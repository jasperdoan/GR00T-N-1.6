"""
Camera abstraction: one interface, two backends.

  StreamCamera  — MJPEG-over-HTTP network stream (color only). Used when the
                  camera hangs off another machine. `CAMERA_SOURCE` = URL str.
  OrbbecCamera  — Orbbec Gemini (pyorbbecsdk) plugged into THIS machine:
                  color + depth hardware-aligned to the color frame, plus the
                  color intrinsics needed for 3D deprojection.
                  `CAMERA_SOURCE` = device index int.

Both expose the same surface:
  read()        -> (ret, frame_bgr)          newest frame, never buffered/stale
  read_depth()  -> (ret, depth_mm|None)      float32 mm aligned to color (Orbbec only)
  has_depth     -> bool
  intrinsics    -> (fx, fy, cx, cy) or None
  grab()/retrieve()                          cv2-compat shims so read_fresh() works
  isOpened() / release()
"""

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


def read_fresh(cap):
    """Grab the most recent frame from any camera object. Returns (ret, bgr)."""
    cap.grab()
    return cap.retrieve()


def read_fresh_after(cap, after_ts: float, timeout: float = 1.5):
    """
    Return a frame CAPTURED after `after_ts` (e.g. the completion time of the
    last arm move) — the deterministic fix for stale-frame servoing: acting on
    a frame taken mid-move systematically over-commands the next correction.
    Polls until cap.last_frame_ts > after_ts or timeout, then falls back to
    the newest frame with a log (never blocks the control loop forever).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if getattr(cap, "last_frame_ts", 0.0) > after_ts:
            return read_fresh(cap)
        time.sleep(0.01)
    print(f"[Vision] No post-move frame within {timeout:.1f}s — using the newest one.")
    return read_fresh(cap)


# =============================================================================
# Network MJPEG stream (color only)
# =============================================================================

class StreamCamera:
    """
    Drains a VideoCapture in a daemon thread and keeps only the NEWEST frame.

    Needed for network streams: the FFMPEG HTTP demuxer ignores
    CAP_PROP_BUFFERSIZE and queues frames, so a plain read() returns
    progressively staler images whenever the control loop runs slower than the
    stream FPS — fatal for visual servoing (the arm would act on pre-move
    frames and oscillate).
    """
    has_depth = False
    intrinsics = None

    def __init__(self, url: str, name: str = "camera"):
        self.last_frame_ts = 0.0
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[Vision] Could not open {name} stream at {url}. "
                f"If the URL host is 0.0.0.0, replace it with the LAN IP of the "
                f"machine hosting the camera (0.0.0.0 is only its listen address)."
            )
        self._lock = threading.Lock()
        self._ret, self._frame = False, None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        # Wait for the first frame so downstream code never sees an empty reader.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ret, _ = self.read()
            if ret:
                print(f"[Vision] Opened {name} stream at {url}.")
                return
            time.sleep(0.05)
        self.release()
        raise RuntimeError(f"[Vision] {name} stream at {url} opened but produced no frames within 5 s.")

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._ret, self._frame = True, frame
                    self.last_frame_ts = time.time()
            else:
                time.sleep(0.01)   # stream hiccup; don't spin at 100% CPU

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def read_depth(self):
        return False, None

    def grab(self):
        return self._ret

    def retrieve(self):
        return self.read()

    def isOpened(self):
        return self._running and self._cap.isOpened()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


# =============================================================================
# Local Orbbec camera (color + aligned depth), via pyorbbecsdk
# =============================================================================

class OrbbecCamera:
    """
    Orbbec Gemini via pyorbbecsdk v2: color + depth streams, depth aligned to
    the color frame by the SDK's AlignFilter (it does the lens-offset warp).
    A daemon thread keeps only the newest (color, depth) PAIR so servoing acts
    on current, mutually-consistent data.
    """
    has_depth = True

    def __init__(self, index: int = 0, name: str = "camera"):
        try:
            import pyorbbecsdk as ob
        except ImportError as e:
            raise RuntimeError(
                "[Vision] pyorbbecsdk is not installed in this environment. "
                "Install the official Linux wheel, e.g.:\n"
                "  pip install https://github.com/orbbec/pyorbbecsdk/releases/"
                "download/v2.1.1/pyorbbecsdk2-2.1.1-cp310-cp310-linux_x86_64.whl"
            ) from e
        self._ob = ob

        ctx = ob.Context()
        devices = ctx.query_devices()
        count = devices.get_count() if hasattr(devices, "get_count") else len(devices)
        if count <= index:
            # NOTE: this index is the ORBBEC enumeration (Orbbec devices only,
            # 0 = first Gemini), NOT the OS /dev/videoN webcam numbering — other
            # cameras on the machine are invisible here and can't shift it.
            listing = []
            for i in range(count):
                try:
                    info = devices.get_device_by_index(i).get_device_info()
                    listing.append(f"  [{i}] {info.get_name()} (SN {info.get_serial_number()})")
                except Exception:
                    listing.append(f"  [{i}] <unreadable>")
            detected = "\n".join(listing) if listing else "  (none)"
            raise RuntimeError(
                f"[Vision] Orbbec device index {index} not found — {count} Orbbec device(s) detected:\n"
                f"{detected}\n"
                f"Is the camera plugged in and are the Orbbec udev rules installed?"
            )
        device = devices.get_device_by_index(index)
        self._pipeline = ob.Pipeline(device)

        config = ob.Config()
        color_profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        depth_profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        self.intrinsics = self._read_intrinsics(color_profile)

        self._pipeline.start(config)
        self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

        self._lock = threading.Lock()
        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self.last_frame_ts = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        deadline = time.time() + 8.0
        while time.time() < deadline:
            ret, _ = self.read()
            if ret:
                print(f"[Vision] Opened Orbbec {name} at index {index} "
                      f"(color {color_profile.get_width()}x{color_profile.get_height()}, "
                      f"depth aligned, intrinsics={'yes' if self.intrinsics else 'NO'}).")
                return
            time.sleep(0.05)
        self.release()
        raise RuntimeError(f"[Vision] Orbbec {name} started but produced no frames within 8 s.")

    @staticmethod
    def _read_intrinsics(color_profile) -> Optional[Tuple[float, float, float, float]]:
        try:
            intr = color_profile.as_video_stream_profile().get_intrinsic() \
                if hasattr(color_profile, "as_video_stream_profile") else color_profile.get_intrinsic()
            return float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
        except Exception as e:
            print(f"[Vision] Could not read color intrinsics ({e}) — 3D localization disabled.")
            return None

    # -------------------------------------------------------------------------
    # Acquisition thread
    # -------------------------------------------------------------------------

    def _loop(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                aligned = self._align.process(frames)
                if aligned is None:
                    continue
                frame_set = aligned.as_frame_set() if hasattr(aligned, "as_frame_set") else aligned

                color_frame = frame_set.get_color_frame()
                depth_frame = frame_set.get_depth_frame()
                if color_frame is None:
                    continue

                bgr = self._color_to_bgr(color_frame)
                depth = self._depth_to_mm(depth_frame) if depth_frame is not None else None
                if bgr is None:
                    continue

                with self._lock:
                    self._color = bgr
                    self._depth = depth
                    self.last_frame_ts = time.time()
            except Exception:
                time.sleep(0.01)   # transient SDK hiccup; keep the thread alive

    def _color_to_bgr(self, color_frame) -> Optional[np.ndarray]:
        ob = self._ob
        w, h = color_frame.get_width(), color_frame.get_height()
        fmt = color_frame.get_format()
        data = np.asanyarray(color_frame.get_data())

        if fmt == ob.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob.OBFormat.RGB:
            return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        if fmt == ob.OBFormat.BGR:
            return data.reshape((h, w, 3))
        if fmt == ob.OBFormat.YUYV:
            return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUYV)
        if fmt == ob.OBFormat.UYVY:
            return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
        if fmt == ob.OBFormat.NV12:
            return cv2.cvtColor(data.reshape((h * 3 // 2, w)), cv2.COLOR_YUV2BGR_NV12)
        if fmt == ob.OBFormat.I420:
            return cv2.cvtColor(data.reshape((h * 3 // 2, w)), cv2.COLOR_YUV2BGR_I420)
        print(f"[Vision] Unsupported Orbbec color format: {fmt}")
        return None

    @staticmethod
    def _depth_to_mm(depth_frame) -> Optional[np.ndarray]:
        try:
            w, h = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
            return raw.astype(np.float32) * float(scale)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def read(self):
        with self._lock:
            if self._color is None:
                return False, None
            return True, self._color.copy()

    def read_depth(self):
        with self._lock:
            if self._depth is None:
                return False, None
            return True, self._depth.copy()

    def grab(self):
        with self._lock:
            return self._color is not None

    def retrieve(self):
        return self.read()

    def isOpened(self):
        return self._running

    def release(self):
        self._running = False
        self._thread.join(timeout=1.5)
        try:
            self._pipeline.stop()
        except Exception:
            pass


# =============================================================================
# Factory
# =============================================================================

def open_camera(source, name: str = "camera"):
    """
    Open a camera from the configured source:
      str URL  → StreamCamera (network MJPEG, color only)
      int      → OrbbecCamera (local Gemini, color + aligned depth)
    Raises RuntimeError with a actionable message when it cannot be opened.
    """
    if isinstance(source, str):
        return StreamCamera(source, name)
    return OrbbecCamera(int(source), name)
