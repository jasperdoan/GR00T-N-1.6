"""
Lite6 Vision Utilities
- SafetyMonitor: multiprocess MediaPipe hand detection (GIL bypass)
- Homography pixel→robot coordinate mapping
- HSV color detection, contour centroid
- Visual Servoing P-controller loop
- Workspace snapshot annotator
"""

import os
import time
import ctypes
import multiprocessing as mp_lib
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

from utils.constants import (
    COLOR_RANGES,
    VS_KP,
    MAX_SERVO_STEP_MM,
    SERVO_DEADBAND_PX,
    GRIPPER_ROI,
    SERVO_CONFIRM_FRAMES,
    FINE_ADJUST_SPEED,
    MATRIX_PATH,
    MIN_BLOB_AREA_PX,
    FRONT_MIN_PRESENCE_PX,
)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# =============================================================================
# Internal helpers
# =============================================================================

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img.copy()

def _enhance_saturation(hsv_img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    h, s, v = cv2.split(hsv_img)
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    return cv2.merge([h, s, v])

def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# =============================================================================
# Camera helpers
# =============================================================================

class LatestFrameReader:
    """
    Drains a VideoCapture in a daemon thread and keeps only the NEWEST frame.

    Needed for network streams: the FFMPEG HTTP demuxer ignores
    CAP_PROP_BUFFERSIZE and queues frames, so a plain read() returns
    progressively staler images whenever the control loop runs slower than the
    stream FPS — fatal for visual servoing (the arm would act on pre-move
    frames and oscillate).

    Mirrors the cv2.VideoCapture surface the rest of the code uses
    (grab/retrieve/read/isOpened/release), so read_fresh() works unchanged.
    """
    def __init__(self, cap):
        import threading
        self._cap = cap
        self._lock = threading.Lock()
        self._ret, self._frame = False, None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._ret, self._frame = True, frame
            else:
                time.sleep(0.01)   # stream hiccup; don't spin at 100% CPU

    def grab(self):
        return self._ret

    def retrieve(self):
        return self.read()

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def isOpened(self):
        return self._running and self._cap.isOpened()

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


def open_camera(source, name: str = "camera"):
    """
    Open a camera from either a local device index (int) or a stream URL (str).
    Raises RuntimeError if it cannot be opened or produces no frames.

    Local device: VideoCapture(index) with BUFFERSIZE=1 as before.
    Stream URL:   VideoCapture(url, CAP_FFMPEG) wrapped in LatestFrameReader so
                  every read is the newest frame, not a buffered stale one.
    """
    if isinstance(source, str):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(
                f"[Vision] Could not open {name} stream at {source}. "
                f"If the URL host is 0.0.0.0, replace it with the LAN IP of the "
                f"machine hosting the camera (0.0.0.0 is only its listen address)."
            )
        reader = LatestFrameReader(cap)
        # Wait for the first frame so downstream code never sees an empty reader.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            ret, _ = reader.read()
            if ret:
                print(f"[Vision] Opened {name} stream at {source}.")
                return reader
            time.sleep(0.05)
        reader.release()
        raise RuntimeError(f"[Vision] {name} stream at {source} opened but produced no frames within 5 s.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"[Vision] Could not open {name} at index {source}.")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass  # not all backends support BUFFERSIZE
    # Warm up: discard the first few frames so exposure/auto-WB settle.
    for _ in range(5):
        cap.read()
    print(f"[Vision] Opened {name} at index {source}.")
    return cap


def read_fresh(cap):
    """
    Grab the most recent frame. With BUFFERSIZE=1 a single read is current, but
    on backends that ignore it we grab twice to drop one buffered frame.
    Returns (ret, frame_bgr).
    """
    cap.grab()
    return cap.retrieve()


# =============================================================================
# Async Safety Monitor (multiprocessing hand detection)
# =============================================================================

class SafetyMonitor:
    """
    Runs MediaPipe Hands in a separate OS process to bypass the GIL.
    Main loop reads hand_detected in O(1) via shared memory.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_MEDIAPIPE

        self.hand_detected = mp_lib.Value(ctypes.c_bool, False)
        self.frame_queue   = mp_lib.Queue(maxsize=1)
        self._process      = None

        if not self.enabled:
            if enabled:
                print("[WARNING] MediaPipe not found — hand safety pause is DISABLED.")
        else:
            self._process = mp_lib.Process(
                target=self._process_loop,
                args=(self.frame_queue, self.hand_detected),
                daemon=True,
            )
            self._process.start()

    def update_frame(self, img_rgb: np.ndarray):
        """Pass a downscaled frame to the background process (lazy, non-blocking)."""
        if not self.enabled:
            return
        if self.frame_queue.empty():
            small = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            try:
                self.frame_queue.put_nowait(small)
            except Exception:
                pass

    def is_hand_present(self) -> bool:
        return bool(self.hand_detected.value)

    @staticmethod
    def _process_loop(queue, shared_flag):
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        while True:
            try:
                frame = queue.get()
                if frame is None:
                    break
                img_u8 = frame if frame.dtype == np.uint8 else (frame * 255).clip(0, 255).astype(np.uint8)
                results = hands.process(img_u8)
                shared_flag.value = results.multi_hand_landmarks is not None
            except Exception:
                pass

    def stop(self):
        if self.enabled:
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except Exception:
                pass
            try:
                self.frame_queue.put_nowait(None)
            except Exception:
                pass
        if hasattr(self, "frame_queue") and hasattr(self.frame_queue, "cancel_join_thread"):
            self.frame_queue.cancel_join_thread()
        if self._process and self._process.is_alive():
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)


# =============================================================================
# Homography utilities
# =============================================================================

def load_homography(path: str = MATRIX_PATH) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"[Vision] Homography matrix not found at {path}. Run calibration first.")
        return None
    H = np.load(path)
    print(f"[Vision] Loaded homography matrix from {path}")
    return H

def pixel_to_robot(u: float, v: float, H: np.ndarray) -> Tuple[float, float]:
    """Map a (u, v) pixel to robot (X_mm, Y_mm) via the 3x3 homography."""
    vec = np.array([u, v, 1.0], dtype=np.float64)
    mapped = H @ vec
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


# =============================================================================
# HSV object detection (top-down camera)
# =============================================================================

def _build_color_mask(hsv_img: np.ndarray, color_name: str) -> np.ndarray:
    ranges = COLOR_RANGES.get(color_name.lower())
    if not ranges:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv_img, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return _clean_mask(mask)


def find_object_centroid(frame: np.ndarray, color_name: str) -> Optional[Tuple[int, int]]:
    """
    Detect a colored object in the given frame.
    Returns pixel (u, v) of the largest blob centroid, or None if not found.
    """
    img_u8  = _ensure_uint8(frame)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR) if img_u8.shape[2] == 3 else img_u8
    hsv     = _enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
    mask    = _build_color_mask(hsv, color_name)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BLOB_AREA_PX:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def find_object_blob(frame: np.ndarray, color_name: str):
    """
    Full blob detection for the servo/grasp phase. Returns
        (cx, cy, (bx, by, bw, bh), angle_deg)
    for the largest color blob, or None if not found.

    angle_deg is the object's in-image rotation from cv2.minAreaRect, normalized
    to [-45, 45) — a cube is 90°-symmetric, so this is always the SMALLEST wrist
    rotation that aligns the jaws with the object's faces.
    """
    img_u8  = _ensure_uint8(frame)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR) if img_u8.shape[2] == 3 else img_u8
    hsv     = _enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
    mask    = _build_color_mask(hsv, color_name)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BLOB_AREA_PX:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    bbox = cv2.boundingRect(largest)

    # minAreaRect angle conventions vary across OpenCV versions; fold everything
    # into [-45, 45) using the cube's 90° symmetry.
    (_, _), (_, _), raw_angle = cv2.minAreaRect(largest)
    angle = raw_angle % 90.0
    if angle >= 45.0:
        angle -= 90.0

    return cx, cy, bbox, angle


def blob_inside_roi(bbox, roi) -> bool:
    """True only if the blob's bounding box is FULLY inside the ROI (no pixel outside)."""
    bx, by, bw, bh = bbox
    rx, ry, rw, rh = roi
    return bx >= rx and by >= ry and (bx + bw) <= (rx + rw) and (by + bh) <= (ry + rh)


def check_color_presence(frame: np.ndarray, target_object: str, zone_roi=None) -> Tuple[bool, int]:
    """
    Check if target_object's color is present above FRONT_MIN_PRESENCE_PX.
    If zone_roi=(x, y, w, h) is given, the search is restricted to that pixel
    region — essential so PRE_CHECK (source zone) and VERIFY (target zone) can
    actually distinguish "in the right zone" from "anywhere on the table".
    """
    color_name = target_object.split()[0].lower()
    img_u8  = _ensure_uint8(frame)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR) if img_u8.shape[2] == 3 else img_u8

    if zone_roi is not None:
        x, y, w, h = zone_roi
        H_img, W_img = img_bgr.shape[:2]
        x = max(0, min(x, W_img - 1)); y = max(0, min(y, H_img - 1))
        w = max(1, min(w, W_img - x)); h = max(1, min(h, H_img - y))
        img_bgr = img_bgr[y:y + h, x:x + w]

    hsv     = _enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
    mask    = _build_color_mask(hsv, color_name)
    px_count = int(np.sum(mask > 0))
    return px_count >= FRONT_MIN_PRESENCE_PX, px_count


# =============================================================================
# Visual Servoing P-controller
# =============================================================================

def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def visual_servo_to_grasp(
    cap,
    robot,
    target_object: str,
    timeout: float = 15.0,
    safety_monitor: Optional[SafetyMonitor] = None,
    should_stop_cb=None,
) -> Tuple[bool, Optional[float]]:
    """
    IBVS proportional-control loop for the non-coaxial wrist camera.

    Drive:  nudge the arm in XY so the blob CENTROID moves toward the center of
            GRIPPER_ROI (the data-measured region where the object appears when
            the gripper is correctly over it).
    Accept: only when the blob's bounding box sits FULLY inside GRIPPER_ROI for
            SERVO_CONFIRM_FRAMES consecutive frames (the arm holds still during
            confirmation, so mask flicker can't fake a lock).

    Returns (locked, angle_deg):
      - locked: True once the containment criterion holds.
      - angle_deg: the object's in-image rotation ([-45, 45), from the LAST
        confirmation frame) so the caller can yaw-align the gripper without
        another camera read. None when not locked.
    """
    color_name = target_object.split()[0].lower()
    rx, ry, rw, rh = GRIPPER_ROI
    aim_x = rx + rw // 2
    aim_y = ry + rh // 2

    start_time = time.time()
    confirm_count = 0
    last_angle = None

    print(f"[Vision] Visual servoing '{target_object}' into gripper ROI {GRIPPER_ROI}...")

    while True:
        if should_stop_cb and should_stop_cb():
            print("[Vision] Stop requested during visual servoing.")
            return False, None

        if time.time() - start_time > timeout:
            print("[Vision] Visual servoing TIMEOUT.")
            return False, None

        if robot.has_error():
            print("[Vision] Arm in error state during servoing — aborting.")
            return False, None

        ret, frame_bgr = read_fresh(cap)
        if not ret:
            time.sleep(0.033)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Hand safety: pause the arm, hold until clear, then resume ---
        if safety_monitor:
            safety_monitor.update_frame(frame_rgb)
            if safety_monitor.is_hand_present():
                print("[SAFETY] Hand detected during servoing — pausing arm...")
                robot.pause()
                while safety_monitor.is_hand_present():
                    time.sleep(0.05)
                    r2, f2 = read_fresh(cap)
                    if r2:
                        safety_monitor.update_frame(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
                robot.resume()
                print("[SAFETY] Hand removed — resuming servoing.")
                start_time = time.time()   # don't penalise the pause against the timeout
                confirm_count = 0
                continue

        blob = find_object_blob(frame_rgb, color_name)
        if blob is None:
            confirm_count = 0
            time.sleep(0.033)
            continue

        obj_cx, obj_cy, bbox, angle = blob

        # --- Acceptance: strict containment, held for N consecutive frames ---
        if blob_inside_roi(bbox, GRIPPER_ROI):
            confirm_count += 1
            last_angle = angle
            if confirm_count >= SERVO_CONFIRM_FRAMES:
                print(f"[Vision] Blob {bbox} inside gripper ROI for "
                      f"{confirm_count} frames. Locked (angle={last_angle:.1f}°).")
                return True, last_angle
            time.sleep(0.033)   # hold still; just re-read to confirm stability
            continue
        confirm_count = 0

        # --- Drive: centroid error toward the ROI center ---
        err_x = obj_cx - aim_x
        err_y = obj_cy - aim_y

        cmd_err_x = 0 if abs(err_x) < SERVO_DEADBAND_PX else err_x
        cmd_err_y = 0 if abs(err_y) < SERVO_DEADBAND_PX else err_y

        pos = robot.get_current_position()
        if pos is None:
            time.sleep(0.033)
            continue

        # P-control with a per-step clamp. Sign mapping (camera px -> robot mm)
        # depends on the wrist-camera orientation; flip signs here after a bench test.
        dx = _clamp(VS_KP * cmd_err_x, MAX_SERVO_STEP_MM)
        dy = _clamp(VS_KP * cmd_err_y, MAX_SERVO_STEP_MM)

        if not robot.move_to(pos[0] + dx, pos[1] + dy, pos[2],
                             speed=FINE_ADJUST_SPEED, wait=True):
            print("[Vision] Servo nudge failed (rejected/faulted) — aborting.")
            return False, None

    return False, None


# =============================================================================
# Workspace snapshot annotator
# =============================================================================

def save_workspace_snapshot(
    frame: np.ndarray,
    filename: str,
    target_object: str,
    output_dir: str,
    zones_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
):
    """
    Save an annotated top-down snapshot.
    Draws white zone boxes and a green bounding box around the detected object.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, os.path.basename(filename))

    img_u8  = _ensure_uint8(frame)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR) if img_u8.shape[2] == 3 else img_u8.copy()
    h_img, w_img = img_bgr.shape[:2]

    color_name = target_object.split()[0].lower()
    ranges = COLOR_RANGES.get(color_name)

    # Draw zone boxes
    if zones_dict:
        for zone_name, (zx, zy, zw, zh) in zones_dict.items():
            cv2.rectangle(img_bgr, (zx, zy), (zx + zw, zy + zh), (255, 255, 255), 1)
            cv2.putText(img_bgr, zone_name, (zx, max(0, zy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw object detection box
    if ranges:
        hsv  = _enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
        mask = _build_color_mask(hsv, color_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_BLOB_AREA_PX:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(img_bgr, target_object.title(), (bx, max(0, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(filepath, img_bgr)
    print(f"[Vision] Snapshot saved to {filepath}")
