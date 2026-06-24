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
    CENTER_TOLERANCE_PX,
    SAFE_Z,
    GRASP_Z,
    FINE_ADJUST_SPEED,
    MATRIX_PATH,
    MIN_BLOB_AREA_PX,
    FRONT_MIN_PRESENCE_PX
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


def check_color_presence(frame: np.ndarray, target_object: str) -> Tuple[bool, int]:
    """Check if target_object color is present (above FRONT_MIN_PRESENCE_PX threshold)."""
    color_name = target_object.split()[0].lower()
    img_u8  = _ensure_uint8(frame)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR) if img_u8.shape[2] == 3 else img_u8
    hsv     = _enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
    mask    = _build_color_mask(hsv, color_name)
    px_count = int(np.sum(mask > 0))
    return px_count >= FRONT_MIN_PRESENCE_PX, px_count


# =============================================================================
# Visual Servoing P-controller
# =============================================================================

def visual_servo_to_center(
    cap_wrist,
    robot,
    target_object: str,
    timeout: float = 15.0,
    safety_monitor: Optional[SafetyMonitor] = None,
    should_stop_cb=None,
) -> bool:
    """
    Proportional control loop: reads wrist camera, moves arm in XY until
    the detected object centroid is within CENTER_TOLERANCE_PX of frame center.

    Returns True on success, False on timeout/abort.
    """
    color_name = target_object.split()[0].lower()
    start_time = time.time()

    print(f"[Vision] Starting visual servoing for '{target_object}'...")

    while True:
        if should_stop_cb and should_stop_cb():
            print("[Vision] Stop requested during visual servoing.")
            return False

        if time.time() - start_time > timeout:
            print("[Vision] Visual servoing TIMEOUT.")
            return False

        ret, frame_bgr = cap_wrist.read()
        if not ret:
            time.sleep(0.033)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Safety check
        if safety_monitor:
            safety_monitor.update_frame(frame_rgb)
            while safety_monitor.is_hand_present():
                print("[SAFETY] Hand detected during visual servoing — holding position...")
                pos = robot.get_current_position()
                if pos:
                    robot.move_to(*pos[:3], speed=FINE_ADJUST_SPEED, wait=False)
                time.sleep(0.1)
                ret2, f2 = cap_wrist.read()
                if ret2:
                    safety_monitor.update_frame(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
                if not safety_monitor.is_hand_present():
                    print("[SAFETY] Hand removed — resuming visual servoing.")
                    start_time = time.time()
                    break

        h, w = frame_bgr.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2

        centroid = find_object_centroid(frame_rgb, color_name)
        if centroid is None:
            time.sleep(0.033)
            continue

        obj_cx, obj_cy = centroid
        err_x = obj_cx - frame_cx
        err_y = obj_cy - frame_cy

        if abs(err_x) < CENTER_TOLERANCE_PX and abs(err_y) < CENTER_TOLERANCE_PX:
            print(f"[Vision] Object centered (err_x={err_x}, err_y={err_y}). Locked.")
            return True

        pos = robot.get_current_position()
        if pos is None:
            time.sleep(0.033)
            continue

        # Camera frame: positive err_x = object is to the right → move +Y in robot frame
        # Mapping depends on camera orientation; adjust signs as needed after physical test
        dx = VS_KP * err_x
        dy = VS_KP * err_y

        robot.move_to(
            pos[0] + dx,
            pos[1] + dy,
            pos[2],
            speed=FINE_ADJUST_SPEED,
            wait=True,
        )

        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining <= 0:
            print("[Vision] Visual servoing TIMEOUT after move.")
            return False

    return False


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
