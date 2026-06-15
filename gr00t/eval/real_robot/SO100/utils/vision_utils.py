"""
SO100 Vision Utilities (Signal-Based Tracking & Safety)
"""

import collections
import time
import threading
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

from utils.constants import (
    COLOR_RANGES,
    MIN_BLOB_AREA_PX,
    FRONT_MIN_PRESENCE_PX,
    WRIST_GRASP_ROI,
    WRIST_PRESENCE_THR,
    WRIST_MIN_PRESENCE_PX,
    WRIST_STABILITY_THR,
    WRIST_CONFIRM_FRAMES,
    VLA_GRASP_MIN_TIME,
    GRIPPER_OPEN_POS,
    GRIPPER_TRANSPORT_THRESHOLD
)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img.copy()

def _enhance_saturation(hsv_img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    h, s, v = cv2.split(hsv_img)
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    return cv2.merge([h, s, v])


# =============================================================================
# Async Safety Monitor (Hand Detection)
# =============================================================================

class SafetyMonitor:
    """Uses MediaPipe in a background thread to prevent control loop lag."""
    def __init__(self, enabled=True):
        self.enabled = enabled and HAS_MEDIAPIPE
        self.hand_detected = False
        
        self._latest_frame = None
        self._running = False
        self._thread = None

        if not self.enabled:
            if enabled:
                print("[WARNING] MediaPipe not found! Hand safety pause is DISABLED.")
        else:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1, # 1 hand is enough to trigger a pause
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._running = True
            self._thread = threading.Thread(target=self._process_loop, daemon=True)
            self._thread.start()

    def update_frame(self, img_rgb: np.ndarray):
        """Passes the frame to the background thread instantly (Non-blocking)."""
        if self.enabled:
            self._latest_frame = img_rgb

    def is_hand_present(self) -> bool:
        """O(1) Instant read for the main control loop."""
        return self.hand_detected

    def _process_loop(self):
        """Background loop running independently of the robot control loop."""
        while self._running:
            frame = self._latest_frame
            self._latest_frame = None # Consume frame
            
            if frame is not None:
                img_u8 = _ensure_uint8(frame)
                # MediaPipe process is CPU heavy (~30-50ms), but now it won't lag the robot!
                try:
                    results = self.hands.process(img_u8)
                    self.hand_detected = results.multi_hand_landmarks is not None
                except Exception as e:
                    print(f"[SafetyMonitor Error] {e}")
            else:
                time.sleep(0.03) # Sleep if no new frame to save CPU
                
    def stop(self):
        """Cleanly shuts down the thread on exit."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)


# =============================================================================
# Workspace Snapshot Annotator (Target Object Bounding Box)
# =============================================================================

def save_workspace_snapshot(
    front_img: np.ndarray, 
    filename: str, 
    zones_dict: Dict[str, Tuple[int, int, int, int]], 
    target_object: str,
    padding: int = 0
):
    """
    Saves the workspace image, drawing a bounding box ONLY for the specific target_object.
    """
    img_bgr = cv2.cvtColor(_ensure_uint8(front_img), cv2.COLOR_RGB2BGR)
    h_img, w_img = img_bgr.shape[:2]

    # Extract target color
    target_color = target_object.split()[0].lower()
    ranges = COLOR_RANGES.get(target_color, None)

    if ranges is None:
        cv2.imwrite(filename, img_bgr)
        print(f"[Vision] Snapshot saved. (No box drawn - unknown color '{target_color}')")
        return

    hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_full = _enhance_saturation(hsv_full, factor=1.4)

    for zone_name, zone_coords in zones_dict.items():
        x, y, w, h = zone_coords
        
        px = max(0, x - padding)
        py = max(0, y - padding)
        pw = min(w_img - px, w + 2 * padding)
        ph = min(h_img - py, h + 2 * padding)

        # Faint dashed box for the valid zone area
        cv2.rectangle(img_bgr, (px, py), (px + pw, py + ph), (255, 255, 255), 1)
        cv2.putText(img_bgr, f"Zone: {zone_name}", (px, max(0, py - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        crop_hsv = hsv_full[py:py+ph, px:px+pw]

        # Scan specifically for the requested color
        mask = np.zeros(crop_hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask |= cv2.inRange(crop_hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        
        mask = _clean_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_BLOB_AREA_PX:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                
                abs_x, abs_y = px + cx, py + cy
                
                # Draw Target Box
                cv2.rectangle(img_bgr, (abs_x, abs_y), (abs_x + cw, abs_y + ch), (0, 255, 0), 2)
                cv2.putText(img_bgr, target_object.title(), (abs_x, max(0, abs_y - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(filename, img_bgr)
    print(f"[Vision] Workspace snapshot saved to {filename}")


# =============================================================================
# Existing Handlers
# =============================================================================

def check_task_success(current_frame, baseline_frame, zone, diff_threshold=25, edge_margin=3, debug=False) -> bool:
    curr_u8, base_u8 = _ensure_uint8(current_frame), _ensure_uint8(baseline_frame)
    x, y, w, h = zone
    diff = cv2.absdiff(curr_u8[y:y+h, x:x+w], base_u8[y:y+h, x:x+w])
    gray_diff = cv2.GaussianBlur(cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, diff_mask = cv2.threshold(gray_diff, diff_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(_clean_mask(diff_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(cnt) >= MIN_BLOB_AREA_PX for cnt in contours)

def check_color_presence_front(front_img, target_object, zone, debug=False) -> Tuple[bool, int]:
    img_u8 = _ensure_uint8(front_img)
    x, y, w, h = zone
    hsv = _enhance_saturation(cv2.cvtColor(img_u8[y:y+h, x:x+w], cv2.COLOR_RGB2HSV), factor=1.4)
    color_name = target_object.split()[0].lower()
    ranges = COLOR_RANGES.get(color_name, None)
    if not ranges: return False, 0
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in ranges: mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    px_count = int(np.sum(mask > 0))
    return px_count >= FRONT_MIN_PRESENCE_PX, px_count

class GraspDetector:
    """
    Robust Signal A-D tracking logic. 
    Handles both the initial grasp validation and the mid-transit drop checks.
    """

    def __init__(self, baseline_wrist_img: np.ndarray):
        self.history = collections.deque(maxlen=WRIST_CONFIRM_FRAMES)
        self.start_time = time.time()
        self.prev_gray = None
        self.transit_start_time = 0.0
        self.locked_color_mass = None
        self.transit_lost_count = 0
        self.roi_area = WRIST_GRASP_ROI[2] * WRIST_GRASP_ROI[3]
        safe_base = _ensure_uint8(baseline_wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        self.baseline_gray = cv2.cvtColor(safe_base[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    def extract_color_pixels(self, rgb_img: np.ndarray, target_object: str) -> int:
        hsv = _enhance_saturation(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV), factor=1.4)
        ranges = COLOR_RANGES.get(target_object.split()[0].lower(), None)
        if not ranges: return 0 
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges: mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        return np.sum(mask > 0)

    def update(self, obs: Dict[str, Any], target_object: str) -> str:
        """
        Returns:
          "SUCCESS": Grasp is stable and correct color.
          "WRONG_OBJECT": Gripper closed on empty space or wrong color.
          "SEARCHING": Still trying.
        """
        # --- Signal D: Time gating ---
        if time.time() - self.start_time < VLA_GRASP_MIN_TIME:
            self._update_prev_frame(obs); self.history.append(False); return "SEARCHING"
        wrist_img = obs.get("wrist")
        if wrist_img is None: self.history.append(False); return "SEARCHING"
        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_curr[y:y+h, x:x+w]

        # --- Signal B: Visual Presence (Color Check instead of old Diff) ---
        color_pixels = self.extract_color_pixels(roi, target_object)
        has_correct_color = color_pixels >= WRIST_MIN_PRESENCE_PX

        # --- Signal C: Gripper STRICT Check ---
        gripper_pos = obs.get("gripper.pos", GRIPPER_OPEN_POS)
        is_gripping = float(gripper_pos) <= GRIPPER_TRANSPORT_THRESHOLD  

        if is_gripping and not has_correct_color: return "WRONG_OBJECT"
        self.history.append(is_gripping and has_correct_color)
        if len(self.history) == self.history.maxlen and all(self.history): return "SUCCESS"
        return "SEARCHING"

    def lock_grasp(self, obs: Dict[str, Any]):
        self.transit_start_time = time.time(); self.transit_lost_count = 0
        wrist_img = obs.get("wrist")
        if wrist_img is not None:
            safe_curr = _ensure_uint8(wrist_img)
            x, y, w, h = WRIST_GRASP_ROI
            self.locked_color_mass = cv2.GaussianBlur(safe_curr[y:y+h, x:x+w], (21, 21), 0)

    def check_grasp_maintained(self, obs: Dict[str, Any]) -> bool:
        # --- Signal A: Time Gate ---
        if time.time() - self.transit_start_time < 0.5: return True
        if self.locked_color_mass is None: return True 
        wrist_img = obs.get("wrist")
        if wrist_img is None: return True 

        # --- Signal B: Mechanical Gripper Check ---
        gripper_pos = float(obs.get("gripper.pos", GRIPPER_OPEN_POS))
        is_gripping = gripper_pos <= GRIPPER_TRANSPORT_THRESHOLD

        # --- Signal C: Color Blob Integrity ---
        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        curr_color_mass = cv2.GaussianBlur(safe_curr[y:y+h, x:x+w], (21, 21), 0)
        
        diff = cv2.absdiff(curr_color_mass, self.locked_color_mass)
        changed_px = np.sum(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 50)
        
        # --- Signal D: Sequential Confirmation ---
        is_visually_maintained = changed_px < (self.roi_area * 0.50)
        if not (is_gripping and is_visually_maintained): self.transit_lost_count += 1
        else: self.transit_lost_count = 0  

        return self.transit_lost_count < 3

    def _update_prev_frame(self, obs: Dict[str, Any]):
        wrist_img = obs.get("wrist")
        if wrist_img is not None:
            safe_img = _ensure_uint8(wrist_img)
            x, y, w, h = WRIST_GRASP_ROI
            self.prev_gray = cv2.cvtColor(safe_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)