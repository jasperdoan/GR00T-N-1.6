"""
SO100 Vision Utilities (Presence-Based)

Two independent detectors:
  1. check_task_success  — confirms a new object is stably placed in a target zone using background subtraction.
  2. GraspDetector       — multi-signal sequential detector using the wrist camera comparing against an empty baseline.
"""

import collections
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from constants import (
    MIN_BLOB_AREA_PX,
    WRIST_GRASP_ROI,
    WRIST_PRESENCE_THR,
    WRIST_MIN_PRESENCE_PX,
    WRIST_STABILITY_THR,
    WRIST_CONFIRM_FRAMES,
    VLA_GRASP_MIN_TIME,
    GRIPPER_OPEN_POS,
    GRIPPER_TRANSPORT_MAX
)

# =============================================================================
# Internal Helpers
# =============================================================================

def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Remove small noise blobs with a morphological open."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img.copy()


# =============================================================================
# Task Success Detection (front camera, zone-based presence check)
# =============================================================================

def check_task_success(
    current_frame: np.ndarray,
    baseline_frame: np.ndarray,
    zone: Tuple[int, int, int, int],
    diff_threshold: int = 25,
    edge_margin: int = 3,
    debug: bool = False,
) -> bool:
    """
    Returns True if a new object of sufficient size has appeared in the zone.
    """
    def to_uint8(img):
        if img.dtype != np.uint8:
            return (img * 255).clip(0, 255).astype(np.uint8)
        return img

    curr_u8 = to_uint8(current_frame)
    base_u8 = to_uint8(baseline_frame)

    x, y, w, h = zone
    crop      = curr_u8[y : y + h, x : x + w]
    base_crop = base_u8[y : y + h, x : x + w]

    # --- Step 1: background subtraction ---
    diff      = cv2.absdiff(crop, base_crop)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, diff_mask = cv2.threshold(gray_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # --- Step 2: Clean the mask ---
    final_mask = _clean_mask(diff_mask)

    # --- Step 3: contour analysis ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_valid_blob = False
    
    if debug:
        print(f"\n--- Vision Debug [Presence Check] ---")
        print(f"Diff mask pixels: {np.sum(final_mask > 0)}")
        print(f"Contours found: {len(contours)}")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        
        # touches_edge = (
        #     bx <= edge_margin
        #     or by <= edge_margin
        #     or (bx + bw) >= (w - edge_margin)
        #     or (by + bh) >= (h - edge_margin)
        # )
        
        if debug:
            print(f"  Contour {i}: Area={area}, TouchesEdge={touches_edge}")

        if area >= MIN_BLOB_AREA_PX:
            found_valid_blob = True
            break

    if debug:
        cv2.imwrite("DEBUG_success_diff.jpg", final_mask)
        full_debug = cv2.cvtColor(curr_u8, cv2.COLOR_RGB2BGR)
        cv2.rectangle(full_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite("DEBUG_full_frame.jpg", full_debug)

    return found_valid_blob


# =============================================================================
# Grasp Confirmation Detector (Wrist Camera)
# =============================================================================

class GraspDetector:
    """
    Requires:
      A) Image stability inside the gripper's bounding box.
      B) Visual Object Presence (significant pixel diff from empty baseline).
      C) The gripper to NOT be fully open.
      D) Time delay: giving the VLA >= 1.0 seconds to act.
    """

    def __init__(self, baseline_wrist_img: np.ndarray):
        self.history = collections.deque(maxlen=WRIST_CONFIRM_FRAMES)
        self.start_time = time.time()
        self.prev_gray = None
        
        # Scale 0-1 float to 0-255 integer and convert to Grayscale
        safe_base = _ensure_uint8(baseline_wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_base[y:y+h, x:x+w]
        
        # Note: Using BGR2GRAY because LeRobot outputs BGR natively
        self.baseline_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def update(self, obs: Dict[str, Any]) -> bool:
        # --- Signal D: Time gating ---
        if time.time() - self.start_time < VLA_GRASP_MIN_TIME:
            self._update_prev_frame(obs)
            self.history.append(False)
            return False

        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is None:
            self.history.append(False)
            return False

        # Scale 0-1 float to 0-255 integer
        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_curr[y:y+h, x:x+w]
        curr_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # --- Signal A: Stability check (% still between frames) ---
        is_stable = False
        if self.prev_gray is not None:
            diff_motion = cv2.absdiff(curr_gray, self.prev_gray)
            changed_pixels = np.sum(diff_motion > WRIST_STABILITY_THR)
            total_pixels = curr_gray.size
            if (changed_pixels / total_pixels) <= 0.15:
                is_stable = True
        self.prev_gray = curr_gray

        # --- Signal B: Visual Presence (Diff from empty baseline) ---
        diff_presence = cv2.absdiff(curr_gray, self.baseline_gray)
        presence_px = np.sum(diff_presence > WRIST_PRESENCE_THR)
        has_object = presence_px >= WRIST_MIN_PRESENCE_PX

        # --- Signal C: Gripper STRICT Check ---
        gripper_pos = obs.get("gripper.pos", GRIPPER_OPEN_POS)
        is_gripping = float(gripper_pos) <= GRIPPER_TRANSPORT_MAX

        frame_success = is_stable and has_object and is_gripping

        print(f'Grasp check: {frame_success} | stable: {is_stable} | has_obj: {has_object} (px diff: {presence_px}) | gripping: {is_gripping} (pos: {float(gripper_pos):.1f})')

        # Optional: Uncomment below to save an image and physically see the difference mask if you need to tune thresholds!
        if presence_px >= 0:
            cv2.imwrite("DEBUG_wrist_diff.jpg", diff_presence)

        self.history.append(frame_success)
        return len(self.history) == self.history.maxlen and all(self.history)

    def check_grasp_maintained(self, obs: Dict[str, Any]) -> bool:
        """
        Fast-path check intended for monitoring the object during transit.
        Disregards time-gating and stability (since the arm is moving).
        Returns False if the object slips out of the gripper.
        """
        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is None:
            return True  # Assume true if camera drops a single frame

        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_curr[y:y+h, x:x+w]
        curr_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Check visual presence (allow slightly lower threshold to account for shifting)
        diff_presence = cv2.absdiff(curr_gray, self.baseline_gray)
        presence_px = np.sum(diff_presence > WRIST_PRESENCE_THR)
        has_object = presence_px >= (WRIST_MIN_PRESENCE_PX * 0.75)

        # Check that the gripper hasn't swung completely wide open
        gripper_pos = obs.get("gripper.pos", GRIPPER_OPEN_POS)
        is_gripping = float(gripper_pos) <= GRIPPER_TRANSPORT_MAX

        return has_object and is_gripping

    def _update_prev_frame(self, obs: Dict[str, Any]):
        """Maintain tracking in the background while time-gated."""
        wrist_img = obs.get("wrist")
        if wrist_img is not None:
            safe_img = _ensure_uint8(wrist_img)
            x, y, w, h = WRIST_GRASP_ROI
            roi = safe_img[y:y+h, x:x+w]
            self.prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)