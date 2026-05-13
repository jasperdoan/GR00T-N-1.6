"""
SO100 Vision Utilities

Two independent detectors:
  1. check_task_success  — confirms a cube is stably placed in a target zone.
  2. GraspDetector       — multi-signal sequential detector using the wrist camera.
"""

import collections
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from constants import (
    COLOR_RANGES,
    MIN_BLOB_AREA_PX,
    WRIST_GRASP_ROI,
    WRIST_COLOR_RANGES,
    WRIST_MIN_COLOR_PX,
    WRIST_STABILITY_THR,
    WRIST_CONFIRM_FRAMES,
    VLA_GRASP_MIN_TIME,
    GRIPPER_OPEN_POS,
)


# =============================================================================
# Internal Helpers
# =============================================================================

def _build_color_mask(hsv: np.ndarray, color_name: str, color_ranges: Dict) -> np.ndarray:
    """Return a binary mask of pixels matching the named color from the provided dict."""
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in color_ranges.get(color_name,[]):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Remove small noise blobs with a morphological open."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _color_pixel_count(image_bgr: np.ndarray, color_name: str, color_ranges: Dict) -> int:
    """Return the total number of pixels matching color_name in image_bgr."""
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = _build_color_mask(hsv, color_name, color_ranges)
    mask = _clean_mask(mask)
    return int(np.sum(mask > 0))


# =============================================================================
# Task Success Detection (front camera, zone-based)
# =============================================================================

def check_task_success(
    current_frame: np.ndarray,
    baseline_frame: np.ndarray,
    zone: Tuple[int, int, int, int],
    color_name: str,
    diff_threshold: int = 25,
    edge_margin: int = 3,
) -> bool:
    """
    Returns True when the target cube is stably placed inside zone AND the robot
    arm has cleared the area.
    """
    x, y, w, h = zone

    crop      = current_frame[y : y + h, x : x + w]
    base_crop = baseline_frame[y : y + h, x : x + w]

    # --- Step 1: background subtraction ---
    diff      = cv2.absdiff(crop, base_crop)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, diff_mask = cv2.threshold(gray_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # --- Step 2: color mask (using front cam ranges) ---
    hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_mask = _build_color_mask(hsv, color_name, COLOR_RANGES)

    # --- Step 3: intersection ---
    final_mask = cv2.bitwise_and(color_mask, diff_mask)
    final_mask = _clean_mask(final_mask)

    # --- Step 4: contour analysis ---
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_BLOB_AREA_PX:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        touches_edge = (
            bx <= edge_margin
            or by <= edge_margin
            or (bx + bw) >= (w - edge_margin)
            or (by + bh) >= (h - edge_margin)
        )

        if not touches_edge:
            return True

    return False


# =============================================================================
# Grasp Confirmation Detector (Wrist Camera)
# =============================================================================

class GraspDetector:
    """
    Stateful detector that ensures multi-signal verification holds for N frames.
    Requires:
      A) Image stability inside the gripper's bounding box (>= 75% still).
      B) Color presence inside that ROI.
      C) The gripper to NOT be fully open.
      D) Time delay: giving the VLA >= 5 seconds to act.
    """

    def __init__(self):
        self.history = collections.deque(maxlen=WRIST_CONFIRM_FRAMES)
        self.start_time = time.time()
        self.prev_gray = None

    def update(self, obs: Dict[str, Any], color_name: str) -> bool:
        # --- Signal D: Time gating ---
        if time.time() - self.start_time < VLA_GRASP_MIN_TIME:
            self._update_prev_frame(obs)
            self.history.append(False)
            return False

        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is None:
            self.history.append(False)
            return False

        x, y, w, h = WRIST_GRASP_ROI
        roi_bgr = wrist_img[y:y+h, x:x+w]
        curr_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # --- Signal A: Stability check (75% still) ---
        is_stable = False
        if self.prev_gray is not None:
            diff = cv2.absdiff(curr_gray, self.prev_gray)
            
            # Count pixels that have changed beyond the threshold
            changed_pixels = np.sum(diff > WRIST_STABILITY_THR)
            total_pixels = curr_gray.size
            
            # If 25% or less of the pixels are moving, we consider it 75% still
            if (changed_pixels / total_pixels) <= 0.25:
                is_stable = True

        self.prev_gray = curr_gray

        # --- Signal B: Color Presence ---
        color_px = _color_pixel_count(roi_bgr, color_name, WRIST_COLOR_RANGES)
        has_color = color_px >= WRIST_MIN_COLOR_PX

        # --- Signal C: Gripper Loose Check ---
        gripper_pos = obs.get("gripper.pos", GRIPPER_OPEN_POS)
        # Verify the arm isn't just flying through empty air wide open
        is_gripping = float(gripper_pos) < (GRIPPER_OPEN_POS - 2.0)

        # Append overall success for this frame
        frame_success = is_stable and has_color and is_gripping

        print(f'Grasp detection frame: {frame_success} - is_stable: {is_stable}, has_color: {has_color}, is_gripping: {is_gripping}')

        self.history.append(frame_success)

        # Only return true if the queue is full and ALL 16 frames were successful
        return len(self.history) == self.history.maxlen and all(self.history)

    def _update_prev_frame(self, obs: Dict[str, Any]):
        """Maintain tracking in the background while time-gated."""
        wrist_img = obs.get("wrist")
        if wrist_img is not None:
            x, y, w, h = WRIST_GRASP_ROI
            roi_bgr = wrist_img[y:y+h, x:x+w]
            self.prev_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)