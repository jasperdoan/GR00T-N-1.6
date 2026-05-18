"""
SO100 Vision Utilities (Feature-Based Tracking)

Two independent detectors:
  1. check_task_success  — confirms a new object is stably placed in a target zone using background subtraction.
  2. GraspDetector       — Uses Optical Flow for robust mid-air grasp monitoring.
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
    Now uses Optical Flow for mid-air transit monitoring.
    1. Initial grasp check is the same (stability, presence, gripper pos).
    2. On success, `lock_grasp` finds unique feature points on the object.
    3. `check_grasp_maintained` tracks those features. If too many are lost, it fails.
    """

    def __init__(self, baseline_wrist_img: np.ndarray):
        self.history = collections.deque(maxlen=WRIST_CONFIRM_FRAMES)
        self.start_time = time.time()
        self.prev_gray = None
        
        # --- Optical Flow Members ---
        self.locked_features = None
        self.last_flow_gray = None
        self.feature_params = dict(maxCorners=40, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        safe_base = _ensure_uint8(baseline_wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_base[y:y+h, x:x+w]
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

        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_curr[y:y+h, x:x+w]
        curr_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # --- Signal A: Stability check ---
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
        self.history.append(frame_success)
        return len(self.history) == self.history.maxlen and all(self.history)

    def lock_grasp(self, obs: Dict[str, Any]):
        """
        Finds strong, unique features on the object to serve as anchors for optical flow tracking.
        """
        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is not None:
            safe_curr = _ensure_uint8(wrist_img)
            x, y, w, h = WRIST_GRASP_ROI
            roi = safe_curr[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            features = cv2.goodFeaturesToTrack(gray_roi, mask=None, **self.feature_params)
            
            if features is not None:
                print(f"  [GraspDetector] Locked onto {len(features)} visual features for tracking.")
                self.locked_features = features
                self.last_flow_gray = gray_roi
            else:
                print("  [GraspDetector] WARNING: Could not find good features to track on object.")
                self.locked_features = None


    def check_grasp_maintained(self, obs: Dict[str, Any]) -> bool:
        """
        Tracks the locked anchor points using Lucas-Kanade Optical Flow.
        This is robust to lighting, motion blur, and perspective shift.
        """
        # Mechanical check first - fastest way to fail
        gripper_pos = obs.get("gripper.pos", GRIPPER_OPEN_POS)
        if float(gripper_pos) > GRIPPER_TRANSPORT_MAX:
            return False

        # If we never found features to track, we can't monitor. Default to True.
        if self.locked_features is None or self.last_flow_gray is None:
            return True

        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is None:
            return True  # Skip single dropped frame

        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_curr[y:y+h, x:x+w]
        curr_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        new_features, status, _err = cv2.calcOpticalFlowPyrLK(
            self.last_flow_gray, curr_gray, self.locked_features, None, **self.lk_params
        )

        # Count how many of our original features were successfully found in the new frame
        if new_features is not None:
            good_new = new_features[status == 1]
            num_initial_features = len(self.locked_features)
            num_found_features = len(good_new)
            
            # --- The Robustness Check ---
            # If we lose more than 40% of our anchor points, the object is gone.
            retention_ratio = num_found_features / num_initial_features
            is_maintained = retention_ratio >= 0.60
            
            if not is_maintained:
                print(f"  [GraspDetector] Grasp Lost! Feature retention dropped to {retention_ratio:.2%}")
            
            # Update the state for the next frame
            self.last_flow_gray = curr_gray.copy()
            self.locked_features = good_new.reshape(-1, 1, 2)
            
            return is_maintained
        else:
            # Catastrophic failure - no features found at all.
            print("  [GraspDetector] Grasp Lost! No features tracked.")
            return False

    def _update_prev_frame(self, obs: Dict[str, Any]):
        """Maintain tracking in the background while time-gated."""
        wrist_img = obs.get("wrist")
        if wrist_img is not None:
            safe_img = _ensure_uint8(wrist_img)
            x, y, w, h = WRIST_GRASP_ROI
            roi = safe_img[y:y+h, x:x+w]
            self.prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)