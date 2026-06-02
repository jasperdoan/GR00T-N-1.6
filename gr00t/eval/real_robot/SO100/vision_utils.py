"""
SO100 Vision Utilities (Signal-Based Tracking)

Two independent detectors:
  1. check_task_success  — confirms a new object is stably placed in a target zone using background subtraction.
  2. GraspDetector       — Uses a robust Signal A-D logic for both initial grasp and mid-transit monitoring.
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
    WRIST_PRESENCE_THR,
    WRIST_MIN_PRESENCE_PX,
    WRIST_STABILITY_THR,
    WRIST_CONFIRM_FRAMES,
    VLA_GRASP_MIN_TIME,
    GRIPPER_OPEN_POS,
    GRIPPER_TRANSPORT_MIN
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
        
        touches_edge = (
            bx <= edge_margin
            or by <= edge_margin
            or (bx + bw) >= (w - edge_margin)
            or (by + bh) >= (h - edge_margin)
        )
        
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
    Robust Signal A-D tracking logic. 
    Handles both the initial grasp validation and the mid-transit drop checks.
    """

    def __init__(self, baseline_wrist_img: np.ndarray):
        # Initial Grasp Trackers
        self.history = collections.deque(maxlen=WRIST_CONFIRM_FRAMES)
        self.start_time = time.time()
        self.prev_gray = None
        
        # Mid-Transit Trackers
        self.transit_start_time = 0.0
        self.locked_color_mass = None
        self.transit_lost_count = 0
        self.roi_area = WRIST_GRASP_ROI[2] * WRIST_GRASP_ROI[3]
        
        safe_base = _ensure_uint8(baseline_wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_base[y:y+h, x:x+w]
        self.baseline_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    def extract_color_pixels(self, rgb_img: np.ndarray, target_object: str) -> int:
        """Returns the number of pixels matching the target color (HSV filter) and saves debug info."""
        
        # FIX: LeRobot uses RGB, so we must use COLOR_RGB2HSV
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        
        # Get the color name from the instruction (e.g., "red" from "red cube")
        color_name = target_object.split()[0].lower()
        ranges = COLOR_RANGES.get(color_name, None)
        
        if not ranges:
            print(f"[Vision Debug] ⚠️ Missing color '{color_name}' in COLOR_RANGES!")
            return 0 # Fallback if color unknown

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask |= cv2.inRange(hsv, lower_np, upper_np)
            
        px_count = np.sum(mask > 0)

        # ==========================================
        # DEBUG VISUALIZER: Save images to disk
        # ==========================================
        # FIX: Convert the RGB image to BGR specifically so cv2.imwrite saves the colors correctly!
        # bgr_for_saving = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
        # cv2.imwrite("DEBUG_01_wrist_camera_raw.jpg", bgr_for_saving)
        # cv2.imwrite(f"DEBUG_02_wrist_mask_{color_name}.jpg", mask)
        
        # # Draw text on the original image for easy debugging
        # debug_img = bgr_for_saving.copy()
        # cv2.putText(debug_img, f"Looking for: {color_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # cv2.putText(debug_img, f"Found Px: {px_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if px_count > WRIST_MIN_PRESENCE_PX else (0, 0, 255), 2)
        # cv2.imwrite("DEBUG_03_wrist_overlay.jpg", debug_img)
        # ==========================================

        return px_count

    # -------------------------------------------------------------------------
    # Phase 1: Initial Grasp Validation
    # -------------------------------------------------------------------------
    def update(self, obs: Dict[str, Any], target_object: str) -> str:
        """
        Returns:
          "SUCCESS": Grasp is stable and correct color.
          "WRONG_OBJECT": Gripper closed on empty space or wrong color.
          "SEARCHING": Still trying.
        """
        # --- Signal D: Time gating ---
        if time.time() - self.start_time < VLA_GRASP_MIN_TIME:
            self._update_prev_frame(obs)
            self.history.append(False)
            return "SEARCHING"

        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is None:
            self.history.append(False)
            return "SEARCHING"

        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_curr[y:y+h, x:x+w]

        # --- Signal B: Visual Presence (Color Check instead of old Diff) ---
        color_pixels = self.extract_color_pixels(roi, target_object)
        has_correct_color = color_pixels >= WRIST_MIN_PRESENCE_PX

        # --- Signal C: Gripper STRICT Check ---
        gripper_pos = obs.get("gripper.pos", GRIPPER_OPEN_POS)
        is_gripping = float(gripper_pos) >= GRIPPER_TRANSPORT_MIN   # inverted: closed = high angle

        # --- Decision Logic ---
        if is_gripping and not has_correct_color:
            # The VLA grabbed something, but it's not our target color!
            print(f"[Vision] 🚨 WRONG OBJECT GRASPED! Expected {target_object}. Found only {color_pixels}px.")
            return "WRONG_OBJECT"

        # (Frame logic is simple: Gripping correctly and has right color)
        frame_success = is_gripping and has_correct_color
        print(f'Grasp check: {frame_success} | has_color: {has_correct_color} (px: {color_pixels}) | gripping: {is_gripping} (pos: {float(gripper_pos):.1f})')
        
        self.history.append(frame_success)
        
        if len(self.history) == self.history.maxlen and all(self.history):
            return "SUCCESS"
            
        return "SEARCHING"

    # -------------------------------------------------------------------------
    # Phase 2: Transit Monitoring Setup
    # -------------------------------------------------------------------------
    def lock_grasp(self, obs: Dict[str, Any]):
        """
        Takes a snapshot of the object exactly when FSM confirms grasp.
        Applies heavy blur to track the "Blob of Color" rather than sharp edges.
        """
        self.transit_start_time = time.time()
        self.transit_lost_count = 0
        
        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is not None:
            safe_curr = _ensure_uint8(wrist_img)
            x, y, w, h = WRIST_GRASP_ROI
            roi = safe_curr[y:y+h, x:x+w]
            
            # Heavy 21x21 Gaussian Blur to eliminate edge/shift sensitivity
            self.locked_color_mass = cv2.GaussianBlur(roi, (21, 21), 0)

    # -------------------------------------------------------------------------
    # Phase 3: Mid-Transit Drop Detection
    # -------------------------------------------------------------------------
    def check_grasp_maintained(self, obs: Dict[str, Any]) -> bool:
        """
        Monitors the grasp mid-transit using independent A-D signal logic.
        """
        # --- Signal A: Time Gating (Lift-off delay) ---
        # Ignore checks for the first 0.5 seconds to let the arm clear the table
        if time.time() - self.transit_start_time < 0.5:
            return True

        if self.locked_color_mass is None:
            return True # Fallback if lock wasn't called properly

        wrist_img: Optional[np.ndarray] = obs.get("wrist")
        if wrist_img is None:
            return True # Ignore single dropped camera frames

        # --- Signal B: Mechanical Gripper Check ---
        gripper_pos = float(obs.get("gripper.pos", GRIPPER_OPEN_POS))
        is_gripping = gripper_pos >= GRIPPER_TRANSPORT_MIN   # inverted: closed = high angle

        # --- Signal C: Color Blob Integrity ---
        safe_curr = _ensure_uint8(wrist_img)
        x, y, w, h = WRIST_GRASP_ROI
        roi = safe_curr[y:y+h, x:x+w]
        
        # Heavily blur current frame to ignore shifting/shaking
        curr_color_mass = cv2.GaussianBlur(roi, (21, 21), 0)
        
        # Compare to locked snapshot
        diff = cv2.absdiff(curr_color_mass, self.locked_color_mass)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Generous threshold: Pixel must change by > 50 brightness levels to be "lost"
        changed_px = np.sum(gray_diff > 50)
        
        # Object is safe if less than 50% of the ROI drastically changed color
        is_visually_maintained = changed_px < (self.roi_area * 0.50)

        # Evaluate this specific frame
        frame_maintained = is_gripping and is_visually_maintained

        # --- Signal D: Sequential Confirmation ---
        # Require 3 consecutive bad frames to abort. This perfectly filters out 
        # a split-second shadow or camera glare ruining the run.
        if not frame_maintained:
            self.transit_lost_count += 1
            print(f"  [Transit Monitor] Warning: Drop detected (Frame {self.transit_lost_count}/3) | grip: {is_gripping} | visual: {is_visually_maintained} (Diff px: {changed_px})")
        else:
            self.transit_lost_count = 0  # Reset counter immediately on a good frame

        # If it fails 3 times in a row, the object is truly gone
        if self.transit_lost_count >= 3:
            return False
            
        return True

    def _update_prev_frame(self, obs: Dict[str, Any]):
        """Maintain background tracking while initial check is time-gated."""
        wrist_img = obs.get("wrist")
        if wrist_img is not None:
            safe_img = _ensure_uint8(wrist_img)
            x, y, w, h = WRIST_GRASP_ROI
            roi = safe_img[y:y+h, x:x+w]
            self.prev_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)