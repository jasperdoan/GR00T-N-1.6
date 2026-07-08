"""
HSV color detection with optional depth-based TOP-FACE refinement.

The perception problem this solves: the wrist camera views the cube at an
angle, so the color blob includes the cube's SIDE face. The 2D centroid of the
full silhouette is dragged toward the visible side, and the gripper ends up
grabbing a corner. With depth aligned to color, side-face pixels are FARTHER
from the camera than top-face pixels — restricting the blob to the
nearest-depth band isolates the TOP face, whose centroid is the true grasp
center.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.constants import (
    COLOR_RANGES,
    MIN_BLOB_AREA_PX,
    FRONT_MIN_PRESENCE_PX,
    DEPTH_TOP_BAND_MM,
    DEPTH_MIN_VALID_PX,
    MASK_FUSE_KERNEL_PX,
    OBJECT_MIN_HEIGHT_MM,
)
from utils.vision.helpers import enhance_saturation, clean_mask, to_bgr


def _fuse_fragments(mask: np.ndarray) -> np.ndarray:
    """
    Morphological CLOSE to re-fuse a mask that split across the object (the HSV
    mask fragments at the lit edge between a cube's top and side faces; the
    largest-contour pick then flip-flops between fragments frame to frame,
    which made the servo chase a teleporting centroid on hardware).
    """
    kernel = np.ones((MASK_FUSE_KERNEL_PX, MASK_FUSE_KERNEL_PX), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def build_color_mask(hsv_img: np.ndarray, color_name: str) -> np.ndarray:
    ranges = COLOR_RANGES.get(color_name.lower())
    if not ranges:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv_img, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return _fuse_fragments(clean_mask(mask))


def height_gate_mask(mask: np.ndarray, depth_mm: Optional[np.ndarray]) -> np.ndarray:
    """
    Keep only mask pixels that rise at least OBJECT_MIN_HEIGHT_MM off the table.
    The table is the robust FAR plane of the depth image (90th percentile of
    valid depths — the table dominates the frame at hover/top view). This is
    per-pixel point-cloud reasoning: reflections, stains and shadows sit AT
    table depth and are physically excluded no matter how red they look.
    Falls back to the input mask when depth is missing/insufficient.
    """
    if depth_mm is None or depth_mm.shape[:2] != mask.shape[:2]:
        return mask

    valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
    if valid.size < 1000:
        return mask
    table_d = float(np.percentile(valid, 90))

    raised = np.isfinite(depth_mm) & (depth_mm > 0) & (depth_mm < table_d - OBJECT_MIN_HEIGHT_MM)
    gated = np.zeros_like(mask)
    gated[(mask > 0) & raised] = 255
    if int((gated > 0).sum()) < DEPTH_MIN_VALID_PX:
        return mask   # object too flat / depth too sparse — don't lose it entirely
    return _fuse_fragments(gated)


def color_mask_of(frame: np.ndarray, color_name: str) -> np.ndarray:
    """RGB frame → cleaned binary mask for the named color."""
    img_bgr = to_bgr(frame)
    hsv = enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
    return build_color_mask(hsv, color_name)


def top_face_mask(mask: np.ndarray, depth_mm: Optional[np.ndarray]) -> np.ndarray:
    """
    Restrict a color mask to the object's TOP face using aligned depth:
    keep only pixels within DEPTH_TOP_BAND_MM of the blob's nearest surface.
    Falls back to the input mask when depth is missing/insufficient.
    """
    if depth_mm is None or depth_mm.shape[:2] != mask.shape[:2]:
        return mask

    in_blob = mask > 0
    depths = depth_mm[in_blob]
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size < DEPTH_MIN_VALID_PX:
        return mask

    # Robust nearest surface (5th percentile rejects speckle noise).
    d_top = float(np.percentile(valid, 5))
    band = in_blob & np.isfinite(depth_mm) & (depth_mm > 0) & (depth_mm <= d_top + DEPTH_TOP_BAND_MM)
    if int(band.sum()) < DEPTH_MIN_VALID_PX:
        return mask

    refined = np.zeros_like(mask)
    refined[band] = 255
    return clean_mask(refined)


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BLOB_AREA_PX:
        return None
    return largest


def _contour_centroid(contour) -> Optional[Tuple[int, int]]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def _grasp_angle(contour) -> float:
    """minAreaRect angle folded into [-45, 45) via the cube's 90° symmetry."""
    (_, _), (_, _), raw_angle = cv2.minAreaRect(contour)
    angle = raw_angle % 90.0
    if angle >= 45.0:
        angle -= 90.0
    return angle


def find_object_centroid(frame: np.ndarray, color_name: str) -> Optional[Tuple[int, int]]:
    """
    Detect a colored object in the given (RGB) frame.
    Returns pixel (u, v) of the largest blob centroid, or None if not found.
    """
    contour = _largest_contour(color_mask_of(frame, color_name))
    if contour is None:
        return None
    return _contour_centroid(contour)


def find_object_blob(frame: np.ndarray, color_name: str, depth_mm: Optional[np.ndarray] = None):
    """
    Full blob detection for the servo/grasp phase. Returns
        (cx, cy, (bx, by, bw, bh), angle_deg)
    for the largest color blob, or None if not found.

    When aligned depth is provided, the mask is first HEIGHT-GATED (only pixels
    that rise off the table survive — kills table-level phantoms), then the
    centroid/angle are computed over the TOP-FACE submask (nearest-depth band)
    so an angled view of the cube's side can't drag the aim point toward a
    corner. The bounding box describes the gated blob (diagnostics/overlays).
    """
    mask = color_mask_of(frame, color_name)
    if depth_mm is not None:
        mask = height_gate_mask(mask, depth_mm)

    blob_contour = _largest_contour(mask)
    if blob_contour is None:
        return None
    bbox = cv2.boundingRect(blob_contour)

    contour = blob_contour
    if depth_mm is not None:
        refined = top_face_mask(mask, depth_mm)
        top_contour = _largest_contour(refined)
        if top_contour is not None:
            contour = top_contour

    centroid = _contour_centroid(contour)
    if centroid is None:
        return None
    cx, cy = centroid

    return cx, cy, bbox, _grasp_angle(contour)


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
    img_bgr = to_bgr(frame)

    if zone_roi is not None:
        x, y, w, h = zone_roi
        H_img, W_img = img_bgr.shape[:2]
        x = max(0, min(x, W_img - 1)); y = max(0, min(y, H_img - 1))
        w = max(1, min(w, W_img - x)); h = max(1, min(h, H_img - y))
        img_bgr = img_bgr[y:y + h, x:x + w]

    hsv  = enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
    mask = build_color_mask(hsv, color_name)
    px_count = int(np.sum(mask > 0))
    return px_count >= FRONT_MIN_PRESENCE_PX, px_count
