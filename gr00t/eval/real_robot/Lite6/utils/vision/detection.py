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

from collections import namedtuple
from typing import List, Optional, Tuple

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
    ZONE_BLOB_MIN_AREA_PX,
    ZONE_BLOB_MAX_AREA_PX,
    SERVO_MAX_BLOB_AREA_PX,
    BLOB_MIN_SOLIDITY,
    BLOB_MAX_ASPECT,
    ZONE_BLOB_MIN_SOLIDITY,
    ZONE_BLOB_MAX_ASPECT,
    ONE_CUBE_AREA_PX,
)
from utils.vision.helpers import enhance_saturation, clean_mask, to_bgr


# One detected color blob, in FULL-FRAME pixel coordinates (the frame the
# intrinsics and snapshot overlays are calibrated in — zone scoping is a
# membership test, never a crop, so no offset bookkeeping).
BlobCandidate = namedtuple("BlobCandidate", "cx cy area bbox angle contour")


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


def _roi_contains(roi, u: int, v: int) -> bool:
    """True when pixel (u, v) lies inside roi=(x, y, w, h) (half-open bounds)."""
    x, y, w, h = roi
    return x <= u < x + w and y <= v < y + h


def _cube_shaped(contour, area: float, min_solidity: float = BLOB_MIN_SOLIDITY,
                 max_aspect: float = BLOB_MAX_ASPECT) -> bool:
    """
    Shape validation: a cube seen top-down is COMPACT — solidity
    (area / convex-hull area) near 1 and a squat minAreaRect. Glare streaks,
    cables and mask ribbons fail; a half-shadowed or rotated cube passes.
    Grasp-time callers use the strict (BLOB_MIN_SOLIDITY/BLOB_MAX_ASPECT)
    defaults; zone-COUNTING callers pass the looser ZONE_BLOB_* bounds so a
    cube seen mostly edge-on (wedged against a neighbor) still counts.
    """
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0 and area / hull_area < min_solidity:
        return False
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if min(w, h) > 0 and max(w, h) / min(w, h) > max_aspect:
        return False
    return True


def _split_touching_blob(contour, depth_mm: Optional[np.ndarray]):
    """
    Attempt to separate an oversized blob (likely two cubes fused by the
    CLOSE-kernel morphology) into its constituent contours via a classic
    distance-transform + watershed split. Returns a list of contours (>= 2)
    on a clean split, or None when the split doesn't yield well-separated
    regions (caller keeps treating it as a single blob).
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    x0, y0 = max(0, x - pad), max(0, y - pad)
    local = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    cv2.drawContours(local, [contour], -1, 255, thickness=cv2.FILLED,
                     offset=(-x0, -y0))

    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    if dist.max() < 1.0:
        return None
    # Peaks in the distance transform = candidate cube centers.
    _ret, peaks = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    peaks = peaks.astype(np.uint8)
    n_peaks, markers = cv2.connectedComponents(peaks)
    if n_peaks < 3:   # background (0) + at least 2 cube peaks
        return None

    markers = markers + 1          # watershed reserves 0 for "unknown"
    unknown = cv2.subtract(local, peaks)
    markers[unknown == 255] = 0
    local_bgr = cv2.cvtColor(local, cv2.COLOR_GRAY2BGR)
    cv2.watershed(local_bgr, markers)

    contours = []
    for label in range(2, n_peaks + 1):
        region = np.zeros_like(local)
        region[markers == label] = 255
        cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if cv2.contourArea(c) < ONE_CUBE_AREA_PX * 0.4:
            continue   # split sliver, not a real second cube
        # Shift back to full-frame coordinates.
        c = c + np.array([[x0, y0]])
        contours.append(c)

    return contours if len(contours) >= 2 else None


def _clip_roi(roi, shape) -> Tuple[int, int, int, int]:
    """Clamp roi=(x, y, w, h) to the image bounds (same rule as check_color_presence)."""
    x, y, w, h = roi
    H_img, W_img = shape[:2]
    x = max(0, min(x, W_img - 1)); y = max(0, min(y, H_img - 1))
    w = max(1, min(w, W_img - x)); h = max(1, min(h, H_img - y))
    return x, y, w, h


def find_all_blobs(
    frame: np.ndarray,
    color_name: str,
    depth_mm: Optional[np.ndarray] = None,
    zone_roi=None,
    min_area: float = MIN_BLOB_AREA_PX,
    max_area: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    shape_gate: bool = True,
    min_solidity: float = BLOB_MIN_SOLIDITY,
    max_aspect: float = BLOB_MAX_ASPECT,
    try_split: bool = False,
) -> List[BlobCandidate]:
    """
    Every color blob in the FULL frame (multi-object aware — no winner-take-all),
    height-gated when depth is given, filtered by:
      - area >= min_area, and <= max_area when set (see try_split below).
      - shape gates (solidity + aspect), unless shape_gate=False. Grasp-time
        callers (the default) use the strict BLOB_MIN_SOLIDITY/BLOB_MAX_ASPECT;
        zone-COUNTING callers pass the looser ZONE_BLOB_* bounds so a cube seen
        mostly edge-on (wedged against a neighbor) still counts as present.
      - centroid inside zone_roi when given. The CENTROID decides zone
        membership: a straddling cube belongs to the zone its centroid is in.
    try_split=True (zone counting only): a blob in roughly [1.5x, 3x]
    ONE_CUBE_AREA_PX — or exceeding max_area, if set — is likely two cubes
    fused by the CLOSE kernel; attempt a distance-transform/watershed split
    before falling back to the merged-blob exclusion. A clean split emits
    each half as its own candidate; an unclean one (or one that also exceeds
    max_area) is excluded as today.
    Pass mask= to reuse an already-computed color mask.
    Returns BlobCandidates sorted largest-first.
    """
    if mask is None:
        mask = color_mask_of(frame, color_name)
        if depth_mm is not None:
            mask = height_gate_mask(mask, depth_mm)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def _accept(c, area) -> Optional[BlobCandidate]:
        centroid = _contour_centroid(c)
        if centroid is None:
            return None
        cx, cy = centroid
        if area < min_area:
            return None
        if shape_gate and not _cube_shaped(c, area, min_solidity, max_aspect):
            return None
        if zone_roi is not None and not _roi_contains(zone_roi, cx, cy):
            return None
        return BlobCandidate(cx, cy, float(area), cv2.boundingRect(c), _grasp_angle(c), c)

    blobs: List[BlobCandidate] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        centroid = _contour_centroid(c)
        if centroid is None:
            continue
        cx, cy = centroid

        oversized = max_area is not None and area > max_area
        # A blob well beyond one cube's reference area is worth a split
        # attempt regardless of whether an explicit max_area guard is
        # configured (ZONE_BLOB_MAX_AREA_PX defaults to None/disabled) — this
        # is the primary way two touching cubes get counted as two.
        looks_like_two = ONE_CUBE_AREA_PX * 1.5 <= area <= ONE_CUBE_AREA_PX * 3.0
        if (oversized or looks_like_two) and try_split:
            parts = _split_touching_blob(c, depth_mm)
            if parts is not None:
                accepted = [b for p in parts
                           for b in [_accept(p, cv2.contourArea(p))] if b is not None]
                if len(accepted) >= 2:
                    print(f"[Vision] Blob at ({cx}, {cy}) area {area:.0f}px split "
                          f"into {len(accepted)} touching cubes.")
                    blobs.extend(accepted)
                    continue
        if oversized:
            print(f"[Vision] Blob at ({cx}, {cy}) area {area:.0f}px exceeds {max_area:.0f}px "
                  f"— likely touching cubes (centroid = the seam); skipping it.")
            continue

        b = _accept(c, area)
        if b is None:
            if shape_gate and not _cube_shaped(c, area, min_solidity, max_aspect):
                print(f"[Vision] Blob at ({cx}, {cy}) area {area:.0f}px rejected by "
                      f"shape gates (not compact/squat enough for a cube).")
            continue
        blobs.append(b)

    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs


def count_objects_in_zone(
    frame: np.ndarray,
    target_object: str,
    zone_roi,
    depth_mm: Optional[np.ndarray] = None,
) -> Tuple[int, List[BlobCandidate], int]:
    """
    Multi-object zone census for PRE_CHECK / VERIFY / task selection:
      count   — blobs (ZONE_BLOB_MIN/MAX area gates) whose centroid is in the ROI
      blobs   — those BlobCandidates (full-frame coords), largest-first
      px_mass — total mask pixels inside the ROI (same measure as
                check_color_presence; VERIFY's merged-blob backstop)
    """
    color_name = target_object.split()[0].lower()
    mask = color_mask_of(frame, color_name)
    if depth_mm is not None:
        mask = height_gate_mask(mask, depth_mm)

    # Looser shape/solidity than grasp-time selection (a cube worth COUNTING
    # may be partially occluded / seen edge-on against a neighbor), plus a
    # touching-cube split attempt so two fused cubes still count as two.
    blobs = find_all_blobs(frame, color_name, zone_roi=zone_roi,
                           min_area=ZONE_BLOB_MIN_AREA_PX,
                           max_area=ZONE_BLOB_MAX_AREA_PX, mask=mask,
                           min_solidity=ZONE_BLOB_MIN_SOLIDITY,
                           max_aspect=ZONE_BLOB_MAX_ASPECT,
                           try_split=True)
    x, y, w, h = _clip_roi(zone_roi, mask.shape) if zone_roi is not None \
        else (0, 0, mask.shape[1], mask.shape[0])
    px_mass = int(np.sum(mask[y:y + h, x:x + w] > 0))
    return len(blobs), blobs, px_mass


def count_all_colors_in_zone(
    frame: np.ndarray,
    zone_roi,
    depth_mm: Optional[np.ndarray] = None,
) -> Tuple[int, List[BlobCandidate]]:
    """
    Occupancy census across EVERY color in COLOR_RANGES: drop clearance and
    zone-full detection must see cubes of ANY color, not just the task's
    target color (a blue cube in storage still occupies a drop spot when
    sorting red). Same per-color path as count_objects_in_zone (mask →
    height gate → ZONE_* blob gates → touching-cube split). COLOR_RANGES
    entries are pairwise disjoint by contract, so summing per-color blobs
    cannot double-count one cube. Returns (total_count, blobs largest-first).
    """
    all_blobs: List[BlobCandidate] = []
    for color_name in COLOR_RANGES:
        count, blobs, _px = count_objects_in_zone(
            frame, f"{color_name} cube", zone_roi, depth_mm=depth_mm)
        all_blobs.extend(blobs)
    all_blobs.sort(key=lambda b: b.area, reverse=True)
    return len(all_blobs), all_blobs


def select_blob_near(
    blobs: List[BlobCandidate],
    near_px: Tuple[int, int],
    max_dist_px: Optional[float] = None,
) -> Optional[BlobCandidate]:
    """
    The candidate whose centroid is nearest near_px=(u, v); None when the list
    is empty or the nearest exceeds max_dist_px (the servo's target-continuity
    gate: a huge inter-frame jump means a DIFFERENT cube, not ours moving).
    """
    if not blobs:
        return None
    u, v = near_px
    best = min(blobs, key=lambda b: (b.cx - u) ** 2 + (b.cy - v) ** 2)
    if max_dist_px is not None:
        if (best.cx - u) ** 2 + (best.cy - v) ** 2 > max_dist_px ** 2:
            return None
    return best


def find_object_centroid(frame: np.ndarray, color_name: str) -> Optional[Tuple[int, int]]:
    """
    Detect a colored object in the given (RGB) frame.
    Returns pixel (u, v) of the largest blob centroid, or None if not found.
    """
    contour = _largest_contour(color_mask_of(frame, color_name))
    if contour is None:
        return None
    return _contour_centroid(contour)


def find_object_blob(frame: np.ndarray, color_name: str, depth_mm: Optional[np.ndarray] = None,
                     near_px: Optional[Tuple[int, int]] = None,
                     max_dist_px: Optional[float] = None):
    """
    Full blob detection for the servo/grasp phase. Returns
        (cx, cy, (bx, by, bw, bh), angle_deg)
    or None if not found.

    Selection: largest blob by default; with near_px the blob whose centroid is
    nearest that point wins instead (multi-object: track OUR cube, not whichever
    is biggest in the wrist view), rejected when farther than max_dist_px.

    When aligned depth is provided, the mask is first HEIGHT-GATED (only pixels
    that rise off the table survive — kills table-level phantoms), then the
    centroid/angle are computed over the SELECTED blob's TOP-FACE submask
    (nearest-depth band, scoped to that one blob so a neighbor cube at the same
    height can't hijack the refinement) — an angled view of the cube's side
    can't drag the aim point toward a corner. The bounding box describes the
    gated blob (diagnostics/overlays).
    """
    mask = color_mask_of(frame, color_name)
    if depth_mm is not None:
        mask = height_gate_mask(mask, depth_mm)

    if near_px is not None:
        candidates = find_all_blobs(frame, color_name, mask=mask)
        chosen = select_blob_near(candidates, near_px, max_dist_px=max_dist_px)
        blob_contour = chosen.contour if chosen is not None else None
    else:
        blob_contour = _largest_contour(mask)
    if blob_contour is None:
        return None

    if SERVO_MAX_BLOB_AREA_PX is not None:
        area = cv2.contourArea(blob_contour)
        if area > SERVO_MAX_BLOB_AREA_PX:
            # Warn-only: suction feedback / VERIFY catch a failed seam grasp.
            print(f"[Vision] WARNING: servo blob area {area:.0f}px exceeds "
                  f"{SERVO_MAX_BLOB_AREA_PX:.0f}px — possibly two touching cubes.")

    return refine_blob(mask, blob_contour, depth_mm)


def refine_blob(mask: np.ndarray, contour, depth_mm: Optional[np.ndarray] = None):
    """
    Grasp-quality measurement of ONE selected blob:
        (cx, cy, (bx, by, bw, bh), angle_deg)
    With aligned depth, the centroid/angle come from the blob-scoped TOP-FACE
    submask (nearest-depth band restricted to this contour, so a neighbor cube
    at the same height can't hijack the refinement); the bounding box always
    describes the full input contour (diagnostics/overlays). Falls back to the
    raw contour when depth is missing/unusable. None if the centroid degenerates.
    """
    bbox = cv2.boundingRect(contour)

    refined_contour = contour
    if depth_mm is not None:
        blob_only = np.zeros_like(mask)
        cv2.drawContours(blob_only, [contour], -1, 255, thickness=cv2.FILLED)
        refined = top_face_mask(cv2.bitwise_and(mask, blob_only), depth_mm)
        top_contour = _largest_contour(refined)
        if top_contour is not None:
            refined_contour = top_contour

    centroid = _contour_centroid(refined_contour)
    if centroid is None:
        return None
    cx, cy = centroid

    return cx, cy, bbox, _grasp_angle(refined_contour)


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
