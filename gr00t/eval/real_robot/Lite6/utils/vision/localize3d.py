"""
3D object localization: pixel + aligned depth → camera-frame XYZ → robot base XYZ.

THE geometry layer of the pipeline (the old one-pose homography is gone):
intrinsics deproject/project pixels, and the calibrated extrinsics map between
the camera and base frames — valid at ANY arm pose, unlike the homography that
was only true at the exact pose it was calibrated from. Model: the camera is
rigidly mounted on the wrist, and all perception happens at a FIXED orientation
(roll/pitch/yaw = -180/0/0), so the camera→base mapping at any TCP position
p_tcp is

    p_base = p_tcp + R · p_cam + t

with CONSTANT R (3×3 rotation) and t (3,) solved once on hardware by
script/lite6_extrinsics.py (Kabsch least-squares over measured pairs) and
stored at EXTRINSICS_PATH. When per-pixel depth is unavailable, 2D→3D still
works by intersecting the pixel ray with a known horizontal plane
(pixel_to_base_on_plane) — e.g. the table plane estimated by estimate_table_z.
"""

import os
from typing import Optional, Tuple

import numpy as np

from utils.constants import EXTRINSICS_PATH, DEPTH_MIN_VALID_PX
from utils.vision.detection import color_mask_of, top_face_mask


# =============================================================================
# Pinhole deprojection
# =============================================================================

def deproject(u: float, v: float, depth_mm: float, intrinsics) -> np.ndarray:
    """Pixel (u, v) at depth_mm → camera-frame (X, Y, Z) in mm."""
    fx, fy, cx, cy = intrinsics
    z = float(depth_mm)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def robust_depth_at(depth_mm, u: int, v: int, half: int = 4) -> Optional[float]:
    """Median valid depth (mm) in a small patch around pixel (u, v); None if unusable."""
    if depth_mm is None:
        return None
    h, w = depth_mm.shape[:2]
    y0, y1 = max(0, v - half), min(h, v + half + 1)
    x0, x1 = max(0, u - half), min(w, u + half + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def project(p_cam: np.ndarray, intrinsics) -> Optional[Tuple[float, float]]:
    """
    Camera-frame point (mm) → pixel (u, v). Inverse of deproject.
    None when the point is at/behind the camera plane (Z <= 0).
    """
    fx, fy, cx, cy = intrinsics
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if z <= 0.0:
        return None
    return fx * x / z + cx, fy * y / z + cy


def localize_object_3d(frame_rgb, depth_mm, color_name: str, intrinsics,
                       blob_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Median 3D point (camera frame, mm) of the object's TOP face — the robust
    center of its point cloud. Returns None if the object or valid depth is
    missing.

    blob_mask (optional): restrict localization to this binary mask (the FSM
    fills the CHOSEN blob's contour). With several identical cubes in view the
    whole-color-mask median lands BETWEEN cubes; scoping it to one blob makes
    the median the center of that cube.
    """
    if depth_mm is None or intrinsics is None:
        return None

    base_mask = blob_mask if blob_mask is not None else color_mask_of(frame_rgb, color_name)
    mask = top_face_mask(base_mask, depth_mm)
    ys, xs = np.nonzero(mask)
    if xs.size < DEPTH_MIN_VALID_PX:
        return None

    ds = depth_mm[ys, xs]
    ok = np.isfinite(ds) & (ds > 0)
    if int(ok.sum()) < DEPTH_MIN_VALID_PX:
        return None
    xs, ys, ds = xs[ok], ys[ok], ds[ok]

    fx, fy, cx, cy = intrinsics
    pts = np.stack([(xs - cx) * ds / fx, (ys - cy) * ds / fy, ds], axis=1)
    return np.median(pts, axis=0)


# =============================================================================
# Camera → base extrinsics
# =============================================================================

def solve_extrinsics(cam_pts: np.ndarray, base_offsets: np.ndarray):
    """
    Least-squares rigid transform (Kabsch/SVD): find R, t minimizing
    ||R·p_cam + t − q|| over pairs, where q_i = p_base_i − p_tcp_i.

    cam_pts:      (N, 3) object positions in the CAMERA frame (mm)
    base_offsets: (N, 3) matching (p_base − p_tcp) offsets (mm)

    Returns (R (3,3), t (3,), rms_mm). Needs N ≥ 3 non-collinear pairs.
    """
    P = np.asarray(cam_pts, dtype=np.float64)
    Q = np.asarray(base_offsets, dtype=np.float64)
    if P.shape[0] < 3 or P.shape != Q.shape:
        raise ValueError(f"Need >=3 matched pairs; got {P.shape} vs {Q.shape}")

    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    H = (P - p_mean).T @ (Q - q_mean)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = q_mean - R @ p_mean

    residuals = (R @ P.T).T + t - Q
    rms = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))
    return R, t, rms


def save_extrinsics(R: np.ndarray, t: np.ndarray, rms: float, path: str = EXTRINSICS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, R=R, t=t, rms=rms)
    print(f"[Vision] Extrinsics saved to {path} (RMS {rms:.2f} mm).")


def load_extrinsics(path: str = EXTRINSICS_PATH) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Returns (R, t) or None if not calibrated yet."""
    if not os.path.exists(path):
        return None
    data = np.load(path)
    print(f"[Vision] Loaded camera→base extrinsics from {path} "
          f"(calibration RMS {float(data['rms']):.2f} mm).")
    return data["R"], data["t"]


def camera_to_base(p_cam: np.ndarray, tcp_xyz, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Camera-frame point (mm) → robot base frame (mm) at the given TCP position."""
    return np.asarray(tcp_xyz, dtype=np.float64) + R @ np.asarray(p_cam, dtype=np.float64) + t


def base_to_camera(p_base: np.ndarray, tcp_xyz, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Robot-base point (mm) → camera frame (mm) at the given TCP position.
    Exact inverse of camera_to_base (R is a rotation, so R⁻¹ = Rᵀ). Used to
    predict WHERE a known cube should appear in the wrist image (project()).
    """
    d = np.asarray(p_base, dtype=np.float64) - np.asarray(tcp_xyz, dtype=np.float64) - t
    return R.T @ d


def pixel_to_base_on_plane(u: float, v: float, plane_z: float, tcp_xyz,
                           R: np.ndarray, t: np.ndarray,
                           intrinsics) -> Optional[Tuple[float, float]]:
    """
    Depth-free 2D→3D: intersect the camera ray through pixel (u, v) with the
    horizontal base-frame plane z = plane_z. The pose-independent replacement
    for the old homography (ray origin = camera center = tcp + t; direction =
    R · [(u−cx)/fx, (v−cy)/fy, 1]). Returns base (x, y) mm, or None when the
    ray is parallel to the plane or the intersection is behind the camera.
    """
    fx, fy, cx, cy = intrinsics
    d_base = R @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    origin = np.asarray(tcp_xyz, dtype=np.float64) + t
    if abs(d_base[2]) < 1e-9:
        return None
    s = (plane_z - origin[2]) / d_base[2]
    if s <= 0:
        return None
    return float(origin[0] + s * d_base[0]), float(origin[1] + s * d_base[1])


def estimate_table_z(depth_mm, tcp_xyz, R: np.ndarray, t: np.ndarray,
                     intrinsics, max_samples: int = 500) -> Optional[float]:
    """
    Robust base-frame height of the TABLE plane from one aligned depth frame:
    the table is the far plane of a top view (same 90th-percentile convention
    as height_gate_mask), so take pixels within a tight band of that depth,
    deproject a subsample with their own depths, map to base, median z.
    Returns None when depth is too sparse to trust.
    """
    if depth_mm is None or intrinsics is None:
        return None
    valid_mask = np.isfinite(depth_mm) & (depth_mm > 0)
    valid = depth_mm[valid_mask]
    if valid.size < 1000:
        return None
    table_d = float(np.percentile(valid, 90))

    band = valid_mask & (np.abs(depth_mm - table_d) < 10.0)
    ys, xs = np.nonzero(band)
    if xs.size < 100:
        return None
    if xs.size > max_samples:
        idx = np.random.default_rng(0).choice(xs.size, max_samples, replace=False)
        xs, ys = xs[idx], ys[idx]

    fx, fy, cx, cy = intrinsics
    ds = depth_mm[ys, xs].astype(np.float64)
    pts_cam = np.stack([(xs - cx) * ds / fx, (ys - cy) * ds / fy, ds], axis=1)
    pts_base = (R @ pts_cam.T).T + np.asarray(tcp_xyz, dtype=np.float64) + t
    return float(np.median(pts_base[:, 2]))
