"""
3D object localization: pixel + aligned depth → camera-frame XYZ → robot base XYZ.

Replaces homography scanning when a depth camera + calibrated extrinsics are
available. Model: the camera is rigidly mounted on the wrist, and all
perception happens at a FIXED orientation (roll/pitch/yaw = -180/0/0), so the
camera→base mapping at any TCP position p_tcp is

    p_base = p_tcp + R · p_cam + t

with CONSTANT R (3×3 rotation) and t (3,) solved once on hardware by
script/lite6_extrinsics.py (Kabsch least-squares over measured pairs) and
stored at EXTRINSICS_PATH.
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


def localize_object_3d(frame_rgb, depth_mm, color_name: str, intrinsics) -> Optional[np.ndarray]:
    """
    Median 3D point (camera frame, mm) of the object's TOP face — the robust
    center of its point cloud. Returns None if the object or valid depth is
    missing.
    """
    if depth_mm is None or intrinsics is None:
        return None

    mask = top_face_mask(color_mask_of(frame_rgb, color_name), depth_mm)
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
