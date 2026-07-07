"""Homography pixel→robot mapping (2D localization, calibrated at TOP_VIEW_POSE)."""

import os
from typing import Optional, Tuple

import numpy as np

from utils.constants import MATRIX_PATH


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
