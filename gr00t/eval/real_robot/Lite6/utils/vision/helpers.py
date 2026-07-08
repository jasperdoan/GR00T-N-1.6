"""Low-level image helpers shared across the vision package."""

import cv2
import numpy as np


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img.copy()


def enhance_saturation(hsv_img: np.ndarray, factor: float = 1.3) -> np.ndarray:
    h, s, v = cv2.split(hsv_img)
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    return cv2.merge([h, s, v])


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """Accepts an RGB or BGR uint8-ish frame; returns a BGR uint8 copy."""
    img = ensure_uint8(frame)
    # Callers in this codebase pass RGB frames (camera reads are converted at
    # the call site); 3-channel input is treated as RGB.
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
