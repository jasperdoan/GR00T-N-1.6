"""Annotated workspace snapshots (zone overlays + detected-object boxes)."""

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from utils.constants import COLOR_RANGES, MIN_BLOB_AREA_PX
from utils.vision.helpers import enhance_saturation, to_bgr
from utils.vision.detection import build_color_mask


def save_workspace_snapshot(
    frame: np.ndarray,
    filename: str,
    target_object: str,
    output_dir: str,
    zones_dict: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
):
    """
    Save an annotated top-down snapshot.
    Draws white zone boxes and a green bounding box around the detected object.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, os.path.basename(filename))

    img_bgr = to_bgr(frame)

    color_name = target_object.split()[0].lower()
    ranges = COLOR_RANGES.get(color_name)

    # Draw zone boxes
    if zones_dict:
        for zone_name, (zx, zy, zw, zh) in zones_dict.items():
            cv2.rectangle(img_bgr, (zx, zy), (zx + zw, zy + zh), (255, 255, 255), 1)
            cv2.putText(img_bgr, zone_name, (zx, max(0, zy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw object detection box
    if ranges:
        hsv  = enhance_saturation(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), factor=1.4)
        mask = build_color_mask(hsv, color_name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_BLOB_AREA_PX:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(img_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(img_bgr, target_object.title(), (bx, max(0, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(filepath, img_bgr)
    print(f"[Vision] Snapshot saved to {filepath}")
