"""
Camera self-test — FIRST thing to run when the Orbbec Gemini arrives.

    python3 script/lite6_camera_test.py --camera 0                       # local Orbbec
    python3 script/lite6_camera_test.py --camera "http://IP:9988/stream.mjpg"

Prints color/depth shapes, FPS, center-pixel depth, and intrinsics, then saves
a side-by-side snapshot (color | depth colormap) to data/camera_test.jpg.
Run from the Lite6 directory (imports utils/*).
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from utils.constants import CAMERA_SOURCE, BASE_DIR
from utils.vision import open_camera


def main():
    parser = argparse.ArgumentParser(description="Lite6 camera self-test")
    parser.add_argument("--camera", type=str, default=None,
                        help="Device index (e.g. 0) or stream URL; defaults to CAMERA_SOURCE")
    parser.add_argument("--seconds", type=float, default=3.0, help="Sampling duration")
    args = parser.parse_args()

    source = CAMERA_SOURCE
    if args.camera is not None:
        source = int(args.camera) if args.camera.isdigit() else args.camera

    print(f"[Test] Opening camera source: {source!r}")
    cam = open_camera(source, "test camera")

    print(f"[Test] has_depth  = {cam.has_depth}")
    print(f"[Test] intrinsics = {getattr(cam, 'intrinsics', None)}")

    # Sample frames for FPS + stability
    n, t0 = 0, time.time()
    frame = depth = None
    while time.time() - t0 < args.seconds:
        ret, f = cam.read()
        if ret:
            frame = f
            n += 1
        if cam.has_depth:
            dret, d = cam.read_depth()
            if dret:
                depth = d
        time.sleep(0.01)
    elapsed = time.time() - t0

    if frame is None:
        print("[Test] FAILED — no color frames received.")
        cam.release()
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"[Test] color: {w}x{h}, ~{n / elapsed:.1f} fps sampled")

    panels = [frame]
    if depth is not None:
        dh, dw = depth.shape[:2]
        cy, cx = dh // 2, dw // 2
        center = depth[cy - 2:cy + 3, cx - 2:cx + 3]
        valid = center[np.isfinite(center) & (center > 0)]
        center_mm = float(np.median(valid)) if valid.size else float("nan")
        cover = float(np.mean(depth > 0)) * 100.0
        print(f"[Test] depth: {dw}x{dh} aligned to color, center ≈ {center_mm:.0f} mm, "
              f"valid coverage {cover:.0f}%")
        print("[Test] Tip: if coverage is poor on the table, try a different depth "
              "mode/preset in OrbbecViewer and note which works best.")
        # Colormap panel for the snapshot
        d8 = np.clip(depth / 4.0, 0, 255).astype(np.uint8)   # 0-1020mm span
        dcol = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
        if dcol.shape[:2] != (h, w):
            dcol = cv2.resize(dcol, (w, h))
        panels.append(dcol)
    else:
        print("[Test] depth: none (stream source, or depth frames not arriving)")

    out = os.path.join(BASE_DIR, "data", "camera_test.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.hstack(panels))
    print(f"[Test] Snapshot saved to {out}")

    cam.release()
    print("[Test] PASSED.")


if __name__ == "__main__":
    main()
