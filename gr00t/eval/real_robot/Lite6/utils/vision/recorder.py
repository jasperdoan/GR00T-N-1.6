"""
Debug/demo run recorder (--video flag): writes an MP4 of a 2x2 mosaic while the
task runs, from its own daemon thread so the control loop is untouched.

  ┌───────────────────────┬───────────────────────┐
  │ [A] color, annotated  │ [B] depth colormap    │
  │  ROI box, aim mark,   │  (placeholder when    │
  │  detected blob/angle  │   streaming, no depth)│
  ├───────────────────────┼───────────────────────┤
  │ [C] raw color mask    │ [D] height-gated /    │
  │                       │     top-face mask     │
  └───────────────────────┴───────────────────────┘

Panels C/D are the views that diagnose detection problems (mask fragmentation,
table-level phantoms). Default off; enable per-run with --video.
"""

import os
import math
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from utils.constants import GRIPPER_ROI, VIDEO_FPS
from utils.vision.detection import (
    color_mask_of,
    height_gate_mask,
    top_face_mask,
    find_object_blob,
)

_PANEL_W, _PANEL_H = 640, 360


def _fit(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (_PANEL_W, _PANEL_H))

def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


class RunRecorder:
    """Records the 2x2 debug mosaic to MP4 until stop() is called."""

    def __init__(self, cam, target_object: str, output_dir: str, fps: int = VIDEO_FPS):
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(output_dir, f"run_{stamp}.mp4")

        self._cam = cam
        self._color_name = target_object.split()[0].lower()
        self._fps = max(1, int(fps))
        self._writer = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps, (_PANEL_W * 2, _PANEL_H * 2),
        )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Video] Recording debug mosaic to {self.path} ({self._fps} fps)")

    def _render_point_cloud(self, frame_bgr, depth_mm):
        canvas = np.zeros((_PANEL_H, _PANEL_W, 3), dtype=np.uint8)
        if depth_mm is None:
            return _label(canvas, "3D Cloud (No Depth)")

        # Subsample for real-time video encoding performance
        step = 6
        h, w = depth_mm.shape[:2]
        ys = np.arange(0, h, step)
        xs = np.arange(0, w, step)
        xx, yy = np.meshgrid(xs, ys)

        z_mm = depth_mm[yy, xx]
        colors = frame_bgr[yy, xx]

        # Filter out invalid depth points (sensor noise or too far)
        valid = (z_mm > 20) & (z_mm < 1000) & np.isfinite(z_mm)
        if not np.any(valid):
            return _label(canvas, "3D Cloud (Empty)")

        x_px = xx[valid]
        y_px = yy[valid]
        z_c = z_mm[valid]
        c_rgb = colors[valid]

        # 1. Pinhole Deprojection (Converts Pixels -> Metric MM)
        fx, fy = 500.0, 500.0  # Approx intrinsics
        cx, cy = w / 2, h / 2
        
        x_c = (x_px - cx) * z_c / fx
        y_c = (y_px - cy) * z_c / fy
        
        # We now have a true (N, 3) metric point cloud!
        pts = np.vstack((x_c, y_c, z_c)).T

        # 2. Translate center of scene to origin (so we can rotate around it)
        z_center = np.median(z_c)
        pts[:, 2] -= z_center

        # 3. Apply Isometric 3D Rotation (Tilt Pitch and Pan Yaw)
        ax = math.radians(-50) # Pitch down 50 deg
        ay = math.radians(35)  # Pan right 35 deg
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(ax), -math.sin(ax)],
            [0, math.sin(ax), math.cos(ax)]
        ])
        Ry = np.array([
            [math.cos(ay), 0, math.sin(ay)],
            [0, 1, 0],
            [-math.sin(ay), 0, math.cos(ay)]
        ])
        
        pts = pts @ Ry.T @ Rx.T

        # 4. Push back along Z and Project to Virtual 2D Camera
        pts[:, 2] += z_center * 1.5  # Push out slightly to fit in frame
        
        v_fx, v_fy = 400.0, 400.0  # Virtual camera focal length
        v_cx, v_cy = _PANEL_W / 2, _PANEL_H / 2

        z_safe = np.maximum(pts[:, 2], 1.0) # Prevent division by zero
        proj_x = (pts[:, 0] * v_fx / z_safe) + v_cx
        proj_y = (pts[:, 1] * v_fy / z_safe) + v_cy

        # 5. Draw
        proj_x = proj_x.astype(np.int32)
        proj_y = proj_y.astype(np.int32)
        
        for px, py, color in zip(proj_x, proj_y, c_rgb):
            if 0 <= px < _PANEL_W and 0 <= py < _PANEL_H:
                cv2.circle(canvas, (px, py), 1, (int(color[0]), int(color[1]), int(color[2])), -1)

        return _label(canvas, "Metric 3D Cloud")

    def _compose(self, frame_bgr, depth_mm):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # [A] Annotated color
        a = frame_bgr.copy()
        rx, ry, rw, rh = GRIPPER_ROI
        cv2.rectangle(a, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
        cv2.drawMarker(a, (rx + rw // 2, ry + rh // 2), (255, 255, 0),
                       cv2.MARKER_CROSS, 24, 2)
        blob = find_object_blob(frame_rgb, self._color_name, depth_mm=depth_mm)
        if blob is not None:
            cx, cy, (bx, by, bw, bh), angle = blob
            cv2.rectangle(a, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.drawMarker(a, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
            cv2.putText(a, f"{angle:+.1f} deg", (bx, max(0, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        a = _label(_fit(a), "color + target center")

        # [B] Depth colormap
        if depth_mm is not None:
            d8 = np.clip(depth_mm / 4.0, 0, 255).astype(np.uint8)   
            b = _label(_fit(cv2.applyColorMap(d8, cv2.COLORMAP_JET)), "depth (mm)")
        else:
            b = _label(_fit(None), "no depth (stream mode)")

        # [C] Raw color mask
        mask = color_mask_of(frame_rgb, self._color_name)
        c = _label(_fit(mask), "color mask")
        
        # [D] 3D Point cloud projection
        d = self._render_point_cloud(frame_bgr, depth_mm)

        return np.vstack([np.hstack([a, b]), np.hstack([c, d])])

    def _loop(self):
        interval = 1.0 / self._fps
        while self._running:
            tic = time.time()
            try:
                ret, frame = self._cam.read()
                if ret:
                    depth = None
                    if getattr(self._cam, "has_depth", False):
                        dret, d = self._cam.read_depth()
                        if dret:
                            depth = d
                    self._writer.write(self._compose(frame, depth))
            except Exception:
                pass   # never let a debug recorder take down the run
            time.sleep(max(0.0, interval - (time.time() - tic)))

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=2.0)
        self._writer.release()
        print(f"[Video] Recording saved: {self.path}")
