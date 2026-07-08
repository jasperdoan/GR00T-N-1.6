"""
Fine visual servoing (IBVS P-controller) with lost-target recovery.

Drive:  nudge the arm in XY so the blob CENTROID moves toward the center of
        GRIPPER_ROI. The pixel error is mapped to a robot-frame delta via the
        HOMOGRAPHY: Δ = SERVO_GAIN · (H(p_obj) − H(p_aim)). H's absolute
        mapping is only valid at TOP_VIEW_POSE, but the camera orientation is
        identical at the hover pose (rigid mount, same roll/pitch/yaw), so H's
        DIRECTION is always right — only its scale is off, which the gain +
        step clamp absorb. No hand-tuned axis signs.

Accept: blob centroid within CENTER_LOCK_TOL_PX of the aim for
        SERVO_CONFIRM_FRAMES consecutive frames while the arm stands still
        (rotation-invariant; bbox containment was unsatisfiable for rotated
        cubes). With a depth camera, the centroid is the TOP-FACE centroid, so
        an angled view can't drag the aim toward a corner.

Lost target: (1) return to the position where the blob was last seen, halving
        the step gain each loss episode; (2) only if still invisible there,
        expanding square spiral; exhaustion aborts early for an FSM re-scan.
"""

import time
from typing import Optional, Tuple

import cv2

import numpy as np

from utils.constants import (
    SERVO_GAIN,
    SERVO_GAIN_3D,
    MAX_SERVO_STEP_MM,
    SERVO_DEADBAND_PX,
    GRIPPER_ROI,
    CENTER_LOCK_TOL_PX,
    SERVO_CONFIRM_FRAMES,
    SERVO_SETTLE_S,
    SEARCH_START_DELAY_S,
    SEARCH_STEP_MM,
    SEARCH_MAX_STEPS,
    FINE_ADJUST_SPEED,
)
from utils.vision.helpers import clamp
from utils.vision.camera import read_fresh
from utils.vision.detection import find_object_blob
from utils.vision.homography import pixel_to_robot
from utils.vision.localize3d import deproject
from utils.vision.safety import SafetyMonitor


def _spiral_offsets(step_mm: float):
    """
    Infinite generator of cumulative (dx, dy) offsets tracing an expanding
    square spiral: legs of 1,1,2,2,3,3,... steps in directions +X, +Y, -X, -Y.
    Used to sweep the camera around the approach point when the target is lost.
    """
    x = y = 0.0
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    d = 0
    run = 1
    while True:
        for _ in range(2):
            ddx, ddy = dirs[d]
            for _ in range(run):
                x += ddx * step_mm
                y += ddy * step_mm
                yield x, y
            d = (d + 1) % 4
        run += 1


def _read_depth_if_available(cap):
    if getattr(cap, "has_depth", False):
        ret, depth = cap.read_depth()
        if ret:
            return depth
    return None


def _robust_depth_at(depth_mm, u: int, v: int, half: int = 4):
    """Median valid depth in a small patch around (u, v); None if unusable."""
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


def _metric_step(obj_px, aim_px, depth_mm, intrinsics, extrinsics):
    """
    Exact base-frame XY step from pixel error using depth + calibration:
        Δ_base = R @ (deproject(obj) − deproject(aim))     [same depth plane]
    Direction AND scale are calibration-true, so no homography scale guess.
    Returns (dx, dy) in mm or None when depth at the object is unusable.
    """
    d = _robust_depth_at(depth_mm, obj_px[0], obj_px[1])
    if d is None:
        return None
    p_obj = deproject(obj_px[0], obj_px[1], d, intrinsics)
    p_aim = deproject(aim_px[0], aim_px[1], d, intrinsics)
    R, _t = extrinsics
    delta = R @ (p_obj - p_aim)
    return float(delta[0]), float(delta[1])


def visual_servo_to_grasp(
    cap,
    robot,
    target_object: str,
    H,
    timeout: float = 15.0,
    safety_monitor: Optional[SafetyMonitor] = None,
    should_stop_cb=None,
    intrinsics=None,
    extrinsics=None,
) -> Tuple[bool, Optional[float]]:
    """
    Returns (locked, angle_deg):
      - locked: True once the centroid lock criterion holds.
      - angle_deg: the object's in-image rotation ([-45, 45), from the LAST
        confirmation frame) so the caller can yaw-align the gripper without
        another camera read. None when not locked.

    Step mapping: with intrinsics + extrinsics + depth (Orbbec runs), pixel
    error converts to an EXACT base-frame delta (SERVO_GAIN_3D). Without them
    (stream runs), the homography-direction step (SERVO_GAIN) is used.
    """
    use_3d = intrinsics is not None and extrinsics is not None
    if H is None and not use_3d:
        print("[Vision] No homography and no 3D calibration — cannot map pixel error.")
        return False, None

    color_name = target_object.split()[0].lower()
    rx, ry, rw, rh = GRIPPER_ROI
    aim_x = rx + rw // 2
    aim_y = ry + rh // 2
    aim_robot = pixel_to_robot(aim_x, aim_y, H) if H is not None else None

    start_time = time.time()
    confirm_count = 0
    last_angle = None

    # Lost-target recovery state
    last_seen = time.time()
    last_seen_pos = None       # arm position at the most recent detection
    returned_to_seen = False   # tried the return-to-last-seen undo this episode?
    loss_backoff = 1.0         # step-size annealing: halved after each loss
                               # episode so a step can't repeatedly overshoot
                               # into a detection dead-zone
    search_center = None
    spiral_iter = None
    search_steps = 0

    print(f"[Vision] Visual servoing '{target_object}' into gripper ROI {GRIPPER_ROI}...")

    while True:
        if should_stop_cb and should_stop_cb():
            print("[Vision] Stop requested during visual servoing.")
            return False, None

        if time.time() - start_time > timeout:
            print("[Vision] Visual servoing TIMEOUT.")
            return False, None

        if robot.has_error():
            print("[Vision] Arm in error state during servoing — aborting.")
            return False, None

        ret, frame_bgr = read_fresh(cap)
        if not ret:
            time.sleep(0.033)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Hand safety: pause the arm, hold until clear, then resume ---
        if safety_monitor:
            safety_monitor.update_frame(frame_rgb)
            if safety_monitor.is_hand_present():
                print("[SAFETY] Hand detected during servoing — pausing arm...")
                robot.pause()
                while safety_monitor.is_hand_present():
                    time.sleep(0.05)
                    r2, f2 = read_fresh(cap)
                    if r2:
                        safety_monitor.update_frame(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
                robot.resume()
                print("[SAFETY] Hand removed — resuming servoing.")
                start_time = time.time()   # don't penalise the pause against the timeout
                confirm_count = 0
                continue

        depth_mm = _read_depth_if_available(cap)
        blob = find_object_blob(frame_rgb, color_name, depth_mm=depth_mm)
        if blob is None:
            confirm_count = 0

            # Brief grace window: don't launch a search over one flickered frame.
            if time.time() - last_seen < SEARCH_START_DELAY_S:
                time.sleep(0.033)
                continue

            # --- First recovery: undo the last approach. Losses cluster at the
            # final-approach frontier (frame-edge clipping / gripper shadow);
            # returning to the exact spot where the blob was last visible
            # breaks the servo↔search limit cycle. ---
            if last_seen_pos is not None and not returned_to_seen:
                returned_to_seen = True
                loss_backoff = max(0.125, loss_backoff * 0.5)
                print(f"[Vision] Target lost — returning to last-seen position "
                      f"({last_seen_pos[0]:.1f}, {last_seen_pos[1]:.1f}) "
                      f"[step backoff → {loss_backoff:.3f}]...")
                if not robot.move_to(last_seen_pos[0], last_seen_pos[1], last_seen_pos[2],
                                     speed=FINE_ADJUST_SPEED, wait=True):
                    print("[Vision] Return move failed (rejected/faulted) — aborting.")
                    return False, None
                time.sleep(SERVO_SETTLE_S)
                continue   # re-read; blob should be visible here again

            # --- Second recovery: spiral search sweeping the camera around the
            # last position until the blob re-enters the frame. ---
            if search_center is None:
                pos0 = robot.get_current_position()
                if pos0 is None:
                    time.sleep(0.033)
                    continue
                search_center = (pos0[0], pos0[1], pos0[2])
                spiral_iter = _spiral_offsets(SEARCH_STEP_MM)
                search_steps = 0
                print(f"[Vision] Target lost — spiral search around "
                      f"({search_center[0]:.1f}, {search_center[1]:.1f})...")

            moved = False
            while search_steps < SEARCH_MAX_STEPS:
                search_steps += 1
                off_x, off_y = next(spiral_iter)
                tx = search_center[0] + off_x
                ty = search_center[1] + off_y
                tz = search_center[2]
                if not robot.in_workspace(tx, ty, tz):
                    continue   # unreachable spiral point; budget still consumed
                print(f"[Vision] Spiral search step {search_steps}/{SEARCH_MAX_STEPS} "
                      f"→ ({tx:.1f}, {ty:.1f})")
                if not robot.move_to(tx, ty, tz, speed=FINE_ADJUST_SPEED, wait=True):
                    print("[Vision] Search nudge failed (rejected/faulted) — aborting.")
                    return False, None
                moved = True
                time.sleep(SERVO_SETTLE_S)   # post-move frame, not a stale one
                # Search has its own budget; don't let it eat the servo timeout.
                start_time = time.time()
                break

            if not moved:
                print("[Vision] Target lost — spiral search exhausted. Aborting for re-scan.")
                return False, None
            continue   # loop re-reads the camera and re-detects after the nudge

        obj_cx, obj_cy, bbox, angle = blob
        last_seen = time.time()
        last_seen_pos = robot.get_current_position() or last_seen_pos
        if search_center is not None or returned_to_seen:
            print("[Vision] Target reacquired — resuming servoing.")
        returned_to_seen = False
        search_center = None
        spiral_iter = None
        search_steps = 0

        err_x = obj_cx - aim_x
        err_y = obj_cy - aim_y

        # --- Acceptance: centroid within tolerance of the aim point, held for
        # N consecutive frames while the arm stands still. ---
        if abs(err_x) <= CENTER_LOCK_TOL_PX and abs(err_y) <= CENTER_LOCK_TOL_PX:
            confirm_count += 1
            last_angle = angle
            if confirm_count >= SERVO_CONFIRM_FRAMES:
                print(f"[Vision] Centroid ({err_x:+d},{err_y:+d}) px from aim for "
                      f"{confirm_count} frames. Locked (angle={last_angle:.1f}°).")
                return True, last_angle
            time.sleep(SERVO_SETTLE_S)   # hold still; re-read to confirm stability
            continue
        confirm_count = 0

        # --- Drive: centroid error toward the ROI center ---
        cmd_x = aim_x if abs(err_x) < SERVO_DEADBAND_PX else obj_cx
        cmd_y = aim_y if abs(err_y) < SERVO_DEADBAND_PX else obj_cy

        pos = robot.get_current_position()
        if pos is None:
            time.sleep(0.033)
            continue

        # Step mapping, best available first:
        #   3D metric (depth + intrinsics + extrinsics): exact base-frame delta.
        #   Homography direction (stream mode): direction exact, scale damped.
        # Both annealed by the loss backoff and bounded by the step clamp.
        step = None
        if use_3d and depth_mm is not None:
            step = _metric_step((cmd_x, cmd_y), (aim_x, aim_y), depth_mm, intrinsics, extrinsics)
            if step is not None:
                gain = SERVO_GAIN_3D * loss_backoff
                dx = clamp(gain * step[0], MAX_SERVO_STEP_MM)
                dy = clamp(gain * step[1], MAX_SERVO_STEP_MM)
        if step is None:
            if aim_robot is None:
                time.sleep(0.033)   # 3D-only run with momentary bad depth: retry
                continue
            obj_robot = pixel_to_robot(cmd_x, cmd_y, H)
            gain = SERVO_GAIN * loss_backoff
            dx = clamp(gain * (obj_robot[0] - aim_robot[0]), MAX_SERVO_STEP_MM)
            dy = clamp(gain * (obj_robot[1] - aim_robot[1]), MAX_SERVO_STEP_MM)

        # Nothing meaningful to command (dead-banded to ~zero): don't spam the
        # controller with no-op moves; just re-read.
        if abs(dx) < 0.05 and abs(dy) < 0.05:
            time.sleep(0.05)
            continue

        print(f"[Vision] err=({err_x:+d},{err_y:+d}) px → step=({dx:+.1f},{dy:+.1f}) mm")
        if not robot.move_to(pos[0] + dx, pos[1] + dy, pos[2],
                             speed=FINE_ADJUST_SPEED, wait=True):
            print("[Vision] Servo nudge failed (rejected/faulted) — aborting.")
            return False, None

        # Let the stream catch up so the next step is computed from a
        # genuinely post-move frame (MJPEG has encode/transmit latency).
        time.sleep(SERVO_SETTLE_S)

    return False, None
