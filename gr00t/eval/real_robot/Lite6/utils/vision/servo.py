"""
PBVS grasp alignment (Position-Based Visual Servoing / industrial
look-then-move for a calibrated eye-in-hand RGB-D wrist camera).

How it works — no pixel chasing:
  Measure: each fresh frame measures the target cube's position IN THE ROBOT
        BASE FRAME (per-blob depth → deproject → extrinsics). The mapping
        p_cube = p_tcp(live) + R·p_cam + t is TCP-relative, so it is valid
        from ANY arm position; every measurement is automatically anchored to
        wherever the arm currently is. All measurements happen at the
        calibration orientation (roll/pitch/yaw = −180/0/0, yaw rotates only
        after lock) because R assumes it.
  Move:  command the TCP directly at the grasp point (the cube's measured
        position + TOOL_OFFSET), damped by PBVS_DAMPING and clamped to
        PBVS_MAX_STEP_MM. The target is the cube's own in-envelope position,
        so the servo can never walk out of the workspace the way the old
        offset-aim pixel loop could. The camera↔gripper offset lives in the
        extrinsics t — no explicit −70 mm constant, and the cube may sit
        anywhere in the frame.
  Refine: once aligned at hover height, descend to PBVS_REFINE_Z_MM and
        re-verify from the shorter lever arm (sharper px/mm, smaller
        calibration-tilt error), then lock.
  Fresh: every measurement waits for a frame CAPTURED AFTER the previous move
        finished (camera timestamps) — stale mid-move frames were the source
        of the old over-stepping.

Select: multi-object aware via WORLD-FRAME TARGET IDENTITY — candidates are
        measured in the base frame and matched to the target's known position
        within TARGET_MATCH_TOL_MM; blobs near already-delivered cubes
        (avoid_xy) are never selected. Depth dropout falls back to matching
        against the target's PROJECTED pixel. A frame matching neither tier =
        target lost; sustained loss aborts for an FSM re-scan (no blind
        spiral — the world already knows where cubes are).
"""

import time
from typing import List, Optional, Tuple

import cv2

import numpy as np

from utils.constants import (
    GRASP_Z,
    FINE_ADJUST_SPEED,
    PBVS_TOL_MM,
    PBVS_CONFIRM_FRAMES,
    PBVS_DAMPING,
    PBVS_MAX_STEP_MM,
    PBVS_MAX_ITERS,
    PBVS_REFINE_Z_MM,
    FRAME_FRESH_TIMEOUT_S,
    LOST_TARGET_GRACE_S,
    TOOL_OFFSET_X,
    TOOL_OFFSET_Y,
    TARGET_MATCH_TOL_MM,
    TARGET_AVOID_RADIUS_MM,
    TARGET_EMA_ALPHA,
    DEPTH_TRUST_MIN_TCP_Z_MM,
    DEPTH_MIN_PLAUSIBLE_MM,
    TRACK_MAX_JUMP_PX,
    SERVO_MAX_BLOB_AREA_PX,
)
from utils.vision.helpers import clamp
from utils.vision.camera import read_fresh_after
from utils.vision.detection import (
    find_all_blobs,
    refine_blob,
    select_blob_near,
    color_mask_of,
    height_gate_mask,
)
from utils.vision.localize3d import (
    deproject,
    project,
    camera_to_base,
    base_to_camera,
    robust_depth_at,
)


def _read_depth_if_available(cap):
    if getattr(cap, "has_depth", False):
        ret, depth = cap.read_depth()
        if ret:
            return depth
    return None


class _TargetTracker:
    """
    World-frame identity of the cube being grasped. The estimate starts at the
    SCANNING localization and is refined by an EMA over gated metric
    measurements (the 45 mm match gate rejects outliers; the EMA averages down
    depth noise) — it lives in the BASE frame, so it survives every arm move.
    """

    def __init__(self, target_base=None, avoid_xy=None):
        if target_base is not None:
            self.est_xy = (float(target_base[0]), float(target_base[1]))
            z = target_base[2] if len(target_base) > 2 else None
            self.est_z = float(z) if z is not None else None
        else:
            self.est_xy = None
            self.est_z = None
        self.initial_est = self.est_xy    # for the calibration-health residual
        self.avoid_xy = [(float(x), float(y)) for x, y in (avoid_xy or [])]
        self.last_match = ""              # "12mm(3D)" / "34px(proj)" — for logs
        self._warned_oversize = False
        self._last_reject_log = 0.0

    def update(self, p_base):
        """EMA the estimate toward a gated measurement (static target)."""
        a = TARGET_EMA_ALPHA
        self.est_xy = ((1 - a) * self.est_xy[0] + a * float(p_base[0]),
                       (1 - a) * self.est_xy[1] + a * float(p_base[1]))
        if self.est_z is None:
            self.est_z = float(p_base[2])
        else:
            self.est_z = (1 - a) * self.est_z + a * float(p_base[2])

    def residual_mm(self) -> Optional[float]:
        """|current estimate − SCANNING's estimate| — calibration health."""
        if self.initial_est is None or self.est_xy is None:
            return None
        return ((self.est_xy[0] - self.initial_est[0]) ** 2
                + (self.est_xy[1] - self.initial_est[1]) ** 2) ** 0.5

    def _log_reject(self, msg):
        """Rate-limited (1/s) — gate rejections repeat every frame while lost."""
        now = time.time()
        if now - self._last_reject_log >= 1.0:
            self._last_reject_log = now
            print(msg)


def _match_target(candidates, tracker, depth_mm, tcp, intrinsics, extrinsics):
    """
    Select OUR cube among the candidate blobs, by robustness tier:
      1. Metric (Orbbec): measure each candidate's base-frame position from its
         own depth; nearest to the target estimate within TARGET_MATCH_TOL_MM
         wins; candidates near delivered cubes (avoid_xy) are discarded. When
         at least one candidate had usable depth, this tier's verdict is FINAL
         (a gated-out frame means the target is genuinely not visible — a
         neighbor cube must not be accepted by a looser tier).
         Trusted only at TCP z >= DEPTH_TRUST_MIN_TCP_Z_MM (wrist Orbbec is
         inside its min range near the table).
      2. Expected pixel: project the target's known 3D position into the image
         (needs no live depth); nearest candidate within TRACK_MAX_JUMP_PX.
    Returns the matched BlobCandidate or None (target lost this frame — never
    a silent adoption of a different cube). On a tier-1 match, EMA-updates the
    estimate.
    """
    if not candidates:
        return None

    if (tracker.est_xy is None or intrinsics is None or extrinsics is None
            or tcp is None):
        tracker._log_reject("[Vision] Target matching needs a target identity + "
                            "intrinsics + extrinsics — none matched this frame.")
        return None

    R, t = extrinsics
    tcp_xyz = tcp[:3]

    # --- Tier 1: per-candidate metric measurement ---
    if depth_mm is not None and tcp[2] >= DEPTH_TRUST_MIN_TCP_Z_MM:
        best, best_dist, best_p, measured = None, None, None, 0
        for c in candidates:
            d = robust_depth_at(depth_mm, c.cx, c.cy)
            if d is None or d < DEPTH_MIN_PLAUSIBLE_MM:
                continue
            measured += 1
            p = camera_to_base(deproject(c.cx, c.cy, d, intrinsics), tcp_xyz, R, t)
            if any((p[0] - ax) ** 2 + (p[1] - ay) ** 2 < TARGET_AVOID_RADIUS_MM ** 2
                   for ax, ay in tracker.avoid_xy):
                continue   # a cube we already delivered — never our target
            dist = ((p[0] - tracker.est_xy[0]) ** 2
                    + (p[1] - tracker.est_xy[1]) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best, best_dist, best_p = c, dist, p
        if measured:
            if best is not None and best_dist <= TARGET_MATCH_TOL_MM:
                tracker.update(best_p)
                tracker.last_match = f"{best_dist:.0f}mm(3D)"
                return best
            tracker._log_reject(
                f"[Vision] {len(candidates)} blob(s) visible but none within "
                f"{TARGET_MATCH_TOL_MM:.0f}mm of the target at "
                f"({tracker.est_xy[0]:.0f}, {tracker.est_xy[1]:.0f}) "
                f"(nearest {best_dist:.0f}mm)" if best is not None else
                f"[Vision] {len(candidates)} blob(s) visible but all are "
                f"delivered cubes / gated out — target not in view.")
            return None

    # --- Tier 2: expected-pixel projection (no live depth needed) ---
    if tracker.est_z is not None:
        p_base = np.array([tracker.est_xy[0], tracker.est_xy[1], tracker.est_z])
        exp = project(base_to_camera(p_base, tcp_xyz, R, t), intrinsics)
        if exp is not None:
            sel = select_blob_near(candidates, (int(exp[0]), int(exp[1])),
                                   max_dist_px=TRACK_MAX_JUMP_PX)
            if sel is not None:
                dist_px = ((sel.cx - exp[0]) ** 2 + (sel.cy - exp[1]) ** 2) ** 0.5
                tracker.last_match = f"{dist_px:.0f}px(proj)"
                return sel
            tracker._log_reject(
                f"[Vision] No blob within {TRACK_MAX_JUMP_PX}px of the target's "
                f"projected pixel ({exp[0]:.0f}, {exp[1]:.0f}).")
            return None

    tracker._log_reject("[Vision] No usable depth and no target height estimate — "
                        "cannot match this frame.")
    return None


def _detect_target(frame_rgb, color_name, depth_mm, tracker, tcp,
                   intrinsics, extrinsics):
    """
    One full perception pass: mask → all candidates → world-frame match →
    blob-scoped top-face refinement of the selection.
    Returns (cx, cy, bbox, angle) or None.
    """
    mask = color_mask_of(frame_rgb, color_name)
    if depth_mm is not None:
        mask = height_gate_mask(mask, depth_mm)
    candidates = find_all_blobs(frame_rgb, color_name, mask=mask)

    sel = _match_target(candidates, tracker, depth_mm, tcp, intrinsics, extrinsics)
    if sel is None:
        return None

    if (SERVO_MAX_BLOB_AREA_PX is not None and sel.area > SERVO_MAX_BLOB_AREA_PX
            and not tracker._warned_oversize):
        tracker._warned_oversize = True
        print(f"[Vision] WARNING: matched blob area {sel.area:.0f}px exceeds "
              f"{SERVO_MAX_BLOB_AREA_PX:.0f}px — possibly two touching cubes.")

    return refine_blob(mask, sel.contour, depth_mm)


def visual_servo_to_grasp(
    cap,
    robot,
    target_object: str,
    timeout: float = 15.0,
    should_stop_cb=None,
    intrinsics=None,
    extrinsics=None,
    target_base=None,
    avoid_xy=None,
) -> Tuple[bool, Optional[float], Optional[Tuple[float, float]], Optional[float]]:
    """
    PBVS alignment over the target cube. Returns (locked, angle_deg, grasp_xy,
    est_z):
      - locked:   True once the TCP sits over the grasp point within
                  PBVS_TOL_MM for PBVS_CONFIRM_FRAMES fresh frames, verified
                  from the PBVS_REFINE_Z_MM stage.
      - angle_deg: the cube's in-image rotation from the last matched frame
                  (for the caller's yaw-align), None when not locked.
      - grasp_xy: the base-frame (x, y) the descend should target — the
                  final measured cube position + TOOL_OFFSET. None when not
                  locked.
      - est_z:    the tracker's base-frame estimate of the cube's TOP-FACE
                  height (EMA over gated metric measurements, including the
                  refine stage) — the caller's depth-adaptive descend input.
                  May be None even when locked (pixel-tier-only tracking).

    target_base: (x, y, z|None) base-frame mm of the SCANNING-chosen cube.
    avoid_xy: [(x, y), ...] of cubes already sitting in the task's target zone.
    Requires intrinsics + extrinsics + target_base (calibrated Orbbec only).
    """
    if intrinsics is None or extrinsics is None:
        print("[Vision] Grasping requires the local Orbbec (intrinsics) + "
              "calibrated extrinsics (script/lite6_extrinsics.py) — aborting.")
        return False, None, None, None
    if target_base is None:
        print("[Vision] No target identity from SCANNING — aborting.")
        return False, None, None, None

    color_name = target_object.split()[0].lower()
    tracker = _TargetTracker(target_base, avoid_xy)

    def grasp_target():
        return (tracker.est_xy[0] + TOOL_OFFSET_X,
                tracker.est_xy[1] + TOOL_OFFSET_Y)

    print(f"[Vision] PBVS target: base ({tracker.est_xy[0]:.1f}, "
          f"{tracker.est_xy[1]:.1f}"
          + (f", z {tracker.est_z:.1f}" if tracker.est_z is not None else "")
          + f") mm, avoiding {len(tracker.avoid_xy)} delivered cube(s).")

    # Reachability precheck: the grasp TCP is the cube's own position, so this
    # only trips when the cube genuinely can't be picked — fail fast instead of
    # walking toward an impossible point.
    pos = robot.get_current_position()
    if pos is None:
        return False, None, None, None
    gx, gy = grasp_target()
    if not (robot.in_workspace(gx, gy, pos[2]) and robot.in_workspace(gx, gy, GRASP_Z)):
        print(f"[Vision] Cube unreachable — grasp TCP ({gx:.1f}, {gy:.1f}) is "
              f"outside the workspace envelope. Aborting.")
        return False, None, None, None
    if not (robot.is_pose_reachable(gx, gy, pos[2])
            and robot.is_pose_reachable(gx, gy, GRASP_Z)):
        print(f"[Vision] Cube unreachable — grasp TCP ({gx:.1f}, {gy:.1f}) is "
              f"outside the kinematic reach sphere. Aborting.")
        return False, None, None, None

    start_time = time.time()
    confirm = 0
    last_angle = None
    last_move_end = 0.0      # first frame needs no freshness wait
    last_seen = time.time()
    at_refine_stage = False
    iters = 0

    while True:
        if should_stop_cb and should_stop_cb():
            print("[Vision] Stop requested during PBVS alignment.")
            return False, None, None, None
        if time.time() - start_time > timeout:
            print("[Vision] PBVS alignment TIMEOUT.")
            return False, None, None, None
        if robot.has_error():
            print("[Vision] Arm in error state during PBVS alignment — aborting.")
            return False, None, None, None

        # Measure only on frames captured AFTER the last move finished.
        ret, frame_bgr = read_fresh_after(cap, last_move_end, FRAME_FRESH_TIMEOUT_S)
        if not ret:
            time.sleep(0.05)
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth_mm = _read_depth_if_available(cap)

        pos = robot.get_current_position()
        if pos is None:
            time.sleep(0.05)
            continue

        blob = _detect_target(frame_rgb, color_name, depth_mm, tracker,
                              pos, intrinsics, extrinsics)
        if blob is None:
            confirm = 0
            if time.time() - last_seen > LOST_TARGET_GRACE_S:
                print("[Vision] Target lost — aborting for an FSM re-scan "
                      "(the world matcher will not adopt a different cube).")
                return False, None, None, None
            time.sleep(0.05)
            continue

        _cx, _cy, _bbox, angle = blob
        last_seen = time.time()
        last_angle = angle

        gx, gy = grasp_target()
        err_x = gx - pos[0]
        err_y = gy - pos[1]
        err = (err_x ** 2 + err_y ** 2) ** 0.5

        if err <= PBVS_TOL_MM:
            confirm += 1
            if confirm < PBVS_CONFIRM_FRAMES:
                continue
            if not at_refine_stage:
                # Stage 2: verify from the shorter lever arm before locking —
                # sharper px/mm and less calibration-tilt sensitivity.
                at_refine_stage = True
                confirm = 0
                if pos[2] > PBVS_REFINE_Z_MM + 5.0:
                    print(f"[Vision] Aligned at hover — descending to refine "
                          f"height Z={PBVS_REFINE_Z_MM:.0f} for verification...")
                    if not robot.move_to(gx, gy, PBVS_REFINE_Z_MM,
                                         speed=FINE_ADJUST_SPEED, wait=True):
                        print("[Vision] Refine descend failed — aborting.")
                        return False, None, None, None
                    last_move_end = time.time()
                continue
            residual = tracker.residual_mm()
            print(f"[Vision] PBVS LOCKED at ({gx:.1f}, {gy:.1f}) "
                  f"[match {tracker.last_match}, angle {last_angle:.1f}°]. "
                  f"Calibration health: scan→final residual "
                  f"{residual:.1f} mm." if residual is not None else
                  f"[Vision] PBVS LOCKED at ({gx:.1f}, {gy:.1f}).")
            return True, last_angle, (gx, gy), tracker.est_z
        confirm = 0

        iters += 1
        if iters > PBVS_MAX_ITERS:
            print(f"[Vision] PBVS move budget ({PBVS_MAX_ITERS}) exhausted at "
                  f"err {err:.1f} mm — aborting for a re-scan.")
            return False, None, None, None

        step_x = clamp(PBVS_DAMPING * err_x, PBVS_MAX_STEP_MM)
        step_y = clamp(PBVS_DAMPING * err_y, PBVS_MAX_STEP_MM)
        tx, ty = pos[0] + step_x, pos[1] + step_y
        if not robot.in_workspace(tx, ty, pos[2]):
            # The step is toward the cube's own (pre-validated) position, so
            # this only fires if the estimate escaped the envelope mid-run.
            print(f"[Vision] PBVS step target ({tx:.1f}, {ty:.1f}) left the "
                  f"envelope — aborting.")
            return False, None, None, None

        print(f"[Vision] PBVS iter {iters}: cube at ({tracker.est_xy[0]:.1f}, "
              f"{tracker.est_xy[1]:.1f}) [match {tracker.last_match}] — "
              f"err {err:.1f} mm → move ({step_x:+.1f}, {step_y:+.1f}).")
        if not robot.move_to(tx, ty, pos[2], speed=FINE_ADJUST_SPEED, wait=True):
            print("[Vision] PBVS correction move failed (rejected/faulted) — aborting.")
            return False, None, None, None
        last_move_end = time.time()

    return False, None, None
