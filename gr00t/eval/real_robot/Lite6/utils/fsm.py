"""
Lite6 Finite State Machine Controller

States:
  PRE_CHECK      → count target objects in the SOURCE zone ROI; baseline the
                   TARGET zone (count/mass/occupancy) from the same frame
  SCANNING       → top-view pose → pick ONE source-zone blob (random among
                   candidates) → 3D depth localization (ray-plane fallback)
                   → robot XY target
  GLOBAL_APPROACH→ move to (X, Y, APPROACH_Z_MM) above the object
                   (height ladder: TOP_VIEW 450 scan → APPROACH 250 PBVS →
                   CARRY 220 transport → GRASP 105; tool-down above ~250 runs
                   out of the Lite6 reach sphere at the table corners)
  FINE_GRASP     → PBVS: measure cube in base frame → move TCP at it →
                   re-measure fresh → refine at 250 mm → lock → yaw-align →
                   descend → vacuum grasp (+ suction feedback) → lift
  TRANSPORT      → blended move to a randomized target-zone drop point
                   (yaw straightens; clearance from objects of ANY color
                   already there), drop
  RETURNING      → the target zone is FULL (no sampled drop point even clears
                   a cube footprint): carry the cube back to its recorded
                   pickup point, place it at the measured grasp height, and
                   end DONE_RETURNED (a distinct success, not a failure)
  RECOVERY       → move fault mid-task → if suction still (maybe) holds,
                   retry TRANSPORT (or RETURNING) with a fresh drop point
                   before ever releasing; only release + re-home +
                   retry-scan/fail once confirmed empty or the hold-retry
                   budget is spent
  VERIFY         → confirm the TARGET zone GREW (count-delta, not presence —
                   the zone may already hold identical cubes)
  DONE / DONE_RETURNED / FAILED

Safety / robustness:
  - Every move checks its return code; faults route to recovery/FAILED.
  - Vision is zone-restricted EVERYWHERE — including SCANNING's localization,
    so a cube already sitting in the target zone (or any other zone) is never
    picked as the task's target.
  - No post-grasp camera check: the object is not visible in the wrist camera
    once gripped (mount geometry) — but the vacuum's payload feedback
    (check_grasp) can catch an empty grasp before TRANSPORT when enabled.
  - Soft stop finishes an in-progress manipulation (SO-ARM semantics): once
    the servo has locked, descend/grasp/transport/drop run to completion and
    only then does the system exit. Pre-lock, a stop aborts immediately.

Grasp orientation:
  The servo runs at yaw=0 (the aim point was measured at yaw=0). Once the
  servo locks, the object's in-image angle from that final frame drives a
  pure yaw rotation about the TCP — which does NOT move the gripper off the
  object — then the descent/grasp/lift carry that yaw. Transport commands
  yaw=0, so the cube is straightened before the drop.
"""

import time
import random
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from utils.constants import (
    ZONES, ZONE_PIXEL_ROI, HOME_POSE, TOP_VIEW_POSE,
    APPROACH_Z_MM, TRANSPORT_Z_MM, GRASP_Z,
    DEFAULT_SPEED, FINE_ADJUST_SPEED,
    YAW_ALIGN_ENABLED, CAMERA_YAW_SIGN, CAMERA_YAW_OFFSET,
    POSE_TOL_MM, POSE_TOL_DEG, MOVE_TIMEOUT_S,
    ZONE_BLOB_MIN_AREA_PX, ZONE_BLOB_MAX_AREA_PX,
    PLACE_MARGIN_MM, PLACE_CLEARANCE_MM, PLACE_SAMPLE_ATTEMPTS,
    PLACE_MIN_FALLBACK_CLEARANCE_MM,
    FREE_SPACE_S_MAX, FREE_SPACE_V_MIN, PLACE_FREE_MIN_CLEARANCE_MM,
    TRANSPORT_BLEND_RADIUS_MM,
    TRANSPORT_IK_RETRIES, MAX_RECOVERY_RETRIES,
    VERIFY_RECHECKS, VERIFY_MASS_DELTA_MIN_PX,
    GRASP_FEEDBACK_ENABLED,
    WORKSPACE_X_RANGE, WORKSPACE_Y_RANGE, WORKSPACE_Z_RANGE,
    OBJECT_HEIGHT_MM, TABLE_Z_FALLBACK_MM, DEPTH_MIN_PLAUSIBLE_MM,
    GRASP_DEPTH_ADAPT_ENABLED, GRASP_Z_MAX_LOWER_MM, GRASP_Z_MAX_RAISE_MM,
    GRASP_TOP_SANE_BAND_MM, GRASP_PRESS_DOWN_MM,
)
from utils.vision import (
    find_all_blobs,
    count_objects_in_zone,
    count_all_colors_in_zone,
    visual_servo_to_grasp,
    read_fresh,
    load_extrinsics,
    localize_object_3d,
    camera_to_base,
    deproject,
    pixel_to_base_on_plane,
    estimate_table_z,
    robust_depth_at,
)
from utils.vision.helpers import to_bgr


class FSMState(Enum):
    PRE_CHECK        = auto()
    SCANNING         = auto()
    GLOBAL_APPROACH  = auto()
    FINE_GRASP       = auto()
    TRANSPORT        = auto()
    RETURNING        = auto()   # target zone full → carry cube back to pickup point
    RECOVERY         = auto()
    VERIFY           = auto()
    DONE             = auto()
    DONE_RETURNED    = auto()   # terminal: cube returned to origin (zone full);
                                # a distinct successful outcome, NOT a failure
    FAILED           = auto()


class Lite6FSM:
    def __init__(
        self,
        robot,
        cap,
        task_type: str,
        target_object: str,
        source_zone: str,
        target_zone: str,
        vla_timeout: float = 15.0,
        max_retries: int = 2,
        should_stop_cb=None,
    ):
        self.robot          = robot
        self.cap            = cap
        self.task_type      = task_type
        self.target_object  = target_object
        self.source_zone    = source_zone
        self.target_zone    = target_zone
        self.vla_timeout    = vla_timeout
        self.max_retries    = max_retries
        self.should_stop_cb = should_stop_cb

        self.state          = FSMState.PRE_CHECK
        self.search_retries = 0
        self._recovery_retries = 0   # hold-through-retry budget, separate from
                                    # search_retries (RECOVERY's own retry path)
        # Calibrated geometry is MANDATORY (the one-pose homography is gone):
        # color intrinsics from the camera + camera→base extrinsics carry ALL
        # pixel→robot mapping (scanning, occupancy, drop rects, servo).
        self.intrinsics     = getattr(cap, "intrinsics", None)
        self.extrinsics     = load_extrinsics()
        if self.intrinsics is None or self.extrinsics is None:
            raise RuntimeError(
                "[FSM] Grasping requires the local Orbbec (color intrinsics) and "
                "calibrated camera→base extrinsics (run script/lite6_extrinsics.py). "
                "The color-only stream camera can no longer run tasks."
            )
        self._table_z       = None   # base-frame table height, estimated from
                                     # the first top-view depth frame (PRE_CHECK)
        self._target_x      = None
        self._target_y      = None

        # Multi-object bookkeeping, captured at PRE_CHECK from the last top view
        # BEFORE the arm is mid-carry (a single wrist camera has no top view
        # while holding the object):
        self._baseline_target_count = 0    # cubes already in the target zone
        self._baseline_target_mass  = 0    # their total mask pixels (VERIFY backstop)
        self._occupied_target_xy    = []   # their robot-mm (x, y): drop clearance +
                                           # servo avoid-list (never re-pick them)
        self._chosen_blob           = None # the SCANNING-selected source blob
        self._target_base           = None # its base-frame (x, y, z|None) mm — the
                                           # servo's world-frame target identity
        # Return-to-origin bookkeeping (target zone full): the PBVS-measured
        # pickup point and the descend height actually used to grasp there.
        self._pickup_xy             = None
        self._grasp_z_used          = GRASP_Z
        self._returning             = False
        # Drop rect needs the table-z estimate, so it is computed lazily on
        # first use (TRANSPORT) rather than here.
        self._drop_rect             = None
        self._drop_rect_computed    = False
        # Free-space drop planning, computed at PRE_CHECK from the top view
        # (white table = empty; distance-transform peak = farthest from every
        # obstacle and the zone edge):
        self._planned_drop          = None   # (x, y) mm, or None
        self._free_space_full       = False  # analysis ran and found NO spot

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run(self) -> FSMState:
        handlers = {
            FSMState.PRE_CHECK:       self._handle_pre_check,
            FSMState.SCANNING:        self._handle_scanning,
            FSMState.GLOBAL_APPROACH: self._handle_global_approach,
            FSMState.FINE_GRASP:      self._handle_fine_grasp,
            FSMState.TRANSPORT:       self._handle_transport,
            FSMState.RETURNING:       self._handle_returning,
            FSMState.RECOVERY:        self._handle_recovery,
            FSMState.VERIFY:          self._handle_verify,
        }
        # Soft-stop semantics (SO-ARM style): a stop may abort the task only
        # BEFORE the arm has committed to the object. From TRANSPORT onward
        # (object in gripper) the FSM runs to completion — finish the carry,
        # drop it in the target zone (or back at its pickup point when the
        # zone is full), verify — and the caller's loop exits afterwards.
        # Commitment point = servo lock inside FINE_GRASP.
        abortable = {FSMState.PRE_CHECK, FSMState.SCANNING,
                     FSMState.GLOBAL_APPROACH, FSMState.FINE_GRASP}
        while self.state not in (FSMState.DONE, FSMState.DONE_RETURNED,
                                 FSMState.FAILED):
            if self.state in abortable and self._check_abort():
                break
            handlers[self.state]()
        return self.state

    # -------------------------------------------------------------------------
    # Guards
    # -------------------------------------------------------------------------

    def _check_abort(self) -> bool:
        if self.should_stop_cb and self.should_stop_cb():
            print("\n[FSM] Stop command detected — aborting FSM.")
            self.state = FSMState.FAILED
            return True
        return False

    def _fail_or_retry(self, reason: str) -> bool:
        """Increment the retry counter; return True if we should hard-fail."""
        self.search_retries += 1
        print(f"[FSM] {reason} (retry {self.search_retries}/{self.max_retries}).")
        return self.search_retries > self.max_retries

    # -------------------------------------------------------------------------
    # Non-blocking move with polling for completion + fault handling.
    #
    # Completion is detected by POSE CONVERGENCE (actual position within
    # POSE_TOL of the target), NOT get_is_moving(): the controller lags a few
    # tens of ms before reporting motion after a non-blocking command, which
    # once let the FSM race into FINE_GRASP while the arm was still parked at
    # TOP_VIEW_POSE. If the arm hasn't started moving yet, it simply isn't at
    # the target — no race is possible.
    #
    # Returns True on success, False on rejection/fault/abort/timeout.
    # -------------------------------------------------------------------------

    def _safe_move(self, x, y, z, roll=-180.0, pitch=0.0, yaw=0.0,
                   speed=DEFAULT_SPEED, interruptible=True) -> bool:
        """
        interruptible=False marks a COMMITTED move (post-grasp-lock: descend,
        lift, transport, recovery homing): a soft stop is ignored so the
        manipulation finishes and the object is delivered before shutdown.
        Fault handling remains active either way.
        """
        if interruptible and self._check_abort():
            return False

        if not self.robot.move_to(x, y, z, roll=roll, pitch=pitch, yaw=yaw, speed=speed, wait=False):
            return False

        deadline = time.time() + MOVE_TIMEOUT_S
        while time.time() < deadline:
            if interruptible and self.should_stop_cb and self.should_stop_cb():
                # Soft stop: do NOT pause — the in-flight trajectory finishes on
                # the controller (xArm queues commands), the FSM unwinds, and the
                # caller's cleanup path drives the arm home. Pausing here used to
                # leave the arm frozen in state 4 so the home move never ran.
                print("[FSM] Soft stop — letting current motion finish, then homing.")
                return False

            if self.robot.has_error():
                print("[FSM] Arm entered error state mid-move — clearing.")
                self.robot.clear_errors()
                return False

            pos = self.robot.get_current_position()
            if pos is not None:
                dist = ((pos[0] - x) ** 2 + (pos[1] - y) ** 2 + (pos[2] - z) ** 2) ** 0.5
                if dist <= POSE_TOL_MM and abs(pos[5] - yaw) <= POSE_TOL_DEG:
                    return True

            time.sleep(0.02)

        print(f"[FSM] Move to ({x:.1f}, {y:.1f}, {z:.1f}) did not converge within {MOVE_TIMEOUT_S:.0f}s.")
        return False

    # -------------------------------------------------------------------------
    # Blended multi-waypoint chain (COMMITTED, like interruptible=False).
    #
    # Corner blending (set_position radius=...) only works when the controller
    # already has the NEXT segment queued as the current one ends — so all
    # waypoints are issued back-to-back with wait=False and convergence is
    # polled on the FINAL target only. _safe_move's per-waypoint convergence
    # polling would force a full stop at every corner and defeat the blend.
    # -------------------------------------------------------------------------

    def _wait_converged(self, wp: dict, timeout_s: float) -> bool:
        """Poll the actual pose until it converges on wp (same criterion as _safe_move)."""
        x, y, z = wp["x"], wp["y"], wp["z"]
        yaw = wp.get("yaw", 0.0)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.robot.has_error():
                # A fault discards the controller's queued segments too.
                print("[FSM] Arm entered error state mid-chain — clearing.")
                self.robot.clear_errors()
                return False

            pos = self.robot.get_current_position()
            if pos is not None:
                dist = ((pos[0] - x) ** 2 + (pos[1] - y) ** 2 + (pos[2] - z) ** 2) ** 0.5
                if dist <= POSE_TOL_MM and abs(pos[5] - yaw) <= POSE_TOL_DEG:
                    return True

            time.sleep(0.02)

        print(f"[FSM] Blended chain to ({x:.1f}, {y:.1f}, {z:.1f}) did not converge "
              f"within {timeout_s:.0f}s.")
        return False

    def _blended_move(self, waypoints, timeout_s=None) -> tuple:
        """
        waypoints: list of dicts {x, y, z, [roll, pitch, yaw, speed, radius]}.
        radius applies to every waypoint EXCEPT the last (forced to None so the
        chain always ends with a normal decelerate-to-stop MoveLine).

        A soft stop does NOT abort the chain — this helper exists only for the
        committed TRANSPORT carry (object in the gripper; pausing mid-chain
        historically froze the arm in state 4). The caller's loop honors the
        stop after the manipulation completes.

        Returns (ok, moved):
          moved=False means NO command was ever issued (pre-flight rejection —
            envelope/IK check failed before any motion). Safe to retry with a
            different point; the arm and payload are exactly as they were.
          moved=True, ok=False means the arm actually moved and then faulted
            mid-chain — the caller must go through error-clearing recovery,
            not a bare retry.
        """
        if not waypoints:
            return True, False

        # Refuse to start a chain we cannot finish: Cartesian envelope AND the
        # controller's IK (the envelope box can't see the reach-sphere limits
        # that faulted a tool-down carry at height on hardware). Nothing has
        # moved yet, so a rejection here is always safe to retry elsewhere.
        for wp in waypoints:
            if not self.robot.in_workspace(wp["x"], wp["y"], wp["z"]):
                print(f"[FSM] Blended chain REJECTED — waypoint "
                      f"({wp['x']:.1f}, {wp['y']:.1f}, {wp['z']:.1f}) outside envelope.")
                return False, False
            if not self.robot.is_pose_reachable(wp["x"], wp["y"], wp["z"],
                                                yaw=wp.get("yaw", 0.0)):
                print(f"[FSM] Blended chain REJECTED — waypoint "
                      f"({wp['x']:.1f}, {wp['y']:.1f}, {wp['z']:.1f}) outside "
                      f"kinematic reach.")
                return False, False

        for i, wp in enumerate(waypoints):
            is_last = i == len(waypoints) - 1
            ok = self.robot.move_to(
                wp["x"], wp["y"], wp["z"],
                roll=wp.get("roll", -180.0), pitch=wp.get("pitch", 0.0),
                yaw=wp.get("yaw", 0.0),
                speed=wp.get("speed", DEFAULT_SPEED),
                wait=False,
                radius=None if is_last else wp.get("radius"),
            )
            if not ok:
                print(f"[FSM] Blended chain: waypoint {i + 1}/{len(waypoints)} rejected — "
                      f"settling at the last accepted one.")
                if i > 0:
                    self._wait_converged(waypoints[i - 1], timeout_s or MOVE_TIMEOUT_S)
                    return False, True
                return False, False

        converged = self._wait_converged(waypoints[-1],
                                         timeout_s or MOVE_TIMEOUT_S * len(waypoints))
        return converged, True

    # -------------------------------------------------------------------------
    # Whole-zone randomized placement
    # -------------------------------------------------------------------------

    def _compute_drop_rect(self):
        """
        Robot-mm rectangle (xmin, xmax, ymin, ymax) to scatter drops across the
        target zone: the 4 corners of the zone's pixel ROI ray-cast onto the
        TABLE plane from TOP_VIEW_POSE (intrinsics + extrinsics; the quad is
        near-axis-aligned from the top view, so its bounding box is a faithful
        stand-in), shrunk inward by PLACE_MARGIN_MM so a dropped cube's
        centroid stays comfortably INSIDE the ROI — which is also the
        cube-drift fix: a cube can no longer walk half-out of its zone over
        hundreds of cycles. Intersected with the workspace envelope. None
        (→ nominal-point fallback) when the table height is unknown or the
        rect degenerates.
        """
        if self._table_z is None:
            return None
        roi = ZONE_PIXEL_ROI.get(self.target_zone)
        if roi is None:
            return None
        R, t = self.extrinsics
        x, y, w, h = roi
        corners = []
        for (u, v) in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
            c = pixel_to_base_on_plane(u, v, self._table_z, TOP_VIEW_POSE[:3],
                                       R, t, self.intrinsics)
            if c is None:
                return None
            corners.append(c)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        xmin = max(min(xs) + PLACE_MARGIN_MM, WORKSPACE_X_RANGE[0])
        xmax = min(max(xs) - PLACE_MARGIN_MM, WORKSPACE_X_RANGE[1])
        ymin = max(min(ys) + PLACE_MARGIN_MM, WORKSPACE_Y_RANGE[0])
        ymax = min(max(ys) - PLACE_MARGIN_MM, WORKSPACE_Y_RANGE[1])
        if xmin >= xmax or ymin >= ymax:
            print(f"[FSM] Drop rect for {self.target_zone} degenerated — "
                  f"falling back to the nominal drop point.")
            return None
        return xmin, xmax, ymin, ymax

    def _plan_drop_point(self, frame_rgb):
        """
        Free-space drop planning from the PRE_CHECK top view. The work surface
        is WHITE, so emptiness is detected directly: white(-ish) pixels inside
        the target zone ROI are free, everything else — a cube of ANY color
        (known or not), a stray object, deep shadow — is occupied, and so is
        everything outside the ROI (the boundary repels drops the same way a
        cube does). A distance transform then finds the free pixel FARTHEST
        from all of that: the drop lands as deep in open space as the zone
        allows, instead of hoping one of the random samples happens to fall in
        the empty pocket.

        Sets self._planned_drop (x, y) mm and self._free_space_full; leaves
        both cleared when the analysis can't run (no table height / bad ROI /
        unmappable pixel) so TRANSPORT falls back to the random sampler.
        """
        self._planned_drop = None
        self._free_space_full = False
        if self._table_z is None:
            return
        roi = ZONE_PIXEL_ROI.get(self.target_zone)
        if roi is None:
            return

        hsv = cv2.cvtColor(to_bgr(frame_rgb), cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, FREE_SPACE_V_MIN),
                            (180, FREE_SPACE_S_MAX, 255))
        x, y, w, h = roi
        H_img, W_img = white.shape[:2]
        x = max(0, min(x, W_img - 1)); y = max(0, min(y, H_img - 1))
        w = max(1, min(w, W_img - x)); h = max(1, min(h, H_img - y))
        free = np.zeros_like(white)
        free[y:y + h, x:x + w] = white[y:y + h, x:x + w]
        # OPEN kills false-free specks (glare dots on cube tops); CLOSE fills
        # false-occupied specks (dust/smudges on the white table) so a clean
        # pocket isn't carved into worthless slivers.
        k = np.ones((5, 5), np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, k)
        free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, k)

        dist = cv2.distanceTransform((free > 0).astype(np.uint8), cv2.DIST_L2, 5)
        _mn, max_px, _mnloc, (u, v) = cv2.minMaxLoc(dist)
        if max_px <= 0:
            self._free_space_full = True
            print(f"[PRE_CHECK] {self.target_zone}: no white free space in the "
                  f"zone at all — FULL.")
            return

        # Pixel radius → mm via two ray-plane casts on the table plane (the
        # px/mm scale varies mildly across the view; this measures it locally).
        R, t = self.extrinsics
        tcp = TOP_VIEW_POSE[:3]
        p0 = pixel_to_base_on_plane(u, v, self._table_z, tcp, R, t, self.intrinsics)
        p1 = pixel_to_base_on_plane(u + max_px, v, self._table_z, tcp, R, t,
                                    self.intrinsics)
        if p0 is None or p1 is None:
            return
        clearance_mm = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        if clearance_mm < PLACE_FREE_MIN_CLEARANCE_MM:
            self._free_space_full = True
            print(f"[PRE_CHECK] {self.target_zone}: largest free spot clears "
                  f"only {clearance_mm:.0f} mm (< "
                  f"{PLACE_FREE_MIN_CLEARANCE_MM:.0f}) — FULL; a picked cube "
                  f"will be returned to its pickup point.")
            return
        self._planned_drop = (float(p0[0]), float(p0[1]))
        print(f"[PRE_CHECK] Planned drop (free-space peak) at "
              f"({p0[0]:.1f}, {p0[1]:.1f}) — {clearance_mm:.0f} mm clear of "
              f"every obstacle and the zone edge.")

    def _sample_drop_point(self):
        """
        Drop point in the target zone, best source first:
          1. The PRE_CHECK free-space plan (white-table distance-transform
             peak — farthest from every obstacle AND the zone edge), consumed
             one-shot so retries don't loop on an unreachable point.
          2. Random sampling over the drop rect: reachable at TRANSPORT_Z_MM
             and GRASP_Z, at least PLACE_CLEARANCE_MM from every object seen
             at PRE_CHECK (any color), best-effort when merely crowded.
        Returns (x, y), or None when the zone is FULL — the free-space
        analysis found no spot, or every sample landed within a cube footprint
        of an occupied spot (caller routes to RETURNING). The caller must
        have validated the zone name (unknown zones are its problem).
        """
        if self._free_space_full:
            print(f"[TRANSPORT] {self.target_zone} is FULL (PRE_CHECK "
                  f"free-space analysis found no spot with "
                  f"{PLACE_FREE_MIN_CLEARANCE_MM:.0f} mm clearance) — "
                  f"returning the cube to its pickup point.")
            return None

        if self._planned_drop is not None:
            px_, py_ = self._planned_drop
            self._planned_drop = None   # one-shot: retries use the sampler
            if (self.robot.in_workspace(px_, py_, TRANSPORT_Z_MM)
                    and self.robot.in_workspace(px_, py_, GRASP_Z)
                    and self.robot.is_pose_reachable(px_, py_, TRANSPORT_Z_MM)
                    and self.robot.is_pose_reachable(px_, py_, GRASP_Z)):
                print(f"[TRANSPORT] Using the free-space planned drop "
                      f"({px_:.1f}, {py_:.1f}).")
                return px_, py_
            print(f"[TRANSPORT] Planned drop ({px_:.1f}, {py_:.1f}) is outside "
                  f"the envelope/reach — falling back to random sampling.")

        zone = ZONES[self.target_zone]
        nominal = (zone["x"], zone["y"])

        if not self._drop_rect_computed:
            self._drop_rect = self._compute_drop_rect()
            self._drop_rect_computed = True
            if self._drop_rect is not None:
                xmin, xmax, ymin, ymax = self._drop_rect
                print(f"[TRANSPORT] {self.target_zone} drop rect: "
                      f"x [{xmin:.0f}, {xmax:.0f}], y [{ymin:.0f}, {ymax:.0f}] mm.")

        if self._drop_rect is None:
            print("[TRANSPORT] No drop rect (table height unknown) — using the "
                  "nominal drop point.")
            return nominal

        xmin, xmax, ymin, ymax = self._drop_rect
        best, best_clearance = None, -1.0
        for _ in range(PLACE_SAMPLE_ATTEMPTS):
            sx = random.uniform(xmin, xmax)
            sy = random.uniform(ymin, ymax)
            if not (self.robot.in_workspace(sx, sy, TRANSPORT_Z_MM)
                    and self.robot.in_workspace(sx, sy, GRASP_Z)):
                continue
            # Envelope box ≠ reach sphere: also require the controller's IK to
            # solve the carry and drop poses (tool-down corners can be out of
            # kinematic reach while inside the box).
            if not (self.robot.is_pose_reachable(sx, sy, TRANSPORT_Z_MM)
                    and self.robot.is_pose_reachable(sx, sy, GRASP_Z)):
                continue
            clearance = min(
                (((sx - ox) ** 2 + (sy - oy) ** 2) ** 0.5
                 for ox, oy in self._occupied_target_xy),
                default=float("inf"),
            )
            if clearance >= PLACE_CLEARANCE_MM:
                detail = f"clearance {clearance:.0f} mm" if self._occupied_target_xy else "zone empty"
                print(f"[TRANSPORT] Drop point ({sx:.1f}, {sy:.1f}) ({detail}).")
                return sx, sy
            if clearance > best_clearance:
                best, best_clearance = (sx, sy), clearance

        # Nothing met the clearance: take the least-crowded sample if it at
        # least clears a cube's footprint.
        if best is not None and best_clearance > PLACE_MIN_FALLBACK_CLEARANCE_MM:
            print(f"[TRANSPORT] Zone crowded — best-clearance drop "
                  f"({best[0]:.1f}, {best[1]:.1f}) ({best_clearance:.0f} mm).")
            return best
        if best is None:
            # Every sample failed the envelope/IK checks — a reachability
            # problem, not fullness; the nominal point is the safe default.
            print("[TRANSPORT] No reachable sample — using the nominal drop point.")
            return nominal
        # Genuinely saturated: even the best sample would land on a cube.
        print(f"[TRANSPORT] {self.target_zone} is FULL (best clearance "
              f"{best_clearance:.0f} mm < {PLACE_MIN_FALLBACK_CLEARANCE_MM:.0f} mm "
              f"over {PLACE_SAMPLE_ATTEMPTS} samples) — returning the cube to "
              f"its pickup point.")
        return None

    # -------------------------------------------------------------------------
    # Camera helpers (single camera; "top-down" = arm parked at TOP_VIEW_POSE)
    # -------------------------------------------------------------------------

    def _goto_top_view(self) -> bool:
        """
        Park the arm at the fixed TOP_VIEW_POSE so the wrist camera sees the whole
        workspace. The zone pixel ROIs (and the table-plane estimate's TCP) are
        defined in this pose's view, so all zone perception happens here.
        """
        return self._safe_move(
            TOP_VIEW_POSE[0], TOP_VIEW_POSE[1], TOP_VIEW_POSE[2],
            roll=TOP_VIEW_POSE[3], pitch=TOP_VIEW_POSE[4], yaw=TOP_VIEW_POSE[5],
            speed=DEFAULT_SPEED,
        )

    def _read_top_rgb(self):
        """Move to TOP_VIEW_POSE, then read the (single) camera as the top-down view."""
        if not self._goto_top_view():
            return None
        ret, frame = read_fresh(self.cap)
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_top_rgbd(self):
        """
        Top view frame + aligned depth (depth is None on stream cameras or a
        failed read). Counting with depth lets the height gate physically
        exclude table-level phantoms (reflections/stains) from zone censuses.
        """
        frame_rgb = self._read_top_rgb()
        if frame_rgb is None:
            return None, None
        depth_mm = None
        if getattr(self.cap, "has_depth", False):
            ret, depth = self.cap.read_depth()
            if ret:
                depth_mm = depth
        return frame_rgb, depth_mm

    # -------------------------------------------------------------------------
    # Base-frame geometry helpers (top-view; TCP = TOP_VIEW_POSE after
    # _goto_top_view convergence)
    # -------------------------------------------------------------------------

    def _ensure_table_z(self, depth_mm):
        """
        Estimate the table plane's base-frame height once per task from the
        first good top-view depth frame; TABLE_Z_FALLBACK_MM covers the day
        depth can't (copy the logged estimate into constants).
        """
        if self._table_z is not None:
            return
        if depth_mm is not None:
            R, t = self.extrinsics
            z = estimate_table_z(depth_mm, TOP_VIEW_POSE[:3], R, t, self.intrinsics)
            if z is not None:
                self._table_z = z
                print(f"[Vision] Table plane z ≈ {z:.1f} mm (base frame).")
                return
        if TABLE_Z_FALLBACK_MM is not None:
            self._table_z = float(TABLE_Z_FALLBACK_MM)
            print(f"[Vision] Table plane from TABLE_Z_FALLBACK_MM = {self._table_z:.1f} mm.")

    def _blobs_to_base_xy(self, blobs, depth_mm):
        """
        Top-view blob centroids → base-frame (x, y): per-blob depth
        deprojection (exact) with a ray-plane fallback at the cube-top plane.
        Blobs that can't be mapped are skipped (occupancy degrades gracefully).
        """
        R, t = self.extrinsics
        tcp = TOP_VIEW_POSE[:3]
        out = []
        for b in blobs:
            d = robust_depth_at(depth_mm, b.cx, b.cy) if depth_mm is not None else None
            if d is not None and d >= DEPTH_MIN_PLAUSIBLE_MM:
                p = camera_to_base(deproject(b.cx, b.cy, d, self.intrinsics), tcp, R, t)
                out.append((float(p[0]), float(p[1])))
                continue
            if self._table_z is not None:
                xy = pixel_to_base_on_plane(b.cx, b.cy, self._table_z + OBJECT_HEIGHT_MM,
                                            tcp, R, t, self.intrinsics)
                if xy is not None:
                    out.append(xy)
        return out

    def _compute_grasp_z(self, est_z) -> float:
        """
        Depth-adaptive descend height. GRASP_Z is the empirically correct TCP
        height for a NOMINAL cube (top at table_z + OBJECT_HEIGHT_MM), so the
        adaptive height is GRASP_Z shifted by the measured top face's DELTA
        from nominal — the working press force is inherited, short cubes get
        a deeper press, tall objects a higher stop. Clamped asymmetrically
        (−GRASP_Z_MAX_LOWER_MM / +GRASP_Z_MAX_RAISE_MM), floored at the
        workspace envelope. Falls back to the fixed GRASP_Z whenever depth or
        the table estimate is missing or the measured height is implausible.

        GRASP_PRESS_DOWN_MM is SUBTRACTED on EVERY path (adaptive or
        fallback): lower Z = closer to the table in this robot's convention,
        so "press further past the measured/nominal surface" means going to
        a SMALLER Z, same direction a short cube's delta already pushes —
        stacking them (not one cancelling the other) is what actually presses
        the cup down past the point where depth says the surface sits.
        """
        if not GRASP_DEPTH_ADAPT_ENABLED:
            return float(GRASP_Z - GRASP_PRESS_DOWN_MM)
        top_z = est_z
        if top_z is None and self._target_base is not None:
            top_z = self._target_base[2]
        if top_z is None or self._table_z is None:
            print("[FINE_GRASP] No depth-measured top height "
                  f"(top_z={top_z}, table_z={self._table_z}) — using fixed "
                  f"GRASP_Z={GRASP_Z} - {GRASP_PRESS_DOWN_MM:.1f} press.")
            return float(GRASP_Z - GRASP_PRESS_DOWN_MM)
        height = top_z - self._table_z
        lo, hi = GRASP_TOP_SANE_BAND_MM
        if not (lo <= height <= hi):
            print(f"[FINE_GRASP] Measured object height {height:.1f} mm outside "
                  f"the sane band [{lo:.0f}, {hi:.0f}] — using fixed "
                  f"GRASP_Z={GRASP_Z} - {GRASP_PRESS_DOWN_MM:.1f} press.")
            return float(GRASP_Z - GRASP_PRESS_DOWN_MM)
        delta = top_z - (self._table_z + OBJECT_HEIGHT_MM)
        delta = max(-GRASP_Z_MAX_LOWER_MM, min(GRASP_Z_MAX_RAISE_MM, delta))
        grasp_z = max(GRASP_Z + delta - GRASP_PRESS_DOWN_MM, WORKSPACE_Z_RANGE[0])
        print(f"[FINE_GRASP] Depth-adaptive grasp Z: top {top_z:.1f} "
              f"(table {self._table_z:.1f}, height {height:.1f}) → "
              f"Z={grasp_z:.1f} (nominal {GRASP_Z} - {GRASP_PRESS_DOWN_MM:.1f} press).")
        return float(grasp_z)

    # -------------------------------------------------------------------------
    # State handlers
    # -------------------------------------------------------------------------

    def _handle_pre_check(self):
        print("\n─── [FSM: PRE_CHECK] Counting objects in source zone ────────")
        time.sleep(0.3)

        frame_rgb, depth_mm = self._read_top_rgbd()
        if frame_rgb is None:
            print("[PRE_CHECK] Could not read top-down camera.")
            self.robot.wrist_wiggle()
            self.state = FSMState.FAILED
            return
        self._ensure_table_z(depth_mm)

        source_roi = ZONE_PIXEL_ROI.get(self.source_zone)
        n_src, _src_blobs, src_mass = count_objects_in_zone(
            frame_rgb, self.target_object, source_roi, depth_mm=depth_mm)

        if n_src == 0:
            print(f"[PRE_CHECK FAILED] '{self.target_object}' not in {self.source_zone} "
                  f"(0 objects, {src_mass} px).")
            self.robot.wrist_wiggle()
            self.state = FSMState.FAILED
            return

        # Baseline the TARGET zone from the same frame — the only top view we
        # get before the arm is mid-carry. VERIFY passes on count/mass GROWTH
        # over this baseline (target color only — that's our verify signal),
        # while the placement sampler must keep clear of cubes of ANY color,
        # so occupancy comes from the all-color census.
        target_roi = ZONE_PIXEL_ROI.get(self.target_zone)
        self._baseline_target_count, _tgt_blobs, self._baseline_target_mass = \
            count_objects_in_zone(frame_rgb, self.target_object, target_roi,
                                  depth_mm=depth_mm)
        n_all, all_blobs = count_all_colors_in_zone(frame_rgb, target_roi,
                                                    depth_mm=depth_mm)
        self._occupied_target_xy = self._blobs_to_base_xy(all_blobs, depth_mm)
        # Primary drop plan: white-table free-space analysis of the same frame
        # (sees obstacles of ANY color; also decides zone-full up front).
        self._plan_drop_point(frame_rgb)

        print(f"[PRE_CHECK PASSED] {n_src} object(s) in {self.source_zone}; "
              f"{self._baseline_target_count} '{self.target_object}' already in "
              f"{self.target_zone} ({n_all} object(s) of any color occupy it). "
              f"Proceeding.")
        self.state = FSMState.SCANNING

    def _choose_source_blob(self, frame_rgb, depth_mm=None):
        """
        All target-color blobs whose CENTROID lies in the SOURCE zone ROI —
        cubes in the target zone (e.g. already checked in) or anywhere else on
        the table are never candidates. With several candidates one is picked
        at RANDOM, so a multi-cube demo doesn't always grab the same cube first.
        Stashes and returns the chosen BlobCandidate, or None.
        """
        color_name = self.target_object.split()[0].lower()
        blobs = find_all_blobs(frame_rgb, color_name, depth_mm=depth_mm,
                               zone_roi=ZONE_PIXEL_ROI.get(self.source_zone),
                               min_area=ZONE_BLOB_MIN_AREA_PX,
                               max_area=ZONE_BLOB_MAX_AREA_PX)
        if not blobs:
            return None
        print(f"[SCANNING] {len(blobs)} candidate(s) in {self.source_zone}: "
              + ", ".join(f"({b.cx},{b.cy} {b.area:.0f}px)" for b in blobs))
        blob = random.choice(blobs)
        if len(blobs) > 1:
            print(f"[SCANNING] Picked ({blob.cx}, {blob.cy}) at random.")
        self._chosen_blob = blob
        return blob

    def _localize_3d(self, frame_rgb):
        """
        Depth-based localization: top-face point cloud median → camera frame →
        base frame via the calibrated extrinsic. Returns (x, y) mm or None.
        """
        ret, depth_mm = self.cap.read_depth()
        if not ret:
            return None

        blob = self._choose_source_blob(frame_rgb, depth_mm=depth_mm)
        if blob is None:
            return None
        # Localize only the CHOSEN blob: with several identical cubes in view
        # the whole-mask point-cloud median lands BETWEEN cubes.
        blob_mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
        cv2.drawContours(blob_mask, [blob.contour], -1, 255, thickness=cv2.FILLED)

        color_name = self.target_object.split()[0].lower()
        p_cam = localize_object_3d(frame_rgb, depth_mm, color_name, self.intrinsics,
                                   blob_mask=blob_mask)
        if p_cam is None:
            return None

        tcp = self.robot.get_current_position()
        if tcp is None:
            return None
        R, t = self.extrinsics
        p_base = camera_to_base(p_cam, tcp[:3], R, t)
        print(f"[SCANNING] 3D localization: camera ({p_cam[0]:.0f}, {p_cam[1]:.0f}, {p_cam[2]:.0f}) "
              f"→ base ({p_base[0]:.1f}, {p_base[1]:.1f}, {p_base[2]:.1f}) mm "
              f"[object top height {p_base[2]:.1f}]")
        # Full 3D position = the servo's world-frame target identity.
        self._target_base = (float(p_base[0]), float(p_base[1]), float(p_base[2]))
        return float(p_base[0]), float(p_base[1])

    def _localize_ray_plane(self, frame_rgb):
        """
        Depth-free localization fallback: intersect the chosen blob's pixel ray
        with the CUBE-TOP plane (table_z + OBJECT_HEIGHT_MM — the visible
        centroid is the top face). Pose-independent via the extrinsics, unlike
        the retired one-pose homography. Returns (x, y) mm or None.
        """
        if self._table_z is None:
            print("[SCANNING] Ray-plane fallback needs the table height — not "
                  "estimated yet (no depth frame so far).")
            return None
        blob = self._choose_source_blob(frame_rgb)
        if blob is None:
            return None
        R, t = self.extrinsics
        top_z = self._table_z + OBJECT_HEIGHT_MM
        xy = pixel_to_base_on_plane(blob.cx, blob.cy, top_z, TOP_VIEW_POSE[:3],
                                    R, t, self.intrinsics)
        if xy is None:
            return None
        print(f"[SCANNING] Ray-plane: pixel ({blob.cx}, {blob.cy}) → "
              f"robot ({xy[0]:.1f}, {xy[1]:.1f}) mm at top z {top_z:.1f}")
        self._target_base = (xy[0], xy[1], top_z)
        return xy

    def _handle_scanning(self):
        print("\n─── [FSM: SCANNING] Top-down detection ──────────────────────")

        frame_rgb = self._read_top_rgb()
        if frame_rgb is None:
            print("[SCANNING] Camera read failed.")
            self.state = FSMState.FAILED
            return

        # Prefer depth-based 3D localization; fall back to depth-free ray-plane.
        target = self._localize_3d(frame_rgb)
        if target is None:
            target = self._localize_ray_plane(frame_rgb)

        if target is None:
            if self._fail_or_retry("Could not locate object in top-down view"):
                self.robot.wrist_wiggle()
                self.state = FSMState.FAILED
            else:
                time.sleep(1.0)
            return

        rob_x, rob_y = target

        if not self.robot.in_workspace(rob_x, rob_y, APPROACH_Z_MM):
            if self._fail_or_retry("Mapped target outside workspace envelope"):
                self.robot.wrist_wiggle()
                self.state = FSMState.FAILED
            else:
                time.sleep(0.5)
            return

        # IK pre-flight: fail unreachable cubes at scan time with a clear log
        # instead of faulting mid-approach or mid-carry.
        if not (self.robot.is_pose_reachable(rob_x, rob_y, APPROACH_Z_MM)
                and self.robot.is_pose_reachable(rob_x, rob_y, GRASP_Z)):
            if self._fail_or_retry("Mapped target outside kinematic reach"):
                self.robot.wrist_wiggle()
                self.state = FSMState.FAILED
            else:
                time.sleep(0.5)
            return

        self._target_x, self._target_y = rob_x, rob_y
        self.state = FSMState.GLOBAL_APPROACH

    def _handle_global_approach(self):
        print("\n─── [FSM: GLOBAL_APPROACH] Moving above object ──────────────")
        if not self._safe_move(self._target_x, self._target_y, APPROACH_Z_MM, speed=DEFAULT_SPEED):
            if self._fail_or_retry("Global approach move failed"):
                self.state = FSMState.FAILED
            else:
                self.state = FSMState.SCANNING
            return
        print(f"[GLOBAL_APPROACH] Above ({self._target_x:.1f}, {self._target_y:.1f}) at Z={APPROACH_Z_MM}.")
        self.state = FSMState.FINE_GRASP

    def _handle_fine_grasp(self):
        print("\n─── [FSM: FINE_GRASP] PBVS alignment + grasp ────────────────")

        locked, obj_angle, grasp_xy, est_z = visual_servo_to_grasp(
            cap=self.cap,
            robot=self.robot,
            target_object=self.target_object,
            timeout=self.vla_timeout,
            should_stop_cb=self.should_stop_cb,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            target_base=self._target_base,
            avoid_xy=self._occupied_target_xy,
        )
        if not locked:
            if self._fail_or_retry("PBVS alignment failed to lock"):
                self.state = FSMState.FAILED
            else:
                self._safe_move(*HOME_POSE[:3])
                self.state = FSMState.SCANNING
            return

        # PBVS hands back the descend target directly: the cube's final
        # measured base position (+ tool offset). No camera-offset math —
        # that geometry lives in the extrinsics.
        grasp_x, grasp_y = grasp_xy

        # Depth-adaptive descend height, and the return-to-origin bookkeeping
        # (if the target zone turns out to be full, RETURNING puts the cube
        # back exactly here at exactly this height).
        grasp_z = self._compute_grasp_z(est_z)
        self._grasp_z_used = grasp_z
        self._pickup_xy = (grasp_x, grasp_y)
        self._returning = False   # fresh manipulation — clear any stale flag

        # Determine target yaw (measurements ran at yaw=0; rotate only now)
        grasp_yaw = 0.0
        if YAW_ALIGN_ENABLED and obj_angle is not None and abs(obj_angle) > 1.0:
            grasp_yaw = CAMERA_YAW_SIGN * obj_angle + CAMERA_YAW_OFFSET

        print(f"[FINE_GRASP] Descend to ({grasp_x:.1f}, {grasp_y:.1f}, {grasp_z:.1f}) "
              f"| Yaw: {grasp_yaw:.1f}°")

        # ONE SIMULTANEOUS MOTION: Move X, Y, Z, and Yaw together.
        if not self._safe_move(grasp_x, grasp_y, grasp_z, yaw=grasp_yaw,
                               speed=FINE_ADJUST_SPEED, interruptible=False):
            if self._fail_or_retry("Combined descent/alignment move failed"):
                self.state = FSMState.FAILED
            else:
                self._safe_move(*HOME_POSE[:3])
                self.state = FSMState.SCANNING
            return

        # 4. Perform Grasp: energize suction
        self.robot.start_vacuum()

        # Lift only to CARRY height — climbing back to scan height with the
        # payload was both wasteful and out of kinematic reach at the corners.
        print(f"[FINE_GRASP] Lifting to Z={TRANSPORT_Z_MM} (keeping yaw {grasp_yaw:.1f}° until clear)...")
        if not self._safe_move(grasp_x, grasp_y, TRANSPORT_Z_MM, yaw=grasp_yaw,
                               speed=FINE_ADJUST_SPEED, interruptible=False):
            self.state = FSMState.RECOVERY
            return

        # 5. Suction feedback: cheap "did I actually grab it" before committing
        # to the carry. Raw values are always logged (Thor calibration); a None
        # (indeterminate) reading proceeds — never stall the demo on a flaky read.
        held, _raw = self.robot.check_grasp()
        if GRASP_FEEDBACK_ENABLED and held is False:
            print("[FINE_GRASP] Suction reports NO OBJECT after lift — recovering.")
            self.state = FSMState.RECOVERY
            return

        self.state = FSMState.TRANSPORT

    def _attempt_transport_drop(self, drop_x, drop_y):
        """
        One blended carry→descend attempt at a specific point. Returns (ok, moved)
        — see _blended_move. Does NOT change self.state; the caller decides.
        """
        print(f"[TRANSPORT] Moving to {self.target_zone} ({drop_x:.1f}, {drop_y:.1f}), "
              f"blended descend to Z={GRASP_Z}...")
        return self._blended_move([
            dict(x=drop_x, y=drop_y, z=TRANSPORT_Z_MM, yaw=0.0,
                 speed=DEFAULT_SPEED, radius=TRANSPORT_BLEND_RADIUS_MM),
            dict(x=drop_x, y=drop_y, z=GRASP_Z, yaw=0.0,
                 speed=FINE_ADJUST_SPEED),
        ])

    def _handle_transport(self):
        print("\n─── [FSM: TRANSPORT] Scripted transport to target zone ──────")

        if ZONES.get(self.target_zone) is None:
            print(f"[TRANSPORT] Unknown target zone: {self.target_zone}")
            self.state = FSMState.RECOVERY
            return

        drop = self._sample_drop_point()
        if drop is None:
            # Zone is FULL (the sampler already logged it) — carry the cube
            # back to its pickup point instead of stacking it on another one.
            self.state = FSMState.RETURNING
            return

        # get_inverse_kinematics has no fixed ref_angles, so a point sampled as
        # reachable can flip by the time we're actually about to move (the arm's
        # current joint configuration seeds the solver). Re-verify right before
        # committing, and redraw a fresh point on rejection — no motion has
        # happened yet, so this is always safe to retry.
        ok = moved = False
        for attempt in range(1, TRANSPORT_IK_RETRIES + 1):
            drop_x, drop_y = drop
            if not (self.robot.is_pose_reachable(drop_x, drop_y, TRANSPORT_Z_MM)
                    and self.robot.is_pose_reachable(drop_x, drop_y, GRASP_Z)):
                print(f"[TRANSPORT] Drop point ({drop_x:.1f}, {drop_y:.1f}) is no "
                      f"longer reachable at move time (attempt {attempt}/"
                      f"{TRANSPORT_IK_RETRIES}) — drawing a fresh point.")
                drop = self._sample_drop_point()
                if drop is None:
                    self.state = FSMState.RETURNING
                    return
                continue

            ok, moved = self._attempt_transport_drop(drop_x, drop_y)
            if ok or moved:
                break   # succeeded, or faulted mid-chain (needs RECOVERY, not a resample)
            print(f"[TRANSPORT] Pre-flight rejection (attempt {attempt}/"
                  f"{TRANSPORT_IK_RETRIES}) — drawing a fresh point.")
            drop = self._sample_drop_point()
            if drop is None:
                self.state = FSMState.RETURNING
                return

        if not ok and not moved:
            # Every resampled point was pre-flight-rejected: try the nominal
            # zone point once as a last resort before giving up to RECOVERY.
            zone = ZONES.get(self.target_zone)
            if zone is not None:
                print("[TRANSPORT] Resampling exhausted — trying the nominal "
                      "zone point as a last resort.")
                ok, moved = self._attempt_transport_drop(zone["x"], zone["y"])
                drop = (zone["x"], zone["y"])

        if not ok:
            self.state = FSMState.RECOVERY
            return

        drop_x, drop_y = drop
        self.robot.stop_vacuum()   # vent, dwell, then de-energize the valve

        print(f"[TRANSPORT] Lifting back to Z={TRANSPORT_Z_MM}...")
        self._safe_move(drop_x, drop_y, TRANSPORT_Z_MM, speed=FINE_ADJUST_SPEED, interruptible=False)
        self.state = FSMState.VERIFY

    def _handle_returning(self):
        """
        Target zone full: carry the cube back to its recorded pickup point and
        place it there — a distinct, successful outcome (DONE_RETURNED), not a
        failure. The pickup spot is free by construction (we just vacated it),
        and the descend reuses the exact grasp height measured for this cube.
        Committed like TRANSPORT (cube in gripper — soft stop won't abort it).
        """
        print("\n─── [FSM: RETURNING] Target zone full — returning cube ──────")
        self._returning = True

        if self._pickup_xy is None:
            # Set at FINE_GRASP before any grasp can succeed; only a code path
            # that skipped the grasp could land here.
            print("[RETURNING] No recorded pickup point — recovering.")
            self.state = FSMState.RECOVERY
            return
        px, py = self._pickup_xy

        # Same at-move-time IK caveat as TRANSPORT: the point was reachable at
        # pick time, but the solver is seeded by the current joint configuration.
        if not (self.robot.is_pose_reachable(px, py, TRANSPORT_Z_MM)
                and self.robot.is_pose_reachable(px, py, self._grasp_z_used)):
            print(f"[RETURNING] Pickup point ({px:.1f}, {py:.1f}) not reachable "
                  f"from here — recovering.")
            self.state = FSMState.RECOVERY
            return

        print(f"[RETURNING] Carrying cube back to ({px:.1f}, {py:.1f}), blended "
              f"descend to Z={self._grasp_z_used:.1f}...")
        ok, _moved = self._blended_move([
            dict(x=px, y=py, z=TRANSPORT_Z_MM, yaw=0.0,
                 speed=DEFAULT_SPEED, radius=TRANSPORT_BLEND_RADIUS_MM),
            dict(x=px, y=py, z=self._grasp_z_used, yaw=0.0,
                 speed=FINE_ADJUST_SPEED),
        ])
        if not ok:
            self.state = FSMState.RECOVERY
            return

        self.robot.stop_vacuum()   # vent, dwell, then de-energize the valve

        print(f"[RETURNING] Lifting back to Z={TRANSPORT_Z_MM}...")
        self._safe_move(px, py, TRANSPORT_Z_MM, speed=FINE_ADJUST_SPEED,
                        interruptible=False)
        print(f"[RETURNING] Cube returned to ({px:.1f}, {py:.1f}) — "
              f"{self.target_zone} is full; task ends DONE_RETURNED.")
        self.state = FSMState.DONE_RETURNED

    def _handle_recovery(self):
        print("\n─── [FSM: RECOVERY] Move fault mid-task ─────────────────────")
        if self.robot.has_error():
            self.robot.clear_errors()

        # Hold-through-retry: a fault doesn't have to mean abandoning the
        # cube. If suction still reports (or can't rule out) a hold, retry
        # TRANSPORT with a fresh drop point instead of releasing wherever the
        # arm happened to fault — today that could be mid-air, outside any
        # tracked zone. Only release once confirmed empty or retries exhaust.
        held, _raw = self.robot.check_grasp()
        if held is not False and self._recovery_retries < MAX_RECOVERY_RETRIES:
            self._recovery_retries += 1
            retry_state = (FSMState.RETURNING if self._returning
                           else FSMState.TRANSPORT)
            print(f"[RECOVERY] Suction still {'holds' if held else 'may hold'} the "
                  f"object — retrying {retry_state.name} instead of releasing "
                  f"(attempt {self._recovery_retries}/{MAX_RECOVERY_RETRIES}).")
            pos = self.robot.get_current_position()
            if pos is not None:
                # Vertical-only recovery lift at the current XY (bypasses the
                # envelope on purpose — a fault can strand the arm slightly
                # outside it, and straight up is safe anywhere).
                self.robot.move_to(pos[0], pos[1], TRANSPORT_Z_MM,
                                   speed=FINE_ADJUST_SPEED, wait=True)
            self.state = retry_state
            return

        # Confirmed empty, or out of hold-retry budget: release, re-home, and
        # fall back to the original scan-and-retry (or fail) path.
        print("[RECOVERY] Releasing and returning to the standard retry path.")
        self.robot.stop_vacuum()    # vent, dwell, then de-energize the valve
        self._safe_move(*HOME_POSE[:3], interruptible=False)

        if self._fail_or_retry("Recovering from lost grasp / fault"):
            self.state = FSMState.FAILED
        else:
            self.state = FSMState.SCANNING

    def _handle_verify(self):
        print("\n─── [FSM: VERIFY] Checking target zone ──────────────────────")
        time.sleep(0.3)

        target_roi = ZONE_PIXEL_ROI.get(self.target_zone)

        # Count-delta, not presence: the target zone may already hold identical
        # cubes, so success = the zone GREW by our cube. The pixel-mass check
        # backstops the count when the dropped cube lands touching an existing
        # one and the two blobs merge into a single (possibly oversized) one.
        count = mass = 0
        for attempt in range(1 + VERIFY_RECHECKS):
            frame_rgb, depth_mm = self._read_top_rgbd()
            if frame_rgb is None:
                print("[VERIFY] Camera unavailable — cannot confirm; marking FAILED to be safe.")
                self.state = FSMState.FAILED
                return

            count, _blobs, mass = count_objects_in_zone(
                frame_rgb, self.target_object, target_roi, depth_mm=depth_mm)

            if count >= self._baseline_target_count + 1:
                print(f"[VERIFY] Confirmed in {self.target_zone}: {count} object(s) "
                      f"(baseline {self._baseline_target_count}).")
                self.state = FSMState.DONE
                return
            if mass >= self._baseline_target_mass + VERIFY_MASS_DELTA_MIN_PX:
                print(f"[VERIFY] Confirmed by pixel mass: {mass} px (baseline "
                      f"{self._baseline_target_mass}) — dropped cube likely "
                      f"touching an existing one.")
                self.state = FSMState.DONE
                return

            if attempt < VERIFY_RECHECKS:
                print(f"[VERIFY] Not confirmed yet ({count} object(s), {mass} px) — "
                      f"cube may still be settling; re-reading...")
                time.sleep(0.5)

        print(f"[VERIFY] {self.target_zone} did not grow ({count} object(s), {mass} px "
              f"vs baseline {self._baseline_target_count}/{self._baseline_target_mass}) "
              f"— bounced out?")
        if self._fail_or_retry("Object not in target zone"):
            self.state = FSMState.FAILED
        else:
            self.state = FSMState.SCANNING
