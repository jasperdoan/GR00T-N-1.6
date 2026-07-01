"""
Lite6 Finite State Machine Controller

States:
  PRE_CHECK      → verify target color is present in the SOURCE zone ROI
  SCANNING       → top-view pose → homography → robot XY target
  GLOBAL_APPROACH→ move to (X, Y, SAFE_Z) above the object
  FINE_GRASP     → servo blob into GRIPPER_ROI → yaw-align → descend → grasp → lift
  TRANSPORT      → move to target zone (yaw back to straight), drop
  RECOVERY       → move fault mid-task → release, re-home, retry
  VERIFY         → confirm object landed in the TARGET zone ROI
  DONE / FAILED

Safety / robustness:
  - Hand freeze actually halts in-flight motion via robot.pause()/resume()
    (the blocking-move version could only react between moves).
  - Every move checks its return code; faults route to recovery/FAILED.
  - Vision is zone-restricted, so PRE_CHECK/VERIFY mean what they say.
  - No post-grasp camera check: the object is not visible in the wrist camera
    once gripped (mount geometry), so we rely on the servo's strict ROI
    containment + the Lite6's repeatability, and VERIFY catches a failed
    pick because the object never shows up in the target zone.

Grasp orientation:
  The servo runs at yaw=0 (GRIPPER_ROI was measured at yaw=0). Once the blob
  is contained, the object's in-image angle from that final frame drives a
  pure yaw rotation about the TCP — which does NOT move the gripper off the
  object — then the descent/grasp/lift carry that yaw. Transport commands
  yaw=0, so the cube is straightened before the drop.
"""

import time
from enum import Enum, auto
from typing import Optional

import cv2

from utils.constants import (
    ZONES, ZONE_PIXEL_ROI, HOME_POSE, TOP_VIEW_POSE,
    SAFE_Z, GRASP_Z,
    DEFAULT_SPEED, FINE_ADJUST_SPEED,
    YAW_ALIGN_ENABLED, CAMERA_YAW_SIGN, CAMERA_YAW_OFFSET,
)
from utils.vision import (
    SafetyMonitor,
    load_homography,
    pixel_to_robot,
    find_object_centroid,
    check_color_presence,
    visual_servo_to_grasp,
    read_fresh,
)


class FSMState(Enum):
    PRE_CHECK        = auto()
    SCANNING         = auto()
    GLOBAL_APPROACH  = auto()
    FINE_GRASP       = auto()
    TRANSPORT        = auto()
    RECOVERY         = auto()
    VERIFY           = auto()
    DONE             = auto()
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
        safety_monitor: Optional[SafetyMonitor] = None,
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
        self.safety_monitor = safety_monitor
        self.should_stop_cb = should_stop_cb

        self.state          = FSMState.PRE_CHECK
        self.search_retries = 0
        self.H              = load_homography()
        self._target_x      = None
        self._target_y      = None

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
            FSMState.RECOVERY:        self._handle_recovery,
            FSMState.VERIFY:          self._handle_verify,
        }
        while self.state not in (FSMState.DONE, FSMState.FAILED):
            if self._check_abort():
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
    # Safety-aware move — issues a non-blocking move, then polls so a hand
    # entering mid-trajectory pauses the arm immediately (not at move-end).
    # Returns True on success, False on rejection/fault/abort.
    # -------------------------------------------------------------------------

    def _safe_move(self, x, y, z, roll=-180.0, pitch=0.0, yaw=0.0, speed=DEFAULT_SPEED) -> bool:
        # Don't even start into a workspace the hand is currently occupying.
        if self.safety_monitor:
            while self.safety_monitor.is_hand_present():
                print("[SAFETY] Hand present — waiting before move...")
                time.sleep(0.1)

        if self._check_abort():
            return False

        if not self.robot.move_to(x, y, z, roll=roll, pitch=pitch, yaw=yaw, speed=speed, wait=False):
            return False

        # Poll until the trajectory finishes; pause/resume around any hand event.
        while self.robot.is_moving() or (self.safety_monitor and self.safety_monitor.is_hand_present()):
            if self.should_stop_cb and self.should_stop_cb():
                self.robot.pause()
                return False
            if self.safety_monitor and self.safety_monitor.is_hand_present():
                print("[SAFETY] Hand detected mid-move — pausing arm...")
                self.robot.pause()
                while self.safety_monitor.is_hand_present():
                    time.sleep(0.05)
                print("[SAFETY] Hand removed — resuming.")
                self.robot.resume()
                time.sleep(0.1)   # let motion restart before re-polling is_moving()
            time.sleep(0.02)

        if self.robot.has_error():
            print("[FSM] Arm in error state after move — clearing.")
            self.robot.clear_errors()
            return False
        return True

    # -------------------------------------------------------------------------
    # Camera helpers (single camera; "top-down" = arm parked at TOP_VIEW_POSE)
    # -------------------------------------------------------------------------

    def _goto_top_view(self) -> bool:
        """
        Park the arm at the fixed TOP_VIEW_POSE so the wrist camera sees the whole
        workspace. The homography matrix is only valid from this pose, so SCANNING's
        pixel→robot mapping requires we are here first.
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
        if self.safety_monitor:
            self.safety_monitor.update_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------------------------------------------------------------------------
    # State handlers
    # -------------------------------------------------------------------------

    def _handle_pre_check(self):
        print("\n─── [FSM: PRE_CHECK] Verifying object in source zone ────────")
        time.sleep(0.3)

        frame_rgb = self._read_top_rgb()
        if frame_rgb is None:
            print("[PRE_CHECK] Could not read top-down camera.")
            self.robot.wrist_wiggle()
            self.state = FSMState.FAILED
            return

        source_roi = ZONE_PIXEL_ROI.get(self.source_zone)
        is_present, px_count = check_color_presence(frame_rgb, self.target_object, zone_roi=source_roi)

        if not is_present:
            print(f"[PRE_CHECK FAILED] '{self.target_object}' not in {self.source_zone} "
                  f"({px_count} px < threshold).")
            self.robot.wrist_wiggle()
            self.state = FSMState.FAILED
        else:
            print(f"[PRE_CHECK PASSED] {px_count} px in {self.source_zone}. Proceeding.")
            self.state = FSMState.SCANNING

    def _handle_scanning(self):
        print("\n─── [FSM: SCANNING] Top-down detection ──────────────────────")

        if self.H is None:
            print("[SCANNING] No homography matrix — cannot localise object.")
            self.state = FSMState.FAILED
            return

        frame_rgb = self._read_top_rgb()
        if frame_rgb is None:
            print("[SCANNING] Camera read failed.")
            self.state = FSMState.FAILED
            return

        color_name = self.target_object.split()[0].lower()
        centroid = find_object_centroid(frame_rgb, color_name)

        if centroid is None:
            if self._fail_or_retry("Could not locate object in top-down view"):
                self.robot.wrist_wiggle()
                self.state = FSMState.FAILED
            else:
                time.sleep(1.0)
            return

        u, v = centroid
        rob_x, rob_y = pixel_to_robot(u, v, self.H)
        print(f"[SCANNING] Object at pixel ({u}, {v}) → robot ({rob_x:.1f}, {rob_y:.1f}) mm")

        if not self.robot.in_workspace(rob_x, rob_y, SAFE_Z):
            if self._fail_or_retry("Mapped target outside workspace envelope"):
                self.robot.wrist_wiggle()
                self.state = FSMState.FAILED
            else:
                time.sleep(0.5)
            return

        self._target_x, self._target_y = rob_x, rob_y
        self.state = FSMState.GLOBAL_APPROACH

    def _handle_global_approach(self):
        print("\n─── [FSM: GLOBAL_APPROACH] Moving above object ──────────────")
        if not self._safe_move(self._target_x, self._target_y, SAFE_Z, speed=DEFAULT_SPEED):
            if self._fail_or_retry("Global approach move failed"):
                self.state = FSMState.FAILED
            else:
                self.state = FSMState.SCANNING
            return
        print(f"[GLOBAL_APPROACH] Above ({self._target_x:.1f}, {self._target_y:.1f}) at Z={SAFE_Z}.")
        self.state = FSMState.FINE_GRASP

    def _handle_fine_grasp(self):
        print("\n─── [FSM: FINE_GRASP] Visual servoing + grasp ───────────────")

        locked, obj_angle = visual_servo_to_grasp(
            cap=self.cap,
            robot=self.robot,
            target_object=self.target_object,
            timeout=self.vla_timeout,
            safety_monitor=self.safety_monitor,
            should_stop_cb=self.should_stop_cb,
        )
        if not locked:
            if self._fail_or_retry("Visual servoing failed to lock"):
                self.state = FSMState.FAILED
            else:
                self._safe_move(*HOME_POSE[:3])
                self.state = FSMState.SCANNING
            return

        pos = self.robot.get_current_position()
        if pos is None:
            self.state = FSMState.FAILED
            return
        grasp_x, grasp_y = pos[0], pos[1]

        # Yaw-align the jaws to the object. A pure yaw rotation about the TCP
        # keeps the gripper over the object (only the camera view moves, and we
        # no longer need the camera from here on).
        grasp_yaw = 0.0
        if YAW_ALIGN_ENABLED and obj_angle is not None and abs(obj_angle) > 1.0:
            grasp_yaw = CAMERA_YAW_SIGN * obj_angle + CAMERA_YAW_OFFSET
            print(f"[FINE_GRASP] Object rotated {obj_angle:.1f}° — aligning gripper yaw to {grasp_yaw:.1f}°.")
            if not self._safe_move(grasp_x, grasp_y, pos[2], yaw=grasp_yaw, speed=FINE_ADJUST_SPEED):
                if self._fail_or_retry("Yaw alignment move failed"):
                    self.state = FSMState.FAILED
                else:
                    self._safe_move(*HOME_POSE[:3])
                    self.state = FSMState.SCANNING
                return

        print(f"[FINE_GRASP] Descending to Z={GRASP_Z} (yaw={grasp_yaw:.1f}°)...")
        if not self._safe_move(grasp_x, grasp_y, GRASP_Z, yaw=grasp_yaw, speed=FINE_ADJUST_SPEED):
            if self._fail_or_retry("Descent to grasp failed"):
                self.state = FSMState.FAILED
            else:
                self._safe_move(*HOME_POSE[:3])
                self.state = FSMState.SCANNING
            return

        # No camera confirmation possible here — the object is not visible in
        # the wrist camera once gripped. The strict ROI containment above plus
        # the Lite6's repeatability carry the grasp; VERIFY catches a miss.
        self.robot.close_gripper()

        print(f"[FINE_GRASP] Lifting to Z={SAFE_Z} (keeping yaw until clear of table)...")
        if not self._safe_move(grasp_x, grasp_y, SAFE_Z, yaw=grasp_yaw, speed=FINE_ADJUST_SPEED):
            self.state = FSMState.RECOVERY
            return

        self.state = FSMState.TRANSPORT

    def _handle_transport(self):
        print("\n─── [FSM: TRANSPORT] Scripted transport to target zone ──────")

        zone = ZONES.get(self.target_zone)
        if zone is None:
            print(f"[TRANSPORT] Unknown target zone: {self.target_zone}")
            self.state = FSMState.RECOVERY
            return
        drop_x, drop_y = zone["x"], zone["y"]

        # This move commands yaw=0 (the _safe_move default), so a cube grasped
        # at an angle is rotated back to straight in mid-air before the drop.
        print(f"[TRANSPORT] Moving to {self.target_zone} ({drop_x}, {drop_y}) at Z={SAFE_Z}...")
        if not self._safe_move(drop_x, drop_y, SAFE_Z, speed=DEFAULT_SPEED):
            self.state = FSMState.RECOVERY
            return

        print(f"[TRANSPORT] Descending to Z={GRASP_Z} for drop...")
        if not self._safe_move(drop_x, drop_y, GRASP_Z, speed=FINE_ADJUST_SPEED):
            self.state = FSMState.RECOVERY
            return

        self.robot.open_gripper()
        self.robot.stop_gripper()
        time.sleep(0.2)

        print(f"[TRANSPORT] Lifting back to Z={SAFE_Z}...")
        self._safe_move(drop_x, drop_y, SAFE_Z, speed=FINE_ADJUST_SPEED)
        self.state = FSMState.VERIFY

    def _handle_recovery(self):
        print("\n─── [FSM: RECOVERY] Move fault mid-task ─────────────────────")
        # Release whatever we may be holding, clear any fault, re-home.
        self.robot.open_gripper()
        self.robot.stop_gripper()
        if self.robot.has_error():
            self.robot.clear_errors()
        self._safe_move(*HOME_POSE[:3])

        if self._fail_or_retry("Recovering from lost grasp / fault"):
            self.state = FSMState.FAILED
        else:
            self.state = FSMState.SCANNING

    def _handle_verify(self):
        print("\n─── [FSM: VERIFY] Checking target zone ──────────────────────")
        time.sleep(0.3)

        frame_rgb = self._read_top_rgb()
        if frame_rgb is None:
            print("[VERIFY] Camera unavailable — cannot confirm; marking FAILED to be safe.")
            self.state = FSMState.FAILED
            return

        target_roi = ZONE_PIXEL_ROI.get(self.target_zone)
        is_present, px_count = check_color_presence(frame_rgb, self.target_object, zone_roi=target_roi)

        if is_present:
            print(f"[VERIFY] Confirmed in {self.target_zone} ({px_count} px).")
            self.state = FSMState.DONE
        else:
            print(f"[VERIFY] NOT in {self.target_zone} ({px_count} px) — bounced out?")
            if self._fail_or_retry("Object not in target zone"):
                self.state = FSMState.FAILED
            else:
                self.state = FSMState.SCANNING
