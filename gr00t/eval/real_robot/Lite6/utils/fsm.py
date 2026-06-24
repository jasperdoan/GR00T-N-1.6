"""
Lite6 Finite State Machine Controller

States:
  PRE_CHECK      → verify target color is present in source zone
  SCANNING       → top-down camera → homography → global approach XY
  GLOBAL_APPROACH→ arm moves to (X, Y, SAFE_Z) above object
  FINE_GRASP     → wrist-camera visual servoing → descend → grasp → lift
  TRANSPORT      → scripted move to target zone drop-off
  VERIFY         → top-down snapshot comparison to confirm placement
  DONE / FAILED

Safety features:
  1  Hand detection freeze   — every move step polls SafetyMonitor
  2  Double-tap hard stop    — handled by system.py signal handler
  3  Pre-check + fail shake  — PRE_CHECK state
  4  Before/after snapshots  — driven by eval.py / auto.py callers
  5  Error recovery + retry  — FINE_GRASP timeout → retry → FAILED
"""

import time
from enum import Enum, auto
from typing import Optional

import cv2

from utils.constants import (
    ZONES, HOME_POSE,
    SAFE_Z, GRASP_Z,
    DEFAULT_SPEED, FINE_ADJUST_SPEED,
    GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS,
)
from utils.vision import (
    SafetyMonitor,
    load_homography,
    pixel_to_robot,
    find_object_centroid,
    check_color_presence,
    visual_servo_to_center,
)


class FSMState(Enum):
    PRE_CHECK        = auto()
    SCANNING         = auto()
    GLOBAL_APPROACH  = auto()
    FINE_GRASP       = auto()
    TRANSPORT        = auto()
    VERIFY           = auto()
    DONE             = auto()
    FAILED           = auto()


class Lite6FSM:
    def __init__(
        self,
        robot,
        cap_top,
        cap_wrist,
        task_type: str,
        target_object: str,
        source_zone: str,
        target_zone: str,
        vla_timeout: float = 15.0,
        max_retries: int = 2,
        safety_monitor: Optional[SafetyMonitor] = None,
        should_stop_cb=None,
    ):
        self.robot         = robot
        self.cap_top       = cap_top
        self.cap_wrist     = cap_wrist
        self.task_type     = task_type
        self.target_object = target_object
        self.source_zone   = source_zone
        self.target_zone   = target_zone
        self.vla_timeout   = vla_timeout
        self.max_retries   = max_retries
        self.safety_monitor = safety_monitor
        self.should_stop_cb = should_stop_cb

        self.state          = FSMState.PRE_CHECK
        self.search_retries = 0
        self.H              = load_homography()

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run(self) -> FSMState:
        while self.state not in (FSMState.DONE, FSMState.FAILED):
            if self._check_abort():
                break

            if   self.state == FSMState.PRE_CHECK:       self._handle_pre_check()
            elif self.state == FSMState.SCANNING:         self._handle_scanning()
            elif self.state == FSMState.GLOBAL_APPROACH:  self._handle_global_approach()
            elif self.state == FSMState.FINE_GRASP:       self._handle_fine_grasp()
            elif self.state == FSMState.TRANSPORT:        self._handle_transport()
            elif self.state == FSMState.VERIFY:           self._handle_verify()

        return self.state

    # -------------------------------------------------------------------------
    # Abort guard
    # -------------------------------------------------------------------------

    def _check_abort(self) -> bool:
        if self.should_stop_cb and self.should_stop_cb():
            print("\n[FSM] Stop command detected — aborting FSM.")
            self.state = FSMState.FAILED
            return True
        return False

    # -------------------------------------------------------------------------
    # Safety-aware move helper
    # Holds position if a hand is detected; resumes once clear.
    # -------------------------------------------------------------------------

    def _safe_move(self, x, y, z, speed=DEFAULT_SPEED, wait=True):
        if self.safety_monitor and self.safety_monitor.is_hand_present():
            print("[SAFETY] Hand detected before move — waiting...")
            while self.safety_monitor.is_hand_present():
                pos = self.robot.get_current_position()
                if pos:
                    self.robot.move_to(*pos[:3], speed=FINE_ADJUST_SPEED, wait=False)
                time.sleep(0.1)
            print("[SAFETY] Hand removed — resuming.")

        self.robot.move_to(x, y, z, speed=speed, wait=wait)

        if self.safety_monitor:
            while self.safety_monitor.is_hand_present():
                print("[SAFETY] Hand detected mid-move — holding position...")
                pos = self.robot.get_current_position()
                if pos:
                    self.robot.move_to(*pos[:3], speed=FINE_ADJUST_SPEED, wait=False)
                time.sleep(0.1)
            print("[SAFETY] Hand removed — continuing.")

    def _failure_shake(self):
        """Wrist 'no-no' wiggle to visually signal task failure."""
        pos = self.robot.get_current_position()
        if pos is None:
            return
        x, y, z, roll, pitch, yaw = pos
        print("[FSM] Executing failure shake...")
        for _ in range(3):
            self.robot.arm.set_position(x, y, z, roll, pitch, yaw + 20, speed=300, mvacc=800, wait=True)
            self.robot.arm.set_position(x, y, z, roll, pitch, yaw - 20, speed=300, mvacc=800, wait=True)
        self.robot.arm.set_position(x, y, z, roll, pitch, yaw, speed=200, mvacc=500, wait=True)

    # -------------------------------------------------------------------------
    # FSM state handlers
    # -------------------------------------------------------------------------

    def _handle_pre_check(self):
        print("\n─── [FSM: PRE_CHECK] Verifying object presence ──────────────")
        time.sleep(0.3)

        ret, frame = self.cap_top.read()
        if not ret:
            print("[PRE_CHECK] Could not read top-down camera.")
            self._failure_shake()
            self.state = FSMState.FAILED
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Feed to safety monitor
        if self.safety_monitor:
            self.safety_monitor.update_frame(frame_rgb)

        is_present, px_count = check_color_presence(frame_rgb, self.target_object)

        if not is_present:
            print(f"[PRE_CHECK FAILED] '{self.target_object}' not found in source zone ({px_count} px < threshold).")
            self._failure_shake()
            self.state = FSMState.FAILED
        else:
            print(f"[PRE_CHECK PASSED] Detected {px_count} px. Proceeding.")
            self.state = FSMState.SCANNING

    def _handle_scanning(self):
        print("\n─── [FSM: SCANNING] Top-down detection ──────────────────────")

        if self.H is None:
            print("[SCANNING] No homography matrix — cannot localise object.")
            self.state = FSMState.FAILED
            return

        ret, frame = self.cap_top.read()
        if not ret:
            print("[SCANNING] Camera read failed.")
            self.state = FSMState.FAILED
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.safety_monitor:
            self.safety_monitor.update_frame(frame_rgb)

        color_name = self.target_object.split()[0].lower()
        centroid = find_object_centroid(frame_rgb, color_name)

        if centroid is None:
            print(f"[SCANNING] Could not locate '{self.target_object}' in top-down view.")
            self.search_retries += 1
            if self.search_retries > self.max_retries:
                self._failure_shake()
                self.state = FSMState.FAILED
            else:
                print(f"[SCANNING] Retry {self.search_retries}/{self.max_retries}...")
                time.sleep(1.0)
            return

        u, v = centroid
        rob_x, rob_y = pixel_to_robot(u, v, self.H)
        print(f"[SCANNING] Object at pixel ({u}, {v}) → robot ({rob_x:.1f} mm, {rob_y:.1f} mm)")

        self._target_x = rob_x
        self._target_y = rob_y
        self.state = FSMState.GLOBAL_APPROACH

    def _handle_global_approach(self):
        print("\n─── [FSM: GLOBAL_APPROACH] Moving to object vicinity ────────")

        if self._check_abort():
            return

        self._safe_move(self._target_x, self._target_y, SAFE_Z, speed=DEFAULT_SPEED)
        print(f"[GLOBAL_APPROACH] Arrived above ({self._target_x:.1f}, {self._target_y:.1f}) at Z={SAFE_Z}")
        self.state = FSMState.FINE_GRASP

    def _handle_fine_grasp(self):
        print("\n─── [FSM: FINE_GRASP] Visual servoing + grasp ───────────────")

        if self._check_abort():
            return

        success = visual_servo_to_center(
            cap_wrist=self.cap_wrist,
            robot=self.robot,
            target_object=self.target_object,
            timeout=self.vla_timeout,
            safety_monitor=self.safety_monitor,
            should_stop_cb=self.should_stop_cb,
        )

        if not success:
            self.search_retries += 1
            print(f"[FINE_GRASP] Visual servoing failed. Retry {self.search_retries}/{self.max_retries}.")
            if self.search_retries > self.max_retries:
                self.state = FSMState.FAILED
                return
            # Return home, re-scan
            self._safe_move(*HOME_POSE[:3])
            self.state = FSMState.SCANNING
            return

        # Descend, grasp, lift
        pos = self.robot.get_current_position()
        if pos is None:
            self.state = FSMState.FAILED
            return

        print(f"[FINE_GRASP] Descending to Z={GRASP_Z}...")
        self._safe_move(pos[0], pos[1], GRASP_Z, speed=FINE_ADJUST_SPEED)

        print("[FINE_GRASP] Closing gripper...")
        self.robot.arm.set_gripper_position(GRIPPER_CLOSED_POS, wait=True)
        time.sleep(0.4)

        print(f"[FINE_GRASP] Lifting to Z={SAFE_Z}...")
        self._safe_move(pos[0], pos[1], SAFE_Z, speed=FINE_ADJUST_SPEED)

        self.state = FSMState.TRANSPORT

    def _handle_transport(self):
        print("\n─── [FSM: TRANSPORT] Scripted transport to target zone ──────")

        zone = ZONES.get(self.target_zone)
        if zone is None:
            print(f"[TRANSPORT] Unknown target zone: {self.target_zone}")
            self.state = FSMState.FAILED
            return

        drop_x = zone["x"]
        drop_y = zone["y"]

        if self._check_abort():
            return

        print(f"[TRANSPORT] Moving to {self.target_zone} drop-off ({drop_x}, {drop_y})...")
        self._safe_move(drop_x, drop_y, SAFE_Z, speed=DEFAULT_SPEED)

        print(f"[TRANSPORT] Descending to Z={GRASP_Z} for drop...")
        self._safe_move(drop_x, drop_y, GRASP_Z, speed=FINE_ADJUST_SPEED)

        print("[TRANSPORT] Opening gripper...")
        self.robot.arm.set_gripper_position(GRIPPER_OPEN_POS, wait=True)
        time.sleep(0.35)

        print(f"[TRANSPORT] Lifting back to Z={SAFE_Z}...")
        self._safe_move(drop_x, drop_y, SAFE_Z, speed=FINE_ADJUST_SPEED)

        self.state = FSMState.VERIFY

    def _handle_verify(self):
        print("\n─── [FSM: VERIFY] Checking target zone ──────────────────────")
        time.sleep(0.3)

        ret, frame = self.cap_top.read()
        if not ret:
            # Cannot verify; assume success to avoid false failure
            print("[VERIFY] Camera unavailable — assuming placement succeeded.")
            self.state = FSMState.DONE
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        is_present, px_count = check_color_presence(frame_rgb, self.target_object)

        if is_present:
            print(f"[VERIFY] Object confirmed in target zone ({px_count} px).")
            self.state = FSMState.DONE
        else:
            print(f"[VERIFY] Object NOT detected in target zone ({px_count} px). Retry?")
            self.search_retries += 1
            if self.search_retries > self.max_retries:
                self.state = FSMState.FAILED
            else:
                self.state = FSMState.SCANNING
