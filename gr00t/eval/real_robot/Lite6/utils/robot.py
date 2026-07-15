"""
Lite6 Robot Controller — thin, safety-aware wrapper over xArmAPI.

Responsibilities:
- Connect / enable / disconnect (connection failure is FATAL, not silent).
- Cartesian moves that REPORT success (callers must react to failures).
- xArm error/warn state detection + clearing (the arm latches on faults and
  silently ignores commands until cleared).
- pause()/resume() that actually halt an in-flight trajectory (set_state 4/0).
- Vacuum head control (start/stop suction + payload feedback) over the Lite6
  tool GPIO.
- Reachability guard against a configured workspace envelope.
"""

import time

from xarm.wrapper import XArmAPI
from utils.constants import (
    DEFAULT_SPEED, DEFAULT_ACCEL, FINE_ADJUST_SPEED,
    VACUUM_GRIP_DWELL_S, VACUUM_RELEASE_DWELL_S,
    WORKSPACE_X_RANGE, WORKSPACE_Y_RANGE, WORKSPACE_Z_RANGE,
)


class Lite6Controller:
    def __init__(self, ip_address):
        self.ip = ip_address
        self.arm = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def connect(self):
        """
        Connect, enable motion, set position-control mode.
        Raises RuntimeError on any failure — a half-connected arm must NOT run the FSM.
        """
        print(f"[Robot] Connecting to Lite 6 at {self.ip}...")
        try:
            self.arm = XArmAPI(self.ip)
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)          # standard position control
            self.arm.set_state(state=0)   # ready
        except Exception as e:
            self.arm = None
            raise RuntimeError(f"[Robot] Connection failed: {e}") from e

        # A latched fault at connect time is common after a crash/collision.
        # Retry clearing it instead of raising immediately — under a supervisor,
        # raising here becomes a crash-restart loop that can never recover.
        for attempt in range(1, 4):
            if not self.has_error():
                break
            code = self.arm.get_err_warn_code()
            print(f"[Robot] Latched fault at connect {code} — clearing (attempt {attempt}/3)...")
            self.clear_errors()
            time.sleep(attempt)
        if self.has_error():
            code = self.arm.get_err_warn_code()
            self.arm = None
            raise RuntimeError(f"[Robot] Arm reports error/warn after connect (unclearable): {code}")

        print("[Robot] Connected, enabled, vacuum ready.")

    def disconnect(self):
        if self.arm:
            try:
                self.vacuum_neutral()
            except Exception:
                pass
            self.arm.disconnect()
            print("[Robot] Disconnected.")

    # -------------------------------------------------------------------------
    # Error / warn state
    # -------------------------------------------------------------------------

    def has_error(self) -> bool:
        """True if the arm is latched in an error or warning state."""
        if not self.arm:
            return False
        code, (err, warn) = self.arm.get_err_warn_code()
        # get_err_warn_code returns (code, [err, warn]); err!=0 means latched fault.
        return bool(err) or bool(warn)

    def clear_errors(self) -> bool:
        """Clear latched faults and re-enable motion. Returns True if arm is healthy after."""
        if not self.arm:
            return False
        print("[Robot] Clearing error/warn state...")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        return not self.has_error()

    # -------------------------------------------------------------------------
    # Motion
    # -------------------------------------------------------------------------

    def get_current_position(self):
        """Returns [x, y, z, roll, pitch, yaw] or None if failed."""
        if not self.arm:
            return None
        code, pos = self.arm.get_position()
        if code == 0:
            return pos
        return None

    def is_moving(self) -> bool:
        """True while a trajectory is executing (used by the non-blocking safe-move loop)."""
        if not self.arm:
            return False
        try:
            return bool(self.arm.get_is_moving())
        except Exception:
            return False

    def in_workspace(self, x, y, z) -> bool:
        """Reject targets outside the configured Cartesian envelope before issuing a move."""
        return (
            WORKSPACE_X_RANGE[0] <= x <= WORKSPACE_X_RANGE[1] and
            WORKSPACE_Y_RANGE[0] <= y <= WORKSPACE_Y_RANGE[1] and
            WORKSPACE_Z_RANGE[0] <= z <= WORKSPACE_Z_RANGE[1]
        )

    def is_pose_reachable(self, x, y, z, roll=-180.0, pitch=0.0, yaw=0.0) -> bool:
        """
        Ask the controller's IK whether a pose is SOLVABLE before committing a
        move — kinematic limits then reject cleanly at planning time instead
        of faulting mid-carry with a cube on the vacuum. (Observed on
        hardware: tool-down at z=450 near the table corners lies outside the
        Lite6's ~440mm reach sphere; the Cartesian envelope check cannot see
        that.) Degrades to True (logged once) if the IK query itself fails,
        so a flaky query can never block valid motion.
        """
        if not self.arm:
            return False
        try:
            code, _angles = self.arm.get_inverse_kinematics(
                [x, y, z, roll, pitch, yaw],
                input_is_radian=False, return_is_radian=False,
            )
        except Exception as e:
            if not getattr(self, "_warned_ik_unavailable", False):
                self._warned_ik_unavailable = True
                print(f"[Robot] IK reachability query unavailable ({e}) — "
                      f"skipping pre-flight reach checks.")
            return True
        if code != 0:
            print(f"[Robot] Pose ({x:.1f}, {y:.1f}, {z:.1f}) is NOT reachable "
                  f"(IK code {code}).")
            return False
        return True

    def move_to(self, x, y, z, roll=-180.0, pitch=0.0, yaw=0.0,
                speed=DEFAULT_SPEED, wait=True, radius=None) -> bool:
        """
        Move to a Cartesian pose. Returns True on success, False on failure
        (not connected, out of envelope, or nonzero SDK return code).
        Callers MUST check the return — a silent failure used to sail through the FSM.

        radius=None → MoveLine (decelerate to a stop at the point, today's
        default). radius>0 → MoveArcLine: the corner into the NEXT queued
        command is blended with a fillet arc of that radius — this only takes
        effect when this and the next command are issued back-to-back with
        wait=False so the controller has both segments queued.
        """
        if not self.arm:
            print("[Robot] Cannot move — not connected.")
            return False

        if not self.in_workspace(x, y, z):
            print(f"[Robot] REJECTED move to ({x:.1f}, {y:.1f}, {z:.1f}) — outside workspace envelope.")
            return False

        print(f"[Robot] Moving to X:{x:.1f} Y:{y:.1f} Z:{z:.1f} (speed={speed}, wait={wait})...")
        code = self.arm.set_position(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=DEFAULT_ACCEL, wait=wait, radius=radius,
        )
        if code != 0:
            print(f"[Robot] Move FAILED — set_position returned code {code}.")
            return False
        if self.has_error():
            print("[Robot] Move left arm in error state.")
            return False
        return True

    def hold_here(self):
        """Re-command the current pose (used while paused)."""
        pos = self.get_current_position()
        if pos:
            self.arm.set_position(*pos, speed=FINE_ADJUST_SPEED, mvacc=DEFAULT_ACCEL, wait=False)

    # -------------------------------------------------------------------------
    # Pause/resume primitives — these actually halt an in-flight move.
    # -------------------------------------------------------------------------

    def pause(self):
        """Halt the current trajectory immediately (xArm state 4 = paused)."""
        if self.arm:
            self.arm.set_state(4)

    def resume(self):
        """Resume from a pause (state 0 = ready/continue)."""
        if self.arm:
            self.arm.set_state(0)

    # -------------------------------------------------------------------------
    # Vacuum head — suction is driven over the Lite6 tool GPIO. Wire mapping
    # (verified against xarm-python-sdk 1.18.4 source):
    #   open_lite6_gripper()  → out0=1, out1=0 → suction ON
    #                           (wire-identical to set_vacuum_gripper(True))
    #   close_lite6_gripper() → out0=0, out1=1 → suction OFF / vent
    #   stop_lite6_gripper()  → out0=0, out1=0 → neutral (lines de-energized)
    # -------------------------------------------------------------------------

    def start_vacuum(self, wait=True):
        """
        Energize suction to grasp. The line stays DRIVEN after this call so the
        vacuum keeps holding through transport — it is only released by
        stop_vacuum(). The dwell gives the suction time to seat on the object
        before the lift. (Driving the line is also what makes check_grasp()
        meaningful: the payload input only reads while suction is energized.)
        """
        if not self.arm:
            return
        print(f"[Robot] Starting vacuum (dwell {VACUUM_GRIP_DWELL_S:.1f}s to seat the grip)...")
        self.arm.open_lite6_gripper()   # reversed
        if wait:
            time.sleep(VACUUM_GRIP_DWELL_S)

    def stop_vacuum(self, wait=True):
        """
        Release: vent the vacuum, dwell so the object drops free, then
        de-energize both valve lines (neutral) to keep the valve cool.
        """
        if not self.arm:
            return
        print(f"[Robot] Stopping vacuum (dwell {VACUUM_RELEASE_DWELL_S:.1f}s to release)...")
        self.arm.close_lite6_gripper()  # reversed
        if wait:
            time.sleep(VACUUM_RELEASE_DWELL_S)

    def vacuum_neutral(self):
        """De-energize both vacuum control lines (safe idle state)."""
        if self.arm:
            self.arm.stop_lite6_gripper()

    def check_grasp(self):
        """
        Read the vacuum payload feedback (arm.get_suction_cup, an SDK alias of
        get_vacuum_gripper): state 1 = object held, 0 = suction on but nothing
        held, -1 = suction not energized. Only meaningful between start_vacuum()
        and stop_vacuum() (the feedback gate needs the suction line driven).

        Returns (held, raw):
          held — True / False / None; None means indeterminate (not connected,
                 nonzero code, or state -1) and callers must NOT treat it as a
                 failed grasp.
          raw  — the raw (code, state) tuple, always logged so Thor runs can
                 calibrate trust in the sensor before enabling enforcement.
        Never raises.
        """
        if not self.arm:
            return None, None
        try:
            code, state = self.arm.get_suction_cup()
        except Exception as e:
            print(f"[Robot] Suction feedback read failed: {e}")
            return None, None
        print(f"[Robot] Suction feedback raw: code={code} state={state}")
        if code != 0 or state == -1:
            return None, (code, state)
        return state == 1, (code, state)

    # -------------------------------------------------------------------------
    # Failure signalling
    # -------------------------------------------------------------------------

    def wrist_wiggle(self, delta_deg=12.0, cycles=3):
        """
        Small, joint-limit-safe 'no-no' wrist shake to visually signal failure.
        Uses a conservative yaw delta (the old ±20° could hit wrist limits).
        """
        pos = self.get_current_position()
        if pos is None:
            return
        x, y, z, roll, pitch, yaw = pos
        for _ in range(cycles):
            self.arm.set_position(x, y, z, roll, pitch, yaw + delta_deg,
                                  speed=300, mvacc=800, wait=True)
            self.arm.set_position(x, y, z, roll, pitch, yaw - delta_deg,
                                  speed=300, mvacc=800, wait=True)
        self.arm.set_position(x, y, z, roll, pitch, yaw,
                              speed=200, mvacc=DEFAULT_ACCEL, wait=True)
