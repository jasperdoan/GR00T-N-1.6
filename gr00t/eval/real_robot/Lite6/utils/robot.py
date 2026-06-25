"""
Lite6 Robot Controller — thin, safety-aware wrapper over xArmAPI.

Responsibilities:
- Connect / enable / disconnect (connection failure is FATAL, not silent).
- Cartesian moves that REPORT success (callers must react to failures).
- xArm error/warn state detection + clearing (the arm latches on faults and
  silently ignores commands until cleared).
- pause()/resume() that actually halt an in-flight trajectory (set_state 4/0),
  used by the hand-safety freeze.
- Lite6 binary gripper (open/close/stop — NOT position control).
- Reachability guard against a configured workspace envelope.
"""

import time

from xarm.wrapper import XArmAPI
from utils.constants import (
    DEFAULT_SPEED, DEFAULT_ACCEL, FINE_ADJUST_SPEED,
    GRIPPER_SETTLE_S,
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
        Connect, enable motion, set position-control mode, enable the Lite6 gripper.
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

        if self.has_error():
            code = self.arm.get_err_warn_code()
            self.arm = None
            raise RuntimeError(f"[Robot] Arm reports error/warn after connect: {code}")

        print("[Robot] Connected, enabled, gripper ready.")

    def disconnect(self):
        if self.arm:
            try:
                self.stop_gripper()
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

    def move_to(self, x, y, z, roll=-180.0, pitch=0.0, yaw=0.0,
                speed=DEFAULT_SPEED, wait=True) -> bool:
        """
        Move to a Cartesian pose. Returns True on success, False on failure
        (not connected, out of envelope, or nonzero SDK return code).
        Callers MUST check the return — a silent failure used to sail through the FSM.
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
            speed=speed, mvacc=DEFAULT_ACCEL, wait=wait,
        )
        if code != 0:
            print(f"[Robot] Move FAILED — set_position returned code {code}.")
            return False
        if self.has_error():
            print("[Robot] Move left arm in error state.")
            return False
        return True

    def hold_here(self):
        """Re-command the current pose (used while paused for safety)."""
        pos = self.get_current_position()
        if pos:
            self.arm.set_position(*pos, speed=FINE_ADJUST_SPEED, mvacc=DEFAULT_ACCEL, wait=False)

    # -------------------------------------------------------------------------
    # Hand-safety freeze primitives — these actually halt an in-flight move.
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
    # Lite6 gripper — BINARY open/close (no position control on this gripper).
    # -------------------------------------------------------------------------

    def open_gripper(self, wait=True):
        if not self.arm:
            return
        print("[Robot] Opening gripper...")
        self.arm.open_lite6_gripper()
        if wait:
            time.sleep(GRIPPER_SETTLE_S)

    def close_gripper(self, wait=True):
        if not self.arm:
            return
        print("[Robot] Closing gripper...")
        self.arm.close_lite6_gripper()
        if wait:
            time.sleep(GRIPPER_SETTLE_S)

    def stop_gripper(self):
        """Stop driving the gripper motor (call after a settle to avoid overheating)."""
        if self.arm:
            self.arm.stop_lite6_gripper()

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
