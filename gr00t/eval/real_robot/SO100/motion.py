"""
SO100 Motion Primitives

Smooth trajectory execution between joint-space waypoints.

Public API:
  lerp_to_waypoint()       — smoothly move from current pose to a target dict.
  move_to_home()           — convenience wrapper back to HOME_ACTION.
  scripted_transport()     — full pick-to-place scripted sequence after grasp.
"""

import time
from typing import Dict, Optional

import numpy as np

from constants import (
    CHECKOUT_PLACE,
    GRIPPER_OPEN_POS,
    HOME_ACTION,
    JOINT_NAMES,
    LERP_DURATION_DROP,
    LERP_DURATION_HOME,
    LERP_DURATION_LIFT,
    LERP_DURATION_PLACE,
    LIFT_OVERRIDE,
    PLACE_VARIATION_DEG,
    POST_DROP_PAUSE,
    STORAGE_PLACE,
)

# Target control frequency (Hz)
CONTROL_HZ = 30


# =============================================================================
# Core Lerp Engine
# =============================================================================

def lerp_to_waypoint(
    robot,
    target: Dict[str, float],
    duration: float,
    fixed_joints: Optional[Dict[str, float]] = None,
) -> None:
    """
    Smoothly interpolate from the robot's current joint positions to target
    using a Smoothstep (ease-in/ease-out) curve.

    Args:
        robot:        LeRobot Robot instance with get_observation / send_action.
        target:       Mapping of joint_name → target angle (degrees).
                      Joints absent from target are held at their current value.
        duration:     Total duration of the move in seconds.
        fixed_joints: Optional overrides applied at every step regardless of
                      interpolation (e.g. keep gripper closed during transport).
    """
    # Read the current state once at the start of the segment
    obs   = robot.get_observation()
    start = {j: float(obs.get(j, HOME_ACTION.get(j, 0.0))) for j in JOINT_NAMES}

    # Merge: joints not in `target` stay at their current position
    end = {**start, **target}

    steps = max(2, int(duration * CONTROL_HZ))

    for step in range(steps):
        tic = time.time()

        t     = step / (steps - 1)                   # 0.0 → 1.0
        alpha = t * t * (3.0 - 2.0 * t)              # Smoothstep

        action = {
            j: start[j] + (end[j] - start[j]) * alpha
            for j in JOINT_NAMES
        }

        # Apply fixed overrides (e.g. keep gripper locked during transport)
        if fixed_joints:
            action.update(fixed_joints)

        robot.send_action(action)

        elapsed = time.time() - tic
        sleep_t = 1.0 / CONTROL_HZ - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# =============================================================================
# Move
# =============================================================================

def move_to_home(robot, duration: float = LERP_DURATION_HOME) -> None:
    """Smoothly move the robot to the predefined Home position."""
    print(f">>> Moving to HOME over {duration:.1f}s …")
    lerp_to_waypoint(robot, HOME_ACTION, duration)


def move_to_ready(robot, duration: float = LERP_DURATION_LIFT) -> None:
    """Smoothly move the robot to the lift position."""
    print(f">>> Moving to LIFT over {duration:.1f}s …")
    d = {
        "shoulder_pan.pos":  36.2,
        "shoulder_lift.pos": -21.1,
        "elbow_flex.pos":    27.1,
        "wrist_flex.pos":    78.4,
    }
    lerp_to_waypoint(robot, d, duration, fixed_joints={"gripper.pos": 40.0})


# =============================================================================
# Scripted Transport Sequence
# =============================================================================

def scripted_transport(robot, task_type: str, grasp_obs: Dict) -> None:
    """
    Execute the deterministic pick-to-place trajectory after a confirmed grasp.

    Sequence:
      1. Lift vertically (keep pan, roll, gripper from grasp pose).
      2. Rotate & approach placement zone (keep gripper closed).
      3. Drop the cube (open gripper).
      4. Pause briefly so the cube settles.
      5. Lift back up (avoid collision with placed cube).
      6. Return to Home.

    Args:
        robot:      LeRobot Robot instance.
        task_type:  "check_in" → cube goes to STORAGE; "check_out" → CHECKOUT.
        grasp_obs:  Observation dict captured at the moment grasp was confirmed.
                    Used to read the locked gripper/wrist_roll from VLA's pose.
    """
    # --- Determine placement target ---
    place_waypoint = (
        STORAGE_PLACE.copy() if task_type == "check_in" else CHECKOUT_PLACE.copy()
    )

    # Add small random variation to prevent repetitive mechanical wear
    rng = np.random.default_rng()
    for joint in ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"):
        place_waypoint[joint] += rng.uniform(-PLACE_VARIATION_DEG, PLACE_VARIATION_DEG)

    # --- Lock gripper closed throughout transport ---
    gripper_closed = float(grasp_obs.get("gripper.pos", 15.0))
    transport_lock = {"gripper.pos": gripper_closed}

    # ── Step 1: Lift up (pan and roll unchanged, gripper locked) ──────────────
    print("  [Transport] Step 1/5 — Lifting arm vertically …")
    lerp_to_waypoint(
        robot,
        LIFT_OVERRIDE,                   # only lift/elbow/wrist_flex change
        LERP_DURATION_LIFT,
        fixed_joints=transport_lock,
    )

    # ── Step 2: Rotate & approach target zone ─────────────────────────────────
    print("  [Transport] Step 2/5 — Rotating to placement zone …")
    lerp_to_waypoint(
        robot,
        place_waypoint,
        LERP_DURATION_PLACE,
        fixed_joints=transport_lock,
    )

    # ── Step 3: Open gripper — release the cube ───────────────────────────────
    print("  [Transport] Step 3/5 — Releasing cube …")
    obs = robot.get_observation()
    drop_pose = {j: float(obs.get(j, place_waypoint.get(j, 0.0))) for j in JOINT_NAMES}
    drop_pose["gripper.pos"] = GRIPPER_OPEN_POS
    lerp_to_waypoint(robot, drop_pose, LERP_DURATION_DROP)

    # ── Step 4: Pause so the cube can settle ──────────────────────────────────
    print(f"  [Transport] Step 4/5 — Pausing {POST_DROP_PAUSE:.1f}s …")
    time.sleep(POST_DROP_PAUSE)

    # ── Step 5: Lift back up before returning home (avoid collision) ──────────
    print("  [Transport] Step 5/5 — Lifting clear of placed cube …")
    lerp_to_waypoint(robot, LIFT_OVERRIDE, LERP_DURATION_LIFT)

    # ── Return to Home ────────────────────────────────────────────────────────
    move_to_home(robot)