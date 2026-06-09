"""
SO100 Motion Primitives

Smooth trajectory execution between joint-space waypoints.

Public API:
  lerp_to_waypoint()      — smoothly interpolate from current pose to a target.
  spline_trajectory()     — smooth Catmull-Rom path through multiple waypoints.
  move_to_home()          — convenience wrapper back to HOME_ACTION.
  move_to_ready()         — task-specific approach position, defined in constants.
  scripted_transport()    — full pick-to-place sequence using spline_trajectory.
  execute_failure_shake() — visual head shake for failure indications.
"""

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.constants import (
    CHECKOUT_PLACE,
    GRIPPER_GRASP_POS,
    GRIPPER_OPEN_POS,
    GRIPPER_TRANSPORT_MIN,
    HOME_ACTION,
    JOINT_NAMES,
    LERP_DURATION_DROP,
    LERP_DURATION_HOME,
    LERP_DURATION_LIFT,
    LERP_DURATION_PLACE,
    LIFT_OVERRIDE,
    PLACE_VARIATION_DEG,
    POST_DROP_PAUSE,
    READY_POSITIONS,
    STORAGE_PLACE,
)

# Target control frequency (Hz)
CONTROL_HZ = 30


# =============================================================================
# Exceptions
# =============================================================================

class GraspLostException(Exception):
    """Raised when an object slips from the gripper during transit."""
    def __init__(self, message: str, last_obs: Dict[str, Any]):
        super().__init__(message)
        self.last_obs = last_obs


# =============================================================================
# Core Lerp Engine  (Quintic Smoothstep)
# =============================================================================

def lerp_to_waypoint(
    robot,
    target: Dict[str, float],
    duration: float,
    fixed_joints: Optional[Dict[str, float]] = None,
    monitor_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> None:
    """
    Smoothly interpolate from the robot's current joint positions to *target*
    using a Quintic Smoothstep curve.

    Classic Smoothstep  (t² (3 − 2t))  zeroes velocity at endpoints but leaves
    acceleration non-zero, creating a subtle snap at the start and finish.
    Quintic Smoothstep  (6t⁵ − 15t⁴ + 10t³)  also zeroes *acceleration* at
    both endpoints, yielding a perceptibly more natural, mechanical-wear-friendly
    motion profile.

    Args:
        robot:            LeRobot Robot instance (get_observation / send_action).
        target:           Mapping of joint_name → target angle (degrees).
                          Joints absent from *target* are held at their current value.
        duration:         Total duration of the move in seconds.
        fixed_joints:     Optional overrides applied at every step regardless of
                          interpolation (e.g. keep gripper closed during transport).
        monitor_callback: Function evaluated every step. If it returns False, execution aborts.
    """
    obs   = robot.get_observation()
    start = {j: float(obs.get(j, HOME_ACTION.get(j, 0.0))) for j in JOINT_NAMES}
    end   = {**start, **target}

    steps = max(2, int(duration * CONTROL_HZ))

    for step in range(steps):
        tic = time.time()

        t     = step / (steps - 1)                        # 0.0 → 1.0
        alpha = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # quintic smoothstep

        action = {j: start[j] + (end[j] - start[j]) * alpha for j in JOINT_NAMES}

        if fixed_joints:
            action.update(fixed_joints)

        robot.send_action(action)

        # Monitor Transit Health (The "Oops" factor)
        if monitor_callback is not None:
            obs_now = robot.get_observation()
            if not monitor_callback(obs_now):
                raise GraspLostException("Grasp lost during linear interpolation.", obs_now)

        elapsed = time.time() - tic
        sleep_t = 1.0 / CONTROL_HZ - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# =============================================================================
# Catmull-Rom Spline Trajectory
# =============================================================================

def _catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    """
    Scalar Catmull-Rom interpolation between p1 and p2 at parameter t ∈ [0, 1].

    The tangent at p1 is estimated as 0.5 * (p2 − p0) and at p2 as
    0.5 * (p3 − p1), so velocity is continuous across segment boundaries —
    the key property that eliminates stop-start jerk at intermediate waypoints.
    """
    return 0.5 * (
          2.0 * p1
        + (-p0 + p2) * t
        + ( 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t ** 2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t ** 3
    )

def arc_trajectory(
    robot,
    p_mid: Dict[str, float],
    p_end: Dict[str, float],
    duration: float,
    fixed_joints: Optional[Dict[str, float]] = None,
    monitor_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> None:
    """
    Executes a smooth Quadratic Bezier curve.
    Instead of rigidly stopping at waypoints, this sweeps smoothly from the
    current pose, pulls gracefully toward p_mid, and lands exactly at p_end.
    """
    obs = robot.get_observation()
    p0 = {j: float(obs.get(j, HOME_ACTION.get(j, 0.0))) for j in JOINT_NAMES}
    p1 = {**p0, **p_mid}
    p2 = {**p0, **p_end}

    steps = max(2, int(duration * CONTROL_HZ))
    for step in range(steps):
        tic = time.time()
        
        # Smoothstep time scaling for gentle acceleration/deceleration
        t = step / (steps - 1)
        t_smooth = t * t * (3.0 - 2.0 * t)

        action = {}
        for j in JOINT_NAMES:
            # Quadratic Bezier formula
            action[j] = (
                ((1 - t_smooth) ** 2) * p0[j] +
                2 * (1 - t_smooth) * t_smooth * p1[j] +
                (t_smooth ** 2) * p2[j]
            )

        if fixed_joints:
            action.update(fixed_joints)

        robot.send_action(action)

        # Monitor Transit Health
        if monitor_callback is not None:
            obs_now = robot.get_observation()
            if not monitor_callback(obs_now):
                raise GraspLostException("Grasp lost during arc trajectory.", obs_now)

        elapsed = time.time() - tic
        sleep_t = 1.0 / CONTROL_HZ - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)


# =============================================================================
# Named Moves
# =============================================================================

def move_to_home(robot, duration: float = LERP_DURATION_HOME) -> None:
    """Smoothly move the robot to the predefined Home position."""
    print(f">>> Moving to HOME over {duration:.1f}s …")
    lerp_to_waypoint(robot, HOME_ACTION, duration)


def move_to_ready(
    robot, 
    task_type: str, 
    duration: float = LERP_DURATION_LIFT,
    pan_override: Optional[float] = None
) -> None:
    """
    Move to the task-specific approach / ready position.

    Target joints are looked up from constants.READY_POSITIONS[task_type].
    If pan_override is provided, the robot will adopt the safe ready elevation
    (lift/elbow/wrist) but hold the requested shoulder pan to look down at a custom area.
    The gripper is always forced open (GRIPPER_OPEN_POS) throughout the move.

    Args:
        robot:        LeRobot Robot instance.
        task_type:    Task identifier string, e.g. "check_in" or "check_out".
        duration:     Move duration in seconds.
        pan_override: Optional custom shoulder pan to use instead of the default.

    Raises:
        ValueError if task_type is not present in READY_POSITIONS.
    """
    base_target = READY_POSITIONS.get(task_type)
    if base_target is None:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Expected one of: {list(READY_POSITIONS.keys())}"
        )
        
    target = base_target.copy()
    if pan_override is not None:
        target["shoulder_pan.pos"] = pan_override
        
    print(f">>> Moving to READY [{task_type}] over {duration:.1f}s …")
    lerp_to_waypoint(
        robot,
        target,
        duration,
        fixed_joints={"gripper.pos": GRIPPER_OPEN_POS},
    )

def execute_failure_shake(robot) -> None:
    """
    Shakes the robot's 'head' (pan and wrist roll) to indicate failure/absence of an object.
    Requires the robot to be starting roughly near HOME_ACTION for visual effect.
    """
    print("  [Motion] Executing failure 'shake' motion...")
    obs = robot.get_observation()
    base_pose = {j: float(obs.get(j, HOME_ACTION.get(j, 0.0))) for j in JOINT_NAMES}
    
    shake_poses = []
    
    # Shake right
    pose1 = base_pose.copy()
    pose1["shoulder_pan.pos"] = -20.0
    pose1["wrist_roll.pos"] = -20.0
    shake_poses.append(pose1)
    
    # Shake left
    pose2 = base_pose.copy()
    pose2["shoulder_pan.pos"] = 20.0
    pose2["wrist_roll.pos"] = 20.0
    shake_poses.append(pose2)

    # Return to base
    shake_poses.append(base_pose)

    # Do it thrice
    for _ in range(3): 
        for pose in shake_poses:
            lerp_to_waypoint(robot, pose, 0.3)


# =============================================================================
# Scripted Transport Sequence
# =============================================================================

def scripted_transport(
    robot, 
    task_type: str, 
    grasp_obs: Dict, 
    monitor_callback: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> None:
    """
    Execute the deterministic pick-to-place trajectory using fluid Bezier arcs.
    Now actively monitored by monitor_callback to detect mid-air object drops.
    """
    if not hasattr(scripted_transport, "drop_counter"):
        scripted_transport.drop_counter = 0
    
    base_waypoint = (
        STORAGE_PLACE.copy() if task_type == "check_in" else CHECKOUT_PLACE.copy()
    )

    # ── Calculate Grid Offset ──
    # Create a 2x2 grid of drop points to prevent stacking collisions
    grid_offsets = [
        {"shoulder_pan.pos": 0.0,  "wrist_flex.pos": 0.0},    # Center
        {"shoulder_pan.pos": 5.0,  "wrist_flex.pos": 0.0},    # Right
        {"shoulder_pan.pos": -5.0, "wrist_flex.pos": 0.0},    # Left
        {"shoulder_pan.pos": 0.0,  "wrist_flex.pos": 5.0},    # Further forward
    ]
    
    offset = grid_offsets[scripted_transport.drop_counter % len(grid_offsets)]
    place_waypoint = base_waypoint.copy()
    place_waypoint["shoulder_pan.pos"] += offset["shoulder_pan.pos"]
    place_waypoint["wrist_flex.pos"] += offset["wrist_flex.pos"]
    
    scripted_transport.drop_counter += 1 # Increment for the next run!

    rng = np.random.default_rng()
    for joint in ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"):
        place_waypoint[joint] += rng.uniform(-PLACE_VARIATION_DEG, PLACE_VARIATION_DEG)

    raw_grip       = float(grasp_obs.get("gripper.pos", GRIPPER_GRASP_POS))
    gripper_locked = min(raw_grip, GRIPPER_TRANSPORT_THRESHOLD) 
    transport_lock = {"gripper.pos": gripper_locked}

    # ── Phase A: Sweeping Arc (Lift to Place) ──
    print("  [Transport] Phase A — Sweeping Arc to placement zone …")
    arc_trajectory(
        robot,
        p_mid=LIFT_OVERRIDE,
        p_end=place_waypoint,
        duration=(LERP_DURATION_LIFT + LERP_DURATION_PLACE) * 0.8, # Speed up slightly
        fixed_joints=transport_lock,
        monitor_callback=monitor_callback,  # Only applied while holding the object!
    )

    # ── Phase B: Open gripper ──
    print("  [Transport] Phase B — Releasing cube …")
    obs       = robot.get_observation()
    drop_pose = {j: float(obs.get(j, place_waypoint.get(j, 0.0))) for j in JOINT_NAMES}
    drop_pose["gripper.pos"] = GRIPPER_OPEN_POS
    lerp_to_waypoint(robot, drop_pose, LERP_DURATION_DROP)

    # ── Phase C: Settle pause ──
    print(f"  [Transport] Phase C — Settling pause ({POST_DROP_PAUSE:.2f}s) …")
    time.sleep(POST_DROP_PAUSE)

    # ── Phase D: Sweeping Arc (Return Home) ──
    print("  [Transport] Phase D — Sweeping Arc to home …")
    arc_trajectory(
        robot,
        p_mid=LIFT_OVERRIDE,
        p_end=HOME_ACTION,
        duration=(LERP_DURATION_LIFT + LERP_DURATION_HOME) * 0.8,
    )