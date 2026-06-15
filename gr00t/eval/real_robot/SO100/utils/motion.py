import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from utils.constants import (
    CHECKIN_PLACE, CHECKOUT_PLACE, GRIPPER_GRASP_POS, GRIPPER_OPEN_POS,
    GRIPPER_TRANSPORT_THRESHOLD, HOME_ACTION, JOINT_NAMES, LERP_DURATION_DROP,
    LERP_DURATION_HOME, LERP_DURATION_LIFT, LERP_DURATION_PLACE, LIFT_OVERRIDE,
    PLACE_VARIATION_DEG, POST_DROP_PAUSE, READY_POSITIONS, STORAGE_PLACE,
)

CONTROL_HZ = 30

class GraspLostException(Exception):
    def __init__(self, message: str, last_obs: Dict[str, Any]):
        super().__init__(message)
        self.last_obs = last_obs

def _pause_if_hand_detected(robot, safety_monitor, hold_action):
    """Holds position if the monitor's flag is triggered."""
    if safety_monitor is None or not safety_monitor.enabled: return
    
    if safety_monitor.is_hand_present():
        print("\n🛑 [SAFETY] Hand detected! Pausing motion...")
        while True:
            robot.send_action(hold_action)
            time.sleep(1.0 / CONTROL_HZ)
            
            # Keep feeding the monitor so it knows when the hand leaves
            obs = robot.get_observation()
            safety_monitor.update_frame(obs["front"])
            
            if not safety_monitor.is_hand_present():
                print("▶️ [SAFETY] Hand removed. Resuming motion...")
                break

def lerp_to_waypoint(
    robot,
    target: Dict[str, float],
    duration: float,
    fixed_joints: Optional[Dict[str, float]] = None,
    monitor_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    safety_monitor = None
) -> None:
    obs = robot.get_observation()
    start = {j: float(obs.get(j, HOME_ACTION.get(j, 0.0))) for j in JOINT_NAMES}
    end = {**start, **target}
    steps = max(2, int(duration * CONTROL_HZ))

    for step in range(steps):
        tic = time.time()
        
        t = step / (steps - 1)
        alpha = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

        action = {j: start[j] + (end[j] - start[j]) * alpha for j in JOINT_NAMES}
        if fixed_joints: action.update(fixed_joints)

        robot.send_action(action)

        # LATENCY FIX: Fetch observation and check every single step
        obs_now = None
        if (safety_monitor is not None and safety_monitor.enabled) or (monitor_callback is not None):
            obs_now = robot.get_observation()

        if safety_monitor is not None and safety_monitor.enabled and obs_now is not None:
            safety_monitor.update_frame(obs_now["front"])
            _pause_if_hand_detected(robot, safety_monitor, action)

        if monitor_callback is not None and obs_now is not None:
            if not monitor_callback(obs_now):
                raise GraspLostException("Grasp lost during linear interpolation.", obs_now)

        elapsed = time.time() - tic
        sleep_t = 1.0 / CONTROL_HZ - elapsed
        if sleep_t > 0: time.sleep(sleep_t)

def arc_trajectory(
    robot,
    p_mid: Dict[str, float],
    p_end: Dict[str, float],
    duration: float,
    fixed_joints: Optional[Dict[str, float]] = None,
    monitor_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    safety_monitor = None
) -> None:
    obs = robot.get_observation()
    p0 = {j: float(obs.get(j, HOME_ACTION.get(j, 0.0))) for j in JOINT_NAMES}
    p1 = {**p0, **p_mid}
    p2 = {**p0, **p_end}

    steps = max(2, int(duration * CONTROL_HZ))
    for step in range(steps):
        tic = time.time()
        
        t = step / (steps - 1)
        t_smooth = t * t * (3.0 - 2.0 * t)

        action = {}
        for j in JOINT_NAMES:
            action[j] = (((1 - t_smooth) ** 2) * p0[j] + 2 * (1 - t_smooth) * t_smooth * p1[j] + (t_smooth ** 2) * p2[j])

        if fixed_joints: action.update(fixed_joints)

        robot.send_action(action)

        # Fetch observation and check every single step
        obs_now = None
        if (safety_monitor is not None and safety_monitor.enabled) or (monitor_callback is not None):
            obs_now = robot.get_observation()

        if safety_monitor is not None and safety_monitor.enabled and obs_now is not None:
            safety_monitor.update_frame(obs_now["front"])
            _pause_if_hand_detected(robot, safety_monitor, action)

        if monitor_callback is not None and obs_now is not None:
            if not monitor_callback(obs_now):
                raise GraspLostException("Grasp lost during arc trajectory.", obs_now)

        elapsed = time.time() - tic
        sleep_t = 1.0 / CONTROL_HZ - elapsed
        if sleep_t > 0: time.sleep(sleep_t)

def move_to_home(robot, duration: float = LERP_DURATION_HOME, safety_monitor=None) -> None:
    print(f">>> Moving to HOME over {duration:.1f}s …")
    lerp_to_waypoint(robot, HOME_ACTION, duration, safety_monitor=safety_monitor)

def move_to_ready(
    robot, task_type: str, duration: float = LERP_DURATION_LIFT,
    pan_override: Optional[float] = None, safety_monitor=None
) -> None:
    base_target = READY_POSITIONS.get(task_type)
    target = base_target.copy()
    if pan_override is not None: target["shoulder_pan.pos"] = pan_override
    print(f">>> Moving to READY [{task_type}] over {duration:.1f}s …")
    lerp_to_waypoint(robot, target, duration, fixed_joints={"gripper.pos": GRIPPER_OPEN_POS}, safety_monitor=safety_monitor)

def execute_failure_shake(robot) -> None:
    obs = robot.get_observation()
    base_pose = {j: float(obs.get(j, HOME_ACTION.get(j, 0.0))) for j in JOINT_NAMES}
    shake_poses = []
    
    pose1 = base_pose.copy(); pose1["shoulder_pan.pos"] = -20.0; pose1["wrist_roll.pos"] = -20.0
    shake_poses.append(pose1)
    pose2 = base_pose.copy(); pose2["shoulder_pan.pos"] = 20.0; pose2["wrist_roll.pos"] = 20.0
    shake_poses.append(pose2)
    shake_poses.append(base_pose)

    for _ in range(3): 
        for pose in shake_poses: lerp_to_waypoint(robot, pose, 0.3)

def scripted_transport(
    robot, task_type: str, grasp_obs: Dict, 
    monitor_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    safety_monitor=None
) -> None:
    if not hasattr(scripted_transport, "drop_counter"): scripted_transport.drop_counter = 0
    
    base_waypoint = STORAGE_PLACE.copy()
    if task_type == "check_out": base_waypoint = CHECKOUT_PLACE.copy()
    elif task_type == "check_back": base_waypoint = CHECKIN_PLACE.copy()

    grid_offsets = [
        {"shoulder_pan.pos": 0.0, "wrist_flex.pos": 0.0},
        {"shoulder_pan.pos": 2.0, "wrist_flex.pos": 0.0},
        {"shoulder_pan.pos": -2.0, "wrist_flex.pos": 0.0},
        {"shoulder_pan.pos": 0.0, "wrist_flex.pos": 2.0},
        {"shoulder_pan.pos": 0.0, "wrist_flex.pos": -2.0},
    ]
    
    offset = grid_offsets[scripted_transport.drop_counter % len(grid_offsets)]
    place_waypoint = base_waypoint.copy()
    place_waypoint["shoulder_pan.pos"] += offset["shoulder_pan.pos"]
    place_waypoint["wrist_flex.pos"] += offset["wrist_flex.pos"]
    scripted_transport.drop_counter += 1

    rng = np.random.default_rng()
    for joint in ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"):
        place_waypoint[joint] += rng.uniform(-PLACE_VARIATION_DEG, PLACE_VARIATION_DEG)

    raw_grip = float(grasp_obs.get("gripper.pos", GRIPPER_GRASP_POS))
    gripper_locked = min(raw_grip, GRIPPER_TRANSPORT_THRESHOLD) 
    transport_lock = {"gripper.pos": gripper_locked}

    print("  [Transport] Sweeping Arc to placement zone …")
    arc_trajectory(
        robot, p_mid=LIFT_OVERRIDE, p_end=place_waypoint,
        duration=(LERP_DURATION_LIFT + LERP_DURATION_PLACE) * 0.8,
        fixed_joints=transport_lock, monitor_callback=monitor_callback,
        safety_monitor=safety_monitor
    )

    print("  [Transport] Releasing cube …")
    obs = robot.get_observation()
    drop_pose = {j: float(obs.get(j, place_waypoint.get(j, 0.0))) for j in JOINT_NAMES}
    drop_pose["gripper.pos"] = GRIPPER_OPEN_POS
    lerp_to_waypoint(robot, drop_pose, LERP_DURATION_DROP, safety_monitor=safety_monitor)

    time.sleep(POST_DROP_PAUSE)

    print("  [Transport] Sweeping Arc to home …")
    arc_trajectory(
        robot, p_mid=LIFT_OVERRIDE, p_end=HOME_ACTION,
        duration=(LERP_DURATION_LIFT + LERP_DURATION_HOME) * 0.8,
        safety_monitor=safety_monitor
    )