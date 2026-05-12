"""
SO100 Real-Robot Gr00T Policy Evaluation Script (WITH AUTO-STOP VISION)

This script runs closed-loop policy evaluation on the SO100 / SO101 robots.
It uses an OpenCV heuristic to detect task completion (cube placed and arm removed)
to prevent infinite execution loops.
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import time
from typing import Any, Dict, List

import cv2
import draccus
import numpy as np

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots import koch_follower  # noqa: F401
from lerobot.robots import so_follower as so100_follower  # noqa: F401
from lerobot.robots import so_follower as so101_follower  # noqa: F401
from lerobot.utils.utils import init_logging, log_say


# =============================================================================
# Task Vision Constants & Home Position
# =============================================================================

STORAGE_ZONE   = (277, 168, 96, 94)
CHECK_IN_ZONE  = (399, 100, 99, 97)
CHECK_OUT_ZONE = (152, 103, 96, 95)

# Based on your calibration. 
# Format: List of (Lower_HSV, Upper_HSV) tuples.
# Note: Red requires two ranges because Hue wraps around 0 and 180.
COLOR_RANGES = {
    "red": [
        (np.array([0, 10, 100]), np.array([10, 255, 255])),
        (np.array([165, 10, 100]), np.array([180, 255, 255]))
    ],
    "blue": [
        # Your blue V was very low (13), meaning it's dark. Range expanded to catch it.
        (np.array([80, 50, 0]), np.array([120, 255, 255]))
    ],
    "yellow": [
        (np.array([20, 50, 50]), np.array([45, 255, 255]))
    ]
}

HOME_ACTION = {
    "shoulder_pan.pos":    0.0,
    "shoulder_lift.pos":  -60.0,
    "elbow_flex.pos":      60.0,
    "wrist_flex.pos":      60.0,
    "wrist_roll.pos":      90.0,
    "gripper.pos":         40.0,
}

# =============================================================================
# Core Helper Functions
# =============================================================================

def recursive_add_extra_dim(obs: Dict) -> Dict:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  
    return obs


def move_to_home(robot: Robot):
    """Sends the robot to the predefined home position."""
    print(">>> Moving to HOME position...")
    # Send command repeatedly for ~1.5 seconds to ensure smooth travel
    for _ in range(45):
        robot.send_action(HOME_ACTION)
        time.sleep(1.0 / 30)


def check_task_success(img_np, baseline_np, zone, color_name) -> bool:
    """
    Checks if a target color blob is securely inside the target zone,
    using background subtraction to ignore static colored paper.
    """
    x, y, w, h = zone
    
    # Crop current frame and baseline frame
    crop = img_np[y:y+h, x:x+w]
    base_crop = baseline_np[y:y+h, x:x+w]
    
    # 1. Find what changed (Background Subtraction)
    diff = cv2.absdiff(crop, base_crop)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    # Threshold: Pixels must change by at least 25 (out of 255) to be considered "new"
    _, diff_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
    
    # 2. Find target colors in the current frame
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_mask = np.zeros((h, w), dtype=np.uint8)
    ranges = COLOR_RANGES.get(color_name,[])
    for (lower, upper) in ranges:
        m = cv2.inRange(hsv, lower, upper)
        color_mask = cv2.bitwise_or(color_mask, m)
        
    # 3. Combine them: Must be the right color AND a newly changed pixel!
    # This completely eliminates the background paper.
    final_mask = cv2.bitwise_and(color_mask, diff_mask)
    
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Check contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Minimum pixel size of the cube
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # Check if the object touches the edges of the cropped zone
            margin = 3
            touches_left = (bx <= margin)
            touches_right = (bx + bw >= w - margin)
            touches_top = (by <= margin)
            touches_bottom = (by + bh >= h - margin)
            
            if not (touches_left or touches_right or touches_top or touches_bottom):
                return True
                
    return False

# =============================================================================
# Policy Adapter
# =============================================================================

class So100Adapter:
    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client
        self.robot_state_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ]
        self.camera_keys = ["front", "wrist"]

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        model_obs = {}
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:5],
            "gripper": state[5:6],
        }
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        single_arm = chunk["single_arm"][0][t]
        gripper = chunk["gripper"][0][t]
        full = np.concatenate([single_arm, gripper], axis=0)
        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]

# =============================================================================
# Evaluation Config & Main Loop
# =============================================================================

@dataclass
class EvalConfig:
    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "Check in red cube" # CHANGE THIS ON CLI
    play_sounds: bool = False
    timeout: int = 40 # Maximum seconds before it forces a stop


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 1. Parse the Instruction to determine Target Zone and Color
    instruction = cfg.lang_instruction.lower()
    
    # Determine Action Type
    if "in" in instruction:
        target_zone = STORAGE_ZONE
    elif "out" in instruction:
        target_zone = CHECK_OUT_ZONE
    else:
        target_zone = STORAGE_ZONE # Default fallback

    # Determine Target Color
    target_color = "red"
    if "blue" in instruction: target_color = "blue"
    elif "yellow" in instruction: target_color = "yellow"

    print(f"\n[PARSER] Task: {cfg.lang_instruction}")
    print(f"[PARSER] Looking for {target_color.upper()} cube in {target_zone}\n")

    # 2. Initialize Hardware & Policy
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = So100Adapter(policy_client)

    log_say("Hardware and Policy initialized.", cfg.play_sounds)

    # 3. Move to Initial Home Position First
    move_to_home(robot)
    time.sleep(0.5)  # Let the camera auto-exposure settle for a split second
    
    print(">>> Taking baseline snapshot of workspace...")
    baseline_obs = robot.get_observation()
    baseline_img = baseline_obs["front"]

    # 4. Main Loop
    start_time = time.time()
    
    while True:
        # Failsafe Timeout
        if time.time() - start_time > cfg.timeout:
            print("\n[TIMEOUT] Max execution time reached. Terminating loop.")
            break

        obs = robot.get_observation()
        obs["lang"] = cfg.lang_instruction 
        
        # --- VISION CHECK: IS TASK COMPLETE? ---
        # Note: We now pass baseline_img in as the second argument!
        is_done = check_task_success(obs["front"], baseline_img, target_zone, target_color)
        
        if is_done:
            print(f"\n🎉 [SUCCESS] {target_color.upper()} cube securely detected in zone!")
            print("🎉 [SUCCESS] Arm has cleared the area. Task Complete.")
            break

        # --- RUN INFERENCE ---
        actions = policy.get_action(obs)

        for i, action_dict in enumerate(actions[: cfg.action_horizon]):
            tic = time.time()
            robot.send_action(action_dict)
            toc = time.time()
            if toc - tic < 1.0 / 30:
                time.sleep(1.0 / 30 - (toc - tic))

    # 5. Terminate: Return to Home 
    print("\n>>> Task Finished. Returning to HOME position...")
    move_to_home(robot)
    print(">>> System Shutting Down cleanly.")


if __name__ == "__main__":
    eval()