"""
SO100 Robot Calibration Tool: Get Joint Positions

Usage:
    PYTHONPATH=. python gr00t/eval/real_robot/SO100/get_pos.py \
      --robot.type=so101_follower \
      --robot.port=/dev/ttyACM0 \
      --robot.id=follower_arm
"""

from dataclasses import dataclass
import time
import draccus
import logging
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.utils import init_logging

@dataclass
class GetPosConfig:
    robot: RobotConfig
    # We include these just so the CLI arguments from eval_so100.py don't cause errors
    policy_host: str = "localhost"
    policy_port: int = 5555
    lang_instruction: str = ""

@draccus.wrap()
def main(cfg: GetPosConfig):
    init_logging()
    
    # 1. Initialize Robot Hardware
    print(f"Connecting to robot on {cfg.robot.port}...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # The joint names we care about for calibration
    joint_keys = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    print("\n" + "="*50)
    print(" ROBOT CALIBRATION MODE")
    print(" Move the arm by hand. Press Ctrl+C to stop.")
    print("="*50 + "\n")

    try:
        while True:
            # Get latest observation from servos
            obs = robot.get_observation()
            
            # Print in a Python-dictionary-friendly format for easy copy-pasting
            print("current_pos = {")
            for k in joint_keys:
                val = obs.get(k, 0.0)
                print(f"    \"{k}\": {val:8.3f},")
            print("}")
            
            # Add a small separator and move cursor back up (terminal trick)
            print("\033[9A", end="") # Move up 9 lines
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nExiting calibration mode.")
    finally:
        # Properly disconnect
        robot.disconnect()

if __name__ == "__main__":
    main()