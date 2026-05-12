"""
SO100 Position Reader (Based on eval_so100 logic)
Use this to calibrate your A and B waypoints.
"""

from dataclasses import asdict, dataclass
import logging
import time
from typing import Dict

import draccus
import numpy as np

# Same imports as your eval script to ensure hardware compatibility
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.utils import init_logging

@dataclass
class EvalConfig:
    """
    Mirroring your exact EvalConfig so the CLI arguments match.
    """
    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = ""
    play_sounds: bool = False
    timeout: int = 30

@draccus.wrap()
def main(cfg: EvalConfig):
    init_logging()
    
    # 1. Initialize Robot Hardware (Identical to eval_so100)
    print(f"\nConnecting to robot on {cfg.robot.port}...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # These are the keys used by your adapter
    robot_state_keys = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    print("\n" + "="*60)
    print(" READING ROBOT POSITIONS")
    print(" Move the arm to your desired A or B points.")
    print(" Press Ctrl+C to exit.")
    print("="*60 + "\n")

    try:
        while True:
            # Read from servos
            obs = robot.get_observation()
            
            # Print in a format ready for code insertion
            print("waypoint = {")
            for k in robot_state_keys:
                val = obs.get(k, 0.0)
                print(f"    \"{k}\": {val:8.3f},")
            print("}")
            
            # Terminal trick: Move cursor back up 8 lines to refresh the view
            print("\033[8A", end="")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopped reading positions.")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()