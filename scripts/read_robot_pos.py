"""
SO100 Waypoint Calibration Tool
Usage: python read_robot_pos.py --port /dev/ttyACM0
"""

import time
import draccus
from dataclasses import dataclass
from lerobot.robots import make_robot_from_config, RobotConfig

@dataclass
class CalibrationConfig:
    # Match your hardware settings
    port: str = "/dev/ttyACM0"
    robot_type: str = "so101_follower"  # or so_follower
    robot_id: str = "follower_arm"

def main():
    cfg = CalibrationConfig()
    
    # 1. Create the LeRobot config manually for the hardware
    # This mirrors what you pass in your CLI command
    robot_cfg = RobotConfig(
        type=cfg.robot_type,
        port=cfg.port,
        id=cfg.robot_id,
        # We don't need cameras for position calibration
        cameras={} 
    )

    print(f"Connecting to {cfg.robot_type} on {cfg.port}...")
    robot = make_robot_from_config(robot_cfg)
    robot.connect()

    # The joint names we care about for the SO100
    joint_keys = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    print("\n" + "="*50)
    print(" ROBOT POSITION CALIBRATOR")
    print(" Move the arm manually and copy the values below.")
    print(" Press Ctrl+C to exit.")
    print("="*50 + "\n")

    try:
        while True:
            # 2. Get current state from servos
            obs = robot.get_observation()
            
            # 3. Print in a dictionary format for easy copy-pasting
            print("action_dict = {")
            for k in joint_keys:
                val = obs.get(k, 0.0)
                print(f'    "{k}": {val:8.3f},')
            print("}\n")
            
            time.sleep(1.0) # Refresh every second
            
    except KeyboardInterrupt:
        print("\nCalibration finished. Closing connection.")
    finally:
        # Important: Don't just kill the script, close the port
        # though LeRobot handles this gracefully, it's good practice.
        pass

if __name__ == "__main__":
    main()