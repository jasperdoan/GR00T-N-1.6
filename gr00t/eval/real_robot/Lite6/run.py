import argparse
import sys
from utils.constants import DEFAULT_IP
from utils.robot import Lite6Controller

def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Lite 6 Hybrid CV Control")
    parser.add_argument("--ip", type=str, default=DEFAULT_IP, 
                        help="IP address of the UFACTORY Lite 6")
    parser.add_argument("--demo", action="store_true", 
                        help="Run a simple robot connection and movement demo")
    
    args = parser.parse_args()

    # 2. Main Logic Route
    if args.demo:
        print("--- RUNNING DEMO MODE ---")
        robot = Lite6Controller(args.ip)
        robot.connect()
        
        if robot.arm:
            # Just grab the current position and print it to prove it works
            pos = robot.get_current_position()
            if pos:
                print(f"Current Position: X:{pos[0]:.1f}, Y:{pos[1]:.1f}, Z:{pos[2]:.1f}")
            
            robot.disconnect()
    else:
        print("No action specified. Run with --help for options.")
        print("To test the connection, run: python run.py --demo")

if __name__ == "__main__":
    main()