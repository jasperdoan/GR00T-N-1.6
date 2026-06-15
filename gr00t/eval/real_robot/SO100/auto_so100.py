"""
SO100 Auto Demo (Continuous Loop)
=================================================================
Continuously runs the robot through check_in, check_out, and check_back.
Uses stateless auto-discovery: It looks at the workspace, figures out 
where the red cube is, and executes the corresponding next logical step.
"""

import logging
import time
import os
import signal
import sys
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots import koch_follower, so100_follower, so101_follower  # noqa: F401
from lerobot.utils.utils import init_logging, log_say

from utils.adapter        import So100Adapter
from utils.motion         import move_to_home
from utils.policy_runner  import AsyncPolicyRunner
from utils.vision_utils   import GraspDetector, check_color_presence_front, SafetyMonitor, save_workspace_snapshot
from utils.fsm_controller import EvaluationFSM, FSMState
from utils.constants      import CHECK_IN_ZONE, CHECK_OUT_ZONE, STORAGE_ZONE

@dataclass
class EvalConfig:
    robot:             RobotConfig
    policy_host:       str   = "localhost"
    policy_port:       int   = 5555
    action_horizon:    int   = 16
    lang_instruction:  str   = "Show the demo" 
    play_sounds:       bool  = False
    vla_timeout:       int   = 15    
    max_retries:       int   = 2     
    replan_every:      int   = 6
    ensemble_temp:     float = 0.1

ZONE_TASK_MAP = {
    "check_in":   {"source": CHECK_IN_ZONE,  "target": STORAGE_ZONE},
    "check_out":  {"source": STORAGE_ZONE,   "target": CHECK_OUT_ZONE},
    "check_back": {"source": CHECK_OUT_ZONE, "target": CHECK_IN_ZONE},
}

ALL_ZONES_DICT = {
    "Check In": CHECK_IN_ZONE,
    "Storage": STORAGE_ZONE,
    "Check Out": CHECK_OUT_ZONE
}

def detect_next_task(front_img, target_object="red cube") -> str:
    """Scans all three zones and returns the appropriate task type."""
    print(f"\n[AUTO-SCAN] Looking for '{target_object}' to determine next action...")
    for task_type, zones in ZONE_TASK_MAP.items():
        is_present, px = check_color_presence_front(front_img, target_object, zones["source"])
        if is_present:
            print(f"  -> Found in {task_type} starting zone! ({px} px)")
            return task_type
    return None


graceful_stop_requested = False

def request_graceful_stop(sig, frame):
    global graceful_stop_requested
    if graceful_stop_requested:
        print("\n🚨 [HARD STOP] Forced exit requested! Terminating immediately (No Home Sequence).")
        os._exit(1)
    print("\n⏳ [SOFT STOP] Stop signal received. Finishing current movement then returning home...")
    graceful_stop_requested = True

signal.signal(signal.SIGINT, request_graceful_stop)
signal.signal(signal.SIGTERM, request_graceful_stop)

def is_stop_requested():
    return graceful_stop_requested or os.path.exists("/tmp/stop_so100.flag")

@draccus.wrap()
def auto_eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    print("\n=======================================================")
    print("🤖 SO100 AUTO DEMO MODE INITIALIZING...")
    print("=======================================================\n")

    target_object = "red cube" 
    safety_monitor = SafetyMonitor(enabled=True)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    if os.path.exists("/tmp/stop_so100.flag"):
        os.remove("/tmp/stop_so100.flag")

    try:
        policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
        adapter       = So100Adapter(policy_client)
        runner = AsyncPolicyRunner(
            policy        = adapter,
            replan_every  = cfg.replan_every,
            ensemble_temp = cfg.ensemble_temp,
        )

        log_say("Hardware and policy initialized.", cfg.play_sounds)

        run_counter = 1

        while True:
            if is_stop_requested():
                print("\n🛑 [SHUTDOWN] Stop command detected. Exiting auto loop gracefully.")
                if os.path.exists("/tmp/stop_so100.flag"): os.remove("/tmp/stop_so100.flag")
                break 

            move_to_home(robot, safety_monitor=safety_monitor)
            time.sleep(1.0)   

            print(f"\n>>> [Run #{run_counter}] Taking baseline snapshots of workspace...")
            baseline_obs = robot.get_observation()
            baseline_img = baseline_obs["front"]
            baseline_wrist = baseline_obs["wrist"]

            # Only draw bounding box for the target object
            save_workspace_snapshot(baseline_img, f"snapshot_run_{run_counter}_before.jpg", ALL_ZONES_DICT, target_object, padding=40)

            task_type = detect_next_task(baseline_img, target_object)

            if not task_type:
                print("⚠️ [AUTO DEMO] Could not find the red cube in any known zone. Waiting 5 seconds before retrying...")
                time.sleep(5)
                continue

            source_zone = ZONE_TASK_MAP[task_type]["source"]
            target_zone = ZONE_TASK_MAP[task_type]["target"]

            print(f"\n[AUTO DEMO] Task Sequence Selected : {task_type.upper()}")
            print(f"[AUTO DEMO] Target Object          : {target_object}")

            grasp_detector = GraspDetector(baseline_wrist_img=baseline_wrist)
            
            fsm = EvaluationFSM(
                cfg=cfg, robot=robot, runner=runner,
                grasp_detector=grasp_detector, safety_monitor=safety_monitor,
                should_stop_cb=is_stop_requested,
                task_type=task_type, target_object=target_object,
                source_zone=source_zone, target_zone=target_zone,
                baseline_img=baseline_img
            )
            
            final_state = fsm.run()

            move_to_home(robot, safety_monitor=safety_monitor)
            time.sleep(1.0)
            final_obs = robot.get_observation()
            
            save_workspace_snapshot(final_obs["front"], f"snapshot_run_{run_counter}_after.jpg", ALL_ZONES_DICT, target_object, padding=40)

            if final_state == FSMState.FAILED:
                print("\n❌ [AUTO DEMO FAILED] FSM aborted. Rescanning...")
            elif final_state == FSMState.DONE:
                print("\n✅ [AUTO DEMO PASSED] Task complete. Ready for next sequence...")
                
            run_counter += 1
                
    except KeyboardInterrupt:
        pass 
    finally:
        print("\n>>> Initiating emergency cleanup...")
        if 'safety_monitor' in locals():
            safety_monitor.stop()

        if 'robot' in locals():
            try:
                print("  [Cleanup] Ensuring arm is safely parked at HOME...")
                move_to_home(robot, duration=1.0)
            except Exception as e:
                print(f"  [Cleanup] Failed to move home: {e}")

        if 'runner' in locals():
            try: runner.reset()
            except Exception: pass
            
        if 'robot' in locals():
            try: robot.disconnect()
            except Exception: pass
            
        print(">>> System shut down cleanly.\n")

if __name__ == "__main__":
    auto_eval()