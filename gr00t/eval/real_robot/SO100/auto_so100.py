"""
SO100 Auto Demo (Continuous Loop)
=================================================================
Continuously runs the robot through check_in, check_out, and check_back.
Uses stateless auto-discovery: It looks at the workspace, figures out 
where the red cube is, and executes the corresponding next logical step.
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.utils.utils import init_logging, log_say

from utils.adapter        import So100Adapter
from utils.motion         import move_to_home
from utils.policy_runner  import AsyncPolicyRunner
from utils.vision_utils   import GraspDetector, check_color_presence_front
from utils.fsm_controller import EvaluationFSM, FSMState
from utils.constants      import CHECK_IN_ZONE, CHECK_OUT_ZONE, STORAGE_ZONE

@dataclass
class EvalConfig:
    robot:             RobotConfig
    policy_host:       str   = "localhost"
    policy_port:       int   = 5555
    action_horizon:    int   = 16
    lang_instruction:  str   = "Show the demo" # Ignored in auto mode
    play_sounds:       bool  = False
    vla_timeout:       int   = 15    
    max_retries:       int   = 2     
    replan_every:      int   = 6
    ensemble_temp:     float = 0.1

# Map the starting zones to the task that needs to be executed
ZONE_TASK_MAP = {
    "check_in":   {"source": CHECK_IN_ZONE,  "target": STORAGE_ZONE},
    "check_out":  {"source": STORAGE_ZONE,   "target": CHECK_OUT_ZONE},
    "check_back": {"source": CHECK_OUT_ZONE, "target": CHECK_IN_ZONE},
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

@draccus.wrap()
def auto_eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    print("\n=======================================================")
    print("🤖 SO100 AUTO DEMO MODE INITIALIZING...")
    print("=======================================================\n")

    target_object = "red cube" # Hardcoded for now per your spec

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    try:
        policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
        adapter       = So100Adapter(policy_client)
        runner = AsyncPolicyRunner(
            policy        = adapter,
            replan_every  = cfg.replan_every,
            ensemble_temp = cfg.ensemble_temp,
        )

        log_say("Hardware and policy initialized.", cfg.play_sounds)

        # ── CONTINUOUS DEMO LOOP ─────────────────────────────────────────────
        while True:
            # 1. Reset to home and let the cameras settle
            move_to_home(robot)
            time.sleep(1.0)   

            print("\n>>> Taking baseline snapshots of workspace...")
            baseline_obs = robot.get_observation()
            baseline_img = baseline_obs["front"]
            baseline_wrist = baseline_obs["wrist"]

            # 2. Auto-Discovery: Find the cube
            task_type = detect_next_task(baseline_img, target_object)

            if not task_type:
                print("⚠️ [AUTO DEMO] Could not find the red cube in any known zone. Waiting 5 seconds before retrying...")
                time.sleep(5)
                continue

            # 3. Setup Task Variables
            source_zone = ZONE_TASK_MAP[task_type]["source"]
            target_zone = ZONE_TASK_MAP[task_type]["target"]

            print(f"\n[AUTO DEMO] Task Sequence Selected : {task_type.upper()}")
            print(f"[AUTO DEMO] Target Object          : {target_object}")

            # 4. Initialize Vision Trackers for this specific run
            grasp_detector = GraspDetector(baseline_wrist_img=baseline_wrist)
            
            fsm = EvaluationFSM(
                cfg=cfg,
                robot=robot,
                runner=runner,
                grasp_detector=grasp_detector,
                task_type=task_type,
                target_object=target_object,
                source_zone=source_zone,
                target_zone=target_zone,
                baseline_img=baseline_img
            )
            
            # 5. Run the FSM!
            final_state = fsm.run()

            # 6. Evaluate outcome
            if final_state == FSMState.FAILED:
                print("\n❌ [AUTO DEMO FAILED] FSM aborted. Returning home to rescan and recover...")
            elif final_state == FSMState.DONE:
                print("\n✅ [AUTO DEMO PASSED] Task complete. Returning home to trigger the next sequence...")
                time.sleep(1.0) # Let the object settle before the next baseline snapshot
                
    except KeyboardInterrupt:
        print("\n⚠️ [INTERRUPTED] User stopped the Auto Demo manually.")
    except Exception as e:
        print(f"\n🚨 [CRITICAL ERROR] Script crashed: {e}")
        raise
    finally:
        print("\n>>> Initiating emergency cleanup...")
        if 'runner' in locals():
            try:
                runner.reset()
                print("  [Cleanup] Policy runner threads joined.")
            except Exception as e:
                pass
        if 'robot' in locals():
            try:
                robot.disconnect()
                print("  [Cleanup] Robot disconnected. Cameras released successfully.")
            except Exception as e:
                pass
        print(">>> System shut down cleanly.\n")

if __name__ == "__main__":
    auto_eval()