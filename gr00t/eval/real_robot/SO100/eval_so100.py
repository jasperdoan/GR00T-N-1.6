"""
SO100 Hybrid Modular Evaluation Script (FSM & Monitored Transit)
=================================================================
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots import koch_follower       # noqa: F401
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.robots import so101_follower  # noqa: F401
from lerobot.utils.utils import init_logging, log_say

from utils.adapter        import So100Adapter
from utils.motion         import move_to_home
from utils.policy_runner  import AsyncPolicyRunner
from utils.vision_utils   import GraspDetector
from utils.nlp_parser     import parse_instruction
from utils.fsm_controller import EvaluationFSM, FSMState

@dataclass
class EvalConfig:
    robot:             RobotConfig
    policy_host:       str   = "localhost"
    policy_port:       int   = 5555
    action_horizon:    int   = 16
    lang_instruction:  str   = "Check in red cube"
    play_sounds:       bool  = False
    vla_timeout:       int   = 10    
    max_retries:       int   = 2     
    replan_every:      int   = 6
    ensemble_temp:     float = 0.1

@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # ── Parse instruction ────────────────────────────────────────────────────
    task_type, target_object, source_zone, target_zone = parse_instruction(cfg.lang_instruction)

    print(f"\n[PARSER] Instruction   : {cfg.lang_instruction}")
    print(f"[PARSER] Task type     : {task_type}")
    print(f"[PARSER] Target object : {target_object}")
    print(f"[PARSER] Source zone   : {source_zone}")
    print(f"[PARSER] Target zone   : {target_zone}\n")

    # ── Initialise hardware and policy ───────────────────────────────────────
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

        # ── Pre-flight: home + baseline snapshot ─────────────────────────────────
        move_to_home(robot)
        time.sleep(0.5)   

        print(">>> Taking baseline snapshots of workspace and empty gripper …")
        baseline_obs = robot.get_observation()
        baseline_img = baseline_obs["front"]
        baseline_wrist = baseline_obs["wrist"]

        grasp_detector = GraspDetector(baseline_wrist_img=baseline_wrist)
        
        # ── Initialize and Run FSM ───────────────────────────────────────────────
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
        
        final_state = fsm.run()

        # ── Final Teardown ───────────────────────────────────────────────────────
        if final_state == FSMState.FAILED:
            print("\n❌ [EVALUATION FAILED] The robot was unable to complete the task.")
            move_to_home(robot)
        elif final_state == FSMState.DONE:
            print("\n✅ [EVALUATION COMPLETE] Finished successfully.")

    except KeyboardInterrupt:
        print("\n⚠️ [INTERRUPTED] User stopped the script manually.")
    except Exception as e:
        print(f"\n🚨 [CRITICAL ERROR] Script crashed: {e}")
        raise  # Re-raise the exception so you can still see the traceback
    finally:
        # ── Guaranteed Cleanup Block ─────────────────────────────────────────────
        print("\n>>> Initiating emergency cleanup...")
        
        # 1. Stop the background inference thread safely
        if 'runner' in locals():
            try:
                runner.reset()
                print("  [Cleanup] Policy runner threads joined.")
            except Exception as e:
                print(f"  [Cleanup] Failed to reset runner: {e}")

        # 2. Safely disconnect the robot (THIS releases the cameras!)
        if 'robot' in locals():
            try:
                robot.disconnect()
                print("  [Cleanup] Robot disconnected. Cameras released successfully.")
            except Exception as e:
                print(f"  [Cleanup] Failed to disconnect robot: {e}")

        print(">>> System shut down cleanly.\n")

if __name__ == "__main__":
    eval()