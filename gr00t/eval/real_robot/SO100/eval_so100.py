"""
SO100 Hybrid Modular Evaluation Script (FSM & Monitored Transit)
=================================================================
"""

import logging
import time
import sys
import os
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
from utils.vision_utils   import GraspDetector, SafetyMonitor, save_workspace_snapshot
from utils.nlp_parser     import parse_instruction
from utils.fsm_controller import EvaluationFSM, FSMState
from utils.constants      import CHECK_IN_ZONE, CHECK_OUT_ZONE, STORAGE_ZONE, DIR_CAMERA, DIR_CAMERA_FRONT, ALL_ZONES_DICT
from utils.system_utils   import setup_signal_handlers, is_stop_requested, set_in_use, clear_in_use, clear_stop_flag

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

    setup_signal_handlers()
    clear_stop_flag()
    set_in_use()

    task_type, target_object, source_zone, target_zone = parse_instruction(cfg.lang_instruction)
    safety_monitor = SafetyMonitor(enabled=True)

    print(f"\n[PARSER] Instruction   : {cfg.lang_instruction}")
    print(f"[PARSER] Task type     : {task_type}")
    print(f"[PARSER] Target object : {target_object}")
    print(f"[PARSER] Source zone   : {source_zone}")
    print(f"[PARSER] Target zone   : {target_zone}\n")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    try:
        policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
        adapter       = So100Adapter(policy_client)
        runner = AsyncPolicyRunner(policy=adapter, replan_every=cfg.replan_every, ensemble_temp=cfg.ensemble_temp)

        move_to_home(robot, safety_monitor=safety_monitor)
        time.sleep(0.5)   

        print(">>> Taking baseline snapshots of workspace...")
        baseline_obs = robot.get_observation()
        baseline_img = baseline_obs["front"]

        save_workspace_snapshot(
            baseline_img, 
            "snapshot_eval_front_before.jpg", 
            ALL_ZONES_DICT, 
            target_object, 
            output_dir=DIR_CAMERA,
            padding=40
        )

        grasp_detector = GraspDetector()
        
        fsm = EvaluationFSM(
            cfg=cfg, robot=robot, runner=runner, grasp_detector=grasp_detector,
            safety_monitor=safety_monitor, should_stop_cb=is_stop_requested,
            task_type=task_type, target_object=target_object, 
            source_zone=source_zone, target_zone=target_zone, 
            baseline_img=baseline_img,
            run_id="eval"
        )
        
        final_state = fsm.run()

        move_to_home(robot, safety_monitor=safety_monitor)
        time.sleep(1.0)
        final_obs = robot.get_observation()
        
        save_workspace_snapshot(
            final_obs["front"], 
            "snapshot_eval_front_after.jpg", 
            ALL_ZONES_DICT, 
            target_object, 
            output_dir=DIR_CAMERA,
            padding=40
        )
        save_workspace_snapshot(
            final_obs["front"], 
            "eval_front.jpg", 
            ALL_ZONES_DICT, 
            target_object, 
            output_dir=DIR_CAMERA_FRONT,
            padding=40
        )

        if final_state == FSMState.FAILED:
            print("\n❌ [EVALUATION FAILED] The robot was unable to complete the task.")
        elif final_state == FSMState.DONE:
            print("\n✅ [EVALUATION COMPLETE] Finished successfully.")

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

        clear_in_use()
        print(">>> System shut down cleanly.\n")

if __name__ == "__main__":
    eval()