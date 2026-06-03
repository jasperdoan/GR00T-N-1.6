"""
SO100 Autonomous Continuous Loop Script
=======================================
Monitors the check-in zone for new items. 
- Waits for physical motion (human hand) and a 5-second settle period.
- Dynamically clears Storage -> Checkout (shakes head if empty).
- Dynamically clears Checkin -> Storage (shakes head if empty).
- Aborts safely on FSM failures to prevent infinite loops.
"""

import logging
import time
import sys
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Tuple, List

import cv2
import numpy as np
import draccus

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.utils import init_logging

from utils.adapter import So100Adapter
from utils.motion import move_to_home, execute_failure_shake
from utils.policy_runner import AsyncPolicyRunner
from utils.vision_utils import GraspDetector, detect_all_objects_in_zone
from utils.fsm_controller import EvaluationFSM, FSMState
from utils.constants import CHECK_IN_ZONE, CHECK_OUT_ZONE, STORAGE_ZONE


@dataclass
class AutoConfig:
    robot:             RobotConfig
    policy_host:       str   = "localhost"
    policy_port:       int   = 5555
    action_horizon:    int   = 16
    play_sounds:       bool  = False
    vla_timeout:       int   = 10    
    max_retries:       int   = 2     
    replan_every:      int   = 6
    ensemble_temp:     float = 0.1
    settle_time:       float = 5.0   # Seconds to wait after human stops moving


def wait_for_human_intervention(robot, zone: Tuple[int, int, int, int], settle_time: float):
    """
    Blocks until ACTUAL MOTION is detected, followed by complete stillness.
    This prevents infinite loops on ungraspable objects by forcing human help.
    """
    print("\n[Auto] ==========================================================")
    print("[Auto] 👁️  Monitoring zone for human intervention...")
    
    prev_gray = None
    motion_state = "WAITING_FOR_MOTION"
    settle_timer = None
    
    while True:
        obs = robot.get_observation()
        img = obs["front"]
        
        x, y, w, h = zone
        crop = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_gray is None:
            prev_gray = gray
            time.sleep(0.1)
            continue
            
        diff = cv2.absdiff(gray, prev_gray)
        prev_gray = gray
        
        # Count pixels that changed significantly
        motion_pixels = np.sum(diff > 15)
        is_moving = motion_pixels > 500  
        
        if motion_state == "WAITING_FOR_MOTION":
            if is_moving:
                sys.stdout.write(f"\r[Auto] ✋ Motion detected! Human is interacting...           ")
                sys.stdout.flush()
                motion_state = "WAITING_FOR_SETTLE"
                settle_timer = time.time()
            else:
                sys.stdout.write(f"\r[Auto] 💤 Waiting for items to be placed...                  ")
                sys.stdout.flush()
                
        elif motion_state == "WAITING_FOR_SETTLE":
            if is_moving:
                # Reset the timer as long as human is moving
                sys.stdout.write(f"\r[Auto] ✋ Human still moving...                               ")
                sys.stdout.flush()
                settle_timer = time.time()
            else:
                elapsed = time.time() - settle_timer
                if elapsed >= settle_time:
                    # Verify there are actually items left behind before triggering
                    items = detect_all_objects_in_zone(img, zone)
                    if len(items) > 0:
                        print(f"\n[Auto] ✅ Zone settled. Target acquired: {items}")
                        return
                    else:
                        print(f"\n[Auto] ❌ Zone settled but empty (False alarm). Resetting...")
                        motion_state = "WAITING_FOR_MOTION"
                else:
                    sys.stdout.write(f"\r[Auto] ⏳ Settling... {settle_time - elapsed:.1f}s remaining                 ")
                    sys.stdout.flush()
                    
        time.sleep(0.1)


def process_zone(robot, runner, cfg, task_type: str, source_zone: Tuple, target_zone: Tuple):
    """
    Dynamically clears a zone. Scans, picks one item, re-scans, picks next item.
    Aborts gracefully if it empties, or if FSM fails.
    """
    zone_name = "Storage" if task_type == "check_out" else "Check-in"
    
    # 1. Initial check (so we can shake head if empty)
    obs = robot.get_observation()
    initial_items = detect_all_objects_in_zone(obs["front"], source_zone)
    
    if not initial_items:
        print(f"\n[Auto] 🤷 {zone_name} is empty. Skipping.")
        execute_failure_shake(robot)
        return

    # 2. Dynamic processing loop
    while True:
        move_to_home(robot)
        time.sleep(0.5)
        
        obs = robot.get_observation()
        current_items = detect_all_objects_in_zone(obs["front"], source_zone)
        
        if not current_items:
            print(f"\n[Auto] 🧹 {zone_name} successfully cleared!")
            break
            
        target_item = current_items[0]  # Just grab the first one we see
        print(f"\n[Auto] Processing {task_type} for: {target_item}")
        
        # Take baseline right before execution for placement verification
        baseline_img = obs["front"]
        grasp_detector = GraspDetector(baseline_wrist_img=obs["wrist"])
        
        fsm = EvaluationFSM(
            cfg=cfg, robot=robot, runner=runner, grasp_detector=grasp_detector,
            task_type=task_type, target_object=target_item,
            source_zone=source_zone, target_zone=target_zone, baseline_img=baseline_img
        )
        
        final_state = fsm.run()
        
        if final_state == FSMState.FAILED:
            print(f"\n[Auto] 🚨 FSM Failed on '{target_item}'. Aborting current zone loop to wait for human help.")
            break


@draccus.wrap()
def auto_loop(cfg: AutoConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # ── Initialize Hardware ──
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    adapter = So100Adapter(policy_client)

    runner = AsyncPolicyRunner(
        policy=adapter,
        replan_every=cfg.replan_every,
        ensemble_temp=cfg.ensemble_temp,
    )
    
    print("\n>>> Hardware connected. Entering Autonomous Loop.")

    while True:
        # 1. Park robot safely and watch the check-in area for a human
        move_to_home(robot)
        wait_for_human_intervention(robot, CHECK_IN_ZONE, cfg.settle_time)

        # ---------------------------------------------------------
        # PHASE A: CLEAR STORAGE (Storage -> Checkout)
        # ---------------------------------------------------------
        print("\n[Auto] --- PHASE A: CLEARING STORAGE ---")
        process_zone(
            robot=robot, runner=runner, cfg=cfg, 
            task_type="check_out", 
            source_zone=STORAGE_ZONE, target_zone=CHECK_OUT_ZONE
        )

        # ---------------------------------------------------------
        # PHASE B: CHECK-IN NEW ITEMS (Checkin -> Storage)
        # ---------------------------------------------------------
        print("\n[Auto] --- PHASE B: CHECKING IN ---")
        process_zone(
            robot=robot, runner=runner, cfg=cfg, 
            task_type="check_in", 
            source_zone=CHECK_IN_ZONE, target_zone=STORAGE_ZONE
        )
                
        print("\n[Auto] 🔄 Cycle complete. Returning to watch state.")

if __name__ == "__main__":
    auto_loop()