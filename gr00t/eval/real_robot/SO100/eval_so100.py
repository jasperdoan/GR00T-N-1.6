"""
SO100 Hybrid Modular Evaluation Script
=======================================

Architecture
------------
  Phase 1 — VLA (GR00T):      Approach and grasp the target cube.
  Phase 2 — Scripted Motion:  Transport, place, and return to Home.

The language instruction passed to GR00T contains only the target color
(e.g. "red"), not the full instruction. Task routing (check-in vs check-out)
is handled here in the orchestration layer.

Usage
-----
  python eval_so100.py --robot <cfg> --lang_instruction "Check in red cube"
"""

# =============================================================================
# Imports
# =============================================================================

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Optional

import draccus

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots import koch_follower       # noqa: F401
from lerobot.robots import so_follower as so100_follower  # noqa: F401
from lerobot.robots import so_follower as so101_follower  # noqa: F401
from lerobot.utils.utils import init_logging, log_say

from adapter      import So100Adapter
from constants    import CHECK_OUT_ZONE, STORAGE_ZONE
from motion       import move_to_home, scripted_transport, move_to_lift
from vision_utils import GraspDetector, check_task_success

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    robot:             RobotConfig
    policy_host:       str  = "localhost"
    policy_port:       int  = 5555
    action_horizon:    int  = 16
    lang_instruction:  str  = "Check in red cube"
    play_sounds:       bool = False
    vla_timeout:       int  = 20    # Max seconds for the VLA grasp phase


# =============================================================================
# Instruction Parser
# =============================================================================

def parse_instruction(instruction: str):
    """
    Parse the free-text instruction into (task_type, color, target_zone).
    """
    lower = instruction.lower()

    if "out" in lower:
        task_type   = "check_out"
        target_zone = CHECK_OUT_ZONE
    else:
        task_type   = "check_in"
        target_zone = STORAGE_ZONE

    if "blue" in lower:
        color = "blue"
    elif "yellow" in lower:
        color = "yellow"
    else:
        color = "red"

    return task_type, color, target_zone


# =============================================================================
# Main Evaluation Loop
# =============================================================================

@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # ── Parse instruction ────────────────────────────────────────────────────
    task_type, color, target_zone = parse_instruction(cfg.lang_instruction)

    print(f"\n[PARSER] Instruction : {cfg.lang_instruction}")
    print(f"[PARSER] Task type   : {task_type}")
    print(f"[PARSER] Target color: {color}")
    print(f"[PARSER] Target zone : {target_zone}\n")

    # ── Initialise hardware and policy ───────────────────────────────────────
    robot         = make_robot_from_config(cfg.robot)
    robot.connect()
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy        = So100Adapter(policy_client)

    log_say("Hardware and policy initialized.", cfg.play_sounds)

    # ── Pre-flight: home + baseline snapshot ─────────────────────────────────
    move_to_home(robot)
    time.sleep(0.1)   # let camera auto-exposure settle
    move_to_lift(robot)

    print(">>> Taking baseline snapshot of workspace …")
    baseline_img = robot.get_observation()["front"]

    # ── Phase 1: VLA — approach and grasp ────────────────────────────────────
    print("\n─── Phase 1: VLA Grasp ─────────────────────────────────────────")
    log_say(f"Searching for {color} cube.", cfg.play_sounds)
    print(">>> Monitoring wrist camera (VLA_GRASP_MIN_TIME gate initiated) ...")

    grasp_obs: Optional[dict] = None
    start_time = time.time()
    
    # Initialize our sequential confirmation detector
    grasp_detector = GraspDetector()

    while True:
        # Failsafe timeout
        if time.time() - start_time > cfg.vla_timeout:
            print("\n[TIMEOUT] VLA grasp phase exceeded limit. Aborting.")
            move_to_home(robot)
            return

        obs = robot.get_observation()
        obs["lang"] = color   # VLA sees the color string only

        # Pass frame into the detector state machine
        if grasp_detector.update(obs, color):
            print(f"\n✅ [GRASP CONFIRMED] {color.upper()} cube secured!")
            grasp_obs = obs
            break

        # Run one VLA inference step
        actions = policy.get_action_chunk(obs)

        for action_dict in actions[: cfg.action_horizon]:
            tic = time.time()
            robot.send_action(action_dict)
            elapsed = time.time() - tic
            if elapsed < 1.0 / 30:
                time.sleep(1.0 / 30 - elapsed)

    # ── Phase 2: Scripted transport ───────────────────────────────────────────
    print("\n─── Phase 2: Scripted Transport ────────────────────────────────")
    log_say("Grasp confirmed. Executing transport.", cfg.play_sounds)

    scripted_transport(robot, task_type, grasp_obs)

    # ── Post-task: verify placement with vision ───────────────────────────────
    time.sleep(0.3)   # let the scene settle before checking
    final_obs = robot.get_observation()
    success   = check_task_success(
        final_obs["front"], baseline_img, target_zone, color
    )

    if success:
        print(f"\n🎉 [SUCCESS] {color.upper()} cube confirmed in target zone.")
        log_say("Task complete.", cfg.play_sounds)
    else:
        print(f"\n⚠️  [WARN] Cube not detected in target zone after placement.")
        log_say("Placement could not be confirmed.", cfg.play_sounds)

    print("\n>>> Evaluation complete. System shutting down cleanly.")


if __name__ == "__main__":
    eval()