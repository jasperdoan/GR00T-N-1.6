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

VLA smoothness
--------------
  Inference runs in a background thread (AsyncPolicyRunner) so the control
  loop never stalls waiting for the GPU.  Overlapping action chunks are blended
  via TemporalEnsemble so the hand-off between chunks is invisible to the robot.
  Together these eliminate the burst-pause-burst pattern of naive chunk execution.

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

from adapter        import So100Adapter
from constants      import CHECK_OUT_ZONE, STORAGE_ZONE, KNOWN_OBJECTS
from motion         import move_to_home, move_to_ready, scripted_transport
from policy_runner  import AsyncPolicyRunner
from vision_utils   import GraspDetector, check_task_success

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvalConfig:
    robot:             RobotConfig
    policy_host:       str   = "localhost"
    policy_port:       int   = 5555
    action_horizon:    int   = 16
    lang_instruction:  str   = "Check in red cube"
    play_sounds:       bool  = False
    vla_timeout:       int   = 15    # Max seconds for the VLA grasp phase

    # --- Async runner settings -----------------------------------------------
    # How many control steps to execute before triggering re-inference.
    # Lower  → more frequent inference, smoother adaptation, higher GPU load.
    # Higher → less GPU load, slightly longer coasting on a stale chunk.
    # Recommended range: 4–8 for a 16-step horizon at 30 Hz.
    replan_every:      int   = 6

    # Temperature for TemporalEnsemble.
    # 0.0 → uniform average of all live chunks (maximum smoothing).
    # 0.1 → moderate bias to the newest chunk (recommended).
    # 1.0 → nearly always takes the newest chunk (minimal blending).
    ensemble_temp:     float = 0.1


# =============================================================================
# Instruction Parser
# =============================================================================

def parse_instruction(instruction: str):
    """
    Parse the free-text instruction into (task_type, target_object, target_zone).
    """
    lower = instruction.lower()

    if "out" in lower:
        task_type   = "check_out"
        target_zone = CHECK_OUT_ZONE
    else:
        task_type   = "check_in"
        target_zone = STORAGE_ZONE

    target_object = None
    for obj in KNOWN_OBJECTS:
        if obj in lower:
            target_object = obj
            break
            
    if target_object is None:
        raise ValueError(
            f"Could not find a known object in the instruction: '{instruction}'. "
            f"Known objects are: {KNOWN_OBJECTS}"
        )

    return task_type, target_object, target_zone

# =============================================================================
# Main Evaluation Loop
# =============================================================================

@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # ── Parse instruction ────────────────────────────────────────────────────
    task_type, target_object, target_zone = parse_instruction(cfg.lang_instruction)

    print(f"\n[PARSER] Instruction   : {cfg.lang_instruction}")
    print(f"[PARSER] Task type     : {task_type}")
    print(f"[PARSER] Target object : {target_object}")
    print(f"[PARSER] Target zone   : {target_zone}\n")

    # ── Initialise hardware and policy ───────────────────────────────────────
    robot         = make_robot_from_config(cfg.robot)
    robot.connect()
    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    adapter       = So100Adapter(policy_client)

    # Build the async runner — inference will overlap with robot execution
    runner = AsyncPolicyRunner(
        policy        = adapter,
        replan_every  = cfg.replan_every,
        ensemble_temp = cfg.ensemble_temp,
    )

    log_say("Hardware and policy initialized.", cfg.play_sounds)

    # ── Pre-flight: home + baseline snapshot ─────────────────────────────────
    move_to_home(robot)
    time.sleep(0.1)   # let camera auto-exposure settle
    move_to_ready(robot, task_type)

    print(">>> Taking baseline snapshots of workspace and empty gripper …")
    baseline_obs = robot.get_observation()
    baseline_img = baseline_obs["front"]
    baseline_wrist = baseline_obs["wrist"]

    # ── Phase 1: VLA — approach and grasp ────────────────────────────────────
    print("\n─── Phase 1: VLA Grasp ─────────────────────────────────────────")
    log_say(f"Searching for {target_object}.", cfg.play_sounds)
    print(
        f">>> Running async inference "
        f"(replan_every={cfg.replan_every}, ensemble_temp={cfg.ensemble_temp}) …"
    )

    grasp_obs:      Optional[dict] = None
    start_time      = time.time()
    # Pass the empty wrist image to the detector!
    grasp_detector  = GraspDetector(baseline_wrist_img=baseline_wrist)

    while True:
        # ── Failsafe timeout ────────────────────────────────────────────────
        if time.time() - start_time > cfg.vla_timeout:
            print("\n[TIMEOUT] VLA grasp phase exceeded limit. Aborting.")
            move_to_home(robot)
            runner.reset()
            return

        tic = time.time()

        obs = robot.get_observation()
        obs["lang"] = target_object  # <-- Passing the exact object string to VLA

        # ── Grasp detection ─────────────────────────────────────────────────
        if grasp_detector.update(obs):  # <-- Removed color dependency
            print(f"\n✅ [GRASP CONFIRMED] {target_object.upper()} secured!")
            grasp_obs = obs
            break

        # ── Async VLA step ──────────────────────────────────────────────────
        # runner.step() returns the temporally-ensembled action for this tick.
        # On the very first call it blocks briefly for the initial inference;
        # all subsequent calls return immediately from the ensemble buffer.
        action = runner.step(obs)
        if action is not None:
            robot.send_action(action)

        # ── Hold 30 Hz regardless of inference time ─────────────────────────
        elapsed = time.time() - tic
        sleep_t = 1.0 / 30 - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    runner.reset()   # clear stale chunk history before scripted phase

    # ── Phase 2: Scripted transport ───────────────────────────────────────────
    print("\n─── Phase 2: Scripted Transport ────────────────────────────────")
    log_say("Grasp confirmed. Executing transport.", cfg.play_sounds)

    scripted_transport(robot, task_type, grasp_obs)

    # ── Post-task: verify placement with vision ───────────────────────────────
    time.sleep(0.3)   # let the scene settle before checking
    final_obs = robot.get_observation()
    success   = check_task_success(
        final_obs["front"], baseline_img, target_zone
    )

    if success:
        print(f"\n🎉 [SUCCESS] {target_object.upper()} confirmed in target zone.")
        log_say("Task complete.", cfg.play_sounds)
    else:
        print(f"\n⚠️  [WARN] Object not detected in target zone after placement.")
        log_say("Placement could not be confirmed.", cfg.play_sounds)

    print("\n>>> Evaluation complete. System shutting down cleanly.")


if __name__ == "__main__":
    eval()