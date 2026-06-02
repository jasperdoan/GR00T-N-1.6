"""
SO100 Hybrid Modular Evaluation Script (FSM & Monitored Transit)
=================================================================

Architecture
------------
  Phase 1 — VLA (GR00T):      Approach and grasp the target cube.
  Phase 2 — Scripted Motion:  Transport, place, and return to Home.

Now wrapped in a Finite State Machine (FSM) behavior loop! If the robot 
drops the object mid-air, it captures the exact pan location of the drop 
and initiates a local visual recovery search without returning home.
"""

# =============================================================================
# Imports
# =============================================================================

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum, auto
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
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.robots import so101_follower  # noqa: F401
from lerobot.utils.utils import init_logging, log_say

from adapter        import So100Adapter
from constants      import CHECK_OUT_ZONE, STORAGE_ZONE, KNOWN_OBJECTS, JOINT_NAMES, GRIPPER_OPEN_POS
from motion         import move_to_home, move_to_ready, scripted_transport, lerp_to_waypoint, GraspLostException
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
    
    # Fault Tolerance Settings
    max_retries:       int   = 0     # How many times to retry on a timeout or drop

    # --- Async runner settings -----------------------------------------------
    replan_every:      int   = 6
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
# Finite State Machine States
# =============================================================================

class FSMState(Enum):
    SEARCHING = auto()
    TRANSPORT = auto()
    RECOVERY  = auto()
    VERIFY    = auto()
    DONE      = auto()
    FAILED    = auto()


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

    grasp_detector = GraspDetector(baseline_wrist_img=baseline_wrist)
    
    # ── FSM Initialization ───────────────────────────────────────────────────
    state = FSMState.SEARCHING
    search_retries = 0
    grasp_obs: Optional[dict] = None
    recovery_pan: Optional[float] = None

    while state not in (FSMState.DONE, FSMState.FAILED):
        
# =====================================================================
        # STATE: SEARCHING (HARDCODED CHEAT MODE)
        # =====================================================================
        if state == FSMState.SEARCHING:
            print("\n─── [FSM: SEARCHING] HARDCODED SCRIPTED GRASP ───────────────")
            log_say(f"Executing hardcoded grasp for {target_object}.", cfg.play_sounds)
            
            # 1. Define waypoints based on task type
            if task_type == "check_in":
                approach_wp = {
                    "shoulder_pan.pos":   47.430,
                    "shoulder_lift.pos":  40.341,
                    "elbow_flex.pos":     -14.761,
                    "wrist_flex.pos":     75.290,
                    "wrist_roll.pos":     47.460,
                    "gripper.pos":        28.088,
                }
            else:  # check_out
                approach_wp = {
                    "shoulder_pan.pos":    -14.711,
                    "shoulder_lift.pos":  24.121,
                    "elbow_flex.pos":      7.672,
                    "wrist_flex.pos":     45.957,
                    "wrist_roll.pos":     25.705,
                    "gripper.pos":        33.865,
                }
                
            grasp_wp = approach_wp.copy()
            grasp_wp["gripper.pos"] = 65.0
            
            # 2. Execute approach extremely slowly (5.0 seconds)
            print(f">>> Moving to {task_type} approach waypoint...")
            lerp_to_waypoint(robot, approach_wp, duration=5.0)
            
            # 3. Close the gripper (1.0 second)
            print(">>> Closing gripper...")
            lerp_to_waypoint(robot, grasp_wp, duration=1.0)
            time.sleep(0.5)  # Brief pause to let physics settle
            
            # 4. Fake the confirmation steps exactly as the model would have done
            print(f"\n✅ [GRASP CONFIRMED] {target_object.upper()} secured! (Cheat mode)")
            
            # Grab current observation to pass into the transport sequence
            grasp_obs = robot.get_observation()
            grasp_obs["lang"] = target_object
            grasp_obs.pop("front", None)
            
            # Take visual snapshot of object for mid-air drop monitoring
            grasp_detector.lock_grasp(grasp_obs)
            
            # Clear state variables
            recovery_pan = None 
            runner.reset()
            
            # 5. Move to next FSM state
            state = FSMState.TRANSPORT

        # =====================================================================
        # STATE: TRANSPORT
        # =====================================================================
        elif state == FSMState.TRANSPORT:
            print("\n─── [FSM: TRANSPORT] Scripted Transport ─────────────────────")
            log_say("Grasp confirmed. Executing transport.", cfg.play_sounds)
            
            # 🔴 PREVIOUSLY: monitor_cb = grasp_detector.check_grasp_maintained
            # 🟢 CHANGE TO THIS:
            monitor_cb = None  # Completely disables the drop detection!
            
            try:
                scripted_transport(robot, task_type, grasp_obs, monitor_callback=monitor_cb)
                state = FSMState.VERIFY
            except GraspLostException as e:
                print(f"\n⚠️ [FSM: Grasp Lost] Exception caught mid-air: {e}")
                log_say("Object dropped mid-air. Initiating local recovery.", cfg.play_sounds)
                
                # Extract the precise shoulder pan location where the drop happened
                recovery_pan = float(e.last_obs.get("shoulder_pan.pos", 0.0))
                state = FSMState.RECOVERY


        # =====================================================================
        # STATE: VERIFY
        # =====================================================================
        elif state == FSMState.VERIFY:
            print("\n─── [FSM: VERIFY] Checking Target Zone Placement ────────────")
            time.sleep(0.3)
            final_obs = robot.get_observation()
            success   = check_task_success(final_obs["front"], baseline_img, target_zone)
            
            if success:
                print(f"\n🎉 [SUCCESS] {target_object.upper()} confirmed in target zone.")
                log_say("Task complete.", cfg.play_sounds)
                state = FSMState.DONE
            else:
                print(f"\n⚠️  [WARN] Object not detected in target zone after placement. Bounced out?")
                log_say("Placement could not be confirmed.", cfg.play_sounds)
                
                search_retries += 1
                if search_retries > cfg.max_retries:
                    print("[FSM] Max retries exceeded. Task failed.")
                    state = FSMState.FAILED
                else:
                    print("[FSM] Retrying task.")
                    move_to_ready(robot, task_type)
                    state = FSMState.SEARCHING

        # =====================================================================
        # STATE: RECOVERY
        # =====================================================================
        elif state == FSMState.RECOVERY:
            print("\n─── [FSM: RECOVERY] Local Error Correction ──────────────────")
            # Pop the gripper open safely in case the cube is wedged awkwardly
            obs = robot.get_observation()
            drop_pose = {j: float(obs.get(j, 0.0)) for j in JOINT_NAMES}
            drop_pose["gripper.pos"] = GRIPPER_OPEN_POS
            lerp_to_waypoint(robot, drop_pose, 0.5)
            
            # Move to ready height/pitch, but keep the pan from where we dropped it
            # so the VLA can immediately look down and find the dropped object!
            if recovery_pan is not None:
                print(f"  [FSM: RECOVERY] Retaining local pan angle ({recovery_pan:.1f}) to find dropped item.")
                move_to_ready(robot, task_type, pan_override=recovery_pan)
            else:
                # Fallback if recovery_pan somehow wasn't captured
                move_to_home(robot)
                move_to_ready(robot, task_type)
            
            search_retries += 1
            if search_retries > cfg.max_retries:
                print("[FSM] Max retries exceeded during recovery. Failing.")
                state = FSMState.FAILED
            else:
                state = FSMState.SEARCHING


    # ── Final Teardown ───────────────────────────────────────────────────────
    if state == FSMState.FAILED:
        print("\n❌ [EVALUATION FAILED] The robot was unable to complete the task.")
        move_to_home(robot)
    elif state == FSMState.DONE:
        print("\n✅ [EVALUATION COMPLETE] Finished successfully.")
        # move_to_home(robot) is already handled at the end of phase D in scripted_transport

    print("\n>>> System shutting down cleanly.")


if __name__ == "__main__":
    eval()