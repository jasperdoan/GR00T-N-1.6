"""
SO100 Finite State Machine Controller
"""

import time
from enum import Enum, auto

from lerobot.utils.utils import log_say

from utils.constants import JOINT_NAMES, GRIPPER_OPEN_POS, DIR_CAMERA, DIR_CAMERA_WRIST
from utils.motion import (
    move_to_home, move_to_ready, scripted_transport, lerp_to_waypoint, execute_failure_shake, GraspLostException
)
from utils.vision_utils import check_task_success, check_color_presence_front, save_workspace_snapshot


class FSMState(Enum):
    PRE_CHECK = auto()
    SEARCHING = auto()
    TRANSPORT = auto()
    RECOVERY  = auto()
    VERIFY    = auto()
    DONE      = auto()
    FAILED    = auto()


class EvaluationFSM:
    def __init__(self, cfg, robot, runner, grasp_detector, task_type, target_object, source_zone, target_zone, baseline_img, run_id="eval", safety_monitor=None, should_stop_cb=None):
        self.cfg = cfg
        self.robot = robot
        self.runner = runner
        self.grasp_detector = grasp_detector
        self.safety_monitor = safety_monitor
        self.should_stop_cb = should_stop_cb
        self.run_id = run_id
        
        self.task_type = task_type
        self.target_object = target_object
        self.source_zone = source_zone
        self.target_zone = target_zone
        self.baseline_img = baseline_img
        
        self.state = FSMState.PRE_CHECK
        self.search_retries = 0
        self.grasp_obs = None
        self.recovery_pan = None
        self.start_time = 0.0

    def check_abort(self) -> bool:
        if self.should_stop_cb and self.should_stop_cb():
            print("\n🛑 [FSM] Stop command detected! Aborting sequence immediately...")
            self.state = FSMState.FAILED
            return True
        return False

    def run(self):
        while self.state not in (FSMState.DONE, FSMState.FAILED):
            if self.check_abort():
                break
                
            if self.state == FSMState.PRE_CHECK: self._handle_pre_check()
            elif self.state == FSMState.SEARCHING: self._handle_searching()
            elif self.state == FSMState.TRANSPORT: self._handle_transport()
            elif self.state == FSMState.VERIFY: self._handle_verify()
            elif self.state == FSMState.RECOVERY: self._handle_recovery()
            
        return self.state

    def _handle_pre_check(self):
        print("\n─── [FSM: PRE_CHECK] Verifying Object Presence ──────────────────")
        time.sleep(0.5) 
        obs = self.robot.get_observation()
        is_present, px_count = check_color_presence_front(obs["front"], self.target_object, zone=self.source_zone)
        
        if not is_present:
            print(f"\n❌ [PRE_CHECK FAILED] Expected color for '{self.target_object}' not found in the starting zone.")
            execute_failure_shake(self.robot)
            self.state = FSMState.FAILED
        else:
            print(f"\n✅ [PRE_CHECK PASSED] Object color detected ({px_count} px). Proceeding to task.")
            move_to_ready(self.robot, self.task_type, safety_monitor=self.safety_monitor)
            
            # Take wrist snapshot safely at the READY position
            obs = self.robot.get_observation()
            save_workspace_snapshot(
                obs["wrist"], 
                f"snapshot_{self.run_id}_wrist.jpg", 
                None, 
                self.target_object, 
                output_dir=DIR_CAMERA
            )
            save_workspace_snapshot(
                obs["wrist"], 
                f"{self.run_id}_wrist.jpg", 
                None, 
                self.target_object, 
                output_dir=DIR_CAMERA_WRIST
            )
            
            self.state = FSMState.SEARCHING

    def _handle_searching(self):
        print("\n─── [FSM: SEARCHING] VLA Grasp Phase ────────────────────────")
        log_say(f"Searching for {self.target_object}.", self.cfg.play_sounds)
        
        self.start_time = time.time()
        self.grasp_detector.start_time = time.time()
        
        while True:
            if self.check_abort():
                break
                
            if time.time() - self.start_time > self.cfg.vla_timeout:
                print("\n[TIMEOUT] VLA grasp phase exceeded limit.")
                self.search_retries += 1
                if self.search_retries > self.cfg.max_retries:
                    self.state = FSMState.FAILED
                else:
                    move_to_home(self.robot, safety_monitor=self.safety_monitor)
                    move_to_ready(self.robot, self.task_type, safety_monitor=self.safety_monitor)
                    self.start_time = time.time()
                    self.grasp_detector.start_time = time.time()
                    self.runner.reset()
                break
            
            tic = time.time()
            obs = self.robot.get_observation()

            # Pass frame to safety monitor thread (Instantly O(1))
            if self.safety_monitor:
                self.safety_monitor.update_frame(obs["front"])

                # Check if the thread saw a hand
                if self.safety_monitor.is_hand_present():
                    print("\n🛑 [SAFETY] Hand detected during SEARCHING! Pausing inference...")
                    hold_pose = {j: float(obs.get(j, 0.0)) for j in JOINT_NAMES}
                    while True:
                        self.robot.send_action(hold_pose)
                        time.sleep(1/30)
                        obs = self.robot.get_observation()
                        self.safety_monitor.update_frame(obs["front"])
                        if not self.safety_monitor.is_hand_present():
                            print("▶️ [SAFETY] Hand removed. Clearing stale chunks and resuming...")
                            break
                    self.runner.reset()
                    self.start_time = time.time()
                    self.grasp_detector.start_time = time.time()
                    continue
            
            obs["lang"] = self.target_object
            obs_for_model = obs.copy()
            obs_for_model.pop("front", None)

            grasp_status = self.grasp_detector.update(obs, self.target_object)
            
            if grasp_status == "SUCCESS":
                print(f"\n✅ [GRASP CONFIRMED] {self.target_object.upper()} secured!")
                self.grasp_obs = obs
                self.grasp_detector.lock_grasp(obs)
                self.recovery_pan = None 
                self.state = FSMState.TRANSPORT
                break
                
            elif grasp_status == "WRONG_OBJECT":
                print("\n⚠️ [FSM: Early Abort] VLA grabbed the wrong object or empty space!")
                drop_pose = {j: float(obs.get(j, 0.0)) for j in JOINT_NAMES}
                drop_pose["gripper.pos"] = GRIPPER_OPEN_POS
                lerp_to_waypoint(self.robot, drop_pose, 0.5, safety_monitor=self.safety_monitor)
                self.search_retries += 1
                
                if self.search_retries > self.cfg.max_retries:
                    self.state = FSMState.FAILED
                    break
                else:
                    move_to_home(self.robot, safety_monitor=self.safety_monitor)
                    move_to_ready(self.robot, self.task_type, safety_monitor=self.safety_monitor)
                    self.start_time = time.time()
                    self.grasp_detector.start_time = time.time()
                    self.runner.reset()
                    continue
            
            action = self.runner.step(obs_for_model)
            if action is not None:
                self.robot.send_action(action)
            
            elapsed = time.time() - tic
            sleep_t = 1.0 / 30 - elapsed
            if sleep_t > 0: time.sleep(sleep_t)
        
        self.runner.reset()

    def _handle_transport(self):
        print("\n─── [FSM: TRANSPORT] Scripted Transport ─────────────────────")
        monitor_cb = self.grasp_detector.check_grasp_maintained
        try:
            scripted_transport(self.robot, self.task_type, self.grasp_obs, monitor_callback=monitor_cb, safety_monitor=self.safety_monitor)
            self.state = FSMState.VERIFY
        except GraspLostException as e:
            print(f"\n⚠️ [FSM: Grasp Lost] Exception caught mid-air: {e}")
            self.recovery_pan = float(e.last_obs.get("shoulder_pan.pos", 0.0))
            self.state = FSMState.RECOVERY

    def _handle_verify(self):
        print("\n─── [FSM: VERIFY] Checking Target Zone Placement ────────────")
        time.sleep(0.3)
        final_obs = self.robot.get_observation()
        success = check_task_success(final_obs["front"], self.baseline_img, self.target_zone)
        
        if success:
            print(f"\n🎉 [SUCCESS] {self.target_object.upper()} confirmed in target zone.")
            self.state = FSMState.DONE
        else:
            print(f"\n⚠️  [WARN] Object not detected in target zone after placement. Bounced out?")
            self.search_retries += 1
            if self.search_retries > self.cfg.max_retries:
                self.state = FSMState.FAILED
            else:
                move_to_ready(self.robot, self.task_type, safety_monitor=self.safety_monitor)
                self.state = FSMState.SEARCHING

    def _handle_recovery(self):
        print("\n─── [FSM: RECOVERY] Local Error Correction ──────────────────")
        obs = self.robot.get_observation()
        drop_pose = {j: float(obs.get(j, 0.0)) for j in JOINT_NAMES}
        drop_pose["gripper.pos"] = GRIPPER_OPEN_POS
        lerp_to_waypoint(self.robot, drop_pose, 0.5, safety_monitor=self.safety_monitor)
        
        if self.recovery_pan is not None:
            move_to_ready(self.robot, self.task_type, pan_override=self.recovery_pan, safety_monitor=self.safety_monitor)
        else:
            move_to_home(self.robot, safety_monitor=self.safety_monitor)
            move_to_ready(self.robot, self.task_type, safety_monitor=self.safety_monitor)
        
        self.search_retries += 1
        if self.search_retries > self.cfg.max_retries:
            self.state = FSMState.FAILED
        else:
            self.state = FSMState.SEARCHING