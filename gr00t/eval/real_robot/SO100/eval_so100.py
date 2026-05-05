import logging
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any, Dict, List

import draccus
import numpy as np
from gr00t.policy.server_client import PolicyClient
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.utils import init_logging, log_say
from lerobot.robots import koch_follower  # noqa: F401
from lerobot.robots import so_follower as so100_follower  # noqa: F401
from lerobot.robots import so_follower as so101_follower  # noqa: F401

# =============================================================================
# 1. So100 Adapter (Data Formatting)
# =============================================================================

class So100Adapter:
    """Formats robot data for GR00T and decodes model outputs."""
    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client
        self.robot_state_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ]
        self.camera_keys = ["front", "wrist"]

    def recursive_add_extra_dim(self, obs: Any) -> Any:
        """Adds B=1, T=1 dimensions required by GR00T server."""
        if isinstance(obs, dict):
            return {k: self.recursive_add_extra_dim(v) for k, v in obs.items()}
        elif isinstance(obs, np.ndarray):
            return obs[np.newaxis, ...]
        return [obs]

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        """Converts raw robot dict to GR00T VLA input format."""
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs = {
            "video": {k: obs[k] for k in self.camera_keys},
            "state": {
                "single_arm": state[:5],  # 5 joints
                "gripper": state[5:6],    # 1 gripper
            },
            "language": {"annotation.human.task_description": obs["lang"]}
        }
        # Add (B=1, T=1)
        return self.recursive_add_extra_dim(self.recursive_add_extra_dim(model_obs))

    def get_action_chunk(self, obs: Dict) -> List[Dict[str, float]]:
        """Queries the server and returns a list of 16 action steps."""
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, _ = self.policy.get_action(model_input)
        
        # Flatten the (B, T, D) results into a list of robot commands
        horizon = action_chunk["single_arm"].shape[1]
        decoded_chunk = []
        for t in range(horizon):
            arm = action_chunk["single_arm"][0][t]
            grip = action_chunk["gripper"][0][t]
            full = np.concatenate([arm, grip])
            decoded_chunk.append({
                joint: float(full[i]) for i, joint in enumerate(self.robot_state_keys)
            })
        return decoded_chunk


# =============================================================================
# 2. Async Policy Handler (The Background "Brain")
# =============================================================================

class AsyncPolicyHandler:
    """Runs model inference in a background thread to prevent robot stutter."""
    def __init__(self, adapter: So100Adapter, robot):
        self.adapter = adapter
        self.robot = robot
        self.latest_chunk = None
        self.chunk_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.lang = ""
        
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        while self.running:
            try:
                # 1. Capture current observation
                obs = self.robot.get_observation()
                obs["lang"] = self.lang
                
                # 2. Inference (This part is slow, but runs in background)
                new_actions = self.adapter.get_action_chunk(obs)
                
                # 3. Update the shared buffer
                with self.lock:
                    self.latest_chunk = new_actions
                    self.chunk_id += 1 # Signal that a new plan is ready
            except Exception as e:
                logging.error(f"Inference error: {e}")
                time.sleep(0.1)

    def get_action_step(self, current_step_idx: int):
        with self.lock:
            if self.latest_chunk is None:
                return None, 0
            
            # Ensure we don't go past the 16th step
            idx = min(current_step_idx, len(self.latest_chunk) - 1)
            return self.latest_chunk[idx], self.chunk_id


# =============================================================================
# 3. Main Evaluation Script
# =============================================================================

@dataclass
class EvalConfig:
    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    lang_instruction: str = "Pick up the red cube and place it in the center."
    hz: int = 20  # Matches your Isaac Lab Decimation 6


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Connect to Robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    log_say("Robot Connected", False)

    # Connect to Policy Server
    client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    adapter = So100Adapter(client)
    
    # Start the Async Brain
    brain = AsyncPolicyHandler(adapter, robot)
    brain.lang = cfg.lang_instruction

    print(f"\n--- Starting 20Hz Control Loop ---")
    print(f"Instruction: {cfg.lang_instruction}")
    
    # Tracking variables
    current_step_in_chunk = 0
    last_processed_chunk_id = -1
    target_dt = 1.0 / cfg.hz

    try:
        while True:
            loop_start = time.perf_counter()

            # 1. Pull the most recent action step from the background brain
            action_dict, chunk_id = brain.get_action_step(current_step_in_chunk)

            if action_dict:
                # If the brain just finished a NEW inference, reset our step counter
                # to 0 to use the freshest possible trajectory immediately.
                if chunk_id != last_processed_chunk_id:
                    current_step_in_chunk = 0
                    last_processed_chunk_id = chunk_id
                    action_dict, _ = brain.get_action_step(current_step_in_chunk)

                # 2. Execute Action
                robot.send_action(action_dict)
                current_step_in_chunk += 1
            else:
                print("Waiting for model warmup...", end="\r")

            # 3. Precise Timing (Throttles to 20Hz)
            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            
            # Print status periodically
            if current_step_in_chunk % 20 == 0:
                actual_hz = 1.0 / (time.perf_counter() - loop_start)
                logging.info(f"Step {current_step_in_chunk}/16 | Freq: {actual_hz:.1f}Hz")

    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        brain.running = False
        robot.disconnect()


if __name__ == "__main__":
    eval()