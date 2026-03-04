"""
SO100 Local Inference — No ZMQ Overhead

This script combines the TRT server and eval client into a single process.
Instead of serializing observations over ZeroMQ (which costs ~20-50ms per
frame for two 640x480 cameras), it calls policy.get_action() directly.

Usage (on Jetson Orin, inside Docker container):
    PYTHONPATH=. python gr00t/eval/real_robot/SO100/eval_so100_local.py \
        --model-path /workspaces/isaac_ros-dev/models/so100_inference_checkpoint \
        --trt-engine-path /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model_fp16.trt \
        --embodiment-tag NEW_EMBODIMENT \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.id=follower_arm \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
        --action-horizon 16 \
        --lang-instruction "Grab pen and place into pen holder"
"""

# =============================================================================
# Imports
# =============================================================================

from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, List

import draccus
import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots import koch_follower  # noqa: F401
from lerobot.robots import so_follower as so100_follower  # noqa: F401
from lerobot.robots import so_follower as so101_follower  # noqa: F401
from lerobot.utils.utils import init_logging, log_say


# =============================================================================
# TensorRT DiT Wrapper (copied from run_gr00t_trt_server.py)
# =============================================================================


class TensorRTDiTWrapper:
    """Wrapper that runs the DiT model through a TensorRT engine."""

    def __init__(self, engine_path: str, device: int = 0):
        import tensorrt as trt

        self.device = device
        self._cuda_device = f"cuda:{device}"

        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.set_device(device)
            logging.info(f"CUDA initialized: device {device}")
        else:
            raise RuntimeError("CUDA not available for TensorRT")

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=device)
        self._output_buffer = None
        self._cached_shapes = {}
        logging.info(f"TensorRT engine loaded: {engine_path}")

    def _set_shape_if_changed(self, name: str, tensor: torch.Tensor):
        shape = tuple(tensor.shape)
        if self._cached_shapes.get(name) != shape:
            self.context.set_input_shape(name, shape)
            self._cached_shapes[name] = shape

    def __call__(self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None):
        sa_embs = sa_embs.to(self._cuda_device).contiguous()
        vl_embs = vl_embs.to(self._cuda_device).contiguous()
        timestep = timestep.to(self._cuda_device).contiguous()

        if image_mask is not None:
            image_mask = image_mask.to(self._cuda_device).contiguous()
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(self._cuda_device).contiguous()

        self._set_shape_if_changed("sa_embs", sa_embs)
        self._set_shape_if_changed("vl_embs", vl_embs)
        self._set_shape_if_changed("timestep", timestep)
        if image_mask is not None:
            self._set_shape_if_changed("image_mask", image_mask)
        if backbone_attention_mask is not None:
            self._set_shape_if_changed("backbone_attention_mask", backbone_attention_mask)

        self.context.set_tensor_address("sa_embs", sa_embs.data_ptr())
        self.context.set_tensor_address("vl_embs", vl_embs.data_ptr())
        self.context.set_tensor_address("timestep", timestep.data_ptr())
        if image_mask is not None:
            self.context.set_tensor_address("image_mask", image_mask.data_ptr())
        if backbone_attention_mask is not None:
            self.context.set_tensor_address(
                "backbone_attention_mask", backbone_attention_mask.data_ptr()
            )

        output_shape = tuple(self.context.get_tensor_shape("output"))
        if self._output_buffer is None or self._output_buffer.shape != output_shape:
            self._output_buffer = torch.empty(
                output_shape, dtype=torch.bfloat16, device=self._cuda_device
            )
        self.context.set_tensor_address("output", self._output_buffer.data_ptr())

        with torch.cuda.stream(self._stream):
            success = self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        if not success:
            raise RuntimeError("TensorRT inference failed")

        return self._output_buffer


def replace_dit_with_tensorrt(policy: Gr00tPolicy, trt_engine_path: str, device: int = 0):
    """Replace the DiT forward method in a Gr00tPolicy with TensorRT inference."""
    trt_dit = TensorRTDiTWrapper(trt_engine_path, device=device)

    def trt_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        output = trt_dit(
            sa_embs=hidden_states,
            vl_embs=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )
        if return_all_hidden_states:
            raise RuntimeError("TensorRT only returns the final output")
        return output

    policy.model.action_head.model.forward = trt_forward
    logging.info("DiT replaced with TensorRT engine")


# =============================================================================
# SO100 Adapter (same as eval_so100.py)
# =============================================================================


def recursive_add_extra_dim(obs: Dict) -> Dict:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class So100Adapter:
    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.robot_state_keys = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        self.camera_keys = ["front", "wrist"]

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        model_obs = {}
        model_obs["video"] = {k: obs[k] for k in self.camera_keys}
        state = np.array([obs[k] for k in self.robot_state_keys], dtype=np.float32)
        model_obs["state"] = {
            "single_arm": state[:5],
            "gripper": state[5:6],
        }
        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        single_arm = chunk["single_arm"][0][t]
        gripper = chunk["gripper"][0][t]
        full = np.concatenate([single_arm, gripper], axis=0)
        return {joint_name: float(full[i]) for i, joint_name in enumerate(self.robot_state_keys)}

    def get_action(self, obs: Dict) -> List[Dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LocalEvalConfig:
    """Combined config for local inference (no ZMQ)."""

    # Robot
    robot: RobotConfig

    # Model + TRT
    model_path: str = ""
    """Path to the model checkpoint directory"""

    trt_engine_path: str = ""
    """Path to the TensorRT engine file (.trt) for the DiT action head"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the backbone on"""

    num_inference_timesteps: int = 4
    """Diffusion denoising steps (default 4). Try 2 for speed."""

    # Eval
    action_horizon: int = 8
    """Number of action steps to execute per inference"""

    lang_instruction: str = "Grab markers and place into pen holder."
    """Language instruction for the task"""

    play_sounds: bool = False
    timeout: int = 30


# =============================================================================
# Main
# =============================================================================


@draccus.wrap()
def main(cfg: LocalEvalConfig):
    init_logging()
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  GR00T Local Inference (No ZMQ)")
    print("=" * 60)
    print(f"  Model path:        {cfg.model_path}")
    print(f"  TRT engine:        {cfg.trt_engine_path}")
    print(f"  Embodiment:        {cfg.embodiment_tag}")
    print(f"  Diffusion steps:   {cfg.num_inference_timesteps}")
    print(f"  Action horizon:    {cfg.action_horizon}")
    print(f"  Instruction:       {cfg.lang_instruction}")
    print("=" * 60)

    # Validate paths
    if not cfg.model_path:
        raise ValueError("--model-path is required")
    if cfg.model_path.startswith("/") and not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model path does not exist: {cfg.model_path}")
    if not cfg.trt_engine_path:
        raise ValueError("--trt-engine-path is required")
    if not os.path.exists(cfg.trt_engine_path):
        raise FileNotFoundError(f"TRT engine does not exist: {cfg.trt_engine_path}")

    # 1. Load policy directly (no server)
    print("\n[1/4] Loading GR00T policy...")
    policy = Gr00tPolicy(
        embodiment_tag=cfg.embodiment_tag,
        model_path=cfg.model_path,
        device=cfg.device,
        strict=True,
    )
    policy.model.num_inference_timesteps = cfg.num_inference_timesteps
    print(f"      Policy loaded (diffusion steps: {cfg.num_inference_timesteps}).")

    # 2. Replace DiT with TRT
    print("[2/4] Replacing DiT with TensorRT...")
    replace_dit_with_tensorrt(policy, cfg.trt_engine_path)
    print("      TensorRT engine active.")

    # 3. Initialize robot
    print("[3/4] Initializing robot hardware...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    log_say("Robot connected", cfg.play_sounds, blocking=True)

    # 4. Create adapter (direct, no ZMQ client)
    adapter = So100Adapter(policy)
    log_say(f'Policy ready: "{cfg.lang_instruction}"', cfg.play_sounds, blocking=True)

    # 5. Main loop with Hz logging
    print("[4/4] Starting inference loop...")
    inference_count = 0
    inference_time_sum = 0.0

    try:
        while True:
            obs = robot.get_observation()
            obs["lang"] = cfg.lang_instruction

            t0 = time.perf_counter()
            actions = adapter.get_action(obs)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            inference_count += 1
            inference_time_sum += elapsed
            hz = 1.0 / elapsed if elapsed > 0 else float("inf")
            avg_hz = inference_count / inference_time_sum if inference_time_sum > 0 else 0
            print(f"[Local] Inference #{inference_count}: {elapsed*1000:.1f}ms ({hz:.1f} Hz) | Avg: {avg_hz:.1f} Hz")

            for i, action_dict in enumerate(actions[: cfg.action_horizon]):
                tic = time.time()
                robot.send_action(action_dict)
                toc = time.time()
                if toc - tic < 1.0 / 30:
                    time.sleep(1.0 / 30 - (toc - tic))

    except KeyboardInterrupt:
        print(f"\nStopping. Total inferences: {inference_count}, Avg Hz: {inference_count / inference_time_sum:.1f}" if inference_time_sum > 0 else "\nStopping.")


if __name__ == "__main__":
    main()
