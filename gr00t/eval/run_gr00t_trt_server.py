"""
GR00T TensorRT Inference Server

A ZeroMQ-based inference server that loads a GR00T policy with TensorRT
acceleration for the DiT (action head). The backbone (vision + language)
still runs in PyTorch; only the DiT forward pass is replaced by TensorRT.

Usage:
    PYTHONPATH=. python gr00t/eval/run_gr00t_trt_server.py \
        --model-path /path/to/checkpoint \
        --trt-engine-path /path/to/dit_model_bf16.trt \
        --embodiment-tag NEW_EMBODIMENT \
        --port 5555

The client side (eval_so100.py) requires no changes — it communicates
over ZMQ and is agnostic to whether the server uses PyTorch or TensorRT.
"""

from dataclasses import dataclass
import logging
import os
import time

import torch
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer


# =============================================================================
# TensorRT DiT Wrapper (self-contained, no external script dependency)
# =============================================================================


class TensorRTDiTWrapper:
    """Wrapper that runs the DiT model through a TensorRT engine.

    Optimizations over naive approach:
    - Pre-allocates output buffer and reuses it across calls
    - Caches input shapes to skip redundant set_input_shape calls
    - Reuses a dedicated CUDA stream for TRT execution
    """

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

        # Dedicated CUDA stream for TRT execution
        self._stream = torch.cuda.Stream(device=device)

        # Cache for pre-allocated output buffer and last-seen input shapes
        self._output_buffer = None
        self._cached_shapes = {}

        logging.info(f"TensorRT engine loaded: {engine_path}")

    def _set_shape_if_changed(self, name: str, tensor: torch.Tensor):
        """Only call set_input_shape when the shape actually changes."""
        shape = tuple(tensor.shape)
        if self._cached_shapes.get(name) != shape:
            self.context.set_input_shape(name, shape)
            self._cached_shapes[name] = shape

    def __call__(self, sa_embs, vl_embs, timestep, image_mask=None, backbone_attention_mask=None):
        """Forward pass through TensorRT DiT."""
        sa_embs = sa_embs.to(self._cuda_device).contiguous()
        vl_embs = vl_embs.to(self._cuda_device).contiguous()
        timestep = timestep.to(self._cuda_device).contiguous()

        if image_mask is not None:
            image_mask = image_mask.to(self._cuda_device).contiguous()
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(self._cuda_device).contiguous()

        # Set shapes only when they change (they stay constant across diffusion steps)
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

        # Pre-allocate or reuse output buffer
        output_shape = tuple(self.context.get_tensor_shape("output"))
        if self._output_buffer is None or self._output_buffer.shape != output_shape:
            self._output_buffer = torch.empty(
                output_shape, dtype=torch.bfloat16, device=self._cuda_device
            )
        self.context.set_tensor_address("output", self._output_buffer.data_ptr())

        # Execute on dedicated stream
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
        """
        TensorRT wrapper matching DiT forward signature.

        Maps DiT parameter names to ONNX export names:
        - hidden_states -> sa_embs
        - encoder_hidden_states -> vl_embs
        - timestep -> timestep
        - image_mask, backbone_attention_mask passed through
        """
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
# Server Configuration & Entry Point
# =============================================================================

DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class TRTServerConfig:
    """Configuration for running the GR00T TensorRT inference server."""

    # Model + TensorRT
    model_path: str = ""
    """Path to the model checkpoint directory (needed for backbone + processor)"""

    trt_engine_path: str = ""
    """Path to the TensorRT engine file (.trt) for the DiT action head"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the backbone on"""

    # Server
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    num_inference_timesteps: int = 4
    """Number of diffusion denoising steps (default 4). Lower = faster but less precise. Try 2 for ~2x DiT speedup."""


def main(config: TRTServerConfig):
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  GR00T TensorRT Inference Server")
    print("=" * 60)
    print(f"  Embodiment tag:    {config.embodiment_tag}")
    print(f"  Model path:        {config.model_path}")
    print(f"  TRT engine path:   {config.trt_engine_path}")
    print(f"  Device:            {config.device}")
    print(f"  Host:              {config.host}")
    print(f"  Port:              {config.port}")
    print("=" * 60)

    # Validate paths
    if not config.model_path:
        raise ValueError("--model-path is required (checkpoint for backbone + processor)")
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path does not exist: {config.model_path}")
    if not config.trt_engine_path:
        raise ValueError("--trt-engine-path is required (.trt file for DiT action head)")
    if not os.path.exists(config.trt_engine_path):
        raise FileNotFoundError(f"TRT engine does not exist: {config.trt_engine_path}")

    # Step 1: Load the full policy from checkpoint (backbone + processor + DiT)
    print("\n[1/3] Loading GR00T policy from checkpoint...")
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )
    policy.model.num_inference_timesteps = config.num_inference_timesteps
    print(f"      Policy loaded (diffusion steps: {config.num_inference_timesteps}).")

    # Step 2: Replace DiT forward with TensorRT engine
    print("[2/3] Replacing DiT action head with TensorRT engine...")
    replace_dit_with_tensorrt(policy, config.trt_engine_path)
    print("      TensorRT engine active.")

    # Step 3: Wrap get_action with Hz logging
    _original_get_action = policy.get_action
    _inference_count = [0]
    _inference_time_sum = [0.0]

    def _timed_get_action(*args, **kwargs):
        t0 = time.perf_counter()
        result = _original_get_action(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _inference_count[0] += 1
        _inference_time_sum[0] += elapsed
        hz = 1.0 / elapsed if elapsed > 0 else float("inf")
        avg_hz = _inference_count[0] / _inference_time_sum[0] if _inference_time_sum[0] > 0 else 0
        print(f"[Server] Inference #{_inference_count[0]}: {elapsed*1000:.1f}ms ({hz:.1f} Hz) | Avg: {avg_hz:.1f} Hz")
        return result

    policy.get_action = _timed_get_action

    # Step 4: Start ZMQ server
    print("[3/3] Starting ZMQ server...")
    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(TRTServerConfig)
    main(config)
