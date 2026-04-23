# Hardware Setup and Dependencies

This section covers camera management, manual calibration overrides, and the technical implementation of Flash Attention on NVIDIA Jetson hardware.

---

## 1. Camera Management

Camera indices can shift between reboots. It is recommended to verify your camera mapping before launching robot sessions.

### Identifying Cameras
Use the built-in LeRobot tool to list active devices:
```bash
lerobot-find-cameras
```
Refer to the [Official LeRobot Camera Documentation](https://huggingface.co/docs/lerobot/en/cameras) for persistent identification methods.

### Updating Launch Arguments
Once identified, update your launch command. Example configuration for a front and wrist camera:
```bash
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"
```

---

## 2. Manual Calibration Configuration

In certain environments (Docker containers or Jetson Orin), the automated calibration save process may fail. If your calibration does not persist, you must manually create the configuration directory and file.

### Manual Setup
```bash
# Create the directory structure
mkdir -p /root/.cache/huggingface/lerobot/calibration/robots/so_follower/

# Create the configuration file
vi /root/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json
```

### Configuration Content (Example)
Paste the following JSON into the file, adjusting values based on your specific arm's calibration:
```json
{
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2023, "range_min": 865, "range_max": 3280},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -2008, "range_min": 817, "range_max": 3361},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -1928, "range_min": 852, "range_max": 3047},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -1345, "range_min": 824, "range_max": 3167},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 1052, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1565, "range_min": 1973, "range_max": 3449}
}
```
*Note: The `scripts/setup_orin.sh` script is designed to automate this process during initial provisioning.*

---



## 3. Flash Attention on Jetson Orin

### Environment Verification
Before installing Flash Attention, verify your CUDA and PyTorch environment to ensure compatibility with prebuilt wheels.
```bash
python3 -V
nvcc --version
python3 -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Prebuilt Wheels
To avoid 90+ minute compilation times, use prebuilt wheels when possible. Check the following repositories for compatibility:
*   [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
*   [Flash-Attn Official Releases](https://flashattn.dev/)

**Wheel Naming Convention:**
`flash_attn-[Version]+cu[CUDA]torch[PyTorch]-cp[Python]-cp[Python]-linux_x86_64.whl`

```
# Example: Python 3.11, CUDA 12.4, PyTorch 2.5, and flash_attn 2.6.3

flash_attn-2.6.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl
```

Download it, then `pip install ./flash_attn-....whl`



## 4. `sm_87` Compatibility Issue

The Jetson AGX Orin uses the **sm_87** architecture. By default, Flash Attention kernels are written for sm_80, sm_86, sm_89, and sm_90. Standard builds often exclude sm_87, leading to the error: `CUDA Error: no kernel image is available for execution`.

This is just incompatibility: flash-attn does not support sm_87 (Jetson Orin) regardless of how it's built. I tried:

- `TORCH_CUDA_ARCH_LIST="8.7" pip install flash-attn --no-build-isolation`
- Different versions of flash-attn (2.6.3, 2.8.3, latest main branch) with different combinations of PyTorch and CUDA versions, seeing it might be forward-compatible.
- Building from source by default

It just doesn't work. Apparently this is a known Jetson limitation — flash-attn's GitHub issues have several reports about Orin specifically.

Although, I found out that sm_87 is very similar to sm_86 (with the primary difference being a larger 192KB unified data cache) according to [NVIDIA docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x), Flash Attention requires explicit enablement to function.

May refer to these two issue on GitHub for more details:
*   [Flash-Attention Issue #449](https://github.com/Dao-AILab/flash-attention/issues/449)
*   [Flash-Attention Issue #1146](https://github.com/Dao-AILab/flash-attention/issues/1146)

And thus, turns out it is possible to compile flash-attn with sm_87 support by modifying the `setup.py` to include the appropriate architecture flag, but if you want to avoid compilation, the recommended approach is to switch to SDPA (Scaled Dot-Product Attention) which is natively supported on Orin and still offers significant performance benefits over standard PyTorch attention.

### Method 1: Compiling from Source with sm_87 Patch
Use this method if you for best performance.

1.  **Preparation:**
    ```bash
    pip uninstall flash-attn -y
    cd /tmp
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    git checkout v2.8.3
    ```

2.  **Modify `setup.py`:**
    Open `setup.py` and locate the block where `cuda_archs()` are checked. Add the `87` check (Look for the block that looks like this):

    ```py
    if "80" in cuda_archs(): 
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_80,code=sm_80")
    ```
    and change, or add this after it:

    ```py
    if "87" in cuda_archs(): 
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_87,code=sm_87")
    ```

3.  **Build:**
    ```bash
    FLASH_ATTN_CUDA_ARCHS=87 python setup.py install
    ```


### Method 2: SDPA Fallback
Scaled Dot-Product Attention (SDPA) is a native PyTorch alternative that fully supports sm_87 via cuBLAS/cuDNN. For single-batch inference on Orin, the performance difference compared to Flash-Attn is negligible.

1.  **Update Config:**
    In `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/config.json`, change:
    `"_attn_implementation": "flash_attention_2"` → `"_attn_implementation": "sdpa"`

2.  **Patch the Modeling Script:**
    Modify `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py` to propagate the attention implementation from the global config:

    **Target Change (Vision Section):**
    ```python
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'intern_vit_6b':
                self.vision_model = InternVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'siglip_vision_model':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = SiglipVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'siglip2_vision_model':
                config.vision_config._attn_implementation = 'flash_attention_2'
                self.vision_model = Siglip2VisionModel(config.vision_config)
            elif config.vision_config.model_type == 'radio':
                self.vision_model = RADIOModel(config.vision_config)
    ```
    to
    ```python
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.model_type == 'intern_vit_6b':
                self.vision_model = InternVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'siglip_vision_model':
                config.vision_config._attn_implementation = config._attn_implementation
                self.vision_model = SiglipVisionModel(config.vision_config)
            elif config.vision_config.model_type == 'siglip2_vision_model':
                config.vision_config._attn_implementation = config._attn_implementation
                self.vision_model = Siglip2VisionModel(config.vision_config)
            elif config.vision_config.model_type == 'radio':
                self.vision_model = RADIOModel(config.vision_config)
    ```

    **Target Change (Language Section):**
    ```python
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen2ForCausalLM':
                assert config.text_config._attn_implementation == 'flash_attention_2', f"Qwen2 must use flash_attention_2 but got {config.text_config._attn_implementation}"
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen3ForCausalLM':
                assert config.text_config._attn_implementation == 'flash_attention_2', f"Qwen3 must use flash_attention_2 but got {config.text_config._attn_implementation}"
                self.language_model = Qwen3ForCausalLM(config.text_config)
    ```
    to
    ```python
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.text_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen2ForCausalLM':
                config.text_config._attn_implementation = config._attn_implementation
                self.language_model = Qwen2ForCausalLM(config.text_config)
            elif config.text_config.architectures[0] == 'Qwen3ForCausalLM':
                config.text_config._attn_implementation = config._attn_implementation
                self.language_model = Qwen3ForCausalLM(config.text_config)
    ```

---

# Issues and Performance

This section documents technical hurdles, hardware-specific limitations (specifically for the NVIDIA Jetson Orin), and a deep dive into model optimization strategies.

---

## 1. Known Issues and Troubleshooting

### Eagle Backbone Cache Conflicts
When switching between attention implementations (e.g., from `flash_attention_2` to `sdpa`), the Hugging Face cache for the Eagle modules may persist with old configurations, causing loading errors. 
**Solution:** Clear the specific module cache:
```bash
rm -rf /root/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2/
```

### Configuration Patching
To ensure the `config.json` settings are respected by the model, `modeling_eagle3_vl.py` must be patched to propagate the `_attn_implementation` flag throughout the vision and language components.

### Flash-Attention Compatibility
Prebuilt Flash-Attention wheels often do not support `sm_87` (NVIDIA Orin architecture) out of the box. If compilation fails or the architecture is not recognized, the system should fall back to SDPA.

---

## 2. Performance Benchmarks: Jetson Orin

The following benchmarks illustrate the performance gap between standard PyTorch and TensorRT (TRT) optimizations on the Jetson Orin.

### Inference Speed Comparison
| Implementation | Latency (ms) | Frequency (Hz) |
| :--- | :--- | :--- |
| **PyTorch (Standard)** | ~312.4ms | 3.2 Hz |
| **TensorRT (Optimized)** | ~208.5ms | 4.8 Hz |

### Component Bottleneck Analysis
Optimization effort is best spent on the **DiT Action Head**, as it represents the primary bottleneck.

| Component | Standard PyTorch | TensorRT |
| :--- | :--- | :--- |
| **Backbone (Vision + Lang)** | ~93ms | ~95ms |
| **DiT Action Head** | **~202ms** | **~72ms** |
| **Total Latency** | **300ms** | **173ms** |

*Note: On a high-end Desktop GPU, expected performance is 12–17 Hz. The Orin's lower frequency (approx. 5.8 Hz with TRT) is a hardware-bound limitation and not necessarily a software inefficiency.*

---

## 3. Attention Mechanisms Comparison

The model's ability to process relationships between pixels and language tokens depends on the "Attention" implementation chosen in `config.json`.

| Feature | Eager (Standard) | SDPA | Flash-Attn 2 |
| :--- | :--- | :--- | :--- |
| **Mechanism** | Standard PyTorch math ops. | Built-in PyTorch dispatcher. | Optimized CUDA kernels. |
| **Orin Support** | Yes | Yes (Recommended) | No (sm_87 compatibility) |
| **Efficiency** | Low ($O(N^2)$ memory). | High (Memory efficient). | Highest (Tiling/SRAM). |
| **Speed** | Slowest. | Middle. | Fastest. |

### Why use SDPA on Orin?
While Flash-Attention is the "gold standard," it is often not compiled for the Orin's `sm_87` architecture. **SDPA (Scaled Dot-Product Attention)** serves as the optimal middle ground, offering significant memory savings and speed increases over "Eager" without the compilation hurdles of Flash-Attention.

---

## 4. Key Model Concepts (GR00T N1.5)

### Diffusion Denoising Steps
GR00T N1.5 utilizes a **Diffusion Transformer (DiT)** to generate actions. Instead of predicting a single coordinate, it starts with random noise and iteratively refines it into a motor command.

*   **Higher Steps:** Increased precision and smoother paths; higher latency.
*   **Lower Steps (Default: 4):** Essential for real-time robotics; faster but may introduce jitter if reduced too far.

### Model Components
1.  **Backbone (System 2):** Handles Vision (SiGLIP2) and Language processing. It is responsible for "understanding" the scene.
2.  **Action Head (System 1):** The DiT (Diffusion Transformer) that converts understanding into physical movements. This is the most computationally expensive part of the inference loop.

---

## 5. Summary of Current Deployment State

| Component | Configuration / Status |
| :--- | :--- |
| **Server** | `run_gr00t_trt_server.py` using ZMQ + TRT. |
| **Attention** | `sdpa` (Flash-attn bypassed for Orin compatibility, or compile from source for `flash-attn`). |
| **Backbone** | Eagle v2 (Patched for config propagation). |
| **Target Hz** | ~4.8 Hz to 5.8 Hz (Production target for Orin). |