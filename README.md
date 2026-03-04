# GR00T-N1.5 Training, Deployment, and Optimization Guide

This document outlines the workflow for migrating from data collection to production-level inference on the NVIDIA Jetson Orin. The process is divided between a Workstation (PC) for training/export and the Jetson Orin for optimized deployment.

---

## 1. Dataset Preparation (PC)

LeRobot records datasets in **v3 format**, while the GR00T training pipeline requires **v2 format**. You must convert the data before beginning the finetuning process.

### Conversion Command
Move your recorded LeRobot dataset into a `demo_data/` directory and run:

```bash
python3 scripts/lerobot_conversion/convert_v3_to_v2.py \
    --repo-id clean_table \
    --root demo_data
```

---

## 2. Model Finetuning (PC)

Finetuning should be performed on a workstation with high VRAM. The following command utilizes a 3B parameter base model and applies data augmentation (color jitter and random rotation) to improve generalization.

```bash
python3 -m gr00t.experiment.launch_finetune \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path demo_data/clean_table \
    --modality_config_path examples/SO100/so100_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus 1 \
    --output_dir ~/Desktop/models/so100_finetune \
    --save_steps 2000 \
    --save_total_limit 5 \
    --max_steps 20000 \
    --random_rotation_angle 5 \
    --warmup_ratio 0.1 \
    --state_dropout_prob 0.1 \
    --weight_decay 1e-4 \
    --learning_rate 1e-4 \
    --global_batch_size 32 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --episode_sampling_rate 1.0 \
    --dataloader_num_workers 4
```

---

## 3. Evaluation and Testing (PC)

Before deploying to the Jetson, verify the model's performance on your PC.

### Start the Policy Server
```bash
PYTHONPATH=. python gr00t/eval/run_gr00t_server.py \
  --model-path ~/Desktop/models/so100_finetune/checkpoint-20000 \
  --embodiment-tag NEW_EMBODIMENT
```

### Start the Robot Client
```bash
PYTHONPATH=. python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --lang_instruction="Grab pen and place into pen holder"
```

---

## 4. Pre-Deployment: Export and Optimization

### Export to ONNX (PC)
Exporting the model to ONNX requires significant RAM and disk space. This step is best performed on the workstation where the training environment and weights are already present.

```bash
PYTHONPATH=. python scripts/deployment/export_onnx_n1d6.py \
  --model_path ~/Desktop/models/so100_finetune/checkpoint-20000 \
  --dataset_path ~/Desktop/Isaac-GR00T/demo_data/clean_table \
  --embodiment_tag new_embodiment \
  --output_dir ~/Desktop/models/groot_n1d6_onnx
```

### Build TensorRT Engine *(Jetson Orin)*
The TensorRT engine **must** be built on the target hardware (Jetson) to match the specific ARM/GPU architecture.

```bash
python3 -m scripts.deployment.build_tensorrt_engine \
    --onnx /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model.onnx \
    --engine /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model_fp16.trt \
    --precision fp16
```

### Precision Options for TensorRT

| Precision | Bits | Speed (Orin) | Accuracy | Best For... |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | Slowest (1x) | Highest | Debugging / Reference |
| **BF16** | 16 | Medium (2x) | Very High | Training / Generative AI |
| **FP16** | 16 | Fast (2.5x) | High | **Standard Jetson Deployment** |
| **FP8** | 8 | Very Fast (3.5x) | Medium | New Ampere/Hopper hardware |
| **INT8** | 8 | Fastest (4x) | Variable | **Maximum Throughput (Requires Calibration)** |


### Technical Note: The Role of DiT and TensorRT
The **Diffusion Transformer (DiT)** serves as the "Action Head." It iteratively denoises visual and language embeddings to generate motor commands. Because it runs multiple iterations per inference, it is the primary bottleneck.
*   **Without TensorRT:** ~3.3 Hz (300ms latency)
*   **With TensorRT:** ~5.8 Hz (173ms latency)

---

## 5. Production Deployment (Jetson Orin)

### Model Stripping (Reducing 125GB to ~9.5GB)
Standard training checkpoints include heavy optimizer states (`optimizer.pt`) that are unnecessary for inference. You only need the following core files ( ~9.5GB total), you may remove all the `checkpoints-*/` directories, as top-level/root model files are the final saved model (all the tensors files). If training ended at step 20000, they should be identical. The checkpoint directory additionally contains training state (optimizer, scheduler, rng) that you don't need for inference:

| File/Directory | Approx. Size | Purpose | Required for Inference? |
| :--- | :--- | :--- | :--- |
| `model-00001-of-00002.safetensors` | ~4.7G | Model Weights (Shard 1) | **Yes** |
| `model-00002-of-00002.safetensors` | ~4.5G | Model Weights (Shard 2) | **Yes** |
| `model.safetensors.index.json` | ~120K | Shard Index Mapping | **Yes** |
| `config.json` | ~4K | Architecture Configuration | **Yes** |
| `processor_config.json` | ~12K | Image/Data Pre-processing Logic | **Yes** |
| `embodiment_id.json` | ~4K | Robot Embodiment Metadata | **Yes** |
| `statistics.json` | ~280K | Dataset Normalization Stats | **Yes** |
| `experiment_cfg/` | ~220K | Hydra/Experiment Config Files | **Yes** |
| `optimizer.pt` | ~13G | Training Optimizer State | **No** |
| `scheduler.pt` | ~4K | Learning Rate Scheduler | **No** |
| `trainer_state.json` | ~276K | Training Progress Metadata | **No** |
| `training_args.bin` | ~8K | Training Hyperparameters | **No** |
| `rng_state.pth` | ~16K | Random Seed States | **No** |

**Total footprint for inference: ~9.5GB**


### Correcting the Directory Structure
The `AutoProcessor` expects JSON files at the root level. If they are nested in a `processor/` subdirectory, the model will fail to load.

**Run this on the Jetson to fix the structure:**
```bash
cp /workspaces/isaac_ros-dev/models/so100_inference_checkpoint/processor/* \
   /workspaces/isaac_ros-dev/models/so100_inference_checkpoint/
```

**Final Required Structure:**
```text
so100_inference_checkpoint
├── config.json
├── embodiment_id.json
├── experiment_cfg
│   ├── config.yaml
│   ├── conf.yaml
│   ├── dataset_statistics.json
│   ├── final_model_config.json
│   └── final_processor_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── processor_config.json
├── statistics.json
├── training_args.bin
└── wandb_config.json
```

---

## 6. Running Production Inference (Jetson)

### Step 1: Start the TensorRT Server
```bash
PYTHONPATH=. python gr00t/eval/run_gr00t_trt_server.py \
  --model-path /workspaces/isaac_ros-dev/models/so100_inference_checkpoint \
  --trt-engine-path /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model_fp16.trt \
  --embodiment-tag NEW_EMBODIMENT \
  --compile-backbone \
  --port 5555
```

### Step 2: Start the Evaluation Client
```bash
PYTHONPATH=. python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --action_horizon 16 \
  --camera_w 320 \ 
  --camera_h 240 \
  --lang_instruction="Grab pen and place into pen holder"
```

---

## 7. Infrastructure: Docker Management

### Launching the Container
Ensure the microcontroller ports and udev rules are mapped correctly for hardware access.

```bash
docker run -it \
    --privileged \
    --net=host \
    --device=/dev/ttyACM0:/dev/ttyACM0 \
    --device=/dev/ttyACM1:/dev/ttyACM1 \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v /mnt/nova_ssd/workspaces/isaac_ros-dev:/workspaces/isaac_ros-dev \
    --name isaac_ros_dev-aarch64-container \
    isaac_ros_dev-aarch64
```

`--rm` flag is omitted to allow for persistent containers that can be stopped and restarted without losing state. But you can add `--rm` if you prefer ephemeral containers.

### Lifecycle Commands
*   **Start/Attach:**
    ```bash
    docker start isaac_ros_dev-aarch64-container
    docker exec -it isaac_ros_dev-aarch64-container /bin/bash
    ```
*   **Stop/Remove:**
    ```bash
    docker stop isaac_ros_dev-aarch64-container
    docker rm isaac_ros_dev-aarch64-container
    ```

*Note: Once inside the container, execute `scripts/setup_orin.sh` to initialize the environment dependencies (PyTorch, TensorRT, etc.) specific to the Jetson Orin.*