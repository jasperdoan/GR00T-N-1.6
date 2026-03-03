Convert:

python3 scripts/lerobot_conversion/convert_v3_to_v2.py \
    --repo-id clean_table \
    --root demo_data


===================================================================


Finetune:

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


===================================================================


Run Server + Client:

PYTHONPATH=. python gr00t/eval/run_gr00t_server.py \
  --model-path ~/Desktop/models/so100_finetune/checkpoint-20000 \
  --embodiment-tag NEW_EMBODIMENT


PYTHONPATH=. python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --lang_instruction="Grab pen and place into pen holder"


===================================================================


Pre-Deployment

Export to ONNX (Should be done on PC - has full training env, weights, high RAM and disk space):

PYTHONPATH=. python scripts/deployment/export_onnx_n1d6.py \
  --model_path ~/Desktop/models/so100_finetune/checkpoint-20000 \
  --dataset_path ~/Desktop/Isaac-GR00T/demo_data/clean_table \
  --embodiment_tag new_embodiment \
  --output_dir ~/Desktop/models/groot_n1d6_onnx


Build TensorRT Engine (Need to be on Jetson or other ARM device with TensorRT installed):

PYTHONPATH=. python scripts/deployment/build_tensorrt_engine.py \
  --onnx_path ./groot_n1d6_onnx/dit_model.onnx \
  --engine_path ./groot_n1d6_onnx/dit_model_bf16.trt \
  --precision bf16

python3 -m scripts.deployment.build_tensorrt_engine \
    --onnx /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model.onnx \
    --engine /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model_bf16.trt \
    --precision bf16






====================================================================


Run Docker

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


(--rm to auto delete container on exit, but I usually like to keep it around for debugging and reuse)


Next day:

Start container:
docker start isaac_ros_dev-aarch64-container

Attach to container:
docker exec -it isaac_ros_dev-aarch64-container /bin/bash

Stop container (after exit)
docker stop isaac_ros_dev-aarch64-container

Delete container (if needed)
docker rm isaac_ros_dev-aarch64-container



=====================================================================


Self notes:


What is DiT and why TensorRT?
DiT = Diffusion Transformer. It's the "action head" of the GR00T model — the part that takes the vision+language embeddings and generates robot actions through a diffusion (denoising) process. It runs multiple denoising steps (typically 4-16), so it's called many times per inference. That makes it the performance bottleneck.

Why TensorRT if you still need the checkpoint? The purpose isn't to eliminate the checkpoint — it's speed. On Orin:

Without TensorRT: 300ms per inference (3.3 Hz)
With TensorRT: 173ms per inference (5.8 Hz)
The backbone processes the camera images and language once, then the DiT runs 4+ denoising iterations. TensorRT makes each of those iterations faster.



You don't need the full 22GB
Looking at your checkpoint, about 13GB is optimizer.pt (training only). For inference you only need:

File	Size	Needed?
model-00001-of-00002.safetensors	4.7G	Yes
model-00002-of-00002.safetensors	4.5G	Yes
model.safetensors.index.json	    120K	Yes
config.json	                      4K	  Yes
processor_config.json	            12K	  Yes
embodiment_id.json	              4K	  Yes
statistics.json	                  280K	Yes
experiment_cfg/	                  220K	Yes
optimizer.pt	                    13G	  No
scheduler.pt	                    4K	  No
trainer_state.json	              276K	No
training_args.bin	                8K	  No
rng_state.pth	                    16K	  No
Total needed: ~9.5GB (not 22GB). You can skip the optimizer, scheduler, trainer state, and rng state.


Is this how production deployment works?
Yes, roughly. In production robotics, the model weights live on the edge device. The Jetson Orin has 16-64GB of shared CPU/GPU memory (depending on variant), and a 3B parameter model in BF16 uses ~6GB of VRAM. This is the standard deployment pattern — the full inference stack runs on the robot's onboard compute.



Step 1: Transfer checkpoint to Jetson (inference files only, ~9.5GB)


Remove checkpoint-20000/. Regarding your question about checkpoint-20000 vs the top-level files: yes, the top-level model files are the final saved model. If training ended at step 20000, they should be identical. The checkpoint directory additionally contains training state (optimizer, scheduler, rng) that you don't need for inference. Using the top-level model files is correct.

The error is straightforward. AutoProcessor.from_pretrained() expects processor_config.json, embodiment_id.json, and statistics.json at the root of the directory, but yours are nested inside a processor/ subdirectory.

In the checkpoint-20000/ directory, these files were at the root level. In the top-level so100_finetune/ directory, they're inside processor/. You're using the top-level files, so they're in the wrong spot.

The fix is to copy those three files up to the root. Run this on the Jetson:

cp /workspaces/isaac_ros-dev/models/so100_inference_checkpoint/processor/* \
   /workspaces/isaac_ros-dev/models/so100_inference_checkpoint/

After that, your directory should look like:

so100_inference_checkpoint/
├── config.json
├── embodiment_id.json          ← now at root
├── experiment_cfg/
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── processor/
├── processor_config.json       ← now at root
├── statistics.json             ← now at root
└── ...


Step 2: Start the TensorRT server (on the Jetson)

PYTHONPATH=. python gr00t/eval/run_gr00t_trt_server.py \
  --model-path /workspaces/isaac_ros-dev/models/so100_inference_checkpoint \
  --trt-engine-path /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model_bf16.trt \
  --embodiment-tag NEW_EMBODIMENT \
  --port 5555


Step 3: Start the client (unchanged, on the Jetson)

PYTHONPATH=. python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --action_horizon 16 \
  --lang_instruction="Grab pen and place into pen holder"





====================================================================




On Jetson stuff:


In GR00T root directory
```
pip install -e .
tyro
transformers
msgpack
av
diffusers
triton
draccus

torchvision==0.20.0

  # Install torchvision dependencies
  sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

  # Clone and build torchvision
  cd /tmp
  git clone --branch v0.20.0 https://github.com/pytorch/vision torchvision
  cd torchvision
  export BUILD_VERSION=0.20.0
  python3 setup.py install --user

torch==2.5.0a0+872d972e41.nv24.8
  NVIDIA-compiled PyTorch 2.5 wheel
  pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

flash-attn
  pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3+cu124torch2.5-cp310-cp310-linux_aarch64.whl

  Looks like I can't use CUDA 12.4 flash-attn on the Jetson because it has CUDA 12.6...

  This will take a while (30-60+ minutes on the Orin) because it compiles the CUDA kernels from source. Make sure you have enough disk space (~2-3GB for the build). If you want to speed it up slightly, you can restrict compilation to only your architecture:
  TORCH_CUDA_ARCH_LIST="8.7" pip install flash-attn --no-build-isolation --no-cache-dir


  https://github.com/Dao-AILab/flash-attention/issues/1146

```

Clone lerobot repo, cd inside it, then:
```
pip install -e . --no-deps

pyserial
deepdiff
feetech-servo-sdk
```



```
mkdir -p /root/.cache/huggingface/lerobot/calibration/robots/so_follower/
vi /root/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json

{
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2023, "range_min": 865, "range_max": 3280},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -2008, "range_min": 817, "range_max": 3361},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -1928, "range_min": 852, "range_max": 3047},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -1345, "range_min": 824, "range_max": 3167},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 1052, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1565, "range_min": 1973, "range_max": 3449}
}

```




=====================================================================











=====================================================================
PYTORCH TEST

root@ubuntu:/workspaces/isaac_ros-dev/src/GR00T-N-1.6# PYTHONPATH=. python -c "
import time, torch, numpy as np
import gr00t.model
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path='/workspaces/isaac_ros-dev/models/so100_inference_checkpoint',
    device='cuda',
)

modality = policy.get_modality_config()
video_T = len(modality['video'].delta_indices)
state_T = len(modality['state'].delta_indices)
lang_key = modality['language'].modality_keys[0]

print('Denoising steps:', policy.model.action_head.num_inference_timesteps)
print('Language key:', lang_key)
print('Video keys:', modality['video'].modality_keys)
print('State keys:', modality['state'].modality_keys)
print('Action horizon:', len(modality['action'].delta_indices))

dummy_obs = {
    'video': {k: np.zeros((1, video_T, 480, 640, 3), dtype=np.uint8)
              for k in modality['video'].modality_keys},
    'state': {'single_arm': np.zeros((1, state_T, 5), dtype=np.float32),
              'gripper':     np.zeros((1, state_T, 1), dtype=np.float32)},
    'language': {lang_key: [['Grab pen']]}
}

# Warmup
"rint(f'Inference: {np.mean(times)*1000:.1f}ms avg  ({1/np.mean(times):.1f} Hz)')
/usr/local/lib/python3.10/dist-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.18). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Tune backbone llm: False
Tune backbone visual: False
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.12.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.12.post_attention_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.13.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.13.post_attention_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.14.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.14.post_attention_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.15.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.15.post_attention_layernorm.weight
Casting trainable parameter model.language_model.model.layers.12.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.post_attention_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.post_attention_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.post_attention_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.post_attention_layernorm.weight to fp32
/workspaces/isaac_ros-dev/src/GR00T-N-1.6/gr00t/model/modules/dit.py:205: FutureWarning: Accessing config attribute `compute_dtype` directly via 'AlternateVLDiT' object attribute is deprecated. Please access 'compute_dtype' over 'AlternateVLDiT's config object instead, e.g. 'unet.config.compute_dtype'.
  embedding_dim=self.inner_dim, compute_dtype=self.compute_dtype
/workspaces/isaac_ros-dev/src/GR00T-N-1.6/gr00t/model/modules/dit.py:236: FutureWarning: Accessing config attribute `output_dim` directly via 'AlternateVLDiT' object attribute is deprecated. Please access 'output_dim' over 'AlternateVLDiT's config object instead, e.g. 'unet.config.output_dim'.
  self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
Total number of DiT parameters:  1091722240
Using AlternateVLDiT for diffusion model
Tune action head projector: True
Tune action head diffusion model: True
Tune action head vlln: True
`use_fast` is set to `True` but the image processor class does not have a fast version.  Falling back to the slow version.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.56it/s]
Denoising steps: 4
Language key: annotation.human.task_description
Video keys: ['front', 'wrist']
State keys: ['single_arm', 'gripper']
Action horizon: 16
Inference: 312.4ms avg  (3.2 Hz)






TRT TEST

root@ubuntu:/workspaces/isaac_ros-dev/src/GR00T-N-1.6# PYTHONPATH=. python -c "
import time, torch, numpy as np
import gr00t.model
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
import sys
sys.path.insert(0, '.')
from gr00t.eval.run_gr00t_trt_server import replace_dit_with_tensorrt

policy = Gr00tPolicy(
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    model_path='/workspaces/isaac_ros-dev/models/so100_inference_checkpoint',
    device='cuda',
)
replace_dit_with_tensorrt(policy, '/workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model_bf16.trt')

modality = policy.get_modality_config()
video_T = len(modality['video'].delta_indices)
state_T = len(modality['state'].delta_indices)
lang_key = modality['language'].modality_keys[0]

dummy_obs = {
    'video': {k: np.zeros((1, video_T, 480, 640, 3), dtype=np.uint8)
              for k in modality['video'].modality_keys},
    'state': {'single_arm': np.zeros((1, state_T, 5), dtype=np.float32),
              'gripper':     np.zeros((1, state_T, 1), dtype=np.float32)},
    'language': {lang_key: [['Grab pen']]}
}

policy.get_action(dummy_obs)  # warmup
torch.cuda.synchronize()

"rint(f'TRT inference: {np.mean(times)*1000:.1f}ms avg  ({1/np.mean(times):.1f} Hz)')
/usr/local/lib/python3.10/dist-packages/albumentations/__init__.py:13: UserWarning: A new version of Albumentations is available: 2.0.8 (you have 1.4.18). Upgrade using: pip install -U albumentations. To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.
  check_for_updates()
Tune backbone llm: False
Tune backbone visual: False
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.12.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.12.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.12.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.12.post_attention_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.13.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.13.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.13.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.13.post_attention_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.14.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.14.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.14.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.14.post_attention_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.q_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.k_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.v_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.o_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.q_norm.weight
Backbone trainable parameter: model.language_model.model.layers.15.self_attn.k_norm.weight
Backbone trainable parameter: model.language_model.model.layers.15.mlp.gate_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.mlp.up_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.mlp.down_proj.weight
Backbone trainable parameter: model.language_model.model.layers.15.input_layernorm.weight
Backbone trainable parameter: model.language_model.model.layers.15.post_attention_layernorm.weight
Casting trainable parameter model.language_model.model.layers.12.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.12.post_attention_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.13.post_attention_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.14.post_attention_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.q_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.k_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.v_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.o_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.q_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.self_attn.k_norm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.mlp.gate_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.mlp.up_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.mlp.down_proj.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.input_layernorm.weight to fp32
Casting trainable parameter model.language_model.model.layers.15.post_attention_layernorm.weight to fp32
/workspaces/isaac_ros-dev/src/GR00T-N-1.6/gr00t/model/modules/dit.py:205: FutureWarning: Accessing config attribute `compute_dtype` directly via 'AlternateVLDiT' object attribute is deprecated. Please access 'compute_dtype' over 'AlternateVLDiT's config object instead, e.g. 'unet.config.compute_dtype'.
  embedding_dim=self.inner_dim, compute_dtype=self.compute_dtype
/workspaces/isaac_ros-dev/src/GR00T-N-1.6/gr00t/model/modules/dit.py:236: FutureWarning: Accessing config attribute `output_dim` directly via 'AlternateVLDiT' object attribute is deprecated. Please access 'output_dim' over 'AlternateVLDiT's config object instead, e.g. 'unet.config.output_dim'.
  self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)
Total number of DiT parameters:  1091722240
Using AlternateVLDiT for diffusion model
Tune action head projector: True
Tune action head diffusion model: True
Tune action head vlln: True
`use_fast` is set to `True` but the image processor class does not have a fast version.  Falling back to the slow version.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.68it/s]
[03/02/2026-22:30:27] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
[03/02/2026-22:30:29] [TRT] [W] Using default stream in enqueueV3() may lead to performance issues due to additional calls to cudaStreamSynchronize() by TensorRT to ensure correct synchronization. Please use non-default stream instead.
TRT inference: 208.5ms avg  (4.8 Hz)











================


rm -rf /root/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2/



config.json currently has "sdpa", so the Eagle backbone will use PyTorch's native SDPA. You can switch between modes at any time by just changing that one line in config.json (and re-syncing + clearing cache):

_attn_implementation	Works on Orin	Speed
"flash_attention_2"	No (sm_87 not compiled)	Fastest
"sdpa"	Yes	Middle
"eager"	Yes	Slowest





Flash-attn is NOT the primary bottleneck. Look at the benchmark numbers:

Component	Orin Eager	Orin TRT
Backbone (vision + language)	93ms	95ms
DiT action head	202ms	72ms
Total	300ms	173ms
Flash-attn only affects the backbone (93ms). TRT already handles the big one — the DiT went from 202ms to 72ms. Switching flash-attn → sdpa → eager on the backbone changes maybe 20-30ms. Not the bottleneck.

Flash-attn affects both vision AND language, not just language. SiGLIP2 (the vision encoder) also uses attention layers. But as the numbers show, the backbone runs in ~93ms regardless of flash-attn vs SDPA.

The arm being slow is simply the Jetson Orin being a smaller computer. On your PC, inference was likely running at 12-17 Hz (every ~60-80ms). On the Orin with TRT, you're getting ~5.8 Hz (every ~173ms). That's a real hardware difference — the Orin has a fraction of the GPU compute of a desktop GPU. This is expected and can't be optimized away.

TRT actually helps significantly — without it on the Orin you'd be at 3.3 Hz (300ms). You're already running the most impactful optimization.







Summary of current state:

Component	Status
run_gr00t_trt_server.py	Ready (clean ZMQ + TRT server)
modeling_eagle3_vl.py	Patched (propagates config._attn_implementation)
config.json	Set to "sdpa" (flash-attn bypassed)
TRT inference speed	~208ms / 4.8 Hz






**GR00T N1.5** is an upgraded version of NVIDIA’s foundation model for humanoid robots (Project GR00T), released in mid-2025. It serves as a "universal brain" for robots, enabling them to understand natural language and execute complex physical tasks.

### 1. What is `denoising_steps`?
In the context of **GR00T N1.5**, `denoising_steps` refers to a parameter used by its **Diffusion Transformer (DiT)** (the "System 1" or action-generating part of the model).

*   **The Process:** Unlike traditional models that predict a single action, GR00T N1.5 uses a **diffusion process**. It starts with a block of random noise representing possible robot movements and iteratively "cleans" (denoises) it to reveal the actual intended action.
*   **The Parameter:** `denoising_steps` determines how many times the model iterates to refine that noise into a smooth, precise motor command.
    *   **Higher steps:** Generally leads to higher precision and smoother movements but takes more time (higher latency).
    *   **Lower steps:** Faster execution (essential for real-time robotics) but can lead to "jittery" or less accurate movements if too low.

---

### 2. Difference between Flash-Attn vs. SDPA vs. Eager
These terms refer to different ways a model calculates **Attention**, which is the mechanism used to process relationships between data (like camera pixels and language tokens).

| Feature | **Eager (Standard)** | **SDPA (Scaled Dot-Product Attention)** | **Flash-Attn (FlashAttention-2/3)** |
| :--- | :--- | :--- | :--- |
| **What it is** | The "manual" way. Uses standard PyTorch math ops (`matmul`, `softmax`). | A built-in PyTorch function that acts as a "smart dispatcher." | A specialized, highly optimized CUDA kernel. |
| **Efficiency** | **Low.** It creates a massive "attention matrix" in memory, which is slow and memory-intensive. | **High.** It automatically chooses the fastest method available for your hardware. | **Highest.** It uses "tiling" to process data in tiny blocks, avoiding slow memory reads. |
| **Memory Usage** | Scales quadratically ($O(N^2)$). Easily runs out of memory on long sequences. | Memory-efficient; often uses 50-80% less memory than Eager. | Extremely efficient; allows for much longer context (longer "memory"). |
| **Speed** | Slowest. | Fast (often uses Flash-Attn under the hood). | Fastest (2-4x faster than Eager). |

#### Summary for GR00T N1.5:
*   **Eager** is rarely used in production now because it is too slow for real-time robotics.
*   **SDPA** is the modern default for developers using PyTorch because it is easy to use and "just works."
*   **Flash-Attn** is the "gold standard" for high-performance models like GR00T N1.5. It allows the robot to process high-resolution visual data and complex instructions simultaneously without the GPU lagging or crashing.





Need to play around with the `denoising_steps` parameter to find the right balance of speed vs. smoothness for your specific robot and tasks (num_inference_timesteps)