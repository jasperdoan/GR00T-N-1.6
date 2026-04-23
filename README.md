Follow these steps to set up the environment, optimize the models for Jetson, and run real-time inference on the SO-100 robot.

## 1. Installation

First, install the necessary system and Python dependencies:

```bash
./install_dependencies.sh
```

## 2. Model Setup

Move the downloaded models into the `models/` directory. Your repository structure should look like this for the scripts to function correctly:

```text
.
├── docs/
├── gr00t/                              # Core logic and policies
├── models/
│   ├── groot_n1d6_onnx/                # ONNX model files
│   └── so100_inference_checkpoint/
├── scripts/                            # Deployment and utility scripts
├── install_dependencies.sh
├── pyproject.toml
└── ...
```

## 3. Build TensorRT Engine

The TensorRT engine **must** be generated on the target hardware (Jetson) to optimize for the specific ARM/GPU architecture. 

Command to convert the ONNX model to a high-performance FP16 TensorRT engine, (one time usage). Only need to convert once.

```bash
python3 -m scripts.deployment.build_tensorrt_engine \
    --onnx {path_to_repo}/models/groot_n1d6_onnx/dit_model.onnx \
    --engine {path_to_repo}/models/groot_n1d6_onnx/dit_model_fp16.trt \
    --precision fp16
```

> Be sure to fill out `{path_to_repo}` with the actual path to the program repository

---

## 4. Running Production Inference

To run model:

### Step A: Start the TensorRT Server
The server handles the heavy computation and model inference.

```bash
PYTHONPATH=. python gr00t/eval/run_gr00t_trt_server.py \
  --model-path {path_to_repo}/models/so100_inference_checkpoint \
  --trt-engine-path {path_to_repo}/models/groot_n1d6_onnx/dit_model_fp16.trt \
  --embodiment-tag NEW_EMBODIMENT \
  --port 5555
```

### Step B: Start the Evaluation Client
The client captures camera feeds, sends them to the server, and executes the returned actions on the robot.

```bash
PYTHONPATH=. python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --action_horizon 16 \
  --lang_instruction="Grab pen and place into pen holder"
```

> Be sure to fill out `{path_to_repo}` with the actual path to the program repository

---

## 🛠 Configuration Details

| Parameter | Description |
| :--- | :--- |
| `--robot.port` | The USB port for your robot (usually `/dev/ttyACM0`). |
| `--robot.cameras` | JSON string defining camera indices and resolutions. |
| `--lang_instruction` | The natural language task for the VLA to perform. |
| `--action_horizon` | Number of future steps to predict (keep at `16` for best results). |