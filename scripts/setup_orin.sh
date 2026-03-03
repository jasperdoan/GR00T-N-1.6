#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# Note: We will temporarily disable this for the apt-get update
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "--- Starting Setup for Jetson Orin ---"

# 1. Install GR00T-N-1.6
echo "--- Installing GR00T-N-1.6 ---"
cd "$WORKSPACE_ROOT/src/GR00T-N-1.6"
pip install -e .

echo "--- Installing GR00T Pip Dependencies ---"
pip install tyro transformers msgpack av diffusers triton draccus

# 2. Install torchvision v0.20.0 from source
echo "--- Building torchvision v0.20.0 ---"
# Again, using || true in case of apt hiccups, but these are standard libs
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev || true

cd /tmp
if [ -d "torchvision" ]; then rm -rf torchvision; fi
git clone --branch v0.20.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.20.0
# We use --user to match your requirement
python3 setup.py install --user

# 3. Install LeRobot
echo "--- Installing LeRobot ---"
cd "$WORKSPACE_ROOT/src/lerobot"
pip install -e . --no-deps

echo "--- Installing LeRobot Pip Dependencies ---"
pip install pyserial deepdiff feetech-servo-sdk

# 4. Calibration JSON Configuration
echo "--- Writing Robot Calibration JSON ---"
CALIBRATION_DIR="/root/.cache/huggingface/lerobot/calibration/robots/so_follower"
mkdir -p "$CALIBRATION_DIR"

cat <<EOF > "$CALIBRATION_DIR/follower_arm.json"
{
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2023, "range_min": 865, "range_max": 3280},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -2008, "range_min": 817, "range_max": 3361},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -1928, "range_min": 852, "range_max": 3047},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": -1345, "range_min": 824, "range_max": 3167},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 1052, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 1565, "range_min": 1973, "range_max": 3449}
}
EOF

# 5. Install Flash Attention (built from source with sm_87 / Jetson Orin support)
# Precompiled wheels do not include sm_87 kernels. We clone the source,
# patch setup.py to add Jetson Orin (compute_87 / sm_87) gencode flags,
# then build from scratch. This takes ~30-60 min on Orin.
echo "--- Building Flash Attention from source with sm_87 support ---"

cd /tmp
if [ -d "flash-attention" ]; then rm -rf flash-attention; fi
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3

# Patch setup.py: insert sm_87 gencode flags after the last existing arch line.
# Based on https://github.com/Dao-AILab/flash-attention/issues/1146
python3 - <<'PYEOF'
with open("setup.py", "r") as f:
    content = f.read()

# The setup.py structure uses if "80" in cuda_archs(): blocks.
# We add an equivalent sm_87 block right after the sm_80 block.
sm80_marker = 'if "80" in cuda_archs():'
sm87_block = '\nif "87" in cuda_archs():\n    cc_flag.append("-gencode")\n    cc_flag.append("arch=compute_87,code=sm_87")'

idx = content.find(sm80_marker)
if idx == -1:
    print("ERROR: could not find 'if \"80\" in cuda_archs():' in setup.py — check manually")
    raise SystemExit(1)

# Find the end of the sm_80 block (after its last cc_flag.append line)
block_start = idx
line_end = content.find("\n", block_start)       # end of if "80" line
line_end = content.find("\n", line_end + 1)      # end of first cc_flag.append
line_end = content.find("\n", line_end + 1)      # end of second cc_flag.append

content = content[:line_end] + sm87_block + content[line_end:]

with open("setup.py", "w") as f:
    f.write(content)
print("setup.py patched: sm_87 block added after sm_80 block")
PYEOF

FLASH_ATTN_CUDA_ARCHS=87 python setup.py install

echo "--- Setup Complete! ---"