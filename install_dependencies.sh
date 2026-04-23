#!/bin/bash

echo "--- Installing Dependencies for GR00T N1.6 for Jetson ---"




# 1. Install LeRobot
echo "--- Cloning and Installing LeRobot ---"
pip install lerobot
pip install pyserial deepdiff feetech-servo-sdk




# 2. Install GR00T-N-1.6
echo "--- Installing GR00T Pip Dependencies ---"
pip install -e .
pip install tyro transformers msgpack av diffusers triton draccus




# 3. Install Flash Attn

echo "--- Installing Flash Attention ---"
pip install flash-attn --no-build-isolation --no-cache-dir


echo "--- Setup Complete! ---"