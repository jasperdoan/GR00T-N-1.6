Note to future Jasper:

Try these:
-Run everytime reboot:

-sudo jetson_clocks


- Need to re-run docker command, added runtime nvidia and shm-size=2g

Run Docker

docker run -it \
    --privileged \
    --net=host \
    --runtime nvidia \
    --device=/dev/ttyACM0:/dev/ttyACM0 \
    --device=/dev/ttyACM1:/dev/ttyACM1 \
    -v /dev:/dev \
    -v /run/udev:/run/udev:ro \
    -v /mnt/nova_ssd/workspaces/isaac_ros-dev:/workspaces/isaac_ros-dev \
    --name isaac_dev_container \
    cached_isaac_run_dev_image_local:latest

(--rm to auto delete container on exit, but I usually like to keep it around for debugging and reuse)


Next day:

Start it
docker start isaac_dev_container

Enter it
docker exec -it isaac_dev_container /bin/bash

Stop it
docker stop isaac_dev_container

Delete container (if needed)
docker rm isaac_dev_container


=================



Step 2: Start the TensorRT server (on the Jetson)

PYTHONPATH=. python gr00t/eval/run_gr00t_trt_server.py \
  --model-path /workspaces/isaac_ros-dev/models/so100_finetune \
  --trt-engine-path /workspaces/isaac_ros-dev/models/groot_n1d6_onnx/dit_model_fp16.trt \
  --embodiment-tag NEW_EMBODIMENT \
  --port 5555 


Step 3: Start the client (unchanged, on the Jetson)

PYTHONPATH=. python gr00t/eval/real_robot/SO100/eval_so100.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --action_horizon 16 \
  --lang_instruction="Check in red cube" \
  --robot.use_degrees=true
  



-----------------------------------------


docker exec -it isaac_dev_container /bin/bash
docker start isaac_dev_container
docker stop isaac_dev_container


-----------------------------------------


pip install packaging
pip install wheel

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="110"
export MAX_JOBS=4 

mkdir -p ./flash_attn_wheels

pip wheel "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git" \
    --no-build-isolation \
    --no-cache-dir \
    -w ./flash_attn_wheels
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm

echo "Build finished!"
echo "Wheel is located at: $(pwd)/flash_attn_wheels"
ls -lh ./flash_attn_wheels




------------------------------------------


python3 scripts/read_robot_pos.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm
