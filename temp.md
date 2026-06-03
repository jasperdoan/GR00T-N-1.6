sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1


lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=true



lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=true \
    --dataset.repo_id=so101/shapes \
    --dataset.num_episodes=10 \
    --dataset.single_task="red cube" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=6 \
    --dataset.reset_time_s=20 \
    --resume=true




------



{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -2000,
        "range_min": 774,
        "range_max": 3308
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -1965,
        "range_min": 946,
        "range_max": 2954
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -1882,
        "range_min": 867,
        "range_max": 3105
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -1285,
        "range_min": 942,
        "range_max": 3017
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 235,
        "range_min": 1103,
        "range_max": 3182
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 1750,
        "range_min": 1590,
        "range_max": 2626
    }
}




----








lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=false \
    --dataset.repo_id=so101/shapes \
    --dataset.num_episodes=10 \
    --dataset.single_task="yellow cube" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=8 \
    --dataset.reset_time_s=20 \
    --resume=true



lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm


lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --display_data=true