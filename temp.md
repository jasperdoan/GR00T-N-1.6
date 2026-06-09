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
        "homing_offset": -2005,
        "range_min": 778,
        "range_max": 3378
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -2018,
        "range_min": 1022,
        "range_max": 3204
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -898,
        "range_min": 887,
        "range_max": 3122
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -1176,
        "range_min": 822,
        "range_max": 2894
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": -221,
        "range_min": 1028,
        "range_max": 3050
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": 1324,
        "range_min": 1868,
        "range_max": 2625
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
    --dataset.num_episodes=5 \
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