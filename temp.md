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
    --robot.cameras="{wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=false \
    --dataset.repo_id=so101/shapes \
    --dataset.num_episodes=25 \
    --dataset.single_task="red cube" \
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
    --display_data=false




---


auto_so100 --> Run auto mode where in the order of check in to check out to check back. If object is in a certain zone already it will go from there not strictly in that order but instead in that loop or sequence.

eval_so100 --> Single eval or any of the check in/out/back

Now I kind of want 2 new additional feature

Object box detection show:
- Before starting and after finishing a task. It should try and detect the object within the front camera frame. I'm not sure how this is done or works but its similar to the one that's online where there's a bounding box drawn where the object is encompasses and there's a label on the box (?) I want that. It should be similar to debug where it saves the image of before and after so we can see the difference, with the bounding box of course. And I'm not sure if this is a good idea but I only want to detect certain object, or in certain area (i.e. around the workspace zone). The work space zone is a bit right so we might want to expand it a little bigger in all 4 direction to allow for the detection. I was thinking it only looks within those zone or something, or is the detection of object full frame ok? Because at the end of the day the image I want saved of the before and after should be the full frame. How ever it is done should be smartly and thought out process.

Hand interrupt stop/resume:
- Since this will be set up as a demo in a show room. Having a physical robot move could be dangerous if humans are near it. So I was thinking of using sort of like the same object detection thing where if it detects a hand in frame (front camera). It should just pauses the program, period. It pauses everything, the movement etc... If the bot is in transit pauses it or something like that. If the hand is no longer detected, or is out of frame. It may continue/resume where it has left off.

What do you think? Is this design or feature good? Push back if my idea sucks, etc...

here was my thought process

Great ideas — let me give you my honest take before diving into code.
Feature 1 (Bounding Box Snapshots): Solid idea, and totally doable with your existing HSV pipeline — no need for a heavyweight model like YOLO. I'll use zone-based detection with padding (not full frame) because full-frame detection would pick up colored objects in the background, hands, etc. The full image IS saved though, just with detection limited to the workspace area. Clean and intentional.

Feature 2 (Hand Pause): Great safety idea for a showroom. A few design realities to push back on slightly:
"Resume where it left off" during transport is doable — the lerp loop can hold position then continue
"Resume where it left off" during SEARCHING is NOT safe — VLA chunks go stale. Better to reset the ensemble and grasp timer after a pause. The arm will just re-attempt the grasp.
I'll use MediaPipe for hand detection (much more reliable than skin-color HSV, especially under showroom lighting). It's an optional dep with a HSV fallback.
Detection refresh during transport: I'll sample the front camera every ~10 lerp steps (~3Hz) so the hand detector always has fresh frames even when the main loop isn't reading observations.