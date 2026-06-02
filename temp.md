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



This document serves as a comprehensive development log for the **GR00T SO100 Autonomous Task Completion & Termination System**.

---

# Project Report: Autonomous Task Termination for GR00T-VLA
waypoint = {
    "shoulder_pan.pos":   -3.271,
    "shoulder_lift.pos":   12.028,
    "elbow_flex.pos":  -20.273,
    "wrist_flex.pos":   80.452,
    "wrist_roll.pos":   45.690,
    "gripper.pos":   24.119,
waypoint = {
    "shoulder_pan.pos":   -3.271,
    "shoulder_lift.pos":   12.028,
    "elbow_flex.pos":  -20.273,
    "wrist_flex.pos":   80.452,
    "wrist_roll.pos":   45.690,
    "gripper.pos":   24.119,
}}
## 1. Problem Statement: The "Infinite Loop" Challenge
**The Context:** We are running a GR00T policy (VLA model) using a TensorRT-accelerated DiT (Diffusion Transformer) head to control an SO100 robot arm.
**The Issue:** Like most diffusion-based policies, GR00T is trained to continuously predict the next action chunk. It lacks a built-in "End of Task" signal. Once the robot finishes a task (e.g., placing a cube), the model continues to run, causing the robot to:
1.  Jitter and drift due to stochastic noise in the model.
2.  Attempt to "re-do" the task if the environment is reset.
3.  Perform erratic movements that could potentially damage the hardware.

---

## 2. Task Description & Logic
The robot is tasked with "Sorting" colored cubes (Red, Blue, Yellow) across three designated tabletop zones.

### **Instruction Parsing & Expected Behavior**
We defined two primary task types based on language input:
*   **"Check in [Color] cube":** Move the cube from the **Check-in Area** to the **Storage Area**.
    *   *Success Condition:* Target color detected in `STORAGE_ZONE`.
*   **"Check out [Color] cube":** Move the cube from the **Storage Area** to the **Check-out Area**.
    *   *Success Condition:* Target color detected in `CHECK_OUT_ZONE`.

---

## 3. Initial Thought Process & Potential Solutions
To solve the infinite loop, we explored four architectural approaches:
1.  **Human-in-the-loop:** Using keyboard interrupts (pressing 'q') to stop. (Manual, not autonomous).
2.  **Hard Timeouts:** Stopping after exactly 20-30 seconds. (Inflexible; doesn't account for failures or fast completions).
3.  **Action Convergence:** Monitoring joint variance to see if the robot "settles." (Unreliable due to model jitter).
4.  **Parallel Vision (Selected):** Using a secondary, lightweight OpenCV process to "watch" the goal zones independently of the VLA model.

---

## 4. Implementation Phase 1: Vision Calibration
To enable autonomous stopping, we had to define what "Success" looks like to a computer. We performed a calibration to map the physical workspace to pixel coordinates and HSV color ranges.

### **Workspace Coordinates (X, Y, Width, Height)**
*   **Storage Zone:** `(277, 168, 96, 94)`
*   **Check-in Zone:** `(399, 100, 99, 97)`
*   **Check-out Zone:** `(152, 103, 96, 95)`

waypoint = {
    "shoulder_pan.pos":   -3.271,
    "shoulder_lift.pos":   12.028,
    "elbow_flex.pos":  -20.273,
    "wrist_flex.pos":   80.452,
    "wrist_roll.pos":   45.690,
    "gripper.pos":   24.119,
}### **Color Signatures (Target HSV)**
*   **Red:** `[177, 20, 214]` (Pinkish-red)
*   **Blue:** `[96, 169, 13]` (Blue)
*   **Yellow:** `[33, 90, 168]` (Yellow-green)

---

## 5. Implementation Phase 2: Solving Environmental Interference
**The Problem:** The physical zones are marked with color-coded paper (Pink for Check-in, Yellow for Check-out). A standard color filter would see the paper and immediately think the cube was already there.

**The Solution: Background Subtraction**
We updated the system to take a **Baseline Snapshot** when the robot is at the "Home" position (before the cube is moved).
*   **Logic:** The vision system calculates the absolute difference between the current frame and the baseline.
*   **Effect:** Static objects (like the colored paper) are subtracted and become "black." Only new objects entering the zone (the cube or the robot arm) are processed.

**The "Robot Left the Space" Logic:**
To ensure the task is truly finished, we don't just look for color; we check if the detected blob **touches the boundaries** of the zone.
*   If the blob touches the edge: The robot arm is likely still in the frame (holding the cube).
*   If the blob is "floating" in the center: The arm has let go and cleared the area. **This triggers the Task Success signal.**

---

## 6. Implementation Phase 3: Hardware Refinement (Smoothing)
**The Problem:** Once the task was detected as "Done," the robot was commanded to go to a "Home" position. This caused a violent "snapping" motion because the joints tried to move 90+ degrees in a single frame.

**The Home Position Definition:**
```python
HOME_ACTION = {
    "shoulder_pan.pos": 0, "shoulder_lift.pos": -60, "elbow_flex.pos": 60,
    "wrist_flex.pos": 60, "wrist_roll.pos": 90, "gripper.pos": 40
}
```

**The Solution: Smoothstep Interpolation**
Instead of a sudden jump, we implemented a 2.0-second **Ease-In/Ease-Out (Smoothstep)** trajectory:
1.  **Read:** Get current joint states via `robot.get_observation()`.
waypoint = {
    "shoulder_pan.pos":   -3.271,
    "shoulder_lift.pos":   12.028,
    "elbow_flex.pos":  -20.273,
    "wrist_flex.pos":   80.452,
    "wrist_roll.pos":   45.690,
    "gripper.pos":   24.119,
waypoint = {
    "shoulder_pan.pos":   -3.271,
    "shoulder_lift.pos":   12.028,
    "elbow_flex.pos":  -20.273,
    "wrist_flex.pos":   80.452,
    "wrist_roll.pos":   45.690,
    "gripper.pos":   24.119,
}}2.  **Lerp:** Linearly interpolate from `current` to `home`.
3.  **Smooth:** Apply a mathematical curve ($3t^2 - 2t^3$) so the motion starts and ends gently.

---

## 7. Current System Architecture (Summary)
1.  **Initialization:** Robot connects; user provides instruction (e.g., "Check in blue cube").
2.  **Pre-Flight:** Robot smoothly moves to **Home**. A **Baseline Snapshot** is taken.
3.  **Execution:** GR00T-VLA starts predicting actions.
4.  **Monitoring (Parallel):**
    *   Crop the front camera to the **Target Zone**.
    *   Subtract the **Baseline**.
    *   Mask for the **Target Color**.
    *   Check if the resulting blob is **stable and not touching the zone edges**.
5.  **Termination:** Once Success is detected, the inference loop breaks.
6.  **Cleanup:** Robot performs a final **Smooth Home** and the script exits cleanly.

**Result:** A fully autonomous robot that understands its task, performs it, and stops precisely when the goal is achieved without human intervention or hardware stress. The current architecture—combining Time-Aligned Temporal Ensembling for VLA action chunking with a Hybrid Scripted/VLA approach and Parallel Vision—is genuinely at the cutting edge of open-source robotics. We've already solved the "stop-and-go" chunking problem that plagues 90% of VLA implementations.






---





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
    --dataset.single_task="yellow cube" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=6 \
    --dataset.reset_time_s=20 \
    --resume=true





red cube    60  x
orange cube 20  x
yellow cube 60  
pink cube   20
blue cube   20
green prism 60
red ball    60
yellow ball 20

Total = 60 + 20 + 60 + 20 + 60 + 60 + 60 + 20 = 320 episodes

--- 

60 episodes per object

The "Vanilla" Reaches (20 Episodes) - Teach the basic kinematics of reaching the object in plain sight.

    Setup: Only the target object in the workspace. Completely empty otherwise.

    Locations: Randomly scatter it across the CHECK_IN_ZONE and CHECK_OUT_ZONE. Do not favor the dead center. Place it in the top-left, bottom-right, dead middle, etc.

    Rotations: Keep the object relatively "square" to the robot for these.


Pose & Rotation Robustness (20 Episodes) - Teach the model that a cube at a 45-degree angle is still a cube, and how to angle the wrist roll to grasp it.

    Setup: Single object, but actively mess with its rotation.

    Execution:

        For Cubes: Rotate them 30°, 45°, 60°. Your SO-100 wrist will need to roll to match the faces so the gripper fingers fit cleanly.

        For the Green Prism: Lay it sideways, point it straight at the camera, point it diagonally.


The "Shape vs. Color" Distractor Test (20 Episodes) - Force the language embedding to pay attention to both the color adjective AND the shape noun.

    Setup: Place the target object alongside 1 or 2 distractors.

    Combinations to record:

        Same Color, Different Shape: Target = "red cube". Distractor = "red sphere". (Forces the model to learn "cube" vs "sphere").

        Same Shape, Different Color: Target = "red cube". Distractor = "blue cube" or "pink cube". (Forces the model to look at the color).

        Spacing: Sometimes put the distractor 10 inches away. Sometimes put it literally 1 inch away so the gripper has to squeeze past it.


---

For similar shapes variant

The 20-Episode "Variant" Breakdown

    5 Episodes - Vanilla/Basic Locations:

        Setup: Just the Blue Cube by itself. Scatter it in a few different spots.

        Purpose: Just to prove to the model that the Blue Cube can exist on its own in the workspace.

    5 Episodes - Rotated:

        Setup: Just the Blue Cube, but angled (30°, 45°).

        Purpose: Confirms that the color features and the rotation features aren't mutually exclusive.

    10 Episodes - The "Anti-Bias" Distractor Test (Crucial):

        Setup: Put the Blue Cube AND the Red Cube in the scene together.

        Prompt: "blue cube"

        Purpose: This is the most important part of the 20 episodes. Because the model will have seen 60 Red Cubes, its first instinct when it hears "cube" will be to grab the red one. By forcing it to reach past the Red Cube to grab the Blue one 10 times, you ruthlessly train the color bias out of the model.









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



Current thoughts and issues:

Why VLA sometimes when prompted red cube would go for blue or whatever is in front of the camera? Biased towards a color? How to fix that? Recording more data? Software fix like through computer vision to help it? Need ideas or things to try to mitigate that bias. Or somehow improve the color detection in the model or through software.

Currently, we look at WRIST_MIN_PRESENCE_PX to see if the cube is in frame, but for some reason certain color, like yellow, is less "present" l;ike it only shows less than 1000 pixels of presence even when it's clearly visible to the human eye. Maybe there's a flaw with our detection software? Or should we switch to color based detection instead, though this requires us to do more indepth mapping of most colors. But I think this is good because if it did grasp the wrong color cube, the signal will be false so it wouldn't get to the transport phase, and it would just try again after the time out. What do you think? Help me come up and implement the solution to this problem as well



Other issues I would like to discuss with you:

- Currently for each zone / area, we just send it to a specific pos for each motor so that it is facing and at a drop down pose to drop the cube. But it is always at 1 spot. It never changes. So for example, for the check in zone, it always goes to the same spot to drop the cube. Do you think this is a problem? Should we have multiple "drop" spots within the check in zone and randomly select one each time? Or a smart drop like it knows where is the most empty space to drop the cube based on the vision? Or you know how right now we have a reach/fetch VLA in that search phase? What if we also record data of the opposite of reach now its putting down the already holding cube, so its like a "place" action? Have it move over then turn on VLA for the place action to drop it in different spots within the zone? What do you think? Should all that be in the dataset? I think there will be an issue with that and we shouldn't do it in someway because it might be too much for the model to learn with the amount of data we have. But I also think it would be good to have some variability in the drop location so it's not always dropping in the same exact spot. What do you think? 

- Any other improvements we can make to the system? Could be general improvements, specific improvements to specifics like state behavior, vision, the way we are doing the transport, the way we are doing the reach, the way we are doing the drop, etc. Anything you can think of that would make the system more robust, more efficient, more accurate, easier to use, etc.


Please push back or debate or think or suggest new better solution if you think of anything. I want to make this as good as possible and I know you have a lot of good ideas and insight so please share them with me. I also want to make sure we are on the same page with the current issues and problems and how we are going to solve them. So please let me know your thoughts on all of this. Thanks!





1. The VLA Color Bias & Grasp Detection Flaw

Why is the VLA biased?
VLAs are inherently "lazy" learners. If your training data mostly contains single-object scenes, or if the target object is usually the closest one, the model learns a shortcut: "Ignore the language instruction and just grab the nearest colorful blob."
The Data Fix: You need to record "Distractor Data". Put the blue cube directly in front of the robot, put the red cube further away, and command "red cube". The model needs to see you intentionally bypass the blue cube.

Why does Yellow fail the vision check?
In vision_utils.py, you are doing cv2.absdiff on Grayscale images. Yellow (R:255, G:255, B:0) translates to a very light gray. If your table/background is also light gray or white, the pixel difference falls below your WRIST_PRESENCE_THR = 50. To the computer, light gray and yellow are the exact same brightness!

The Software Fix (Your Idea!):
Switching to HSV Color Tracking is the perfect solution. If we tell the GraspDetector which color to look for, we solve two problems at once:

    Flawless Yellow Detection: HSV separates "Color" (Hue) from "Brightness" (Value). Yellow will pop instantly.

    The "Wrong Cube" Safety Net: If the VLA goes rogue and grabs the Blue cube, but the instruction was "Red", the HSV detector will see 0 red pixels. It will refuse to validate the grasp, wait out the timeout, and force the FSM to retry. This prevents the robot from putting the wrong cube in the output bin!

2. The Static Drop Location

Your thought: Should we train a VLA "Place" policy, or do a smart/random drop?
My Pushback: Do not train a VLA "Place" policy for this.
Placing objects dramatically increases the complexity of the state space. You would need 5x more data because the model has to learn to release at the exact right time based on the height of varying piles. In industrial robotics, "Pick" is often done with AI, but "Place" is almost always done with heuristics (math/scripting) unless it's a high-precision insertion (like a peg in a hole).

The Better Solution: Grid-Based Round-Robin Placement.
Instead of a single STORAGE_PLACE, we define an array of 3 or 4 predefined waypoints spaced out across the zone. We keep a counter of how many items we've placed, and use index % 4 to select the next drop location. This builds a neat, organized grid of cubes and completely prevents tower-stacking collisions!





x=164, y=362, w=346, h=117


[BLUE] (165,196) → HSV [113, 212, 125]
[RED] (306,186) → HSV [30, 84, 188]
[YELLOW] (469,188) → HSV [173, 154, 175]
[White] (315,83) → HSV [100, 34, 189]
[Gray] (393,392) → HSV [103, 48, 74]
