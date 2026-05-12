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
    --dataset.repo_id=so101/cube_man \
    --dataset.num_episodes=10 \
    --dataset.single_task="Check in yellow cube" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=8 \
    --dataset.reset_time_s=20 \
    --resume=true




------



This document serves as a comprehensive development log for the **GR00T SO100 Autonomous Task Completion & Termination System**.

---

# Project Report: Autonomous Task Termination for GR00T-VLA

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

### **Color Signatures (Target HSV)**
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
2.  **Lerp:** Linearly interpolate from `current` to `home`.
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

**Result:** A fully autonomous robot that understands its task, performs it, and stops precisely when the goal is achieved without human intervention or hardware stress.






Next Phase:


Pivoting often referred to in robotics as moving from **End-to-End Control** to **Hybrid Modular Control**. **Proposal: Transitioning to a Hybrid Modular Architecture for Increased Reliability**

Could we significantly increase task success rates by narrowing the GR00T-VLA’s responsibility to high-entropy 'Approach and Grasp' maneuvers while delegating low-entropy 'Transport and Placement' to programmatic trajectories? Currently, the model exhibits cumulative error over long horizons, leading to inconsistency during the transport phase—a portion of the task that requires high spatial precision but low visual reasoning. By treating the VLA as a 'Visual Servoing' specialist, we would leverage its ability to handle varied cube placements to achieve a secure grasp; once the gripper's state indicates a successful hold, the system would hand off control to a scripted, interpolated kinematic path (A-to-B). This approach is backed by the principle of **Dimensionality Reduction in Action Space**: by removing the transport phase from the neural network's burden, we eliminate the risk of 'model drift' during movement, ensuring that once an object is secured, the final delivery is mathematically guaranteed and executionally smooth.

With this new change I want the following

`obs["lang"] = cfg.lang_instruction ` will now only assigned the color so `obs["lang"]` is only assigned red/blue/yellow. Since the task we will be training the robot on will only be about identifying and grasping color cubes. while the lang_instruction full will still be parsed to determined whether it is a check in or check out.

The robot (SO ARM) now will call VLA to identify and grasp the appropriate color cube. The scripted part will take over by moving the arm up vertically then over to the target location, lowering it and releasing the cube, then vertically going up again (to avoid collisions). Then back to the home position in this order.

Now the motion will be relatively the same, but the angle would be different because of the area. For example, from Check In to Storage. We run VLA so it go to the cube for grasping. Then once we confirmed grasp. We will take the current position (angle of all motors), and update lerp it to go vertially up (so most of the motor except for shoulder_pan.pos will change, since shoulder_pan.pos is what's keeping the arm in that area and direction). Then move shoulder_pan.pos to rotate to the storage area. Lower it back and drop something of that nature. We will have to define a specific set of waypoints. Which are below (Note that some steps and ideas are abstract and left out but the general concept and flow is the same as what we have before)


Lets say: Check in red cube

obs["lang"] should only be "red"
Task type is check in from "Check in" so from **Check-in Area** to the **Storage Area**

1. move to home
2. snapshot
3. run VLA to identify and grasp the red cube (inference)
    3.5 detection := reached + grasped (to be discussed later), break and switch to scripted motion

From current position from VLA / GR00T after confirmed grasped
```
{
    "shoulder_pan.pos": __,
    "shoulder_lift.pos": -__,
    "elbow_flex.pos": __,
    "wrist_flex.pos": __,
    "wrist_roll.pos": __,
    "gripper.pos": __
}
```
lerp to "vertically up/picking up" position
```
{
    "shoulder_pan.pos": __,     <-- Keep the same
    "shoulder_lift.pos": -__,
    "elbow_flex.pos": __,
    "wrist_flex.pos": __,
    "wrist_roll.pos": __,
    "gripper.pos": __
}
```


Now this begs the question. How do we detect when the robot has successfully grasped the cube? This is a hard one because we are unsure of the exact moment when it has reached the cube and has a secure grip onto it.