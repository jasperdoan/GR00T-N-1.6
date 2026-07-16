# SO100 Hybrid Modular Robotic System: Technical Architecture & Operational Report

## Executive Summary
The SO100 robotic system is an advanced, hybrid-modular pipeline designed for autonomous object manipulation (pick-and-place) based on natural language instructions. The architecture fuses a **Vision-Language-Action (VLA) neural policy (GR00T)** for dynamic grasping with **deterministic algorithmic motion** for reliable transport. 

The system operates under a rigid **Finite State Machine (FSM)** ensuring step-by-step validation, complete with multi-layered Computer Vision (CV) verification, dynamic trajectory smoothing, temporal ensembling for neural execution, and highly robust local error-recovery behaviors. 

This document serves as a comprehensive technical breakdown of the system’s capabilities, operational pipeline, and fail-safes.

---

## Table of Contents
1. [Natural Language Processing & Command Routing](#1-natural-language-processing--command-routing)
2. [Vision & Perception Subsystem](#2-vision--perception-subsystem)
3. [Kinematics & Motion Primitives](#3-kinematics--motion-primitives)
4. [Asynchronous Neural Execution (Temporal Ensembling)](#4-asynchronous-neural-execution-temporal-ensembling)
5. [Operational Lifecycle (FSM Execution Steps)](#5-operational-lifecycle-fsm-execution-steps)
6. [Fault Tolerance & Error Recovery](#6-fault-tolerance--error-recovery)

---

## 1. Natural Language Processing & Command Routing
The system initiates by passing the human instruction through the `nlp_parser.py` module.

*   **Synonym & Task Routing**: It uses regex word boundaries (`\b`) to prevent substring matching errors (e.g., preventing "bored" from triggering "red"). It scans for task synonyms (e.g., "check out", "retrieve", "get") to determine the `task_type`.
*   **Zone Mapping**: Based on the `task_type`, the parser maps a `source_zone` and a `target_zone`. 
    *   *Check In*: Source = `CHECK_IN_ZONE`, Target = `STORAGE_ZONE`
    *   *Check Out*: Source = `STORAGE_ZONE`, Target = `CHECK_OUT_ZONE`
*   **Target Identification**: Identifies the target object (e.g., "red cube") from a strict list of `KNOWN_OBJECTS`. It includes a fallback mechanism to dynamically parse valid color/shape combinations if phrased unconventionally.

---

## 2. Vision & Perception Subsystem
The system leverages a dual-camera setup (Front and Wrist), relying heavily on highly calibrated HSV color space analysis and morphological operations to eliminate environmental noise.

### 2.1 Hardware Calibrations & Saturation Boosting
Camera auto-exposure (e.g., on ZED cameras) often washes out colors, causing pinks and oranges to bleed into the "red" spectrum. To counteract this:
*   **Algorithmic Saturation Enhancement**: The `_enhance_saturation()` function multiplies the HSV saturation channel by a scalar (`factor = 1.4` for front, `1.4` for wrist) before masking.
*   **Strict Color Bounds**: The red HSV boundaries are heavily tightened (Hue: `0-5` and `170-180`, Min Saturation: `100`) to absolutely reject orange and pink.

### 2.2 Front Camera: Pre-Check & Verification
*   **Target Presence Pre-Check**: Before the robot moves, it crops the front camera feed to the instruction's `source_zone`. It applies the saturation boost and HSV mask to count color pixels. It requires at least `FRONT_MIN_PRESENCE_PX` (500px) to proceed.
*   **Task Success Verification**: Post-transport, the system crops the target zone and performs background subtraction against a baseline snapshot taken at boot. It applies a Gaussian blur, binary thresholding (`diff_threshold = 25`), morphological opening (5x5 kernel noise removal), and checks if any resulting contour exceeds `MIN_BLOB_AREA_PX` (100px).

### 2.3 Wrist Camera: Grasp & Transit Monitoring
The Wrist camera operates a 4-signal (A-D) logic system to definitively confirm an object is securely held.
*   **Signal A (Time-Gating)**: Ignores visual checks during the initial `VLA_GRASP_MIN_TIME` (1.0s) or transit lift-off (0.5s) to allow for mechanical settling.
*   **Signal B (Visual Presence)**: Extracts a strictly defined Region of Interest (`WRIST_GRASP_ROI`). Applies the target's HSV mask and requires `WRIST_MIN_PRESENCE_PX` (10,000px) of the correct color.
*   **Signal C (Mechanical Gripper)**: Verifies the physical gripper servo state is equal to or tighter than `GRIPPER_TRANSPORT_MIN` (70.0°).
*   **Signal D (Sequential Confirmation)**: A state must be maintained for `WRIST_CONFIRM_FRAMES` (3 consecutive frames) to trigger a "Success" or "Drop" event, filtering out split-second camera glares.
*   **Mid-Transit Blob Tracking**: Once a grasp is confirmed, the system captures a locked snapshot of the wrist ROI. A heavy 21x21 Gaussian Blur is applied. During transit, current frames are blurred and subtracted from the locked frame. If the pixel difference exceeds a 50% ROI shift, a mid-air drop is detected.

---

## 3. Kinematics & Motion Primitives
The `motion.py` module handles the deterministic physical execution.

### 3.1 Trajectory Smoothing
*   **Quintic Smoothstep (`lerp_to_waypoint`)**: Used for linear transitions (e.g., returning home, opening gripper). Standard lerping creates mechanical "snaps" at waypoints. Quintic smoothstep $(6t^5 - 15t^4 + 10t^3)$ zeroes out both velocity *and acceleration* at the start and end of the move, drastically reducing wear on the SO100 servos.
*   **Quadratic Bezier Arcs (`arc_trajectory`)**: Used for fluid transport. Instead of moving up, stopping, moving over, and stopping, the arm sweeps in a continuous, non-stop curve from the starting point, pulling towards a `LIFT_OVERRIDE` apex, and gently landing at the destination.

### 3.2 Dynamic Grid Offsetting
To prevent objects from stacking on top of each other and crashing the system, the `scripted_transport` utilizes a `drop_counter` to offset the placement in a 2x2 grid (Center, Right +5°, Left -5°, Forward +5° on the wrist flex). Additionally, a $\pm 2.0°$ uniform random noise is applied to placement joints to organically scatter the objects.

### 3.3 Visual Telemetry (Failure Shake)
If the FSM determines an unrecoverable visual failure (e.g., no object in the zone), the robot calls `execute_failure_shake()`, panning and rolling the wrist $\pm 20°$ three times to mimic a human "head shake" indicating to the operator that the target is missing.

---

## 4. Asynchronous Neural Execution (Temporal Ensembling)
VLA inferences block the main thread and take time (causing jitter). The `AsyncPolicyRunner` completely masks this latency using **Time-Aligned Temporal Ensembling (TATE)**.

1.  **Background Inference**: Policy requests are pushed to a background thread to maintain a rigid 30 Hz control loop on the FSM.
2.  **Time Alignment**: Because inference takes time, an action chunk requested at Step 0 but arriving at Step 2 is inherently outdated. TATE dynamically skips the first two actions in the newly arrived chunk to perfectly realign the robot's timeline.
3.  **Exponential Averaging (Temperature Blend)**: The robot simultaneously executes overlapping action chunks. An older chunk provides stability, while a newly arrived chunk provides updated visual reactions. The actions are blended via a weighted average using `ensemble_temp = 0.1`.

---

## 5. Operational Lifecycle (FSM Execution Steps)
The `EvaluationFSM` orchestrates the task perfectly through a rigorous cycle:

1.  **Boot & Baseline**: Robot homes, takes a baseline image of the empty workspace and an empty wrist gripper.
2.  **State 1: `PRE_CHECK`**: 
    *   Crops front camera to the `source_zone`.
    *   Executes HSV/Saturation color counting.
    *   If absent: Shakes head, transitions to `FAILED`. 
    *   If present: Moves arm to hover over the source zone (`READY_POSITION`).
3.  **State 2: `SEARCHING` (The VLA Phase)**:
    *   Hands control to the GR00T neural policy.
    *   Constantly polls the `GraspDetector`. 
    *   If the VLA secures the object, locks the visual blob snapshot and transitions to `TRANSPORT`.
4.  **State 3: `TRANSPORT`**:
    *   Executes Bezier arc to the `target_zone`.
    *   Actively evaluates `monitor_callback` 30 times a second.
    *   Drops the object using the grid offset and pauses (`POST_DROP_PAUSE` = 0.35s) to let the object settle before arcing home.
5.  **State 4: `VERIFY`**:
    *   Takes a new front camera image.
    *   Runs background subtraction against the Step 1 baseline on the `target_zone`.
    *   If diff > 100px, transitions to `DONE`.

---

## 6. Fault Tolerance & Error Recovery
The system is built for autonomous continuity and defines a `max_retries` limit of **2 attempts** per task before total failure.

### 6.1 "Wrong Object" Detection
During `SEARCHING`, the VLA might successfully grab a blue cube when instructed to grab a red cube. The `GraspDetector` identifies that `gripper.pos` is closed, but the HSV pixel count for "red" is below 10,000. 
*   **Action**: The FSM instantly aborts the VLA, pops the gripper open, increments the retry counter, returns to the ready position to clear the camera view, and resets the VLA history to try again.

### 6.2 Timeouts
If the VLA gets stuck in a loop or cannot find the object for `vla_timeout` (15 seconds) during the `SEARCHING` phase:
*   **Action**: Execution is forcefully terminated. The retry counter is incremented, the robot goes all the way home to reset its field of view, and then returns to the ready position for another attempt.

### 6.3 Local Mid-Air Drop Recovery
If the robot's `TRANSPORT` phase throws a `GraspLostException` (verified by 3 consecutive frames of wrist ROI blob shifting > 50% or gripper unspooling):
*   **Action**: The robot halts the Bezier trajectory immediately. 
*   **Recovery Logic**: Instead of blindly returning home, the FSM transitions to `RECOVERY`. It captures the exact `shoulder_pan.pos` coordinate where the drop occurred. The arm resets to the safe ready elevation/pitch, but **overrides its default pan with the captured pan**. This forces the VLA to look directly down at the table exactly where it dropped the object, seamlessly resuming the `SEARCHING` state to pick it back up.