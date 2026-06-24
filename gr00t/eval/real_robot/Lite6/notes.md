# `LITE6_HYBRID_CV_ARCHITECTURE.md`

## 1. System Overview & Philosophy
This project operates a UFACTORY Lite 6 robotic arm for pick-and-place tasks. It mimics the state-machine logic of the previous **SO ARM 101** system (supporting NLP-driven `eval` tasks and continuous `auto` tasks like *Check In*, *Check Out*, and *Check Back*). 

However, it **deprecates End-to-End Neural Policies (VLAs)**. VLAs traditionally cause heavy jitter, require GPUs, and struggle with millimeter precision. Instead, this system uses a **Deterministic Hybrid Computer Vision Pipeline**, or also known as Image-Based Visual Servoing (IBVS).

### The Hybrid Approach:
1. **Global Phase (Top-Down Camera):** Fast, math-based localization using a Homography Matrix to map pixels to real-world millimeters. Moves the arm rapidly to the general vicinity.
2. **Local Phase (Wrist Camera):** Real-time Visual Servoing (Proportional Control) to perfectly center the gripper over the object, eliminating jitter and ensuring a flawless grasp.

---

## 2. Setup & Calibration (The Homography Matrix)
Before the system can run, the top-down camera's perspective must be mapped to the robot's physical 2D Cartesian plane ($X, Y$). 

**How it is done:**
1. A ChArUco board is placed flat on the table.
2. The user runs the interactive calibration script (`script/lite6_homography.py`).
3. The user clicks 4 corners on the camera image, capturing pixel coordinates $(u, v)$.
4. The user physically moves the Lite 6 gripper to touch those same 4 corners, recording robot base coordinates $(X_{mm}, Y_{mm})$.
5. OpenCV calculates a 3x3 transformation matrix ($H$) using `cv2.findHomography`.
6. This matrix is saved as `homography_matrix.npy`. Any future pixel coordinate $(u, v)$ can now be multiplied by $H$ to instantly yield the exact robot $X, Y$ coordinate.

---

## 3. The Core Pipeline: Step-by-Step

Every time the robot attempts to pick an object, it executes the following **Finite State Machine (FSM)**.

### STEP 1: NLP parsing (Eval Mode Only)
* **Action:** The system receives a natural language string (e.g., `"check in red cube"`).
* **Process:** Regex parsing extracts the `task_type` (check_in) and the `target_object` (red cube).

### STEP 2: The Global Snapshot (State: `SCANNING`)
* **Action:** The top-down camera takes a single snapshot of the workspace.
* **Process:** OpenCV applies HSV color thresholding (e.g., looking for red) to isolate the object. It draws a contour around the blob and calculates the geometric centroid/moments to find the exact center pixel: $(u, v)$.

### STEP 3: Coordinate Translation (State: `GLOBAL_APPROACH`)
* **Action:** The system translates pixels to millimeters.
* **Process:** The pixel $(u, v)$ is passed through the Homography Matrix. The output is a real-world robot coordinate: e.g., $X = 310.5\text{mm}, Y = -120.2\text{mm}$.
* **Execution:** The Lite 6 uses its native, buttery-smooth trajectory planner to move to $(X, Y, Z_{safe})$. $Z_{safe}$ is a predefined height (e.g., $150\text{mm}$) ensuring the arm travels *above* all obstacles.

### STEP 4: Visual Servoing (State: `FINE_GRASP`)
* **Action:** The arm is now hovering above the object, but due to slight calibration errors or object height, it might be off by a few centimeters. The **Wrist Camera** takes over.
* **Process:**
  1. The wrist camera continually captures frames at ~30fps.
  2. It detects the target object and finds its center pixel.
  3. It calculates the **Error**: The distance between the object's center pixel and the *exact center of the camera frame*.
  4. **P-Controller:** It multiplies this pixel error by a proportional gain ($K_p$, e.g., 0.08) to get a tiny movement delta in millimeters ($dX, dY$).
  5. The arm moves by $(dX, dY)$ and takes another picture.
  6. This loop repeats rapidly until the error is $< 10$ pixels (the object is dead center).
* **Execution:** Once centered, the robot lowers straight down on the Z-axis to $Z_{grasp}$ (e.g., $30\text{mm}$), closes the gripper, and lifts back to $Z_{safe}$.

### STEP 5: Scripted Transport (State: `TRANSPORT`)
* **Action:** The robot moves the object to its final destination based on the task type.
* **Process:** The system looks up the hardcoded $(X, Y)$ drop-off coordinates for the specified zone, moves there at $Z_{safe}$, descends, opens the gripper, and returns to the `HOME` position.

---

## 4. Operational Modes (Task Semantics)

Just like the SO ARM 101 codebase, this system organizes logical tasks into physical zones. The zones are pre-defined physical coordinates mapped in `modules/constants.py`.

### The Zones:
* **Check-In Zone:** The intake area where new items arrive.
* **Storage Zone:** The main warehousing/holding area.
* **Check-Out Zone:** The outbound area where items are dispatched.

### Task Dictionary:
* **`check_in`**: Object is picked from *Check-In Zone* (or anywhere on the table) and transported to the *Storage Zone*.
* **`check_out`**: Object is picked from the *Storage Zone* and transported to the *Check-Out Zone*.
* **`check_back`**: Object is returned from the *Check-Out Zone* to the *Check-In Zone*.

---

## 5. Execution Scripts

### `eval.py` (Instruction-Driven)
Designed for single-shot, directed operations. 
* **Input:** Command line argument (e.g., `python eval.py --instruction "check in blue cube"`).
* **Flow:** Parses the instruction, runs the FSM once for that specific object and task, returns home, and exits.

### `auto.py` (Continuous Loop)
Designed for endless, autonomous workspace sorting.
* **Input:** None (`python auto.py`).
* **Flow:** 
  1. Sweeps the top-down camera frame for *any* known object in the `KNOWN_OBJECTS` list.
  2. If it sees a target, it automatically assigns a default task (e.g., `check_in`), runs the FSM to move it to storage.
  3. Returns home, sleeps for 3 seconds, takes another picture, and repeats endlessly until terminated via `Ctrl+C`.

---

## 6. File Architecture Overview
If expanding or debugging the code, refer to this structure:

* `README.md` - Setup and run instructions.
* `eval.py` - Single-shot NLP evaluation script.
* `auto.py` - Continuous autonomous script.
* `script/lite6_homography.py` - Standalone interactive calibration tool.
* `modules/`
  * `constants.py` - Single source of truth (IP, camera IDs, Z-heights, Zone X/Y coords, Color thresholds).
  * `system.py` - OS signals (Graceful Ctrl+C exiting).
  * `nlp.py` - Regex string parsing for Eval mode.
  * `robot.py` - Wrapper class for `xArmAPI` (connect, move, gripper control).
  * `vision.py` - Computer Vision logic (Homography math, color masking, contour tracking, Visual Servoing P-Controller loop).
  * `fsm.py` - The Finite State Machine bridging Vision and Robot commands.

---

## 7. Baseline Logic Alignment & Critical Safety Features (SO ARM Match)

To maintain absolute behavioral consistency with the original SO ARM codebase, the orchestration and safety logic must be ported exactly. The only differences are how the physical arm is commanded (Cartesian `xarm-python-sdk` vs. joint-space degrees) and the camera translation setup. 

The following critical system-level features must be implemented precisely as they were in the SO ARM setup:

### 7.1. Hand Detection & Safety Motion Freeze
*   **The Problem:** Humans reaching into the workspace during automatic operation is a high-risk safety concern. 
*   **The Baseline Logic:** We utilize a multiprocessing `SafetyMonitor` running Google MediaPipe Hands in a separate Python process to bypass the GIL (Global Interpreter Lock). 
*   **The Mechanism:**
    *   The main control loop passes downscaled frames $(256 \times 256)$ lazily to a shared-memory queue `frame_queue` (with a max capacity of 1 to prevent queue latency buildup).
    *   A shared boolean flag `hand_detected` is updated in $O(1)$ real-time.
    *   At **every single step** of Cartesian travel or visual servoing, the robot queries this flag.
    *   If `hand_detected` is `True`, the arm immediately halts, enters a loop sending its *current* pose continuously, and pauses all execution.
    *   The moment the hand leaves the frame, the FSM automatically clears any stale commands and safely resumes execution from where it paused.

### 7.2. Graceful Shutdown & Signal Handling
*   **The Mechanism:** Standard Ctrl+C (SIGINT) or SIGTERM must not result in the Lite 6 dropping payloads or locking up in an awkward position.
*   **The Logic:**
    *   Signal handlers catch exit requests and raise a soft-stop flag. 
    *   If a movement is underway, the robot finishes the localized travel step safely, immediately halts the FSM, releases the object if safely hovering, executes a controlled transit back to the standard `HOME_POSE`, disconnects the API cleanly, and removes any `/tmp` flags.
    *   If a user hits Ctrl+C a *second* time (double-tap), the system triggers a hard stop, bypassing the home sequence to act as a secondary software-level emergency stop.

### 7.3. Pre-Check Phase & Failure Shake
*   **The Mechanism:** Before the robot even begins moving, it must verify the target actually exists.
*   **The Logic:**
    *   In both `eval` and `auto` modes, the robot transitions to `PRE_CHECK` first.
    *   The top-down camera masks for the target color inside the designated source zone.
    *   The task only proceeds if the detected color pixel mass exceeds `FRONT_MIN_PRESENCE_PX` (default: 1500 px).
    *   If the target object is missing, the robot remains in place, executes a fast, physical "no-no shake" (rotating the wrist back and forth to visually signal failure to the user), and aborts the FSM without trying to pick up empty space.

### 7.4. Before/After Snapshot Verification
*   **The Mechanism:** High-accuracy logging and auditing.
*   **The Logic:**
    *   Before executing any FSM step, the top-down camera saves a baseline image: `snapshot_eval_front_before.jpg`.
    *   After the drop-off is complete and the arm returns home, another snapshot is taken: `snapshot_eval_front_after.jpg`.
    *   A visualization function overlays white bounding boxes around the active check-in/storage/check-out zones and draws a green bounding box around the detected target object (using HSV contours) to verify successful task execution.

### 7.5. FSM Error Recovery & Retries
*   **The Mechanism:** Handling unexpected failures (e.g., the object slips, or visual servoing loses target tracking).
*   **The Logic:**
    *   If visual servoing fails to lock onto the target within `vla_timeout` (default: 15 seconds), the search retries counter increments.
    *   If the retry limit (`max_retries`) is not exceeded, the arm returns to `HOME_POSE`, resets its state, approaches the workspace again, and restarts the visual servoing loop.
    *   If the visual servoing fails repeatedly, the robot drops back to `HOME_POSE` and halts in a `FAILED` state to prevent endless looping or erratic behaviors.