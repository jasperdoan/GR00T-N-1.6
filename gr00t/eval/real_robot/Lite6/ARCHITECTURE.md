# `LITE6_HYBRID_CV_ARCHITECTURE.md`

## 1. System Overview & Philosophy
This project operates a UFACTORY Lite 6 robotic arm for pick-and-place tasks. It implements the state-machine logic of the previous **SO ARM 101** system (supporting NLP-driven `eval` tasks and continuous `auto` tasks like *Check In*, *Check Out*, and *Check Back*). 

The system relies on a **Deterministic 3D Position-Based Visual Servoing (PBVS)** pipeline. It replaces the classic 2D Homography matrix and pixel-chasing Image-Based Visual Servoing (IBVS) with metric, camera-pose-independent 3D reasoning. By integrating aligned depth sensing with known camera intrinsics and hand-eye extrinsics, the robot represents the workspace directly in the 3D robot base frame.

### The Hybrid Approach:
1. **Global Phase (Top-Down Wrist View):** The arm parks at a high home position (`TOP_VIEW_POSE` at $Z = 450\text{mm}$). Color-segmented blobs are located in 3D by deprojecting depth or intersecting pixel rays with an estimated table plane. The arm then travels rapidly to the approach height (`APPROACH_Z_MM` at $Z = 250\text{mm}$).
2. **Local Phase (Wrist Camera PBVS):** Hovering above the target, the eye-in-hand camera continually measures the object's position in the robot base frame using per-blob depth deprojection. The controller commands the Tool Center Point (TCP) directly to the estimated base-frame target coordinates (damped and step-clamped) instead of chasing pixel offsets.

---

## 2. Geometry, Calibration & Extrinsics
The entire pixel-to-robot coordinate mapping is governed by projective geometry and a rigid 3D transform, removing the restriction of a static calibration pose.

### 2.1. Pinhole Camera Models
Using the color camera intrinsics $(f_x, f_y, c_x, c_y)$ provided by the Orbbec SDK, any pixel coordinate $(u, v)$ with a known depth $d$ (in mm) is deprojected to its camera-frame coordinates $(X_c, Y_c, Z_c)$ via:
$$X_c = \frac{(u - c_x) \cdot d}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot d}{f_y}, \quad Z_c = d$$

Conversely, any 3D camera-frame point is projected back to pixel space using:
$$u = \frac{X_c \cdot f_x}{Z_c} + c_x, \quad v = \frac{Y_c \cdot f_y}{Z_c} + c_y$$

### 2.2. Camera-to-Base Extrinsics
Because the camera is rigidly mounted to the wrist and the system keeps a fixed tool orientation (roll/pitch/yaw = $-180/0/0$) during all sensing operations, the relationship between a camera-frame coordinate $p_{cam}$ and the robot base-frame coordinate $p_{base}$ at any given TCP position $p_{tcp}$ is:
$$p_{base} = p_{tcp} + R \cdot p_{cam} + t$$
where $R$ is a constant $3 \times 3$ rotation matrix and $t$ is a constant $3 \times 1$ translation vector. These parameters are calibrated using SVD-based Kabsch rigid registration (`script/lite6_extrinsics.py`) over paired physical measurements and are stored in `data/extrinsics.npz`.

### 2.3. Dynamic Table Height Estimation
When the robot is parked at `TOP_VIEW_POSE`, the table plane $Z_{table}$ in the robot base frame is estimated dynamically from the aligned depth frame. The system samples the depth values corresponding to the background (determined as the 90th percentile of valid depths), deprojects them, and transforms them to the base frame. This estimated $Z$ plane provides:
*   A ground plane for depth-gating (removing table-level reflections).
*   A ray-intersection plane (`pixel_to_base_on_plane`) for depth-free 3D fallback localization.

---

## 3. The Core Pipeline: Step-by-Step

### STEP 1: NLP parsing (Eval Mode Only)
*   **Action:** Receives a natural language instruction (e.g., `"check out blue cube"`).
*   **Process:** Parses task intent and target color using predefined task aliases and HSV color range keys.

### STEP 2: Pre-Check & Target Verification (State: `PRE_CHECK`)
*   **Action:** Verifies target presence and maps workspace occupancy before initiating physical motion.
*   **Process:**
    1. The arm moves to `TOP_VIEW_POSE` to capture a global RGB-D image.
    2. The system filters the target color mask within the specific `source_zone` pixel ROI.
    3. Proceeds if the pixel mass exceeds `FRONT_MIN_PRESENCE_PX` (1500 px).
    4. Records the baseline target color count and pixel mass in the `target_zone` to allow verification of delivery later.
    5. Runs a multi-color census across the target zone to map occupied $(X, Y)$ coordinate points of existing cubes. This map is used for obstacle avoidance and drop-point planning.
    6. Conducts a **Free-Space Drop-Point Analysis** (see Step 5). If the target zone is determined to be fully occupied, the FSM flags this state early.

### STEP 3: Top-Down Localization (State: `SCANNING`)
*   **Action:** Selects a target cube and determines its initial coordinates.
*   **Process:**
    1. Locates all target-color blobs within the source zone ROI.
    2. If multiple candidates exist, selects one at random to distribute picks in multi-object tasks.
    3. **3D Localization:** Isolates the chosen blob's contour, applies a morphological close to fuse split fragments, and computes the median 3D camera-frame position of its top face (using a localized depth-based top-face mask). Converts this to base-frame coordinates.
    4. **Ray-Plane Fallback:** If depth is invalid, intersects the camera ray of the blob's 2D centroid with the plane $Z = Z_{table} + \text{OBJECT\_HEIGHT\_MM}$ to resolve the coordinate point.
    5. Verifies that the targeted $(X, Y)$ is inside `WORKSPACE` boundaries and reachable by the kinematics solver.

### STEP 4: Global Approach (State: `GLOBAL_APPROACH`)
*   **Action:** Rapid transit to the approach hover position.
*   **Process:** Commands the robot to move to $(X_{target}, Y_{target}, \text{APPROACH\_Z\_MM})$ at $200\text{mm/s}$.

### STEP 5: Position-Based Visual Servoing (State: `FINE_GRASP`)
*   **Action:** Closed-loop 3D centering and grasp execution.
*   **Process:**
    1. **Target Tracking:** An instance of `_TargetTracker` maintains a running estimate of the cube's base coordinates using an Exponential Moving Average (EMA) with $\alpha = 0.4$.
    2. **Frame Freshness Guard:** To prevent overshooting, the FSM polls the camera and discards frames captured *before* the previous movement ended.
    3. **Multi-Object Gating:** From the approach view, candidates are deprojected to the base frame. The tracker matches only the candidate within $45\text{mm}$ of the running target estimate and rejects any candidate within $40\text{mm}$ of already-delivered cubes. If depth drops out, it falls back to matching the nearest candidate within $120\text{px}$ of the projected target coordinate.
    4. **Metric P-Control:** Computes the distance error between the current TCP coordinate and the matched cube's measured base-frame coordinate. Applies proportional damping (correction step $= 0.75 \times \text{error}$) clamped to a maximum step of $60\text{mm}$.
    5. **Two-Stage Refinement:** Once the error is within $2.5\text{mm}$ at hover, the arm descends to `PBVS_REFINE_Z_MM` ($250\text{mm}$, skipped if already at approach height), re-measures from the shorter focal distance to minimize angular tilt errors, and locks when consecutive frames stay within tolerance.
    6. **Yaw Alignment:** Evaluates the target's rotation in the final image frame. Rotates the gripper joint to align with the object's principal axis.
    7. **Depth-Adaptive Descend:** Instead of descending to a fixed Z height, the robot calculates the target's measured top-face base $Z$. It shifts the nominal `GRASP_Z` ($105\text{mm}$) by the target's height deviation from nominal, applies asymmetric safety clamps, and adds a `GRASP_PRESS_DOWN_MM` ($5\text{mm}$) offset to compress the suction cup.
    8. **Grasp & Lift:** Activates the vacuum line, dwells for $3.0\text{s}$ to establish suction, and lifts to `TRANSPORT_Z_MM` ($220\text{mm}$).
    9. **Vacuum Feedback:** Queries the physical sensor (`get_suction_cup`). If vacuum flow reads $0$ (indicating no object is held), the system routes directly to `RECOVERY`.

### STEP 6: Transport and Drop-off Planning (State: `TRANSPORT` / `RETURNING`)
*   **Action:** Places the object in a clear spot or returns it if the target zone is full.
*   **Process:**
    1. **Free-Space Drop-Point Selection:** Operates primarily on the free-space plan established during `PRE_CHECK`. It extracts HSV pixels matching the white table, masks out the boundaries and existing obstacles, applies a distance transform, and selects the coordinate point furthest from all obstacles.
    2. **Randomized Fallback:** If the free-space plan is unavailable, it samples random coordinates inside the zone's base-frame bounding box, requiring a minimum clearance of $45\text{mm}$ from all occupied coordinates.
    3. **Kinematic Pre-Flight:** Verifies the IK solvability of the planned trajectory before starting movement. If the solver fails, it re-samples up to 3 times before falling back to the nominal zone center.
    4. **Blended Path Execution:** Executes the carry and descent as a single, continuous trajectory. The horizontal arrival at the drop coordinate and the vertical descent are blended using a corner fillet radius (`TRANSPORT_BLEND_RADIUS_MM` = $30\text{mm}$) to prevent intermediate stops.
    5. **Suction Release:** De-energizes the vacuum, vents the lines, dwells for $1.5\text{s}$, and returns the arm to `TRANSPORT_Z_MM`.
    6. **The `RETURNING` Alternate Path:** If the target zone is determined to be full, the robot carries the cube back to its original pickup coordinate, places it down at its measured grasp height, and terminates in the `DONE_RETURNED` state.

### STEP 7: Verification (State: `VERIFY`)
*   **Action:** Confirms delivery.
*   **Process:** Re-parks at `TOP_VIEW_POSE`. Compares target zone occupancy against the baseline captured in `PRE_CHECK`. Verification passes if the target-color count increased by $\ge 1$ or if the total target-color pixel mass increased by $\ge 900\text{px}$ (detecting cases where the dropped cube landed touching an existing one, merging the contours).

---

## 4. Error Recovery & Safety Features

```
          [Move Fault / Loss of Vacuum]
                        │
                        ▼
               [Check check_grasp()]
               /                   \
        (Still Held)           (Empty/Lost)
             /                       \
  [Increment Recovery Retry]       [Stop Vacuum]
  [Lift Vertically to Carry Z]     [Safe Home Transit]
  [Re-Plan Drop to Destination]    [Increment Search Retry]
  [Resume TRANSPORT/RETURNING]     [FSM State -> SCANNING]
```

### 4.1. Hold-Through-Retry Recovery
If a movement command is interrupted or fails mid-transit, the robot clears the error and queries the vacuum payload sensor before releasing the object.
*   If `check_grasp()` indicates the payload is still held, the robot performs a vertical-only recovery lift to safe carry height, marks the previous drop coordinates as obstructed, re-samples a fresh drop-off coordinate, and resumes `TRANSPORT` (or `RETURNING`).
*   This recovery loop is allowed up to `MAX_RECOVERY_RETRIES` (2), preventing the robot from dropping held cubes in untracked workspace locations during transient faults.

### 4.2. Safe Home Transit
If the payload is confirmed lost or if the recovery retries are exhausted, the robot vents the suction line, executes a slow, direct transit back to `HOME_POSE`, increments the task's general search retry counter, and restarts from `SCANNING` (up to `max_retries`).

### 4.3. Failure Signalling
If a task fails permanently (e.g., the source zone is empty during `PRE_CHECK` or search retries are exhausted), the robot moves to its current height's center and performs a joint-limit-safe "no-no shake" (rotating the wrist axis back and forth by $\pm 12^\circ$) to visually signal the failure before halting.

### 4.4. Graceful Soft Stop
*   **Signal Handlers:** Standard SIGINT (Ctrl+C) and SIGTERM are caught by `system.py` to raise a soft-stop flag.
*   **Interruption Rules:**
    *   During non-committed phases (`PRE_CHECK`, `SCANNING`, `GLOBAL_APPROACH`, and active alignment in `FINE_GRASP`), a soft stop halts motion immediately, aborts the FSM, and executes a controlled transit back to `HOME_POSE`.
    *   During committed phases (descending, active vacuum gripping, `TRANSPORT`, `RETURNING`, and releasing), the soft stop is ignored. The FSM completes the current pick, transport, and drop-off sequence to deliver the payload safely before returning to `HOME_POSE` and shutting down.
*   **Hard Stop Fallback:** If the user triggers SIGINT a second time, the system calls `os._exit(1)` immediately, functioning as a software emergency stop.

---

## 5. File Architecture

*   `auto.py` — Continuous autonomous sorting loop. Manages task selection priority, task-blocking for full zones, and empty-workspace shutdown.
*   `eval.py` — Multi-object single-instruction evaluation script. Parses NLP input and drains all matching objects from the source zone up to a safety limit (`EVAL_MAX_CUBES` = 6).
*   `utils/`
    *   `constants.py` — Single source of truth containing cartesian boundaries, zone ROIs, color thresholds, and PBVS gain values.
    *   `system.py` — Inter-process communication flags and soft/hard signal handling.
    *   `nlp.py` — Regex parsing of commands into task types and target colors.
    *   `robot.py` — Wrapper for the `XArmAPI`. Handles motion safety, pose reachability checks, and GPIO-driven suction cup states.
    *   `vision/`
        *   `camera.py` — Stream and Orbbec camera drivers. Integrates a frame-timestamp tracking thread to guarantee frame freshness.
        *   `detection.py` — HSV thresholding, morphological fragment fusion, touching-blob watershed segmentation, and depth-based top-face masking.
        *   `localize3d.py` — Geometric coordinate math. Includes pinhole camera projection, 3D point cloud localization, and table height estimation.
        *   `servo.py` — Position-Based Visual Servoing. Implements the target tracker, candidate matching, and metric alignment loops.
        *   `recorder.py` — Real-time diagnostics. Generates a 2x2 MP4 video mosaic during runs.
        *   `snapshot.py` — Generates before/after workspace images with annotated zone boundaries and detected object contours.

---

## 6. Diagnostic 2x2 Video Mosaic (`RunRecorder`)

When initialized with the `--video` flag, a background thread records a diagnostic MP4 mosaic at $10\text{ fps}$ to monitor perception performance without blocking the arm's control loops.

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ PANEL A: Annotated Color              │ PANEL B: Depth Colormap               │
│ Shows the RGB feed with bounding      │ Displays normalized depth values      │
│ boxes, 3D metric coordinates, and     │ using the Jet colormap. Filled with   │
│ active target tracking crosses.       │ black in stream-only (no-depth) mode. │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ PANEL C: Raw Color Mask               │ PANEL D: Isometric 3D Point Cloud     │
│ The raw, binary HSV mask before       │ An interactive metric reconstruction  │
│ height gating or top-face processing  │ of the scene, projected onto a virtual│
│ is applied. Helps diagnose lighting.  │ 3D camera to evaluate alignment.      │
└───────────────────────────────────────┴───────────────────────────────────────┘
```