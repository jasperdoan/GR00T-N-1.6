import os

# --- NETWORK & SYSTEM ---
DEFAULT_IP = "192.168.1.189"
# Camera source: int → local /dev/video device index; str → network stream URL.
# The wrist camera is hosted on a separate machine serving MJPEG over HTTP.
CAMERA_SOURCE = "http://172.21.2.83:9988/stream.mjpg"

INUSE_FLAG_PATH = "/tmp/lite6_inuse.flag"
STOP_FLAG_PATH  = "/tmp/stop_lite6.flag"


# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH = os.path.join(BASE_DIR, "data", "homography_matrix.npy")
# Separate output dirs so eval's before/after snapshots don't land in the auto dir.
OUTPUT_DIR_EVAL = "gr00t/eval/real_robot/Lite6/data/outputs/GR00T-N-1.6/lite6_eval"
OUTPUT_DIR_AUTO = "gr00t/eval/real_robot/Lite6/data/outputs/GR00T-N-1.6/lite6_auto"

# --- ROBOT MOTION SETTINGS ---
DEFAULT_SPEED = 200        # mm/s
DEFAULT_ACCEL = 500        # mm/s^2
FINE_ADJUST_SPEED = 80     # mm/s (for visual servoing)

GRIPPER_SETTLE_S = 0.4     # pause after an open/close so the gripper finishes actuating

DEFAULT_TASK  = "check_in"
SCAN_INTERVAL = 3.0  # seconds between scans when idle


# --- Robot Poses & Heights (Cartesian mm) ---
# Safe height to travel above all objects
SAFE_Z = 200.0
# Height to execute the actual grasp
GRASP_Z = 100.0
# Default Home Position [X, Y, Z, Roll, Pitch, Yaw]
HOME_POSE = [0.0, -150.0, 200.0, -180.0, 0.0, 0.0]
TOP_VIEW_POSE = [-50.0, -150.0, 300.0, -180.0, 0.0, 0.0]

# --- Workspace envelope (Cartesian mm) ---
# Reject any homography/servo target outside this box before commanding a move,
# so a bad detection can't fault the arm by driving it out of reach.
WORKSPACE_X_RANGE = (-170.0, 10.0)
WORKSPACE_Y_RANGE = (-310.0, -140.0)
WORKSPACE_Z_RANGE = (90.0, 310.0)


# --- Zone Definitions (Robot Cartesian Coordinates in mm) ---
ZONES = {
    "check_in":  {"x": -60.0, "y": -300.0},
    "storage":   {"x": -160.0, "y": -300.0},
    "check_out": {"x": -160.0, "y": -150.0}
}

# Top-down-camera pixel ROIs (x, y, w, h) per zone, keyed by the INTERNAL zone
# names the FSM/NLP use. These restrict color masking to a specific zone so that
# PRE_CHECK (source), VERIFY (target), and auto's task selection actually work.
ZONE_PIXEL_ROI = {
    "check_in":  (166, 425, 235, 226),
    "storage":   (158, 171, 255, 231),
    "check_out": (542, 160, 249, 241),
}

# Display-name view of the same ROIs, used only for drawing snapshot overlays.
ALL_ZONES_DICT = {
    "Check In":  ZONE_PIXEL_ROI["check_in"],
    "Storage":   ZONE_PIXEL_ROI["storage"],
    "Check Out": ZONE_PIXEL_ROI["check_out"],
}


# --- Language & Vision Definitions ---
KNOWN_OBJECTS = ["red cube", "blue cube", "yellow cube", "green cube"]

# HSV Color Ranges (Lower Bound, Upper Bound)
COLOR_RANGES = {
    "red":    [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (180, 255, 255))], 
    "blue":   [((100, 150, 50), (140, 255, 255))],
    "yellow": [((20, 100, 100), (30, 255, 255))],
    "green":  [((40, 50, 50), (90, 255, 255))],
}

# --- Visual Servoing (Image-Based Visual Servoing P-controller) ---
# The pixel error is mapped to a robot-frame delta THROUGH THE HOMOGRAPHY
# (H(p_obj) - H(p_aim)): the camera's orientation never changes between the
# calibration pose and the hover pose, so H's direction is always correct and
# only its scale is off (camera is closer at hover -> H overestimates mm/px by
# roughly the height ratio). SERVO_GAIN damps that overestimate.
SERVO_GAIN = 0.6              # fraction of the homography-mapped error per step
MAX_SERVO_STEP_MM = 8.0       # clamp per-iteration delta so a big initial error
                              # can't command an overshoot at FINE_ADJUST_SPEED
SERVO_DEADBAND_PX = 3         # ignore sub-pixel jitter below this error

# --- Move completion (pose convergence) ---
# _safe_move issues non-blocking moves and detects completion by polling the
# actual pose — get_is_moving() lags after a non-blocking command and once let
# the FSM race ahead while the arm was still at TOP_VIEW_POSE.
POSE_TOL_MM    = 2.0    # XYZ distance to target to count as "arrived"
POSE_TOL_DEG   = 2.0    # yaw distance (covers yaw-only alignment moves)
MOVE_TIMEOUT_S = 30.0   # give up (and fail the move) after this long

# Gripper ROI: where the object appears in the wrist camera when the gripper is
# correctly over it (camera is NOT coaxial with the TCP). Measured from 3 samples
# with the gripper hovering over the object at yaw=0:
#   (627, 591, 162, 128), (609, 578, 130, 126), (618, 582, 150, 133)
# ROI = union of the samples (609, 578, 180, 146) + 15 px margin per side.
# The servo drives the blob centroid toward the ROI center and accepts only when
# the blob's bounding box is FULLY inside this ROI.
GRIPPER_ROI = (594, 563, 210, 176)   # (x, y, w, h) in wrist-cam pixels

# Consecutive frames the blob must stay fully inside GRIPPER_ROI to lock the
# servo (HSV mask edges flicker ~1-2 px; a single strict frame would chatter).
SERVO_CONFIRM_FRAMES = 3

# --- Yaw alignment (grasping rotated objects) ---
# The object's in-image angle (cv2.minAreaRect at the hover pose) is mapped to a
# gripper yaw command. Sign/offset depend on how the camera is mounted relative
# to the gripper jaws — verify on hardware with a deliberately rotated cube.
YAW_ALIGN_ENABLED  = True
CAMERA_YAW_SIGN    = 1.0    # flip to -1.0 if the gripper rotates the wrong way
CAMERA_YAW_OFFSET  = 0.0    # fixed mount rotation (deg) between image axes and jaws

# --- FSM retry / timeout defaults ---
VLA_TIMEOUT  = 15.0   # seconds for the visual-servo lock attempt
MAX_RETRIES  = 2

# --- HSV object detection (top-down) ---
MIN_BLOB_AREA_PX      = 100
FRONT_MIN_PRESENCE_PX = 1500