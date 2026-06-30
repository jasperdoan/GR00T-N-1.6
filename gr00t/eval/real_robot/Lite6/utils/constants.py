import os

# --- NETWORK & SYSTEM ---
DEFAULT_IP = "192.168.1.150"
WRIST_CAM_IDX = 0
TOP_DOWN_CAM_IDX = 1

INUSE_FLAG_PATH = "/tmp/lite6_inuse.flag"
STOP_FLAG_PATH  = "/tmp/stop_lite6.flag"


# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX_PATH = os.path.join(BASE_DIR, "homography_matrix.npy")
# Separate output dirs so eval's before/after snapshots don't land in the auto dir.
OUTPUT_DIR_EVAL = "/workspaces/data/outputs/GR00T-N-1.6/lite6_eval"
OUTPUT_DIR_AUTO = "/workspaces/data/outputs/GR00T-N-1.6/lite6_auto"

# --- ROBOT MOTION SETTINGS ---
DEFAULT_SPEED = 200        # mm/s
DEFAULT_ACCEL = 500        # mm/s^2
FINE_ADJUST_SPEED = 80     # mm/s (for visual servoing)

GRIPPER_SETTLE_S = 0.4     # pause after an open/close so the gripper finishes actuating

DEFAULT_TASK  = "check_in"
SCAN_INTERVAL = 3.0  # seconds between scans when idle


# --- Robot Poses & Heights (Cartesian mm) ---
# Safe height to travel above all objects
SAFE_Z = 150.0
# Height to execute the actual grasp
GRASP_Z = 50.0
# Default Home Position [X, Y, Z, Roll, Pitch, Yaw]
HOME_POSE = [0.0, -150.0, 200.0, -180.0, 0.0, 0.0]
TOP_VIEW_POSE = [-50.0, -150.0, 300.0, -180.0, 0.0, 0.0]

# --- Workspace envelope (Cartesian mm) ---
# Reject any homography/servo target outside this box before commanding a move,
# so a bad detection can't fault the arm by driving it out of reach.
WORKSPACE_X_RANGE = (-160.0, 0.0)
WORKSPACE_Y_RANGE = (-300.0, -150.0)
WORKSPACE_Z_RANGE = (50.0, 300.0)


# --- Zone Definitions (Robot Cartesian Coordinates in mm) ---
ZONES = {
    "check_in":  {"x": -60.0, "y": -300.0},
    "storage":   {"x": -160.0, "y": -300.0},
    "check_out": {"x": -160.0, "y": -150.0}
}

# Top-down-camera pixel ROIs (x, y, w, h) per zone, keyed by the INTERNAL zone
# names the FSM/NLP use. These restrict color masking to a specific zone so that
# PRE_CHECK (source), VERIFY (target), and auto's task selection actually work.
# TODO: replace with real pixel ROIs after calibrating the top-down camera.
ZONE_PIXEL_ROI = {
    "check_in":  (0,   0, 200, 200),
    "storage":   (200, 0, 200, 200),
    "check_out": (400, 0, 200, 200),
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
VS_KP = 0.08                  # proportional gain: pixel error -> mm delta
CENTER_TOLERANCE_PX = 10      # within this pixel error the object is "centered"
MAX_SERVO_STEP_MM = 8.0       # clamp per-iteration delta so a big initial error
                              # can't command an overshoot at FINE_ADJUST_SPEED
SERVO_DEADBAND_PX = 3         # ignore sub-pixel jitter below this error

# Hand-eye offset: the wrist camera is NOT coaxial with the gripper TCP, so the
# gripper sits over the object when the object is at (frame_center + offset), not
# at the raw frame center. Tune (dx_px, dy_px) on hardware; (0,0) = no correction.
# TODO: 
CAMERA_CENTER_OFFSET = (0, 0)

# --- FSM retry / timeout defaults ---
VLA_TIMEOUT  = 15.0   # seconds for the visual-servo lock attempt
MAX_RETRIES  = 2

# --- HSV object detection (top-down) ---
MIN_BLOB_AREA_PX      = 100
FRONT_MIN_PRESENCE_PX = 1500

# --- Wrist-camera grasp confirmation ---
# After closing the gripper, the target color must occupy at least this many
# pixels inside the gripper ROI to confirm we actually grabbed the object
# (instead of closing on air). Tune from the printed pixel counts on hardware.
# TODO: 
WRIST_GRASP_ROI    = (160, 360, 320, 120)   # (x, y, w, h) in wrist-cam pixels
WRIST_GRASP_MIN_PX = 4000