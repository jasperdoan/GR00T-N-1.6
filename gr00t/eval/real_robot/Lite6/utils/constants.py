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
OUTPUT_DIR = "/tmp/lite6_auto"

# --- ROBOT MOTION SETTINGS ---
DEFAULT_SPEED = 200        # mm/s
DEFAULT_ACCEL = 500        # mm/s^2
FINE_ADJUST_SPEED = 80     # mm/s (for visual servoing)

DEFAULT_TASK  = "check_in"
SCAN_INTERVAL = 3.0  # seconds between scans when idle


# --- Robot Poses & Heights (Cartesian mm) ---
# Safe height to travel above all objects
SAFE_Z = 150.0  
# Height to execute the actual grasp
GRASP_Z = 50.0  
# Default Home Position [X, Y, Z, Roll, Pitch, Yaw]
HOME_POSE = [250.0, 0.0, 200.0, -180.0, 0.0, 0.0]


# --- Zone Definitions (Robot Cartesian Coordinates in mm) ---
# TODO: Jog your robot to these zones in UFACTORY Studio and update the X, Y!
ZONES = {
    "check_in":  {"x": 150.0, "y": -200.0},
    "storage":   {"x": 300.0, "y": 0.0},
    "check_out": {"x": 150.0, "y": 200.0}
}

ALL_ZONES_DICT = {
    "Check In":  (0, 0, 200, 200),   # TODO: replace with real pixel ROIs after calibration
    "Storage":   (200, 0, 200, 200),
    "Check Out": (400, 0, 200, 200),
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

# Visual Servoing Gain (Pixel error to mm)
VS_KP = 0.08
CENTER_TOLERANCE_PX = 10

# --- Gripper positions (xArm gripper range 0–800, 0=closed, 800=open) ---
GRIPPER_OPEN_POS   = 800
GRIPPER_CLOSED_POS = 0

# --- FSM retry / timeout defaults ---
VLA_TIMEOUT  = 15.0   # seconds
MAX_RETRIES  = 2

# --- HSV object detection ---
MIN_BLOB_AREA_PX   = 100
FRONT_MIN_PRESENCE_PX = 1500