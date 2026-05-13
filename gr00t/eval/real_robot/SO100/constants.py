"""
SO100 Constants: Workspace zones, color ranges, home position, and scripted waypoints.
"""

import numpy as np

# =============================================================================
# Workspace Zone Definitions (x, y, width, height) in front-camera pixels
# =============================================================================

STORAGE_ZONE   = (277, 168, 96, 94)
CHECK_IN_ZONE  = (399, 100, 99, 97)
CHECK_OUT_ZONE = (152, 103, 96, 95)

# =============================================================================
# HSV Color Ranges for Front Camera Detection
# =============================================================================

COLOR_RANGES = {
    "red": [
        (np.array([0,   10, 100]), np.array([10,  255, 255])),
        (np.array([165, 10, 100]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([80,  50,  0]), np.array([120, 255, 255])),
    ],
    "yellow": [
        (np.array([20,  50, 50]), np.array([45,  255, 255])),
    ],
}

# Minimum pixel area for a detected blob to be considered a real cube (not noise)
MIN_BLOB_AREA_PX = 100

# =============================================================================
# Wrist Camera Grasp Verification Constants (NEW)
# =============================================================================

# Bounding box between gripper fingers from calibration
WRIST_GRASP_ROI = (276, 293, 193, 186)

# Calibrated HSV Ranges for the Wrist Camera
WRIST_COLOR_RANGES = {
    "red": [
        (np.array([165, 80, 120]), np.array([180, 255, 255])),
        (np.array([0,   80, 120]), np.array([10,  255, 255])),
    ],
    "blue": [
        (np.array([90, 40, 120]), np.array([115, 255, 255])),
    ],
    "yellow": [
        (np.array([10, 15, 150]), np.array([35, 255, 255])),
    ],
}

WRIST_MIN_COLOR_PX   = 5000      # Min pixels matching target color inside ROI
WRIST_STABILITY_THR  = 3.0     # Diff threshold; any pixel changing > X is "moving"
WRIST_CONFIRM_FRAMES = 3      # Number of consecutive true frames required
VLA_GRASP_MIN_TIME   = 1.0     # Number of seconds the VLA runs before checking for grasps

# =============================================================================
# Robot Joint Names (ordered, matches state/action arrays)
# =============================================================================

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# =============================================================================
# Home Position
# =============================================================================

HOME_ACTION = {
    "shoulder_pan.pos":   0.0,
    "shoulder_lift.pos": -60.0,
    "elbow_flex.pos":     60.0,
    "wrist_flex.pos":     60.0,
    "wrist_roll.pos":     90.0,
    "gripper.pos":        40.0,
}

# =============================================================================
# Gripper State Thresholds
# =============================================================================

GRIPPER_OPEN_POS    = 40.0   # degrees — fully open
GRIPPER_GRASP_POS   = 15.0   # degrees — expected closed-on-cube position
GRIPPER_GRASP_TOL   = 1.5    # ± tolerance in degrees

# =============================================================================
# Scripted Transport Waypoints
# =============================================================================

LIFT_OVERRIDE = {
    "shoulder_lift.pos":  -10.8,
    "elbow_flex.pos":    3.8,
    "wrist_flex.pos":    80.2,
}

STORAGE_PLACE = {
    "shoulder_pan.pos":   0.0,
    "shoulder_lift.pos": 19.5,
    "elbow_flex.pos":   -12.1,
    "wrist_flex.pos":    72.6,
    "wrist_roll.pos":    63.8,
}

CHECKOUT_PLACE = {
    "shoulder_pan.pos":  -48.2,
    "shoulder_lift.pos": 19.5,
    "elbow_flex.pos":   -12.1,
    "wrist_flex.pos":    72.6,
    "wrist_roll.pos":    63.8,
}

PLACE_VARIATION_DEG = 2.0   # uniform random in [-X, +X] degrees per joint
POST_DROP_PAUSE = 0.25      # Pause duration after dropping the cube (seconds)

LERP_DURATION_LIFT  = 1.0
LERP_DURATION_PLACE = 1.5
LERP_DURATION_DROP  = 0.1
LERP_DURATION_HOME  = 2.0