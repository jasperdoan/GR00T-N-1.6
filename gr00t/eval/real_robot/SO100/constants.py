"""
SO100 Constants: Workspace zones, color ranges, home position, and scripted waypoints.
"""

import numpy as np

# =============================================================================
# Workspace Zone Definitions (x, y, width, height) in front-camera pixels
# =============================================================================

STORAGE_ZONE   = (277, 168, 96, 94)
CHECK_IN_ZONE  = (152, 103, 96, 95)
CHECK_OUT_ZONE = (399, 100, 99, 97)

# =============================================================================
# Language & Object Definitions
# =============================================================================

# Add any new objects you train here! Must be lowercase.
KNOWN_OBJECTS = [
    "red cube",
    "blue cube",
    "yellow cube",
    "pink prism",
    "orange sphere",
    "dice",
    "number 10 block"
]

# =============================================================================
# Front Camera Task Success Verification Constants
# =============================================================================

# Minimum pixel area for a detected blob to be considered a real object
MIN_BLOB_AREA_PX = 100

# =============================================================================
# Wrist Camera Grasp Verification Constants
# =============================================================================

# Bounding box between gripper fingers from calibration
WRIST_GRASP_ROI = (276, 293, 193, 186)

WRIST_PRESENCE_THR   = 30     # Pixel difference intensity to count as "changed" from baseline
WRIST_MIN_PRESENCE_PX= 1500   # Min changed pixels to confirm an object is in the gripper
WRIST_STABILITY_THR  = 6.0    # Diff threshold; any pixel changing > X is "moving"
WRIST_CONFIRM_FRAMES = 3      # Number of consecutive true frames required
VLA_GRASP_MIN_TIME   = 1.0    # Seconds the VLA runs before checking for grasps

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

# Maximum gripper position allowed during transport.
# Clamps the locked value so a slightly-open grasp (e.g. 22°) can't slip further.
GRIPPER_TRANSPORT_MAX = GRIPPER_GRASP_POS + 5.0

# =============================================================================
# Task-Specific Ready / Approach Positions
# =============================================================================
# Keyed by task_type string. Each dict is a partial joint map — joints absent
# here are filled from current state in move_to_ready(). Gripper is always
# forced open (GRIPPER_OPEN_POS) by move_to_ready() independently.

READY_POSITIONS = {
    "check_in": {
        "shoulder_pan.pos":   36.2,
        "shoulder_lift.pos": -21.1,
        "elbow_flex.pos":     27.1,
        "wrist_flex.pos":     78.4,
    },
    "check_out": {
        "shoulder_pan.pos":    0.0,
        "shoulder_lift.pos": -21.1,
        "elbow_flex.pos":     27.1,
        "wrist_flex.pos":     78.4,
    },
}

# =============================================================================
# Scripted Transport Waypoints
# =============================================================================

LIFT_OVERRIDE = {
    "shoulder_lift.pos": -33.3,
    "elbow_flex.pos":    23.4,
    "wrist_flex.pos":    70.1,
}

STORAGE_PLACE = {
    "shoulder_pan.pos":   0.0,
    "shoulder_lift.pos": 19.5,
    "elbow_flex.pos":   -12.1,
    "wrist_flex.pos":    72.6,
    "wrist_roll.pos":    63.8,
}

CHECKOUT_PLACE = {
    "shoulder_pan.pos":  -50.2,
    "shoulder_lift.pos": 19.5,
    "elbow_flex.pos":   -12.1,
    "wrist_flex.pos":    72.6,
    "wrist_roll.pos":    63.8,
}

PLACE_VARIATION_DEG = 2.0   # uniform random in [-X, +X] degrees per joint
POST_DROP_PAUSE     = 0.35  # pause after dropping cube (seconds) — increased for cleaner settle

LERP_DURATION_LIFT  = 1.0
LERP_DURATION_PLACE = 1.0
LERP_DURATION_DROP  = 0.4   # increased from 0.1 — prevents flinging the cube on release
LERP_DURATION_HOME  = 1.0