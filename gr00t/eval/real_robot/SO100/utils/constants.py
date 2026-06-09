"""
SO100 Constants: Workspace zones, color ranges, home position, and scripted waypoints.
"""

# =============================================================================
# Workspace Zone Definitions (x, y, width, height) in front-camera pixels
# =============================================================================

STORAGE_ZONE = (869, 483, 267, 249)
CHECK_IN_ZONE = (595, 323, 263, 221)
CHECK_OUT_ZONE = (1133, 311, 268, 216)

# =============================================================================
# Language & Object Definitions
# =============================================================================

# Add any new objects you train here! Must be lowercase.
KNOWN_OBJECTS = [
    "red cube",
    "blue cube",
    "yellow cube",
    "orange cube",
    "pink cube",
    "green prism",
    "red ball",
    "yellow ball"
]

# =============================================================================
# HSV Color Ranges for Vision Tracking (Hue, Saturation, Value)
# Ranges are (Lower Bound), (Upper Bound). Red needs two ranges because it wraps around 180.
# =============================================================================

COLOR_RANGES = {
    "black":  [((0, 0, 0), (180, 255, 65))],
    "white":  [((0, 0, 160), (180, 60, 255))],
    "gray":   [((0, 0, 66), (180, 65, 159))],
    "red":    [((0, 100, 40), (5, 255, 255)), ((170, 100, 40), (180, 255, 255))], 
    "yellow": [((15, 66, 50), (45, 255, 255))],
    "blue":   [((95, 66, 50), (130, 255, 255))],
    "green":  [((46, 66, 50), (94, 255, 255))],
    "purple": [((131, 66, 50), (160, 255, 255))],
    "orange": [((10, 66, 50), (20, 255, 255))], 
}

# =============================================================================
# Front Camera Task Success Verification Constants
# =============================================================================

# Minimum pixel area for a detected blob to be considered a real object
MIN_BLOB_AREA_PX = 100

# Minimum color pixel count in front camera to confirm object exists before starting
# You can tune this based on the printed output from the Pre-Check step
FRONT_MIN_PRESENCE_PX = 2000  

# =============================================================================
# Wrist Camera Grasp Verification Constants
# =============================================================================
WRIST_GRASP_ROI = (164, 362, 346, 117)
# WRIST_GRASP_ROI = (133, 311, 413, 166)

WRIST_PRESENCE_THR   = 50     # Pixel difference intensity to count as "changed" from baseline
WRIST_MIN_PRESENCE_PX= 10000  # Min changed pixels to confirm an object is in the gripper
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
    "wrist_roll.pos":     0.0,
    "gripper.pos":        95.0,   # open (inverted design)
}

# =============================================================================
# Gripper State Thresholds
# =============================================================================

GRIPPER_OPEN_POS    = 95.0   # UPDATED: degrees — fully open
GRIPPER_GRASP_POS   = 25.0   # UPDATED: degrees — good grasp on cube
GRIPPER_FULLY_CLOSED= 1.0    # UPDATED: degrees — closed empty

# Minimum gripper position allowed during transport.
GRIPPER_TRANSPORT_THRESHOLD = GRIPPER_GRASP_POS + 5.0

# =============================================================================
# Task-Specific Ready / Approach Positions
# =============================================================================
# Keyed by task_type string. Each dict is a partial joint map — joints absent
# here are filled from current state in move_to_ready(). Gripper is always
# forced open (GRIPPER_OPEN_POS) by move_to_ready() independently.

READY_POSITIONS = {
    "check_in": {
        "shoulder_pan.pos":   45.0,
        "shoulder_lift.pos": -65.0,
        "elbow_flex.pos":     60.0,
        "wrist_flex.pos":     72.0,
    },
    "check_out": {
        "shoulder_pan.pos":    0.0,
        "shoulder_lift.pos": -60.0,
        "elbow_flex.pos":     60.0,
        "wrist_flex.pos":     45.0,
    },
    "check_back": {
        "shoulder_pan.pos":  -45.0,
        "shoulder_lift.pos": -65.0,
        "elbow_flex.pos":     60.0,
        "wrist_flex.pos":     72.0,
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

CHECKIN_PLACE = {
    "shoulder_pan.pos":  45.0,
    "shoulder_lift.pos": 19.5,
    "elbow_flex.pos":   -2.1,
    "wrist_flex.pos":    72.6,
    "wrist_roll.pos":    33.8,
}

STORAGE_PLACE = {
    "shoulder_pan.pos":   0.0,
    "shoulder_lift.pos": 19.5,
    "elbow_flex.pos":   -2.1,
    "wrist_flex.pos":    72.6,
    "wrist_roll.pos":    33.8,
}

CHECKOUT_PLACE = {
    "shoulder_pan.pos":  -50.2,
    "shoulder_lift.pos": 19.5,
    "elbow_flex.pos":   -2.1,
    "wrist_flex.pos":    72.6,
    "wrist_roll.pos":    33.8,
}

PLACE_VARIATION_DEG = 2.0   # uniform random in [-X, +X] degrees per joint
POST_DROP_PAUSE     = 0.35  # pause after dropping cube (seconds) — increased for cleaner settle

LERP_DURATION_LIFT  = 1.0
LERP_DURATION_PLACE = 1.0
LERP_DURATION_DROP  = 0.4   # increased from 0.1 — prevents flinging the cube on release
LERP_DURATION_HOME  = 2.0