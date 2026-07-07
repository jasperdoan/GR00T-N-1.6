"""
Lite6 vision package.

Modules by responsibility:
  helpers     — low-level image utils (uint8, saturation, mask cleanup, clamp)
  camera      — StreamCamera (MJPEG URL) / OrbbecCamera (local, color+depth), open_camera
  safety      — SafetyMonitor (multiprocess MediaPipe hand detection)
  homography  — 2D pixel→robot mapping (top-view calibration)
  detection   — HSV masks, blob finding, depth-based top-face refinement
  servo       — fine visual-servoing loop with lost-target recovery
  snapshot    — annotated workspace snapshots
  localize3d  — depth deprojection, camera→base extrinsics, 3D localization

Public API is re-exported here so `from utils.vision import X` keeps working.
"""

from utils.vision.camera import (          # noqa: F401
    StreamCamera,
    OrbbecCamera,
    open_camera,
    read_fresh,
)
from utils.vision.safety import (          # noqa: F401
    SafetyMonitor,
    HAS_MEDIAPIPE,
)
from utils.vision.homography import (      # noqa: F401
    load_homography,
    pixel_to_robot,
)
from utils.vision.detection import (       # noqa: F401
    find_object_centroid,
    find_object_blob,
    check_color_presence,
    blob_inside_roi,
    color_mask_of,
    top_face_mask,
)
from utils.vision.servo import (           # noqa: F401
    visual_servo_to_grasp,
)
from utils.vision.snapshot import (        # noqa: F401
    save_workspace_snapshot,
)
from utils.vision.localize3d import (      # noqa: F401
    deproject,
    localize_object_3d,
    solve_extrinsics,
    save_extrinsics,
    load_extrinsics,
    camera_to_base,
)
