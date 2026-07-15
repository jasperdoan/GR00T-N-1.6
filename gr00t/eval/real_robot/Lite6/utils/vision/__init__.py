"""
Lite6 vision package.

Modules by responsibility:
  helpers     — low-level image utils (uint8, saturation, mask cleanup, clamp)
  camera      — StreamCamera (MJPEG URL) / OrbbecCamera (local, color+depth), open_camera
  detection   — HSV masks, blob finding, depth-based top-face refinement
  servo       — fine visual-servoing loop with lost-target recovery
  snapshot    — annotated workspace snapshots
  localize3d  — ALL pixel↔robot geometry: intrinsics (de)projection, camera→base
                extrinsics, 3D localization, ray-plane intersection, table-z
                estimation (the one-pose homography was removed)

Public API is re-exported here so `from utils.vision import X` keeps working.
"""

from utils.vision.camera import (          # noqa: F401
    StreamCamera,
    OrbbecCamera,
    open_camera,
    read_fresh,
)
from utils.vision.detection import (       # noqa: F401
    BlobCandidate,
    find_object_centroid,
    find_object_blob,
    find_all_blobs,
    count_objects_in_zone,
    select_blob_near,
    refine_blob,
    check_color_presence,
    blob_inside_roi,
    color_mask_of,
    top_face_mask,
    height_gate_mask,
)
from utils.vision.servo import (           # noqa: F401
    visual_servo_to_grasp,
)
from utils.vision.snapshot import (        # noqa: F401
    save_workspace_snapshot,
)
from utils.vision.localize3d import (      # noqa: F401
    deproject,
    project,
    robust_depth_at,
    localize_object_3d,
    solve_extrinsics,
    save_extrinsics,
    load_extrinsics,
    camera_to_base,
    base_to_camera,
    pixel_to_base_on_plane,
    estimate_table_z,
)
from utils.vision.recorder import (        # noqa: F401
    RunRecorder,
)
