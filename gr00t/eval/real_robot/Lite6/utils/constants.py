import os

# --- NETWORK & SYSTEM ---
DEFAULT_IP = "192.168.1.189"
# Camera source: int → local Orbbec device index; str → network MJPEG URL.
# Default is the local Orbbec: grasping REQUIRES depth + intrinsics +
# calibrated extrinsics (the homography was removed) — the color-only stream
# remains usable only for viewing/recording tools.
CAMERA_SOURCE = 0

INUSE_FLAG_PATH = "/tmp/lite6_inuse.flag"
STOP_FLAG_PATH  = "/tmp/stop_lite6.flag"


# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Camera→robot rigid transform (R, t) solved by script/lite6_extrinsics.py —
# THE single calibration of the pipeline. All pixel→robot geometry (scanning,
# occupancy, drop rects, servo steps) runs through it + the color intrinsics.
EXTRINSICS_PATH = os.path.join(BASE_DIR, "data", "extrinsics.npz")
# Separate output dirs so eval's before/after snapshots don't land in the auto dir.
OUTPUT_DIR_EVAL = "/workspaces/data/outputs/GR00T-N-1.6/lite6_eval"
OUTPUT_DIR_AUTO = "/workspaces/data/outputs/GR00T-N-1.6/lite6_auto"

# --- ROBOT MOTION SETTINGS ---
DEFAULT_SPEED = 200        # mm/s
DEFAULT_ACCEL = 500        # mm/s^2
FINE_ADJUST_SPEED = 80     # mm/s (for visual servoing)

# Vacuum head timing (tool GPIO-driven suction cup; see robot.py for the
# open/close_lite6_gripper → suction ON/OFF wire mapping).
VACUUM_GRIP_DWELL_S    = 3.0  # dwell after energizing suction so it fully seats
                              # on the object before lifting
VACUUM_RELEASE_DWELL_S = 1.5  # dwell after venting so the object drops free
                              # before de-energizing the valve lines

# --- Whole-zone randomized placement ---
# Drop points are sampled uniformly across the target zone (its pixel ROI
# corners ray-cast onto the table plane via intrinsics + extrinsics) instead
# of jittered around one nominal spot: repeated cycles scatter naturally, and
# because samples stay PLACE_MARGIN_MM inside the ROI, a placed cube can no
# longer drift half-out of its zone over hundreds of cycles.
PLACE_MARGIN_MM        = 20.0  # shrink the zone drop rectangle inward by this much
PLACE_CLEARANCE_MM     = 45.0  # min drop distance from cubes already in the zone
PLACE_SAMPLE_ATTEMPTS  = 20    # rejection-sampling budget before falling back
PLACE_MIN_FALLBACK_CLEARANCE_MM = 25.0  # best-effort sample must at least clear
                                        # a cube footprint, else use the nominal spot

# --- Free-space drop planning (the work surface is WHITE) ---
# The PRIMARY drop-point chooser detects emptiness directly instead of
# inferring it from the color census: white(-ish) pixels inside the target
# zone ROI are free space, everything else — a cube of ANY color (known to
# COLOR_RANGES or not), a stray object, a shadow — is occupied. A distance
# transform then finds the free pixel farthest from every occupied pixel AND
# from the zone boundary, so the drop lands as deep in open space as the zone
# allows (spread out, never against an edge). The random sampler above remains
# only as the fallback when this analysis is unavailable (no table-z) or its
# point is kinematically unreachable.
FREE_SPACE_S_MAX = 70    # HSV saturation ceiling for "white table" pixels
FREE_SPACE_V_MIN = 120   # HSV value floor for "white table" pixels (deep
                         # shadow fails this → reads occupied → conservative)
PLACE_FREE_MIN_CLEARANCE_MM = 30.0  # min distance from the drop CENTER to the
                                    # nearest occupied pixel / zone edge (half a
                                    # cube ~20 + margin); below this = zone FULL

DEFAULT_TASK  = "check_in"
SCAN_INTERVAL = 3.0  # seconds between scans when idle

# Consecutive empty scans (target object in NO source zone) before auto.py
# parks the arm, sets the stop flag and exits 0. The stop flag is what makes
# auto_supervised.sh stop instead of endlessly restarting an idle demo.
EMPTY_SCAN_GIVE_UP = 3

# Consecutive failed top-view reads before auto.py EXITS nonzero so a
# supervisor can restart it with a fresh camera. Without this, a dead MJPEG
# stream / unplugged Orbbec loops "read failed; retrying" forever — the
# process never dies, so no watchdog can help.
CAMERA_FAIL_LIMIT = 10   # ~10 × SCAN_INTERVAL ≈ 30 s of blindness


# --- Robot Poses & Heights (Cartesian mm) ---
# Height ladder: TOP_VIEW 450 (scan) → APPROACH 250 (PBVS) → CARRY 220
# (transport) → GRASP 105. The Lite6's ~440mm reach sphere is centered at the
# shoulder (~z 243): TOOL-DOWN AT 450 IS OUT OF REACH AT THE TABLE CORNERS
# (measured fault at (-180, -330, 450) ≈ 462mm wrist-center distance), so only
# the central TOP_VIEW_POSE lives that high.
APPROACH_Z_MM  = 250.0  # GLOBAL_APPROACH hover = PBVS working height (== the
                        # PBVS refine height, so the refine descend auto-skips).
                        # NOT higher — see the reach note above.
TRANSPORT_Z_MM = 220.0  # post-grasp lift + carry height: reach-safe across the
                        # whole table (worst corner ≈ 378mm), clears standing
                        # cubes (~45mm) plus the held payload.
# Height to execute the actual grasp
GRASP_Z = 105

# --- Depth-adaptive grasp height ---
# GRASP_Z is the empirically correct TCP height for a NOMINAL cube (top face at
# table_z + OBJECT_HEIGHT_MM). When enabled, the FINE_GRASP descend shifts
# GRASP_Z by the DIFFERENCE between the depth-measured top face (PBVS est_z,
# EMA over refine-stage frames; SCANNING's 3D fix as fallback) and the nominal
# top — short cubes get a deeper press, tall objects a higher stop. The clamps
# are asymmetric on purpose: lowering is capped hard by the workspace floor
# (WORKSPACE_Z_RANGE[0] = 100 leaves only 5 mm under GRASP_Z — raise both
# together only after physically checking the suction tip can't strike the
# table), while raising has generous headroom so we never crush an
# unexpectedly tall object.
GRASP_DEPTH_ADAPT_ENABLED = True
GRASP_Z_MAX_LOWER_MM = 7.5
GRASP_Z_MAX_RAISE_MM = 30.0
# The depth measurement lands pin-point on the object's surface, which isn't
# enough travel for the suction cup to seat (Thor testing: consistently a few
# mm short / hovering). Press this far PAST the measured/nominal surface on
# every descend — adaptive or fixed-fallback alike. Tune directly on Thor.
GRASP_PRESS_DOWN_MM = 5.0
# Default Home Position [X, Y, Z, Roll, Pitch, Yaw]
HOME_POSE = [0.0, -150.0, 200.0, -180.0, 0.0, 0.0]
TOP_VIEW_POSE = [0.0, -150.0, 450.0, -180.0, 0.0, 0.0]

# --- Workspace envelope (Cartesian mm) ---
# Reject any vision/servo target outside this box before commanding a move,
# so a bad detection can't fault the arm by driving it out of reach.
WORKSPACE_X_RANGE = (-220.0, 90.0)
WORKSPACE_Y_RANGE = (-395.0, -75.0)
WORKSPACE_Z_RANGE = (95, 510.0)


# --- Zone Definitions (Robot Cartesian Coordinates in mm) ---
ZONES = {
    "check_in":  {"x": 5.0, "y": -327.5},
    "storage":   {"x": -155.0, "y": -327.5},
    "check_out": {"x": -155.0, "y": -145.0}
}

# Top-down-camera pixel ROIs (x, y, w, h) per zone, keyed by the INTERNAL zone
# names the FSM/NLP use. These restrict color masking to a specific zone so that
# PRE_CHECK (source), VERIFY (target), and auto's task selection actually work.
ZONE_PIXEL_ROI = {
    "check_in":  (223, 375, 265, 253),
    "storage":   (230, 99, 254, 251),
    "check_out": (538, 94, 258, 251),
}

# Display-name view of the same ROIs, used only for drawing snapshot overlays.
ALL_ZONES_DICT = {
    "Check In":  ZONE_PIXEL_ROI["check_in"],
    "Storage":   ZONE_PIXEL_ROI["storage"],
    "Check Out": ZONE_PIXEL_ROI["check_out"],
}


# --- Language & Vision Definitions ---
KNOWN_OBJECTS = [
    "red cube", "blue cube", "yellow cube", "green cube",
    "orange cube", "pink cube", "purple cube",
]

# HSV Color Ranges (Lower Bound, Upper Bound)
# Red's V floor is 60 (not 100): during the final grasp approach the gripper
# casts a shadow on the cube and V drops — a V>=100 floor made the cube vanish
# from detection precisely at the grasp pose.
#
# CONTRACT: ranges must stay pairwise disjoint in (H, S) space — the all-color
# zone census (occupancy for drop clearance / zone-full detection) sums blobs
# across colors, and overlapping ranges would double-count a single cube.
# Neighboring boundaries are pure HUE splits: red|orange at H 8/9,
# orange|yellow at 19/20, blue|purple at 128/129, purple|pink at 155/156,
# pink|red at 172/173 (a saturation split was tried first, but the production
# enhance_saturation(1.4) boost pushes real pinks past any workable S ceiling
# and they read as red). Trade-off: a shadowed red cube whose hue drifts
# below 173 reads as pink — if that shows up on hardware, move the 172/173
# split down rather than re-overlapping the ranges.
# These are starting values: masks are computed AFTER enhance_saturation(1.4),
# so calibrate on hardware with the snapshot tool before trusting them.
COLOR_RANGES = {
    "red":    [((0, 100, 60), (8, 255, 255)), ((173, 100, 60), (180, 255, 255))],
    "orange": [((9, 120, 100), (19, 255, 255))],
    "yellow": [((20, 100, 100), (30, 255, 255))],
    "green":  [((40, 50, 50), (90, 255, 255))],
    "blue":   [((100, 150, 50), (128, 255, 255))],
    "purple": [((129, 60, 60), (155, 255, 255))],
    "pink":   [((156, 40, 100), (172, 255, 255))],
}

# --- PBVS (Position-Based Visual Servoing / industrial look-then-move) ---
# The servo does NOT chase pixels: each fresh frame measures the cube's
# position IN THE ROBOT BASE FRAME (depth deprojection + extrinsics, valid at
# any pose since R,t are TCP-relative), and the TCP is commanded directly at
# the grasp point — damped, clamped, and always workspace-validated (the
# target is the cube's own in-envelope position, so the servo can no longer
# walk out of the workspace chasing an offset aim pixel).
PBVS_TOL_MM         = 2.5   # |measured cube − grasp target| to count as aligned
PBVS_CONFIRM_FRAMES = 2     # consecutive fresh frames within tol to lock
PBVS_DAMPING        = 0.75  # fraction of the measured error corrected per move
PBVS_MAX_STEP_MM    = 60.0  # per-move clamp (moves are exact, so steps can be big)
PBVS_MAX_ITERS      = 12    # correction-move budget before aborting for a re-scan
PBVS_REFINE_Z_MM    = 250.0 # two-stage: after hover alignment, descend here and
                            # re-measure (sharper px/mm, shorter calibration lever
                            # arm) before the final blind descend; must stay >=
                            # DEPTH_TRUST_MIN_TCP_Z_MM. Equals APPROACH_Z_MM, so
                            # when GLOBAL_APPROACH already parks at 250 the
                            # refine descend is skipped automatically.
FRAME_FRESH_TIMEOUT_S = 1.5 # max wait for a post-move frame before using newest
LOST_TARGET_GRACE_S   = 1.0 # target unmatched this long → abort for an FSM re-scan

# --- Tool offset (suction cup vs TCP axis) ---
# The camera→gripper "watch on the wrist vs finger" geometry lives in the
# extrinsics t — PBVS needs no explicit camera offset. These are ONLY for
# residual cup-vs-TCP misalignment measured on hardware (expect ~0; if grasps
# land consistently offset by a fixed vector, put it here).
TOOL_OFFSET_X = 0.0
TOOL_OFFSET_Y = 0.0

# Servo target-continuity gate for the expected-pixel matching tier: a
# candidate farther than this from the target's projected pixel is a
# DIFFERENT cube, treated as a lost frame.
TRACK_MAX_JUMP_PX = 120

# --- World-frame target identity (multi-object servo) ---
# A cube's identity is its metric position in the ROBOT BASE frame (measured
# per-blob from Orbbec depth + extrinsics) — static while the arm moves, so
# the servo can never drift onto a different cube the way pixel-space
# nearest-to-aim selection could.
TARGET_MATCH_TOL_MM    = 45.0  # candidate must measure within this of the
                               # target's known base position to be OUR cube
TARGET_AVOID_RADIUS_MM = 40.0  # candidates within this of an already-delivered
                               # cube's position are never selected
DEPTH_TRUST_MIN_TCP_Z_MM = 250.0  # trust metric depth matching only at/above this
                                  # TCP height — the wrist Orbbec is inside its
                                  # minimum range near the table (grainy garbage)
DEPTH_MIN_PLAUSIBLE_MM   = 120.0  # reject per-candidate depths shorter than this
                                  # (inside the sensor's min range = invalid)

# --- Move completion (pose convergence) ---
# _safe_move issues non-blocking moves and detects completion by polling the
# actual pose — get_is_moving() lags after a non-blocking command and once let
# the FSM race ahead while the arm was still at TOP_VIEW_POSE.
POSE_TOL_MM    = 2.0    # XYZ distance to target to count as "arrived"
POSE_TOL_DEG   = 2.0    # yaw distance (covers yaw-only alignment moves)
MOVE_TIMEOUT_S = 30.0   # give up (and fail the move) after this long

# --- Yaw alignment (grasping rotated objects) ---
# The object's in-image angle (cv2.minAreaRect at the hover pose) is mapped to a
# gripper yaw command. Sign/offset depend on how the camera is mounted relative
# to the gripper jaws — verify on hardware with a deliberately rotated cube.
YAW_ALIGN_ENABLED  = True
CAMERA_YAW_SIGN    = 1.0    # flip to -1.0 if the gripper rotates the wrong way
CAMERA_YAW_OFFSET  = 0.0    # fixed mount rotation (deg) between image axes and jaws

# --- FSM retry / timeout defaults ---
VLA_TIMEOUT  = 20.0   # seconds for the PBVS lock attempt (2-4 exact moves plus
                      # fresh-frame waits fit comfortably; the budget covers
                      # lost-target grace periods too)
MAX_RETRIES  = 2

# --- HSV object detection (top-down) ---
MIN_BLOB_AREA_PX      = 100
FRONT_MIN_PRESENCE_PX = 1500

# --- Multi-object counting (top-down zone scans) ---
# A blob only counts as a cube in a zone when its area is at least this —
# stricter than MIN_BLOB_AREA_PX so mask fragments/phantoms don't inflate the
# per-zone count that PRE_CHECK/VERIFY compare. Calibrate on Thor: one cube at
# top view should land well above this.
ZONE_BLOB_MIN_AREA_PX = 800
# Merged-blob guard: a blob LARGER than this is most likely two touching cubes
# fused by the CLOSE kernel — its centroid is the seam between them, so it is
# excluded from counting/targeting (logged). None disables; set to ~1.8x the
# measured one-cube area on Thor.
ZONE_BLOB_MAX_AREA_PX = None
# Same guard for the wrist-camera servo view — WARN-ONLY (the servo keeps
# tracking; suction feedback / VERIFY catch a failed seam grasp). None disables.
SERVO_MAX_BLOB_AREA_PX = None

# Reference "one clean cube" top-view area (mm² in px, calibrate on Thor from
# a single unobstructed cube's logged blob area). Used only to decide whether
# an oversized blob is worth attempting a touching-cube split: candidates in
# roughly [1.5x, 3x] this area are tried; clearly-single or clearly-a-crowd
# blobs are left alone.
ONE_CUBE_AREA_PX = 3000

# Zone-COUNTING-only shape/area gates — looser than the servo's grasp-time
# gates (BLOB_MIN_SOLIDITY/BLOB_MAX_ASPECT below, MIN_BLOB_AREA_PX above): a
# cube worth GRASPING should look clean, but a cube worth COUNTING may be
# partially occluded or seen edge-on (mostly side face) because a neighbor is
# touching it — PRE_CHECK/VERIFY/auto task-detection should still see it.
ZONE_BLOB_MIN_SOLIDITY = 0.35
ZONE_BLOB_MAX_ASPECT   = 4.5

# --- VERIFY (count-delta) ---
# VERIFY passes when the target zone GREW: blob count >= baseline+1, or —
# when the dropped cube lands touching an existing one and the blobs merge —
# pixel mass grew by at least this (~0.6 of a cube's top-view area).
VERIFY_RECHECKS          = 2    # extra reads (0.5 s apart) before consuming a retry
VERIFY_MASS_DELTA_MIN_PX = 900

# --- Multi-cube eval ---
EVAL_MAX_CUBES = 6   # safety cap on eval.py's drain-the-source-zone loop

# --- Blended transport ---
# Corner-blend radius for the TRANSPORT chain (horizontal arrival → descend):
# the arm rounds the corner above the drop point instead of stopping. Keep
# <= ~1/3 of the shortest adjacent segment or the controller may reject it.
TRANSPORT_BLEND_RADIUS_MM = 30.0

# get_inverse_kinematics has no fixed ref_angles, so its solvability verdict
# for a Cartesian pose can shift with the arm's CURRENT joint configuration —
# a drop point can look reachable while sampled (arm near the grasp pose) and
# then fail its final pre-flight check (arm already lifted for transport).
# On that pre-flight rejection (no motion committed — safe to retry), redraw
# a fresh drop point and try again this many times before finally trying the
# nominal ZONES point once as a last resort.
TRANSPORT_IK_RETRIES = 3

# --- Recovery: hold-through-retry ---
# A mid-task fault does not have to mean dropping the cube: if check_grasp()
# still reports (or can't rule out) a hold, RECOVERY re-attempts TRANSPORT
# with a fresh drop point instead of releasing wherever the arm happened to
# fault. Bounded so a persistently hostile zone still terminates (releasing
# safely at HOME) instead of looping forever.
MAX_RECOVERY_RETRIES = 2

# --- Suction grasp feedback ---
# When True, a FALSE reading from check_grasp() after the post-grasp lift
# routes the FSM to RECOVERY instead of transporting an empty gripper.
# Keep False until ~10 grasps of logged "Suction feedback raw:" values on
# Thor confirm state==1 reliably tracks a real hold.
GRASP_FEEDBACK_ENABLED = False

# --- Depth-assisted perception (Orbbec local camera only) ---
# The wrist camera views the cube at an angle, so the color blob includes the
# cube's SIDE face and the 2D centroid is dragged toward it (the gripper then
# grabs a corner). With aligned depth, side-face pixels are FARTHER from the
# camera than top-face pixels: restrict the blob to the nearest-depth band and
# its centroid is the TRUE top-face center.
DEPTH_TOP_BAND_MM  = 30.0   # keep blob pixels within this depth of the nearest face
DEPTH_MIN_VALID_PX = 50     # min valid-depth pixels in the blob to trust depth logic

# Fragment fusion: the HSV mask splits on real cubes (top vs side face at the
# lit edge — hardware snapshot showed TWO 'red cube' boxes on one cube), and
# largest-contour selection then flip-flops between fragments, making the
# servo chase a teleporting centroid. CLOSE with this kernel re-fuses them.
MASK_FUSE_KERNEL_PX = 15

# Height-above-table gate: with depth, object candidates must rise at least
# this far off the table plane (robust far plane of the depth image). Excludes
# table-level phantoms (reflections/stains/shadows) no matter how red they look.
OBJECT_MIN_HEIGHT_MM = 5.0

# --- Target estimate smoothing ---
# The world-frame target estimate is an EMA over gated measurements: the
# 45 mm match gate rejects outliers, the EMA averages down depth noise so
# PBVS converges on a stable point instead of chasing per-frame jitter.
TARGET_EMA_ALPHA = 0.4     # weight of each new gated measurement

# --- Blob shape gates (lenient; reject glare streaks / cables, not cubes) ---
# A cube seen top-down is compact: solidity (area / convex-hull area) near 1
# and a squat minAreaRect. Bounds are deliberately loose so a half-shadowed
# or angled cube still passes.
BLOB_MIN_SOLIDITY = 0.65
BLOB_MAX_ASPECT   = 2.5    # long side / short side of the minAreaRect

# --- Ray-plane geometry (depth-free 2D→3D fallbacks) ---
# The table plane's base-frame z is ESTIMATED from the top-view depth frame at
# the start of each task (logged as "[Vision] Table plane z ≈ ..."); this
# override is only for the day depth can't estimate it — copy the logged value.
TABLE_Z_FALLBACK_MM = None
# Nominal object height: ray-plane localization of a blob centroid intersects
# the CUBE-TOP plane (table_z + this), since the visible centroid is the top face.
OBJECT_HEIGHT_MM = 40.0

# Depth-adaptive grasp sanity band: the measured height-above-table of the
# object's top face must fall inside this band, else the depth estimate is
# treated as garbage and the descend falls back to the fixed GRASP_Z.
GRASP_TOP_SANE_BAND_MM = (OBJECT_MIN_HEIGHT_MM, 90.0)

# --- Debug/demo video recording (--video flag) ---
VIDEO_FPS = 10   # mosaic recording rate; recorder runs in its own thread


