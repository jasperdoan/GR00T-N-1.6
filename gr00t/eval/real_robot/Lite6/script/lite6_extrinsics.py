"""
Camera→robot extrinsic calibration (run once the Orbbec is mounted and working).

Solves the constant (R, t) in   p_base = p_tcp + R · p_cam + t
valid because the camera is rigidly mounted and perception always happens at
the fixed orientation (roll/pitch/yaw = -180/0/0).

PROCEDURE (repeat for >= 4 well-spread object placements):
  1. The arm parks at TOP_VIEW_POSE and detects the object → p_cam recorded
     automatically (top-face point-cloud median) along with the TCP position.
  2. You jog the arm (UFACTORY Studio or freehand) until the TCP tip touches
     the object's TOP CENTER, then press Enter — the script reads p_base from
     the robot.
  3. The arm returns to TOP_VIEW_POSE for the next placement; move the object
     somewhere else and repeat.

Finally the script solves Kabsch least-squares, prints the RMS residual, and
saves data/extrinsics.npz. RMS should be a few mm; if it is large, re-run with
placements spread wider across the workspace.

    python3 script/lite6_extrinsics.py [--camera 0] [--ip 192.168.1.189] [--object "red cube"]
    python3 script/lite6_extrinsics.py --dry-run    # logic self-test, no hardware

Run from the Lite6 directory.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from utils.constants import CAMERA_SOURCE, DEFAULT_IP, TOP_VIEW_POSE, EXTRINSICS_PATH
from utils.vision import open_camera, read_fresh, localize_object_3d, solve_extrinsics, save_extrinsics


def dry_run():
    """No-hardware self-test: synthetic R,t must be recovered from noisy pairs."""
    rng = np.random.default_rng(42)
    # Ground-truth mount: downward-looking camera = 180° rotation about the
    # (1,1,0) axis — a PROPER rotation (det +1), as any physical mount must be.
    Rz = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
    t_true = np.array([35.0, -12.0, -180.0])

    cam_pts = rng.uniform([-80, -60, 150], [80, 60, 320], size=(8, 3))
    base_offsets = (Rz @ cam_pts.T).T + t_true + rng.normal(0, 0.5, size=(8, 3))

    R, t, rms = solve_extrinsics(cam_pts, base_offsets)
    print("Recovered R:\n", np.round(R, 3))
    print("Recovered t:", np.round(t, 2), " (true:", t_true, ")")
    print(f"RMS: {rms:.2f} mm (should be ~0.5 = the injected noise)")
    # t tolerance is loose: with few noisy samples the translation estimate has
    # a lever-arm amplification from the camera-frame point distances.
    assert rms < 2.0 and np.allclose(R, Rz, atol=0.01) and np.allclose(t, t_true, atol=3.0)
    print("DRY RUN PASSED — solver is correct.")


def main():
    parser = argparse.ArgumentParser(description="Lite6 camera→base extrinsic calibration")
    parser.add_argument("--camera", type=str, default=None, help="Orbbec device index (default CAMERA_SOURCE)")
    parser.add_argument("--ip",     type=str, default=DEFAULT_IP)
    parser.add_argument("--object", type=str, default="red cube")
    parser.add_argument("--dry-run", action="store_true", help="Solver self-test, no hardware")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    source = CAMERA_SOURCE
    if args.camera is not None:
        source = int(args.camera) if args.camera.isdigit() else args.camera
    if isinstance(source, str):
        print("ERROR: extrinsic calibration needs the LOCAL Orbbec (depth). "
              "Pass --camera <index>, not a stream URL.")
        sys.exit(1)

    from utils.robot import Lite6Controller   # deferred: needs xarm sdk + robot
    color_name = args.object.split()[0].lower()

    cam = open_camera(source, "calibration camera")
    if not cam.has_depth or cam.intrinsics is None:
        print("ERROR: camera provides no depth/intrinsics — cannot calibrate.")
        sys.exit(1)

    robot = Lite6Controller(args.ip)
    robot.connect()

    cam_pts, base_offsets = [], []
    try:
        while True:
            n = len(cam_pts)
            print(f"\n=== Sample {n + 1} (have {n}, need >= 4; Ctrl+C when done) ===")
            input(f"Place the {args.object} somewhere new in the workspace, then press Enter...")

            print("[Cal] Parking at TOP_VIEW_POSE...")
            robot.move_to(*TOP_VIEW_POSE)

            ret, frame_bgr = read_fresh(cam)
            dret, depth_mm = cam.read_depth()
            if not (ret and dret):
                print("[Cal] Camera read failed — try again.")
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            p_cam = localize_object_3d(frame_rgb, depth_mm, color_name, cam.intrinsics)
            if p_cam is None:
                print(f"[Cal] Could not localize the {args.object} in 3D — adjust placement/lighting.")
                continue
            tcp = robot.get_current_position()
            if tcp is None:
                print("[Cal] Could not read TCP position — try again.")
                continue
            print(f"[Cal] p_cam = ({p_cam[0]:.1f}, {p_cam[1]:.1f}, {p_cam[2]:.1f}) mm, "
                  f"TCP at ({tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f})")

            input("Now JOG the TCP tip to touch the object's TOP CENTER, then press Enter...")
            touch = robot.get_current_position()
            if touch is None:
                print("[Cal] Could not read touch position — sample discarded.")
                continue
            print(f"[Cal] p_base = ({touch[0]:.1f}, {touch[1]:.1f}, {touch[2]:.1f})")

            cam_pts.append(np.asarray(p_cam, dtype=np.float64))
            base_offsets.append(np.asarray(touch[:3], dtype=np.float64) - np.asarray(tcp[:3], dtype=np.float64))

            print("[Cal] Returning to TOP_VIEW_POSE (clear your hands)...")
            robot.move_to(*TOP_VIEW_POSE)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        robot.disconnect()

    if len(cam_pts) < 3:
        print(f"\nOnly {len(cam_pts)} sample(s) — need >= 3 (>= 4 recommended). Nothing saved.")
        sys.exit(1)

    R, t, rms = solve_extrinsics(np.array(cam_pts), np.array(base_offsets))
    print(f"\nSolved extrinsics from {len(cam_pts)} samples — RMS {rms:.2f} mm")
    if rms > 8.0:
        print("WARNING: RMS is high. Re-run with the object placed more widely apart, "
              "and make sure the TCP truly touches the object's top center.")
    save_extrinsics(R, t, rms, EXTRINSICS_PATH)
    print("Done. SCANNING will now use 3D localization automatically (with --camera <index>).")


if __name__ == "__main__":
    main()
