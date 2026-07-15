"""
Lite6 Continuous Auto Script
Usage: python auto.py [--ip 192.168.1.150]

Endlessly scans the workspace. Like SO100, it is zone-driven: it figures out
WHICH zone holds the target and picks the matching task (check_in / check_out /
check_back), then runs the FSM and loops. Ctrl+C for graceful shutdown.
"""

import argparse
import sys
import time

import cv2

from utils.constants import (
    DEFAULT_IP,
    CAMERA_SOURCE,
    HOME_POSE, TOP_VIEW_POSE,
    ZONE_PIXEL_ROI,
    ALL_ZONES_DICT,
    OUTPUT_DIR_AUTO,
    SCAN_INTERVAL,
    VLA_TIMEOUT,
    KNOWN_OBJECTS,
    CAMERA_FAIL_LIMIT,
    EMPTY_SCAN_GIVE_UP,
)
from utils.system  import (
    setup_signal_handlers, clear_stop_flag, set_stop_flag,
    set_in_use, clear_in_use, is_stop_requested,
)
from utils.vision  import count_objects_in_zone, save_workspace_snapshot, open_camera, read_fresh, RunRecorder
from utils.robot   import Lite6Controller
from utils.fsm     import Lite6FSM, FSMState
from utils.nlp     import TASK_ZONE_MAP

# Default object the auto loop sorts; override per-run with --object.
AUTO_TARGET = "red cube"


def detect_next_task(front_rgb, target_object=AUTO_TARGET, depth_mm=None):
    """
    Zone-driven task selection (ported from SO100): scan each task's SOURCE zone
    ROI and return the first task whose source zone currently holds >= 1 of the
    object. TASK_ZONE_MAP's ordering IS the priority — e.g. a cube sitting in
    check_out legitimately triggers check_back; that's the intended sorting
    behavior, not a stray pick. With depth, the height gate physically excludes
    table-level phantoms from the census.
    Returns (task_type, source_zone, target_zone) or (None, None, None).
    """
    for task_type, zones in TASK_ZONE_MAP.items():
        source_zone = zones["source"]
        roi = ZONE_PIXEL_ROI.get(source_zone)
        count, _blobs, px = count_objects_in_zone(front_rgb, target_object,
                                                  zone_roi=roi, depth_mm=depth_mm)
        if count >= 1:
            print(f"[AUTO-SCAN] {count}x '{target_object}' in {source_zone} ({px} px) "
                  f"→ task {task_type}.")
            return task_type, source_zone, zones["target"]
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Lite6 Auto — continuous autonomous loop")
    parser.add_argument("--ip",        type=str,   default=DEFAULT_IP)
    parser.add_argument("--camera",    type=str,   default=None,
                        help="Camera source override: Orbbec device index (e.g. 0) or stream URL")
    parser.add_argument("--object",    type=str,   default=AUTO_TARGET,
                        help=f"Object to sort (must be in KNOWN_OBJECTS; default: {AUTO_TARGET!r})")
    parser.add_argument("--timeout",   type=float, default=VLA_TIMEOUT)
    parser.add_argument("--retries",   type=int,   default=2)
    parser.add_argument("--video",     action="store_true",
                        help="Record a 2x2 debug mosaic MP4 (color+detection | depth | masks) during the run")
    parser.add_argument("--max-runtime-h", type=float, default=0.0,
                        help="Exit cleanly at an idle boundary after this many hours (0 = run forever). "
                             "Used by the supervisor for periodic fresh restarts instead of a blind cron kill.")
    args = parser.parse_args()

    camera_source = CAMERA_SOURCE
    if args.camera is not None:
        camera_source = int(args.camera) if args.camera.isdigit() else args.camera

    target_object = args.object.strip().lower()
    if target_object not in KNOWN_OBJECTS:
        print(f"Unknown object {args.object!r}. Known objects:")
        for obj in KNOWN_OBJECTS:
            print(f"  - {obj}")
        sys.exit(1)

    setup_signal_handlers()
    clear_stop_flag()
    set_in_use()

    print("\n=======================================================")
    print("  LITE6 AUTO MODE INITIALIZING...")
    print("=======================================================\n")

    cap = None
    robot = recorder = None
    exit_code = 0
    start_time = time.time()
    camera_failures = 0
    try:
        # Single physical camera (wrist), reused for the top-down view at TOP_VIEW_POSE.
        cap = open_camera(camera_source, "wrist camera")
        if args.video:
            recorder = RunRecorder(cap, target_object, OUTPUT_DIR_AUTO)

        robot = Lite6Controller(args.ip)
        robot.connect()

        consecutive_empty = 0
        while True:
            if is_stop_requested():
                print("\n[AUTO] Stop command received — exiting loop gracefully.")
                break

            # Idle boundary: robot is parked, no task in flight — the only safe
            # place to honor the periodic fresh-restart request.
            if args.max_runtime_h > 0 and (time.time() - start_time) >= args.max_runtime_h * 3600:
                print(f"\n[AUTO] Max runtime ({args.max_runtime_h:.1f} h) reached at idle boundary — "
                      f"exiting for a fresh restart.")
                break

            # Park at the top-view pose so the wrist camera sees the whole workspace.
            robot.move_to(*TOP_VIEW_POSE)
            time.sleep(0.5)

            ret, frame = read_fresh(cap)
            if not ret:
                camera_failures += 1
                print(f"[AUTO] Top-view read failed ({camera_failures}/{CAMERA_FAIL_LIMIT}); retrying...")
                if camera_failures >= CAMERA_FAIL_LIMIT:
                    raise RuntimeError(
                        f"Camera produced no frames for {CAMERA_FAIL_LIMIT} consecutive scans — "
                        f"exiting so a supervisor can restart with a fresh camera."
                    )
                time.sleep(SCAN_INTERVAL)
                continue
            camera_failures = 0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_mm = None
            if getattr(cap, "has_depth", False):
                dret, depth = cap.read_depth()
                if dret:
                    depth_mm = depth

            save_workspace_snapshot(frame_rgb, "snapshot_auto_front_before.jpg",
                                    target_object, OUTPUT_DIR_AUTO, ALL_ZONES_DICT)

            task_type, source_zone, target_zone = detect_next_task(frame_rgb, target_object,
                                                                   depth_mm=depth_mm)
            if task_type is None:
                consecutive_empty += 1
                print(f"[AUTO] '{target_object}' not found in any zone "
                      f"({consecutive_empty}/{EMPTY_SCAN_GIVE_UP}).")
                if consecutive_empty >= EMPTY_SCAN_GIVE_UP:
                    # Empty workspace: park, set the stop flag (so the 24/7
                    # supervisor stops instead of restarting an idle loop) and
                    # exit cleanly. A human restarts the demo with new cubes.
                    print("[AUTO] Workspace empty — parking, setting the stop flag, "
                          "exiting cleanly.")
                    robot.move_to(*HOME_POSE[:3])
                    set_stop_flag()
                    break
                time.sleep(SCAN_INTERVAL)
                continue
            consecutive_empty = 0

            print(f"\n[AUTO] Task: {task_type.upper()}  |  Object: {target_object}")

            fsm = Lite6FSM(
                robot=robot, cap=cap,
                task_type=task_type, target_object=target_object,
                source_zone=source_zone, target_zone=target_zone,
                vla_timeout=args.timeout, max_retries=args.retries,
                should_stop_cb=is_stop_requested,
            )
            final_state = fsm.run()

            # Top-view "after" snapshot, then park home.
            robot.move_to(*TOP_VIEW_POSE)
            ret, frame = read_fresh(cap)
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_workspace_snapshot(frame_rgb, "snapshot_auto_front_after.jpg",
                                        target_object, OUTPUT_DIR_AUTO, ALL_ZONES_DICT)
            robot.move_to(*HOME_POSE[:3])
            time.sleep(1.0)

            if final_state == FSMState.DONE:
                print("\n[AUTO] Task complete. Rescanning...")
            else:
                print("\n[AUTO] Task failed. Rescanning after pause...")
                time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"\n[AUTO] Fatal: {e}")
        exit_code = 1   # nonzero so a supervisor logs this as a crash, not a clean exit
    finally:
        print("\n[AUTO] Cleaning up...")
        if recorder is not None:
            recorder.stop()
        if robot is not None:
            try:
                robot.resume()   # clear any lingering pause so the home move actually runs
                robot.move_to(*HOME_POSE[:3])
            except Exception:
                pass
            robot.disconnect()
        if cap is not None:
            cap.release()
        clear_in_use()
        print("[AUTO] Shutdown complete.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
