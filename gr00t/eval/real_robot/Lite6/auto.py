"""
Lite6 Continuous Auto Script
Usage: python auto.py [--ip 192.168.1.150]

Endlessly scans the workspace. Like SO100, it is zone-driven: it figures out
WHICH zone holds the target and picks the matching task (check_in / check_out /
check_back), then runs the FSM and loops. Ctrl+C for graceful shutdown.
"""

import argparse
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
)
from utils.system  import setup_signal_handlers, clear_stop_flag, set_in_use, clear_in_use, is_stop_requested
from utils.vision  import SafetyMonitor, check_color_presence, save_workspace_snapshot, open_camera, read_fresh
from utils.robot   import Lite6Controller
from utils.fsm     import Lite6FSM, FSMState
from utils.nlp     import TASK_ZONE_MAP

# Object the auto loop sorts. Could be extended to iterate KNOWN_OBJECTS.
AUTO_TARGET = "red cube"


def detect_next_task(front_rgb, target_object=AUTO_TARGET):
    """
    Zone-driven task selection (ported from SO100): scan each task's SOURCE zone
    ROI and return the first task whose source zone currently holds the object.
    Returns (task_type, source_zone, target_zone) or (None, None, None).
    """
    for task_type, zones in TASK_ZONE_MAP.items():
        source_zone = zones["source"]
        roi = ZONE_PIXEL_ROI.get(source_zone)
        is_present, px = check_color_presence(front_rgb, target_object, zone_roi=roi)
        if is_present:
            print(f"[AUTO-SCAN] '{target_object}' in {source_zone} ({px} px) → task {task_type}.")
            return task_type, source_zone, zones["target"]
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Lite6 Auto — continuous autonomous loop")
    parser.add_argument("--ip",        type=str,   default=DEFAULT_IP)
    parser.add_argument("--camera",    type=str,   default=None,
                        help="Camera source override: device index (e.g. 0) or stream URL")
    parser.add_argument("--timeout",   type=float, default=15.0)
    parser.add_argument("--retries",   type=int,   default=2)
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()

    camera_source = CAMERA_SOURCE
    if args.camera is not None:
        camera_source = int(args.camera) if args.camera.isdigit() else args.camera

    setup_signal_handlers()
    clear_stop_flag()
    set_in_use()

    print("\n=======================================================")
    print("  LITE6 AUTO MODE INITIALIZING...")
    print("=======================================================\n")

    cap = None
    robot = safety_monitor = None
    try:
        # Single physical camera (wrist), reused for the top-down view at TOP_VIEW_POSE.
        cap = open_camera(camera_source, "wrist camera")

        robot = Lite6Controller(args.ip)
        robot.connect()
        safety_monitor = SafetyMonitor(enabled=not args.no_safety)

        while True:
            if is_stop_requested():
                print("\n[AUTO] Stop command received — exiting loop gracefully.")
                break

            # Park at the top-view pose so the wrist camera sees the whole workspace.
            robot.move_to(*TOP_VIEW_POSE)
            time.sleep(0.5)

            ret, frame = read_fresh(cap)
            if not ret:
                print("[AUTO] Top-view read failed; retrying...")
                time.sleep(SCAN_INTERVAL)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            save_workspace_snapshot(frame_rgb, "snapshot_auto_front_before.jpg",
                                    AUTO_TARGET, OUTPUT_DIR_AUTO, ALL_ZONES_DICT)

            task_type, source_zone, target_zone = detect_next_task(frame_rgb)
            if task_type is None:
                print(f"[AUTO] '{AUTO_TARGET}' not found in any zone. Sleeping {SCAN_INTERVAL}s...")
                time.sleep(SCAN_INTERVAL)
                continue

            print(f"\n[AUTO] Task: {task_type.upper()}  |  Object: {AUTO_TARGET}")

            fsm = Lite6FSM(
                robot=robot, cap=cap,
                task_type=task_type, target_object=AUTO_TARGET,
                source_zone=source_zone, target_zone=target_zone,
                vla_timeout=args.timeout, max_retries=args.retries,
                safety_monitor=safety_monitor, should_stop_cb=is_stop_requested,
            )
            final_state = fsm.run()

            # Top-view "after" snapshot, then park home.
            robot.move_to(*TOP_VIEW_POSE)
            ret, frame = read_fresh(cap)
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_workspace_snapshot(frame_rgb, "snapshot_auto_front_after.jpg",
                                        AUTO_TARGET, OUTPUT_DIR_AUTO, ALL_ZONES_DICT)
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
    finally:
        print("\n[AUTO] Cleaning up...")
        if safety_monitor is not None:
            safety_monitor.stop()
        if robot is not None:
            try:
                robot.move_to(*HOME_POSE[:3])
            except Exception:
                pass
            robot.disconnect()
        if cap is not None:
            cap.release()
        clear_in_use()
        print("[AUTO] Shutdown complete.")


if __name__ == "__main__":
    main()
