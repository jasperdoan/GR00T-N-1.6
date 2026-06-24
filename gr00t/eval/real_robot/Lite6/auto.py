"""
Lite6 Continuous Auto Script
Usage: python auto.py [--ip 192.168.1.150]

Endlessly scans the workspace and performs check_in → storage for any
detected known object, then loops. Ctrl+C for graceful shutdown.
"""

import time
import sys

import cv2

from utils.constants import (
    DEFAULT_IP,
    TOP_DOWN_CAM_IDX, WRIST_CAM_IDX,
    HOME_POSE,
    KNOWN_OBJECTS,
    ZONES,
    ALL_ZONES_DICT,
    OUTPUT_DIR,
    DEFAULT_TASK,
    SCAN_INTERVAL
)
from utils.system  import setup_signal_handlers, clear_stop_flag, set_in_use, clear_in_use, is_stop_requested
from utils.vision  import SafetyMonitor, check_color_presence, save_workspace_snapshot
from utils.robot   import Lite6Controller
from utils.fsm     import Lite6FSM, FSMState
from utils.nlp     import TASK_ZONE_MAP


def detect_any_object(cap_top) -> tuple:
    """
    Scan the top-down camera for any KNOWN_OBJECTS.
    Returns (target_object, task_type) for the first match, or (None, None).
    """
    ret, frame = cap_top.read()
    if not ret:
        return None, None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for obj in KNOWN_OBJECTS:
        is_present, px = check_color_presence(frame_rgb, obj)
        if is_present:
            print(f"[AUTO-SCAN] Found '{obj}' ({px} px).")
            return obj, DEFAULT_TASK

    return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lite6 Auto — continuous autonomous loop")
    parser.add_argument("--ip",        type=str,   default=DEFAULT_IP)
    parser.add_argument("--timeout",   type=float, default=15.0)
    parser.add_argument("--retries",   type=int,   default=2)
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()

    setup_signal_handlers()
    clear_stop_flag()
    set_in_use()

    print("\n=======================================================")
    print("  LITE6 AUTO MODE INITIALIZING...")
    print("=======================================================\n")

    cap_top   = cv2.VideoCapture(TOP_DOWN_CAM_IDX)
    cap_wrist = cv2.VideoCapture(WRIST_CAM_IDX)

    robot          = Lite6Controller(args.ip)
    safety_monitor = SafetyMonitor(enabled=not args.no_safety)

    try:
        robot.connect()

        while True:
            if is_stop_requested():
                print("\n[AUTO] Stop command received — exiting loop gracefully.")
                break

            robot.move_to(*HOME_POSE[:3])
            time.sleep(1.0)

            # Before snapshot
            ret, frame = cap_top.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_workspace_snapshot(frame_rgb, "snapshot_auto_front_before.jpg",
                                        "red cube", OUTPUT_DIR, ALL_ZONES_DICT)

            target_object, task_type = detect_any_object(cap_top)

            if target_object is None:
                print(f"[AUTO] No known objects found. Sleeping {SCAN_INTERVAL}s before retry...")
                time.sleep(SCAN_INTERVAL)
                continue

            zones      = TASK_ZONE_MAP[task_type]
            source_zone = zones["source"]
            target_zone = zones["target"]

            print(f"\n[AUTO] Task: {task_type.upper()}  |  Object: {target_object}")

            fsm = Lite6FSM(
                robot=robot,
                cap_top=cap_top,
                cap_wrist=cap_wrist,
                task_type=task_type,
                target_object=target_object,
                source_zone=source_zone,
                target_zone=target_zone,
                vla_timeout=args.timeout,
                max_retries=args.retries,
                safety_monitor=safety_monitor,
                should_stop_cb=is_stop_requested,
            )

            final_state = fsm.run()

            robot.move_to(*HOME_POSE[:3])
            time.sleep(1.0)

            # After snapshot
            ret, frame = cap_top.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_workspace_snapshot(frame_rgb, "snapshot_auto_front_after.jpg",
                                        target_object, OUTPUT_DIR, ALL_ZONES_DICT)

            if final_state == FSMState.DONE:
                print("\n[AUTO] Task complete. Rescanning...")
            else:
                print("\n[AUTO] Task failed. Rescanning after pause...")
                time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[AUTO] Cleaning up...")
        safety_monitor.stop()
        try:
            robot.move_to(*HOME_POSE[:3])
        except Exception:
            pass
        robot.disconnect()
        cap_top.release()
        cap_wrist.release()
        clear_in_use()
        print("[AUTO] Shutdown complete.")


if __name__ == "__main__":
    main()
