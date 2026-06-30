"""
Lite6 Single-Shot Evaluation Script
Usage: python eval.py --instruction "check in red cube"
"""

import argparse
import sys
import time

import cv2

from utils.constants import (
    DEFAULT_IP,
    WRIST_CAM_IDX,
    HOME_POSE, TOP_VIEW_POSE,
    ALL_ZONES_DICT,
    OUTPUT_DIR_EVAL,
)
from utils.system  import setup_signal_handlers, clear_stop_flag, set_in_use, clear_in_use, is_stop_requested
from utils.nlp     import parse_instruction
from utils.vision  import SafetyMonitor, save_workspace_snapshot, open_camera, read_fresh
from utils.robot   import Lite6Controller
from utils.fsm     import Lite6FSM, FSMState


def _top_view_snapshot(robot, cap, filename, target_object):
    """Park at TOP_VIEW_POSE (the single camera's top-down view), then snapshot."""
    robot.move_to(*TOP_VIEW_POSE)
    ret, frame = read_fresh(cap)
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        save_workspace_snapshot(frame_rgb, filename, target_object,
                                OUTPUT_DIR_EVAL, ALL_ZONES_DICT)


def main():
    parser = argparse.ArgumentParser(description="Lite6 Eval — single-shot NLP task")
    parser.add_argument("--instruction", type=str, default="check in red cube",
                        help='Natural language instruction e.g. "check in red cube"')
    parser.add_argument("--ip",        type=str,   default=DEFAULT_IP)
    parser.add_argument("--timeout",   type=float, default=15.0, help="Visual servoing timeout (s)")
    parser.add_argument("--retries",   type=int,   default=2,    help="Max retries on failure")
    parser.add_argument("--no-safety", action="store_true",      help="Disable hand detection safety")
    args = parser.parse_args()

    setup_signal_handlers()
    clear_stop_flag()
    set_in_use()

    # Parse instruction up front so a bad string fails before touching hardware.
    try:
        task_type, target_object, source_zone, target_zone = parse_instruction(args.instruction)
    except ValueError as e:
        print(e)
        clear_in_use()
        sys.exit(1)

    print(f"\n[EVAL] Instruction   : {args.instruction}")
    print(f"[EVAL] Task type     : {task_type}")
    print(f"[EVAL] Target object : {target_object}")
    print(f"[EVAL] Source zone   : {source_zone}")
    print(f"[EVAL] Target zone   : {target_zone}\n")

    cap = None
    robot = safety_monitor = None
    try:
        # Single physical camera (wrist), reused for the top-down view at TOP_VIEW_POSE.
        cap = open_camera(WRIST_CAM_IDX, "wrist camera")

        robot = Lite6Controller(args.ip)
        robot.connect()                       # raises on failure — no blind runs
        safety_monitor = SafetyMonitor(enabled=not args.no_safety)

        robot.move_to(*HOME_POSE[:3])
        time.sleep(0.5)

        _top_view_snapshot(robot, cap, "snapshot_eval_front_before.jpg", target_object)

        fsm = Lite6FSM(
            robot=robot, cap=cap,
            task_type=task_type, target_object=target_object,
            source_zone=source_zone, target_zone=target_zone,
            vla_timeout=args.timeout, max_retries=args.retries,
            safety_monitor=safety_monitor, should_stop_cb=is_stop_requested,
        )
        final_state = fsm.run()

        _top_view_snapshot(robot, cap, "snapshot_eval_front_after.jpg", target_object)
        robot.move_to(*HOME_POSE[:3])
        time.sleep(1.0)

        if final_state == FSMState.DONE:
            print("\n[EVAL] COMPLETE — task succeeded.")
        else:
            print("\n[EVAL] FAILED — task did not complete.")

    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"\n[EVAL] Fatal: {e}")
    finally:
        print("\n[EVAL] Cleaning up...")
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
        print("[EVAL] Shutdown complete.")


if __name__ == "__main__":
    main()
