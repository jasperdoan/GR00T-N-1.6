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
    TOP_DOWN_CAM_IDX, WRIST_CAM_IDX,
    HOME_POSE,
    ZONES,
)
from utils.system  import setup_signal_handlers, clear_stop_flag, set_in_use, clear_in_use, is_stop_requested
from utils.nlp     import parse_instruction
from utils.vision  import SafetyMonitor, save_workspace_snapshot
from utils.robot   import Lite6Controller
from utils.fsm     import Lite6FSM, FSMState

OUTPUT_DIR = "/tmp/lite6_eval"

ALL_ZONES_DICT = {
    "Check In":  (0, 0, 200, 200),   # TODO: replace with real pixel ROIs after calibration
    "Storage":   (200, 0, 200, 200),
    "Check Out": (400, 0, 200, 200),
}


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

    # Parse instruction
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

    # Open cameras
    cap_top   = cv2.VideoCapture(TOP_DOWN_CAM_IDX)
    cap_wrist = cv2.VideoCapture(WRIST_CAM_IDX)

    robot          = Lite6Controller(args.ip)
    safety_monitor = SafetyMonitor(enabled=not args.no_safety)

    try:
        robot.connect()

        # Home position
        robot.move_to(*HOME_POSE[:3])
        time.sleep(0.5)

        # Before snapshot
        ret, frame = cap_top.read()
        if ret:
            import cv2 as _cv2
            frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            save_workspace_snapshot(frame_rgb, "snapshot_eval_front_before.jpg",
                                    target_object, OUTPUT_DIR, ALL_ZONES_DICT)

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

        # Return home
        robot.move_to(*HOME_POSE[:3])
        time.sleep(1.0)

        # After snapshot
        ret, frame = cap_top.read()
        if ret:
            import cv2 as _cv2
            frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            save_workspace_snapshot(frame_rgb, "snapshot_eval_front_after.jpg",
                                    target_object, OUTPUT_DIR, ALL_ZONES_DICT)

        if final_state == FSMState.DONE:
            print("\n[EVAL] COMPLETE — task succeeded.")
        else:
            print("\n[EVAL] FAILED — task did not complete.")

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[EVAL] Cleaning up...")
        safety_monitor.stop()
        try:
            robot.move_to(*HOME_POSE[:3])
        except Exception:
            pass
        robot.disconnect()
        cap_top.release()
        cap_wrist.release()
        clear_in_use()
        print("[EVAL] Shutdown complete.")


if __name__ == "__main__":
    main()
