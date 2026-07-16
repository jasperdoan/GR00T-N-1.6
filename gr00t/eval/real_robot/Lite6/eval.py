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
    CAMERA_SOURCE,
    HOME_POSE, TOP_VIEW_POSE,
    ZONE_PIXEL_ROI,
    ALL_ZONES_DICT,
    OUTPUT_DIR_EVAL,
    VLA_TIMEOUT,
    EVAL_MAX_CUBES,
)
from utils.system  import setup_signal_handlers, clear_stop_flag, set_in_use, clear_in_use, is_stop_requested
from utils.nlp     import parse_instruction
from utils.vision  import count_objects_in_zone, save_workspace_snapshot, open_camera, read_fresh, RunRecorder
from utils.robot   import Lite6Controller
from utils.fsm     import Lite6FSM, FSMState


def _read_top_frame(robot, cap):
    """Park at TOP_VIEW_POSE (the single camera's top-down view) and read a frame (RGB)."""
    robot.move_to(*TOP_VIEW_POSE)
    ret, frame = read_fresh(cap)
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _read_top_rgbd(robot, cap):
    """Top view frame + aligned depth (depth None on stream cameras / failed read)."""
    frame_rgb = _read_top_frame(robot, cap)
    if frame_rgb is None:
        return None, None
    depth_mm = None
    if getattr(cap, "has_depth", False):
        ret, depth = cap.read_depth()
        if ret:
            depth_mm = depth
    return frame_rgb, depth_mm


def _top_view_snapshot(robot, cap, filename, target_object):
    frame_rgb = _read_top_frame(robot, cap)
    if frame_rgb is not None:
        save_workspace_snapshot(frame_rgb, filename, target_object,
                                OUTPUT_DIR_EVAL, ALL_ZONES_DICT)


def main():
    parser = argparse.ArgumentParser(description="Lite6 Eval — single-shot NLP task")
    parser.add_argument("--instruction", type=str, default="check in red cube",
                        help='Natural language instruction e.g. "check in red cube"')
    parser.add_argument("--ip",        type=str,   default=DEFAULT_IP)
    parser.add_argument("--camera",    type=str,   default=None,
                        help="Camera source override: device index (e.g. 0) or stream URL")
    parser.add_argument("--timeout",   type=float, default=VLA_TIMEOUT, help="Visual servoing timeout (s)")
    parser.add_argument("--retries",   type=int,   default=2,    help="Max retries on failure")
    parser.add_argument("--video",     action="store_true",
                        help="Record a 2x2 debug mosaic MP4 (color+detection | depth | masks) during the run")
    args = parser.parse_args()

    camera_source = CAMERA_SOURCE
    if args.camera is not None:
        camera_source = int(args.camera) if args.camera.isdigit() else args.camera

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
    robot = recorder = None
    try:
        # Single physical camera (wrist), reused for the top-down view at TOP_VIEW_POSE.
        cap = open_camera(camera_source, "wrist camera")
        if args.video:
            recorder = RunRecorder(cap, target_object, OUTPUT_DIR_EVAL)

        robot = Lite6Controller(args.ip)
        robot.connect()                       # raises on failure — no blind runs

        robot.move_to(*HOME_POSE[:3])
        time.sleep(0.5)

        _top_view_snapshot(robot, cap, "snapshot_eval_front_before.jpg", target_object)

        # Multi-object: drain the source zone — "check in red cube" with two
        # red cubes in check-in checks in BOTH (one FSM run per cube, fresh
        # retries and zone baselines each time). EVAL_MAX_CUBES is a runaway
        # guard against a miscounting mask.
        completed = 0
        final_state = None
        source_roi = ZONE_PIXEL_ROI.get(source_zone)
        for cycle in range(1, EVAL_MAX_CUBES + 1):
            if is_stop_requested():
                print("[EVAL] Stop requested — ending early.")
                break

            frame_rgb, depth_mm = _read_top_rgbd(robot, cap)
            if frame_rgb is None:
                print("[EVAL] Could not read the top view — ending.")
                break
            count, _blobs, _mass = count_objects_in_zone(frame_rgb, target_object,
                                                         source_roi, depth_mm=depth_mm)
            if count == 0:
                if cycle == 1:
                    # Preserve the showroom "no-no" failure signal the FSM's
                    # PRE_CHECK used to give for an impossible instruction.
                    print(f"[EVAL] No '{target_object}' in {source_zone} — signalling failure.")
                    robot.wrist_wiggle()
                else:
                    print(f"[EVAL] Source zone {source_zone} empty after {completed} cube(s).")
                break

            print(f"\n[EVAL] Cycle {cycle}: {count} '{target_object}'(s) in {source_zone}.")
            fsm = Lite6FSM(
                robot=robot, cap=cap,
                task_type=task_type, target_object=target_object,
                source_zone=source_zone, target_zone=target_zone,
                vla_timeout=args.timeout, max_retries=args.retries,
                should_stop_cb=is_stop_requested,
            )
            final_state = fsm.run()
            if final_state != FSMState.DONE:
                # DONE_RETURNED included: the target zone is full, so draining
                # further cubes would just carry each one there and back.
                break
            completed += 1
        else:
            print(f"[EVAL] Safety cap EVAL_MAX_CUBES={EVAL_MAX_CUBES} reached.")

        _top_view_snapshot(robot, cap, "snapshot_eval_front_after.jpg", target_object)
        robot.move_to(*HOME_POSE[:3])
        time.sleep(1.0)

        if final_state == FSMState.DONE_RETURNED:
            print(f"\n[EVAL] RETURNED — {target_zone} is full; the cube went back "
                  f"to its pickup point ({completed} cube(s) delivered first).")
        elif final_state is not None and final_state != FSMState.DONE:
            print(f"\n[EVAL] FAILED — task did not complete ({completed} cube(s) delivered first).")
        elif completed == 0:
            print(f"\n[EVAL] NOTHING TO DO — no '{target_object}' found in {source_zone}.")
        else:
            print(f"\n[EVAL] COMPLETE — {completed} cube(s) delivered.")

    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"\n[EVAL] Fatal: {e}")
    finally:
        print("\n[EVAL] Cleaning up...")
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
        print("[EVAL] Shutdown complete.")


if __name__ == "__main__":
    main()
