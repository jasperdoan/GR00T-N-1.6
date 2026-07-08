"""
Wrist Camera Grasp Zone Calibrator

Lets you define:
  1. The grasp ROI bounding box — the region between the pincher fingers
     where a cube would appear when successfully grasped.
  2. Wrist-camera-specific HSV color ranges for each cube color.

Outputs a ready-to-paste constants block for vision_utils.py.

Usage:
  # From a saved frame:
  python calibrate_wrist.py --image ~/Downloads/wrist_frame.png

  # From a live camera feed (capture a frame first, then calibrate):
  python calibrate_wrist.py --camera 3
"""

import argparse
import cv2
import numpy as np
import os


def capture_frame_from_camera(camera_index: int) -> np.ndarray:
    print(f"Opening camera index {camera_index}. Press SPACE to capture a frame, 'q' to quit.")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            continue
        cv2.imshow("Wrist Camera — Press SPACE to capture", f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame = f.copy()
            print("Frame captured.")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


def calibrate(img: np.ndarray):
    # ── Step 1: Draw the grasp ROI ──────────────────────────────────────────
    print("\n==================================================")
    print(" GRASP ZONE ROI")
    print("==================================================")
    print("Draw a box around the region BETWEEN the gripper fingers.")
    print("This is where the cube face would appear when grasped.")
    print("Keep it tight — just the inner pinch area.")
    print("Click and drag. Press ENTER to confirm, 'c' to retry.\n")

    roi = cv2.selectROI("Select Grasp ROI", img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No ROI selected. Exiting.")
        return

    x, y, w, h = roi
    print(f"Grasp ROI saved: x={x}, y={y}, w={w}, h={h}\n")

    # Show a preview of the crop
    crop = img[y:y+h, x:x+w]
    cv2.imshow("Grasp ROI Preview (press any key)", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Step 2: Color picking ───────────────────────────────────────────────
    print("==================================================")
    print(" WRIST CAMERA COLOR CALIBRATION")
    print("==================================================")
    print("Click on the cube color in the image to print its HSV values.")
    print("Click multiple spots on the same cube to get a range.")
    print("Press 'r', 'b', 'y' to label the next clicks as red/blue/yellow.")
    print("Press 'q' when done.\n")

    hsv_img     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    display_img = img.copy()
    current_color_label = "red"

    # Track clicks per color for range computation
    samples: dict = {"red": [], "blue": [], "yellow": []}

    label_colors_bgr = {
        "red":    (0, 0, 255),
        "blue":   (255, 100, 0),
        "yellow": (0, 220, 220),
    }

    def click_hsv(event, x_c, y_c, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv_img[y_c, x_c]
            samples[current_color_label].append((int(h), int(s), int(v)))
            print(f"  [{current_color_label.upper()}] ({x_c},{y_c}) → HSV [{h}, {s}, {v}]")
            cv2.circle(display_img, (x_c, y_c), 4, label_colors_bgr[current_color_label], -1)
            cv2.imshow("Color Picker", display_img)

    cv2.imshow("Color Picker", display_img)
    cv2.setMouseCallback("Color Picker", click_hsv)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            current_color_label = "red"
            print("[Switched to RED]")
        elif key == ord('b'):
            current_color_label = "blue"
            print("[Switched to BLUE]")
        elif key == ord('y'):
            current_color_label = "yellow"
            print("[Switched to YELLOW]")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    # ── Step 3: Compute ranges and print output ─────────────────────────────
    print("\n==================================================")
    print(" CALIBRATION OUTPUT — paste into constants.py")
    print("==================================================\n")

    print(f"WRIST_GRASP_ROI = ({x}, {y}, {w}, {h})")
    print()

    print("WRIST_COLOR_RANGES = {")
    for color_name, pts in samples.items():
        if not pts:
            print(f'    "{color_name}": [],  # no samples taken')
            continue

        arr = np.array(pts)
        h_vals = arr[:, 0]
        s_vals = arr[:, 1]
        v_vals = arr[:, 2]

        # Add a margin around the sampled range
        margin_h = 10
        margin_s = 40
        margin_v = 40

        h_lo = max(0,   int(h_vals.min()) - margin_h)
        h_hi = min(180, int(h_vals.max()) + margin_h)
        s_lo = max(0,   int(s_vals.min()) - margin_s)
        v_lo = max(0,   int(v_vals.min()) - margin_v)

        print(f'    "{color_name}": [')

        # Red wraps around hue 0/180 — split into two ranges if needed
        if color_name == "red" and (h_lo < 15 or h_hi > 165):
            print(f"        (np.array([0,   {s_lo}, {v_lo}]), np.array([10,  255, 255])),")
            print(f"        (np.array([165, {s_lo}, {v_lo}]), np.array([180, 255, 255])),")
        else:
            print(f"        (np.array([{h_lo}, {s_lo}, {v_lo}]), np.array([{h_hi}, 255, 255])),")

        print("    ],")

        print(f"    # Sampled HSV: H={list(h_vals)} S={list(s_vals)} V={list(v_vals)}")

    print("}")
    print()
    print("# Minimum colour pixel count inside WRIST_GRASP_ROI to confirm cube presence")
    print("WRIST_MIN_COLOR_PX = 80  # tune this — start low, raise if false-positives appear")
    print()
    print("# Stability detection: max allowed mean-abs frame diff inside ROI (0-255)")
    print("WRIST_STABILITY_THR = 6.0  # tune — lower = stricter stillness requirement")
    print()
    print("# Number of consecutive stable+color frames required for grasp confirmation")
    print("WRIST_CONFIRM_FRAMES = 8")


def main():
    
    path = os.path.expanduser("~/Downloads/lite6/storage.jpg")
    img  = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    calibrate(img)


if __name__ == "__main__":
    main()