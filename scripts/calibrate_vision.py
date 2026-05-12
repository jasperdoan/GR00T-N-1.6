import cv2
import numpy as np
import os

def calibrate():
    # 1. Load the image
    img_path = os.path.expanduser("~/Downloads/opencv__dev_video3.png")
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    img = cv2.imread(img_path)
    # Optional: Resize if the image is too huge to fit on screen
    # img = cv2.resize(img, (640, 480)) 
    
    zones = {}
    zone_names = ["Storage Area", "Check-in Area", "Check-out Area"]

    print("==================================================")
    print(" ZONE CALIBRATION")
    print("==================================================")
    for name in zone_names:
        print(f"-> Please draw a box around the {name}.")
        print("   Click and drag to draw. Press ENTER to confirm, or 'c' to cancel/retry.")
        
        # Select ROI opens a window, lets you draw, and returns (x, y, w, h)
        roi = cv2.selectROI(f"Select {name}", img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        
        zones[name] = roi
        print(f"   {name} saved as: {roi}\n")

    print("==================================================")
    print(" COLOR CALIBRATION")
    print("==================================================")
    print("A window will now open.")
    print("-> CLICK on the red, blue, and yellow cubes to print their HSV values.")
    print("-> Press 'q' to quit when you are done.\n")

    # Convert image to HSV for color picking
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mouse callback function to print HSV values on click
    def click_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, s, v = hsv_img[y, x]
            print(f"Clicked at (X:{x}, Y:{y}) -> HSV: [{h}, {s}, {v}]")
            
            # Draw a tiny circle where clicked
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
            cv2.imshow("Color Picker", img)

    cv2.imshow("Color Picker", img)
    cv2.setMouseCallback("Color Picker", click_color)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\n==================================================")
    print(" SUMMARY FOR eval_so100.py")
    print("==================================================")
    print("Copy these into your main script later:\n")
    print(f"STORAGE_ZONE = {zones['Storage Area']}")
    print(f"CHECK_IN_ZONE = {zones['Check-in Area']}")
    print(f"CHECK_OUT_ZONE = {zones['Check-out Area']}")
    print("\nUse the HSV values you clicked to set your color ranges!")
    print("Note: OpenCV Hue goes from 0-180, Saturation 0-255, Value 0-255.")

if __name__ == "__main__":
    calibrate()