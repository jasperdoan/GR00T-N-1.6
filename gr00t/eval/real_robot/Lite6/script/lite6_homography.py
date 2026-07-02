import cv2
import numpy as np
import os
# from xarm.wrapper import XArmAPI

# ============================================================================
# HOW TO USE THIS SCRIPT (STEP-BY-STEP)
# ============================================================================
#
# STEP 1: PHYSICAL SETUP
#   - Mount your camera looking at the workspace. Tape your ChArUco board down.
#   - Ensure your image is saved at ~/Downloads/captured_image_custom_zed.jpg
#
# STEP 2: RUN CALIBRATION
#   - Set `MODE = "CALIBRATE"` at the bottom of this script and run it.
#   - An image window will open. CLICK 4 POINTS on the calibration board.
#     (e.g., the 4 outermost corners). Red dots will appear as you click.
#
# STEP 3: ENTER ROBOT COORDINATES
#   - After clicking 4 points, the image window will close.
#   - Look at your terminal/console. It will prompt you for the Robot X and Y 
#     coordinates (in mm) for each point you clicked.
#   - Use UFACTORY Studio to jog the arm to those physical points and type 
#     the values into the console.
#   - The script will calculate and save `homography_matrix.npy`.
#
# STEP 4: RUNTIME (PICK AND PLACE)
#   - Remove the calibration board.
#   - Set `MODE = "RUNTIME"` and run the script.
#   - It will load the matrix, simulate finding an object, translate the 
#     pixels to real mm, and command the Lite 6 to move there.
# ============================================================================

MATRIX_FILENAME = "homography_matrix.npy"
IMAGE_PATH = os.path.expanduser("~/Downloads/captured_image.jpg")

# Global variables for the OpenCV mouse callback
clicked_points = []
display_img = None

def mouse_callback(event, x, y, flags, param):
    """
    OpenCV callback to capture X,Y pixel coordinates when the user left-clicks.
    """
    global clicked_points, display_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            point_num = len(clicked_points)
            print(f"Point {point_num} selected at pixel: (u={x}, v={y})")
            
            # Draw a circle and point number on the image for visual feedback
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_img, f"{point_num}", (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Select 4 Points", display_img)


def get_float_input(prompt):
    """Helper function to ensure the user types a valid number."""
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("  Invalid input. Please enter a valid number (e.g., 250.5).")


def calibrate_and_save():
    """
    Opens the image, lets the user click 4 points, prompts for robot coords,
    and computes/saves the Homography matrix.
    """
    global clicked_points, display_img
    print("--- RUNNING INTERACTIVE CALIBRATION ---")
    
    # 1. Load the image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return
        
    display_img = cv2.imread(IMAGE_PATH)
    
    # 2. Show image and wait for user to click 4 points
    cv2.namedWindow("Select 4 Points", cv2.WINDOW_NORMAL) # Allows resizing if image is huge
    cv2.imshow("Select 4 Points", display_img)
    cv2.setMouseCallback("Select 4 Points", mouse_callback)
    
    print("Please click 4 distinct points on the calibration board in the image window...")
    
    # Loop until 4 points are clicked
    while len(clicked_points) < 4:
        # Wait for 50ms, check if user closed window manually
        if cv2.waitKey(50) & 0xFF == 27: # ESC key to exit early
            print("Calibration cancelled.")
            cv2.destroyAllWindows()
            return

    # Wait a brief moment so the user sees the 4th dot, then close window
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    # 3. Prompt user for Robot Coordinates via console
    print("\n--- ENTER ROBOT COORDINATES ---")
    print("For each pixel point you clicked, enter the physical Robot X,Y coordinates in mm.")
    
    robot_points = []
    for i, pt in enumerate(clicked_points):
        print(f"\nPoint {i+1} [Pixel u={pt[0]}, v={pt[1]}]:")
        rob_x = get_float_input(f"  Enter Robot X for Point {i+1} (mm): ")
        rob_y = get_float_input(f"  Enter Robot Y for Point {i+1} (mm): ")
        robot_points.append([rob_x, rob_y])
        
    # 4. Convert lists to float32 numpy arrays for OpenCV
    pixel_array = np.array(clicked_points, dtype=np.float32)
    robot_array = np.array(robot_points, dtype=np.float32)
    
    # 5. Compute Homography Matrix
    H, status = cv2.findHomography(pixel_array, robot_array, cv2.RANSAC)
    
    if H is not None:
        print("\nCalculated Homography Matrix:\n", H)
        # Save the matrix to a file
        np.save(MATRIX_FILENAME, H)
        print(f"Success! Matrix saved to {MATRIX_FILENAME}")
    else:
        print("\nError: Could not calculate Homography matrix from these points.")


def pixel_to_robot(u, v, H_matrix):
    """
    Converts a single (u, v) pixel coordinate to Robot (X, Y) in mm.
    """
    pixel_vector = np.array([u, v, 1.0])
    mapped_vector = np.dot(H_matrix, pixel_vector)
    
    # Normalize by the 3rd element 'Z' to flatten back to 2D
    robot_x = mapped_vector[0] / mapped_vector[2]
    robot_y = mapped_vector[1] / mapped_vector[2]
    
    return robot_x, robot_y


def run_pick_and_place(lite6_ip):
    """
    Loads the matrix, takes a mock object detection pixel, and moves the Lite 6.
    """
    print("--- RUNNING RUNTIME PHASE ---")
    if not os.path.exists(MATRIX_FILENAME):
        print("Error: Matrix not found! Run calibration first.")
        return
    
    # Load the saved Matrix
    H = np.load(MATRIX_FILENAME)
    print("Loaded Homography Matrix.")

    # -------------------------------------------------------------
    # SIMULATED VISION STEP
    # -------------------------------------------------------------
    # Example: YOLO found the object at pixel (640, 360)
    target_u, target_v = 640, 360 
    print(f"Vision System found object at pixels: (u={target_u}, v={target_v})")

    # Map pixels to Robot mm
    target_x, target_y = pixel_to_robot(target_u, target_v, H)
    print(f"Mapped to Robot Coordinates: X={target_x:.2f} mm, Y={target_y:.2f} mm")

    # -------------------------------------------------------------
    # ROBOT EXECUTION STEP
    # -------------------------------------------------------------
    try:
        print(f"Connecting to Lite 6 at {lite6_ip}...")
        arm = XArmAPI(lite6_ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)
        
        pre_grasp_z = 150.0 
        speed = 200 
        accel = 500

        print(f"Moving to Pre-Grasp position above the object...")
        arm.set_position(
            x=target_x, 
            y=target_y, 
            z=pre_grasp_z, 
            roll=-180.0, 
            pitch=0.0, 
            yaw=0.0, 
            speed=speed, 
            mvacc=accel, 
            wait=True
        )
        print("Robot is now positioned above the object!")
        
        arm.disconnect()
        
    except Exception as e:
        print(f"Could not connect or move robot: {e}")
        print("Did you enter the correct Lite 6 IP address?")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # CHANGE THIS TO "CALIBRATE" OR "RUNTIME"
    MODE = "CALIBRATE" 
    
    # CHANGE THIS TO YOUR LITE 6 IP ADDRESS
    LITE6_IP = "192.168.1.150" 

    if MODE == "CALIBRATE":
        calibrate_and_save()
    elif MODE == "RUNTIME":
        run_pick_and_place(LITE6_IP)
    else:
        print("Invalid MODE selected.")