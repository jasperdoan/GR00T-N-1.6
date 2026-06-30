import cv2
import time

def instant_capture(filename="captured_image.jpg"):
    # Initialize the webcam (0 is usually the default built-in camera)
    cap = cv2.VideoCapture(11)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Optional: Give the camera a second to warm up and adjust exposure/autofocus
    time.sleep(1)

    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was captured successfully, save it
    if ret:
        cv2.imwrite(filename, frame)
        print(f"Success! Image saved as {filename}")
    else:
        print("Error: Failed to capture an image.")

    # Release the camera resource
    cap.release()

if __name__ == "__main__":
    instant_capture()




# import cv2

# def test_cameras():
#     print("Searching for available camera indices...")
#     available_cameras = []
    
#     # Check the first 10 indices
#     for i in range(10):
#         # We use cv2.CAP_V4L2 to explicitly tell OpenCV to use Linux video drivers
#         # This prevents the annoying GStreamer warnings you were seeing
#         cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 print(f"[SUCCESS] Camera index {i} is working and returning frames.")
#                 available_cameras.append(i)
#             else:
#                 print(f"[WARNING] Camera index {i} opened, but could not read frames (might be depth/metadata).")
#             cap.release()
            
#     print(f"\nUsable camera indices: {available_cameras}")

# if __name__ == "__main__":
#     test_cameras()