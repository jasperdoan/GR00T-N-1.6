For the future:

Will be using Orbbec Gemini336 3D Camera. Maybe leverage pyorbbecsdk for all the 3D stuff like Depth Data estimation, depth pixels to the color pixels (e.g., finding the depth of a specific colored object). The lenses are physically offset from one another. The SDK performs the complex matrix math to warp and align the depth map to the color map, IR projector (the laser pattern that helps it see in the dark), adjust the depth sensing mode, or read the built-in IMU (gyroscope/accelerometer) data, etc... to this arm. Ask question if you are unsure. See where any of these could be used / applicable to our current task and application so that it is more accurate. Like one of them is Point Clouds: The SDK has built-in functions to instantly convert the depth frames into 3D X, Y, Z coordinates (Point Clouds). Which I think is super useful considering the fact we can merge this with the lite6 mm 3D set position system to know exactly where everything is!!!! Honestly I prefer it this way over what we have currently where it scans everything and knows wher everything is and where to go to grab the object for example.



Will need to change later, if string like "http://172.21.2.83:9988/stream.mjpg" then use what we have. Or given index like 0, 1, 2, etc... then use the pyorbbecsdk to get the depth data and color data from the camera. We need to be able to use both, depending on what was given. Maybe some form of toggle like local/stream. If local, then use the pyorbbecsdk to get the depth data and color data from the camera since the camera is physically connected to the computer. If stream, then use what we have currently where it streams the data from the camera over the network. This will be useful for testing and debugging since we can use a local camera or a remote camera depending on what is needed. Note that for streaming, we are only streaming the color data, not the depth data.



Clean up vision.py, file too long. Maybe we can split it, or organize it a bit better into classes, or composite. I just want to declutter it as some functions are way too long and deserves to be split or, make helper functions that calls so it is readable and clear to programmers later on.


Perception problem where it goes for not entirely center since it detects it at an angle so it sees the side, thus end up going for one of the corners. Need a way where fine tune movement goes for dead center of the object. (might need to use depth camera)




Might need to use Yolo Model for detection instead of whatever it is right now


---


python3 gr00t/eval/real_robot/Lite6/auto.py --camera 0 --video


---

— Targeted validations (one run each)
Rotated cube (~30°): should servo, yaw-align, grasp cleanly. If the gripper rotates the wrong way, flip CAMERA_YAW_SIGN to -1.0 in constants.py — that's the last unverified sign.

Tunables if needed: DEPTH_TOP_BAND_MM (20 mm) — if your cube is very short or the depth is noisy at the hover height, widen/narrow this band; the self-test snapshot helps judge depth noise.


---