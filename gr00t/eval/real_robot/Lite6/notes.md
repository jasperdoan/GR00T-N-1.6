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


Because of how the camera is mounted on the wrist, and the actualy object that's below the gripper is just about at the bottom of the frame instead of the center. So how I am doing it is now instead of the ROI thing at the bottom of the frame / defined in constants; we want to go to and center the object in the middle of the frame, dead center of the frame, so now the ROI would be a square in the middle of the frame instead of at the bottom or as defined in constants. This way, the camera is looking at the object dead center (after the fine tune search), then once it is dead center. We can move x direction -70mm to line up the gripper for pick up. Basically, it will do a global move then fine tune search to center object in middle of frame, after confirm like within ROI box or 3/5 consecutive frames, then move x direction -70mm to line up the gripper for pick up; we could even move the x -70 and lower z to the one defined in constants at the same time so its in 1 motion (Do defined the 70mm in constants). So like for example if it is placed at an angle like 30 degrees. It will first center the object in frame. And its current position is like [-50.0, -150.0, 300.0, -180.0, 0.0, 0.0] (AFTER FINE TUNE SEARCH). Then it should now move to [-120.0, -150.0, GRASP_Z, -180.0, 0.0, 30.0] (GRASP_Z is defined in constants). So it will move x direction -70mm and lower z to GRASP_Z at the same time, not in separate motions. Don't forgot the rotate because it is currently not doing that...

Also for the video, currently the point cloud 3D viz does not look good. Like it looks like it is screwed instead of titled at an angle to show the 3D depth of the scene. Which by the way is what I want to show.
