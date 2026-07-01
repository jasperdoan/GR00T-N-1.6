
Will be using Orbbec Gemini336 3D Camera. Maybe leverage pyorbbecsdk for all the 3D stuff like Depth Data estimation, depth pixels to the color pixels (e.g., finding the depth of a specific colored object). The lenses are physically offset from one another. The SDK performs the complex matrix math to warp and align the depth map to the color map, IR projector (the laser pattern that helps it see in the dark), adjust the depth sensing mode, or read the built-in IMU (gyroscope/accelerometer) data, etc... to this arm. Ask question if you are unsure. See where any of these could be used / applicable to our current task and application so that it is more accurate. Like one of them is Point Clouds: The SDK has built-in functions to instantly convert the depth frames into 3D X, Y, Z coordinates (Point Clouds). Which I think is super useful considering the fact we can merge this with the lite6 mm 3D set position system to know exactly where everything is!!!! Honestly I prefer it this way over what we have currently where it scans everything and knows wher everything is and where to go to grab the object for example.



http://0.0.0.0:9988/stream.mjpg
