Added TOP_VIEW_POSE to use. Currently for Lite6 code base (and the SO ARM codebase), it uses top-down camera for the top view for snapshot, and verification and stuff. But now with the lite6, there's only 1 camera, which is the wrist camera now. So instead we are going to repurpose and reuse the wrist camera as well for the top-view by letting the lite6 go to this top view pose position for any snapshot or anything involving the topview. I think this system should work. Its almost like a "set and ready" state, like going back to a known position that's fixed to check and do the task etc...


---

Run `scripts/calibrate_vision.py` for ZONE_PIXEL_ROI

Update CAMERA_CENTER_OFFSET, WRIST_GRASP_ROI