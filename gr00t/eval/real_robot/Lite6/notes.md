For the future:

Might need to use Yolo Model for detection instead of whatever it is right now

Expert-techniques menu (user asked for proposals)

    Adopted this round: PBVS look-then-move; timestamp-gated frame freshness; two-stage coarse-to-fine refine; EMA state estimation with innovation gating (45 mm world gate = the gate, EMA = the smoother); calibration health monitoring; shape validation gates; (previously: world-frame target identity, depth height-gating, suction feedback, blended transport).



Roadmap (documented, not implemented now):

    Flat-spot suction targeting — pick the grasp point as the centroid of the flattest/smoothest region of the top-face depth patch (maximizes seal quality) instead of the color centroid; classic vacuum-picking refinement.

    Kalman filter upgrade — constant-position KF with Mahalanobis gating replaces EMA if measurement noise ever warrants it (per-axis variance tracking comes free).

    AprilTag/ChArUco auto-recalibration — a tag glued to the table lets a script re-solve extrinsics in seconds whenever the calibration-health residual drifts; removes the manual Kabsch sampling session.

    Learned detection (YOLO/FastSAM segmentation) — replaces HSV thresholds outright; solves lighting drift and enables arbitrary object classes (already on notes.md's future list; pairs with locking camera exposure/WB).

    Guarded descend — monitor joint currents (get_joints_torque) during the final blind descend and stop on contact; protects against height-estimate errors without an F/T sensor.

    Grasp-pose ranking — when multiple cubes are candidates, pick the one with the best margin (distance from zone edges/neighbors) instead of random; reduces edge-case failures in crowded zones.

    Use depth for pick up instead of fixed height defined in constants. Generally the height defined in constants are good with most items, but some objects/cubes are slightly shorter than the other, hence it might miss the object by some mm. Maybe we can do depth estimate, and grab the larger height for grasping, not sure...Plan this out, what do you think?

    For the drop destination / target space, ideally we want to drop where there's empty space, so that the cubes are spread out and doesn't collide with each other. Other wise if there's no empty space / meaning the whole space is occupied. We must return the cube to where it was picked up. For example if the cube was picked up at check in at X1 Y2 Z3, going to storage, we see that there's other cubes in storage that's taking up spaces (could be red, blue, or any color, we need to be able to detect prescene), but there's an empty space top right, then it shall drop off the picked up object there. If there's no empty spaces, meaning in this case top right corner is also occupied, so are other spaces. It will return the cube back to X1 Y2 Z3 and end task.

---


python3 gr00t/eval/real_robot/Lite6/auto.py --camera 0 --video


---


Files/folders of interest to update:
@GR00T-N-1.6/gr00t/eval/real_robot/Lite6/
@lerobot_app/
@lerobot_app/demo/
@lerobot_app/lite6/
@lerobot_app/run-console.sh
@lerobot_app-setup/
@lerobot_app-setup/update-lerobot-app.sh/


Requests:

Add more color options (like orange, pink, purple, etc...) | Side notes, what about shapes, does it currently check for solely cubes, or would a hexagon work also as long if its the same color?

Use depth for pick up instead of fixed height defined in constants. Generally the height defined in constants are good with most items, but some objects/cubes are slightly shorter than the other, hence it might miss the object by some mm. Maybe we can do depth estimate, and grab the larger height for grasping, not sure...Plan this out, what do you think?

For the drop destination / target space, ideally we want to drop where there's empty space, so that the cubes are spread out and doesn't collide with each other. Other wise if there's no empty space / meaning the whole space is occupied. We must return the cube to where it was picked up. For example if the cube was picked up at check in at X1 Y2 Z3, going to storage, we see that there's other cubes in storage that's taking up spaces (could be red, blue, or any color, we need to be able to detect prescene), but there's an empty space top right, then it shall drop off the picked up object there. If there's no empty spaces, meaning in this case top right corner is also occupied, so are other spaces. It will return the cube back to X1 Y2 Z3 and end task.