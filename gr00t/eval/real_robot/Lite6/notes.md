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
        What happens when 2 cubes of similar or idential are placed close / next / touching each other. From what I noticed it is grouped up into 1 whole single cube, anyway around this?

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