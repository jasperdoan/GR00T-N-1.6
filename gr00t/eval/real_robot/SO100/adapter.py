"""
SO100 Policy Adapter

Translates between LeRobot observation dicts and the GR00T policy server's
expected input/output format.
"""

from typing import Any, Dict, List

import numpy as np

from constants import JOINT_NAMES


def _recursive_add_extra_dim(obs: Dict) -> Dict:
    """Wrap every array/scalar in an extra batch dimension (required by GR00T)."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = _recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class So100Adapter:
    """
    Bridges the LeRobot observation format and the GR00T PolicyClient API.

    VLA receives:
      - video:    {"front": <H,W,3>, "wrist": <H,W,3>}
      - state:    {"single_arm": <5,>, "gripper": <1,>}
      - language: {"annotation.human.task_description": <str>}
                  NOTE: language is now the *color only* (e.g. "red"),
                  not the full instruction string. The full instruction is
                  handled by the orchestration layer for task-type routing.

    Policy outputs an action chunk which is decoded joint-by-joint.
    """

    CAMERA_KEYS    = ["front", "wrist"]
    LANGUAGE_KEY   = "annotation.human.task_description"

    def __init__(self, policy_client) -> None:
        self.policy = policy_client

    # -------------------------------------------------------------------------
    # Observation → Policy Input
    # -------------------------------------------------------------------------

    def obs_to_policy_inputs(self, obs: Dict[str, Any]) -> Dict:
        state = np.array(
            [obs[j] for j in JOINT_NAMES],
            dtype=np.float32,
        )

        model_obs = {
            "video": {k: obs[k] for k in self.CAMERA_KEYS},
            "state": {
                "single_arm": state[:5],
                "gripper":    state[5:6],
            },
            "language": {
                self.LANGUAGE_KEY: obs["lang"],  # canonical object string ("dice", "pink prism")
            },
        }

        # GR00T requires two batch dimensions
        model_obs = _recursive_add_extra_dim(model_obs)
        model_obs = _recursive_add_extra_dim(model_obs)
        return model_obs

    # -------------------------------------------------------------------------
    # Policy Output → Action Dict
    # -------------------------------------------------------------------------

    def _decode_action_chunk(self, chunk: Dict, t: int) -> Dict[str, float]:
        single_arm = chunk["single_arm"][0][t]
        gripper    = chunk["gripper"][0][t]
        full       = np.concatenate([single_arm, gripper], axis=0)
        return {joint: float(full[i]) for i, joint in enumerate(JOINT_NAMES)}

    def get_action_chunk(self, obs: Dict) -> List[Dict[str, float]]:
        """
        Run one forward pass through the VLA and return the full action chunk
        as a list of per-timestep joint dicts.
        """
        model_input           = self.obs_to_policy_inputs(obs)
        action_chunk, _info   = self.policy.get_action(model_input)
        horizon               = action_chunk[next(iter(action_chunk))].shape[1]
        return [self._decode_action_chunk(action_chunk, t) for t in range(horizon)]