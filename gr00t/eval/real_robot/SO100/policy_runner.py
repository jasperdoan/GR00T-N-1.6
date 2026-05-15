"""
SO100 Async Policy Runner with Temporal Ensembling
====================================================

Eliminates the burst-pause-burst pattern of naive action-chunk execution by
combining two complementary techniques:

  1. Async double-buffer  — GR00T inference runs in a background thread while
     the main control loop keeps sending actions at 30 Hz.  The moment a new
     chunk is ready it is swapped in; the robot never stalls waiting for the GPU.

  2. Temporal ensembling  — Overlapping action chunks are blended with
     exponential weights (newest chunk wins hardest).  This smooths the seam
     between consecutive chunks because adjacent predictions agree on the
     shared timesteps, averaging out any small disagreement.

Public API
----------
  runner = AsyncPolicyRunner(policy, replan_every=6, ensemble_temp=0.1)
  action = runner.step(obs)   # call at 30 Hz; returns a joint dict or None
  runner.reset()              # call between episodes

Typical call pattern in the VLA loop
-------------------------------------
  runner = AsyncPolicyRunner(policy)
  while True:
      obs = robot.get_observation()
      obs["lang"] = color
      if grasp_detector.update(obs, color):
          break
      action = runner.step(obs)
      if action is not None:
          robot.send_action(action)
      time.sleep(1.0 / 30)
"""

import threading
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Temporal Ensemble
# =============================================================================

class TemporalEnsemble:
    """
    Sliding-window blend of overlapping action chunks.

    Each time a new action chunk arrives (via add_chunk), it is stored with the
    step index at which it arrived.  Every call to get_action returns an
    exponentially-weighted average of all chunks that cover the current step:

        w_i = exp(−temperature × age_i)

    where age_i = 0 for the most recently added chunk and increases for older
    ones.  This gives the latest inference the most authority while still
    smoothing away any step-to-step discontinuity between consecutive chunks.

    Args:
        temperature: Controls how sharply older chunks are down-weighted.
                     0.0  → uniform average of all chunks (maximum smoothing).
                     0.1  → moderate bias toward the newest (recommended).
                     1.0+ → nearly always takes the newest chunk (less blending).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature
        # Stored as (arrival_step, chunk_list) pairs, oldest first
        self._chunks: Deque[Tuple[int, List[Dict[str, float]]]] = deque()
        self._step   = 0

    # -------------------------------------------------------------------------

    def add_chunk(self, chunk: List[Dict[str, float]]) -> None:
        """Register a newly arrived action chunk at the current step index."""
        self._chunks.append((self._step, chunk))

    def get_action(self) -> Optional[Dict[str, float]]:
        """
        Return the blended action for the current step, then advance the
        internal clock by one tick.

        Chunks that no longer cover the current step are pruned automatically.
        Returns None if no chunk covers this step (should only happen during
        the very first inference latency window).
        """
        # Prune expired chunks (those that ended before this step)
        while self._chunks and (self._chunks[0][0] + len(self._chunks[0][1]) <= self._step):
            self._chunks.popleft()

        if not self._chunks:
            self._step += 1
            return None

        candidates: List[Dict[str, float]] = []
        weights:    List[float]            = []

        # Iterate newest → oldest to assign age 0 to the most recent chunk
        for age, (start_step, chunk) in enumerate(reversed(self._chunks)):
            idx = self._step - start_step
            if 0 <= idx < len(chunk):
                candidates.append(chunk[idx])
                weights.append(np.exp(-self.temperature * age))

        if not candidates:
            self._step += 1
            return None

        total_w  = sum(weights)
        norm_w   = [w / total_w for w in weights]
        joints   = list(candidates[0].keys())

        blended  = {
            j: float(sum(w * c[j] for w, c in zip(norm_w, candidates)))
            for j in joints
        }

        self._step += 1
        return blended

    def reset(self) -> None:
        """Clear all state — call between episodes."""
        self._chunks.clear()
        self._step = 0


# =============================================================================
# Async Double-Buffer Runner
# =============================================================================

class AsyncPolicyRunner:
    """
    Wraps a So100Adapter and drives inference asynchronously so the 30 Hz
    control loop never blocks on the GPU.

    How it works
    ~~~~~~~~~~~~
    *   The control loop calls step(obs) at 30 Hz.
    *   step() pops actions from the TemporalEnsemble.
    *   After every *replan_every* steps, a background thread is launched to
        run get_action_chunk() on a snapshot of the current observation.
    *   When the thread finishes, its chunk is fed into the ensemble on the
        next step() call; the hand-off is invisible because the ensemble blends
        the seam between the outgoing and incoming chunks.
    *   On the very first call, step() blocks briefly until at least one chunk
        is available — this is the only unavoidable stall, and it happens before
        the robot has started moving.

    Args:
        policy:         So100Adapter instance.  get_action_chunk must be safe to
                        call from a daemon thread (no shared mutable state).
        replan_every:   Steps between re-inference triggers.  Smaller values
                        give more reactive, smoother behaviour at the cost of
                        higher GPU load.  4–8 is a good range for a 16-step
                        horizon at 30 Hz.
        ensemble_temp:  Passed directly to TemporalEnsemble.  0.1 works well.
    """

    def __init__(
        self,
        policy,
        replan_every:   int   = 6,
        ensemble_temp:  float = 0.1,
    ) -> None:
        self.policy        = policy
        self.replan_every  = replan_every
        self.ensemble      = TemporalEnsemble(temperature=ensemble_temp)

        self._thread:         Optional[threading.Thread]       = None
        self._pending_chunk:  Optional[List[Dict[str, float]]] = None
        self._lock            = threading.Lock()
        self._steps_since_replan = 0

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _infer_background(self, obs_snapshot: Dict) -> None:
        """Target for the background inference thread."""
        try:
            chunk = self.policy.get_action_chunk(obs_snapshot)
        except Exception as exc:
            print(f"[AsyncPolicyRunner] Inference error (chunk dropped): {exc}")
            chunk = None
        with self._lock:
            self._pending_chunk = chunk

    def _flush_pending(self) -> None:
        """If a background chunk has arrived since last tick, add it to the ensemble."""
        with self._lock:
            if self._pending_chunk is not None:
                self.ensemble.add_chunk(self._pending_chunk)
                self._pending_chunk = None

    def _maybe_trigger_infer(self, obs: Dict) -> None:
        """Launch a new background inference thread when it is time to replan."""
        thread_idle    = self._thread is None or not self._thread.is_alive()
        time_to_replan = self._steps_since_replan >= self.replan_every

        if time_to_replan and thread_idle:
            # Shallow copy: deep-copy arrays so inference doesn't race with the
            # control loop, but avoid copying large camera frames unnecessarily.
            obs_snap = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
            }
            self._thread = threading.Thread(
                target=self._infer_background,
                args=(obs_snap,),
                daemon=True,
                name="gr00t-infer",
            )
            self._thread.start()
            self._steps_since_replan = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def step(self, obs: Dict) -> Optional[Dict[str, float]]:
        """
        Advance by one control tick (call at 30 Hz).

        Returns the blended action dict for this timestep, or None during the
        very first inference latency window (before any chunk is available).
        In the None case, the caller should skip send_action for that tick.

        Args:
            obs: Current robot observation dict (including "lang" key for GR00T).
        """
        # 1. Collect any chunk that finished in the background
        self._flush_pending()

        # 2. Decide whether to kick off a new inference
        self._maybe_trigger_infer(obs)

        # 3. Pop the blended action for this timestep
        action = self.ensemble.get_action()
        self._steps_since_replan += 1

        # 4. Bootstrap block: only on the very first call, wait for initial chunk
        if action is None and self._thread is not None:
            print("[AsyncPolicyRunner] Waiting for first inference chunk …")
            self._thread.join()
            self._flush_pending()
            action = self.ensemble.get_action()

        return action

    def reset(self) -> None:
        """
        Clear all internal state.  Call between evaluation episodes so stale
        chunk history from the previous run doesn't bleed into the next one.
        """
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.ensemble.reset()
        with self._lock:
            self._pending_chunk = None
        self._steps_since_replan = 0