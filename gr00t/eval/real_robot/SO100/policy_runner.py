"""
SO100 Async Policy Runner (Time-Aligned Temporal Ensemble)
==========================================================
State-of-the-art Action Chunking execution. 
1. Runs VLA inference in a background thread to prevent blocking.
2. Fast-forwards new chunks based on the exact step the thread started.
3. Exponentially averages overlapping chunks (Temporal Ensembling) to smooth 
   out VLA noise and create flawless transitions between trajectories.
"""

import threading
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import time

class TemporalEnsemble:
    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature
        # Stores tuples of (start_step, chunk_list)
        self._chunks: Deque[Tuple[int, List[Dict[str, float]]]] = deque()

    def add_chunk(self, start_step: int, chunk: List[Dict[str, float]]) -> None:
        """Register a chunk bound to the exact global timestep it was requested."""
        self._chunks.append((start_step, chunk))

    def get_action(self, current_step: int) -> Optional[Dict[str, float]]:
        # 1. Prune expired chunks (if their end step is in the past)
        while self._chunks and (self._chunks[0][0] + len(self._chunks[0][1]) <= current_step):
            self._chunks.popleft()

        if not self._chunks:
            return None

        candidates: List[Dict[str, float]] = []
        weights: List[float] = []

        # 2. Iterate from newest chunk (age 0) to oldest
        for age, (start_step, chunk) in enumerate(reversed(self._chunks)):
            # Calculate exactly where we are inside THIS chunk's timeline
            idx = current_step - start_step
            
            # If the chunk is valid for the current moment in time, use it
            if 0 <= idx < len(chunk):
                candidates.append(chunk[idx])
                weights.append(np.exp(-self.temperature * age))

        if not candidates:
            return None

        # 3. Blend the overlapping actions
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]
        joints = list(candidates[0].keys())

        blended = {
            j: float(sum(w * c[j] for w, c in zip(norm_w, candidates)))
            for j in joints
        }
        return blended

    def reset(self) -> None:
        self._chunks.clear()

class AsyncPolicyRunner:
    def __init__(
        self,
        policy,
        replan_every: int = 6,
        ensemble_temp: float = 0.1,
    ) -> None:
        self.policy = policy
        self.replan_every = replan_every
        self.ensemble = TemporalEnsemble(temperature=ensemble_temp)

        self._thread: Optional[threading.Thread] = None
        self._pending_chunk: Optional[Tuple[int, List[Dict[str, float]]]] = None
        self._lock = threading.Lock()
        
        self._step = 0
        self._steps_since_replan = 0

    def _infer_background(self, obs_snapshot: Dict, start_step: int) -> None:
        try:
            chunk = self.policy.get_action_chunk(obs_snapshot)
        except Exception as exc:
            print(f"[AsyncPolicyRunner] Inference error: {exc}")
            chunk = None
        with self._lock:
            if chunk is not None:
                self._pending_chunk = (start_step, chunk)

    def step(self, obs: Dict) -> Optional[Dict[str, float]]:
        # 1. Catch completed background chunks and add them to the ensemble
        with self._lock:
            if self._pending_chunk is not None:
                start_step, chunk = self._pending_chunk
                self.ensemble.add_chunk(start_step, chunk)
                self._pending_chunk = None

        # 2. Check if we need to launch a new inference thread
        thread_idle = self._thread is None or not self._thread.is_alive()
        if self._steps_since_replan >= self.replan_every and thread_idle:
            # Safely copy to prevent PyTorch tensor errors
            obs_snap = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    obs_snap[k] = v.copy()
                elif hasattr(v, "clone"):
                    obs_snap[k] = v.clone()
                else:
                    obs_snap[k] = v

            self._thread = threading.Thread(
                target=self._infer_background,
                args=(obs_snap, self._step),
                daemon=True,
            )
            self._thread.start()
            self._steps_since_replan = 0

        # 3. Bootstrap block (only on the very first frame)
        action = self.ensemble.get_action(self._step)
        if action is None and self._thread is not None and self._thread.is_alive():
            print("[AsyncPolicyRunner] Waiting for first inference chunk...")
            self._thread.join()
            with self._lock:
                if self._pending_chunk is not None:
                    start_step, chunk = self._pending_chunk
                    self.ensemble.add_chunk(start_step, chunk)
                    self._pending_chunk = None
            action = self.ensemble.get_action(self._step)

        # 4. Advance clock
        self._step += 1
        self._steps_since_replan += 1

        return action

    def reset(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.ensemble.reset()
        with self._lock:
            self._pending_chunk = None
        self._step = 0
        self._steps_since_replan = 0