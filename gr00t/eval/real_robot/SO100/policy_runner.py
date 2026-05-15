"""
SO100 Async Policy Runner (Time-Aligned Queue)
==============================================
Runs VLA inference in a background thread. When a new chunk arrives, it 
calculates exactly how many steps elapsed during inference and fast-forwards 
the chunk so the robot never jumps backward in time.
"""

import threading
import time
from typing import Dict, Optional, List
import numpy as np

class AsyncPolicyRunner:
    def __init__(
        self,
        policy,
        replan_every: int = 6,
        ensemble_temp: float = 0.1,  # Kept in signature so eval_so100.py doesn't break
    ) -> None:
        self.policy = policy
        self.replan_every = replan_every
        
        self._thread = None
        self._lock = threading.Lock()
        
        self._pending_chunk = None
        self._pending_start_step = 0
        self._current_chunk = []
        self._step = 0
        self._steps_since_replan = 0

    def _infer_background(self, obs_snapshot: Dict, start_step: int) -> None:
        try:
            chunk = self.policy.get_action_chunk(obs_snapshot)
        except Exception as exc:
            print(f"[AsyncPolicyRunner] Inference error: {exc}")
            chunk = None
            
        with self._lock:
            self._pending_chunk = chunk
            self._pending_start_step = start_step

    def step(self, obs: Dict) -> Optional[Dict[str, float]]:
        # 1. Promote a newly arrived chunk, aligned to the current timeline
        with self._lock:
            if self._pending_chunk is not None:
                elapsed = self._step - self._pending_start_step
                if elapsed < len(self._pending_chunk):
                    self._current_chunk = self._pending_chunk[elapsed:]
                else:
                    self._current_chunk = [self._pending_chunk[-1]] # Keep last action if we overran
                self._pending_chunk = None

        # 2. Trigger background inference if it's time
        thread_idle = self._thread is None or not self._thread.is_alive()
        if self._steps_since_replan >= self.replan_every and thread_idle:
            # Safely copy observation (handles both numpy arrays and torch tensors)
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

        self._step += 1
        self._steps_since_replan += 1

        # 3. Bootstrap stall (only happens on the very first step)
        if not self._current_chunk and self._thread is not None and self._thread.is_alive():
            print("[AsyncPolicyRunner] Waiting for first inference chunk...")
            self._thread.join()
            with self._lock:
                if self._pending_chunk is not None:
                    elapsed = self._step - self._pending_start_step
                    if elapsed < len(self._pending_chunk):
                        self._current_chunk = self._pending_chunk[elapsed:]
                    else:
                        self._current_chunk = [self._pending_chunk[-1]]
                    self._pending_chunk = None

        # 4. Pop the next action (but repeat the last one if we run completely dry)
        if self._current_chunk:
            if len(self._current_chunk) > 1:
                return self._current_chunk.pop(0)
            else:
                return self._current_chunk[0]
        return None

    def reset(self) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            self._pending_chunk = None
        self._current_chunk = []
        self._step = 0
        self._steps_since_replan = 0