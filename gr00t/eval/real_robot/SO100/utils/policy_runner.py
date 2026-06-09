"""
SO100 Async Policy Runner (Time-Aligned Temporal Ensemble)
==========================================================
State-of-the-art Action Chunking execution. 
1. Runs VLA inference in a background thread to prevent blocking.
2. Fast-forwards new chunks based on the exact step the thread started.
3. Exponentially averages overlapping chunks (Temporal Ensembling) to smooth 
   out VLA noise and create flawless transitions between trajectories.


=======================================================================================================================
THE TIME-ALIGNED TEMPORAL ENSEMBLE TIMELINE (Steps 0 - 24)
=======================================================================================================================

LEGEND:
[*] = Policy Inference Requested (Picture Taken)
[#] = GPU Processing (Inference Delay: 2 steps)
[A][B][C][D] = Chunks being executed
[---] = Actions skipped (Time-Alignment) to match the current global step.

GLOBAL | VLA ACTIVITY    | CHUNK A         | CHUNK B         | CHUNK C         | CHUNK D         | VOTING POWER (%)
STEP   | (Inference)     | (Capture S:0)   | (Capture S:6)   | (Capture S:12)  | (Capture S:18)  | (Weighted Average)
-------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------------
0      | * Request A     | [WAITING...]    |                 |                 |                 | (Bootstrapping)
1      | # Thinking...   | [WAITING...]    |                 |                 |                 | (Bootstrapping)
2      | A Arrives!      | [---][---][A2]  |                 |                 |                 | 100% A
3      |                 | [A3]            |                 |                 |                 | 100% A
4      |                 | [A4]            |                 |                 |                 | 100% A
5      |                 | [A5]            |                 |                 |                 | 100% A
-------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------------
6      | * Request B     | [A6]            | [WAITING...]    |                 |                 | 100% A
7      | # Thinking...   | [A7]            | [WAITING...]    |                 |                 | 100% A
8      | B Arrives!      | [A8]            | [---][---][B2]  |                 |                 | 52% B (New), 48% A
9      |                 | [A9]            | [B3]            |                 |                 | 52% B, 48% A
10     |                 | [A10]           | [B4]            |                 |                 | 52% B, 48% A
11     |                 | [A11]           | [B5]            |                 |                 | 52% B, 48% A
-------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------------
12     | * Request C     | [A12]           | [B6]            | [WAITING...]    |                 | 52% B, 48% A
13     | # Thinking...   | [A13]           | [B7]            | [WAITING...]    |                 | 52% B, 48% A
14     | C Arrives!      | [A14]           | [B8]            | [---][---][C2]  |                 | 37% C, 33% B, 30% A
15     |                 | [A15]           | [B9]            | [C3]            |                 | 37% C, 33% B, 30% A
16     | [A EXPIRES]     |                 | [B10]           | [C4]            |                 | 52% C, 48% B
17     |                 |                 | [B11]           | [C5]            |                 | 52% C, 48% B
-------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------------
18     | * Request D     |                 | [B12]           | [C6]            | [WAITING...]    | 52% C, 48% B
19     | # Thinking...   |                 | [B13]           | [C7]            | [WAITING...]    | 52% C, 48% B
20     | D Arrives!      |                 | [B14]           | [C8]            | [---][---][D2]  | 37% D, 33% C, 30% B
21     |                 |                 | [B15]           | [C9]            | [D3]            | 37% D, 33% C, 30% B
22     | [B EXPIRES]     |                 |                 | [C10]           | [D4]            | 52% D, 48% C
23     |                 |                 |                 | [C11]           | [D5]            | 52% D, 48% C
24     | * Request E     |                 |                 | [C12]           | [D6]            | 52% D, 48% C
-------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------------

=======================================================================================================================
VOTING POWER BREAKDOWN (The "Why it works")
=======================================================================================================================

1. THE MERGE (Step 8): 
   Chunk B arrives. It is "Age 0" (Weight: 1.0). Chunk A is "Age 1" (Weight: 0.90).
   Total Weight: 1.90.
   - Chunk B (Newest) gets 1.0 / 1.90 = 52.6% power.
   - Chunk A (Oldest) gets 0.9 / 1.90 = 47.4% power.
   The robot is now 52% controlled by Chunk B, but A is still "smoothing" the transition.

2. THE 3-WAY BLEND (Step 14):
   Chunk C arrives. 
   - Chunk C (Age 0): 1.00 weight -> 36.7% Power
   - Chunk B (Age 1): 0.90 weight -> 33.2% Power
   - Chunk A (Age 2): 0.82 weight -> 30.1% Power
   The robot is now averaging three different opinions! If Chunk C is slightly "noisy," the combined
   momentum of A and B (63% total) stabilizes the arm while C slowly pulls it toward the new goal.

3. THE SKIPPING (Time-Alignment):
   Notice Step 14. Chunk C was requested at Step 12. 
   When C arrives, it contains actions [C0, C1, C2...].
   If we played [C0] at Step 14, we would be playing an action meant for Step 12.
   The code skips C0 and C1, and starts blending at [C2]. 
   Because [A14], [B8], and [C2] all represent the EXACT same point in the future (Step 14),
   the robot stays perfectly on the timeline.

4. THE TEMPERATURE EFFECT:
   - If Temperature = 0.0: Every active chunk gets exactly equal power (33/33/33).
   - If Temperature = 0.1: Newest chunk is slightly stronger (37/33/30). [CURRENT SETTING]
   - If Temperature = 0.5: Newest chunk dominates heavily (58/35/7). Use this if you want maximum reactivity.

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