"""Async hand-safety monitor: MediaPipe Hands in a separate OS process."""

import ctypes
import multiprocessing as mp_lib

import cv2
import numpy as np

try:
    import mediapipe as mp  # noqa: F401  (presence check; real import in child)
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class SafetyMonitor:
    """
    Runs MediaPipe Hands in a separate OS process to bypass the GIL.
    Main loop reads hand_detected in O(1) via shared memory.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_MEDIAPIPE

        self.hand_detected = mp_lib.Value(ctypes.c_bool, False)
        self.frame_queue   = mp_lib.Queue(maxsize=1)
        self._process      = None

        if not self.enabled:
            if enabled:
                print("[WARNING] MediaPipe not found — hand safety pause is DISABLED.")
        else:
            self._process = mp_lib.Process(
                target=self._process_loop,
                args=(self.frame_queue, self.hand_detected),
                daemon=True,
            )
            self._process.start()

    def update_frame(self, img_rgb: np.ndarray):
        """Pass a downscaled frame to the background process (lazy, non-blocking)."""
        if not self.enabled:
            return
        if self.frame_queue.empty():
            small = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            try:
                self.frame_queue.put_nowait(small)
            except Exception:
                pass

    def is_hand_present(self) -> bool:
        return bool(self.hand_detected.value)

    @staticmethod
    def _process_loop(queue, shared_flag):
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        while True:
            try:
                frame = queue.get()
                if frame is None:
                    break
                img_u8 = frame if frame.dtype == np.uint8 else (frame * 255).clip(0, 255).astype(np.uint8)
                results = hands.process(img_u8)
                shared_flag.value = results.multi_hand_landmarks is not None
            except Exception:
                pass

    def stop(self):
        if self.enabled:
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except Exception:
                pass
            try:
                self.frame_queue.put_nowait(None)
            except Exception:
                pass
        if hasattr(self, "frame_queue") and hasattr(self.frame_queue, "cancel_join_thread"):
            self.frame_queue.cancel_join_thread()
        if self._process and self._process.is_alive():
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
