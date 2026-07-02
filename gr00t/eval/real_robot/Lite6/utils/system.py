"""
Lite6 System Utilities
Graceful shutdown, SIGINT/SIGTERM signal handling, and /tmp state flags.
"""

import os
import signal
from utils.constants import (
    INUSE_FLAG_PATH,
    STOP_FLAG_PATH
)

_graceful_stop_requested = False


def request_graceful_stop(sig, frame):
    global _graceful_stop_requested
    if _graceful_stop_requested:
        print("\n[HARD STOP] Forced exit — terminating immediately (no home sequence).")
        os._exit(1)
    print("\n[SOFT STOP] Signal received. Finishing current move then returning home...")
    _graceful_stop_requested = True


def setup_signal_handlers():
    signal.signal(signal.SIGINT,  request_graceful_stop)
    signal.signal(signal.SIGTERM, request_graceful_stop)


def is_stop_requested() -> bool:
    return _graceful_stop_requested or os.path.exists(STOP_FLAG_PATH)


def clear_stop_flag():
    if os.path.exists(STOP_FLAG_PATH):
        try:
            os.remove(STOP_FLAG_PATH)
        except Exception:
            pass


def set_in_use():
    try:
        with open(INUSE_FLAG_PATH, "w") as f:
            pass
    except Exception as e:
        print(f"[System] Failed to set in-use flag: {e}")


def clear_in_use():
    if os.path.exists(INUSE_FLAG_PATH):
        try:
            os.remove(INUSE_FLAG_PATH)
        except Exception as e:
            print(f"[System] Failed to clear in-use flag: {e}")
