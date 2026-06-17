"""
SO100 System Utilities
Handles OS signals, graceful shutdowns, and state flags (e.g. in-use/stop flags).
"""

import os
import signal
from utils.constants import INUSE_FLAG_PATH, STOP_FLAG_PATH

graceful_stop_requested = False

def request_graceful_stop(sig, frame):
    global graceful_stop_requested
    if graceful_stop_requested:
        print("\n🚨 [HARD STOP] Forced exit requested! Terminating immediately (No Home Sequence).")
        os._exit(1)
    print("\n⏳ [SOFT STOP] Stop signal received. Finishing current movement then returning home...")
    graceful_stop_requested = True

def setup_signal_handlers():
    """Binds Ctrl+C (SIGINT) and termination signals to the graceful shutdown logic."""
    signal.signal(signal.SIGINT, request_graceful_stop)
    signal.signal(signal.SIGTERM, request_graceful_stop)

def is_stop_requested():
    """Checks if a software stop was triggered via signal or the stop flag file."""
    return graceful_stop_requested or os.path.exists(STOP_FLAG_PATH)

def clear_stop_flag():
    """Removes the stop flag if it exists so old stop requests don't trigger immediately."""
    if os.path.exists(STOP_FLAG_PATH):
        try:
            os.remove(STOP_FLAG_PATH)
        except Exception:
            pass

def set_in_use():
    """Creates a flag file to indicate the robot is currently active/in-use."""
    try:
        with open(INUSE_FLAG_PATH, 'w') as f:
            # f.write("1")
            pass
    except Exception as e:
        print(f"Failed to set in-use flag: {e}")

def clear_in_use():
    """Removes the in-use flag file when the robot has cleanly stopped."""
    if os.path.exists(INUSE_FLAG_PATH):
        try:
            os.remove(INUSE_FLAG_PATH)
        except Exception as e:
            print(f"Failed to clear in-use flag: {e}")