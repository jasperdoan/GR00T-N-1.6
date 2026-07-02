"""
Lite6 NLP Parser
Extracts task_type and target_object from a natural language instruction string.
"""

import re
from utils.constants import ZONES, KNOWN_OBJECTS

# Maps spoken task phrases to internal task_type keys
_TASK_ALIASES = {
    "check_in":   ["check in", "check-in", "checkin", "store", "put away"],
    "check_out":  ["check out", "check-out", "checkout", "dispatch", "send out"],
    "check_back": ["check back", "check-back", "checkback", "return", "bring back"],
}

# Source and target zones for each task type
TASK_ZONE_MAP = {
    "check_in":   {"source": "check_in",  "target": "storage"},
    "check_out":  {"source": "storage",   "target": "check_out"},
    "check_back": {"source": "check_out", "target": "check_in"},
}


def parse_instruction(instruction: str):
    """
    Parse a natural language instruction into (task_type, target_object, source_zone, target_zone).
    Raises ValueError if the instruction cannot be parsed.
    """
    text = instruction.lower().strip()

    # Identify task type
    task_type = None
    for key, aliases in _TASK_ALIASES.items():
        for alias in aliases:
            if alias in text:
                task_type = key
                break
        if task_type:
            break

    if task_type is None:
        raise ValueError(f"[NLP] Could not identify task type in: '{instruction}'")

    # Identify target object
    target_object = None
    for obj in KNOWN_OBJECTS:
        if obj.lower() in text:
            target_object = obj
            break

    if target_object is None:
        # Fallback: extract color + "cube" heuristically
        color_match = re.search(r"\b(red|blue|yellow|green|orange|pink|purple|black|white)\b", text)
        if color_match:
            target_object = color_match.group(1) + " cube"
        else:
            raise ValueError(f"[NLP] Could not identify target object in: '{instruction}'")

    zones = TASK_ZONE_MAP[task_type]
    source_zone = zones["source"]
    target_zone = zones["target"]

    return task_type, target_object, source_zone, target_zone
