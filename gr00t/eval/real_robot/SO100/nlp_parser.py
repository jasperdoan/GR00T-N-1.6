"""
SO100 NLP Parser
Extracts task type, target object, source zone, and target zone from free-text instructions.
"""

import re
from typing import Tuple

from constants import KNOWN_OBJECTS, CHECK_IN_ZONE, CHECK_OUT_ZONE, STORAGE_ZONE

TASK_SYNONYMS = {
    "check_out": ["out", "retrieve", "get", "fetch", "checkout"],
    "check_in": ["in", "store", "put", "place", "checkin", "return"]
}

def parse_instruction(instruction: str) -> Tuple[str, str, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Parse the free-text instruction into (task_type, target_object, source_zone, target_zone).
    """
    lower = instruction.lower()

    # Determine task_type and zones
    task_type = "check_in" # Default
    source_zone = CHECK_IN_ZONE
    target_zone = STORAGE_ZONE
    
    for t_type, synonyms in TASK_SYNONYMS.items():
        if any(re.search(rf"\b{syn}\b", lower) for syn in synonyms):
            task_type = t_type
            if task_type == "check_out":
                source_zone = STORAGE_ZONE
                target_zone = CHECK_OUT_ZONE
            break

    # Determine target_object
    target_object = None
    for obj in KNOWN_OBJECTS:
        obj_pattern = r"\b" + re.escape(obj.lower()) + r"\b"
        if re.search(obj_pattern, lower):
            target_object = obj
            break
            
    # Fallback to color/shape parsing
    if target_object is None:
        colors = ["red", "blue", "yellow", "orange", "pink", "green", "purple", "black", "white", "gray"]
        shapes = ["cube", "prism", "ball"]
        
        found_color = next((c for c in colors if re.search(rf"\b{c}\b", lower)), None)
        found_shape = next((s for s in shapes if re.search(rf"\b{s}\b", lower)), None)
        
        if found_color and found_shape:
            target_object = f"{found_color} {found_shape}"
            if target_object not in KNOWN_OBJECTS:
                print(f"[PARSER] Warning: '{target_object}' constructed but not in KNOWN_OBJECTS.")
    
    if target_object is None:
        raise ValueError(
            f"Could not find a known object in the instruction: '{instruction}'. "
            f"Known objects are: {KNOWN_OBJECTS}"
        )

    return task_type, target_object, source_zone, target_zone