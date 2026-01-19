"""
Event Serializer
================

Utilities for cleaning SSE event data for JSON serialization.

Handles:
- Removing numpy arrays (embeddings)
- Converting non-JSON-serializable types
- Cleaning nested structures
"""

import numpy as np
from typing import Any, Dict, Set


# Keys to skip entirely (embedding-related fields)
SKIP_KEYS: Set[str] = {
    "_embedding",
    "embedding",
    "embeddings",
    "query_embeddings",
    "query_embedding",
    "query_embeddings_cached"  # Don't send cached embeddings to frontend
}


def clean_value(value: Any) -> Any:
    """
    Recursively clean a value for JSON serialization.

    Args:
        value: Any value to clean

    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    elif isinstance(value, np.ndarray):
        return None  # Remove numpy arrays entirely
    elif isinstance(value, dict):
        cleaned_dict = {}
        for k, val in value.items():
            if k in SKIP_KEYS:
                continue  # Skip embedding keys
            cleaned_val = clean_value(val)
            if cleaned_val is not None or not isinstance(val, np.ndarray):
                # Keep None values only if original wasn't a numpy array
                if not isinstance(val, np.ndarray):
                    cleaned_dict[k] = cleaned_val
        return cleaned_dict
    elif isinstance(value, list):
        return [clean_value(item) for item in value if not isinstance(item, np.ndarray)]
    elif isinstance(value, (int, float, str, bool)):
        return value
    else:
        # For other types, try to convert or skip
        try:
            # Check if it's numpy scalar
            if hasattr(value, 'item'):
                return value.item()
            return value
        except Exception:
            return str(value)  # Fallback to string


def clean_event_for_json(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean event data for JSON serialization by removing non-serializable objects.

    Removes:
    - Embeddings (numpy arrays, _embedding fields)
    - Other non-JSON-serializable data

    Note: Segment objects are already converted to dicts in the pipeline.

    Args:
        event: Event dict with potential non-serializable data

    Returns:
        Cleaned event dict safe for JSON serialization
    """
    cleaned = {}
    for key, value in event.items():
        if key == "data":
            cleaned["data"] = clean_value(value)
        else:
            cleaned[key] = value

    return cleaned
