"""JSON serialization shared by preparation and experiment reporting."""

import json
from pathlib import Path


def save_json(path, value):
    """Write a JSON-serializable value as indented UTF-8 text to path.
    
    The caller creates the parent directory; this function does not transform values."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
