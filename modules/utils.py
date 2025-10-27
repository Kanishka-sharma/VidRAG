"""
Utility functions for JSON I/O and path handling.
"""
from pathlib import Path
import json

def save_json(obj, path: Path):
    """
    Save a Python object to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    """
    Load a Python object from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
