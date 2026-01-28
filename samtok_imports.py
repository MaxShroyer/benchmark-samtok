import os
import sys
from typing import Optional


def _candidate_paths() -> list[str]:
    candidates = []
    for env_var in ("SAMTOK_SA2VA_PATH", "SA2VA_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(value)
    candidates.append(os.path.join(os.getcwd(), "Sa2VA"))
    return candidates


def ensure_samtok_imports() -> Optional[str]:
    for path in _candidate_paths():
        if os.path.isdir(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    return None
