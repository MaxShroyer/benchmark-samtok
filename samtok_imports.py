import os
import sys
from typing import Optional


def _looks_like_sa2va_repo(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    # Sa2VA contains `projects/samtok/...` which we import from benchmark.py.
    return os.path.isdir(os.path.join(path, "projects", "samtok"))


def _candidate_paths() -> list[str]:
    candidates = []
    for env_var in ("SAMTOK_SA2VA_PATH", "SA2VA_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(value)
    # Common local clone locations (relative to this repo and cwd).
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.join(repo_root, "Sa2VA"),
            os.path.join(repo_root, "..", "Sa2VA"),
            os.path.join(os.getcwd(), "Sa2VA"),
            os.path.join(os.getcwd(), "..", "Sa2VA"),
        ]
    )
    return candidates


def ensure_samtok_imports() -> Optional[str]:
    for path in _candidate_paths():
        if _looks_like_sa2va_repo(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            return path
    return None
