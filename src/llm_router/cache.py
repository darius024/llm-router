"""On-disk cache keyed by a sha256 of the normalised prompt.

A cache hit skips triage, drafting, and any model call entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_DIR = Path(".cache")
_WHITESPACE = re.compile(r"\s+")


def _normalise(prompt: str) -> str:
    # Strip surrounding whitespace and collapse runs so trivially different
    # prompts (extra spaces, trailing newline) share a cache entry.
    return _WHITESPACE.sub(" ", prompt.strip())


def _path(prompt: str, directory: Path) -> Path:
    digest = hashlib.sha256(_normalise(prompt).encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def _dir() -> Path:
    return Path(os.environ.get("CACHE_DIR", DEFAULT_DIR))


def get(prompt: str) -> dict | None:
    """Return the cached payload for `prompt`, or None on miss."""
    directory = _dir()
    target = _path(prompt, directory)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def put(prompt: str, payload: dict) -> None:
    """Store `payload` for `prompt`."""
    directory = _dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = _path(prompt, directory)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
