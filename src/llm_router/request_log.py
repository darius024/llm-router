"""Append one JSON line per request to a local log file."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("logs/requests.jsonl")


def log(event: dict[str, Any], *, path: Path | None = None) -> None:
    """Append `event` as a single JSON line, stamped with an ISO timestamp."""
    target = path or Path(os.environ.get("LOG_PATH", DEFAULT_PATH))
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
