"""Classify a request as safe or reject it before any expensive call."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import ollama

VERDICTS = ("safe", "reject")

SYSTEM_PROMPT = """You are a request triage classifier.
Decide whether a user request should be answered.

Reject when the request is:
- nonsense, empty, or pure spam,
- asking for clearly illegal or dangerous instructions,
- abusive or hateful with no legitimate intent.

Otherwise mark it safe.

Respond with a single JSON object and nothing else:
{"verdict": "safe" | "reject", "reason": "<short reason>"}"""


@dataclass
class Verdict:
    verdict: str
    reason: str


def classify(prompt: str, model: str) -> Verdict:
    """Return a triage verdict for `prompt`."""
    raw = ollama.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    return _parse(raw)


def _parse(raw: str) -> Verdict:
    # Fail open: if the classifier is unparseable we let the request through
    # rather than block legitimate traffic.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return Verdict(verdict="safe", reason="unparseable classifier output")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return Verdict(verdict="safe", reason="invalid classifier json")

    verdict = str(data.get("verdict", "safe")).lower().strip()
    if verdict not in VERDICTS:
        verdict = "safe"
    reason = str(data.get("reason", "")).strip()
    return Verdict(verdict=verdict, reason=reason)
