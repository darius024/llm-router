"""Draft an answer using the local small model, with a self-rated confidence."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import ollama

# The small model is asked to return strict JSON so we can parse a confidence score.
SYSTEM_PROMPT = """You are a careful assistant.
Answer the user's request, then rate how confident you are that your answer is
correct, complete, and unambiguous on a scale from 0.0 to 1.0.

Respond with a single JSON object and nothing else:
{"answer": "<your answer>", "confidence": <number between 0 and 1>}"""


@dataclass
class Draft:
    answer: str
    confidence: float
    raw: str


def draft(prompt: str, model: str) -> Draft:
    """Return the small model's answer together with its self-rated confidence."""
    raw = ollama.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    answer, confidence = _parse(raw)
    return Draft(answer=answer, confidence=confidence, raw=raw)


def _parse(raw: str) -> tuple[str, float]:
    # If the model wraps JSON in prose, isolate the first {...} block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return raw.strip(), 0.0
    try:
        data = json.loads(raw[start : end + 1])
        answer = str(data.get("answer", "")).strip()
        confidence = float(data.get("confidence", 0.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw.strip(), 0.0
    # Clamp to [0, 1] so a misbehaving model can't poison routing decisions.
    confidence = max(0.0, min(1.0, confidence))
    return answer, confidence
