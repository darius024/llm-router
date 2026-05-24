"""Classify a request into a category used to pick a large model."""

from __future__ import annotations

import json

from . import ollama

CATEGORIES = ("reasoning", "chat", "fast", "code")
DEFAULT_CATEGORY = "chat"

SYSTEM_PROMPT = """You assign a category to a user request so the right model
can answer it.

Categories:
- reasoning: multi-step logic, maths, planning, complex analysis.
- code: writing, debugging, explaining, or reviewing source code.
- fast: short factual lookups, greetings, trivial questions.
- chat: everything else (open-ended discussion, advice, summaries).

Pick exactly one. Respond with a single JSON object and nothing else:
{"category": "reasoning" | "code" | "fast" | "chat"}"""


def classify(prompt: str, model: str) -> str:
    """Return one of CATEGORIES for `prompt`."""
    raw = ollama.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    return _parse(raw)


def _parse(raw: str) -> str:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return DEFAULT_CATEGORY
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return DEFAULT_CATEGORY
    category = str(data.get("category", "")).lower().strip()
    return category if category in CATEGORIES else DEFAULT_CATEGORY
