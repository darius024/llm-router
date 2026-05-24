"""Decide between using the local draft or escalating to a large model."""

from __future__ import annotations

from dataclasses import dataclass

from . import draft as draft_mod
from . import openrouter


@dataclass
class Answer:
    text: str
    route: str  # "small" or "large"
    confidence: float
    large_model: str | None = None


def answer(
    prompt: str,
    *,
    small_model: str,
    large_model: str,
    threshold: float,
) -> Answer:
    """Draft locally; escalate to the large model when confidence is too low."""
    drafted = draft_mod.draft(prompt, model=small_model)
    if drafted.confidence >= threshold and drafted.answer:
        return Answer(text=drafted.answer, route="small", confidence=drafted.confidence)

    text = openrouter.chat([{"role": "user", "content": prompt}], model=large_model)
    return Answer(
        text=text,
        route="large",
        confidence=drafted.confidence,
        large_model=large_model,
    )
