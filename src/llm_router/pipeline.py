"""Decide between using the local draft or escalating to a large model."""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import draft as draft_mod
from . import filter as filter_mod
from . import openrouter
from . import request_log


@dataclass
class Answer:
    text: str
    route: str  # "small", "large", or "reject"
    confidence: float
    large_model: str | None = None
    reason: str | None = None


def answer(
    prompt: str,
    *,
    small_model: str,
    large_model: str,
    threshold: float,
) -> Answer:
    """Filter, draft locally, then escalate to the large model when needed."""
    started = time.perf_counter()

    verdict = filter_mod.classify(prompt, model=small_model)
    if verdict.verdict in ("reject", "injection"):
        label = "refused" if verdict.verdict == "reject" else "blocked (prompt injection)"
        result = Answer(
            text=f"Request {label}: {verdict.reason}".strip().rstrip(":"),
            route=verdict.verdict,
            confidence=0.0,
            reason=verdict.reason,
        )
    else:
        drafted = draft_mod.draft(prompt, model=small_model)
        if drafted.confidence >= threshold and drafted.answer:
            result = Answer(
                text=drafted.answer, route="small", confidence=drafted.confidence
            )
        else:
            text = openrouter.chat(
                [{"role": "user", "content": prompt}], model=large_model
            )
            result = Answer(
                text=text,
                route="large",
                confidence=drafted.confidence,
                large_model=large_model,
            )

    request_log.log(
        {
            "prompt": prompt,
            "route": result.route,
            "confidence": result.confidence,
            "small_model": small_model,
            "large_model": result.large_model,
            "reason": result.reason,
            "latency_s": round(time.perf_counter() - started, 3),
        }
    )
    return result
