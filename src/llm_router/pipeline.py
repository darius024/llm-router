"""Decide between using the local draft or escalating to a large model."""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import draft as draft_mod
from . import openrouter
from . import request_log
from . import router as router_mod
from . import triage as triage_mod


@dataclass
class Answer:
    text: str
    route: str  # "small", "large", "reject", or "injection"
    confidence: float
    category: str | None = None
    large_model: str | None = None
    reason: str | None = None


def answer(
    prompt: str,
    *,
    small_model: str,
    large_models: dict[str, str],
    threshold: float,
) -> Answer:
    """Triage, draft locally, then escalate to the category's large model."""
    started = time.perf_counter()

    decision = triage_mod.triage(prompt, model=small_model)
    if decision.verdict in ("reject", "injection"):
        label = "refused" if decision.verdict == "reject" else "blocked (prompt injection)"
        result = Answer(
            text=f"Request {label}: {decision.reason}".strip().rstrip(":"),
            route=decision.verdict,
            confidence=0.0,
            category=decision.category,
            reason=decision.reason,
        )
    else:
        drafted = draft_mod.draft(prompt, model=small_model)
        if drafted.confidence >= threshold and drafted.answer:
            result = Answer(
                text=drafted.answer,
                route="small",
                confidence=drafted.confidence,
                category=decision.category,
            )
        else:
            # Fall back to chat if the category has no configured slug.
            chosen = (
                large_models.get(decision.category)
                or large_models[router_mod.DEFAULT_CATEGORY]
            )
            text = openrouter.chat(
                [{"role": "user", "content": prompt}], model=chosen
            )
            result = Answer(
                text=text,
                route="large",
                confidence=drafted.confidence,
                category=decision.category,
                large_model=chosen,
            )

    request_log.log(
        {
            "prompt": prompt,
            "route": result.route,
            "confidence": result.confidence,
            "category": result.category,
            "small_model": small_model,
            "large_model": result.large_model,
            "reason": result.reason,
            "latency_s": round(time.perf_counter() - started, 3),
        }
    )
    return result
