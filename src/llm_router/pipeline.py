"""Decide between using the local draft or escalating to a large model."""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import draft as draft_mod
from . import openrouter
from . import preprocess
from . import request_log
from . import router as router_mod
from . import triage as triage_mod


@dataclass
class Answer:
    text: str
    route: str  # "small", "large", "reject", or "injection"
    verdict: str  # the triage verdict regardless of route
    confidence: float
    category: str | None = None
    large_model: str | None = None
    reason: str | None = None
    prompt_chars: int | None = None  # length of the prompt actually sent to the large model


def answer(
    prompt: str,
    *,
    small_model: str,
    large_models: dict[str, str],
    threshold: float,
    condense_threshold_chars: int = 0,
) -> Answer:
    """Triage, draft locally, then escalate to the category's large model."""
    started = time.perf_counter()

    decision = triage_mod.triage(prompt, model=small_model)
    if decision.verdict in ("reject", "injection"):
        label = "refused" if decision.verdict == "reject" else "blocked (prompt injection)"
        result = Answer(
            text=f"Request {label}: {decision.reason}".strip().rstrip(":"),
            route=decision.verdict,
            verdict=decision.verdict,
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
                verdict=decision.verdict,
                confidence=drafted.confidence,
                category=decision.category,
            )
        else:
            # Fall back to chat if the category has no configured slug.
            chosen = (
                large_models.get(decision.category)
                or large_models[router_mod.DEFAULT_CATEGORY]
            )
            forwarded = preprocess.condense(
                prompt, model=small_model, threshold_chars=condense_threshold_chars
            )
            text = openrouter.chat(
                [{"role": "user", "content": forwarded}], model=chosen
            )
            result = Answer(
                text=text,
                route="large",
                verdict=decision.verdict,
                confidence=drafted.confidence,
                category=decision.category,
                large_model=chosen,
                prompt_chars=len(forwarded),
            )

    request_log.log(
        {
            "prompt": prompt,
            "verdict": result.verdict,
            "route": result.route,
            "escalated": result.route == "large",
            "confidence": result.confidence,
            "category": result.category,
            "small_model": small_model,
            "large_model": result.large_model,
            "reason": result.reason,
            "prompt_chars_in": len(prompt),
            "prompt_chars_sent": result.prompt_chars,
            "latency_s": round(time.perf_counter() - started, 3),
        }
    )
    return result
