"""Pre-process prompts before they reach the large model."""

from __future__ import annotations

from . import ollama

CONDENSE_SYSTEM_PROMPT = """You rewrite a long user request into a concise,
self-contained version for a more expensive model to answer.

Rules:
- Preserve the original intent and every constraint.
- Keep verbatim any quoted text, code, numbers, names, or identifiers.
- Drop greetings, pleasantries, and repetition.
- Output only the rewritten request. No preamble, no explanation."""


def condense(prompt: str, model: str, *, threshold_chars: int) -> str:
    """Return a shorter equivalent of `prompt`, or `prompt` itself when short."""
    if threshold_chars <= 0 or len(prompt) <= threshold_chars:
        return prompt
    rewritten = ollama.chat(
        [
            {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
    ).strip()
    # Guard against pathological outputs: if the model produced something
    # empty or longer than the original, keep the original.
    if not rewritten or len(rewritten) >= len(prompt):
        return prompt
    return rewritten
