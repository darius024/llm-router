"""Pre-process prompts before they reach the large model."""

from __future__ import annotations

import json

from . import ollama

CONDENSE_SYSTEM_PROMPT = """You rewrite a long user request into a concise,
self-contained version for a more expensive model to answer.

Rules:
- Preserve the original intent and every constraint.
- Keep verbatim any quoted text, code, numbers, names, or identifiers.
- Drop greetings, pleasantries, and repetition.
- Output only the rewritten request. No preamble, no explanation."""


CODE_STRUCTURE_SYSTEM_PROMPT = """You extract structured fields from a code
request so a stronger model can answer it precisely.

Respond with a single JSON object and nothing else:
{"language": "<programming language or empty>",
 "intent": "<one of: write | debug | explain | review | refactor>",
 "constraints": ["<short constraint>", ...]}

If a field is unknown, use "" or []. Output only the JSON."""


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


def structure_code_request(prompt: str, model: str) -> str:
    """Annotate a code request with extracted fields for the large model.

    Returns the original prompt with a short structured header prepended.
    On any failure, returns the original prompt unchanged.
    """
    try:
        raw = ollama.chat(
            [
                {"role": "system", "content": CODE_STRUCTURE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
        )
    except Exception:
        return prompt

    fields = _parse_structure(raw)
    if not fields:
        return prompt

    header_lines = ["Structured request (extracted by triage):"]
    if fields.get("language"):
        header_lines.append(f"- language: {fields['language']}")
    if fields.get("intent"):
        header_lines.append(f"- intent: {fields['intent']}")
    constraints = fields.get("constraints") or []
    if constraints:
        header_lines.append("- constraints:")
        header_lines.extend(f"  - {c}" for c in constraints)
    header_lines.append("\nOriginal request:\n")
    return "\n".join(header_lines) + prompt


def _parse_structure(raw: str) -> dict | None:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    # Coerce types defensively so a misbehaving model can't break formatting.
    constraints = data.get("constraints", [])
    if not isinstance(constraints, list):
        constraints = []
    return {
        "language": str(data.get("language", "")).strip(),
        "intent": str(data.get("intent", "")).strip(),
        "constraints": [str(c).strip() for c in constraints if str(c).strip()],
    }
