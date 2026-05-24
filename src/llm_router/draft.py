"""Draft an answer using the local small model."""

from __future__ import annotations

from . import ollama


def draft(prompt: str, model: str) -> str:
    """Return the small model's answer to `prompt`."""
    return ollama.chat([{"role": "user", "content": prompt}], model=model)
