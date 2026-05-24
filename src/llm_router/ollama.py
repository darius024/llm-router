"""Minimal client for a local Ollama server."""

from __future__ import annotations

import os

import httpx

Message = dict[str, str]

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0


def chat(messages: list[Message], model: str, *, host: str | None = None) -> str:
    """Send a chat request to Ollama and return the assistant's text.

    `messages` follows the OpenAI shape: [{"role": "user", "content": "..."}].
    """
    base = host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    response = httpx.post(
        f"{base.rstrip('/')}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
