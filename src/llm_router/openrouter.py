"""Minimal client for the OpenRouter chat completions API."""

from __future__ import annotations

import os

import httpx

Message = dict[str, str]

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 120.0


def chat(messages: list[Message], model: str, *, api_key: str | None = None) -> str:
    """Send a chat request to OpenRouter and return the assistant's text."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
