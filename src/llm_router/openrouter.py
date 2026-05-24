"""Minimal client for the OpenRouter chat completions API."""

from __future__ import annotations

import json
import os
from typing import Iterator

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


def stream(messages: list[Message], model: str, *, api_key: str | None = None) -> Iterator[str]:
    """Yield assistant text chunks as they arrive over server-sent events."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    with httpx.stream(
        "POST",
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, "stream": True},
        timeout=DEFAULT_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta
