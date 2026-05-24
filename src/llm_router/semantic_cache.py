"""Embedding-based cache that catches paraphrases the exact cache misses.

Each entry stores an embedding plus the cached payload. On lookup we embed
the incoming prompt and return the entry with the highest cosine similarity
above `SEMANTIC_THRESHOLD`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import httpx

DEFAULT_HOST = "http://localhost:11434"
# Tuned for llama3.1:8b chat-model embeddings, which score paraphrases
# around 0.78 and unrelated prompts around 0.50. A dedicated embedding
# model (e.g. nomic-embed-text) would warrant a higher threshold.
DEFAULT_THRESHOLD = 0.75
DEFAULT_PATH = Path(".cache") / "semantic.jsonl"
DEFAULT_TIMEOUT = 60.0


def _path() -> Path:
    return Path(os.environ.get("SEMANTIC_CACHE_PATH", DEFAULT_PATH))


def _threshold() -> float:
    return float(os.environ.get("SEMANTIC_THRESHOLD", DEFAULT_THRESHOLD))


def embed(text: str, model: str, *, host: str | None = None) -> list[float]:
    """Return an embedding vector for `text` from the local Ollama server."""
    url = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST) + "/api/embed"
    response = httpx.post(
        url, json={"model": model, "input": text}, timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def get(prompt: str, *, model: str) -> dict | None:
    """Return the cached payload for the closest prompt above threshold, else None."""
    path = _path()
    if not path.exists():
        return None
    query = embed(prompt, model)
    best_score = -1.0
    best_payload: dict | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = _cosine(query, entry["embedding"])
            if score > best_score:
                best_score = score
                best_payload = entry["payload"]
    if best_payload is not None and best_score >= _threshold():
        return best_payload
    return None


def put(prompt: str, payload: dict, *, model: str) -> None:
    """Append an entry keyed by the prompt's embedding."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    vector = embed(prompt, model)
    entry = {"prompt": prompt, "embedding": vector, "payload": payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
