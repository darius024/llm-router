"""Tests for the semantic cache (no network — embed() is monkeypatched)."""

from __future__ import annotations

import pytest

from llm_router import semantic_cache


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_CACHE_PATH", str(tmp_path / "sem.jsonl"))
    return tmp_path


def _fake_embed(table: dict[str, list[float]]):
    def inner(text, model, *, host=None):
        return table[text]
    return inner


def test_miss_when_empty(tmp_store, monkeypatch):
    monkeypatch.setattr(
        semantic_cache, "embed", _fake_embed({"hi": [1.0, 0.0]})
    )
    assert semantic_cache.get("hi", model="m") is None


def test_hit_on_identical_vector(tmp_store, monkeypatch):
    monkeypatch.setattr(
        semantic_cache, "embed", _fake_embed({"hello": [1.0, 0.0]})
    )
    semantic_cache.put("hello", {"text": "hi"}, model="m")
    assert semantic_cache.get("hello", model="m") == {"text": "hi"}


def test_hit_on_close_paraphrase(tmp_store, monkeypatch):
    # Two near-parallel vectors -> cosine ~0.995, above the default threshold.
    monkeypatch.setattr(
        semantic_cache,
        "embed",
        _fake_embed({"a": [1.0, 0.0], "b": [0.995, 0.0999]}),
    )
    semantic_cache.put("a", {"text": "answer"}, model="m")
    assert semantic_cache.get("b", model="m") == {"text": "answer"}


def test_miss_when_below_threshold(tmp_store, monkeypatch):
    # Orthogonal vectors -> cosine 0.0, well below threshold.
    monkeypatch.setattr(
        semantic_cache,
        "embed",
        _fake_embed({"a": [1.0, 0.0], "c": [0.0, 1.0]}),
    )
    semantic_cache.put("a", {"text": "answer"}, model="m")
    assert semantic_cache.get("c", model="m") is None


def test_custom_threshold_can_admit_more(tmp_store, monkeypatch):
    monkeypatch.setenv("SEMANTIC_THRESHOLD", "0.3")
    monkeypatch.setattr(
        semantic_cache,
        "embed",
        _fake_embed({"a": [1.0, 0.0], "c": [0.5, 0.5]}),
    )
    semantic_cache.put("a", {"text": "answer"}, model="m")
    # cosine([1,0],[0.5,0.5]) ≈ 0.707, above 0.3
    assert semantic_cache.get("c", model="m") == {"text": "answer"}
