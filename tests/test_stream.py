"""Tests for pipeline.answer_stream (clients are monkeypatched)."""

from __future__ import annotations

import pytest

from llm_router import pipeline


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "exact"))
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "log.jsonl"))


def test_large_route_streams_chunks(monkeypatch, isolated_caches):
    monkeypatch.setattr(
        pipeline.triage_mod,
        "triage",
        lambda prompt, model: pipeline.triage_mod.Triage("safe", "chat", ""),
    )
    monkeypatch.setattr(
        pipeline.draft_mod,
        "draft",
        lambda prompt, model: pipeline.draft_mod.Draft("low", 0.1, "{}"),
    )
    monkeypatch.setattr(
        pipeline.preprocess, "condense", lambda prompt, model, threshold_chars: prompt
    )
    monkeypatch.setattr(
        pipeline.openrouter, "stream", lambda messages, model: iter(["Hel", "lo ", "world"])
    )

    seen: list[str] = []
    result = pipeline.answer_stream(
        "anything",
        small_model="s",
        large_models={"chat": "L"},
        threshold=0.9,
        on_chunk=seen.append,
    )

    assert seen == ["Hel", "lo ", "world"]
    assert result.text == "Hello world"
    assert result.route == "large"


def test_small_route_emits_single_chunk(monkeypatch, isolated_caches):
    monkeypatch.setattr(
        pipeline.triage_mod,
        "triage",
        lambda prompt, model: pipeline.triage_mod.Triage("safe", "chat", ""),
    )
    monkeypatch.setattr(
        pipeline.draft_mod,
        "draft",
        lambda prompt, model: pipeline.draft_mod.Draft("local answer", 1.0, "{}"),
    )

    seen: list[str] = []
    result = pipeline.answer_stream(
        "hi",
        small_model="s",
        large_models={"chat": "L"},
        threshold=0.5,
        on_chunk=seen.append,
    )

    assert seen == ["local answer"]
    assert result.route == "small"
