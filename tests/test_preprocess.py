"""Unit tests for preprocess.condense (no network for the short-circuit paths)."""

from llm_router import preprocess


def test_returns_prompt_unchanged_when_under_threshold():
    short = "hello world"
    assert preprocess.condense(short, model="unused", threshold_chars=100) == short


def test_returns_prompt_unchanged_when_threshold_disabled():
    long = "x" * 10_000
    assert preprocess.condense(long, model="unused", threshold_chars=0) == long


def test_keeps_original_when_condenser_makes_it_longer(monkeypatch):
    long = "abc " * 1000  # 4000 chars
    monkeypatch.setattr(
        preprocess.ollama,
        "chat",
        lambda messages, model: long + " plus extra",
    )
    assert preprocess.condense(long, model="m", threshold_chars=100) == long


def test_keeps_original_when_condenser_returns_empty(monkeypatch):
    long = "abc " * 1000
    monkeypatch.setattr(preprocess.ollama, "chat", lambda messages, model: "   ")
    assert preprocess.condense(long, model="m", threshold_chars=100) == long


def test_uses_condensed_when_shorter(monkeypatch):
    long = "abc " * 1000
    monkeypatch.setattr(preprocess.ollama, "chat", lambda messages, model: "short")
    assert preprocess.condense(long, model="m", threshold_chars=100) == "short"
