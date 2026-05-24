"""Tests for preprocess.structure_code_request (no network via monkeypatch)."""

from llm_router import preprocess


def test_returns_original_prompt_when_parse_fails(monkeypatch):
    prompt = "fix this bug in my code"
    monkeypatch.setattr(preprocess.ollama, "chat", lambda messages, model: "not json")
    assert preprocess.structure_code_request(prompt, model="m") == prompt


def test_returns_original_prompt_when_ollama_raises(monkeypatch):
    prompt = "fix this bug in my code"

    def boom(messages, model):
        raise RuntimeError("network down")

    monkeypatch.setattr(preprocess.ollama, "chat", boom)
    assert preprocess.structure_code_request(prompt, model="m") == prompt


def test_prepends_structured_header(monkeypatch):
    prompt = "debug: def add(a,b): return a-b"
    monkeypatch.setattr(
        preprocess.ollama,
        "chat",
        lambda messages, model: (
            '{"language": "python", "intent": "debug", '
            '"constraints": ["must return a+b", "preserve signature"]}'
        ),
    )
    result = preprocess.structure_code_request(prompt, model="m")
    assert "Structured request" in result
    assert "language: python" in result
    assert "intent: debug" in result
    assert "must return a+b" in result
    assert result.endswith(prompt)


def test_skips_empty_fields(monkeypatch):
    prompt = "explain this code"
    monkeypatch.setattr(
        preprocess.ollama,
        "chat",
        lambda messages, model: '{"language": "", "intent": "explain", "constraints": []}',
    )
    result = preprocess.structure_code_request(prompt, model="m")
    assert "language:" not in result
    assert "intent: explain" in result
    assert "constraints:" not in result
