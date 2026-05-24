"""Parser-level tests for the triage module (no network)."""

from llm_router.triage import _parse


def test_parses_full_object():
    result = _parse('{"verdict": "safe", "category": "code", "reason": ""}')
    assert result.verdict == "safe"
    assert result.category == "code"


def test_injection_with_reason():
    result = _parse(
        '{"verdict": "injection", "category": "chat", "reason": "override"}'
    )
    assert result.verdict == "injection"
    assert result.reason == "override"


def test_invalid_category_falls_back():
    result = _parse('{"verdict": "safe", "category": "weird"}')
    assert result.category == "chat"


def test_invalid_verdict_falls_back():
    result = _parse('{"verdict": "nope", "category": "code"}')
    assert result.verdict == "safe"


def test_unparseable_input():
    result = _parse("hello world")
    assert result.verdict == "safe"
    assert result.category == "chat"


def test_model_refusal_becomes_reject():
    result = _parse("I can't help with that request.")
    assert result.verdict == "reject"
    assert "refused" in result.reason
