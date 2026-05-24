"""Parser-level tests for the filter module (no network)."""

from llm_router.filter import _parse


def test_parses_safe_verdict():
    result = _parse('{"verdict": "safe", "reason": "ok"}')
    assert result.verdict == "safe"
    assert result.reason == "ok"


def test_parses_reject_verdict():
    result = _parse('{"verdict": "reject", "reason": "spam"}')
    assert result.verdict == "reject"


def test_parses_injection_verdict():
    result = _parse('{"verdict": "injection", "reason": "override attempt"}')
    assert result.verdict == "injection"


def test_unknown_verdict_falls_back_to_safe():
    result = _parse('{"verdict": "weird", "reason": "x"}')
    assert result.verdict == "safe"


def test_extracts_json_from_surrounding_prose():
    raw = 'Here you go: {"verdict": "reject", "reason": "abuse"}. thanks!'
    result = _parse(raw)
    assert result.verdict == "reject"
    assert result.reason == "abuse"


def test_unparseable_input_fails_open():
    result = _parse("totally not json")
    assert result.verdict == "safe"


def test_invalid_json_fails_open():
    result = _parse('{"verdict": "reject", "reason":}')
    assert result.verdict == "safe"
