"""Parser-level tests for the router module (no network)."""

from llm_router.router import DEFAULT_CATEGORY, _parse


def test_parses_each_category():
    for category in ("reasoning", "code", "fast", "chat"):
        assert _parse(f'{{"category": "{category}"}}') == category


def test_unknown_category_falls_back():
    assert _parse('{"category": "weird"}') == DEFAULT_CATEGORY


def test_unparseable_falls_back():
    assert _parse("not json") == DEFAULT_CATEGORY


def test_extracts_json_from_prose():
    assert _parse('sure: {"category": "code"} done') == "code"
