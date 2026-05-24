"""Tests for the prompt cache (no network)."""

import os

import pytest

from llm_router import cache


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    return tmp_path


def test_miss_returns_none(tmp_cache):
    assert cache.get("anything") is None


def test_round_trip(tmp_cache):
    cache.put("hello", {"text": "world"})
    assert cache.get("hello") == {"text": "world"}


def test_whitespace_is_normalised(tmp_cache):
    cache.put("hello   world", {"text": "x"})
    assert cache.get("\thello world\n") == {"text": "x"}


def test_distinct_prompts_get_distinct_entries(tmp_cache):
    cache.put("a", {"text": "1"})
    cache.put("b", {"text": "2"})
    assert cache.get("a") == {"text": "1"}
    assert cache.get("b") == {"text": "2"}


def test_corrupted_file_returns_none(tmp_cache):
    cache.put("hello", {"text": "world"})
    # Corrupt the on-disk file.
    for name in os.listdir(tmp_cache):
        (tmp_cache / name).write_text("not json")
    assert cache.get("hello") is None
