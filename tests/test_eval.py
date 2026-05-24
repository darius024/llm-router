"""Tests for the evaluation harness (pipeline is monkeypatched)."""

from __future__ import annotations

from llm_router import eval as eval_mod
from llm_router import pipeline


def test_run_produces_one_row_per_prompt(monkeypatch):
    def fake_answer(prompt, **kwargs):
        return pipeline.Answer(
            text="ok",
            route="small",
            verdict="safe",
            confidence=0.9,
            category="chat",
        )

    monkeypatch.setattr(eval_mod.pipeline, "answer", fake_answer)
    rows = eval_mod.run()
    assert len(rows) == len(eval_mod.PROMPTS)
    assert {row["route"] for row in rows} == {"small"}


def test_render_includes_summary_and_table():
    rows = [
        {
            "label": "easy",
            "prompt": "hi",
            "route": "small",
            "category": "chat",
            "confidence": 0.9,
            "latency_s": 0.1,
            "text": "ok",
        },
        {
            "label": "reject",
            "prompt": "no",
            "route": "reject",
            "category": "chat",
            "confidence": 0.0,
            "latency_s": 0.2,
            "text": "Request refused",
        },
    ]
    report = eval_mod.render(rows)
    assert "prompts: **2**" in report
    assert "small" in report and "reject" in report
    assert report.count("|") >= 16  # header + 2 data rows of 7 cells
