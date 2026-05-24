"""Tests for the HTTP server (pipeline is monkeypatched)."""

from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

import pytest

from llm_router import pipeline, server


@pytest.fixture
def running_server(monkeypatch):
    def fake_answer(prompt, **kwargs):
        return pipeline.Answer(
            text=f"echo: {prompt}",
            route="small",
            verdict="safe",
            confidence=1.0,
            category="chat",
        )

    monkeypatch.setattr(server.pipeline, "answer", fake_answer)
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


def _post(port: int, path: str, body: dict | bytes) -> tuple[int, dict | None]:
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            return response.status, json.loads(response.read())
    except Exception as exc:  # urllib raises on non-2xx
        return getattr(exc, "code", 0), None


def test_chat_returns_answer(running_server):
    status, payload = _post(running_server, "/chat", {"prompt": "hello"})
    assert status == 200
    assert payload["text"] == "echo: hello"
    assert payload["route"] == "small"


def test_missing_prompt_is_400(running_server):
    status, _ = _post(running_server, "/chat", {})
    assert status == 400


def test_bad_json_is_400(running_server):
    status, _ = _post(running_server, "/chat", b"not json")
    assert status == 400


def test_unknown_path_is_404(running_server):
    status, _ = _post(running_server, "/other", {"prompt": "x"})
    assert status == 404
