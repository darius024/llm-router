"""Tiny HTTP server exposing POST /chat over the cascade.

Built on stdlib http.server to avoid extra dependencies. Suitable for local
experiments only — there is no auth, rate limiting, or TLS.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dotenv import load_dotenv

from . import pipeline


def _large_models() -> dict[str, str]:
    default = "anthropic/claude-sonnet-4.5"
    return {
        "reasoning": os.environ.get("LARGE_MODEL_REASONING", default),
        "chat": os.environ.get("LARGE_MODEL_CHAT", default),
        "fast": os.environ.get("LARGE_MODEL_FAST", default),
        "code": os.environ.get("LARGE_MODEL_CODE", default),
    }


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path != "/chat":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            self._json(400, {"error": "missing prompt"})
            return
        try:
            result = pipeline.answer(
                prompt,
                small_model=os.environ.get("SMALL_MODEL", "llama3.1:8b"),
                large_models=_large_models(),
                threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")),
                condense_threshold_chars=int(
                    os.environ.get("CONDENSE_THRESHOLD_CHARS", "0")
                ),
                use_semantic_cache=os.environ.get("SEMANTIC_CACHE", "0") == "1",
            )
        except Exception as exc:  # surface upstream failures as 502
            self._json(502, {"error": str(exc)})
            return
        self._json(200, asdict(result))

    def log_message(self, format: str, *args) -> None:  # silence default access log
        return

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    load_dotenv()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"llm-router listening on http://{host}:{port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    import sys

    bind_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve(port=bind_port)
