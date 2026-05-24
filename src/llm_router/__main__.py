"""CLI entrypoint: route a prompt through the cascade."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from . import pipeline


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m llm_router <prompt>", file=sys.stderr)
        return 2

    prompt = " ".join(args)
    result = pipeline.answer(
        prompt,
        small_model=os.environ.get("SMALL_MODEL", "llama3.1:8b"),
        large_model=os.environ.get("LARGE_MODEL", "anthropic/claude-sonnet-4.5"),
        threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")),
    )
    print(
        f"[route={result.route} confidence={result.confidence:.2f}"
        + (f" model={result.large_model}" if result.large_model else "")
        + "]",
        file=sys.stderr,
    )
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
