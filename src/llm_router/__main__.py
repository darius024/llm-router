"""CLI entrypoint: forward a single prompt to the large model."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from . import openrouter


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m llm_router <prompt>", file=sys.stderr)
        return 2

    prompt = " ".join(args)
    model = os.environ.get("LARGE_MODEL", "anthropic/claude-sonnet-4.5")
    reply = openrouter.chat([{"role": "user", "content": prompt}], model=model)
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
