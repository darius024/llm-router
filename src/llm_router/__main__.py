"""CLI entrypoint: forward a single prompt to the large model."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from . import draft


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m llm_router <prompt>", file=sys.stderr)
        return 2

    prompt = " ".join(args)
    small_model = os.environ.get("SMALL_MODEL", "llama3.1:8b")
    reply = draft.draft(prompt, model=small_model)
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
