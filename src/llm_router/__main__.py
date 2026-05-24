"""CLI entrypoint: route a prompt through the cascade."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from . import pipeline


def _large_models() -> dict[str, str]:
    """Read the per-category large-model slugs from the environment."""
    default = "anthropic/claude-sonnet-4.5"
    return {
        "reasoning": os.environ.get("LARGE_MODEL_REASONING", default),
        "chat": os.environ.get("LARGE_MODEL_CHAT", default),
        "fast": os.environ.get("LARGE_MODEL_FAST", default),
        "code": os.environ.get("LARGE_MODEL_CODE", default),
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m llm_router [--stream] <prompt>", file=sys.stderr)
        return 2

    streaming = False
    if args and args[0] == "--stream":
        streaming = True
        args = args[1:]
    if not args:
        print("usage: python -m llm_router [--stream] <prompt>", file=sys.stderr)
        return 2

    prompt = " ".join(args)
    common = dict(
        small_model=os.environ.get("SMALL_MODEL", "llama3.1:8b"),
        large_models=_large_models(),
        threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7")),
        condense_threshold_chars=int(os.environ.get("CONDENSE_THRESHOLD_CHARS", "0")),
        use_semantic_cache=os.environ.get("SEMANTIC_CACHE", "0") == "1",
    )
    if streaming:
        def emit(chunk: str) -> None:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        result = pipeline.answer_stream(prompt, on_chunk=emit, **common)
        sys.stdout.write("\n")
    else:
        result = pipeline.answer(prompt, **common)
    print(
        f"[route={result.route} confidence={result.confidence:.2f}"
        + (f" category={result.category}" if result.category else "")
        + (f" model={result.large_model}" if result.large_model else "")
        + (f" reason={result.reason!r}" if result.reason else "")
        + "]",
        file=sys.stderr,
    )
    if not streaming:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
