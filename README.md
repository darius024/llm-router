# llm-router

A learning project exploring a **two-tier LLM cascade**: a small open-source
model running locally (via [Ollama](https://ollama.com)) sits in front of a
larger hosted model (via [OpenRouter](https://openrouter.ai)) and progressively
takes on more responsibility.

## Goal

Iterate on the small model's role one step at a time, keeping each step
minimal and easy to understand:

1. **Drafter** — small model answers directly; escalate to the large model
   only when it is not confident.
2. **Filter / guard** — small model rejects junk, unsafe, or off-topic
   requests before they reach the large model.
3. **Router** — small model classifies the request and picks the most
   appropriate downstream model (e.g. code vs. chat vs. reasoning).
4. **Preprocessor** — small model rewrites / summarises / extracts
   structure so the large model gets a tighter, cheaper prompt.

Each stage is a separate commit (or small set of commits) so the diff shows
exactly what changed.

## Stack

- **Small model:** Ollama, local. Default `llama3.1:8b`.
- **Large models:** OpenRouter, picked per request. Initial pool:
  - `anthropic/claude-sonnet-4.5` — general chat / code (default)
  - `openai/gpt-5` — hard reasoning
  - `google/gemini-2.5-flash` — fast / cheap fallback
- **Language:** Python (kept tiny — standard lib + one HTTP client).

> Model slugs change as providers release new versions. Check
> [openrouter.ai/models](https://openrouter.ai/models) and update `.env`
> if a slug 404s.

## Setup

```bash
cp .env.example .env   # then fill in OPENROUTER_API_KEY
ollama pull llama3.1:8b
```
