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
5. **Productionisation** — exact + semantic caches, streaming output,
   a tiny HTTP server, and a fixed-prompt evaluation harness.

Each stage is a separate commit (or small set of commits) so the diff shows
exactly what changed.

## Stack

- **Small model:** Ollama, local. Default `llama3.1:8b`.
- **Large models:** OpenRouter, picked per request. Initial pool:
  - `anthropic/claude-sonnet-4.5` — general chat / code (default)
  - `openai/gpt-5` — hard reasoning
  - `google/gemini-2.5-flash` — fast / cheap fallback
- **Language:** Python 3.11+ (standard library + `httpx` + `python-dotenv`).

> Model slugs change as providers release new versions. Check
> [openrouter.ai/models](https://openrouter.ai/models) and update `.env`
> if a slug 404s.

## Setup

```bash
cp .env.example .env       # then fill in OPENROUTER_API_KEY
ollama pull llama3.1:8b
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# one-shot CLI
python -m llm_router "what is the capital of France"

# stream the large-model answer as it arrives
python -m llm_router --stream "write a haiku about routers"

# turn on the semantic cache (paraphrase-aware)
SEMANTIC_CACHE=1 python -m llm_router "tell me france's capital city"

# expose POST /chat on http://127.0.0.1:8000
python -m llm_router.server

# evaluate the cascade on a fixed prompt set, write reports/eval.md
python -m llm_router.eval
```

## Tests

```bash
.venv/bin/pytest -q              # unit tests only (offline)
.venv/bin/pytest -q -m live      # live tests against Ollama
```

## Learning notes

These notes explain each piece at a high level — what it does, how it is
built, and where the same pattern shows up in production systems.

### 1. Small model as drafter with self-rated confidence
**Idea.** Ask the small model to answer **and** to rate its own
confidence in a single JSON reply. If confidence is below a threshold,
escalate to the larger model.
**How.** A system prompt tells the model to reply as
`{"answer": "...", "confidence": 0..1}`. A defensive parser pulls the
first `{...}` block out of the raw text and clamps the score.
**Real-world use.** This is the core of cost-saving cascades like
[FrugalGPT](https://arxiv.org/abs/2305.05176) and the routing layers used by
Martian, OpenRouter "auto", and Anthropic's prompt caching demos. Self-rated
confidence is cheap but noisy; production systems usually combine it with
verifier models or held-out evaluation sets.

### 2. Filter / safety triage
**Idea.** Before any escalation, classify the prompt as `safe`,
`reject` (unsafe / off-topic) or `injection` (prompt-injection attempt).
**How.** Few-shot system prompt that lists explicit injection patterns
(`ignore previous instructions`, `reveal your system prompt`, role-swap
attempts like "you are now DAN"). The parser **fails open to `safe`** so a
malformed reply does not silently block legitimate traffic.
**Real-world use.** Mirrors layers like OpenAI's moderation endpoint,
Anthropic's constitutional classifiers, Lakera Guard, and Llama Guard.
Filtering with a cheap local model is also how teams keep paid-API spend
low: refusals never hit the expensive provider.

### 3. Router by category
**Idea.** The small model labels each prompt as `reasoning`, `chat`,
`fast` or `code`, and the pipeline maps that label to a configured
OpenRouter slug. This lets you route hard reasoning to GPT-5, casual chat
to Claude, and short factual questions to Gemini Flash.
**How.** A second JSON-returning classification call, with a default
category on parse failure so the system always picks *something*.
**Real-world use.** This is exactly what mixture-of-models products do
(OpenRouter auto, Not Diamond, Martian, RouteLLM). The pattern also
appears inside agentic frameworks that pick between a "fast" and "smart"
tool depending on task difficulty.

### 4. Combined triage
**Idea.** Merge filter + router into a single small-model call returning
`{verdict, category, reason}` to cut latency in half. Watch out: merging
weakens injection detection unless you re-add explicit examples to the
combined system prompt — which is why the regression test suite exists.
**How.** Single `Triage` dataclass; few-shot system prompt with both
safety rules and category definitions, plus injection patterns.
**Real-world use.** Real systems batch many policy checks into one
classifier head for the same reason — every extra round trip is latency
the user feels.

### 5. Preprocessor — condensing and structuring
**Condensing.** If the prompt is longer than a threshold, the small
model rewrites it into a shorter form before escalation. The
implementation refuses to use the rewrite if it is empty or *longer*
than the original — a defensive guard against rewrite drift.
**Structuring code requests.** For `code`-category prompts, the small
model extracts `{language, intent, constraints}` and prepends them as a
labelled header, so the large model spends fewer tokens guessing what
the user wants.
**Real-world use.** Identical to "prompt compression" papers like
[LLMLingua](https://github.com/microsoft/LLMLingua), and to the
"summarise the conversation so far" loops in long-running agents.
Coding assistants prepend structured context (file paths, language,
error messages) for the same reason.

### 6. Exact-prompt cache
**Idea.** Hash the normalised prompt with SHA-256 and store the full
answer on disk. A repeat request returns instantly and costs nothing.
**How.** Normalisation collapses whitespace so trivially different
prompts share an entry. Refusals are deliberately **not** cached so a
later policy change can revisit them.
**Real-world use.** Every chat product caches at some layer — even
just at the CDN. Anthropic's
[prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
and OpenAI's automatic input caching are server-side flavours of this
idea; the client-side version here is what tools like LiteLLM and
LangChain's `InMemoryCache` provide.

### 7. Semantic cache
**Idea.** Catch *paraphrases* the exact cache misses. Embed every
stored prompt; on lookup, embed the new prompt and return the best
match above a cosine-similarity threshold.
**How.** Calls Ollama's `/api/embed` (any chat model also exposes
embeddings). Threshold is tuned per embedding model — llama3.1 chat
embeddings score paraphrases around 0.78 and unrelated prompts around
0.50, so the default lives at 0.75.
**Real-world use.** This is how systems like
[GPTCache](https://github.com/zilliztech/GPTCache) and RAG-style retrieval
caches work. Production systems use dedicated embedding models
(`text-embedding-3-small`, `nomic-embed-text`) and store vectors in
Pinecone, Qdrant or pgvector. Picking the right threshold is the hard
part: too high misses paraphrases, too low serves stale answers.

### 8. Streaming output
**Idea.** Don't wait for the whole large-model reply — stream tokens as
they arrive. The first byte feels instant; total time is unchanged.
**How.** OpenRouter speaks server-sent events; the client iterates
lines, strips `data: ` prefixes, parses each JSON delta and yields the
`content` field. The pipeline takes a callback so the CLI can write to
stdout chunk-by-chunk.
**Real-world use.** Every chat UI streams for the same reason. SSE is
the de-facto standard; WebSockets and HTTP/2 push exist but SSE is
simpler and works through every proxy.

### 9. Minimal HTTP server
**Idea.** Wrap the cascade behind a tiny `POST /chat` endpoint so other
tools can call it like any other LLM provider.
**How.** Standard-library `http.server.ThreadingHTTPServer` plus a
hand-written `BaseHTTPRequestHandler`. No framework dependency. The body
is JSON `{"prompt": "..."}` and the response is the full `Answer`
dataclass serialised with `dataclasses.asdict`.
**Real-world use.** The same shape as an OpenAI-compatible
`/v1/chat/completions` endpoint, which is what makes products like
LiteLLM and Ollama drop-in replacements for paid APIs. A real
deployment would add auth, rate limiting, TLS and structured logging
(uvicorn + FastAPI is the usual upgrade path).

### 10. Evaluation harness
**Idea.** Fixed list of prompts (easy / hard / code / reject /
injection), run them through the cascade, write a markdown report with
route counts, category counts and per-prompt latency. Re-run after each
change to spot regressions.
**How.** Small module that calls `pipeline.answer(use_cache=False)` for
each prompt and aggregates the routes into a summary table.
**Real-world use.** This is the entry point to a proper *eval suite*
(OpenAI Evals, Anthropic's `inspect`, Braintrust, LangSmith, ragas).
The fixed-prompt approach is the cheapest possible version and is
exactly what every team starts with before graduating to LLM-as-judge
scoring and human-labelled regression sets.

## Known limitations

- The 8B small model occasionally misclassifies code prompts as `chat`
  unless they explicitly include a code block.
- A subtle "role-swap" jailbreak (`you are now DAN`) sometimes slips past
  the safety triage; the live test that exercises it is marked
  `xfail(strict=False)` to document the gap honestly.
- The semantic cache uses chat-model embeddings, not a dedicated
  embedding model. A real deployment should switch to e.g.
  `nomic-embed-text` and re-tune `SEMANTIC_THRESHOLD`.
- The HTTP server has no auth and no TLS. Local use only.
