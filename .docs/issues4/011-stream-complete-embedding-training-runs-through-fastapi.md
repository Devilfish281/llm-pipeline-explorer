---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "011"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 011: Stream complete Embedding Training Runs through FastAPI

## What to build

Complete the Phase 4 tracer bullet through `POST /train-embed`. Add the dedicated request contract, register the route without disturbing completed Learning Demos, orchestrate one request-owned deterministic Embedding Training Run through bounded same-process worker-thread intervals, stream the exact Embedding Event Stream over the shared SSE transport, persist the complete Saved Embedding Model successfully, and emit one final `done` event only after persistence completes.

Disconnects, cancellations, numerical failures, result-construction failures, and persistence failures must stop safely without false completion, a new SSE error contract, or client-visible internal details.

## Acceptance Criteria

- [ ] The request body exposes only `words`, `epochs`, `dimensions`, `windowSize`, and `negativeSamples`; unknown extra fields are ignored.
- [ ] `words` is required and accepts one through ten strings, each containing at least one character, without trimming or normalization.
- [ ] `epochs` is a strict integer from 10 through 10,000 and defaults to 10,000.
- [ ] `dimensions` is a strict integer from 4 through 64 and defaults to 32.
- [ ] `windowSize` is a strict integer from 1 through 5 and defaults to 2, using the exact camelCase request name.
- [ ] `negativeSamples` is a strict integer from 1 through 10 and defaults to 5, using the exact camelCase request name.
- [ ] Numeric strings, booleans, fractional numbers, missing required words, zero or eleven words, and empty-string entries produce standard FastAPI/Pydantic HTTP `422` before streaming or training begins.
- [ ] Minimum, maximum, default, and non-divisible epoch reporting cases are covered without making the automated suite wait for production presentation delays or execute unnecessary full-size training.
- [ ] A valid request returns HTTP `200`, `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no` through the existing shared SSE transport.
- [ ] Every SSE `data:` line is valid JSON and every event uses the exact registered name and field set.
- [ ] The first event is exactly one `init` containing only `vocabSize`, `sentenceCount`, `embeddingDim`, `windowSize`, and `totalPairs`.
- [ ] Progress contains approximately fifty `epoch` events, includes epoch zero and the requested final epoch, and each payload contains only `epoch` and six-decimal `loss`.
- [ ] The route requests a `0.02`-second presentation wait after every `epoch` event and no presentation wait after `init` or `done`; tests assert requested values rather than wall-clock duration.
- [ ] A successful stream is exactly `init → epoch × N → done`, with one `done` and no later event.
- [ ] The `done` payload contains exactly `embeddings`, `neighbors`, `similarities`, `analogies`, and `warnings`, with no complete Saved Embedding Model, output weights, paths, or other internal state.
- [ ] Training advances one bounded public reporting interval at a time through same-process thread offloading, returning control to async orchestration between intervals.
- [ ] The route checks the current request's disconnect state between intervals; after a controlled disconnect, no later interval starts, no Saved Embedding Model is persisted, and no `done` or SSE `error` event is emitted.
- [ ] One already-started bounded interval may finish before a disconnect is observed, but the entire remaining Training Run is never offloaded as one uncancellable operation.
- [ ] The complete Saved Embedding Model is persisted through Ticket 008's boundary before `done` is emitted.
- [ ] Serialization, write, replacement, numerical-training, and result-construction failures after streaming begins are logged internally, preserve any prior saved model, and terminate without `done`, an SSE `error` event, exception text, traceback data, paths, or numerical state in the response.
- [ ] Cancellation is not swallowed or transformed into an ordinary training failure; the implementation does not catch `BaseException` broadly.
- [ ] Sequential and controlled concurrent requests keep independent Query Words, hyperparameters, Training Runs, disconnect state, completion payloads, and temporary files.
- [ ] Registering `POST /train-embed` preserves current observable behavior for `GET /health`, `POST /simple-chat`, `POST /bpe-tokenize`, and `POST /neural-net`.
- [ ] Adding the Train Embed request model does not change the existing request models or their validation behavior.
- [ ] Focused tests, the complete pytest suite, Ruff, and strict mypy are run through Poetry, and actual command results are reported honestly during implementation.
- [ ] A manual two-server browser or Vite-proxy check is recorded when practical, without treating backend automated tests as proof of browser rendering.

## Testing Expectations

- **Approved test seam:** The registered FastAPI endpoint exercised through `TestClient` or an equivalent in-process ASGI client, with the real public Word2Vec and persistence boundaries covered by focused integration cases.
- **Behavior to verify:** Exact schema and aliases, HTTP `422` failures, shared SSE headers and framing, exact event order and payload fields, reporting schedule, presentation-wait requests, bounded thread offloading, persistence-before-`done`, disconnect and cancellation behavior, failure privacy, request isolation, registration, and completed-route regressions.
- **Relevant prior art:** Existing Simple Chat, BPE Tokenizer, and Neural Network `TestClient`/SSE tests; the Neural Network route's bounded `asyncio.to_thread()` orchestration, disconnect seam, quiet post-stream failure handling, and persistence-before-completion pattern.
- **Do not test through:** Private route helpers, exact generator class identity, a particular thread-pool object, a particular temporary-file API, exact wall-clock timing, frontend component rendering, CSS, or TypeScript execution as part of the Python test suite.

## Blocked By

- [Ticket 008 — Persist complete Saved Embedding Models safely](008-persist-complete-saved-embedding-models-safely.md)
- [Ticket 010 — Construct exact Embedding Results and Saved Embedding Models](010-construct-exact-embedding-results-and-saved-embedding-models.md)

## Constraints and Out of Scope

- Reuse the shared request and SSE infrastructure and preserve the unchanged TypeScript/Vite Frontend Contract.
- Keep HTTP orchestration separate from reusable Word2Vec mathematics.
- Do not add frontend changes, Transformer training, Node or TypeScript backend runtime code, multiprocessing, process pools, shared memory, external task queues, global training locks, semaphores, rate limits, timeouts, quotas, or a new SSE `error` event.
- Do not add model loading, caching, resumption, fine-tuning, model history, a registry, checkpoints, or a frontend download feature.
- Do not expose output weights or the complete saved model in `done`.
- Do not change dependencies or lockfiles unless current repository inspection proves a new dependency is necessary.

## Source

- `SPEC.md` — request, Frontend Contract, approved FastAPI seam, failure behavior, persistence ordering, regressions, and quality checks.
- `CONTEXT.md` — canonical Embedding Training Run, Embedding Event Stream, Embedding Result, Saved Embedding Model, Python Backend, and TypeScript Reference Implementation terminology.
- ADR 0001 — deterministic compatibility and persistence-before-`done`.
- Tickets 008 and 010, plus the latest Python Backend source export and TypeScript Reference Implementation.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, Tickets 008 and 010, `SPEC.md`, `CONTEXT.md`, ADR 0001, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
