---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: 03
source_document: SPEC.md
recommended_next_prompt: to-plan-prompt
---

# Ticket 03: Stream complete BPE Tokenization Runs through FastAPI

## What to build

Deliver the complete Phase Migration for the Basic Tokenizer learning demo through a registered `POST /bpe-tokenize` FastAPI endpoint.

For every valid request, the route must accept the shared `ChatRequest`, train a new in-memory Merge Table from the submitted BPE Training Text using the reusable BPE Tokenizer, and stream exactly one `init` event, zero or more learned `merge` events in order, and exactly one final `result` event. The route must use the shared SSE transport while retaining feature-specific payload construction, event sequencing, and production animation behavior.

The complete HTTP and public BPE seams must be covered by self-contained Python tests. No Node, pnpm, Hono, TypeScript runtime, frontend modification, persistence, caching, or future-phase tokenizer functionality is part of this ticket.

## Acceptance Criteria

- [ ] `POST /bpe-tokenize` is registered in the FastAPI application while `GET /health` and `POST /simple-chat` remain available.
- [ ] A valid JSON body containing a non-empty `message` receives HTTP `200` with the `text/event-stream` media type.
- [ ] The SSE response includes `Cache-Control: no-cache` and `X-Accel-Buffering: no`.
- [ ] Every event has a valid `event:` line, a `data:` line containing valid JSON, and a terminating blank line.
- [ ] A successful Tokenization Run emits exactly one `init` event, zero or more `merge` events in learned order, and exactly one `result` event, then completes.
- [ ] The `init` payload has exactly `corpus`, `characters`, `charCount`, and `wordCount`.
- [ ] `characters` contains no more than the first 200 submitted characters while `charCount` reports the complete submitted length.
- [ ] Every `merge` payload has exactly `step`, `pair`, `frequency`, `newToken`, `vocabSize`, and `tokenCount`, with reference-compatible step numbering and learned order.
- [ ] The `result` payload has exactly `inputTokens`, `tokenCount`, `originalCharCount`, and `compressionRatio`.
- [ ] `tokenCount` equals the number of returned `inputTokens`, and `originalCharCount` equals the submitted message length.
- [ ] `compressionRatio` uses the established one-decimal multiplier representation.
- [ ] An empty `message` receives the standard FastAPI/Pydantic HTTP `422` response.
- [ ] Whitespace-only, punctuation-only, and single-character messages remain valid and produce a final result.
- [ ] Submitted text is processed without trimming or normalization.
- [ ] Each request trains a new Merge Table from only its own BPE Training Text, and one request cannot affect a later request.
- [ ] The production stream retains the 800-millisecond initialization delay.
- [ ] Automated tests replace the route's referenced sleep operation with an immediate async substitute and assert event order rather than wall-clock duration.
- [ ] Unexpected failures do not intentionally serialize stack traces, secrets, or environment information into client event payloads.
- [ ] The complete configured pytest suite, Ruff check, and strict mypy check are the required validation path and use only the backend's existing dependency set.

## Testing Expectations

- **Approved test seam:** Exercise `POST /bpe-tokenize`, `POST /simple-chat`, and `GET /health` through FastAPI's `TestClient`; also exercise the public reusable BPE interface for representative failure isolation.
- **Behavior to verify:** Router registration, request validation, headers, media type, SSE framing, valid JSON, exact event order, exact camelCase field sets, initialization truncation, merge numbering, final token consistency, reference-compatible compression formatting, per-request isolation, preserved Simple Chat behavior, and preserved health behavior.
- **Relevant prior art:** The supplied Simple Chat route demonstrates an async stream generator, named SSE events, explicit JSON data, `StreamingResponse`, and the required cache/buffering headers. No existing backend tests were found in the supplied source snapshot.
- **Do not test through:** Private BPE helpers, local variable names, internal generator construction, exact internal data containers, exact wall-clock animation duration, browser rendering, or Vite proxy behavior.

## Blocked By

- [Ticket 01: Preserve Simple Chat through shared request and SSE infrastructure](01-preserve-simple-chat-through-shared-request-and-sse-infrastructure.md)
- [Ticket 02: Provide a deterministic reference-compatible BPE Tokenizer](02-provide-deterministic-reference-compatible-bpe-tokenizer.md)

## User Stories Addressed

- User stories 1–4 and 15–18 — A learner receives initialization, learned BPE Merges, and a consistent final tokenization result.
- User stories 19–30 — The complete HTTP, validation, SSE, payload, camelCase, header, delay, and unmodified-input Frontend Contract is preserved.
- User stories 31–36 — Per-request isolation, router registration, reusable BPE separation, shared request validation, shared SSE transport, and Simple Chat regression protection are delivered.
- User stories 40–44 — Tests bypass real waiting, use Python parity cases, diagnose seam divergence, and preserve existing routes.
- User stories 45–47 — Temporary state is discarded, unexpected failures avoid leaking internal information, and configured quality checks remain authoritative.

## Constraints and Out of Scope

- Use only Python and the dependencies already declared for the backend.
- Preserve the unchanged TypeScript/Vite frontend and its Frontend Contract.
- Keep BPE event construction, sequencing, and delay behavior in the route; keep reusable tokenization in the BPE module; keep common wire formatting in the shared SSE transport.
- Do not add a new SSE error-event design, application-level maximum message length, client-disconnect redesign, frontend change, Node or TypeScript backend dependency, or exact wall-clock test assertion.
- Do not add `train_bpe_on_text()`, custom Pre-token patterns, Word2Vec, transformer, neural-network, persistence, caching, multiprocessing, shared memory, or unrelated cleanup.
- Manual browser or Vite-proxy verification may be useful during implementation but is not a blocker for the approved Python test suite.

## Assumptions and Evidence Limitations

- The exact narrow mechanism used to replace the route-level sleep reference in tests is an implementation choice, provided production behavior remains unchanged.
- Automated backend tests do not prove browser rendering or Vite proxy integration.
- The supplied source snapshot contains no existing backend tests, so the implementation may need to introduce reusable SSE parsing test support.

## Source

- [Phase 2 BPE Tokenizer specification](../../../SPEC.md)

## Recommended Next Step

Run `to-plan-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
