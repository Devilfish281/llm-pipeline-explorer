---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: 01
source_document: SPEC.md
recommended_next_prompt: to-plan-prompt
---

# Ticket 01: Preserve Simple Chat through shared request and SSE infrastructure

## What to build

Create the shared request-validation and Server-Sent Events transport seams required by the Python Backend, then move Simple Chat onto those seams without changing any behavior visible through its HTTP interface.

The shared request contract must continue to accept a JSON `message` string containing at least one character. It must not trim or normalize the submitted value, reject whitespace-only content, or add a maximum length.

The shared SSE transport must produce the established named-event wire format and standard streaming-response configuration. Simple Chat must continue to emit its established `start`, `word`, and `done` sequence with the same payloads, animation behavior, media type, cache behavior, and proxy-buffering behavior.

## Acceptance Criteria

- [ ] One shared `ChatRequest` model requires `message` to be a string with a minimum length of one.
- [ ] The shared request model does not trim, normalize, reject whitespace-only input, or impose a maximum message length.
- [ ] One shared SSE formatter emits an `event:` line, a `data:` line containing valid JSON, and a terminating blank line for each event.
- [ ] One shared SSE response configuration uses the `text/event-stream` media type and includes `Cache-Control: no-cache` and `X-Accel-Buffering: no`.
- [ ] `POST /simple-chat` uses the shared request and SSE seams.
- [ ] A valid Simple Chat request still emits exactly `start`, zero or more `word` events as applicable to the existing response, and `done` in the established order.
- [ ] Simple Chat event payloads and production animation delays remain unchanged.
- [ ] An empty Simple Chat `message` still receives the standard FastAPI/Pydantic HTTP `422` response.
- [ ] A whitespace-only Simple Chat `message` remains valid.
- [ ] Existing `GET /health` behavior remains unchanged.

## Testing Expectations

- **Approved test seam:** Exercise `POST /simple-chat` and `GET /health` through FastAPI's `TestClient`.
- **Behavior to verify:** Shared request validation, exact Simple Chat event order and payloads, valid SSE framing, media type, cache and buffering headers, empty-message rejection, whitespace-only acceptance, and unchanged health behavior.
- **Relevant prior art:** The supplied Simple Chat route already contains the established async stream generator, named SSE events, JSON event data, response media type, and cache/buffering headers.
- **Do not test through:** Private helper identity, local variable names, exact internal response-generator structure, or exact wall-clock timing.

## Blocked By

- None — can start immediately.

## User Stories Addressed

- User story 34 — One shared `ChatRequest` validation model is used by Simple Chat and BPE.
- User story 35 — One shared SSE wire-format and response configuration is used by streaming routes.
- User story 36 — Simple Chat preserves its existing request, events, payloads, headers, and response behavior.
- User story 44 — `GET /health` and `POST /simple-chat` remain available.
- User story 47 — The configured pytest, Ruff, and mypy checks remain the backend validation path.

## Constraints and Out of Scope

- Preserve the established Simple Chat Frontend Contract.
- Keep feature-specific event order and delays in the Simple Chat route's stream generator.
- Do not add BPE behavior in this ticket.
- Do not add frontend changes, new dependencies, future-phase abstractions, persistence, caching, or multiprocessing.
- Do not introduce a new application-specific SSE error-event contract.

## Assumptions and Evidence Limitations

- No existing backend tests were visible in the supplied source snapshot, so regression coverage may need to establish the first test pattern for Simple Chat.
- The exact internal organization of shared transport helpers is an implementation choice as long as the observable contract remains unchanged.

## Source

- [Phase 2 BPE Tokenizer specification](../../../SPEC.md)

## Recommended Next Step

Run `to-plan-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
