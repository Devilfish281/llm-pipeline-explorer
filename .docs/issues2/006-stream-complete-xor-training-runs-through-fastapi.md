---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: 006
source_document: SPEC.md
recommended_next_prompt: to-plan-prompt
---

# Ticket 006: Stream complete XOR Training Runs through FastAPI

## What to build

Complete the Phase 3 vertical slice by exposing and registering `POST /neural-net` through the Python Backend while preserving the unchanged TypeScript/Vite frontend’s Frontend Contract.

The endpoint must validate one requested model mode and epoch count, create one independent Training Run, advance CPU work in bounded same-process worker-thread intervals, stream reference-compatible Epoch Updates, and cooperatively check whether the browser disconnected between intervals.

A successful request must persist the mode-specific Saved Weight Snapshot before emitting exactly one final `done` event. A disconnected or failed request must stop without replacing the previous snapshot, must emit no `done` event, must not invent an SSE `error` event, and must not expose internal exception details.

The route must reuse the existing shared SSE formatting and response behavior and must preserve the completed Health, Simple Chat, and BPE contracts.

## Acceptance Criteria

- [ ] A dedicated `NeuralNetRequest` requires `mode` to be `single-layer` or `multi-layer`.
- [ ] `epochs` is optional, defaults to `5000`, must be an integer, and accepts only values from `100` through `100000`, inclusive.
- [ ] Missing or unknown modes, non-integer epochs, `99`, and `100001` return standard FastAPI/Pydantic HTTP `422` responses rather than SSE events.
- [ ] No seed, learning rate, hidden size, activation, optimizer, output path, persistence switch, or dtype field is added to the request.
- [ ] The FastAPI application registers `POST /neural-net` while retaining `GET /health`, `POST /simple-chat`, and `POST /bpe-tokenize`.
- [ ] A valid request returns HTTP `200` with `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- [ ] The route reuses the existing shared SSE formatter and response factory.
- [ ] Each request creates an independent production Weight Initialization and independent Training Run state.
- [ ] CPU-bound work advances in bounded same-process worker-thread intervals and returns control to the async route between Epoch Updates.
- [ ] The implementation does not create a process pool, worker process, shared-memory region, external task queue, unbounded executor, or global thread-pool capacity change.
- [ ] Every progress event is named `epoch` and contains exactly `epoch` and `loss`.
- [ ] Epoch Updates include epoch zero, all reference-compatible reporting boundaries, and the final requested epoch.
- [ ] After each `epoch` event, production requests a `0.02`-second presentation delay; no delay is requested after `done`.
- [ ] Tests replace only the neural-network route’s referenced delay operation and verify requested delays rather than elapsed wall-clock time.
- [ ] A successful stream contains exactly `epoch × N → done` and no event after `done`.
- [ ] The `done` payload contains exactly `architecture`, `predictions`, and `verdict`.
- [ ] The `done` payload contains four XOR Predictions in the required truth-table order and never contains weights.
- [ ] Exact architecture labels, rounded values, and Training Verdict strings match the public numerical result.
- [ ] The correct mode-specific Saved Weight Snapshot is successfully replaced before `done` is emitted.
- [ ] A request containing only `{"mode":"single-layer"}` behaves as a `5000`-epoch Training Run.
- [ ] A controlled disconnect after a selected Epoch Update allows at most the already-started interval to finish, starts no later interval, writes or replaces no snapshot, emits no `done`, and emits no error event.
- [ ] A controlled training failure after one or more Epoch Updates terminates the stream without `done`, without an SSE error event, and without serialized exception text, stack traces, or filesystem paths.
- [ ] A controlled persistence failure after training preserves the prior destination, cleans temporary files, and terminates without `done` or leaked internal details.
- [ ] Unexpected post-stream exceptions are logged through the existing standard Python logging path.
- [ ] Two controlled requests do not share mutable weights, epoch state, predictions, disconnect state, or temporary destination files.
- [ ] Existing Health, Simple Chat, and BPE tests remain green after router registration.
- [ ] The focused neural-network tests, complete pytest suite, Ruff check, and strict mypy check are run and their actual results are reported honestly.

## Testing Expectations

- **Approved test seam 1:** Exercise `POST /neural-net` through FastAPI’s `TestClient` or an equivalent in-process ASGI client.
- **Approved test seam 2:** Exercise disconnect, training-failure, and persistence-failure behavior through narrow injected or patched boundaries owned by the neural-network route.
- **Behavior to verify:** Request validation, router registration, response headers, shared SSE framing, exact event names and field sets, event order, default epochs, reporting boundaries, delay requests, worker-thread interval orchestration, final payload, persistence-before-completion ordering, disconnect behavior, failure behavior, request isolation, and regressions.
- **Relevant prior art:** Existing Simple Chat and BPE route tests parse named SSE events through FastAPI’s in-process client, patch route-level sleep references, assert exact payload keys and headers, and retain health-route regression coverage.
- **Required deterministic controls:** Use the verified numerical seed or controlled public Training Run state from Ticket 004, temporary snapshot directories from Ticket 005, a fake disconnect boundary, injected training and save failures, and patched production delays.
- **Do not test through:** Exact private helper names, local variables, a particular executor instance, a particular state-container class, internal loop syntax, exact wall-clock duration, or a real network interruption.

## Blocked By

- [Ticket 004: Provide reference-compatible XOR Training Runs](004-provide-reference-compatible-xor-training-runs.md)
- [Ticket 005: Persist completed XOR Training Runs safely](005-persist-completed-xor-training-runs-safely.md)

## User Stories Addressed

- User stories 16–28 — The unchanged frontend can call the validated endpoint and receive the exact Neural Network Event Stream and completion contract.
- User story 29 — Separate requests receive independent Weight Initializations and state.
- User stories 37–41 — Shared SSE behavior is reused, work is thread-offloaded inside the process, disconnects are testable, and production delays are removed from tests.
- User story 44 — Existing Simple Chat and BPE tests remain green.
- User stories 45–49 — Existing routes remain available, abandoned work stops cooperatively, failed work does not replace snapshots or emit false completion, and internal details remain private.
- User stories 50–51 — Successful persistence ordering is honored and the full Poetry quality path remains required.

## Constraints and Out of Scope

- Preserve the unchanged TypeScript/Vite frontend and all exact serialized field names, event names, event ordering, labels, verdicts, and validation behavior.
- Keep reusable XOR mathematics in the public numerical module; keep request orchestration, streaming, disconnect checks, presentation delays, persistence integration, and completion behavior at the route boundary.
- Use only the existing Python backend dependencies and same-process thread-offloading facilities.
- Do not add frontend changes, Node/Hono runtime code, runtime TypeScript execution, multiprocessing, process pools, shared memory, external task queues, global thread-pool configuration, or forceful thread termination.
- Do not add a new SSE `error` event, client-visible exception details, Saved Weight Snapshot data in `done`, saved-weight loading, checkpoint history, model registries, general matrix abstractions, new optimizers, Word2Vec, or transformer work.
- Do not require exact wall-clock timing tests or claim automated Python tests prove browser rendering or Vite proxy behavior.

## Assumptions and Evidence Limitations

- The exact bounded state-machine representation and thread-offloading call are implementation choices provided the async route regains control between reports and tests observe the required behavior.
- Cooperative cancellation may finish one interval already executing before the disconnect is observed.
- Random production Multi-Layer Mode runs may legitimately return the confirmed failure verdict.
- Automated backend tests do not prove browser graph rendering or Vite proxy integration; a manual end-to-end check is recommended after the Python quality checks pass.

## Source

- [Phase 3 XOR Neural Network specification](../SPEC.md)
- [Canonical project context](../CONTEXT.md)
- [Ticket 004](004-provide-reference-compatible-xor-training-runs.md)
- [Ticket 005](005-persist-completed-xor-training-runs-safely.md)
- Latest complete Python backend source snapshot supplied with the specification.

## Recommended Next Step

Run `to-plan-prompt` in a fresh conversation using this ticket, Tickets 004 and 005, the source specification, the canonical context, and the latest complete backend source snapshot.
