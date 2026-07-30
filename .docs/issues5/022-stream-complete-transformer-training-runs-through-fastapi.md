---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "022"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 022: Stream complete Transformer Training Runs through FastAPI

## What to build

Complete Phase 5 by registering `POST /train-transformer` in the Python Backend and orchestrating one complete Transformer Training Run through the established frontend request and Server-Sent Events contract. Validate before reservation, permit only one active run per FastAPI process, prepare the immutable preprocessing and exact init payload before streaming, then allocate request-scoped workers and shared memory only after init is yielded.

For every reported epoch, coordinate the worker group, apply Ordered Gradient Reduction and Adam, generate one independent deterministic sample, stream the exact epoch payload, and request the established presentation delay. After the final update, evaluate final loss, construct and persist the complete Saved Transformer Model, then emit exactly one done event. Every disconnect, cancellation, failure, timeout, and cleanup path must prevent false persistence or completion and must keep internal details out of client-visible data.

## Acceptance Criteria

- [ ] The registered endpoint is exactly `POST /train-transformer`, and existing Health, Simple Chat, BPE Tokenizer, XOR Neural Network, and Word2Vec endpoints retain their established observable behavior.
- [ ] The public request contains only `epochs`, `temperature`, `topP`, `numLayers`, and `maxTokens`; unknown extra fields are ignored.
- [ ] `epochs` is a strict integer defaulting to `300` and accepts only `50` through `2000` inclusive.
- [ ] `temperature` is a strict finite JSON number defaulting to `0.8` and accepts only `0.1` through `2.0` inclusive.
- [ ] `topP` is a strict finite JSON number defaulting to `0.9` and accepts only `0.1` through `1.0` inclusive.
- [ ] `numLayers` is a strict integer defaulting to `2` and accepts only `1` through `6` inclusive.
- [ ] `maxTokens` is a strict integer defaulting to `40` and accepts only `3` through `500` inclusive.
- [ ] Numeric strings, Booleans, fractional integer values, NaN, infinity, and out-of-range values return standard HTTP `422` before run reservation, preprocessing, streaming, process creation, or shared-memory allocation.
- [ ] After validation, one module-local nonblocking run slot permits one active Transformer Training Run per FastAPI process; an overlapping valid request receives immediate HTTP `429` with no queue and no SSE body.
- [ ] The run slot covers preprocessing retrieval, request-dependent init construction, streaming, workers, shared memory, generation, final evaluation, persistence, and cleanup and is released last on every path.
- [ ] Preprocessing and complete init construction happen before returning the streaming response and before worker or shared-memory allocation.
- [ ] A preprocessing or init-construction failure releases the slot and returns a sanitized HTTP `500` with no SSE framing, worker, pipe, process, or shared-memory allocation.
- [ ] A successful response uses the shared SSE transport and returns HTTP `200`, `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- [ ] The first event is exactly `init`, yielded before shared-memory allocation or worker startup.
- [ ] The init payload contains exactly `vocabSize`, `contextLen`, `embeddingDim`, `numHeads`, `ffDim`, `numLayers`, `totalParams`, `temperature`, `topP`, `corpusSentences`, and `trainingSequences` with no additional or snake_case fields.
- [ ] The route creates fresh weights for every valid run and never reads a saved model, skips training, resumes, fine-tunes, or writes an intermediate checkpoint.
- [ ] The route creates one Request-Scoped Worker Group and exactly five Request-Scoped Shared Memory blocks only for the owning request.
- [ ] Epochs advance inclusively from zero through the exact requested final epoch, with progress at `max(1, floor(epochs / 50))`, epoch zero, every report boundary, and the exact final epoch.
- [ ] Each complete epoch waits for committed results from all four Logical Training Shards, applies Ordered Gradient Reduction and one parent-side Adam update, and checks finite state before later work.
- [ ] Each reported epoch creates one Generated Text Sample from an independent `(42 + epoch) modulo 2^32` Sample Random Stream.
- [ ] Each `epoch` SSE payload contains exactly `epoch`, six-decimal `loss`, and `sample`.
- [ ] The route requests approximately `0.02` seconds of presentation delay exactly once after each epoch event and never after init or done; tests replace the delay seam rather than assert wall-clock time.
- [ ] The collected sample list preserves report order as objects containing the report epoch and generated text.
- [ ] After the final Adam update, final loss is recomputed from final weights through the bounded cooperative evaluation boundary rather than copied from a pre-update epoch loss.
- [ ] The complete final Saved Transformer Model is constructed and persisted to the exact configuration-specific destination before done is formatted or yielded.
- [ ] The single `done` payload contains exactly `architecture`, six-decimal `finalLoss`, and the complete ordered `samples` collection, and the stream emits no later event.
- [ ] The architecture text preserves the established decoder-only layer, embedding, head, and feed-forward description without a cached or resumed label.
- [ ] Worker-pipe and sentinel waits use bounded polling away from the event-loop thread, and browser disconnection is checked after every process poll and before each later blocking stage.
- [ ] Generated Text Sample creation and final evaluation run through bounded thread offloading with one request-owned cooperative cancellation event; the route waits for an active helper thread to return before releasing numerical memory it may access.
- [ ] Startup uses the 30-second group deadline, while each complete epoch, each sample, and final evaluation uses its approved five-minute deadline.
- [ ] A disconnect, worker failure, protocol failure, timeout, forced termination, non-finite state, generation failure, final-evaluation failure, serialization failure, write failure, or replacement failure after init terminates quietly with no done, no new SSE error event, no false persistence, and no client-visible exception text, traceback, path, shared-memory name, protocol detail, or numerical state.
- [ ] `asyncio.CancelledError` is not converted into an ordinary training failure; required cleanup completes and cancellation propagates.
- [ ] Cleanup attempts every stage even after an earlier cleanup failure, preserves the original outcome, logs secondary cleanup failures separately, releases shared memory only after no worker remains alive, and releases the process-local run slot last.
- [ ] A forced process termination always marks the run failed and prevents persistence and done.
- [ ] Sequential valid requests use fresh weights, optimizer state, worker processes, pipes, shared memory, cancellation state, samples, and temporary files while reusing only the immutable preprocessing snapshot.
- [ ] Focused endpoint tests use controlled seams for expensive work, clocks, delays, disconnects, failures, and persistence, while at least one minimum bounded integration case reaches the real public Transformer boundaries without maximum training.
- [ ] The ordinary implementation validation runs the focused tests, complete pytest suite, Ruff, and strict mypy through Poetry and reports actual results honestly; a practical two-server browser or Vite-proxy check is recorded separately when possible.

## Testing Expectations

- **Approved test seam:** The registered FastAPI endpoint exercised through `TestClient` or an equivalent in-process ASGI client, with narrow seams for worker orchestration, clocks, disconnect checks, presentation delay, generation/evaluation deadlines, and persistence.
- **Behavior to verify:** Exact strict request validation, run reservation and overlap, pre-stream failures, SSE headers and framing, exact init/epoch/done field sets and order, report schedule, sampling, delay requests, final-loss order, persistence-before-done, process deadlines, disconnect and cancellation, failure privacy, exhaustive cleanup, slot release, request isolation, registration, and completed-route regressions.
- **Relevant prior art:** Existing Simple Chat, BPE, XOR, and Word2Vec `TestClient`/SSE patterns; completed atomic persistence tests; Tickets 017–021 public numerical and process boundaries.
- **Do not test through:** Private route helper names, exact generator class identity, a particular monkeypatch mechanism, generated process/shared-memory names, exact wall-clock timing, frontend component rendering, CSS, TypeScript execution in the Python test suite, or maximum-configuration endurance in ordinary pytest.

## Blocked By

- [Ticket 017 — Advance Transformer epochs with Ordered Gradient Reduction and Adam](017-advance-transformer-epochs-with-ordered-reduction-and-adam.md)
- [Ticket 018 — Generate deterministic text and construct Saved Transformer Models](018-generate-deterministic-text-and-construct-saved-transformer-models.md)
- [Ticket 020 — Coordinate Request-Scoped Worker Groups and cleanup](020-coordinate-request-scoped-worker-groups-and-cleanup.md)
- [Ticket 021 — Persist configuration-specific Saved Transformer Models safely](021-persist-configuration-specific-saved-transformer-models-safely.md)

## Constraints and Out of Scope

- Preserve the unchanged TypeScript/Vite Frontend Contract and reuse the shared SSE formatter and response factory.
- Keep HTTP orchestration separate from reusable Transformer mathematics and worker implementation.
- Do not add frontend changes, a TypeScript or Node backend runtime, a machine-learning framework, a hosted AI service, a task queue, a persistent worker pool, dynamic logical sharding, or cross-request shared memory.
- Do not add request controls for corpus, seed, optimizer, dimensions, heads, feed-forward size, context, sequence length, worker count, shard count, timeouts, save path, checkpoints, or concurrency policy.
- Do not load or resume saved models, write intermediate checkpoints, add model history or registry behavior, or expose a model-download feature.
- Do not add a new SSE `error` event, expose internal exceptions or numerical state, or replace post-stream failures with misleading HTTP status claims.
- Do not queue overlapping requests or introduce a machine-wide or cross-process training lock; the approved run slot is process-local.
- Do not use exact elapsed-time assertions for presentation delays or process deadlines.
- Maximum epochs/layers endurance remains slow/manual and is not required in ordinary pytest.
- No new dependency or lockfile change is expected unless live implementation inspection proves the supplied project configuration has changed.

## Source

- `SPEC.md` — complete Phase 5 request, SSE, training, lifecycle, persistence, failure, and testing contract.
- `CONTEXT.md` — Transformer Training Run, Transformer Event Stream, Transformer Epoch Update, Generated Text Sample, Saved Transformer Model, Request-Scoped Worker Group, and Request-Scoped Shared Memory terminology.
- ADR 0002 — binding stable Python process and lifecycle architecture.
- Tickets 017, 018, 020, and 021 — parent training, completion, worker-group, and persistence boundaries.
- Latest Python Backend source export — current application registration, shared schema/SSE conventions, and completed route regressions.
- Latest TypeScript Reference Implementation — exact frontend request, eleven-field init payload, event discrimination, architecture text, and browser behavior evidence.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
