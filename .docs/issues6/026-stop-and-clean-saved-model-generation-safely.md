---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "026"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 026: Stop and Clean Saved-Model Generation Safely

## What to build

Harden the complete Saved Transformer Generation Run lifecycle so training and loading safely share one process-local nonblocking Transformer request slot while blocking file and generation work stays away from the FastAPI event-loop thread. The route must remain responsive enough to observe disconnects, cancellation, and one monotonic five-minute generation deadline between token calculations, then drain already-started helper work, discard request-owned state, and release the slot as the final cleanup action.

This ticket completes every lifecycle outcome for the named and latest workflows. It must prevent queues, leaked slot ownership, misleading successful completion after interruption, stale loaded state, and accidental creation of training processes or shared memory.

## Acceptance Criteria

- [ ] `POST /train-transformer` and `POST /load-transformer` reserve the same route-owned, process-local, nonblocking Transformer request slot.
- [ ] Slot ownership is attempted only after standard Pydantic request validation succeeds and begins before training preparation or saved-model selection starts.
- [ ] While either route owns the slot, a second valid training or loading request is rejected immediately with HTTP `429` and exactly `{"detail":"Another Transformer request is already running."}`; no request is queued or waits for acquisition.
- [ ] The shared slot is explicitly process-local and does not claim a machine-wide, cross-process, distributed, or multi-server lock.
- [ ] The training route uses the same approved overlap wording without changing its fresh-weight training behavior, event payload fields, numerical behavior, worker protocol, or persistence lifecycle.
- [ ] File enumeration, metadata checks, file read, JSON decode, strict validation, parameter reconstruction, blocking tokenization, and generation execute away from the FastAPI event-loop thread.
- [ ] Same-process offloading does not create a child process, Request-Scoped Worker Group, pipe, queue, manager, or shared-memory region for Saved Transformer Generation Runs and is not described as multi-core inference.
- [ ] Async orchestration can observe disconnect, cancellation, and deadline state while helper work proceeds, and unrelated lightweight request handling remains responsive under controlled blocking work.
- [ ] Saved-model token generation uses one monotonic five-minute deadline measured for the generation phase.
- [ ] When the deadline expires, an already-started token calculation may finish, but no later token calculation begins; the stream emits `Saved Transformer generation exceeded its time limit.` and emits no `result` or `done`.
- [ ] When a browser disconnect is observed, an already-started token calculation may finish, but no later token calculation begins and no later successful SSE event is emitted.
- [ ] Cancellation follows the same cooperative boundary: active helper work is drained before request-owned numerical state is discarded, and no success event is emitted after stopping is observed.
- [ ] The slot is released after successful named generation, successful latest generation, named-model failure, no-valid-latest failure, prompt error, generation error, deadline, disconnect, and cancellation.
- [ ] Slot release is the final lifecycle action after active helper work is drained and request-owned model, token, sampler, and generation state is discarded.
- [ ] Sequential requests do not share loaded model containers, prompt state, random streams, cancellation flags, selected filenames, or generated tokens.
- [ ] A failed, disconnected, cancelled, or timed-out request cannot leave later valid Transformer requests permanently blocked.
- [ ] No interrupted request emits a false `result` or `done`, and successful requests retain exactly one `loaded`, one `result`, and one `done`.
- [ ] Lifecycle and concurrency behavior remains deterministic under controlled monotonic clocks, disconnect observers, cancellation events, and blocking generation helpers.

## Testing Expectations

- **Approved test seam:** FastAPI `TestClient` route-level HTTP/SSE and lifecycle/concurrency tests using controlled nonblocking slot ownership, fake monotonic clocks, disconnect/cancellation observers, controlled blocking helpers, and the existing exact SSE parser.
- **Behavior to verify:** Bidirectional training/loading exclusion, immediate exact `429`, no queue, off-event-loop responsiveness, five-minute monotonic deadline, between-token cooperative stopping, draining of active work, absence of late success events, all-outcome state disposal, and final slot release.
- **Relevant prior art:** Existing Transformer route run-slot tests, controlled worker-group cleanup tests, monotonic deadline fixtures, cancellation/disconnect seams, exact SSE parsing, and bounded route responsiveness tests in the supplied Python Backend.
- **Do not test through:** Exact private thread identity, exact wall-clock duration, local task names, a particular executor API, internal lock type, polling-loop implementation, or private cleanup helper ordering beyond externally proving that active work is drained before state and slot release.

## Blocked By

- [Ticket 024 — Stream Deterministic Generation from an Exact Saved Model](024-stream-deterministic-generation-from-an-exact-saved-model.md)

## Constraints and Out of Scope

- Do not add a waiting queue, application-wide worker pool, machine-wide lock, cross-process coordination, session state, or retained loaded-model cache.
- Saved-model generation remains in the backend parent process and must not start Transformer training workers.
- Do not change the training algorithm, optimizer, Logical Training Shards, shared-memory design, numerical fixtures, or persistence-before-training-`done` guarantee.
- Do not claim thread offloading provides multi-core acceleration.
- Do not expose cancellation tokens, resource identifiers, internal paths, raw exceptions, or numerical state to clients.
- Do not add a new production dependency.

## Source

- `SPEC.md` — shared slot, overlap response, off-event-loop execution, deadline, disconnect, cancellation, cooperative stopping, cleanup, and no-cache decisions.
- `GRILL_WITH_DOCS_RESULT.md` — confirmed lifecycle risks and safeguards.
- `CONTEXT.md` — canonical Request-Scoped Worker Group and Saved Transformer Generation Run boundaries.
- ADR 0003 — process-local mutual exclusion and stateless inference lifecycle.
- [Ticket 024](024-stream-deterministic-generation-from-an-exact-saved-model.md) — established public load route and generation stream.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
