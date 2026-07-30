---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "019"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 019: Compute Logical Training Shards through a spawn-safe worker protocol

## What to build

Provide the top-level, spawn-importable worker protocol and worker entry point that attach to Request-Scoped Shared Memory, validate canonical layouts and ownership, compute assigned Logical Training Shards, and communicate only small typed control records through one dedicated duplex pipe.

A successful worker round trip must prove real Windows-compatible `spawn` behavior, non-writeable shared weights, assigned gradient ownership, sanitized failure records, exact state transitions, commit-marker semantics, and correct process exit codes without introducing a global worker pool.

## Acceptance Criteria

- [ ] Every protocol record is a top-level frozen, slotted dataclass with protocol version `1` and only primitive values, stable closed enums, and immutable tuples.
- [ ] The protocol defines the approved startup, ready, compute, result, failure, stop, and stopped records with exact worker identity and required epoch or shard information.
- [ ] Worker-originated records always identify the worker index, and the parent can validate exact record type, version, fields, and assigned shards.
- [ ] Worker failures use closed phase and code enums and never include exception text, traceback data, filesystem paths, shared-memory names, model values, gradients, losses, or other numerical state.
- [ ] The worker follows only `STARTING → READY → COMPUTING → READY → STOPPING → STOPPED` and rejects illegal transitions or a second outstanding compute command.
- [ ] A worker sends ready only after attaching every required shared-memory block and validating minimum capacity, canonical layout, dtype, C-contiguity, and ownership rules.
- [ ] The worker attaches to one flat `float32` weight block and the assigned Logical Training Shard gradient blocks by generated name, while limiting every view to the exact canonical range.
- [ ] Worker weight views are non-writeable, and an attempted worker mutation is rejected without changing parent-visible weights.
- [ ] The worker writes only gradient blocks for its assigned shard IDs and never writes another worker's or an unassigned shard's block.
- [ ] Each assigned shard gradient block is zeroed immediately before that shard is computed.
- [ ] Assigned shards are processed in ascending shard-ID order, and an empty shard produces zero loss and an all-zero gradient block.
- [ ] A worker sends exactly one matching result record only after every assigned shard gradient and loss is complete and finite; that result is the commit marker authorizing the parent to read those buffers.
- [ ] A controlled worker failure sends one sanitized failure record when possible and exits with status `1`.
- [ ] A cooperative stop produces one stopped record, closes attached shared-memory handles in `finally`, and exits with status `0`.
- [ ] Workers close but never unlink shared-memory blocks.
- [ ] Numerical arrays are never serialized through pipes, queues, managers, files, or pickled protocol payloads.
- [ ] Ordinary pytest uses the real `multiprocessing.get_context("spawn")` path to prove importability, pickling, pipe communication, shared-memory attachment, gradient publication, exit codes, and cleanup for a tiny deterministic fixture.
- [ ] Tests cover malformed or wrong-version commands, invalid capacity or layout, illegal state transitions, non-finite shard output, stop behavior, controlled failure privacy, and repeated worker creation without state reuse.

## Testing Expectations

- **Approved test seam:** Top-level protocol records and a real spawned worker entry point exercised with one tiny shared-memory round trip in ordinary pytest.
- **Behavior to verify:** Spawn importability, exact protocol records, state transitions, one outstanding command, shared-memory attachment and validation, read-only weights, assigned gradient ownership, zero-before-compute, result commit markers, sanitized failures, exit codes, and worker-side cleanup.
- **Relevant prior art:** Python's real-spawn process tests required by Phase 5 and the stable canonical layout and shard-math boundaries.
- **Do not test through:** Mock-only process substitutes, private worker loop variables, shared-memory generated names, operating-system scheduling order, or maximum-model endurance.

## Blocked By

- [Ticket 016 — Execute reference-compatible Transformer forward and backward passes](016-execute-reference-compatible-transformer-forward-and-backward-passes.md)

## Constraints and Out of Scope

- Use a local `spawn` context; do not change the application's global multiprocessing start method.
- Use one dedicated duplex pipe per worker; do not use Queue, Manager, socket, file, or global IPC abstractions.
- Do not create a persistent or application-wide worker pool.
- Do not allow workers to mutate shared weights, Adam state, preprocessing, or another shard's gradient block.
- Do not expose exception text or internal resource identifiers in protocol records.
- Do not implement multi-worker supervision, HTTP orchestration, persistence, or route-level deadlines in this ticket.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — approved real-spawn worker seam, protocol records, ownership, and failure requirements.
- `CONTEXT.md` — Request-Scoped Shared Memory, Logical Training Shard, and Request-Scoped Worker Group terminology.
- ADR 0002 — exact worker protocol, state machine, shared-memory validation, commit markers, and exit codes.
- Ticket 016 — reusable Logical Training Shard computation.
- Official Python multiprocessing behavior verified during specification work.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
