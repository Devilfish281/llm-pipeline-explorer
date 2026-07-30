---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "020"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 020: Coordinate Request-Scoped Worker Groups and cleanup

## What to build

Provide the parent-side Request-Scoped Worker Group that creates one through four spawned workers and exactly five shared-memory blocks for one Transformer Training Run, assigns the four fixed Logical Training Shards deterministically, validates every worker signal, returns committed per-shard results, and releases every request-owned resource after success, failure, timeout, cancellation, or forced termination.

Physical worker count may affect performance only. Logical shard boundaries, gradient contents, reduction inputs, and later public results must remain unchanged across one through four workers and across different worker completion orders.

## Acceptance Criteria

- [ ] Actual worker count is calculated once as `min(4, max(1, os.cpu_count() or 1))`, with controlled tests for missing, zero, one, two, four, and larger reported CPU counts.
- [ ] Exactly four fixed Logical Training Shards are assigned statically by `shard_id % actual_worker_count`, and each worker receives its assigned shard IDs in ascending order.
- [ ] Every run creates exactly one shared `float32` weight block and four separate shared `float32` gradient blocks sized from the canonical layout.
- [ ] Adam moments, reduction workspaces, generated samples, and route state remain parent-local and are never placed in shared memory.
- [ ] Each worker receives one dedicated duplex pipe, and the parent closes its local copy of the worker endpoint immediately after a successful process start.
- [ ] The complete group must become ready within the approved 30-second monotonic deadline; a missing, failed, malformed, exited, or late worker fails startup.
- [ ] The parent issues at most one compute command per worker for an epoch and prohibits epoch pipelining.
- [ ] The parent waits on worker pipe endpoints and process sentinels through bounded `multiprocessing.connection.wait()` polls executed away from the async event-loop thread.
- [ ] Each poll uses the approved `0.1`-second interval and exposes a boundary for route-level disconnect observation.
- [ ] A complete four-shard epoch must finish within the approved five-minute monotonic deadline.
- [ ] The parent rejects wrong-version, stale, duplicate, missing, malformed, wrong-worker, unassigned-shard, or non-finite results before exposing any shard buffer for reduction.
- [ ] A matching result record is required as the commit marker before the parent reads each worker's assigned gradient blocks.
- [ ] Worker completion order does not change the ordered collection of shard losses and gradients returned to the parent.
- [ ] One-through-four real-worker configurations produce equivalent shard results for the same tiny deterministic snapshot within approved numerical tolerances.
- [ ] A cooperative shutdown sends stop, waits up to two seconds for stopped records and exit, and treats clean exits as status zero integrity evidence.
- [ ] Surviving workers are terminated, waited for up to two more seconds, then killed when still alive; any forced termination fails the run and prevents a successful completion outcome.
- [ ] Cleanup is non-short-circuiting: cancellation signaling, cooperative stop, waits, terminate/kill escalation, joins, exit-code recording, pipe closure, process-object closure, view release, shared-memory close, and each unlink are attempted even after an earlier cleanup failure.
- [ ] The parent is the sole unlinking owner and waits until no worker remains alive before releasing shared numerical memory.
- [ ] The original run outcome is preserved while secondary cleanup failures are logged or returned separately for internal handling.
- [ ] Repeated sequential and controlled concurrent group creation uses fresh process, pipe, shared-memory, cancellation, and state objects with no request-scoped resource reuse.
- [ ] Ordinary pytest covers success, startup failure, worker failure, protocol corruption, startup timeout, epoch timeout, cooperative stop, terminate escalation, kill escalation, cleanup failures, and leak checks using tiny bounded fixtures.

## Testing Expectations

- **Approved test seam:** The parent-side Request-Scoped Worker Group exercised with real spawned processes, pipes, sentinels, and shared memory plus narrow deterministic clock and failure seams.
- **Behavior to verify:** Worker-count calculation, static shard assignment, five-block ownership, startup and epoch deadlines, bounded waits, commit validation, worker-count-independent results, staged shutdown, non-short-circuiting cleanup, parent-only unlinking, and request isolation.
- **Relevant prior art:** Ticket 019's real worker round trip and the specification's ordinary-pytest lifecycle requirements.
- **Do not test through:** Private supervisor helper names, generated process or shared-memory names, exact operating-system scheduling, live browser disconnection, route lock ownership, or maximum-configuration endurance.

## Blocked By

- [Ticket 019 — Compute Logical Training Shards through a spawn-safe worker protocol](019-compute-logical-shards-through-a-spawn-safe-worker-protocol.md)

## Constraints and Out of Scope

- Use request-scoped non-daemonic processes from a local `spawn` context; do not create a global pool or alter the global start method.
- Keep exactly four logical shards regardless of physical worker count.
- Do not reduce gradients in worker completion order or allow workers to update weights.
- Do not use Queue, Manager, shared optimizer buffers, cross-request shared memory, or process reuse.
- Do not own or release the FastAPI process-local run slot; route orchestration releases that slot after group cleanup.
- Do not implement request validation, SSE events, Generated Text Samples, final persistence, or frontend behavior in this ticket.
- Maximum model and epoch endurance remains separately marked slow/manual work, not ordinary pytest.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — worker-count, shared-memory, protocol supervision, deadlines, cleanup, and real-spawn testing decisions.
- `CONTEXT.md` — Request-Scoped Worker Group, Request-Scoped Shared Memory, Logical Training Shard, and Ordered Gradient Reduction terminology.
- ADR 0002 — parent polling, static assignment, staged shutdown, and parent-only ownership contract.
- Ticket 019 — spawn-safe worker protocol and entry point.
- Official Python multiprocessing and shared-memory behavior verified during specification work.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
