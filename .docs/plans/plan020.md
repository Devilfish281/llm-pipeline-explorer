---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "020"
source_work_item: 020-coordinate-request-scoped-worker-groups-and-cleanup.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(56).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 020: Coordinate Request-Scoped Worker Groups and cleanup

## Initial checklist

- Confirm Ticket 020 is the only selected work item and that its Ticket 019 blocker is represented by the completed spawn-safe protocol, top-level worker entry point, shared-memory attachment boundary, commit-marker publication, cooperative stop behavior, and real-spawn tests in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(56).md` as the source of truth for current Python code, tests, dependencies, typing style, and established Transformer seams; older exports, prior plans, and historical snippets must not override it.
- Use Ticket 020 as the immediate acceptance authority and use `SPEC.md`, `CONTEXT.md`, and ADR 0002 for the durable Request-Scoped Worker Group, Request-Scoped Shared Memory, Logical Training Shard, Ordered Gradient Reduction, deadline, and cleanup decisions.
- Preserve Ticket 019's protocol records, validators, worker process target, and worker-only close behavior rather than creating a second protocol or process entry point.
- Limit production work to parent-side worker-group supervision in `src/how_llms_work/ml/transformer_worker.py`; do not implement FastAPI request validation, route reservation, SSE, Generated Text Samples, final evaluation, persistence, or frontend behavior.
- Add a focused `tests/test_transformer_worker_group.py` suite at the approved public group seam, using real local-`spawn` processes, dedicated pipes, process sentinels, and five real shared-memory blocks plus narrow deterministic clock and failure seams.
- Re-establish the implementation-session baseline before editing, then finish with focused group tests, existing worker and Transformer regressions, formatting checks, the complete pytest suite, Ruff, strict mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit correction: Ticket 020 is now supplied, the high-level goal remains conversion of the approved TypeScript behavior to Python, and the latest complete Python Backend export is current-code truth.
2. `020-coordinate-request-scoped-worker-groups-and-cleanup.md` for immediate scope, all acceptance criteria, the approved real-process test seam, blocker, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(56).md` for the current implementation, tests, fixtures, dependencies, paths, public symbols, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for durable Phase 5 decisions and the canonical terms Request-Scoped Worker Group, Request-Scoped Shared Memory, Logical Training Shard, and Ordered Gradient Reduction.
5. The completed Ticket 019 worker protocol and current `tests/test_transformer_worker.py` as the blocker evidence and reusable process boundary.
6. The current `TransformerTrainingRun`, canonical parameter layout, `LogicalTrainingShardResult`, and `calculate_logical_training_shard()` boundaries in `src/how_llms_work/ml/transformer.py`; Ticket 020 must consume these boundaries rather than duplicate Transformer mathematics or Adam.
7. The latest `llm_works_file_structure.md`, especially the TypeScript worker and training orchestrator, as limited behavior evidence for shared weight/gradient storage and worker completion signaling only. Host-dependent partitioning and weak lifecycle behavior are not authoritative.
8. Official Python 3.12 documentation for local multiprocessing contexts, spawned processes, duplex pipes, process sentinels, `multiprocessing.connection.wait()`, process termination/kill/close rules, `SharedMemory.close()`/`unlink()`, and `asyncio.to_thread()` as technical cross-checks.
9. Older Python exports, earlier plans, prior assumptions, generated process names, and incidental operating-system scheduling are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 020 adds the parent-side process supervisor that Ticket 019 intentionally deferred.

For one Transformer Training Run, the new Request-Scoped Worker Group must:

- calculate one actual worker count from the host CPU count, bounded to one through four;
- keep exactly four fixed Logical Training Shards and assign them statically by shard ID modulo worker count;
- create exactly five parent-owned shared-memory blocks: one canonical `float32` weight block and four canonical `float32` shard-gradient blocks;
- create one non-daemonic spawned process and one dedicated duplex pipe per actual worker;
- supervise startup through pipe endpoints and process sentinels under one 30-second monotonic group deadline;
- issue no more than one compute command to each worker for an epoch and prohibit overlapping epochs;
- poll worker endpoints and sentinels in bounded `0.1`-second `multiprocessing.connection.wait()` calls off the async event-loop thread;
- expose one generic poll-observation boundary that later route orchestration can use for disconnect and cancellation checks without importing FastAPI into the worker module;
- supervise one complete four-shard epoch under a five-minute monotonic deadline;
- treat validated matching result records as commit markers before copying any shard-gradient buffer into independent parent-owned results;
- return exactly four `LogicalTrainingShardResult` values in canonical shard order regardless of worker completion order;
- shut workers down cooperatively first, then terminate and kill survivors under the approved two-second stages;
- attempt every cleanup operation independently, preserve the primary run outcome, and report secondary cleanup failures separately;
- close and unlink parent-owned shared memory only after no worker remains alive; and
- create fresh process, pipe, cancellation, state, and shared-memory objects for every sequential or concurrent group.

This ticket does not apply Adam, emit an epoch event, generate text, evaluate final loss, persist a model, manage the FastAPI run slot, or release that route-owned slot.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, mypy, Black, process-spawn, shared-memory, or timing command was executed while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and report the actual results honestly.

## Current code observations from the latest source

- `src/how_llms_work/ml/transformer_worker.py` already owns protocol version `1`, closed worker/failure enums, seven top-level frozen slotted protocol records, strict parent/worker validators, shared-memory attachment and layout validation, and the top-level spawn-importable `run_transformer_worker()` process target.
- Ticket 019's worker attaches to one parent-created weight block and only its assigned gradient blocks, exposes non-writeable weight views, zeros each assigned gradient before computation, computes assigned shards in ascending order, sends one finite matching `ResultMessage` only after complete publication, sends sanitized failures, handles cooperative `StopMessage → StoppedMessage`, closes worker-side handles, never unlinks, and exits with the approved status.
- `tests/test_transformer_worker.py` already proves the protocol shape and validators plus real local-`spawn` happy paths, malformed startup and command handling, non-finite failure, no-pipelining behavior, cooperative stop, serial reuse, parent-only unlink ownership, and repeated worker creation.
- `src/how_llms_work/ml/transformer.py` already owns the canonical parameter layout and views, exactly four Logical Training Shards, `LogicalTrainingShardResult`, direct shard calculation, `TransformerTrainingRun`, Ordered Gradient Reduction, parent-local Adam moments and workspaces, and deterministic epoch advancement.
- `TransformerTrainingRun` currently copies initialized parameters into its own parent-local canonical `float32` storage. Ticket 020 should not redesign that class. The smallest integration seam is for the group to accept the current parent weight storage for each epoch, validate it, and copy it into the request-owned shared weight block before dispatch.
- `src/how_llms_work/routes/train_transformer.py` is still empty. It must remain unchanged in this ticket because HTTP/SSE orchestration and run-slot ownership belong to later route work.
- No current source defines a Request-Scoped Worker Group, actual worker-count calculation, static multi-worker assignment, five-block parent owner, group startup supervisor, async bounded wait loop, epoch supervisor, cleanup report, staged termination coordinator, or request-isolation tests.
- `pyproject.toml` already provides Python 3.12, NumPy, pytest, pytest-asyncio, Ruff, mypy, and Black. Standard-library multiprocessing and shared memory require no new dependency or lockfile change.

## Acceptance criteria coverage

No Ticket 020 acceptance criterion is fully satisfied because the parent-side group does not yet exist. Ticket 019 provides reusable worker-side foundations for several criteria, but each still needs parent supervision and group-level evidence.

### Already satisfied and evidenced

- **Blocker only:** Ticket 019's exact worker protocol, spawn-safe entry point, assigned-only gradient ownership, result commit marker, cooperative stopped record, clean worker exit status, worker-side close-only cleanup, and parent-only unlink contract are present and covered by current tests.
- **Reusable mathematical boundary only:** Four fixed Logical Training Shards, canonical parameter sizing, direct shard calculation, and parent-owned Ordered Gradient Reduction/Adam are present in `transformer.py`.

### Behavior present but evidence incomplete

- None at the complete Ticket 020 group seam.

### Partially implemented

- **AC 3:** Canonical `float32` size and worker attachment validation exist; parent creation of exactly five blocks does not.
- **AC 5:** Ticket 019 real-spawn tests create one duplex pipe and close the duplicate child endpoint; multi-worker group ownership and immediate close-on-success are absent.
- **AC 7:** The worker rejects pipelined commands; the parent does not yet enforce one group epoch at a time or one command per worker.
- **AC 11–12:** Strict result validators and worker commit messages exist; the parent does not yet supervise all workers, reject cross-worker/group corruption, or defer all buffer reads until every expected commit is validated.
- **AC 15:** Worker cooperative stop/stopped and exit status zero exist; the group-wide two-second cooperative shutdown stage does not.
- **AC 18:** Worker-side no-unlink behavior exists; the parent group does not yet own and sequence all five unlinks after every process is dead.

### Not implemented

- **AC 1:** One-time actual worker-count calculation and controlled CPU-count coverage.
- **AC 2:** Multi-worker static modulo assignment and ascending per-worker shard IDs.
- **AC 3–5:** Request-owned five-block allocation, parent-local non-shared state evidence, and dedicated group pipe/process construction.
- **AC 6:** Complete-group 30-second startup deadline and failure handling.
- **AC 7–10:** Parent epoch-state enforcement, off-event-loop bounded waits, `0.1`-second poll observation boundary, and five-minute epoch deadline.
- **AC 11–14:** Complete group result-set validation, commit-gated gradient materialization, completion-order independence, and one-through-four-worker numerical equivalence.
- **AC 15–19:** Cooperative shutdown, terminate/kill escalation, non-short-circuiting cleanup, parent-only final release, and primary-outcome preservation with separate cleanup diagnostics.
- **AC 20:** Fresh sequential and controlled concurrent group resources.
- **AC 21:** Ordinary-pytest coverage for all success, failure, timeout, escalation, cleanup, and leak scenarios.

### Evidence limitations

- The current code was inspected through the user's complete source export rather than a live repository checkout. `implement-prompt` must inspect the live files and signatures again before editing.
- No process group was created and no shared-memory block was allocated during planning.
- Exact operating-system timing and scheduling are intentionally not acceptance evidence; deadline tests require deterministic clocks and bounded real-process fixtures.
- On Windows, shared-memory deletion is tied to closing all handles even though the parent must still call its sole-owner `unlink()` operation. Tests must verify the ownership contract and absence of live handles without assuming POSIX-only name behavior.
- Deterministic kill-escalation coverage may require a narrow process-control seam around a real spawned child because Windows termination semantics do not reliably provide a naturally surviving process after `terminate()`.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer_worker.py` — current protocol records, validators, `_AttachedWorkerResources`, `run_transformer_worker()`, module exports, worker-side cleanup, and destination for the parent-side Request-Scoped Worker Group.
2. `src/how_llms_work/ml/transformer.py` — `LOGICAL_TRAINING_SHARD_COUNT`, `TransformerParameterLayout`, `InitializedTransformerParameters`, `TransformerTrainingSequence`, `LogicalTrainingShard`, `LogicalTrainingShardResult`, `TransformerGradientBuffer`, `build_logical_training_shards()`, `build_transformer_parameter_layout()`, `build_transformer_parameter_views()`, `create_transformer_gradient_buffer()`, `calculate_logical_training_shard()`, and `TransformerTrainingRun`.
3. `tests/test_transformer_worker.py` — existing protocol fixtures, spawn context use, parent-created shared-memory helpers, subprocess cleanup patterns, and Ticket 019 regression authority.
4. `tests/test_transformer_training.py` — parent result validation, canonical reduction order, completion-order independence, finite-state handling, and direct `TransformerTrainingRun.advance_epoch()` prior art.
5. `tests/test_transformer_math.py` — tiny deterministic sequence/shard construction and direct shard-result comparison patterns.
6. `tests/test_transformer.py` — canonical layout, parameter-count, initialization, immutability, and exact public-symbol conventions.
7. `tests/test_transformer_worker_group.py` — new focused public-seam suite for Ticket 020.
8. `src/how_llms_work/routes/train_transformer.py` — confirm it remains empty and out of scope.
9. `pyproject.toml` — Python version, pytest-asyncio configuration, Ruff, Black, and strict-mypy settings; confirm no dependency change.
10. `020-coordinate-request-scoped-worker-groups-and-cleanup.md` — direct acceptance authority.
11. `019-compute-logical-shards-through-a-spawn-safe-worker-protocol.md` and the current Ticket 019 implementation — blocker contract that must be reused unchanged.
12. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` — worker count, static assignment, async polling, deadlines, ownership, cancellation, staged shutdown, and cleanup decisions.
13. `llm_works_file_structure.md` — TypeScript `train-worker.ts` and `train.ts` as low-level behavior evidence only; do not copy host-dependent sequence slicing, Worker Threads, `SharedArrayBuffer`, cached-model skipping, or incomplete cleanup.

## Step 1 — Freeze the parent-side public group contract and lifecycle states

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — proposed public `RequestScopedWorkerGroup`, public creation boundary, immutable group cleanup report, one public group failure boundary, and minimal group lifecycle state.
- `tests/test_transformer_worker_group.py` — public contract, state transition, and exact export tests.

**Purpose:**

Define one stable parent-side seam before implementing process mechanics. The route ticket must later be able to create, start, compute through, cancel, and clean one group without reaching into private process collections or shared-memory objects.

**Actions:**

- Introduce one public `RequestScopedWorkerGroup` abstraction in the existing worker module because ADR 0002 assigns both the worker protocol and worker-group supervision to that module.
- Define a minimal lifecycle that distinguishes at least:
  - newly allocated or starting;
  - ready;
  - computing one epoch;
  - stopping/cleaning;
  - closed;
  - failed.
- Define one async creation/start boundary that accepts:
  - the exact layer count and canonical current weight storage;
  - the immutable complete Training Sequence tuple;
  - the exact four canonical Logical Training Shards;
  - one generic async poll observer or cancellation observer that is independent of FastAPI;
  - only narrow optional runtime seams needed by deterministic tests.
- Define one async epoch operation that accepts the epoch and current parent weight storage, copies the validated weights into the shared weight block before command dispatch, and returns an immutable four-element tuple of independent `LogicalTrainingShardResult` values in shard order.
- Define one explicit, idempotent async cleanup operation that returns an immutable report containing:
  - whether cooperative shutdown completed;
  - whether terminate or kill was required;
  - recorded process exit codes;
  - secondary cleanup failures represented without exception text or resource names.
- Establish the outcome rule:
  - a startup/epoch/cancellation failure remains the primary outcome;
  - cleanup failures are recorded separately and do not replace that primary failure;
  - when cleanup is the only failure, the group cannot be treated as a successful completed run;
  - any forced termination marks the group unsuccessful.
- Keep process, connection, array, memoryview, and `SharedMemory` objects private. The public epoch result must be an independent copy so callers cannot retain a live view into request-owned shared memory.
- Add exact public-symbol and state misuse tests:
  - compute before ready;
  - overlapping compute calls;
  - compute after cleanup;
  - repeated cleanup;
  - mutation of returned result data cannot alter internal or shared buffers.

**Guardrails:**

- Do not expose generated process IDs, process names, pipe handles, shared-memory names, private worker lists, or internal polling helpers as the test/API contract.
- Do not add FastAPI `Request`, SSE payloads, route locks, persistence, sample generation, final evaluation, or Adam to the group.
- Do not create a second set of protocol records or a second worker target.
- Do not make cleanup rely solely on `__del__`; explicit lifecycle ownership is required.

**Expected result:**

- Ticket 020 has one precise reusable parent-side API whose public results survive group cleanup and whose lifecycle makes illegal overlap or reuse fail deterministically.

**Verification:**

- Run contract/state-focused tests in `tests/test_transformer_worker_group.py` before adding multi-process happy paths.

## Step 2 — Implement one-time worker-count calculation and deterministic shard assignment

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — proposed pure actual-worker-count calculation, one production call to `os.cpu_count()`, and static assignment construction.
- `tests/test_transformer_worker_group.py` — CPU-count table and assignment table.

**Purpose:**

Make physical worker count a performance choice only and protect the fixed logical partition from host-dependent behavior.

**Actions:**

- Add one pure calculation boundary that maps a reported CPU count to:

  ```text
  min(4, max(1, reported_cpu_count or 1))
  ```

- Have group creation call `os.cpu_count()` exactly once and store the completed worker count for the group lifetime.
- Test reported CPU counts of:
  - `None`;
  - `0`;
  - `1`;
  - `2`;
  - `4`;
  - a value greater than `4`.
- Build assignments only from the exact four canonical shard IDs:
  - `worker_index = shard_id % actual_worker_count`;
  - every worker receives its shard IDs in ascending order;
  - no shard is omitted or duplicated;
  - the combined assignment is always exactly `(0, 1, 2, 3)`.
- Assert the expected assignment tables for one, two, three, and four workers.
- Construct each `WorkerStartupConfig` with only the aligned gradient shared-memory names for that worker's assigned shard IDs and reuse the existing Ticket 019 validator before process start.

**Guardrails:**

- Do not derive shard boundaries from worker count.
- Do not slice Training Sequences by host CPU count.
- Do not schedule shards dynamically or rebalance after startup.
- Do not permit zero workers or more than four workers.
- Do not call `os.cpu_count()` again during startup, epoch computation, or cleanup.

**Expected result:**

- Every supported host maps to one stable worker count and one exact static assignment without changing logical work or later reduction order.

**Verification:**

- Run the worker-count and assignment test subset without spawning processes, then retain those tests alongside real one-through-four-worker cases.

## Step 3 — Allocate exactly five request-owned shared-memory blocks and private parent views

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — parent resource container, five-block allocator, exact-range parent views, weight publication, and request-owned cancellation/state objects.
- `tests/test_transformer_worker_group.py` — allocation count, dtype, capacity, ownership, and isolation tests.

**Purpose:**

Create one unambiguous resource owner for the numerical process boundary and ensure no optimizer, route, or generated-output state leaks into shared memory.

**Actions:**

- Rebuild the canonical `TransformerParameterLayout` from the validated layer count and use its `total_float_count` and `float32` item size as the sole requested size authority.
- Create exactly:
  - one `SharedMemory(create=True, size=canonical_bytes)` weight block;
  - four independent `SharedMemory(create=True, size=canonical_bytes)` gradient blocks.
- Let the operating system generate every name; do not construct predictable names.
- Limit each parent buffer and NumPy array to the exact canonical byte/float range even when the platform reports a larger allocation.
- Construct exactly five flat C-contiguous arrays with dtype exactly `np.float32` and shape exactly `(layout.total_float_count,)`.
- Validate the supplied current parent weights through the existing Transformer numerical contract and copy them into the shared weight array transactionally before workers are started.
- Initialize all four gradient arrays to zero.
- Keep the following outside shared memory:
  - `TransformerTrainingRun` first and second Adam moments;
  - reduction workspace and Adam scratch arrays;
  - Generated Text Samples;
  - route/disconnect state;
  - cleanup diagnostics;
  - process/group state.
- Create one fresh request-owned cancellation signal and fresh collections/state for every group. No module-global mutable worker or shared-memory registry is permitted.
- Make allocation transactional:
  - if any block, view, or later process setup fails, retain every successfully created resource in the cleanup owner;
  - cleanup must still attempt close and unlink for each partial allocation.
- In tests, observe the group through a narrow immutable diagnostic snapshot or allocation seam that reports counts/dtypes/capacities without exposing or asserting generated names.

**Guardrails:**

- Do not use `SharedMemoryManager`, `Queue`, `Manager`, a process pool, a global shared-memory cache, or cross-request buffers.
- Do not place Adam buffers or result collections into shared memory.
- Do not assume `SharedMemory.size` is exactly the requested byte count; require at least the canonical capacity and constrain all views to the exact logical range.
- Do not unlink during ordinary startup or epoch operation.

**Expected result:**

- Every complete or partially constructed group has one parent owner tracking exactly five canonical numerical blocks and no unrelated shared state.

**Verification:**

- Run allocation-failure and five-block ownership tests, including a failure after each creation position and sequential/concurrent group construction.

## Step 4 — Spawn dedicated workers and supervise complete startup under one deadline

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — local spawn context, per-worker pipe/process records, startup send/receive supervision, process-sentinel handling, and 30-second deadline.
- `tests/test_transformer_worker_group.py` — real-spawn ready, malformed, failed, exited, missing, and late startup cases.

**Purpose:**

Make worker-group readiness an all-or-nothing state with independent control and process-exit evidence for every worker.

**Actions:**

- Obtain a local context with `multiprocessing.get_context("spawn")`; never change the application-wide start method.
- For each worker:
  - create one fresh dedicated `context.Pipe(duplex=True)`;
  - create one fresh non-daemonic `context.Process` targeting the existing top-level `run_transformer_worker()`;
  - pass that worker's exact expected index, validated startup config, and child endpoint;
  - start the process once;
  - immediately close the parent's duplicate child endpoint only after `start()` succeeds;
  - retain the parent endpoint and process sentinel in the supervisor.
- If `start()` fails, close both available endpoints where applicable, retain the process object for safe cleanup, and fail startup without attempting later workers as if the group were valid.
- Establish one absolute startup deadline from `time.monotonic()` before the first worker start/wait sequence. Do not restart the 30-second allowance per worker.
- Wait on all unready worker parent endpoints and all corresponding process sentinels.
- For every ready endpoint:
  - receive exactly one object;
  - accept only a strictly validated matching `ReadyMessage`;
  - reject duplicate ready, wrong version, wrong worker, wrong assigned shards, `FailureMessage`, unknown records, EOF, or receive failure.
- If a process sentinel becomes ready before a valid ready record has been accepted, record its exit status when available and fail startup.
- Mark the group ready only after every actual worker has one matching ready commit before the shared deadline.
- Any startup failure must transition directly to the same full cleanup path used by later failures.

**Guardrails:**

- Do not treat process liveness alone as readiness.
- Do not ignore a ready pipe because another worker already failed.
- Do not wait on workers serially with one 30-second timeout each.
- Do not use daemon processes.
- Do not retain the parent's duplicate child endpoint after a successful start.

**Expected result:**

- A group is either completely ready with one validated signal from every live worker or it fails atomically and owns all partial resources until cleanup finishes.

**Verification:**

- Parameterize real one-through-four-worker successful startup and controlled worker targets that exit, fail, send malformed data, omit ready, or become ready after the deterministic deadline.

## Step 5 — Add bounded off-event-loop polling with a route-neutral observation boundary

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — shared async poll operation around `multiprocessing.connection.wait()`, `0.1`-second timeout constant, deadline checks, cancellation checks, and poll observer.
- `tests/test_transformer_worker_group.py` — delegated wait spy, thread identity, event-loop heartbeat, poll cadence, cancellation, and timeout tests.

**Purpose:**

Prevent process waiting from blocking FastAPI's event-loop thread while giving later route orchestration a stable place to observe browser disconnects after every bounded poll.

**Actions:**

- Centralize startup, epoch, and shutdown waiting through one behaviorally shared async boundary.
- For each poll:
  - build the current set of worker pipe endpoints and process sentinels;
  - invoke `multiprocessing.connection.wait(waitables, timeout=0.1)` through `asyncio.to_thread()`;
  - return to the event loop after the wait completes;
  - invoke the supplied generic async poll observer;
  - check the request-owned cancellation signal;
  - compare `time.monotonic()` with the operation's absolute deadline.
- The observer must not import FastAPI or know about HTTP. A later route can provide a callback that checks `Request.is_disconnected()` and signals cancellation.
- Treat observer cancellation or an already-set group cancellation signal as a primary cancellation outcome that proceeds through complete cleanup.
- Preserve `asyncio.CancelledError`: cleanup must run, then cancellation must propagate rather than becoming a generic worker failure.
- Keep the `0.1` interval exact as the maximum wait argument; do not assert or promise exact elapsed wall time.
- Provide one narrow immutable runtime seam for tests to:
  - delegate to real `connection.wait()` while observing the timeout value and thread identity;
  - use a deterministic monotonic clock for deadline cases;
  - select top-level controlled worker targets for failure scenarios.
- Keep this seam limited to clock/wait/process-control behavior and out of the route API.

**Guardrails:**

- Do not call blocking `connection.wait()`, `join()`, or long sleeps directly on the event-loop thread.
- Do not busy-loop with zero-timeout polling.
- Do not hide a deadline extension inside every poll.
- Do not make tests depend on a private helper name or exact internal list order; test public startup/compute/cleanup behavior through the narrow runtime seam.
- Do not perform live browser disconnect tests in this ticket.

**Expected result:**

- All process supervision yields to the async loop at least every bounded poll and exposes exactly one route-neutral observation point after each wait.

**Verification:**

- Prove the wait delegate runs on a thread different from the event-loop thread, a heartbeat task progresses while a worker is delayed, every delegated timeout is `0.1`, and observer-triggered cancellation exits through cleanup.

## Step 6 — Dispatch one epoch, validate every commit, and materialize canonical shard results

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — epoch state enforcement, shared-weight publication, one command per worker, five-minute deadline, result-set validation, and independent canonical result copying.
- `tests/test_transformer_worker_group.py` — happy epoch, completion-order variation, corruption, timeout, overlap, and commit-gating tests.

**Purpose:**

Turn worker-originated commit records plus shared gradient blocks into the exact four independent `LogicalTrainingShardResult` values needed by the existing parent `TransformerTrainingRun`.

**Actions:**

- Reject an epoch call unless the group is fully ready, not already computing, not cancelled, and not previously failed or closed.
- Validate:
  - exact integer epoch;
  - exact canonical current weight array type, dtype, shape, C-contiguity, and finiteness;
  - consistency with the group's canonical layout.
- Copy the complete current parent weight candidate into the shared weight array before sending any compute command. Validate the completed shared copy before dispatch.
- Establish one absolute five-minute epoch deadline.
- Send exactly one matching `ComputeMessage` to every worker. If any send fails, mark the epoch failed and do not send a second command to that worker.
- Record the expected worker, epoch, assigned shard tuple, and one outstanding-command state.
- Wait on every outstanding worker's parent pipe and process sentinel through the Step 5 poll boundary.
- For each received object:
  - accept only one strictly validated matching `ResultMessage` from that endpoint;
  - accept a matching `FailureMessage` only as a run-failing signal;
  - reject wrong protocol version, stale/future epoch, duplicate result, wrong worker, wrong or unassigned shard IDs, missing or duplicate shard IDs, loss-count mismatch, NaN/infinity, unknown object, EOF, or receive failure.
- Treat a worker sentinel before its expected result as an epoch failure even if another worker has already completed.
- Do not read a gradient block when its owning worker's matching result has not been validated.
- After every worker result is validated:
  - prove the committed shard-ID union is exactly `{0, 1, 2, 3}`;
  - map each committed loss to its aligned shard ID;
  - validate the complete corresponding shared gradient array for shape, dtype, contiguity, and finiteness;
  - copy each gradient into a fresh canonical `TransformerGradientBuffer`;
  - construct one `LogicalTrainingShardResult` with the known canonical shard and processed sequence count;
  - return exactly four results ordered `0, 1, 2, 3`.
- Clear outstanding-command state only after the complete result set has been materialized.
- On any failure or timeout, expose no successful partial result collection and transition the group to failed cleanup.

**Guardrails:**

- Do not reduce gradients in this module.
- Do not apply Adam or mutate `TransformerTrainingRun`.
- Do not return completion order.
- Do not expose shared NumPy arrays or memoryviews to the caller.
- Do not reuse a prior epoch's result or gradient if the current commit set is incomplete.
- Do not pipeline another epoch while result materialization is in progress.

**Expected result:**

- One successful public epoch operation returns four finite independent shard results that can be passed directly to `TransformerTrainingRun.advance_epoch()` with no behavior dependent on worker count or completion order.

**Verification:**

- Compare one-through-four real-worker results with direct `calculate_logical_training_shard()` results for the same tiny deterministic snapshot using the approved tight tolerances.
- Force reverse and mixed completion orders with controlled top-level worker targets and require identical ordered results.
- Cover stale, duplicate, missing, malformed, wrong-worker, unassigned-shard, non-finite, exited-worker, send failure, receive failure, and epoch-timeout cases.

## Step 7 — Implement cooperative shutdown, terminate/kill escalation, and outcome preservation

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — idempotent group cleanup coordinator, cancellation signaling, cooperative stop, two escalation deadlines, joins, exit-code capture, and cleanup report.
- `tests/test_transformer_worker_group.py` — cooperative, terminate, kill, forced-failure, and primary-outcome preservation cases.

**Purpose:**

Guarantee that every worker is gone before numerical memory is released and that forced or defective shutdown can never be reported as a successful run.

**Actions:**

- Start cleanup by setting the group-owned cancellation signal so no new startup/epoch operation can begin.
- Attempt cooperative shutdown for every process that may still receive commands:
  - send one validated `StopMessage` where the pipe is still usable;
  - wait for matching `StoppedMessage` records and process sentinels;
  - use one shared absolute two-second cooperative deadline, not two seconds per worker;
  - record clean exit code `0` as independent integrity evidence.
- Treat a clean cooperative worker as requiring both:
  - one matching stopped record when the protocol state permits it; and
  - process termination with exit code `0`.
- After the cooperative window, call `terminate()` on each surviving worker and mark the group as forcibly terminated.
- Wait through the same bounded off-event-loop poll boundary for one additional shared absolute two-second deadline.
- Call `kill()` on every process still alive after terminate, mark the group as forcibly killed, and continue to bounded joins until each process is confirmed dead.
- Record exit codes before closing process objects.
- A terminate or kill call, a nonzero/missing clean exit, or a worker still alive after the complete escalation sequence must prevent a successful completion outcome.
- Implement cleanup so a primary startup, epoch, cancellation, or protocol failure remains the primary error. Append stop/send/wait/terminate/kill/join/close/unlink failures to the cleanup report rather than replacing the primary error.
- If normal work succeeded but forced termination or cleanup integrity failed, surface a group cleanup failure instead of returning success.
- Make repeated cleanup safe: it must not start new processes, resend compute, recreate shared memory, or report success after a prior forced stage.

**Guardrails:**

- Do not call `Process.close()` while a process is alive.
- Do not assume `terminate()` runs worker `finally` blocks.
- Do not release shared numerical memory immediately after sending stop.
- Do not treat a stopped message without process exit, or exit code zero without the expected stopped message, as complete clean evidence.
- Do not swallow `asyncio.CancelledError`; preserve it after cleanup.

**Expected result:**

- Every cleanup path has a deterministic cooperative-first escalation sequence, forced termination is visible in the internal outcome, and the original failure remains diagnosable.

**Verification:**

- Use real spawned children plus narrow controlled process/worker seams to prove:
  - all-clean cooperative shutdown;
  - stop ignored or pipe unavailable, requiring terminate;
  - a deterministic survivor after the terminate stage, requiring kill;
  - nonzero exit after stopped;
  - a primary epoch failure plus secondary cleanup failures;
  - a successful epoch followed by cleanup-only failure, which cannot be reported as success.

## Step 8 — Make cleanup non-short-circuiting and release parent-owned resources in safe order

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — stage ledger, per-resource best-effort loops, view release, connection/process closure, shared-memory close/unlink, and final closed state.
- `tests/test_transformer_worker_group.py` — injected cleanup failures, order evidence, repeated cleanup, and leak checks.

**Purpose:**

Prevent one cleanup defect from hiding or causing additional process, pipe, view, or shared-memory leaks.

**Actions:**

- Implement cleanup as explicit independent stages. Every stage must continue after an earlier stage records a failure:
  1. set cancellation;
  2. cooperative stop sends;
  3. cooperative waits;
  4. terminate survivors;
  5. second waits;
  6. kill survivors;
  7. joins and final liveness checks;
  8. exit-code recording;
  9. parent pipe closure;
  10. process-object closure;
  11. release semantic views, flat arrays, and exact-range memoryviews;
  12. close each parent `SharedMemory` handle;
  13. attempt each parent-owned unlink independently;
  14. publish the immutable cleanup report and closed state.
- Do not begin view/shared-memory release until every process is confirmed not alive.
- Release references in a deliberate order so NumPy or `memoryview` exports do not cause `SharedMemory.close()` to fail with hidden live buffers.
- Attempt close and unlink independently for all five blocks, even if one block fails.
- Call unlink only from the parent owner and at most once per block.
- Preserve platform semantics:
  - POSIX uses unlink to remove the named block;
  - Windows frees the block after every handle is closed, while the owner still attempts its sole unlink operation.
- Record cleanup failures in stable internal categories without exposing generated names, paths, tracebacks, or numerical values.
- After cleanup, clear internal resource references so a later call cannot accidentally reuse closed process, pipe, array, or memory objects.
- Add a narrow cleanup-operation seam that can fail selected stages while still delegating all other stages to real resources.

**Guardrails:**

- Do not place all cleanup in one `try` block that stops at the first exception.
- Do not unlink while any worker is alive.
- Do not depend on garbage collection as the primary owner; explicit release is required.
- Do not assert generated shared-memory names in tests.
- Do not let one failed pipe close skip process close or any shared-memory attempt.

**Expected result:**

- Cleanup always attempts every owned resource operation in a safe order, returns complete secondary diagnostics, and leaves the group permanently closed.

**Verification:**

- Parameterize failures at each cleanup category and assert all later categories were still attempted.
- Verify all real workers are absent from the post-test active-child set, all group endpoints/process objects are closed, and no group-owned block remains attachable where the platform permits that check.
- Treat resource-tracker warnings as test failures or explicit leak evidence rather than ignoring them.

## Step 9 — Prove request isolation, numerical equivalence, and all required failure paths in ordinary pytest

**Files and symbols:**

- `tests/test_transformer_worker_group.py` — complete Ticket 020 public-seam matrix.
- `tests/test_transformer_worker.py` — unchanged Ticket 019 regression suite unless a narrowly reusable fixture must be extracted without weakening current coverage.
- `tests/test_transformer_training.py` and `tests/test_transformer_math.py` — unchanged numerical regressions.

**Purpose:**

Provide bounded, Windows-compatible evidence that the supervisor works with real operating-system resources and does not merely pass mock-only lifecycle tests.

**Actions:**

Organize the new focused suite into these public-behavior groups:

1. **Pure deterministic policy**
   - CPU-count mapping for missing, zero, one, two, four, and larger counts.
   - Static assignment for one through four workers.
   - Exact four-shard union and ascending per-worker order.

2. **Real-spawn success**
   - One through four workers with the same tiny deterministic canonical snapshot.
   - Exactly five request-owned shared blocks.
   - One dedicated duplex pipe per worker.
   - All workers ready under the group deadline.
   - One epoch result tuple in shard order.
   - Direct numerical parity with `calculate_logical_training_shard()`.
   - Different completion orders with unchanged results.
   - Cooperative stopped records and exit code zero.
   - No live child or shared-memory leak.

3. **Startup failures**
   - process-start failure;
   - worker-controlled failure;
   - malformed ready record;
   - wrong worker/version/assignment;
   - EOF or worker exit before ready;
   - one missing ready worker;
   - startup timeout.

4. **Epoch failures**
   - overlapping epoch call;
   - send failure;
   - worker failure;
   - malformed result;
   - wrong worker/version/epoch;
   - stale or duplicate result;
   - missing or unassigned shard;
   - mismatched loss count;
   - non-finite loss or gradient;
   - sentinel before result;
   - epoch timeout;
   - cancellation at the poll observer.

5. **Shutdown escalation**
   - cooperative stop;
   - stop-send or stopped-record failure;
   - terminate escalation;
   - kill escalation;
   - nonzero or missing clean exit status;
   - forced stage prevents success.

6. **Cleanup integrity**
   - every cleanup stage attempted after selected earlier failures;
   - original startup/epoch/cancellation outcome preserved;
   - cleanup-only failure prevents success;
   - parent-only unlink;
   - no shared-memory release while a worker is alive;
   - repeated cleanup idempotence.

7. **Request isolation**
   - repeated sequential groups use fresh process, pipe, cancellation, state, and shared-memory objects;
   - two controlled concurrent groups produce independent results;
   - failure/cleanup of one group does not alter the other;
   - returned result mutation from one group does not alter another group or later epoch.

- Keep fixtures tiny:
  - one canonical one-layer layout;
  - a minimal finite initialized weight snapshot;
  - a small immutable Training Sequence tuple that still exercises non-empty gradients;
  - exact four canonical shards, including empty shards where useful.
- Reuse existing public Transformer constructors and direct shard calculation for expected numerical behavior. Do not generate expected results by calling the new group under test.
- Keep maximum layers, maximum corpus, and long endurance outside ordinary pytest.

**Guardrails:**

- Do not replace real processes, pipes, sentinels, and shared memory with a wholly fake supervisor.
- Do not assert exact process IDs, process/shared-memory names, OS scheduling order, or exact elapsed time.
- Do not test live browser disconnection, route lock ownership, SSE, persistence, or frontend behavior.
- Do not weaken Ticket 019 protocol tests to simplify the group implementation.
- Do not add broad sleeps; use explicit records, sentinels, deterministic clocks, and bounded deadlines.

**Expected result:**

- Ordinary pytest contains direct evidence for every Ticket 020 success, corruption, timeout, escalation, cleanup, and isolation criterion while remaining bounded on Windows.

**Verification:**

- Run the focused group suite repeatedly during implementation, then run the existing worker and Transformer regression suites before the full project suite.

## Step 10 — Reconfirm formatting, typing, complete regression safety, and final scope

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — final production diff.
- `tests/test_transformer_worker_group.py` — final focused test diff.
- Any conditionally touched existing worker test file — confirm only narrowly necessary fixture/public-export changes.
- `pyproject.toml` and `poetry.lock` — confirm unchanged.

**Purpose:**

Ensure the lifecycle implementation is reviewable, strictly typed, dependency-neutral, and limited to Ticket 020.

**Actions:**

- Run Black formatting/checking only through the repository's configured command and avoid unrelated formatting churn.
- Run focused Ticket 020 tests first.
- Run Ticket 019 worker regressions and affected Transformer training/math tests.
- Run the complete pytest suite once at the end.
- Run Ruff and strict mypy.
- Run `git diff --check` and inspect `git status --short`.
- Inspect the final diff for:
  - no route, schema, main, SSE, persistence, or frontend work;
  - no Queue, Manager, pool, global start-method call, or process reuse;
  - no new dependency or lockfile change;
  - no generated process/shared-memory names in assertions or logs;
  - no broad exception detail in stable cleanup diagnostics;
  - every acceptance criterion represented by implementation and test evidence.
- Record actual command outputs honestly in the implementation result.

**Guardrails:**

- Do not claim success based only on the user's earlier baseline.
- Do not repair unrelated failures under Ticket 020.
- Do not create the commit until focused and complete checks pass.
- Do not include generated caches or `.data/` artifacts in the diff.

**Expected result:**

- The final change is limited to the parent-side Request-Scoped Worker Group and its focused tests, with all existing behavior preserved.

**Verification:**

- Compare the final changed-file list with the expected/prohibited lists below before commit.

## Focused verification plan

Run from the backend directory:

```powershell
poetry run pytest tests/test_transformer_worker_group.py -q
poetry run pytest tests/test_transformer_worker.py tests/test_transformer_training.py tests/test_transformer_math.py -q
poetry run black --check src/how_llms_work/ml/transformer_worker.py tests/test_transformer_worker_group.py
poetry run ruff check src/how_llms_work/ml/transformer_worker.py tests/test_transformer_worker_group.py
poetry run mypy src/how_llms_work/ml/transformer_worker.py
```

Expected result:

- Worker-count and assignment policy tests pass.
- Real one-through-four-worker group tests pass with equivalent canonical shard results.
- Startup, protocol-corruption, timeout, cancellation, staged shutdown, cleanup-failure, and leak tests pass.
- Ticket 019 worker protocol and existing Transformer numerical/training regressions remain passing.
- Changed files satisfy Black, Ruff, and strict mypy.

For tests whose timing or process behavior is susceptible to accidental flakiness, repeat the focused group suite locally before the full suite:

```powershell
1..3 | ForEach-Object {
    poetry run pytest tests/test_transformer_worker_group.py -q
}
```

Expected result:

- All three bounded repetitions pass without resource-tracker warnings, surviving group workers, or leaked shared-memory evidence.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
poetry run black --check .
git diff --check
git status --short
```

Expected result:

- All tests pass.
- Ruff reports no findings.
- Strict mypy reports no issues.
- Black reports no formatting changes required.
- `git diff --check` reports no whitespace errors.
- `git status --short` shows only Ticket 020 production/test changes and no generated cache, `.data`, dependency, lockfile, route, or frontend artifacts.

## Manual acceptance checklist

- [ ] Exactly one actual worker count was calculated from one `os.cpu_count()` observation and bounded to one through four.
- [ ] The exact four Logical Training Shards were assigned by modulo, with ascending IDs per worker.
- [ ] One run created exactly one weight block and four separate canonical gradient blocks, all `float32`.
- [ ] Adam moments, reduction workspaces, Generated Text Samples, and route state remained parent-local and non-shared.
- [ ] Every worker had one dedicated duplex pipe, and the parent's duplicate child endpoint was closed immediately after successful start.
- [ ] The complete group required all matching ready records within one 30-second monotonic deadline.
- [ ] One epoch sent at most one compute command per worker and prohibited overlap/pipelining.
- [ ] Pipe endpoints and process sentinels were polled through `connection.wait()` with a `0.1` timeout off the event-loop thread.
- [ ] The generic poll observer can later support route disconnect observation without importing FastAPI.
- [ ] One complete four-shard epoch used one five-minute monotonic deadline.
- [ ] Wrong-version, stale, duplicate, missing, malformed, wrong-worker, unassigned-shard, and non-finite results failed before buffer exposure.
- [ ] A matching result record was required before each assigned gradient block was copied.
- [ ] Results were returned in shard order `0 → 1 → 2 → 3` regardless of completion order.
- [ ] One through four real-worker configurations matched the same direct tiny snapshot within approved tolerances.
- [ ] Cooperative shutdown used one two-second group deadline and required stopped records plus clean exit code zero.
- [ ] Survivors were terminated, waited for up to two more seconds, then killed if still alive.
- [ ] Any terminate/kill stage prevented a successful completion outcome.
- [ ] Cancellation, stop, waits, escalation, joins, exit capture, pipe/process closure, view release, shared-memory close, and every unlink were attempted non-short-circuitingly.
- [ ] No numerical memory was released while a worker remained alive, and only the parent attempted unlink.
- [ ] A primary failure was preserved while cleanup failures were retained separately; a cleanup-only failure did not become success.
- [ ] Repeated sequential and controlled concurrent groups used fresh resource/state objects and did not interfere.
- [ ] Ordinary pytest covered every required success, failure, timeout, escalation, cleanup, and leak category.
- [ ] No route, request schema, SSE, persistence, frontend, dependency, or lockfile behavior changed.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer_worker.py
tests/test_transformer_worker_group.py
```

Conditionally changed only when implementation inspection proves a narrowly shared test fixture or public-export assertion is necessary:

```text
tests/test_transformer_worker.py
```

The default expectation is no new fixture file. Ticket 016 already proves shard mathematics, and Ticket 020 can compare real group output with direct public shard calculations on a tiny deterministic snapshot.

No package or lockfile change is expected.

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/ml/__init__.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/word2vec.py
src/how_llms_work/routes/
tests/test_simple_chat.py
tests/test_bpe.py
tests/test_bpe_tokenize.py
tests/test_math_utils.py
tests/test_matrix.py
tests/test_neural_net.py
tests/test_neural_net_persistence.py
tests/test_neural_net_route.py
tests/test_train_embed_persistence.py
tests/test_train_embed_route.py
tests/test_transformer.py
tests/test_transformer_completion.py
tests/test_word2vec.py
tests/test_word2vec_training.py
tests/test_word2vec_results.py
tests/fixtures/
.data/
README.md
pyproject.toml
poetry.lock
poetry.toml
frontend/
SPEC.md
CONTEXT.md
0002-stabilize-python-transformer-training-and-process-lifecycle.md
019-compute-logical-shards-through-a-spawn-safe-worker-protocol.md
020-coordinate-request-scoped-worker-groups-and-cleanup.md
```

## Risk notes and safeguards

1. **Risk:** Worker count is recalculated during the run and assignments change.
   - **Safeguard:** Call `os.cpu_count()` once at creation, store the bounded result, and protect the exact one-through-four assignment tables with pure tests.

2. **Risk:** Physical worker count changes logical shard boundaries or reduction input order.
   - **Safeguard:** Consume the existing four canonical shards and return results strictly by shard ID, never by worker or completion order.

3. **Risk:** Group creation introduces a global pool or changes the application's start method.
   - **Safeguard:** Use only one local `multiprocessing.get_context("spawn")` per group and non-daemonic request-owned processes.

4. **Risk:** A process target or controlled test target is not importable under Windows spawn.
   - **Safeguard:** Keep production and controlled test targets at module top level and prove every failure seam with real spawn.

5. **Risk:** The parent retains the duplicate child pipe endpoint and prevents EOF or leaks a handle.
   - **Safeguard:** Close it immediately after successful `start()` and cover start-failure/partial-start cleanup separately.

6. **Risk:** A worker exits but the parent waits forever on its pipe.
   - **Safeguard:** Include every process sentinel in every relevant `connection.wait()` set and apply absolute deadlines.

7. **Risk:** Blocking waits freeze FastAPI's event loop.
   - **Safeguard:** Execute every bounded `connection.wait()` through `asyncio.to_thread()` and prove event-loop heartbeat progress.

8. **Risk:** Polling becomes a busy loop or disconnects are observed too slowly.
   - **Safeguard:** Pass the exact `0.1` timeout to every wait and invoke the generic observer after every returned poll.

9. **Risk:** Recomputing a deadline from each poll permits an indefinite hang.
   - **Safeguard:** Compute one absolute monotonic deadline per startup, epoch, cooperative stop, and terminate-wait stage.

10. **Risk:** A partial or stale shared gradient is read before its commit marker.
    - **Safeguard:** Treat a strictly matching `ResultMessage` as the only read authorization and copy nothing from an uncommitted block.

11. **Risk:** A valid-looking result arrives on the wrong worker pipe.
    - **Safeguard:** Validate exact endpoint-to-worker identity, assigned shards, epoch, version, and one-result cardinality.

12. **Risk:** One worker's completion order changes training.
    - **Safeguard:** Accumulate records by shard ID and materialize the public tuple only in canonical `0 → 1 → 2 → 3` order.

13. **Risk:** Parent weight storage and shared weight storage diverge across epochs.
    - **Safeguard:** Validate and copy the complete current parent weights into the shared block before each dispatch, then validate the completed copy.

14. **Risk:** Parent result objects retain shared-memory aliases and become invalid after cleanup.
    - **Safeguard:** Copy every committed gradient into a fresh canonical `TransformerGradientBuffer` before returning.

15. **Risk:** A second compute call overlaps the first and corrupts shared gradient ownership.
    - **Safeguard:** Enforce one group-level computing state and reject overlapping calls before any command is sent.

16. **Risk:** Cooperative shutdown trusts only a message or only an exit code.
    - **Safeguard:** Require matching stopped records and clean status zero as independent integrity evidence.

17. **Risk:** `terminate()` or `kill()` leaves pipe/shared-memory state unsafe.
    - **Safeguard:** Mark forced shutdown as failure, join until no process is alive, and only then release numerical resources.

18. **Risk:** `Process.close()` is attempted while the process is alive.
    - **Safeguard:** Keep process-object closure after final liveness and exit-code capture, and record rather than short-circuit on failure.

19. **Risk:** One cleanup exception prevents later resources from being released.
    - **Safeguard:** Use per-stage and per-resource best-effort loops with an immutable secondary failure ledger.

20. **Risk:** Cleanup replaces the useful startup/epoch error.
    - **Safeguard:** Preserve the primary exception/outcome and attach or return cleanup diagnostics separately.

21. **Risk:** NumPy arrays or memoryviews keep exported buffers alive and make shared-memory close fail.
    - **Safeguard:** Release semantic views, flat arrays, and exact-range memoryviews deliberately before handle closure, and test injected close failures without abandoning later unlinks.

22. **Risk:** Shared memory is unlinked while a worker still has a live handle.
    - **Safeguard:** Assert every process is dead before view release, close, or unlink begins.

23. **Risk:** Windows and POSIX shared-memory deletion differ.
    - **Safeguard:** Test the parent ownership and live-handle contract portably; use platform-appropriate no-reattach evidence without assuming POSIX unlink timing.

24. **Risk:** Kill escalation cannot be made deterministic on Windows.
    - **Safeguard:** Wrap a real spawned process with the narrow approved process-control seam so only the first escalation behavior is controlled; retain real sentinel, join, exit, pipe, and shared-memory evidence.

25. **Risk:** Real-spawn lifecycle tests are slow or flaky.
    - **Safeguard:** Use one-layer tiny fixtures, explicit protocol synchronization, deterministic clocks, shared group deadlines, no broad sleeps, and repeated focused runs.

26. **Risk:** Concurrent groups accidentally share cancellation, process, pipe, or memory state.
    - **Safeguard:** Allocate every mutable object in the group factory and run controlled concurrent isolation tests.

27. **Risk:** Resource-tracker warnings are ignored and leaks reach later tests.
    - **Safeguard:** Treat warnings, surviving active children, attachable old blocks, or unclosed handles as test failures and use `try/finally` cleanup in every test fixture.

28. **Risk:** The group supervisor expands into route orchestration or optimizer ownership.
    - **Safeguard:** Return canonical shard results only; leave Adam to `TransformerTrainingRun` and leave HTTP/SSE/disconnect interpretation/run-slot/persistence to the later route ticket.

29. **Risk:** Standard-library functionality triggers a dependency or lockfile change.
    - **Safeguard:** Use only Python 3.12 multiprocessing/asyncio/shared-memory APIs plus existing NumPy.

30. **Risk:** User-reported baseline is mistaken for planning-session verification.
    - **Safeguard:** Re-run all baseline commands in `implement-prompt` before editing and report observed outputs.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Coordinate request-scoped Transformer workers
```

Commit body should mention:

- one-time bounded worker-count calculation and exact static four-shard assignment;
- one local spawn context, one dedicated duplex pipe per worker, and immediate duplicate-child-end closure;
- exactly one request-owned shared weight block and four shard-gradient blocks;
- complete-group startup and epoch supervision through pipes plus process sentinels;
- off-event-loop `0.1`-second bounded waits and the route-neutral poll observation boundary;
- matching result commits before independent canonical gradient copies;
- completion-order-independent one-through-four-worker shard results;
- cooperative stop, two-second terminate stage, kill escalation, and forced-failure behavior;
- non-short-circuiting cleanup, parent-only unlink, safe release ordering, and primary-outcome preservation;
- sequential/concurrent request isolation and real-spawn leak coverage;
- no Adam, generation, route, SSE, persistence, frontend, dependency, lockfile, Queue, Manager, pool, global start-method, or cross-request resource work;
- the exact focused and full verification commands actually executed and their observed results.

Do not create a commit during `to-plan-prompt`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan020.md`;
- `020-coordinate-request-scoped-worker-groups-and-cleanup.md`;
- Ticket 019 and its completed current implementation/tests;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- the latest `py_llm_pipeline_explorer_file_structure(56).md`;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the live repository again, establish its own baseline before editing, preserve user changes, implement only Ticket 020, reuse the Ticket 019 protocol and worker target, keep four logical shards fixed, create exactly five request-owned numerical blocks, supervise real spawned processes through dedicated pipes and sentinels with bounded off-event-loop waits, commit-gate all gradient reads, return canonical independent shard results, perform staged non-short-circuiting cleanup, preserve primary outcomes, prove one-through-four-worker equivalence and request isolation in ordinary pytest, run focused and full verification, report actual command outcomes honestly, inspect final scope, and create the implementation commit only after every required check passes.
