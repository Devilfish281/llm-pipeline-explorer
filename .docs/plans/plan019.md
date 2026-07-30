---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "019"
source_work_item: 019-compute-logical-shards-through-a-spawn-safe-worker-protocol.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure.md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 019: Compute Logical Training Shards through a spawn-safe worker protocol

## Initial checklist

- Confirm Ticket 019 is the only selected work item and that its Ticket 016 blocker is represented by the completed public Transformer forward, backward, Training Sequence, and Logical Training Shard calculations in the latest Python Backend export.
- Treat the latest supplied `py_llm_pipeline_explorer_file_structure.md`, created after Ticket 018, as the current-code source of truth; do not let older exports, snippets, plans, or TypeScript implementation details override it.
- Preserve the existing canonical flat Transformer parameter layout, semantic view builder, `TransformerGradientBuffer`, `LogicalTrainingShard`, `TransformerTrainingSequence`, and `calculate_logical_training_shard()` boundaries instead of introducing a second layout or numerical implementation.
- Limit production work to the top-level spawn-importable worker protocol and worker entry point in `src/how_llms_work/ml/transformer_worker.py`.
- Add ordinary pytest coverage through real `multiprocessing.get_context("spawn")` processes, dedicated duplex pipes, and parent-owned `multiprocessing.shared_memory.SharedMemory`; do not use mock processes, Queue, Manager, a process pool, or the application's global start method.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Finish with focused worker tests, affected Transformer regressions, Black and Ruff formatting checks for changed Python files, the complete pytest suite, Ruff, strict mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript worker behavior to Python and treat the latest complete Python Backend export as current-code truth.
2. `019-compute-logical-shards-through-a-spawn-safe-worker-protocol.md` for the immediate scope, acceptance criteria, approved real-spawn test seam, blocker, constraints, and out-of-scope boundaries.
3. The latest `py_llm_pipeline_explorer_file_structure.md`, created `2026-07-28T14:02:11Z`, for current source files, completed Transformer APIs, tests, fixtures, project configuration, typing style, and repository commands.
4. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for the durable Request-Scoped Shared Memory, Logical Training Shard, protocol, state-machine, ownership, privacy, commit-marker, and exit-code decisions.
5. The completed Ticket 016 implementation and tests in the current source, plus Tickets 015, 017, and 018 only where they establish reusable current boundaries that Ticket 019 must not rewrite.
6. The latest `llm_works_file_structure.md`, especially TypeScript `train-worker.ts`, as behavioral evidence for zero-before-compute, fixed sequence iteration, forward/backward reuse, shared gradient publication, and ready/result signaling only.
7. Official Python 3.12 multiprocessing, shared-memory, dataclass, and NumPy array-flag documentation as technical cross-checks for local spawn contexts, importable process targets, duplex pipes, process exit codes, shared-memory close/unlink ownership, frozen/slotted records, C-contiguity, and read-only views.
8. Older Python exports, prior plans, stale specification observations, historical worker-thread partitioning, and host-dependent TypeScript behavior are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 019 creates the first real operating-system process boundary for Transformer training without yet creating the Request-Scoped Worker Group supervisor or FastAPI orchestration.

The current Python Backend can already:

- build one immutable Transformer Preprocessing Snapshot;
- represent exactly four deterministic Logical Training Shards;
- construct one canonical flat parameter layout and exact semantic views;
- initialize finite C-contiguous `float32` weights;
- allocate a canonical flat `float32` gradient buffer;
- execute Transformer forward, cross-entropy, and analytical backward passes;
- calculate one complete Logical Training Shard in fixed Training Sequence order;
- return zero loss and an all-zero canonical gradient for an empty shard;
- validate and reduce four shard results in the parent-owned `TransformerTrainingRun` boundary;
- generate samples, evaluate final loss, and construct Saved Transformer Models independently of process infrastructure.

What remains is a spawn-importable worker module that accepts one immutable startup record, attaches by generated name to one parent-owned weight block and only the assigned shard-gradient blocks, validates the canonical memory contract, publishes one ready record, accepts one compute command at a time, computes assigned shards in ascending ID order, writes complete gradients only to owned blocks, and sends one result commit marker only after every assigned loss and gradient is finite.

The worker must keep the following responsibilities separate:

- **Parent-owned and out of scope here:** shared-memory creation and unlinking, actual worker-count selection, multi-worker assignment, group startup supervision, epoch deadlines, result collection across workers, Ordered Gradient Reduction, Adam, sample generation, final evaluation, route cancellation, SSE, persistence, and run-slot ownership.
- **Worker-owned in this ticket:** startup/configuration validation, shared-memory attachment, exact-range NumPy views, read-only weight enforcement, assigned gradient ownership, one-command-at-a-time state transitions, shard computation, finite publication, sanitized failure reporting, cooperative stop, handle closure, and exit status.

The protocol should use the seven ADR-approved top-level record types:

- `WorkerStartupConfig`
- `ReadyMessage`
- `ComputeMessage`
- `ResultMessage`
- `FailureMessage`
- `StopMessage`
- `StoppedMessage`

Every record must be a top-level `@dataclass(frozen=True, slots=True)` with protocol version `1`. Runtime validation must treat received records as untrusted objects: exact record class, exact protocol version, strict integer-versus-Boolean behavior, exact worker identity, exact assigned shard tuple, finite loss values, and closed enum membership must be checked before a transition or parent read authorization.

A practical exact schema to freeze in Step 1 is:

- `WorkerStartupConfig`: protocol version, worker index, layer count, declared numeric dtype, declared canonical float count, weight shared-memory name, assigned shard descriptors, aligned assigned gradient shared-memory names, and immutable primitive-tuple Training Sequence data needed by `calculate_logical_training_shard()`.
- `ReadyMessage`: protocol version, worker index, and exact assigned shard IDs.
- `ComputeMessage`: protocol version, worker index, epoch, and exact assigned shard IDs.
- `ResultMessage`: protocol version, worker index, epoch, exact assigned shard IDs, and finite per-shard losses aligned to those IDs.
- `FailureMessage`: protocol version, worker index, closed failure phase, closed failure code, optional epoch, and shard IDs only.
- `StopMessage`: protocol version and worker index.
- `StoppedMessage`: protocol version and worker index.

The implementation may refine field names after tracing the current public Transformer signatures, but it must not weaken this information boundary or add NumPy arrays, model coordinates, gradients, paths, exception strings, tracebacks, or shared-memory names to worker-originated messages.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, Black, mypy, spawn-process, or shared-memory command was executed while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and report the actual command output honestly.

## Current code observations from the latest source

- `src/how_llms_work/ml/transformer.py` already exposes the complete blocker seam, including `LogicalTrainingShard`, `TransformerTrainingSequence`, `LogicalTrainingShardResult`, `TransformerGradientBuffer`, `TransformerParameterLayout`, `TransformerParameterViews`, `build_transformer_parameter_layout()`, `build_transformer_parameter_views()`, `create_transformer_gradient_buffer()`, and `calculate_logical_training_shard()`.
- The current `calculate_logical_training_shard()` behavior is already covered through the Transformer mathematical tests, including fixed sequence order, unaveraged gradients, finite-state checks, and the exact empty-shard result. Ticket 019 should call this public operation rather than copy forward/backward loops into the worker module.
- `TransformerTrainingRun` already owns parent-side four-shard validation, Ordered Gradient Reduction, Adam, and epoch lifecycle. Ticket 019 must not move those responsibilities into workers.
- `src/how_llms_work/ml/transformer_worker.py` exists as the intended Phase 5 worker boundary, but the latest export shows no completed worker protocol, worker entry point, or worker-specific public test module.
- `tests/test_transformer.py`, `tests/test_transformer_math.py`, `tests/test_transformer_training.py`, and `tests/test_transformer_completion.py` already separate preprocessing/layout, mathematical, parent-training, and completion concerns. A new `tests/test_transformer_worker.py` is the smallest cohesive location for Ticket 019.
- The current test suite has no `tests/test_transformer_worker.py`, so real spawn importability, pipe records, shared-memory attachment, worker exit codes, and worker-side cleanup are not yet evidenced.
- `pyproject.toml` targets Python 3.12, uses pytest discovery under `tests`, configures Ruff and Black for line length 100, and enables strict mypy over `src`. No dependency change is needed because multiprocessing, shared memory, pipes, dataclasses, enums, and logging are standard-library facilities.
- The TypeScript reference worker zeros one shared gradient buffer, executes existing forward/backward behavior over its assigned sequences, posts a completion record, and sends ready after setup. The accepted Python contract deliberately strengthens that behavior with fixed Logical Training Shards, closed records, exact state validation, non-writeable weight views, per-shard ownership, sanitized failures, and independent exit-code integrity.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - One canonical flat parameter layout and semantic view boundary.
  - Exact `float32` C-order parameter and gradient representation.
  - Reusable public Logical Training Shard computation.
  - Fixed Training Sequence order and unaveraged gradients.
  - Exact empty-shard zero-loss/all-zero-gradient behavior.
  - Parent-owned Ordered Gradient Reduction and Adam boundaries that workers must not mutate.
- **Behavior present but evidence incomplete:**
  - The placeholder worker module and project configuration provide the intended location and dependencies, but no public protocol or real-spawn evidence exists.
- **Partially implemented:** None of the Ticket 019 worker behavior is sufficiently implemented to count as partial acceptance.
- **Not implemented:**
  - The seven typed protocol records and closed worker enums.
  - Exact parent/worker record validators.
  - A top-level spawn-importable worker entry point.
  - Shared-memory attachment, capacity/layout/dtype/contiguity/ownership validation.
  - Read-only worker weight views and a rejected mutation probe.
  - Assigned-only gradient publication with zero-before-compute.
  - The worker state machine and one-outstanding-command enforcement.
  - Result commit markers, sanitized failure records, stop/stopped behavior, worker-side close-only cleanup, and exact exit statuses.
  - Real-spawn pytest coverage and repeated-process isolation.
- **Evidence limitation:** The current source was inspected through the user's complete code export rather than a live repository checkout. The implementation run must inspect the live files and signatures again before editing. No process was spawned and no shared-memory block was allocated during planning.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer_worker.py` — confirm its current placeholder contents, module exports, and the smallest location for all protocol records, enums, validators, attachment helpers, and the top-level process target.
2. `src/how_llms_work/ml/transformer.py` — inspect the exact signatures and invariants of `TransformerTrainingSequence`, `LogicalTrainingShard`, `LogicalTrainingShardResult`, `TransformerGradientBuffer`, `TransformerParameterLayout`, `TransformerParameterViews`, `build_transformer_parameter_layout()`, `build_transformer_parameter_views()`, `create_transformer_gradient_buffer()`, and `calculate_logical_training_shard()`.
3. `tests/test_transformer.py` — reuse canonical layout, initialization, capacity, immutability, and exact public-symbol testing patterns without merging worker exports into the Transformer mathematical module.
4. `tests/test_transformer_math.py` — reuse tiny deterministic sequence/shard construction helpers and direct `calculate_logical_training_shard()` comparison patterns; do not duplicate its independent numerical fixture mission.
5. `tests/test_transformer_training.py` — inspect strict shard-result validation, finite-state, failure preservation, and completion-order prior art so worker records align with the existing parent transition without implementing supervision.
6. `tests/test_transformer_completion.py` — inspect isolation and mutation-preservation patterns only; completion behavior remains out of scope.
7. `tests/fixtures/transformer_preprocessing_reference.json` — use only when a tiny immutable Training Sequence needs a stable token-ID fixture; do not make a real-spawn test train the maximum model.
8. `tests/fixtures/transformer_layout_initialization_reference.json` — use canonical one-layer parameter-count evidence where useful for shared-memory sizing; do not duplicate the layout fixture.
9. `pyproject.toml` — confirm Python 3.12, pytest, Ruff, Black, and strict-mypy commands and verify that no dependency change is necessary.
10. `019-compute-logical-shards-through-a-spawn-safe-worker-protocol.md` — direct acceptance authority.
11. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` — protocol names, Request-Scoped Shared Memory ownership, state machine, static shard assignment, failure privacy, commit markers, and exit codes.
12. `llm_works_file_structure.md` — TypeScript `train-worker.ts` as behavioral evidence only; do not copy Worker Threads, `SharedArrayBuffer`, host-dependent slicing, or weak message validation.

## Step 1 — Freeze the protocol records, enums, and exact validation contract

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — `WORKER_PROTOCOL_VERSION`, worker state enum, failure phase/code enums, numeric-dtype marker, seven protocol dataclasses, and parent/worker validation functions.
- `tests/test_transformer_worker.py` — exact protocol-surface tests.

**Purpose:**

Create one stable, spawn-pickleable protocol before implementing process behavior. This prevents later group supervision from inferring fields, accepting look-alike objects, or leaking numerical/internal state.

**Actions:**

- Define `WORKER_PROTOCOL_VERSION` as the sole version authority with value `1`.
- Define one closed `StrEnum` for worker states with exactly `STARTING`, `READY`, `COMPUTING`, `STOPPING`, and `STOPPED`.
- Define minimal closed `StrEnum` sets for failure phase and failure code. The exact members should cover startup/configuration, command/state, compute/non-finite output, and shutdown/cleanup failures without embedding exception text.
- Define a closed dtype marker containing only the approved `float32` protocol value, or an equivalently strict primitive marker validated against exactly `float32`.
- Define `WorkerStartupConfig`, `ReadyMessage`, `ComputeMessage`, `ResultMessage`, `FailureMessage`, `StopMessage`, and `StoppedMessage` at module top level with `@dataclass(frozen=True, slots=True)`.
- Keep all fields limited to `int`, `float`, `str`, closed enum values, `None`, and recursively immutable tuples of those primitive values.
- Include exact worker identity in every worker-originated record.
- Include exact assigned shard IDs in ready, compute, result, and failure records whenever relevant.
- Carry per-shard losses only in `ResultMessage`, aligned one-to-one with ascending shard IDs. Do not include gradients because those remain in shared memory.
- Keep startup-only generated shared-memory names in `WorkerStartupConfig`; never echo them in ready, result, failure, or stopped records.
- Represent immutable Training Sequence and assigned shard metadata in primitive nested tuples suitable for spawn pickling. Reconstruct the current `TransformerTrainingSequence` and `LogicalTrainingShard` values inside the child rather than pickling NumPy arrays or mutable model objects.
- Add exact runtime validators at both trust boundaries:
  - worker-side startup and parent-command validation;
  - parent-side ready, result, failure, and stopped validation for a supplied expected worker index, epoch, and assigned shard tuple.
- Reject Boolean values where an exact integer is required, negative worker indices or epochs, unsupported layer counts, duplicate/unsorted/out-of-range shard IDs, mismatched tuple lengths, malformed shard boundaries, malformed sequence lengths, unsupported dtype markers, wrong declared canonical float counts, non-finite result losses, wrong record classes, and wrong versions.
- Make validators return the exact validated record or raise one stable protocol-validation exception that contains no shared-memory or numerical state in its public message.
- Define an explicit `__all__` for the worker module only if that module already follows the project's public-export pattern; do not append worker symbols to `transformer.py.__all__`.

**Guardrails:**

- Do not add Pydantic, JSON, dictionaries, arbitrary mappings, generic payload objects, `Any`-based dispatch, dynamic enum values, or untyped string message kinds.
- Do not serialize NumPy arrays, parameter arrays, gradient arrays, Adam state, model views, losses beyond the small result tuple, filesystem paths, exceptions, or tracebacks.
- Do not add actual worker-count selection or shard-to-worker assignment logic; Ticket 020 or later parent supervision owns that policy.
- Do not reject deliberately malformed test records only in their constructor; validation must occur at the receiving trust boundary so wrong-version and look-alike records can be tested.

**Expected result:**

- The protocol is top-level, exact, immutable, spawn-pickleable, versioned, and sufficiently self-describing for both worker and future parent validation without carrying numerical arrays.

**Verification:**

- Assert exact dataclass field order, `frozen=True`, `slots=True`, no instance `__dict__`, exact enum members, protocol version, pickle round trips, and rejection of wrong class/version/type/identity/shard/loss records.
- Recursively inspect every worker-originated record value and prove that only approved primitives, enums, `None`, and tuples are present.

## Step 2 — Build the startup conversion and canonical shared-memory attachment boundary

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — startup reconstruction, canonical layout construction, shared-memory attachment owner, exact-range array/view construction, and attached-resource container.
- `tests/test_transformer_worker.py` — startup validation and capacity/layout failure cases.

**Purpose:**

Ensure a worker cannot send ready until it has proven that every resource it will use matches the accepted canonical layout and its exact shard ownership.

**Actions:**

- Rebuild the sole canonical `TransformerParameterLayout` from the validated layer count; do not accept a serialized offset table or maintain worker-specific offsets.
- Compare any declared parameter count in the startup record with the rebuilt canonical total and reject mismatches as a layout failure.
- Attach to the parent-created weight shared-memory block by generated name with `SharedMemory(name=..., create=False)`.
- Attach only to the gradient shared-memory names aligned with this worker's assigned shard IDs. Do not attach to names for unassigned shards.
- For every block, validate `SharedMemory.size >= canonical_float_count * np.dtype(np.float32).itemsize`.
- Limit every memoryview and NumPy array to exactly the canonical byte and float range even when the operating system allocated a larger block.
- Construct flat arrays with dtype exactly `np.float32`, shape exactly `(canonical_float_count,)`, and C-contiguous layout; validate those properties before ready.
- Build semantic weight and gradient views exclusively through the existing canonical Transformer view boundary. Do not duplicate record offsets or reshape rules.
- Set the flat weight array non-writeable before exposing semantic views. Validate that every reachable worker weight view is non-writeable.
- Perform one safe startup enforcement probe that attempts an idempotent write through a worker weight view and requires NumPy to reject it. Verify the parent-visible bytes are unchanged before proceeding. This directly evidences the attempted-mutation acceptance criterion without introducing a test-only protocol command.
- Keep gradient arrays writeable but store them in a mapping keyed only by assigned shard ID.
- Reconstruct immutable `TransformerTrainingSequence` and `LogicalTrainingShard` values from validated primitive tuples. Validate exact sequence length, bounds, ascending shard order, and consistency with the supplied sequence count.
- Keep strong references to shared-memory handles while arrays/views exist, and define deterministic release order so array/view references and memoryviews are dropped before handle closure.
- Do not unlink from any worker code path.

**Guardrails:**

- Do not call `SharedMemory(create=True)` in the worker.
- Do not use `np.frombuffer()` or slicing in a way that accidentally exposes excess capacity or yields non-contiguous arrays.
- Do not mark gradient storage read-only or mark parent weight storage read-only; read-only enforcement is local to worker-created NumPy views.
- Do not import route, SSE, persistence, Adam, generation, or request code.
- Do not add a shared-memory name pattern assertion; generated names are opaque resource identifiers.

**Expected result:**

- A worker can either produce one fully validated attached-resource state or fail before ready with no partial successful state and no unlink attempt.

**Verification:**

- Spawn with valid exact-size and intentionally oversized blocks and require ready in both cases while proving all work is limited to the canonical range.
- Spawn with undersized weight or assigned-gradient capacity, unsupported layer count, wrong declared canonical count, wrong dtype marker, duplicate/unsorted shard descriptors, or mismatched name ownership and require one sanitized startup failure plus exit status `1`.
- After every startup failure, reopen the parent-owned shared-memory name and clean it from the parent, proving the worker closed but did not unlink it.

## Step 3 — Implement the exact worker state machine and one-command-at-a-time loop

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — top-level worker entry point, state transition helper, command receive loop, and no-pipelining checks.
- `tests/test_transformer_worker.py` — legal and illegal transition tests through a real spawned process.

**Purpose:**

Provide one importable Windows-compatible process target whose externally visible behavior is completely described by the approved records and state sequence.

**Actions:**

- Add one top-level process target, such as `run_transformer_worker(startup_config, connection)`, whose arguments are spawn-pickleable and whose module import has no process-starting side effects.
- Begin in `STARTING`, validate startup, attach every resource, and transition to `READY` only after all startup invariants and the read-only mutation probe pass.
- Send exactly one `ReadyMessage` after entering `READY`.
- In `READY`, accept only an exact matching `ComputeMessage` or `StopMessage`.
- Validate protocol version, worker index, assigned shard tuple, and non-negative epoch before changing state.
- Transition `READY → COMPUTING` for one compute and `READY → STOPPING` for one cooperative stop.
- Reject dictionaries, tuples, strings, look-alike dataclasses, worker-originated records sent as commands, wrong-version commands, wrong worker identity, wrong shard assignments, and every command not legal in the current state.
- Enforce no epoch pipelining. Poll the dedicated connection before each assigned shard and immediately before result publication; if another command is pending while the current compute remains outstanding, fail with the closed invalid-state/outstanding-command code and do not send a successful result.
- After one successful result send, transition `COMPUTING → READY` and accept the next command.
- On stop, transition `STOPPING → STOPPED`, send exactly one `StoppedMessage`, and return normally.
- Keep the connection dedicated to this one worker. Close the child endpoint in worker `finally`; future parent code will close its child-end copy after successful process start.

**Guardrails:**

- Do not call `multiprocessing.set_start_method()`.
- Do not create a child process, pool, Queue, Manager, socket, file IPC channel, background thread, or second pipe inside the worker.
- Do not accept stop as successful while a compute remains outstanding.
- Do not rely on operating-system scheduling order or timing sleeps to define state correctness.
- Do not expose private state variables as the test seam; test only records, process liveness/exit code, shared buffers, and public validators.

**Expected result:**

- The only successful lifecycle is `STARTING → READY → COMPUTING → READY` repeated serially as needed, followed by `READY → STOPPING → STOPPED`.

**Verification:**

- Through real spawn, prove one ready, one compute/result, another legal compute/result, one stop/stopped, and exit status `0`.
- Send malformed, wrong-version, wrong-worker, wrong-shard, and illegal record types after ready and require one sanitized failure plus exit status `1`.
- Send two compute commands before consuming the first result and require the no-pipelining failure path with no successful commit marker for the outstanding computation.

## Step 4 — Compute assigned shards and publish only committed owned gradients

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — compute handler, assigned gradient map, zero-before-compute operation, direct shard-calculation call, finite validation, shared-memory copy, and result construction.
- `tests/test_transformer_worker.py` — tiny deterministic round trip, ownership sentinels, empty shard, and commit-marker assertions.

**Purpose:**

Bridge the completed Ticket 016 numerical boundary into Request-Scoped Shared Memory without changing mathematical behavior or allowing partial/unowned data to be treated as committed.

**Actions:**

- Process assigned Logical Training Shards strictly in ascending shard-ID order regardless of startup tuple construction order; preferably require canonical ascending order during startup and iterate that immutable tuple directly.
- Immediately before calculating each shard, fill that shard's complete canonical shared gradient block with exact `np.float32(0.0)`.
- Call the existing `calculate_logical_training_shard()` with the reconstructed immutable shard/sequences and read-only weight views.
- Validate that the returned result has the expected shard ID, canonical layout, exact `float32` dtype, exact canonical length, C-contiguous storage, finite loss, and entirely finite gradient.
- Copy the complete returned gradient into only the matching assigned shared block using one explicit exact-dtype copy operation. Never expose a view or destination for an unassigned shard.
- Preserve Ticket 016 empty-shard behavior: the block remains all zero and the aligned loss is exactly `0.0`.
- Accumulate only the small ordered tuple of per-shard Python losses for the result record; do not reduce, average, or combine gradients in the worker.
- After every assigned shard has been calculated, validated, and copied, revalidate all owned published blocks and losses for finiteness.
- Send exactly one matching `ResultMessage` only after complete publication. Treat that record as the sole commit marker authorizing the future parent supervisor to read those shared buffers.
- If any calculation, layout, copy, ownership, finiteness, or queued-command check fails, do not send a `ResultMessage`.

**Guardrails:**

- Do not copy weights into a worker-owned model, modify weights, write Adam state, reduce gradients, average shard loss, or apply an optimizer update.
- Do not calculate an unassigned shard even if its metadata is present in startup data.
- Do not serialize a gradient, parameter array, or complete numerical result through the pipe.
- Do not send one result per shard; send one result per matching compute command after every assigned shard is complete.
- Do not replace the existing shard calculator with duplicated forward/backward code.

**Expected result:**

- Parent-visible weight bytes remain unchanged, only assigned gradient blocks change, unassigned sentinel blocks remain byte-for-byte unchanged, and one valid result record commits the complete assigned publication.

**Verification:**

- Build a tiny deterministic one-layer fixture with parent-created weight and gradient blocks, one or more assigned shards, and at least one unassigned gradient block prefilled with a nonzero sentinel.
- Calculate the same assigned shard(s) directly through the completed public Ticket 016 operation in the parent test process and compare each committed shared gradient/loss to that existing boundary. This comparison verifies process transport and publication rather than re-testing the mathematical formulas with a second independent fixture.
- Snapshot parent-visible weight bytes before spawn and assert exact equality after ready, compute, stop, and failure cases.
- Assert unassigned and excess-capacity sentinel bytes remain unchanged.
- Include one empty-shard startup fixture and assert exact zero loss and all-zero canonical gradient after the result commit marker.

## Step 5 — Sanitize failures and guarantee close-only worker cleanup with exact exit codes

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py` — controlled failure mapper, send-once guard, internal logging boundary, worker `finally`, and explicit failure exit.
- `tests/test_transformer_worker.py` — failure privacy, exit-code, close-versus-unlink, and cleanup tests.

**Purpose:**

Make pipe records safe for future route consumption while preserving full internal diagnostics and ensuring every worker outcome has an independent process-exit integrity signal.

**Actions:**

- Map each controlled startup, command/state, compute/non-finite, and shutdown failure to one closed failure phase and code.
- Send at most one `FailureMessage` when the connection and validated worker identity permit it.
- Include only protocol version, worker index, enum phase, enum code, optional epoch, and shard IDs in the failure record.
- Never derive phase/code strings from exception messages.
- Log the complete exception internally with standard Python logging, but do not send exception text, type names, tracebacks, paths, shared-memory names, model values, losses, gradients, or array summaries.
- After a controlled failure, close all attached resources and the connection, then exit with status `1`, preferably through an explicit no-traceback `SystemExit(1)` after the failure record is sent.
- On successful stop, send stopped, release arrays/views/memoryviews, close all attached `SharedMemory` handles and the pipe endpoint in `finally`, and return normally for exit status `0`.
- Attempt every worker-owned close even if an earlier close fails. Log secondary cleanup failures without replacing an already selected primary failure record.
- Never call `unlink()` in production worker code.
- Ensure no exported NumPy view remains alive when its shared-memory handle is closed, preventing `BufferError` from accidental buffer ownership.

**Guardrails:**

- Do not catch and silently convert parent termination or kill into successful stopped behavior.
- Do not include generic exception `repr()` or `str()` in any dataclass field.
- Do not emit a second failure if sending the first failure or closing a later resource fails.
- Do not use exit code alone as the failure protocol, and do not treat a failure record alone as proof that the process exited correctly.

**Expected result:**

- Controlled failures yield one sanitized record when possible and exit `1`; cooperative stop yields one stopped record and exit `0`; all worker attachments are closed and remain parent-owned/unlinked until parent cleanup.

**Verification:**

- Force startup and compute failures using secret markers in invalid resource names or exception-producing input, then recursively inspect the received failure record and its `repr()` to prove the marker, name, path, traceback words, model state, losses, and gradients are absent.
- Join every process and assert exact exit code `1` for controlled failures and `0` for cooperative stop.
- After worker exit, reopen each still-parent-owned block by its runtime name, proving the worker did not unlink it; then close and unlink from the parent fixture.
- Run the same failure case more than once and assert no reused worker state, duplicate failure record, or stale connection message appears.

## Step 6 — Add exact protocol and validator tests without process mocks

**Files and symbols:**

- `tests/test_transformer_worker.py` — protocol structure, pickle, strict validation, privacy, and parent-commit validation tests.

**Purpose:**

Protect the stable public control surface cheaply before invoking slower operating-system processes for lifecycle evidence.

**Actions:**

- Assert the seven exact top-level dataclass types, field order, frozen/slotted behavior, version default or required field behavior, and absence of `__dict__`.
- Assert exact closed enum member sets and stable string values.
- Round-trip every valid record with standard pickle and preserve exact equality and type.
- Assert valid records contain no mutable list, dict, set, bytearray, memoryview, NumPy scalar, NumPy array, exception, path object, or arbitrary class instance.
- Exercise worker-side startup/command validators and parent-side ready/result/failure/stopped validators with exact valid records.
- Parameterize wrong class, wrong version, Boolean-as-integer, negative worker/epoch, mismatched identity, duplicate/unsorted/out-of-range shard IDs, mismatched loss count, NaN/infinite loss, malformed sequence tuple, wrong dtype marker, and canonical-count mismatch.
- Prove a similar-looking test dataclass or dictionary is rejected even when it has matching field names.
- Prove parent result validation checks exact expected epoch and assigned shards before a shared buffer could be considered committed.
- Keep protocol unit tests independent of generated shared-memory names and operating-system scheduling.

**Guardrails:**

- Do not test private loop variables or exact private helper decomposition.
- Do not use monkeypatching as a substitute for the real-spawn tests in later steps.
- Do not make protocol constructors auto-correct, sort, coerce, or normalize malformed data.

**Expected result:**

- The protocol contract fails fast and deterministically before process tests, while retaining the ability to send deliberately malformed objects across a pipe for worker rejection tests.

**Verification:**

- Run the protocol-only subset by test name before implementing the process loop, then retain it in the complete focused worker suite.

## Step 7 — Prove one complete real-spawn shared-memory round trip

**Files and symbols:**

- `tests/test_transformer_worker.py` — parent-owned shared-memory fixture, local spawn context, duplex pipe, production worker target, tiny deterministic computation, stop, join, and cleanup.

**Purpose:**

Prove the exact Windows-compatible seam that mock-only tests cannot cover: module importability, spawn pickling, connection communication, attachment by name, shared gradient publication, commit markers, exit codes, and cleanup.

**Actions:**

- Obtain a local context with `multiprocessing.get_context("spawn")`; never call global `set_start_method()`.
- Create one dedicated `context.Pipe(duplex=True)` and one non-daemonic `context.Process` targeting the top-level production worker function.
- Create parent-owned SharedMemory blocks for one canonical flat weight array and the test's assigned and unassigned gradient arrays.
- Initialize or construct one finite one-layer weight fixture through existing Transformer public APIs, copy it into parent shared memory, and retain an exact byte snapshot.
- Construct a tiny immutable Training Sequence/shard startup payload using primitive tuples. Keep the model and sequence count minimal while exercising a real non-empty gradient.
- Start the worker and close the parent's duplicate copy of the child connection endpoint immediately after successful `start()`.
- Receive and validate one exact ready record.
- Send one matching compute command; do not read any gradient buffer as committed until one matching result record has passed the parent validator.
- After validation, compare owned gradients and aligned losses to direct Ticket 016 calculations, assert finite complete publication, and assert weights and unassigned blocks are unchanged.
- Send one matching stop command, receive and validate exactly one stopped record, join the process, and assert exit `0`.
- Close the parent connection, reopen blocks to prove worker close-only behavior when useful, and close/unlink every parent-owned shared-memory block in a non-short-circuiting fixture `finally`.
- Close the `Process` object after join to release parent process resources.

**Guardrails:**

- Do not use `fork`, the platform default implicitly, ProcessPoolExecutor, Pool, Queue, Manager, socket, temporary file, or a fake connection.
- Do not assert wall-clock duration, PID value, generated shared-memory name format, scheduling order, or maximum-model throughput.
- Do not leave cleanup to Python interpreter shutdown or the resource tracker.
- Do not hide a worker failure behind a test timeout; include bounded parent receives/joins and report the worker record and exit code on assertion failure without leaking those details into production protocol.

**Expected result:**

- Ordinary pytest proves one complete production worker lifecycle through real `spawn`, exact records, shared-memory publication, cooperative stop, exit `0`, and parent-owned cleanup.

**Verification:**

- Run the single happy-path real-spawn test repeatedly on Windows and at least once under the complete suite to expose import or leaked-resource problems.

## Step 8 — Cover real-spawn negative paths, no-pipelining, and repeated creation

**Files and symbols:**

- `tests/test_transformer_worker.py` — parameterized real-spawn failure/lifecycle cases.

**Purpose:**

Prove that invalid startup, corrupt commands, non-finite computation, illegal sequencing, and repeated process creation fail safely rather than publishing stale or unauthorized numerical state.

**Actions:**

- Add bounded real-spawn cases for:
  - undersized weight memory;
  - undersized assigned gradient memory;
  - invalid canonical count or unsupported layer layout;
  - wrong dtype marker;
  - duplicate, unsorted, missing, out-of-range, or mismatched assigned shard ownership;
  - malformed record class and wrong protocol version;
  - wrong worker identity, epoch, or shard tuple;
  - a second outstanding compute command;
  - an illegal worker-originated record sent as a parent command;
  - non-finite compute input/output, produced deterministically by corrupting parent-visible shared weights after ready or another controlled tiny fixture;
  - cooperative stop before any compute;
  - two legal serial compute commands with distinct epochs;
  - worker creation, compute, stop, cleanup, and recreation repeated at least twice with fresh process-local state.
- In every failure case, assert no valid result commit marker, exactly one sanitized failure when possible, exit status `1`, unchanged parent weights, no unassigned gradient write, no worker unlink, and successful parent cleanup.
- In the non-finite case, prefill owned gradient blocks with distinguishable sentinels and assert zero-before-compute occurred where the state reached computation, but no result authorized the parent to consume a partial block.
- In repeated creation, use fresh pipes/process objects and verify no stale message, state, handle, or outstanding epoch survives from the prior worker.
- Keep timeouts generous enough for Windows spawn startup but bounded; do not assert exact elapsed time.

**Guardrails:**

- Do not patch private worker functions inside a child process as the main evidence.
- Do not depend on a race to deliver the second compute; send commands back-to-back and rely on the explicit connection-poll no-pipelining checks.
- Do not interpret a timeout as the expected failure. Every controlled case should end with a record when possible and an observed process exit.
- Do not add application-level worker supervision, terminate/kill escalation, connection waiting across workers, or route deadlines to make these tests pass.

**Expected result:**

- Every specified corruption path terminates deterministically, preserves ownership/privacy, and leaves no state that affects a later worker.

**Verification:**

- Run the complete focused worker suite multiple times locally, then include it in the full project verification without a special slow-test-only marker.

## Step 9 — Preserve scope and finalize implementation verification

**Files and symbols:**

- `src/how_llms_work/ml/transformer_worker.py`
- `tests/test_transformer_worker.py`
- conditional files only if live-signature inspection proves a minimal compatibility seam is unavoidable.

**Purpose:**

Ensure the worker tracer bullet is complete without absorbing Ticket 020+ supervision, route, timeout, persistence, or frontend responsibilities.

**Actions:**

- Re-run the current user-reported baseline before editing and record its actual result.
- After implementation, run protocol-only, happy-path spawn, negative spawn, affected Transformer, formatting, full pytest, Ruff, and strict-mypy checks.
- Inspect `git diff --check`, `git status --short`, and a path-limited diff.
- Confirm no generated shared-memory names, temporary files, model snapshots, logs, cache entries, coverage files, or runtime artifacts are included in the change.
- Confirm every acceptance criterion maps to a public test, a real-spawn test, or an explicit static ownership invariant.
- Confirm no global multiprocessing start method, global worker pool, Queue, Manager, socket, file IPC, process supervisor, HTTP route, SSE, persistence, or dependency change appears in the diff.

**Guardrails:**

- Do not broaden the implementation merely to prepare Ticket 020 parent orchestration.
- Do not refactor completed Transformer numerical code unless a live signature makes one minimal destination/input adaptation unavoidable; document and test any such conditional change separately.
- Do not format unrelated files.
- Do not create a commit until every required verification command has passed in the implementation session.

**Expected result:**

- A small, reviewable worker module and one focused test module satisfy Ticket 019 while leaving every completed Learning Demo and later Phase 5 concern unchanged.

**Verification:**

- Compare the final changed-file list with the expected-file section below and reject unrelated churn before commit.

## Focused verification plan

Run from the backend directory in Windows PowerShell:

```powershell
poetry run pytest tests/test_transformer_worker.py -q
```

Expected result:

- Protocol structure and validation tests pass.
- Real local-spawn happy-path and failure-path tests pass.
- Every spawned process is joined and closed.
- Every parent-owned shared-memory block is closed and unlinked by test cleanup.
- No resource-tracker, leaked shared-memory, unclosed connection, or unclosed process warning is emitted.

Run affected Transformer regressions:

```powershell
poetry run pytest `
    tests/test_transformer.py `
    tests/test_transformer_math.py `
    tests/test_transformer_training.py `
    tests/test_transformer_completion.py `
    tests/test_transformer_worker.py `
    -q
```

Expected result:

- Existing preprocessing, layout, numerical, parent-training, and completion behavior remains unchanged while the new worker boundary passes.

Check changed-file formatting and linting:

```powershell
poetry run black --check `
    src/how_llms_work/ml/transformer_worker.py `
    tests/test_transformer_worker.py

poetry run ruff check `
    src/how_llms_work/ml/transformer_worker.py `
    tests/test_transformer_worker.py

poetry run mypy src
```

Expected result:

- Changed Python files satisfy Black and Ruff.
- Strict mypy reports no issues in `src`.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run black --check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Black reports no formatting changes required.
- Strict mypy reports no issues.

Final repository inspection:

```powershell
git diff --check
git status --short
git diff -- `
    src/how_llms_work/ml/transformer_worker.py `
    tests/test_transformer_worker.py
```

Expected result:

- No whitespace errors.
- Only intended source/test files and any explicitly justified conditional file are changed.
- No runtime shared-memory artifact, cache, model file, dependency file, route, frontend file, or unrelated refactor is present.

## Manual acceptance checklist

- [ ] `WorkerStartupConfig`, `ReadyMessage`, `ComputeMessage`, `ResultMessage`, `FailureMessage`, `StopMessage`, and `StoppedMessage` are top-level frozen slotted dataclasses with protocol version `1`.
- [ ] Protocol values are restricted to primitives, closed enums, `None`, and immutable tuples; no NumPy array or mutable container crosses the pipe.
- [ ] Every worker-originated record identifies the exact worker index.
- [ ] Parent validators reject wrong class, version, worker, epoch, assigned shards, tuple lengths, and non-finite losses before treating a result as committed.
- [ ] Failure phase/code enums are closed and failure records contain no exception text, traceback, path, shared-memory name, model value, gradient, loss, or numerical array.
- [ ] The worker follows only `STARTING → READY → COMPUTING → READY → STOPPING → STOPPED`.
- [ ] A queued second compute while one is outstanding is rejected and produces no successful result commit marker.
- [ ] Ready is sent only after every assigned attachment, capacity, canonical-count, dtype, C-contiguity, view, and ownership check succeeds.
- [ ] Worker arrays are limited to the exact canonical range even when shared-memory capacity is larger.
- [ ] Worker weight storage and every semantic weight view are non-writeable.
- [ ] A worker-side attempted weight mutation raises and parent-visible weight bytes remain unchanged.
- [ ] The worker attaches and writes only assigned shard-gradient blocks.
- [ ] Unassigned and excess-capacity sentinels remain byte-for-byte unchanged.
- [ ] Each assigned gradient block is zeroed immediately before its shard calculation.
- [ ] Assigned shard IDs are processed in ascending order.
- [ ] An empty shard commits exact zero loss and an all-zero canonical gradient.
- [ ] One result record is sent only after all assigned gradients/losses are complete and finite.
- [ ] No shared gradient is considered readable before parent validation of the matching result commit marker.
- [ ] Controlled startup, command, state, compute, non-finite, and shutdown failures send at most one sanitized failure when possible and exit `1`.
- [ ] Cooperative stop sends exactly one stopped record, closes worker attachments in `finally`, and exits `0`.
- [ ] Workers close but never unlink shared-memory blocks; parent fixtures remain the creator/unlink owner.
- [ ] Real `multiprocessing.get_context("spawn")`, a real non-daemonic process, and one dedicated duplex pipe are used in ordinary pytest.
- [ ] Queue, Manager, Pool, ProcessPoolExecutor, socket, file IPC, global start-method mutation, and a global worker pool are absent.
- [ ] Repeated worker creation proves no state, handle, message, epoch, or shared buffer is reused.
- [ ] Existing Transformer, Word2Vec, XOR, BPE, Simple Chat, route, persistence, and frontend behavior remains unchanged.
- [ ] No multi-worker supervision, HTTP orchestration, route deadline, SSE, persistence, dependency, or lockfile work was added.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer_worker.py
tests/test_transformer_worker.py
```

Conditionally changed only if live inspection proves the completed Ticket 016 public operation cannot consume reconstructed immutable startup data or publish through a caller-owned canonical destination without duplicating numerical logic:

```text
src/how_llms_work/ml/transformer.py
tests/test_transformer_math.py
```

The default expectation is **no `transformer.py` change**. Prefer calculating through the existing public shard operation and copying its fully validated result into the assigned shared gradient block.

Optional only if a compact, independently fixed worker publication fixture is materially clearer than constructing the tiny fixture in tests:

```text
tests/fixtures/transformer_worker_reference.json
```

The default expectation is **no new numerical fixture** because Ticket 016 already proves shard math; Ticket 019 can compare spawned publication against the same completed public operation to test process transport and ownership.

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
tests/test_word2vec.py
tests/test_word2vec_training.py
tests/test_word2vec_results.py
tests/fixtures/math_utils_reference.json
tests/fixtures/matrix_reference.json
tests/fixtures/transformer_preprocessing_reference.json
tests/fixtures/transformer_layout_initialization_reference.json
tests/fixtures/transformer_forward_backward_reference.json
tests/fixtures/transformer_training_reference.json
tests/fixtures/transformer_completion_reference.json
tests/fixtures/word2vec_preprocessing_reference.json
tests/fixtures/word2vec_training_reference.json
tests/fixtures/word2vec_results_reference.json
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
```

## Risk notes and safeguards

1. **Risk:** The process target or one protocol type is nested, locally defined, or otherwise not importable under Windows spawn.
   - **Safeguard:** Define every record, enum, validator, and process target at module top level and prove a real `spawn` round trip in ordinary pytest.

2. **Risk:** Calling the global multiprocessing start-method setter changes FastAPI or test behavior.
   - **Safeguard:** Use only `multiprocessing.get_context("spawn")` in parent test/orchestration code and never call `set_start_method()`.

3. **Risk:** The worker duplicates the canonical parameter offset table and drifts from Ticket 015.
   - **Safeguard:** Rebuild and use the existing `TransformerParameterLayout` and semantic view builder as the only offset authority.

4. **Risk:** Startup accepts a larger or differently described layout because capacity happens to be sufficient.
   - **Safeguard:** Validate layer count, declared canonical count, dtype marker, exact reconstructed layout, and exact view shapes separately from minimum capacity.

5. **Risk:** Shared-memory capacity greater than the logical model size exposes unrelated bytes to NumPy operations.
   - **Safeguard:** Slice the buffer to the exact canonical byte range before constructing arrays and protect excess bytes with sentinels.

6. **Risk:** Weight views remain writeable even though the flat base array was marked read-only too late.
   - **Safeguard:** Set the base array read-only before semantic view construction, assert every view flag, perform an idempotent rejected write probe, and compare parent-visible bytes.

7. **Risk:** The read-only probe accidentally changes a weight.
   - **Safeguard:** Snapshot the selected coordinate/bytes, attempt an assignment of the same value only after `writeable=False`, require the expected exception, and verify exact equality.

8. **Risk:** A worker receives or attaches another worker's gradient block.
   - **Safeguard:** Put only aligned assigned names in startup configuration, validate one-to-one ownership, and store no unassigned gradient handle or array in worker state.

9. **Risk:** Stale gradients from a prior epoch survive when computation fails midway.
   - **Safeguard:** Zero each complete assigned block immediately before that shard and send no result commit marker after any failure.

10. **Risk:** A result is sent after one shard but before another assigned shard is complete.
    - **Safeguard:** Collect losses locally and construct/send exactly one result only after all assigned buffers pass final finiteness checks.

11. **Risk:** Worker completion order later affects reduction order.
    - **Safeguard:** Result records retain exact shard IDs and aligned losses; workers process ascending IDs, while parent Ordered Gradient Reduction remains outside this ticket and fixed at `0 → 1 → 2 → 3`.

12. **Risk:** A second compute is silently queued and begins after the first, violating no-pipelining semantics.
    - **Safeguard:** Poll the dedicated connection during compute and immediately before commit; treat any pending command as an outstanding-command state failure and add a back-to-back compute test.

13. **Risk:** A malformed object with matching field names is accepted.
    - **Safeguard:** Require exact dataclass classes and exact field values, not duck typing or dictionary keys.

14. **Risk:** Python treats `True` as integer `1` in worker identity, epoch, layer, count, or shard fields.
    - **Safeguard:** Use exact `type(value) is int`-style runtime checks at protocol trust boundaries and parameterize Boolean rejection.

15. **Risk:** Failure records leak a secret through exception text, `repr`, path, shared-memory name, or numerical value.
    - **Safeguard:** Map exceptions only to closed phase/code enums, construct records from validated context fields, and recursively inspect records using secret-marker tests.

16. **Risk:** Sending a failure fails and triggers a second failure or hides cleanup.
    - **Safeguard:** Use a send-once guard, preserve the primary failure classification, log secondary issues, and continue non-short-circuiting close attempts.

17. **Risk:** NumPy arrays or memoryviews remain exported when `SharedMemory.close()` runs, causing `BufferError` and leaked handles.
    - **Safeguard:** Drop semantic views, flat arrays, and sliced memoryviews before closing handles; test repeated process creation and clean parent unlinking.

18. **Risk:** A worker unlinks a block and races with its parent owner.
    - **Safeguard:** Keep every `unlink()` call out of `transformer_worker.py` and prove names remain reopenable after worker exit.

19. **Risk:** Process exit code and pipe message disagree but tests check only one signal.
    - **Safeguard:** Assert both the exact terminal record and exact joined exit code for success and controlled failure.

20. **Risk:** Real-spawn tests hang indefinitely on a defect.
    - **Safeguard:** Use bounded `poll`/receive and join timeouts, deterministic cleanup escalation only in test `finally`, and diagnostic assertions that report process state without changing production behavior.

21. **Risk:** Tests pass on fork but fail on Windows spawn.
    - **Safeguard:** Explicitly obtain the `spawn` context even on platforms where another method is default and use the production top-level target.

22. **Risk:** Tests become circular by using the same shard calculator as both worker implementation and expected numerical oracle.
    - **Safeguard:** Treat Ticket 016 as the already-approved mathematical oracle; Ticket 019 tests compare direct versus spawned publication only to verify IPC, ownership, commit, and lifecycle behavior, not to re-prove formulas.

23. **Risk:** A full reference shard makes ordinary pytest too slow.
    - **Safeguard:** Pass compact immutable primitive-tuple Training Sequence data through startup and use one tiny deterministic one-layer fixture; maximum-model endurance is explicitly out of scope.

24. **Risk:** Startup payload design accidentally serializes NumPy arrays or mutable preprocessing objects.
    - **Safeguard:** Convert startup sequence/shard data to recursively immutable primitive tuples and add recursive type assertions before process start.

25. **Risk:** Worker tests assert generated names, PIDs, timing, or scheduling and become platform fragile.
    - **Safeguard:** Assert only records, bytes, ownership, process exit, reopenability, and bounded completion.

26. **Risk:** Ticket 019 expands into parent worker-group supervision, process-count selection, deadlines, route cancellation, or cleanup escalation.
    - **Safeguard:** Restrict production changes to one worker module; parent multi-process lifecycle remains a later ticket.

27. **Risk:** Ticket 019 moves Adam, reduction, generation, final evaluation, or persistence into the child.
    - **Safeguard:** Import only the canonical layout/view and shard-calculation boundaries needed by the worker; reject unrelated Transformer state in the final diff.

28. **Risk:** A dependency or lockfile is changed for standard-library process functionality.
    - **Safeguard:** Use only Python 3.12 standard library plus existing NumPy and leave project metadata unchanged.

29. **Risk:** Broad formatting or refactoring obscures the process change.
    - **Safeguard:** Format only changed files, run `git diff --check`, and reject unrelated churn before commit.

30. **Risk:** User-reported baseline is mistaken for current-session verification.
    - **Safeguard:** Re-run the baseline in `implement-prompt` before editing and report actual output honestly.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Compute Transformer shards through spawned workers
```

Commit body should mention:

- top-level frozen slotted protocol version `1` records and closed worker/failure enums;
- exact parent/worker validation of record type, identity, epoch, assigned shards, canonical count, and finite losses;
- one top-level Windows-spawn-compatible worker entry point and one dedicated duplex pipe;
- canonical shared-memory attachment with exact-range `float32` C-contiguous views;
- worker-local non-writeable weights and rejected mutation with unchanged parent bytes;
- assigned-only gradient ownership, zero-before-compute, ascending shard processing, and empty-shard behavior;
- one finite matching result commit marker after complete publication;
- one-outstanding-compute state enforcement;
- sanitized send-once failure records, cooperative stopped records, and exit codes `1`/`0`;
- worker close-only cleanup and parent-only unlink ownership;
- real-spawn happy, malformed, capacity/layout, non-finite, illegal-state, privacy, stop, and repeated-creation tests;
- no Queue, Manager, pool, global start-method change, multi-worker supervision, route, SSE, persistence, frontend, dependency, or lockfile work;
- the exact focused and full verification commands actually executed and their observed results.

Do not create a commit during `to-plan-prompt`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan019.md`;
- `019-compute-logical-shards-through-a-spawn-safe-worker-protocol.md`;
- completed blocker Ticket 016 or the current public forward/backward/Logical Training Shard implementation and tests as equivalent evidence;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- the latest `py_llm_pipeline_explorer_file_structure.md` export created after Ticket 018;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the live repository again, establish its own baseline before editing, preserve user changes, implement only Ticket 019, freeze the exact protocol before the process loop, reuse the completed canonical layout and shard-calculation boundaries, create real local-spawn shared-memory evidence in ordinary pytest, verify privacy/ownership/commit/exit/cleanup behavior, run focused and full verification, report actual command outcomes honestly, inspect final scope, and create the implementation commit only after all required checks pass.
