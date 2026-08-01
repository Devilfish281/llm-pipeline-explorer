---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "026"
source_work_item: 026-stop-and-clean-saved-model-generation-safely.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure.md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 026: Stop and Clean Saved-Model Generation Safely

## Initial checklist

- Confirm Ticket 026 is the only selected work item and that its Ticket 024 blocker is satisfied by the existing public `POST /load-transformer` route, strict Saved Transformer Model loading, prompt preparation, deterministic generation, shared Transformer request slot, and `loaded → result → done` stream. Treat completed Ticket 025 latest-valid selection as current prerequisite behavior that must remain intact.
- Treat the latest supplied `py_llm_pipeline_explorer_file_structure.md`, created after Ticket 025, as the current Python Backend source of truth. Re-inspect the live repository before implementation and do not let older exports, plans, snippets, or the historical TypeScript backend override current Python behavior.
- Preserve the existing process-local nonblocking Transformer request slot and exact overlap response while hardening only the Saved Transformer Generation Run lifecycle required by Ticket 026.
- Keep model selection, parsing, strict validation, parameter materialization, prompt tokenization, and generation in same-process helper threads, never in a child process, Request-Scoped Worker Group, process pool, queue, manager, pipe, or shared-memory region.
- Add one absolute monotonic five-minute generation deadline, cooperative stopping between token calculations, active-helper draining, request-state disposal, and slot-release-last behavior without changing deterministic generation mathematics or successful event payloads.
- Build failure-first route-level HTTP/SSE and lifecycle tests around controlled clocks, disconnect/cancellation observers, blocking helpers, the existing exact SSE parser, and observable slot ownership rather than private thread, task, executor, lock, or polling implementation details.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing it as independently verified in this planning session.
- Finish implementation with focused lifecycle tests, affected Transformer regressions, the complete backend suite, Ruff lint and format checks, strict mypy, and a scope-only Git diff review.

## Source-of-truth hierarchy

1. The user's latest explicit direction: plan Ticket 026 only, convert the approved behavior into the Python Backend, and use the supplied current project exports as evidence.
2. `026-stop-and-clean-saved-model-generation-safely.md` for immediate scope, exact acceptance wording, approved test seam, blocker, constraints, and exclusions.
3. The latest supplied `py_llm_pipeline_explorer_file_structure.md`, created `2026-07-31T19:57:18Z`, for current source, tests, dependencies, public symbols, typing conventions, and the completed Tickets 024 and 025 implementation.
4. `SPEC.md` for the durable Phase 6 decisions concerning one shared process-local slot, parent-process inference, off-event-loop work, one monotonic five-minute generation deadline, disconnect and cancellation handling, cooperative stopping, sanitized errors, stateless request ownership, and final cleanup.
5. `0003-load-saved-transformer-models-for-stateless-generation.md` for the accepted separation between fresh-weight Transformer Training Runs and stateless Saved Transformer Generation Runs, including parent-process inference, no model cache, request-local snapshots, and shared process-local mutual exclusion.
6. `CONTEXT.md` for the canonical meanings and boundaries of Saved Transformer Model, Saved Transformer Generation Run, Saved Transformer Event Stream, Transformer Training Run, Request-Scoped Worker Group, and Request-Scoped Shared Memory.
7. The completed current public boundaries in:
   - `src/how_llms_work/routes/train_transformer.py` — `_TRANSFORMER_RUN_SLOT`, `_TRANSFORMER_REQUEST_OVERLAP_DETAIL`, `_run_unbounded_transformer_helper()`, `_raise_if_transformer_client_disconnected()`, `stream_transformer_training()`, `stream_saved_transformer_generation()`, `train_transformer()`, `load_transformer()`, `load_named_transformer_model()`, `load_latest_transformer_model()`, and `LoadedTransformerModelSnapshot`;
   - `src/how_llms_work/ml/transformer.py` — `PreparedSavedTransformerPrompt`, `prepare_saved_transformer_prompt()`, `generate_saved_transformer_text()`, and the existing between-token cooperative cancellation checks;
   - `tests/test_load_transformer_route.py` — strict request validation, exact SSE parser, controlled route dependencies, safe errors, named/latest dispatch, successful event contract, slot-release tests, no-worker tripwires, request isolation, and deterministic integration;
   - `tests/test_train_transformer_route.py` — process-local nonblocking slot, exact overlap wording, training lifecycle, cancellation/helper-draining, worker cleanup, and slot-release-last prior art;
   - `tests/test_transformer_loading.py` — strict named/latest selection, one-read snapshots, no cache, immutable request-owned materialization, path safety, and no fallback;
   - existing Transformer completion and generation tests — seed `42`, latest-sixteen context, stable top-p sampling, exact prompt prefix, finite validation, and sequential request isolation.
8. `llm_works_file_structure.md` only as historical TypeScript behavior evidence for autoregressive token calculation, temperature scaling, top-p nucleus selection, and successful text reconstruction. It is not lifecycle authority and must not reintroduce TypeScript worker threads, cached model loading, host-dependent concurrency, or permissive model validation.
9. Official Python 3.12 `asyncio` documentation, the event-loop monotonic clock contract, Starlette request-disconnection documentation, and FastAPI streaming-response documentation only as technical cross-checks for asynchronous helper observation, cancellation-safe draining, monotonic deadlines, disconnect polling, and streamed chunks.
10. Older Python exports, prior plans, generated caches, production `.data` contents, frontend work reserved for later tickets, and implementation assumptions unsupported by the latest source are non-authoritative.

## Work-item summary

Ticket 026 completes the lifecycle contract for every named and latest Saved Transformer Generation Run without changing the successful generation result or the fresh-weight training system.

The current Python Backend already reserves one process-local nonblocking slot for `POST /train-transformer` and `POST /load-transformer`, rejects overlap immediately with the approved HTTP `429` detail, loads either an exact named model or the newest strictly valid model, creates a fresh request-owned snapshot, prepares the prompt from that snapshot's Vocabulary and Merge Table, runs deterministic generation in the backend process, and emits `loaded → result → done` on success.

The current load stream also routes loading, prompt preparation, and generation through same-process helper offloading. However, the complete generation call is presently wrapped as one helper operation, the stream has no dedicated five-minute generation deadline, and its `finally` path sets cancellation and releases the slot without an explicit lifecycle boundary proving that any already-started helper has completed, all request-owned model/token/sampler/generated-state references have been discarded, and the slot is the final action. Disconnect and task cancellation handling exist in basic form, but Ticket 026 requires deterministic evidence that no later token calculation or successful event can occur after stopping is observed.

The implementation must therefore add route-owned cooperative orchestration around the existing numerical generator. One absolute monotonic deadline begins for the generation phase and is never reset. While an offloaded helper is active, the async route must remain available to observe disconnect, outer-task cancellation, and deadline state. When any stop condition wins, it sets the request-owned cancellation event, allows the token calculation already in progress to finish, prevents the numerical boundary from starting another token, asynchronously drains the helper, emits only the approved deadline error when the stop reason is timeout, emits no later success event for disconnect or cancellation, discards all request-owned state, and releases the shared Transformer slot last.

This is backend lifecycle hardening only. It must not add a waiting queue, application-wide worker pool, machine-wide lock, cross-process coordination, retained loaded-model cache, session state, forceful thread termination, new production dependency, frontend change, or claim that thread offloading supplies multi-core inference.

## Baseline evidence

- **Status:** User-reported.
- **Commands reported:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Reported result:** The user states that pytest and Ruff passed and mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No repository command was executed against the user's live checkout while producing this plan. The supplied source export was inspected as planning evidence only.
- **Implementation rule:** `implement-prompt` must re-inspect the live repository, preserve user changes, run or reconfirm its own baseline before editing, and report only commands actually executed during implementation.

## Current-code observations

### Shared request slot and route boundary

- `train_transformer()` and `load_transformer()` already call the same `_TRANSFORMER_RUN_SLOT.acquire(blocking=False)` after FastAPI/Pydantic request validation has completed.
- Both routes use `_TRANSFORMER_REQUEST_OVERLAP_DETAIL`, whose approved public value is `Another Transformer request is already running.`
- Acquisition is nonblocking. A second valid request receives HTTP `429`; the route does not await, queue, or retry acquisition.
- Training preparation begins only after successful acquisition. Saved-model selection occurs inside the load stream after successful acquisition.
- Route-preparation exceptions currently release the slot before raising a safe HTTP `500` response.
- The slot is a module/process-local object. Current behavior does not provide a machine-wide, multi-process, distributed, or multi-server lock.

### Saved-model selection and request isolation

- `LoadTransformerRequest` already supports required nullable `modelFile`; a string dispatches to `load_named_transformer_model()` and `None` dispatches to `load_latest_transformer_model()`.
- Named selection remains exact-only and never falls back. Latest selection sorts and validates candidates through the completed Ticket 025 trust boundary.
- Both loaders return one fresh `LoadedTransformerModelSnapshot` containing the selected filename, ordered Vocabulary and Merge Table, and independent canonical `float32` parameter storage.
- Existing loader tests already prove one file read per selected request, no cross-request model cache, rereading after disk changes, strict current-format validation, request-owned arrays, immutable returned metadata, and no candidate mutation.

### Blocking-work offloading

- `stream_saved_transformer_generation()` currently invokes `_run_unbounded_transformer_helper()` for:
  - named/latest model selection, enumeration, metadata, read, JSON parse, strict validation, and parameter materialization;
  - prompt preparation and tokenization;
  - the complete deterministic generation call.
- This establishes the intended same-process off-event-loop seam, but Ticket 026 still needs controlled responsiveness evidence and a complete active-helper lifecycle contract.
- Saved-model loading does not call `create_request_scoped_worker_group()` and does not allocate training processes, pipes, queues, managers, or Request-Scoped Shared Memory.

### Current generation and event behavior

- `generate_saved_transformer_text()` already owns a fresh request-seeded Mulberry32 random stream, uses seed `42`, preserves the exact prompt prefix, uses only the latest sixteen IDs for each forward calculation, and checks a supplied cancellation event between generated tokens.
- Existing tests prove deterministic output for identical inputs and independence from earlier calls with different generation settings.
- On successful loading and prompt validation, the route emits one `loaded` event, runs generation, emits one `result`, then emits one empty `done`.
- Model and prompt failures occur before `loaded`. A controlled generation failure after `loaded` maps to the existing safe generic generation error and emits no `result` or `done`.
- There is currently no dedicated timeout outcome or exact deadline error `Saved Transformer generation exceeded its time limit.`

### Current cleanup gap

- The load stream currently creates one request-owned `Event`, catches disconnect/cancellation/failure, and executes `cancellation_event.set()` followed by `_TRANSFORMER_RUN_SLOT.release()` in `finally`.
- The current structure does not explicitly retain and drain an active helper handle in every cancellation path before releasing the slot.
- An outer `asyncio.CancelledError` can cancel the awaiting coroutine while same-process thread work continues unless the orchestration deliberately shields or otherwise retains and awaits that helper completion.
- Current locals such as `loaded_snapshot`, `prepared_prompt`, and `complete_text` are function-scoped and request-local, but Ticket 026 requires final lifecycle evidence that they and any active sampler/token state are discarded before slot release.
- Training has stronger prior art: its tests already verify an active helper is drained, worker cleanup completes, and only then the run slot is released. The load path should reuse the lifecycle principles without creating training resources.

## Acceptance-criteria classification

### Already satisfied and evidenced

- `POST /train-transformer` and `POST /load-transformer` use the same route-owned, process-local, nonblocking Transformer request slot.
- Standard Pydantic request validation occurs before route slot acquisition.
- Overlap rejection is immediate HTTP `429`, uses the approved shared detail, and has no waiting queue.
- The lock is process-local and does not claim machine-wide or distributed coordination.
- Training remains a fresh-weight Transformer Training Run with its existing request, numerical, worker, shared-memory, event, cleanup, and persistence behavior.
- Named and latest model selection, strict current-format validation, prompt preparation, and deterministic request-owned generation are implemented.
- Blocking selection, file work, validation, prompt tokenization, and generation already pass through a same-process helper-offloading seam.
- Saved-model generation does not create a Request-Scoped Worker Group, child process, pipe, queue, manager, or shared memory.
- Successful named and latest requests already emit exactly one `loaded`, one `result`, and one `done`.
- Sequential generation calls already use fresh deterministic random streams and fresh loaded snapshots with no model cache.

### Partially implemented or evidence incomplete

- The route can observe disconnects through the existing helper boundary, but the complete no-late-success and between-token stop behavior is not yet proven under a controlled blocked token calculation.
- `generate_saved_transformer_text()` checks cancellation between tokens, but route task cancellation and disconnect currently need stronger orchestration to retain and drain the active helper before cleanup.
- The slot is released in `finally`, but active-helper draining, explicit request-state disposal, and release-as-the-final-action are not fully established for the load stream.
- Off-event-loop execution exists, but controlled tests still need to prove an unrelated lightweight request remains responsive while file or generation work is deliberately blocked.
- Existing slot-release tests cover several ordinary outcomes, but Ticket 026 requires one complete named/latest/prompt/generation/deadline/disconnect/cancellation matrix and proof that each interrupted request cannot block a later valid Transformer request.

### Not implemented

- One absolute monotonic five-minute deadline measured for the generation phase.
- The exact timeout SSE message `Saved Transformer generation exceeded its time limit.`
- Deadline behavior that allows the active token calculation to finish, begins no later token calculation, emits no `result` or `done`, drains helper work, discards request state, and releases the slot last.
- Complete disconnect and outer-task-cancellation lifecycle evidence with no later successful event.
- A retained active-helper handle and cancellation-safe drain path that survives cancellation of the route's current await.
- Explicit final request-state disposal before slot release.
- Deterministic fake-clock tests for timeout races and no deadline reset.
- Controlled event-loop-responsiveness tests while loading/token calculation is blocked.
- Complete all-outcome slot recovery and no-permanent-blocking evidence.

## Files to inspect before editing

1. `src/how_llms_work/routes/train_transformer.py`
   - `_TRANSFORMER_RUN_SLOT` and `_TRANSFORMER_REQUEST_OVERLAP_DETAIL`;
   - `_run_unbounded_transformer_helper()` and any current polling/disconnect behavior;
   - `_raise_if_transformer_client_disconnected()` and `_TransformerClientDisconnected`;
   - training helper-drain and cleanup patterns in `stream_transformer_training()`;
   - `stream_saved_transformer_generation()`;
   - `train_transformer()` and `load_transformer()` preparation-failure release paths;
   - current safe load, prompt, generation, start, and timeout-related constants.
2. `src/how_llms_work/ml/transformer.py`
   - `PreparedSavedTransformerPrompt`;
   - `prepare_saved_transformer_prompt()`;
   - `generate_saved_transformer_text()`;
   - exact placement of the existing cancellation check relative to each token calculation;
   - request-owned Mulberry32 creation and generated-ID storage;
   - whether a narrow public cooperative-step boundary is genuinely required or whether the existing full generator can satisfy Ticket 026 unchanged when its cancellation event is driven correctly.
3. `tests/test_load_transformer_route.py`
   - exact SSE parser and event assertions;
   - controlled named/latest loaders, prompt preparer, generator, request, slot, and order recorder;
   - existing success, safe failure, overlap, slot release, no-worker, and request-isolation tests;
   - best insertion points for fake monotonic clock, disconnect/cancellation observer, blocking helper, and next-request recovery cases.
4. `tests/test_train_transformer_route.py`
   - `ControlledRunSlot` or equivalent public slot seam;
   - bidirectional overlap tests and exact detail assertions;
   - task-cancellation tests that drain active helpers;
   - worker cleanup and slot-release-last ordering evidence;
   - training regression points if a shared helper abstraction is changed.
5. Existing Transformer generation/completion test module(s)
   - between-token cancellation tests;
   - seed `42`, latest-sixteen context, top-p, prompt prefix, and request-isolation fixtures;
   - whether one focused public test must be added to prove a cancellation event set during a token calculation prevents the following calculation.
6. `tests/test_transformer_loading.py`
   - named/latest strict-loading and no-cache regressions that must remain unchanged;
   - no production `.data` access rule.
7. `src/how_llms_work/sse.py`
   - shared `format_sse()` and `create_sse_response()` contracts; no change expected.
8. `src/how_llms_work/schemas.py` and `src/how_llms_work/main.py`
   - confirm current request and route registration remain sufficient; no change expected.
9. `pyproject.toml`
   - verify existing Python, pytest, pytest-asyncio, FastAPI/Starlette, Ruff, and mypy tooling; no dependency change expected.

## Step 1 — Establish failure-first lifecycle tests around the existing public route

**Files and symbols:**

- `tests/test_load_transformer_route.py` — existing request builder, exact SSE parser, route dependency installer, controlled slot, loaders, prompt preparer, generator, and order recorder.
- `tests/test_train_transformer_route.py` — shared-slot overlap prior art and training regression assertions.
- Conditionally, `tests/test_saved_transformer_lifecycle.py` — only if keeping the complete Ticket 026 matrix in the existing route file would materially obscure its current focused structure.

**Purpose:**

Freeze the externally observable Ticket 026 contract before changing production orchestration. The tests should fail for missing deadline/drain/disposal behavior while preserving existing named/latest success and safe-error evidence.

**Actions:**

- Extend the current controlled route dependencies with narrow public-facing seams for:
  - a monotonic clock or event-loop time source whose value can advance deterministically;
  - a request disconnect observer whose result can change while a helper is active;
  - a blocking load helper and a blocking generation/token helper controlled by `threading.Event` barriers;
  - a request-owned cancellation event observer;
  - an order recorder for externally meaningful milestones such as helper started, stop observed, helper finished, state finalized, and slot released.
- Keep the existing exact SSE parser as the sole event decoder for route tests.
- Add failure-first tests proving:
  - the generation deadline is one absolute `300.0`-second monotonic budget and is not reset by polling or token progress;
  - when the deadline wins during a token calculation, that calculation may finish, no later calculation begins, one exact timeout `error` is emitted, and no `result` or `done` appears;
  - when disconnect wins during a token calculation, the helper drains but no later successful SSE event is emitted;
  - cancelling the async stream-consumer task sets cooperative cancellation, drains the helper, propagates `asyncio.CancelledError`, and emits no late success;
  - helper completion precedes slot release on every stop path;
  - the next valid training or loading request can acquire the slot after each outcome.
- Build an all-outcome table covering at least:
  - named success;
  - latest success;
  - named load failure;
  - no-valid-latest failure;
  - empty prompt;
  - unsupported prompt;
  - overlength prompt;
  - generation failure;
  - deadline;
  - disconnect;
  - task cancellation;
  - response/stream preparation failure where the existing route seam makes this observable.
- Assert successful named/latest streams contain exactly one `loaded`, one `result`, and one `done` with their current exact key sets.
- Assert every interrupted path contains no false `result` or `done`; deadline has exactly its approved `error`; disconnect remains quiet; cancellation propagates after cleanup.
- Add a controlled responsiveness test in which file selection or generation is blocked in same-process helper work while a lightweight endpoint such as `GET /health` still completes. Use barriers and bounded test waits, not exact elapsed-time assertions.
- Add or retain tripwires proving the load route never calls training worker-group creation, process construction, shared-memory allocation, pipe/queue/manager creation, or worker-process label formatting.
- Keep bidirectional overlap evidence:
  - active training rejects loading;
  - active loading rejects training;
  - both responses are immediate `429` with exactly `{"detail":"Another Transformer request is already running."}`;
  - no queued request starts after ownership is released unless the caller explicitly sends a new request.

**Guardrails:**

- Do not wait five real minutes; use a controlled monotonic clock.
- Do not assert exact thread identity, event-loop task names, executor class, lock type, polling frequency, or private helper names.
- Do not make test success depend on scheduler races. Use explicit barriers to establish helper-started and helper-finished states.
- Do not access, replace, or delete real backend `.data` files.
- Do not generate expected model snapshots or event payloads from the production function under test.
- Do not weaken or replace the existing exact named/latest loader tests.

**Expected result:**

- The new deadline, cancellation-safe drain, state-finalization, no-late-success, and responsiveness tests initially identify the precise lifecycle gaps in the current load stream.
- Existing successful named/latest event contract, strict errors, no-worker behavior, and shared-slot behavior remain green.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q -k `
    "deadline or timeout or disconnect or cancellation or drain or responsiveness or slot or overlap"

poetry run pytest tests/test_train_transformer_route.py -q -k `
    "overlap or slot or cancellation or drain or cleanup"
```

If a new focused lifecycle module is created, run it in full before editing production code.

## Step 2 — Make same-process helper execution retainable and drainable

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py` — `_run_unbounded_transformer_helper()` or a narrowly related route-owned helper lifecycle boundary; `stream_saved_transformer_generation()`.
- `tests/test_load_transformer_route.py` — blocking helper, outer-task cancellation, disconnect, and release-order tests.
- `tests/test_train_transformer_route.py` — regression tests if a helper boundary shared with training changes.

**Purpose:**

Ensure cancellation of the async caller cannot abandon an already-running same-process thread while request-owned model state and the only Transformer slot are released.

**Actions:**

- Re-inspect `_run_unbounded_transformer_helper()` in the live repository and choose the smallest implementation that allows the caller to retain the active awaitable/future until helper work has completed.
- Keep helper execution in the framework/standard-library same-process thread offloading already used by the project. Do not introduce a custom global executor or alter global pool size.
- When starting one blocking operation, retain one request-local handle representing its completion.
- While awaiting that handle, allow async orchestration to observe disconnect, deadline, and cancellation without synchronously blocking the event loop.
- If the outer route task is cancelled while helper work is active:
  - set the request-owned cooperative cancellation event;
  - prevent cancellation from discarding the only handle to the still-running helper;
  - await the helper to completion through a cancellation-safe drain path;
  - preserve the original `asyncio.CancelledError` as the primary outcome and re-raise it only after cleanup.
- If disconnect or deadline occurs:
  - set the same request-owned cancellation event;
  - await the active helper asynchronously;
  - do not start another blocking stage or successful emission after the stop condition has been recorded.
- Preserve primary failures and log only secondary drain/cleanup problems without replacing a timeout, disconnect, or cancellation outcome with misleading success.
- Keep ordinary successful helper execution and exception propagation compatible with current loader, prompt, training, persistence, and generation callers.
- Prefer a load-stream-specific lifecycle wrapper if changing the shared helper would risk altering completed training behavior. Share code only when the live implementation proves the lifecycle semantics are truly identical and existing training tests cover the change.

**Guardrails:**

- Do not try to kill or forcibly stop a Python thread.
- Do not call blocking `Future.result()`, `Thread.join()`, or `Event.wait()` directly on the event-loop thread.
- Do not lose the active helper reference when `asyncio.CancelledError` interrupts an await.
- Do not swallow cancellation after cleanup.
- Do not create an application-wide executor, queue, process pool, or worker pool.
- Do not describe same-process thread offloading as multi-core acceleration.

**Expected result:**

- Every started blocking operation has one completion handle retained until it is drained.
- Cancellation, disconnect, and deadline cannot release request resources while the active helper still uses them.
- Unrelated lightweight async request handling remains responsive during controlled blocking work.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q -k `
    "helper and (cancel or disconnect or deadline or drain or responsive)"

poetry run pytest tests/test_train_transformer_route.py -q -k `
    "helper or cancellation or cleanup"
```

## Step 3 — Add one absolute monotonic generation deadline

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py` — new route-owned generation-time-limit constant, timeout safe-message constant, monotonic deadline calculation, and `stream_saved_transformer_generation()` orchestration.
- `tests/test_load_transformer_route.py` — fake monotonic clock and exact timeout event tests.

**Purpose:**

Bound only the saved-model generation phase to five monotonic minutes while preserving already completed selection and prompt validation and allowing the active token calculation to finish cooperatively.

**Actions:**

- Add one route-owned duration constant equal to `300.0` seconds and one exact safe public message:

  ```text
  Saved Transformer generation exceeded its time limit.
  ```

- Use a monotonic source, preferably the running event loop's `time()` boundary or an equivalent injectable route-owned callable that is straightforward to control in tests.
- Calculate one absolute deadline exactly once when the generation phase is about to start. Do not reset it per poll, per token, after `loaded`, or when helper progress occurs.
- Keep model selection and prompt preparation outside this generation budget unless the live specification/source explicitly establishes a different start point. The generation phase begins immediately before the first generated-token calculation is scheduled.
- Orchestrate the generation helper so the route can determine which outcome wins:
  - helper completion;
  - browser disconnect;
  - absolute deadline;
  - outer task cancellation.
- When the deadline is reached:
  - record timeout as the terminal reason;
  - set the request-owned cancellation event;
  - let the token calculation already in progress finish;
  - rely on the numerical boundary's between-token cancellation check to prevent the next token calculation;
  - asynchronously drain the helper;
  - emit exactly one `error` event with the approved timeout message;
  - emit no `result` or `done`.
- After helper completion, re-check terminal state immediately before any `result` or `done` emission so a near-simultaneous stop cannot be followed by success.
- Keep ordinary internal generation failures mapped to the existing safe generic generation error, distinct from timeout.
- Keep disconnect quiet and cancellation propagating after cleanup.
- Define deterministic tie handling for tests: once timeout, disconnect, or cancellation is observed and recorded, later helper success cannot replace it. Avoid exposing internal race details to clients.

**Guardrails:**

- Do not use wall-clock time, `datetime.now()`, file timestamps, or exact elapsed sleeps.
- Do not create a fresh five-minute timeout for each generated token.
- Do not cancel the thread and assume the computation stopped.
- Do not emit the generic generation error for a known deadline outcome.
- Do not emit timeout after a complete successful `result` has already been committed.
- Do not change the request's existing `maxTokens` range or generation mathematics.

**Expected result:**

- Every Saved Transformer Generation Run has one deterministic monotonic generation budget.
- Deadline expiration stops at the approved between-token boundary, drains current work, emits only the timeout error, and leaves no success completion.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q -k `
    "deadline or timeout or monotonic or no_late_success"
```

Expected controlled cases should include expiry before a later token, non-expiry just below the boundary, no budget reset, and timeout after `loaded` with no `result`/`done`.

## Step 4 — Enforce the cooperative between-token stopping boundary

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — `generate_saved_transformer_text()` only if the live implementation's existing cancellation placement is insufficient.
- Existing Transformer generation/completion tests — controlled forward/token-calculation seam and cancellation event.
- `tests/test_load_transformer_route.py` — route-level timeout/disconnect/cancellation integration.

**Purpose:**

Prove that stopping never interrupts an already-started numerical calculation but always prevents the following generated-token calculation and successful completion.

**Actions:**

- First inspect the live `generate_saved_transformer_text()` implementation and retain it unchanged if it already checks the supplied cancellation event immediately before each new token calculation and before successful return.
- Add or strengthen a public numerical test that blocks one forward/token calculation, sets cancellation while it is active, then releases it and proves:
  - that active calculation is allowed to return;
  - no subsequent forward/token calculation begins;
  - the numerical operation returns/raises its existing cancellation outcome rather than a successful complete text;
  - model parameters and caller-owned prompt/vocabulary remain unchanged.
- If the existing cancellation check is only after a later calculation starts, make the smallest mathematical-boundary correction:
  - check before every token calculation;
  - preserve all existing seed, context, logits, temperature, softmax, top-p, sampling, and text-reconstruction behavior;
  - keep cancellation cooperative and request-owned.
- Ensure the route's cancellation event is the same event observed by the numerical generator; do not create an unconnected second flag.
- After the generator finishes because cancellation was requested, let route orchestration map the already-recorded terminal reason rather than treating the numerical cancellation exception as an ordinary generation error.

**Guardrails:**

- Do not split or reorder the numerical operations inside one token calculation merely to create more cancellation points.
- Do not check cancellation inside matrix multiplication or attention loops.
- Do not alter Mulberry32 state consumption on uninterrupted success.
- Do not change exact deterministic fixtures, latest-sixteen context, top-p tie behavior, or prompt reconstruction.
- Do not expose the cancellation event or internal token IDs to clients.

**Expected result:**

- The cancellation contract is exactly between generated-token calculations.
- Uninterrupted requests remain byte-for-byte/event-for-event compatible at public boundaries.

**Verification:**

```powershell
poetry run pytest <live-transformer-generation-test-module> -q -k `
    "saved_generation and (cancel or token or deterministic or context or seed)"

poetry run pytest tests/test_load_transformer_route.py -q -k `
    "deadline or disconnect or cancellation"
```

Replace the placeholder module with the actual live test filename discovered during implementation.

## Step 5 — Centralize the Saved Transformer Generation Run terminal state

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py` — `stream_saved_transformer_generation()` and any small route-owned terminal-reason/state representation.
- `tests/test_load_transformer_route.py` — event-order, no-late-success, error mapping, and state-isolation tests.

**Purpose:**

Make success, load/prompt failure, generation failure, deadline, disconnect, and cancellation mutually exclusive so no interrupted run can accidentally emit `result` or `done`.

**Actions:**

- Keep all mutable lifecycle state request-local. A small private enum/dataclass is acceptable, but tests must assert behavior rather than require its name or shape.
- Track a single terminal reason once stopping is observed. Later helper completion must not overwrite it with success.
- Preserve current pre-generation error mapping:
  - named-load failure → `The saved Transformer model could not be loaded.`;
  - latest no-valid failure → `No valid saved Transformer model was found.`;
  - empty prompt → existing exact empty-prompt message;
  - unsupported text → existing exact unsupported-tokenization message;
  - overlength prompt → existing exact sixteen-token message.
- Preserve `loaded` only after complete model and prompt validation.
- Permit `result` only when:
  - generation helper completed successfully;
  - no deadline, disconnect, cancellation, or internal failure was recorded;
  - the active helper has been drained;
  - complete generated text belongs to this request.
- Permit `done` only immediately after that request's `result`, exactly once, with the existing empty payload.
- Map a known deadline to the new exact timeout error.
- Keep a browser disconnect quiet with no later success or generic error.
- Re-raise task cancellation only after all cleanup.
- Keep ordinary internal generation failure private, log it, emit the existing generic safe generation error, and emit no success.
- Add explicit checks at successful emission boundaries so a stop observed after helper completion but before emission still prevents success.

**Guardrails:**

- Do not expose terminal enums, cancellation tokens, paths, exceptions, numerical values, token IDs, or resource identifiers.
- Do not add token-stream events, training `init`/`epoch` fields, architecture, loss, or sample collection to the load stream.
- Do not send both timeout and generic generation errors for one run.
- Do not emit `done` in a `finally` block.
- Do not turn disconnect or task cancellation into a successful empty stream after swallowing cancellation.

**Expected result:**

- Every run has one unambiguous terminal outcome.
- No stopped or failed run emits false `result` or `done`.
- Successful named and latest runs remain exactly `loaded → result → done`.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q -k `
    "event_order or safe_error or no_late_success or timeout or disconnect or cancellation"
```

## Step 6 — Drain active work, discard request-owned state, and release the slot last

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py` — finalization path of `stream_saved_transformer_generation()`; `load_transformer()` preparation-failure path; shared slot.
- `tests/test_load_transformer_route.py` — lifecycle order recorder, owned-state probes, next-request recovery, and release-on-all-outcomes matrix.
- `tests/test_train_transformer_route.py` — training slot and cleanup regressions.

**Purpose:**

Complete Ticket 026's strongest ownership rule: after any route outcome, no active helper or request-owned numerical state may remain when the shared slot becomes available.

**Actions:**

- Structure finalization so every path attempts the same ordered lifecycle:
  1. record/retain the primary terminal outcome;
  2. set the request-owned cancellation event when stopping or cleanup begins;
  3. asynchronously drain any active helper;
  4. discard references to generated text, prepared prompt/token IDs, loaded snapshot, parameter views/storage, Vocabulary/Merge containers, selected filename, sampler/generation state, and helper handle;
  5. release `_TRANSFORMER_RUN_SLOT` as the final lifecycle action.
- Ensure no `await`, log call that depends on request state, event emission, state mutation, or cleanup operation follows slot release.
- Preserve route-preparation behavior before a stream is returned: if response creation fails after slot acquisition but before stream ownership, release exactly once and return the existing safe start failure.
- Avoid double release between route preparation and stream finalization by making ownership transfer explicit in code structure.
- Preserve the primary outcome if state disposal or secondary cleanup logging encounters a problem; log secondary failures without emitting false success or leaking details.
- Add externally observable tests that use controlled owned objects/barriers to prove:
  - helper finish occurs before slot release;
  - the slot cannot be acquired by a second request while helper work still uses request state;
  - after release, a new valid request receives fresh model, prompt, random stream, cancellation event, selected filename, and generated-token state;
  - failure, timeout, disconnect, or cancellation cannot permanently block a later request.
- Use weak-reference/finalizer probes only if the live request-owned container types support them cleanly and without coupling tests to private structure. Otherwise prove disposal through non-reuse, mutation isolation, barriers, and slot ordering.

**Guardrails:**

- Do not release the slot before active helper completion.
- Do not perform another request-owned operation after release.
- Do not store loaded snapshots, prompts, selected filenames, random generators, cancellation events, or generated tokens in module-level state.
- Do not add a loaded-model cache or retain a “current model.”
- Do not make tests require `del` statements, exact local names, garbage-collection timing, or a particular container implementation.
- Do not double-release the slot when stream setup fails.

**Expected result:**

- Slot release is observably the final lifecycle action.
- Every later request starts from fresh request-owned state.
- No stopped run can leave the process-local Transformer route permanently busy.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q -k `
    "release or final or drain or dispose or isolation or subsequent or recovery"

poetry run pytest tests/test_train_transformer_route.py -q -k `
    "slot or overlap or cleanup"
```

## Step 7 — Preserve training behavior and prove bidirectional shared-slot exclusion

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py` — `_TRANSFORMER_RUN_SLOT`, `_TRANSFORMER_REQUEST_OVERLAP_DETAIL`, `train_transformer()`, `load_transformer()`, and any shared helper touched by Ticket 026.
- `tests/test_train_transformer_route.py` — training request validation, overlap, fresh-weight, event, worker, persistence, cancellation, and cleanup regressions.
- `tests/test_load_transformer_route.py` — load-versus-training exclusion and no-training-resource assertions.

**Purpose:**

Ensure lifecycle hardening does not change the completed Transformer Training Run or weaken the shared nonblocking admission rule.

**Actions:**

- Retain one exact process-local slot object and one approved overlap detail constant for both routes.
- Preserve acquisition after Pydantic validation and before any expensive preparation/selection.
- Retain immediate nonblocking acquisition; do not add retries, conditions, backoff, or a queue.
- Keep the route/API description explicit that ownership is per backend process. Do not imply coordination across Uvicorn workers or servers.
- Re-run or add focused bidirectional tests with controlled active ownership:
  - training owns slot → valid load gets immediate exact `429`;
  - load owns slot → valid training gets immediate exact `429`;
  - invalid requests remain `422` and never touch the slot;
  - no rejected request later starts automatically after slot release.
- If `_run_unbounded_transformer_helper()` is changed, prove training still:
  - creates fresh parameters and one Request-Scoped Worker Group;
  - emits unchanged `init`, `epoch`, and `done` fields and values;
  - preserves Logical Training Shards, worker protocol, Ordered Gradient Reduction, Adam behavior, generation samples, cleanup, and persistence-before-`done`;
  - drains active helpers and releases the slot last.
- Preserve load-route absence of worker processes, shared memory, training labels, training events, and persistence.

**Guardrails:**

- Do not add a machine-wide lock, file lock, Redis lock, database lock, or cross-process coordination.
- Do not alter training request bounds, numerical fixtures, optimizer, workers, shared memory, report schedule, event payloads, saved-model format, or persistence ordering.
- Do not let loading initialize weights, resume training, or use a training worker group.
- Do not introduce a new public endpoint or request field.

**Expected result:**

- Training and loading remain mutually exclusive within one backend process with immediate exact rejection.
- Completed training behavior remains unchanged.

**Verification:**

```powershell
poetry run pytest tests/test_train_transformer_route.py tests/test_load_transformer_route.py -q -k `
    "validation or overlap or slot or fresh or worker or persistence or event"
```

## Step 8 — Complete lifecycle, determinism, and no-resource regression coverage

**Files and symbols:**

- `tests/test_load_transformer_route.py` or the conditionally created lifecycle module.
- Existing Transformer generation/completion tests.
- `tests/test_transformer_loading.py`.
- `tests/test_train_transformer_route.py`.
- Existing worker-group, worker, persistence, and completion regression modules discovered in the live repository.

**Purpose:**

Prove the entire Ticket 026 acceptance surface through stable public behavior and controlled lifecycle seams.

**Actions:**

- Consolidate the final all-outcome matrix and assert for each case:
  - exact client-visible events/status;
  - absence of forbidden success events;
  - active helper drained;
  - slot released exactly once and only last;
  - a later valid Transformer request succeeds;
  - no request-owned state is reused.
- Run named and latest success with distinct controlled snapshots and prove each request's selected filename, prompt, parameters, Vocabulary, Merge Table, and generated text remain aligned.
- Run two sequential identical requests and prove deterministic public text with independent loader calls, independent snapshots, independent cancellation events, and independent generator state.
- Run changed model/settings cases and prove state from an earlier request cannot contaminate later output.
- Assert the timeout error contains only the exact safe message and no path, exception, traceback, task, future, thread, token ID, model array, resource name, or clock value.
- Assert disconnect/cancellation emits no later success even if the blocked helper eventually returns a plausible complete string.
- Retain exact tests for named/latest load errors, prompt errors, generic generation failure, and successful payload key sets.
- Retain no-worker tripwires and verify no process, Request-Scoped Worker Group, pipe, queue, manager, or shared-memory region is created by loading.
- Re-run strict loader tests to prove Ticket 026 does not weaken file safety, one-read snapshots, latest ordering, invalid skipping, no named fallback, or no-cache behavior.
- Re-run deterministic generation tests to prove unchanged seed, context window, top-p, prompt prefix, finite validation, and model immutability.

**Guardrails:**

- Do not assert private implementation decomposition or exact polling count.
- Do not use real maximum-size generation or real five-minute waits in ordinary tests.
- Do not broaden Ticket 026 into frontend parsing/display, worker-count presentation, model management, caching, token streaming, or model repair.
- Do not create fixtures from current production output when independent expected values already exist.

**Expected result:**

- Every Ticket 026 lifecycle outcome is deterministic, privacy-safe, nonblocking at the route level, recoverable, and isolated.
- All completed Tickets 023–025 behavior remains intact.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q
poetry run pytest tests/test_train_transformer_route.py -q
poetry run pytest tests/test_transformer_loading.py -q
poetry run pytest <live-transformer-generation-test-module> -q
```

Also run the actual worker-group, Transformer completion, and persistence modules identified in the live repository if any shared route helper changed.

## Step 9 — Run focused, full, quality, and scope verification

**Files and symbols:**

- All files changed by Ticket 026.
- Repository test and tooling configuration.
- Git status and diff.

**Purpose:**

Prove the implementation is complete, typed, formatted, regression-safe, and restricted to Ticket 026.

**Actions:**

- Run focused lifecycle tests first and correct failures before running broad suites.
- Run affected Transformer route, generation, loading, worker-group, completion, and persistence regressions.
- Run the complete backend pytest suite once focused suites are green.
- Run Ruff lint and Ruff format check exactly as configured by the project.
- Run strict mypy against `src`.
- Inspect `git diff --check`, the complete diff, and `git status --short`.
- Confirm no frontend, schema, route-registration, worker protocol, shared-memory layout, model-format, persistence, dependency, lockfile, `.data`, specification, ADR, ticket, or unrelated source change entered the diff.
- Report actual command output honestly. Do not repeat the user's reported baseline as implementation-session proof.

**Focused verification commands:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q
poetry run pytest tests/test_train_transformer_route.py -q
poetry run pytest tests/test_transformer_loading.py -q
poetry run pytest <live-transformer-generation-test-module> -q
```

If Ticket 026 uses a new focused lifecycle module:

```powershell
poetry run pytest tests/test_saved_transformer_lifecycle.py -q
```

Run any actual live worker-group, completion, and persistence test modules affected by a shared route helper.

**Full verification commands:**

```powershell
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
git diff --check
git status --short
```

**Expected result:**

- All focused and full tests pass.
- Ruff lint and formatting checks pass.
- Strict mypy reports no issues.
- Git diff is whitespace-clean and contains only Ticket 026 lifecycle work.

## Manual validation checklist

Automated controlled tests are the primary evidence for deadline and cancellation. After they pass, perform a practical two-server check only where useful:

- Start the FastAPI backend and Vite frontend using the project's current documented commands.
- Run one valid named `File:` command and confirm the current loaded filename, prompt, and complete generated result still display.
- Run one valid latest `File:` command and confirm the selected actual filename displays and completes normally.
- While one real Transformer training or load request owns the slot, send the other kind of valid request and confirm the immediate exact `429` response; do not wait for queued execution.
- Abort a saved-model browser request and confirm no later completed result appears in that request's UI; then send a new valid Transformer request and confirm it can start.
- Confirm loading never displays `Transformer worker processes` and does not create training-worker output.
- Do not wait five real minutes to validate timeout. The fake-clock test is authoritative for the five-minute boundary.
- Record browser/Vite observations separately from automated test outcomes and do not claim browser proof for behavior that was not observed.

## Expected files changed

### Likely production changes

```text
src/how_llms_work/routes/train_transformer.py
```

### Likely test changes

```text
tests/test_load_transformer_route.py
tests/test_train_transformer_route.py
```

### Conditional changes only if live evidence requires them

```text
src/how_llms_work/ml/transformer.py
<existing Transformer generation/completion test module>
tests/test_saved_transformer_lifecycle.py
```

Conditions:

- Change `ml/transformer.py` only if the current public cancellation check is not located at the required between-token boundary.
- Create a new lifecycle test module only if the existing load-route test file cannot remain coherent with the complete Ticket 026 matrix.
- Change training tests only for shared-slot or shared-helper regressions; do not rewrite established numerical fixtures.

### Files not expected to change

```text
src/how_llms_work/schemas.py
src/how_llms_work/main.py
src/how_llms_work/sse.py
src/how_llms_work/ml/transformer_worker.py
all worker-group/shared-memory protocol modules
all Saved Transformer loading and persistence formats
frontend/**
pyproject.toml
poetry.lock
.data/**
SPEC.md
CONTEXT.md
0003-load-saved-transformer-models-for-stateless-generation.md
026-stop-and-clean-saved-model-generation-safely.md
llm_works_file_structure.md
```

No dependency or lockfile change is expected.

## Risks and safeguards

1. **Risk: outer task cancellation abandons a running thread and releases the slot early.**
   - **Safeguard:** Retain the active helper handle, set cooperative cancellation, use a cancellation-safe asynchronous drain, and re-raise the original cancellation only after helper completion and state disposal.
2. **Risk: the five-minute budget is accidentally reset during polling or after each token.**
   - **Safeguard:** Calculate one absolute monotonic deadline once at generation start and compare all later observations against it.
3. **Risk: deadline is measured with wall-clock time and changes when the system clock moves.**
   - **Safeguard:** Use the event loop's monotonic clock or an equivalent monotonic route seam, controlled by fake-clock tests.
4. **Risk: a helper returns success after timeout/disconnect/cancellation and the route emits late `result`/`done`.**
   - **Safeguard:** Record one terminal reason, re-check it after helper drain and immediately before each successful emission, and never let later helper completion overwrite a stop outcome.
5. **Risk: cancellation is checked inside a token calculation and changes numerical behavior or random consumption.**
   - **Safeguard:** Check only before starting each token calculation and before successful return; allow an active calculation to finish.
6. **Risk: helper draining blocks the event-loop thread.**
   - **Safeguard:** Await the retained completion handle asynchronously; never call thread joins, blocking waits, or synchronous future results on the event loop.
7. **Risk: timeout is mapped to the generic generation failure or emits two errors.**
   - **Safeguard:** Use a dedicated terminal reason and exact timeout message; one terminal event policy per request.
8. **Risk: the slot becomes available while helper work still reads loaded model state.**
   - **Safeguard:** Drain first, discard request-owned state second, release slot last, and prove a second request cannot acquire before the helper-finished barrier.
9. **Risk: explicit cleanup introduces double slot release between route preparation and stream finalization.**
   - **Safeguard:** Make ownership transfer to the stream explicit and cover preparation failure plus every stream outcome with exact acquire/release counts.
10. **Risk: tests overfit a particular executor, task layout, or private helper.**
    - **Safeguard:** Assert HTTP/SSE behavior, barriers, helper completion, no later work, state isolation, and slot availability only.
11. **Risk: same-process offloading is inaccurately described as multi-core inference.**
    - **Safeguard:** Document it only as event-loop responsiveness and cooperative lifecycle isolation.
12. **Risk: lifecycle hardening alters training workers, numerical fixtures, or persistence.**
    - **Safeguard:** Prefer load-specific orchestration unless a shared helper change is demonstrably safe; run all affected training regressions.
13. **Risk: request-local model or sampler state is accidentally cached for convenience.**
    - **Safeguard:** Keep all snapshot, prompt, random, cancellation, filename, and generated-token state local; add sequential mutation/isolation tests.
14. **Risk: a real-time test becomes slow or flaky.**
    - **Safeguard:** Use fake monotonic clocks, deterministic barriers, and bounded waits; never sleep for the production deadline.
15. **Risk: user-reported baseline is presented as tool-verified.**
    - **Safeguard:** Label it user-reported and require `implement-prompt` to run and report its own commands.
16. **Risk: broad formatting or refactoring obscures lifecycle changes.**
    - **Safeguard:** Format only changed files as needed, inspect the full diff, and reject unrelated churn.

## Commit guidance after all checks pass

Do not create a commit during `to-plan-prompt`.

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Stop saved Transformer generation safely
```

The commit body should mention:

- one shared process-local nonblocking Transformer slot and exact immediate `429` overlap response;
- same-process offloading for selection, file work, validation, prompt tokenization, and generation;
- one absolute monotonic five-minute generation deadline;
- cooperative stopping only between generated-token calculations;
- cancellation-safe active-helper draining;
- exact timeout message and absence of false `result`/`done` after interruption;
- request-owned model, prompt, sampler, cancellation, filename, and generated-token isolation;
- drain-before-discard-before-slot-release ordering;
- all-outcome slot recovery and later-request success;
- preserved named/latest loading, deterministic generation, training numerics, workers, shared memory, event payloads, and persistence-before-training-`done`;
- no child process, queue, manager, pipe, shared memory, frontend, dependency, lockfile, model cache, session, or machine-wide-lock change;
- the exact focused, full, Ruff, mypy, and manual commands actually executed and their observed results.

## Handoff to `implement-prompt`

Run `implement-prompt` in a fresh conversation using:

- this `plan026.md`;
- `026-stop-and-clean-saved-model-generation-safely.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0003-load-saved-transformer-models-for-stateless-generation.md`;
- the latest `py_llm_pipeline_explorer_file_structure.md` source export created after Ticket 025;
- the latest `llm_works_file_structure.md` behavior reference;
- the current live repository.

`implement-prompt` must re-inspect the live repository, preserve user changes, establish its own baseline before editing, implement only Ticket 026, create failure-first lifecycle tests, retain the completed named/latest loading and deterministic generation boundaries, use one monotonic generation deadline, observe stop conditions while helper work runs, drain active same-process work before discarding request-owned state, release the process-local slot as the final action, preserve training behavior, run focused and complete verification, report actual outcomes honestly, inspect final scope, and create the implementation commit only after every required check passes.
