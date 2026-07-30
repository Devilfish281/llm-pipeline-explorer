---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: 006
source_work_item: 006-stream-complete-xor-training-runs-through-fastapi.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(11).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 006: Stream Complete XOR Training Runs through FastAPI

## Initial checklist

- Confirm Ticket 006 is the only selected work item and that Tickets 004 and 005 are represented in the latest current code.
- Treat `py_llm_pipeline_explorer_file_structure(11).md` as the current Python code source of truth.
- Preserve the unchanged TypeScript/Vite Frontend Contract and use `llm_works_file_structure.md` only as a behavior reference.
- Reconfirm the user-reported pytest, Ruff, and strict mypy baseline before editing.
- Implement only request validation, route registration, bounded threaded streaming orchestration, disconnect/failure behavior, and route-level tests.
- Reuse the existing Training Run, Saved Weight Snapshot persistence boundary, and shared SSE helpers instead of rewriting them.
- Finish with focused route tests, affected-area tests, the complete pytest suite, Ruff, strict mypy, and a manual browser/proxy check.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript behavior to Python and treat the latest complete Python backend export as the source of truth.
2. `006-stream-complete-xor-training-runs-through-fastapi.md` for required behavior, acceptance criteria, approved test seams, blockers, and scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(11).md` for current implementation, tests, paths, dependencies, and repository conventions.
4. `SPEC.md` and the latest `CONTEXT.md` for durable Phase 3 decisions and the canonical terms Training Run, Epoch Update, Neural Network Event Stream, Training Verdict, and Saved Weight Snapshot.
5. `llm_works_file_structure.md`, especially `src/schemas/neural-net-request.ts` and `src/routes/neural-net/index.ts`, as a compatibility reference only.
6. Older code exports, prior plans, and earlier assumptions are non-authoritative when they conflict with the sources above.

## Work-item summary

Complete the Phase 3 vertical slice by exposing `POST /neural-net` through FastAPI. The endpoint must validate one required model mode and an optional bounded epoch count, create an independent existing `TrainingRun`, advance that iterator one reporting interval at a time in same-process worker threads, emit exact shared-SSE `epoch` events with the requested presentation delays, check for client disconnects between intervals, persist the completed mode-specific Saved Weight Snapshot, and emit exactly one `done` event only after persistence succeeds.

The route must terminate quietly after disconnect, training failure, or persistence failure: no snapshot replacement for incomplete or failed runs, no `done`, no invented SSE `error` event, and no client-visible exception details. Unexpected post-stream failures must be logged through standard Python logging. Existing Health, Simple Chat, BPE, numerical XOR, and persistence behavior must remain unchanged.

## Baseline evidence

- **Status:** User-reported.
- **Command:** `poetry run pytest`
- **Result:** The user reports that all tests passed before planning.
- **Command:** `poetry run ruff check .`
- **Result:** The user reports that Ruff passed before planning.
- **Command:** `poetry run mypy src`
- **Result:** The user reports `Success: no issues found` before planning.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing. None of these commands were tool-verified in this planning session.

## Current code observations from the latest source

- `backend/src/how_llms_work/ml/neural_net.py` already implements Ticket 004. `create_training_run()` creates a fresh NumPy generator and mutable state per request unless a deterministic generator is explicitly supplied by a test.
- `TrainingRun.__next__()` already advances from one reporting boundary to the next rather than completing the entire run in one call. It emits `EpochUpdate` objects at epoch zero, each reference-compatible boundary, and the requested final epoch, followed by one `TrainingResult`.
- `EpochUpdate.to_payload()` already returns exactly `epoch` and `loss`.
- `TrainingResult.to_frontend_payload()` already returns exactly `architecture`, `predictions`, and `verdict`, excluding the Saved Weight Snapshot.
- Existing numerical tests already prove exact architecture labels, verdict strings, prediction order, rounded outputs, deterministic educational contrast, reporting boundaries, and mutable-state isolation.
- `backend/src/how_llms_work/routes/neural_net.py` already implements Ticket 005's mode-specific snapshot persistence, exact JSON serialization, same-directory temporary files, atomic replacement, cleanup, and concurrency behavior.
- Existing persistence tests already prove prior-snapshot preservation on serialization, write, and replacement failure; cleanup behavior; different-mode isolation; and last-successful-finisher-wins behavior for same-mode saves.
- `backend/src/how_llms_work/sse.py` already provides the required `format_sse()` formatter and `create_sse_response()` factory with `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- Existing Simple Chat and BPE route tests establish the repository pattern of exercising routes through `TestClient`, parsing named SSE events, patching the route module's referenced delay operation, and asserting exact headers and payload keys.
- `backend/src/how_llms_work/schemas.py` currently contains only `ChatRequest`; no `NeuralNetRequest` exists.
- `backend/src/how_llms_work/main.py` currently registers only the Simple Chat and BPE routers; `POST /neural-net` is unavailable.
- `backend/src/how_llms_work/routes/neural_net.py` has persistence functions but no `APIRouter`, endpoint, async stream, disconnect check, presentation delay, worker-thread orchestration, or post-stream logging.
- `backend/tests/test_neural_net_route.py` does not exist.
- The current dependency set is sufficient. No package, lockfile, process-pool, executor-capacity, or frontend change is required.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - The public numerical module supports only `single-layer` and `multi-layer` modes.
  - Each Training Run creates independent production initialization and mutable numerical state.
  - Epoch Updates include epoch zero, all reference-compatible reporting boundaries, and the requested final epoch.
  - Epoch payloads contain exactly `epoch` and `loss`.
  - Final result payload construction contains exactly `architecture`, `predictions`, and `verdict`, with four ordered XOR Predictions and no weights.
  - Exact architecture labels, rounded values, Training Verdict strings, and deterministic reference results are covered by `backend/tests/test_neural_net.py`.
  - Mode-specific snapshot selection, exact JSON documents, safe replacement, failure preservation, temporary-file cleanup, and concurrency behavior are covered by `backend/tests/test_neural_net_persistence.py`.
  - Shared SSE formatting, media type, and required headers already exist and are exercised by the Simple Chat and BPE route tests.
  - Health, Simple Chat, and BPE routes are currently registered and tested.
- **Behavior present but evidence incomplete:**
  - `TrainingRun.__next__()` is already the bounded reporting-interval operation needed by the route, but no route currently offloads each call to a worker thread.
  - The persistence boundary already returns only after successful replacement, but no route test proves persistence-before-`done` ordering.
  - Independent Training Runs and persistence temporary files are separately proven, but no route-level test proves that two requests retain isolated orchestration and completion state.
- **Partially implemented:**
  - `backend/src/how_llms_work/routes/neural_net.py` owns the completed persistence boundary but not the HTTP/SSE route portion of Ticket 006.
- **Not implemented:**
  - Dedicated `NeuralNetRequest` validation and defaulting.
  - Router creation and application registration for `POST /neural-net`.
  - Shared SSE response creation for the neural-network route.
  - Same-process worker-thread advancement one reporting interval at a time.
  - Route-level `0.02`-second delay requests after every `epoch` event and no delay after `done`.
  - Cooperative disconnect checks between intervals.
  - Persistence-before-completion orchestration.
  - Quiet stream termination and standard logging after training or persistence failures.
  - Route-level validation, success, disconnect, failure, isolation, and regression tests.
- **Evidence limitation:**
  - Baseline commands are user-reported rather than tool-verified in this planning session.
  - Automated backend tests cannot prove browser graph rendering or Vite proxy behavior.
  - The exact internal helper names used to isolate thread advancement and disconnect checks are implementation choices; tests must target the route-owned behavior seam rather than private loop syntax or a particular executor instance.

## Files to inspect before editing

1. `backend/src/how_llms_work/schemas.py` — `ChatRequest`; location for the dedicated `NeuralNetRequest`.
2. `backend/src/how_llms_work/main.py` — application construction and current router registration.
3. `backend/src/how_llms_work/routes/neural_net.py` — existing persistence symbols and destination for `router`, the endpoint, streaming orchestration, disconnect handling, delay, and logging.
4. `backend/src/how_llms_work/ml/neural_net.py` — `NetworkMode`, `EpochUpdate`, `TrainingResult`, `TrainingEvent`, `TrainingRun`, and `create_training_run()`.
5. `backend/src/how_llms_work/sse.py` — `format_sse()` and `create_sse_response()`.
6. `backend/tests/test_neural_net.py` — deterministic result, reporting schedule, payload, and isolation evidence that must not be duplicated or weakened.
7. `backend/tests/test_neural_net_persistence.py` — persistence failure, cleanup, and concurrency evidence to reuse during route integration.
8. `backend/tests/test_bpe_tokenize.py` — `TestClient`, named SSE parsing, exact headers, patched route dependency, and internal-detail suppression prior art.
9. `backend/tests/test_simple_chat.py` — patched route-level delay and health regression prior art.
10. `backend/pyproject.toml` — Python 3.12, pytest, Ruff, and strict mypy configuration.
11. `006-stream-complete-xor-training-runs-through-fastapi.md`, `SPEC.md`, and the latest `CONTEXT.md` — approved route behavior, terminology, test seams, failure behavior, and out-of-scope boundaries.
12. `llm_works_file_structure.md` — TypeScript request schema and neural-network event ordering reference only.

## Step 1 — Establish the request-validation contract

**Files and symbols:**
- `backend/tests/test_neural_net_route.py` — new validation-focused route tests.
- `backend/src/how_llms_work/schemas.py` — new `NeuralNetRequest`.
- `backend/src/how_llms_work/ml/neural_net.py` — existing `NetworkMode` type alias for the allowed mode values.

**Purpose:**
Prove the ticket's HTTP `422` boundary before adding stream behavior, including the exact required mode enum, strict integer epoch semantics, inclusive limits, and `5000` default. This covers the validation acceptance criteria without placing validation logic inside the SSE generator.

**Actions:**
- Add a dedicated route test module and a local named-SSE parser following the existing BPE/Simple Chat style.
- Add parameterized request tests for a missing mode, unknown mode, non-integer epoch values, `99`, and `100001`; assert standard JSON `422` responses and no SSE framing.
- Include boundary acceptance for `100` and `100000` using a controlled route training seam so the test does not execute a real `100000`-epoch Training Run.
- Add a request containing only `{"mode": "single-layer"}` and capture the arguments passed to the Training Run factory to prove `epochs=5000`.
- Add `NeuralNetRequest` with a required mode restricted to the existing two literal values and a strict integer `epochs` field defaulting to `5000` with inclusive `100` and `100000` bounds.
- Keep all validation declarative in Pydantic so failures remain standard FastAPI validation responses.

**Guardrails:**
- Do not add seed, learning rate, hidden size, activation, optimizer, output path, persistence switch, dtype, or any other request field.
- Do not reuse `ChatRequest`.
- Do not convert validation failures into SSE events or custom error payloads.
- Do not expose a deterministic seed through HTTP.
- Do not alter the existing numerical module's `epochs >= 0` internal guard; the stricter public HTTP range belongs to `NeuralNetRequest`.

**Expected result:**
- Valid request objects contain one allowed mode and a strict bounded integer epoch count.
- Omitted epochs become `5000`.
- Invalid request bodies fail before a stream or Training Run is created.

**Verification:**
- Run the validation-focused tests in `backend/tests/test_neural_net_route.py`.
- Confirm invalid responses use HTTP `422` with an application-JSON content type and contain no `event:` or `data:` lines.

## Step 2 — Register the neural-network router and establish the successful SSE shell

**Files and symbols:**
- `backend/src/how_llms_work/routes/neural_net.py` — new `router`, `POST /neural-net`, and neural-network stream entry point.
- `backend/src/how_llms_work/main.py` — neural-network router import and `app.include_router(...)`.
- `backend/src/how_llms_work/sse.py` — existing `format_sse()` and `create_sse_response()` reuse.
- `backend/tests/test_neural_net_route.py` — registration, header, and exact successful event-shell tests.

**Purpose:**
Expose the missing endpoint without changing the established Health, Simple Chat, or BPE contracts, and prove that the route uses the shared SSE behavior.

**Actions:**
- Create an `APIRouter` in the existing neural-network route module without moving or duplicating its persistence functions.
- Add `POST /neural-net` accepting both the validated `NeuralNetRequest` and the FastAPI/Starlette request object needed for disconnect observation.
- Create one independent Training Run inside the endpoint by calling the existing `create_training_run(mode, epochs)` production factory.
- Return the existing shared `create_sse_response()` around the neural-network async stream.
- Register the neural-network router in `main.py` while retaining the existing router registrations and health endpoint.
- Add a controlled successful request test asserting HTTP `200`, `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- Add route-registration regression assertions for `GET /health`, `POST /simple-chat`, and `POST /bpe-tokenize`, reusing existing tests rather than duplicating their full contracts.

**Guardrails:**
- Do not hand-build `StreamingResponse` headers or copy `format_sse()` logic into the route.
- Do not change the existing endpoint paths or add an `/api` prefix to FastAPI.
- Do not add frontend or Vite changes.
- Do not create a second neural-network persistence module.
- Do not add dependencies.

**Expected result:**
- FastAPI exposes `POST /neural-net`.
- A valid controlled request returns the standard shared SSE response.
- Existing routes remain registered and unchanged.

**Verification:**
- Run the focused route registration/header test.
- Inspect `app.routes` or make in-process requests to confirm all four preserved endpoints remain available.

## Step 3 — Advance bounded Training Run intervals in same-process worker threads

**Files and symbols:**
- `backend/src/how_llms_work/routes/neural_net.py` — neural-network async stream and route-owned interval-advancement seam.
- `backend/src/how_llms_work/ml/neural_net.py` — existing `TrainingRun.__next__()`, `EpochUpdate`, and `TrainingResult`.
- `backend/tests/test_neural_net_route.py` — worker-thread, interval ordering, event payload, and delay tests.

**Purpose:**
Keep CPU-bound training off the route's async event-loop thread while preserving the numerical module's existing reporting-interval iterator. Each offloaded call must produce at most the next public Training Event, allowing the async route to regain control between Epoch Updates.

**Actions:**
- Advance the existing Training Run one `next()` call at a time through Python's same-process thread-offloading facility.
- Keep the default executor configuration untouched; do not instantiate an unbounded executor or alter global thread-pool capacity.
- Distinguish the returned public event by its existing `EpochUpdate` or `TrainingResult` type instead of inspecting private Training Run state.
- For every `EpochUpdate`, emit `format_sse("epoch", update.to_payload())`.
- After each yielded `epoch` event, request exactly `asyncio.sleep(0.02)` through the route module's referenced sleep operation.
- Do not request a presentation delay after `done`.
- Add controlled tests whose fake iterator records one advancement per public event and proves that advancement occurs off the async route thread while remaining in the same process.
- Assert exact successful event order `epoch × N → done`, exact `epoch` payload key sets, no event after `done`, and delay requests equal one `0.02` entry per Epoch Update.
- Use a controlled Training Run or deterministic factory patch for route tests; do not expose or add a seed request field.

**Guardrails:**
- Do not call `list(training_run)` or otherwise complete the Training Run before streaming.
- Do not offload the entire Training Run as one worker-thread task.
- Do not create a process pool, worker process, shared-memory region, external queue, forceful thread termination path, or executor-capacity change.
- Do not test a particular executor object or private loop syntax.
- Do not duplicate numerical assertions already covered by `test_neural_net.py`.

**Expected result:**
- The async route regains control after every public Epoch Update.
- Training work is bounded by the numerical iterator's next reporting interval.
- Each progress event is exact and receives one requested presentation delay.

**Verification:**
- Run focused worker-thread and event-order tests.
- Confirm a controlled run produces only exact `epoch` events followed by one `done`, with no delay after completion.

## Step 4 — Persist successfully before emitting the final completion event

**Files and symbols:**
- `backend/src/how_llms_work/routes/neural_net.py` — `save_network()`, successful `TrainingResult` handling, and final SSE emission.
- `backend/src/how_llms_work/ml/neural_net.py` — `TrainingResult.to_frontend_payload()` and `TrainingResult.weights`.
- `backend/tests/test_neural_net_route.py` — persistence-before-`done`, exact final payload, default epoch, and request-isolation tests.
- `backend/tests/test_neural_net_persistence.py` — existing mode-specific save and concurrency evidence.

**Purpose:**
Join the completed numerical and persistence tickets at the route boundary so the frontend receives completion only after the correct Saved Weight Snapshot has been safely replaced.

**Actions:**
- When the worker-thread interval returns a `TrainingResult`, call the existing route-owned `save_network(result.weights)` boundary and wait for successful completion before constructing or yielding `done`.
- Offload blocking persistence work without introducing a new process or executor configuration.
- Emit `format_sse("done", result.to_frontend_payload())` exactly once after persistence returns successfully, then end the stream without another Training Run advancement.
- Add a successful test using a temporary snapshot directory and a controlled `TrainingResult`; assert the mode-specific file exists and contains the expected weights while the `done` payload contains only `architecture`, `predictions`, and `verdict`.
- Prove ordering through a controlled persistence failure test: when save fails, no `done` can appear.
- Assert exactly four predictions in `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]` order and assert no weight key appears anywhere in the `done` payload.
- Add two controlled requests with distinct result/weight objects and assert the route passes each request's own result to persistence and emits each request's own completion payload.
- Rely on existing numerical tests for exact reference architecture labels, rounded values, and verdict strings; the route test should prove transparent serialization of those already-public values.

**Guardrails:**
- Do not emit `done` before persistence.
- Do not include Saved Weight Snapshot data, destination paths, seed, or internal state in the frontend payload.
- Do not load an existing snapshot into a new Training Run.
- Do not add snapshot history, manifests, registries, or request IDs.
- Do not duplicate or weaken the safe-replacement logic already implemented by Ticket 005.

**Expected result:**
- A successful request produces one complete mode-specific snapshot before one final `done`.
- The frontend payload contains no persistence data.
- Separate requests retain independent run and completion state.

**Verification:**
- Run focused success, payload, persistence-order, default-epoch, and request-isolation tests.
- Re-run the existing numerical and persistence test modules to confirm unchanged contracts.

## Step 5 — Stop cooperatively after a controlled browser disconnect

**Files and symbols:**
- `backend/src/how_llms_work/routes/neural_net.py` — request-disconnect observation and route-owned disconnect seam.
- `backend/tests/test_neural_net_route.py` — controlled disconnect test with interval-count and persistence assertions.

**Purpose:**
Ensure an abandoned browser request cannot start later training intervals or replace a prior successful snapshot, while accepting that one interval already running may finish before disconnection is observed.

**Actions:**
- Add a narrow route-owned async boundary that obtains the current request's disconnect state from the injected request object.
- Check that boundary after each Epoch Update has been yielded and its required presentation delay has been requested, before starting the next worker-thread interval.
- When disconnected, return from the stream without advancing again, persisting weights, emitting `done`, or inventing an SSE `error` event.
- Add a deterministic fake disconnect sequence that becomes true after a selected Epoch Update.
- Use an instrumented fake Training Run to assert that no later interval starts after the disconnect is observed; allow only an interval that was already started before the check to finish.
- Assert `save_network()` is not called and the prior snapshot document remains unchanged.
- Assert the collected stream contains only the Epoch Updates produced before termination and contains no `done`, `error`, exception text, traceback, or path.

**Guardrails:**
- Do not simulate a fragile real TCP interruption in automated tests.
- Do not use forceful thread termination.
- Do not persist a partial or disconnected Training Run.
- Do not store disconnect state globally or on a shared Training Run.
- Do not treat a normal disconnect as a client-visible error event.

**Expected result:**
- The route cooperatively stops between reporting intervals.
- No later interval, persistence, or completion begins after the disconnect is observed.
- Existing successful snapshots remain untouched.

**Verification:**
- Run the focused controlled-disconnect test.
- Confirm advancement count, save-call count, event names, and prior snapshot bytes exactly match expectations.

## Step 6 — Contain training and persistence failures after streaming begins

**Files and symbols:**
- `backend/src/how_llms_work/routes/neural_net.py` — module logger, post-stream exception boundary, Training Run advancement, and `save_network()`.
- `backend/tests/test_neural_net_route.py` — controlled training-failure, persistence-failure, privacy, cleanup, and logging tests.
- `backend/tests/test_neural_net_persistence.py` — existing prior-destination preservation and temporary-file cleanup tests.

**Purpose:**
After the first SSE event has been sent, HTTP status replacement is no longer the contract. The stream must therefore terminate quietly, preserve the previous snapshot, and log the unexpected failure internally without serializing details to the client.

**Actions:**
- Add a module-level logger using standard Python logging.
- Wrap post-stream Training Run advancement and persistence so unexpected exceptions are logged and the async generator returns without yielding another event.
- Do not catch or transform pre-stream Pydantic validation failures.
- Add a controlled Training Run that yields one or more Epoch Updates and then raises an exception marker; assert the response contains only earlier `epoch` events and no `done`, `error`, marker, traceback, or filesystem path.
- Use log capture to assert the unexpected training exception follows the standard logger path.
- Add a route-level persistence failure integration test using `tmp_path`, a pre-existing destination, and one injected persistence failure boundary from the existing module.
- Assert the previous destination remains byte-for-byte unchanged, temporary files are removed when cleanup succeeds, no `done` or `error` event is emitted, and no exception marker or path leaks into the SSE body.
- Retain the existing dedicated persistence tests as the detailed evidence for serialization, write, replacement, and cleanup failure behavior.

**Guardrails:**
- Do not add an SSE `error` event.
- Do not serialize exception messages, `repr`, stack traces, or paths.
- Do not catch `BaseException` or suppress process cancellation semantics.
- Do not delete or truncate the prior successful destination on failure.
- Do not move persistence failure handling into the numerical module.
- Do not replace standard logging with print statements or a new logging dependency.

**Expected result:**
- Training and persistence failures after progress terminate the stream without false completion or client-visible internals.
- Unexpected failures remain visible to operators through normal Python logging.
- Existing Saved Weight Snapshots remain valid.

**Verification:**
- Run the focused training-failure and persistence-failure route tests with log capture.
- Run `backend/tests/test_neural_net_persistence.py` to reconfirm all detailed filesystem safeguards.

## Step 7 — Run focused, affected-area, and complete automated verification

**Files and symbols:**
- `backend/tests/test_neural_net_route.py` — all Ticket 006 route acceptance tests.
- `backend/tests/test_neural_net.py` — Ticket 004 numerical regressions.
- `backend/tests/test_neural_net_persistence.py` — Ticket 005 persistence regressions.
- `backend/tests/test_simple_chat.py` — shared SSE and health regressions.
- `backend/tests/test_bpe.py` and `backend/tests/test_bpe_tokenize.py` — BPE regressions.
- `backend/pyproject.toml` — pytest, Ruff, and mypy configuration.

**Purpose:**
Prove the new vertical slice and the preserved contracts with the project's established Poetry quality path, while reporting actual results honestly.

**Actions:**
- Run the dedicated route test module first.
- Run the three neural-network test modules together.
- Run the complete pytest suite once after focused tests are green.
- Run Ruff over the repository.
- Run strict mypy over `src`.
- Correct only Ticket 006 regressions; do not use broad refactoring or formatting churn to address unrelated findings.
- Record the exact command outputs in the implementation completion report.

**Guardrails:**
- Do not claim a command passed unless it was executed successfully in the implementation session.
- Do not change dependencies or lockfiles to bypass a test or type error.
- Do not weaken exact payload, ordering, failure, or isolation assertions.
- Do not alter existing numerical constants or persistence semantics merely to simplify route tests.

**Expected result:**
- Focused neural-network route tests pass.
- Numerical and persistence tests remain green.
- The complete suite, Ruff, and strict mypy pass.

**Verification:**
- Execute the commands in the Focused and Full Verification sections below in order.

## Step 8 — Perform manual end-to-end acceptance and final scope inspection

**Files and symbols:**
- `backend/src/how_llms_work/main.py` — registered application routes.
- `backend/src/how_llms_work/routes/neural_net.py` — final stream behavior.
- `backend/.data/` — runtime-only Saved Weight Snapshots.
- `frontend/` — unchanged frontend used only for manual acceptance.
- Git diff/status — final scope inspection.

**Purpose:**
Confirm the unchanged frontend can consume the new endpoint through the existing development proxy, and ensure the implementation contains only the planned backend vertical slice.

**Actions:**
- Start the FastAPI backend and Vite frontend with the repository's documented PowerShell commands.
- Submit both Single-Layer and Multi-Layer requests through the frontend or a proxy-targeted `curl.exe -N` request.
- Confirm visible progress, one final result, exact architecture/verdict text, ordered predictions, and no weights in browser-visible data.
- Confirm the correct mode-specific JSON snapshot appears under `backend/.data/` only after a successful completion.
- Interrupt or close one browser request during training and confirm no new completion appears; treat this as a manual observation rather than automated proof.
- Inspect the final diff and repository status for unplanned files, real test snapshots, cache files, frontend edits, dependency changes, or unrelated formatting.
- Exclude runtime `.data` artifacts from the implementation commit according to the repository's ignore policy; do not add a new ignore-file change unless current repository evidence requires it.

**Guardrails:**
- Do not claim automated tests prove browser rendering or Vite proxy integration.
- Do not commit generated snapshots, temporary files, caches, or virtual-environment files.
- Do not modify the frontend to compensate for a backend contract mismatch.
- Do not expand into Train Embeddings, Train Transformer, general matrix, model loading, or snapshot-history work.

**Expected result:**
- The unchanged frontend consumes `POST /neural-net` successfully.
- Runtime snapshots are created only for successful runs.
- The final diff is limited to Ticket 006.

**Verification:**
- Complete the manual acceptance checklist.
- Review `git diff --check`, `git diff`, and `git status --short` before committing.

## Focused verification plan

Run from the backend directory:

```powershell
poetry run pytest tests/test_neural_net_route.py -q
```

Expected result:

- Validation, registration, headers, exact SSE events, worker-thread interval behavior, delays, persistence ordering, disconnect handling, failure privacy/logging, default epochs, and request isolation pass.

Then run the affected neural-network area:

```powershell
poetry run pytest `
    tests/test_neural_net.py `
    tests/test_neural_net_persistence.py `
    tests/test_neural_net_route.py `
    -q
```

Expected result:

- Tickets 004, 005, and 006 remain mutually compatible.

## Full verification plan

Run once after focused tests pass:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.

Final scope checks:

```powershell
git diff --check
git diff
git status --short
```

Expected result:

- No whitespace errors.
- Only planned Ticket 006 files and any explicitly justified conditional file appear.

## Manual acceptance checklist

- [ ] `GET /health` still returns `{"status":"healthy"}`.
- [ ] `POST /simple-chat` still streams `start → word × N → done`.
- [ ] `POST /bpe-tokenize` still streams its existing BPE contract.
- [ ] `POST /neural-net` rejects missing/unknown mode and invalid epoch values with HTTP `422`.
- [ ] A request containing only `{"mode":"single-layer"}` uses `5000` epochs.
- [ ] A valid request returns `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- [ ] Progress includes epoch zero, reference reporting boundaries, and the requested final epoch.
- [ ] Every progress event is named `epoch` and contains only `epoch` and `loss`.
- [ ] A successful stream is `epoch × N → done` with nothing after `done`.
- [ ] The final payload contains only `architecture`, `predictions`, and `verdict`.
- [ ] The four predictions appear in XOR truth-table order and no weights are visible to the client.
- [ ] Single-Layer and Multi-Layer architecture labels and verdict strings match the established contract.
- [ ] The correct mode-specific Saved Weight Snapshot exists before completion is observed.
- [ ] A disconnected request produces no new snapshot, no `done`, and no SSE `error`.
- [ ] A failed Training Run produces no snapshot replacement, no `done`, no SSE `error`, and no leaked internals.
- [ ] A persistence failure preserves the prior snapshot, removes its temporary file when cleanup succeeds, and emits no `done`.
- [ ] Two controlled requests retain independent Training Runs, completion data, disconnect state, and temporary files.
- [ ] No frontend file, dependency manifest, lockfile, future-phase module, or generated snapshot is included in the implementation diff.
- [ ] Actual pytest, Ruff, and mypy results are recorded honestly.

## Expected files changed

Likely changed:

```text
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/main.py
backend/src/how_llms_work/routes/neural_net.py
backend/tests/test_neural_net_route.py
```

Conditionally changed only if focused verification exposes a genuine integration or typing gap:

```text
backend/src/how_llms_work/ml/neural_net.py
backend/tests/test_neural_net.py
backend/tests/test_neural_net_persistence.py
```

No package or lockfile change is expected.

## Files not to change

```text
frontend/
backend/src/how_llms_work/sse.py
backend/src/how_llms_work/ml/bpe.py
backend/src/how_llms_work/ml/math_utils.py
backend/src/how_llms_work/ml/matrix.py
backend/src/how_llms_work/ml/word2vec.py
backend/src/how_llms_work/ml/transformer.py
backend/src/how_llms_work/ml/transformer_worker.py
backend/src/how_llms_work/routes/bpe_tokenize.py
backend/src/how_llms_work/routes/simple_chat.py
backend/src/how_llms_work/routes/train_embed.py
backend/src/how_llms_work/routes/train_transformer.py
backend/tests/test_bpe.py
backend/tests/test_bpe_tokenize.py
backend/tests/test_simple_chat.py
backend/pyproject.toml
backend/poetry.lock
backend/.data/*.json
```

The existing Simple Chat and BPE test files should be executed as regressions, not edited, unless implementation evidence reveals a direct Ticket 006 registration regression that cannot be covered in the new route test module.

## Risk notes and safeguards

1. **Risk:** The route performs all epochs on the async event-loop thread and stalls other requests.
   - **Safeguard:** Offload exactly one existing `TrainingRun.__next__()` reporting interval at a time and return to async orchestration after every public event.

2. **Risk:** Offloading the whole run prevents cooperative disconnect checks.
   - **Safeguard:** Never convert the iterator to a list or wrap the complete Training Run in one thread task; check disconnect state before every later interval.

3. **Risk:** `StopIteration` is propagated through an async future incorrectly.
   - **Safeguard:** Stop immediately after the iterator's guaranteed `TrainingResult`; do not request another interval after `done`. Keep any route-owned advancement adapter typed to return a public `TrainingEvent`.

4. **Risk:** A disconnect occurs just after an interval begins.
   - **Safeguard:** Accept completion of only the already-started bounded interval, then observe disconnect before starting the next interval.

5. **Risk:** Presentation-delay tests become slow or timing-dependent.
   - **Safeguard:** Patch only the route module's referenced sleep operation and assert requested values, never elapsed wall-clock duration.

6. **Risk:** Persistence succeeds after `done` or a persistence failure still reports completion.
   - **Safeguard:** Await successful `save_network()` completion before yielding `done`; prove the ordering by injecting a save failure and asserting `done` is absent.

7. **Risk:** Post-stream exceptions leak markers, tracebacks, or paths to the frontend.
   - **Safeguard:** Log through a module logger, terminate without a new SSE event, and assert the response body contains no internal details.

8. **Risk:** Broad exception handling hides validation errors or cancellation.
   - **Safeguard:** Keep validation outside the body iterator and catch only ordinary post-stream exceptions, not `BaseException`.

9. **Risk:** A new route implementation duplicates SSE framing or headers and drifts from existing endpoints.
   - **Safeguard:** Reuse `format_sse()` and `create_sse_response()` directly and assert the shared headers.

10. **Risk:** Route tests write into the real `backend/.data/` directory.
    - **Safeguard:** Patch the existing route-owned snapshot-directory boundary to `tmp_path` for every route test that permits persistence.

11. **Risk:** Test doubles bypass the contract so thoroughly that the real numerical and persistence modules are no longer covered.
    - **Safeguard:** Keep focused route tests controlled, then run the existing numerical and persistence suites together and include at least one temporary-directory route integration test using the real persistence boundary.

12. **Risk:** Pydantic coercion accepts values that do not satisfy strict integer request semantics.
    - **Safeguard:** Use strict integer validation and include numeric-string, fractional, boolean, and boundary cases in parameterized tests.

13. **Risk:** Registering the router accidentally removes or shadows existing routes.
    - **Safeguard:** Retain all current `include_router()` calls and run Health, Simple Chat, and BPE regressions after registration.

14. **Risk:** Runtime snapshots or temporary files are committed.
    - **Safeguard:** Use temporary directories in tests, inspect `git status --short`, and keep generated `.data` contents out of the implementation commit.

15. **Risk:** The implementation drifts into numerical redesign, model loading, or future phases.
    - **Safeguard:** Treat Tickets 004 and 005 as completed and restrict production changes to schema, registration, and route orchestration.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Suggested outcome:

```text
Stream XOR training through FastAPI
```

Commit body should mention:

- validated and registered `POST /neural-net`;
- bounded same-process thread-offloaded Training Run intervals;
- exact shared-SSE progress/completion behavior and presentation delays;
- persistence-before-`done`, cooperative disconnect handling, quiet failure behavior, and request isolation;
- the exact focused and full verification commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, `006-stream-complete-xor-training-runs-through-fastapi.md`, `SPEC.md`, the latest `CONTEXT.md`, `py_llm_pipeline_explorer_file_structure(11).md`, and the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 006, verify the complete change, report actual command results honestly, and create the implementation commit.
