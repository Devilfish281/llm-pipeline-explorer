---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "022"
source_work_item: 022-stream-complete-transformer-training-runs-through-fastapi.md
source_specification: SPEC.md
source_context: CONTEXT.md
architecture_decision: 0002-stabilize-python-transformer-training-and-process-lifecycle.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(86).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 022: Stream complete Transformer Training Runs through FastAPI

## Initial checklist

- Confirm Ticket 022 is the only selected work item and that completed Tickets 017, 018, 020, and 021 are represented in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(86).md` as current-code authority; do not let older exports, plans, snippets, or historical TypeScript process behavior override it.
- Preserve the unchanged TypeScript/Vite Frontend Contract while using the TypeScript server only as behavior evidence for request fields, event payloads, and architecture text.
- Reuse the completed Transformer Training Run, Generated Text Sample, final-evaluation, worker-group, cleanup, and persistence boundaries rather than duplicating their internals in the route.
- Limit production changes to the dedicated request schema, FastAPI router registration, and request/stream lifecycle orchestration in `routes/train_transformer.py`.
- Add endpoint-focused tests with controlled seams plus one bounded integration case that reaches real public Transformer boundaries without running maximum training.
- Finish with focused route and Transformer regressions, the complete pytest suite, formatting/lint checks, strict mypy, a scope-only diff review, and a separately recorded browser or Vite-proxy check when practical.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript behavior to the Python Backend, plan Ticket 022 only, and treat the latest complete Python export as current-code truth.
2. `022-stream-complete-transformer-training-runs-through-fastapi.md` for immediate scope, acceptance criteria, approved test seam, blockers, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(86).md` for current source, tests, fixtures, dependencies, typing conventions, public symbols, and existing route/persistence patterns.
4. `SPEC.md` for the accepted complete Phase 5 request, event-stream, numerical-order, lifecycle, timeout, failure, persistence, and test decisions.
5. `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for the binding process-local run slot, four Logical Training Shards, request-owned resources, deadline, cleanup, forced-termination, and persistence-before-completion architecture.
6. `CONTEXT.md` for canonical terminology, especially Transformer Training Run, Transformer Event Stream, Transformer Epoch Update, Generated Text Sample, Saved Transformer Model, Request-Scoped Worker Group, and Request-Scoped Shared Memory.
7. The completed current public boundaries in:
   - `src/how_llms_work/ml/transformer.py`;
   - `src/how_llms_work/ml/transformer_worker.py`;
   - `src/how_llms_work/routes/train_transformer.py`;
   - their existing focused tests and fixtures.
8. Existing registered XOR and Word2Vec routes and tests as prior art for FastAPI, shared SSE framing, presentation-delay seams, disconnect observation, persistence-before-`done`, and client-visible failure privacy.
9. The latest `llm_works_file_structure.md` only for the unchanged request shape, eleven-field `init`, `epoch`/`done` field sets, frontend event discrimination, and exact architecture string:

   ```text
   Decoder-Only Transformer (<layers> layers, 32d, 2h, 128ff)
   ```

   Do not copy cached-model skipping, resumption, intermediate checkpoints, host-dependent partitioning, direct process behavior, or other TypeScript details superseded by ADR 0002.
10. Official FastAPI, Starlette, Pydantic, Python `asyncio`, multiprocessing, and pytest documentation only as technical cross-checks for streaming responses, disconnection, strict finite validation, thread offloading, cancellation, and in-process endpoint testing.
11. Older plans are historical examples only. They do not override the selected ticket or current export.

## Work-item summary

Complete the Phase 5 vertical slice by exposing and registering `POST /train-transformer` through FastAPI around the Transformer capabilities already completed by Tickets 017 through 021.

A valid available request must be strictly validated before it can reserve the one process-local Transformer run slot. While holding that slot, the endpoint must obtain immutable preprocessing and construct the complete request-dependent eleven-field `init` payload before returning a streaming response. Pre-stream preparation failure must release the slot and return a sanitized HTTP `500`; an overlapping valid request must receive immediate HTTP `429` with no queue and no SSE body.

The stream must yield `init` first, before worker creation or shared-memory allocation. It must then initialize fresh deterministic weights, create one parent-owned `TransformerTrainingRun`, and create one Request-Scoped Worker Group through the existing public factory. Each inclusive epoch must be computed by the worker group and committed through `TransformerTrainingRun.advance_epoch()`. At every public report boundary, the route must create one bounded thread-offloaded Generated Text Sample, emit an exact `epoch` event, append the exact `{epoch, text}` sample object in report order, and request the established `0.02`-second presentation delay once.

After the final Adam update, the route must recompute final loss through the existing bounded cooperative evaluation boundary. It must then complete worker cleanup and require a fully successful cleanup result before persistence: forced termination, kill escalation, nonzero exit, cleanup failure, or any earlier worker failure must prevent persistence and `done`. Only after the worker group has closed successfully may the route build the complete Saved Transformer Model, persist it through Ticket 021's configuration-specific atomic boundary, and emit exactly one `done` event with the reference architecture text, final loss, and complete ordered sample collection.

Every disconnect, timeout, cancellation, worker/protocol failure, numerical failure, generation/evaluation failure, model-construction failure, serialization failure, write failure, replacement failure, or cleanup failure after `init` must terminate quietly without `done`, a new SSE `error` event, or client-visible internal details. `asyncio.CancelledError` must propagate after required cleanup. The request-owned cooperative cancellation event must be set on disconnect/failure/cancellation, and any active helper thread must be awaited before numerical state or shared memory it could access is released. The process-local run slot is released last on every path.

This ticket does not redesign Transformer mathematics, worker protocol, shared-memory layout, Adam, generation, evaluation, Saved Transformer Model construction, persistence, shared SSE infrastructure, or frontend code.

## Readiness determination

**Status:** Ready for implementation.

The selected ticket has concrete behavior, explicit acceptance criteria, an approved endpoint seam, completed blocker implementations in the latest export, and no unresolved product or architecture decision. The work is large but forms one coherent final vertical slice around already-separated public boundaries.

## Baseline evidence

- **Evidence level:** User-reported before planning.
- **Command:** `poetry run pytest`
- **Reported result:** All tests passed.
- **Command:** `poetry run ruff check .`
- **Reported result:** Ruff passed.
- **Command:** `poetry run mypy src`
- **Reported result:** `Success: no issues found`.
- **Planning-session limitation:** None of these commands was executed while producing this plan. This planning session inspected supplied artifacts and performed read-only documentation/source analysis only.
- **Implementation rule:** `implement-prompt` must establish or reconfirm its own baseline before editing and report only outcomes it actually observes.

## Current code observations

### Application and shared transport

- `src/how_llms_work/main.py` currently imports and registers the Simple Chat, BPE Tokenizer, XOR Neural Network, and Train Embed routers.
- `POST /train-transformer` is not registered.
- `src/how_llms_work/sse.py` already owns the required shared transport:
  - `format_sse()`;
  - `create_sse_response()`;
  - `text/event-stream`;
  - `Cache-Control: no-cache`;
  - `X-Accel-Buffering: no`.
- No new response class, header helper, or SSE formatter is needed.

### Request schema

- `src/how_llms_work/schemas.py` currently defines `ChatRequest`, `NeuralNetRequest`, and `TrainEmbedRequest`.
- It does not define the dedicated Train Transformer request model.
- Existing models use `Annotated`, strict `Field` constraints, aliases, and model-local `ConfigDict(extra="ignore")`, which is the correct local convention to preserve.

### Transformer core

`src/how_llms_work/ml/transformer.py` already provides the reusable public boundaries Ticket 022 must orchestrate:

- immutable `TransformerPreprocessingSnapshot` and `get_transformer_preprocessing()`;
- fixed model constants and architecture dimensions;
- canonical parameter layout and `transformer_parameter_count()`;
- deterministic `initialize_transformer_parameters()` with request-owned `Mulberry32`;
- `TransformerTrainingRun` and `create_transformer_training_run()`;
- ordered `advance_epoch()` with four-shard validation, Ordered Gradient Reduction, parent-side Adam, finite-state enforcement, inclusive reporting, and completion state;
- `generate_transformer_text()` using independent `(42 + epoch) modulo 2^32` sampling and cooperative cancellation;
- `evaluate_transformer_final_loss()` using final post-Adam weights and cooperative cancellation;
- `build_saved_transformer_model()` for the exact complete persistence-ready artifact.

The route must consume these boundaries. It must not reimplement gradient reduction, Adam, report scheduling, generation, final loss, or model serialization structure.

### Request-Scoped Worker Group

`src/how_llms_work/ml/transformer_worker.py` already provides:

- `create_request_scoped_worker_group()`;
- `RequestScopedWorkerGroup.compute_epoch()`;
- `RequestScopedWorkerGroup.cleanup()`;
- one-through-four local-spawn workers;
- exactly one shared weight block plus four shared gradient blocks;
- static modulo shard assignment;
- 30-second complete-group startup deadline;
- five-minute epoch deadline;
- `connection.wait()` polling every `0.1` seconds through `asyncio.to_thread()`;
- a route-neutral asynchronous `poll_observer` invoked after each process poll;
- staged cooperative stop, terminate, and kill cleanup;
- immutable cleanup diagnostics;
- `successful`, `primary_failure_code`, and `cleanup_report` completion gates;
- cancellation propagation and idempotent cleanup.

The route should supply disconnect observation through `poll_observer`, use the public group methods only, and treat `group.successful` as a required pre-persistence completion gate.

### Transformer persistence

`src/how_llms_work/routes/train_transformer.py` is already a substantial persistence-only module. It currently provides:

- strict Saved Transformer Model validation;
- exact epoch/layer configuration validation;
- configuration-specific filename construction;
- backend-root `.data` resolution independent of CWD;
- deterministic two-space JSON plus final newline;
- complete serialization before filesystem mutation;
- unique same-directory temporary-file creation;
- explicit write, flush, file `fsync`, close, and atomic replacement;
- prior-destination preservation;
- owned-temporary cleanup;
- `save_transformer_model()`.

Ticket 022 must preserve those public persistence operations and extend this module with HTTP/SSE orchestration rather than replacing the persistence implementation.

### Existing tests and prior art

- `tests/test_train_transformer_persistence.py` already covers Ticket 021 independently and must continue to pass.
- `tests/test_transformer_training.py` covers report schedules, ordered reduction, Adam, finite state, and run isolation.
- `tests/test_transformer_completion.py` covers Generated Text Samples, final evaluation, complete model construction, cancellation, and finite state.
- `tests/test_transformer_worker.py` and `tests/test_transformer_worker_group.py` cover the real spawn protocol, process polling, deadlines, five shared-memory blocks, failure handling, cleanup, and request isolation.
- No `tests/test_train_transformer_route.py` exists in the current export.
- `routes/train_embed.py` and `tests/test_train_embed_route.py` provide close application-level patterns, but Ticket 022 requires stronger run reservation, process cleanup gating, bounded helper-task lifecycle, and cancellation semantics than Train Embed.

### Dependencies and configuration

- Current dependencies already contain FastAPI, Pydantic, NumPy, pytest, pytest-asyncio, Ruff, mypy, HTTPX/TestClient support, and Black.
- Standard-library `asyncio`, `threading`, logging, and existing multiprocessing code are sufficient.
- No dependency or lockfile change is expected.

## Acceptance-criteria coverage before implementation

### Already satisfied and evidenced by completed public boundaries

- Fixed Transformer corpus, preprocessing, Vocabulary, Training Sequences, generation seeds, and four Logical Training Shards.
- Fixed model dimensions and configurable one-through-six layer depth.
- Fresh deterministic initialization capability.
- Inclusive epoch schedule and exact report boundaries.
- Four-shard Ordered Gradient Reduction and one parent-side Adam update per epoch.
- Six-decimal public epoch updates.
- Independent epoch-seeded Generated Text Sample behavior.
- Final post-Adam loss evaluation.
- Exact Saved Transformer Model construction.
- Request-Scoped Worker Group creation, startup and epoch deadlines, off-event-loop polling, commit-gated shard results, cleanup, and five request-owned shared-memory blocks.
- Configuration-specific atomic Transformer persistence.
- Shared SSE formatting and response headers.
- Existing Learning Demo route regression suites.

### Behavior present but not yet integrated

- `routes/train_transformer.py` owns persistence but not `APIRouter`, request orchestration, streaming, disconnect handling, run reservation, or lifecycle cleanup.
- The worker group accepts a route-owned `poll_observer`, but no route currently supplies browser-disconnect observation.
- Generation and final evaluation support cooperative cancellation, but no request-owned cancellation event or bounded route-level thread wrapper currently coordinates them.
- Exact model and persistence boundaries exist separately, but no endpoint proves successful cleanup and persistence-before-`done` ordering.

### Not implemented or not evidenced

- Dedicated exact five-field Train Transformer request schema.
- Validation-before-reservation evidence and complete invalid-input matrix.
- Process-local nonblocking run slot and HTTP `429` overlap behavior.
- Sanitized pre-stream HTTP `500` behavior.
- Exact eleven-field `init` preparation before `StreamingResponse` creation.
- Registered `POST /train-transformer`.
- First-event-before-shared-memory/startup proof.
- Fresh route-owned initialization and training orchestration.
- Exact `init → epoch × N → done` endpoint stream.
- Per-report sample collection and `0.02`-second presentation-delay requests.
- Route-owned five-minute helper deadlines and active-helper draining.
- Browser-disconnect propagation through every worker poll and later blocking stage.
- Final worker cleanup success gate before persistence.
- Quiet post-stream failure privacy across every route stage.
- `asyncio.CancelledError` propagation after cleanup.
- Run-slot release-last proof on all pre-stream, stream, failure, timeout, disconnect, and cancellation paths.
- Sequential and controlled overlapping request isolation at the endpoint.
- One bounded route integration through real public Transformer boundaries.
- Practical two-server browser or Vite-proxy result.

### Evidence limitations

- The supplied source is a complete text export rather than a live Git checkout. `implement-prompt` must inspect the actual working tree and preserve user changes.
- Baseline results are user-reported, not tool-verified in this planning session.
- A browser result cannot be inferred from `TestClient` tests.
- Maximum-configuration endurance is intentionally outside ordinary pytest.

## Files to inspect before editing

1. `022-stream-complete-transformer-training-runs-through-fastapi.md` — reread the exact ticket first.
2. `src/how_llms_work/schemas.py` — destination for the dedicated request model; preserve existing models unchanged.
3. `src/how_llms_work/main.py` — add only the missing router import and `include_router()` call.
4. `src/how_llms_work/routes/train_transformer.py` — preserve Ticket 021 persistence and add the complete route-owned orchestration boundary.
5. `src/how_llms_work/sse.py` — reuse unchanged.
6. `src/how_llms_work/ml/transformer.py` — inspect and call public preprocessing, layout, initialization, training, sampling, evaluation, and model-construction operations; do not modify by default.
7. `src/how_llms_work/ml/transformer_worker.py` — inspect public worker-group state, failure, poll-observer, compute, cleanup, and success semantics; do not modify by default.
8. `src/how_llms_work/routes/train_embed.py` — inspect logging, disconnect, thread-offload, presentation-delay, and shared-SSE conventions; do not copy its weaker lifecycle assumptions blindly.
9. `tests/test_train_embed_route.py` and `tests/test_neural_net_route.py` — reuse strict SSE parsing, controlled seams, thread observations, failure privacy, registration, and request-isolation patterns.
10. `tests/test_train_transformer_persistence.py` — preserve every public persistence symbol and exact behavior.
11. `tests/test_transformer_training.py` — use existing public Training Run helpers and reporting semantics.
12. `tests/test_transformer_completion.py` — use existing Generated Text Sample, final-evaluation, cancellation, and model-boundary test patterns.
13. `tests/test_transformer_worker_group.py` — use public controlled runtime and cleanup-report patterns; do not test generated names or private wait helpers.
14. `llm_works_file_structure.md` — confirm exact request aliases, eleven `init` fields, event discrimination, sample object shape, and exact architecture text only.
15. `SPEC.md`, `CONTEXT.md`, and ADR 0002 — reconfirm ownership, sequencing, deadline, cleanup, privacy, and no-loading/no-checkpoint decisions.
16. `pyproject.toml` and `README.md` — confirm exact project validation and run commands before implementation.

## Step 1 — Establish the complete endpoint acceptance seam before production orchestration

**Files and symbols:**

- New `tests/test_train_transformer_route.py`.
- Existing public route module import `how_llms_work.routes.train_transformer`.
- Existing `app` from `how_llms_work.main`.

**Purpose:**

Create a fast, deterministic, public-endpoint test seam that can prove ordering, lifecycle, privacy, and isolation without executing full production Transformer training in every test.

**Actions:**

- Add one strict SSE parser that records event names, JSON payloads, order, duplicate fields, and trailing data.
- Exercise the registered route through `TestClient` or equivalent in-process ASGI calls rather than invoking private generator helpers as the primary acceptance seam.
- Define narrow controlled substitutes for:
  - immutable preprocessing retrieval;
  - parameter initialization and Training Run creation;
  - worker-group creation, epoch results, state, and cleanup report;
  - disconnect observation;
  - presentation delay;
  - bounded generation and final-evaluation work;
  - model construction and persistence;
  - monotonic/deadline behavior only where the existing public group seam does not already cover it.
- Keep controlled doubles behaviorally faithful:
  - one active epoch at a time;
  - no hidden pipelining;
  - exact public update/report schedule;
  - explicit cleanup outcome;
  - no private resource names.
- Create an autouse test fixture that restores the module-local run slot to an available state after every test, even if a test fails.
- Redirect persistence to `tmp_path` or replace the persistence seam in route-unit cases so tests never touch production `.data`.
- Add a small sentinel exception containing fake path, shared-memory name, protocol detail, traceback marker, and numerical state. Reuse it to prove client-visible privacy across failure stages.
- Add a call-order recorder for:

  ```text
  validate → reserve → preprocess/init → init yield → initialize → worker startup
  → epoch compute → Adam commit → sample → epoch yield → delay
  → final evaluation → worker cleanup success → model build → persist → done
  → slot release
  ```

- Do not require private route helper names or one exact monkeypatch mechanism.

**Guardrails:**

- Do not build expected results by calling the production route implementation under test.
- Do not assert generated process IDs, pipe identities, shared-memory names, temporary filenames, exact scheduling, or elapsed wall-clock time.
- Do not replace all public Transformer boundaries in the eventual bounded integration case.
- Do not add frontend rendering or TypeScript execution to Python tests.

**Expected result:**

- The new tests define the exact endpoint contract and initially fail only at missing schema/registration/orchestration behavior.
- Existing persistence, numerical, and worker tests remain independent.

**Verification:**

```powershell
poetry run pytest tests/test_train_transformer_route.py -q
```

Expected initial result:

- Focused tests fail because the request model and registered route are absent, not because the test seam depends on private implementation details.

## Step 2 — Add the exact strict Train Transformer request model

**Files and symbols:**

- `src/how_llms_work/schemas.py`
  - new `TrainTransformerRequest`.
- `tests/test_train_transformer_route.py`
  - validation/default/boundary tests.

**Purpose:**

Ensure FastAPI/Pydantic rejects malformed bodies before the endpoint can acquire the run slot or invoke preprocessing, streaming, worker, or shared-memory code.

**Actions:**

- Add one dedicated `BaseModel` with model-local `ConfigDict(extra="ignore")`.
- Preserve exactly these public fields and defaults:
  - `epochs`: strict integer, default `300`, range `50..2000`;
  - `temperature`: strict finite JSON number, default `0.8`, range `0.1..2.0`;
  - `top_p`: alias `topP`, strict finite JSON number, default `0.9`, range `0.1..1.0`;
  - `num_layers`: alias `numLayers`, strict integer, default `2`, range `1..6`;
  - `max_tokens`: alias `maxTokens`, strict integer, default `40`, range `3..500`.
- Use field-level strictness and finite-number enforcement local to this model. Do not enable global strict or alias behavior that could change existing request models.
- Treat ordinary JSON integer-valued numbers as valid for `temperature`/`topP` when they satisfy the existing frontend's JSON-number contract, while rejecting Booleans and strings. Protect this explicitly because Pydantic strict-float behavior must not accidentally narrow or coerce the TypeScript `number` contract.
- Parameterize exact minimum, maximum, defaults, representative middle values, and ignored extras.
- Parameterize rejection of:
  - numeric strings;
  - Booleans in every numerical field;
  - fractional values in integer fields;
  - `NaN` and positive/negative infinity;
  - values just below and above every range;
  - non-object bodies and unsupported container values.
- For every invalid body, assert:
  - HTTP `422`;
  - normal JSON validation response, not SSE;
  - no run-slot acquire attempt;
  - no preprocessing/init call;
  - no worker-group or shared-memory creation;
  - no persistence call.
- Preserve `ChatRequest`, `NeuralNetRequest`, and `TrainEmbedRequest` exactly.

**Guardrails:**

- Do not add fields for corpus, seed, optimizer, learning rate, dimensions, heads, feed-forward width, context, sequence length, workers, shards, timeouts, paths, checkpoints, or concurrency policy.
- Do not normalize or reinterpret submitted values after validation.
- Do not create a custom 422 body or SSE validation event.

**Expected result:**

- FastAPI owns malformed-input rejection before route execution.
- Valid payloads arrive at the endpoint with exact request-owned values and aliases.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    -q `
    -k "request or validation or default or boundary or extra"
```

## Step 3 — Register the endpoint and implement nonblocking reservation plus pre-stream `init`

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - `logger`;
  - `router`;
  - process-local nonblocking run slot;
  - exact `init` payload builder;
  - route endpoint;
  - stream ownership transfer.
- `src/how_llms_work/main.py`
  - router import and registration.
- `tests/test_train_transformer_route.py`
  - registration, `422`, `429`, pre-stream `500`, headers, and `init` ordering.

**Purpose:**

Expose the exact FastAPI boundary while guaranteeing that validation occurs before reservation, overlap is rejected immediately, and all replaceable failures occur before SSE starts.

**Actions:**

- Add an `APIRouter` to the existing persistence module without renaming or weakening any Ticket 021 public persistence operation.
- Add one module-local process-scoped synchronization primitive with nonblocking acquire semantics. A `threading.Lock` is appropriate if live inspection confirms no existing route-level abstraction.
- Ensure the endpoint function is entered only after Pydantic validation.
- Acquire the slot with no await, queue, retry, sleep, or waiting list.
- On an unavailable slot, raise sanitized HTTP `429` immediately with no streaming response.
- After acquisition and before returning `StreamingResponse`:
  - capture every validated request value into local immutable scalars;
  - retrieve `get_transformer_preprocessing()`;
  - construct or retrieve the canonical layout for `numLayers` without allocating shared memory;
  - calculate `totalParams` from the canonical layout/public count boundary;
  - build the complete exact `init` dictionary.
- The `init` payload must contain exactly, in stable insertion order:

  ```text
  vocabSize
  contextLen
  embeddingDim
  numHeads
  ffDim
  numLayers
  totalParams
  temperature
  topP
  corpusSentences
  trainingSequences
  ```

- Derive values only from validated request data, immutable preprocessing, and fixed public architecture constants.
- If preprocessing, layout, parameter-count, or `init` construction fails:
  - release the run slot;
  - return a generic HTTP `500`;
  - emit no SSE framing;
  - create no worker, pipe, process, cancellation helper, or shared-memory block;
  - expose no internal exception content.
- Transfer ownership of the acquired slot to the stream lifecycle only after all pre-stream preparation succeeds.
- Return the stream exclusively through `create_sse_response()`.
- Make the first generator action yield exactly one `init` event before fresh weights, worker startup, or shared-memory allocation.
- Add the router import and `include_router()` call to `main.py` while preserving every existing route.
- Test an overlap using one controlled open stream that owns the slot and a second valid request:
  - second response is immediate `429`;
  - no queue;
  - no SSE body;
  - no preprocessing/worker/persistence for the rejected request.
- Test slot release after a pre-stream `500` and after stream completion/failure so the next valid request can start.

**Guardrails:**

- Do not allocate NumPy shared memory, spawn workers, or create pipes before `init` is yielded.
- Do not acquire the run slot in middleware or before Pydantic validation.
- Do not use an `asyncio.Lock` with awaiting semantics for overlap; the requirement is immediate rejection, not serialized queuing.
- Do not introduce a machine-wide or cross-process lock.
- Do not include weights, paths, worker count, shard count, timeout values, or hidden state in `init`.

**Expected result:**

- The route is registered and produces exact shared-SSE headers.
- Invalid requests fail with `422`, overlapping valid requests fail with `429`, replaceable preparation failures fail with sanitized `500`, and a successful stream begins with exact `init` before resource allocation.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    -q `
    -k "registered or header or init or reservation or overlap or pre_stream or status"
```

## Step 4 — Create fresh training state and orchestrate one non-pipelined inclusive epoch loop

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - post-`init` fresh initialization;
  - `TransformerTrainingRun` creation;
  - Request-Scoped Worker Group factory call;
  - sequential epoch loop;
  - exact epoch event and sample collection.
- `tests/test_train_transformer_route.py`
  - lifecycle order, report schedule, no overlap/pipelining, sample, and delay tests.

**Purpose:**

Connect the completed deterministic parent training state to the completed request-scoped spawned worker boundary without duplicating numerical or process behavior.

**Actions:**

- After yielding `init` and confirming the client remains connected:
  - create one fresh `Mulberry32(42)` initialization stream;
  - initialize fresh parameters through `initialize_transformer_parameters()`;
  - create one fresh `TransformerTrainingRun` with the immutable sequence count and exact requested inclusive final epoch;
  - create one Request-Scoped Worker Group through `create_request_scoped_worker_group()` using the run's current weights, immutable Training Sequences, canonical logical shards, and the route's poll observer.
- Do not read any Saved Transformer Model before initialization.
- Do not skip, resume, fine-tune, or checkpoint training.
- Keep all mutable run state request-local:
  - parameters;
  - Adam moments and reduction workspaces inside the Training Run;
  - worker group and its five shared-memory blocks;
  - cancellation state;
  - active helper task;
  - samples;
  - temporary persistence ownership.
- Advance exactly one epoch at a time:
  1. identify `training_run.next_epoch`;
  2. await `worker_group.compute_epoch(epoch, training_run.weights)`;
  3. pass the complete independent four-shard result collection to `training_run.advance_epoch()`;
  4. begin no later epoch until the prior result has committed and any report-boundary work has completed.
- Never send more than one compute command per worker per epoch and never pipeline epochs.
- Rely on `TransformerTrainingRun` for:
  - exact four-shard canonicalization;
  - ordered `0 → 1 → 2 → 3` reduction;
  - one parent-side Adam update;
  - finite-state checks;
  - inclusive epoch state;
  - report schedule.
- When `observation.update` is present:
  - create exactly one sample through the bounded helper boundary described in Step 5;
  - append `{"epoch": report_epoch, "text": generated_text}` to a request-owned list;
  - emit an `epoch` payload containing exactly:

    ```text
    epoch
    loss
    sample
    ```

  - use the existing six-decimal `update.loss` rather than rerounding hidden state;
  - request `presentation_sleep(0.02)` exactly once after yielding the event.
- Do not delay after `init` or `done`.
- Test minimum `50`, default `300`, maximum `2000`, and a non-divisible reporting case with controlled lightweight worker/sample seams rather than executing full production work.
- Assert epoch zero and the exact final epoch appear once; assert no duplicate or missing report boundaries.
- Assert `temperature`, `topP`, and `maxTokens` reach sample generation only and never change initialization, worker epoch inputs, Training Run updates, weights, or losses.

**Guardrails:**

- Do not implement report arithmetic in the route.
- Do not mutate worker shared weights directly from route code.
- Do not average gradients or losses in the route.
- Do not access worker protocol messages or shared-memory arrays directly.
- Do not use a Queue, Manager, process pool, persistent worker pool, dynamic shards, or global start-method mutation.

**Expected result:**

- Every successful controlled run advances inclusively and serially through the existing public worker and parent-training seams.
- Every public report produces exactly one sample, one epoch event, one ordered sample record, and one delay request.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    -q `
    -k "epoch or report or sample or delay or sequential or fresh or generation_controls"
```

## Step 5 — Implement bounded helper-task ownership, disconnect observation, and cancellation propagation

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - request-owned `threading.Event`;
  - route-level disconnect seam;
  - worker-group poll observer;
  - bounded generation/final-evaluation helper wrapper;
  - active-helper draining;
  - cancellation-safe stream cleanup.
- `tests/test_train_transformer_route.py`
  - disconnect, timeout, helper-thread, and cancellation tests.

**Purpose:**

Prevent parent-side numerical helpers from blocking the event loop or outliving the numerical/shared-memory resources they may read, while preserving true `asyncio.CancelledError` control flow.

**Actions:**

- Create one request-owned cooperative `threading.Event` after `init` and before any helper or worker work.
- Define a narrow asynchronous `request_is_disconnected()` seam around `Request.is_disconnected()` for controlled tests.
- Supply a poll observer to `create_request_scoped_worker_group()` that:
  - runs after every existing `0.1`-second process poll;
  - checks the current request only;
  - sets the request cancellation event on disconnect;
  - stops the current worker-group operation through a sanitized controlled failure path;
  - exposes no route, process, or resource details.
- Check disconnect explicitly:
  - immediately after `init`;
  - before worker-group creation;
  - before each epoch compute where no poll has yet occurred;
  - before each sample;
  - after each sample and before event emission;
  - before final evaluation;
  - after final evaluation;
  - before worker cleanup completion is accepted;
  - before model construction;
  - before persistence;
  - before `done`.
- Create a bounded same-process helper wrapper for generation and final evaluation:
  - start the callable with `asyncio.to_thread()` in a tracked `Task`;
  - use a five-minute monotonic timeout;
  - use shielding or equivalent ownership so timeout/caller cancellation does not lose the underlying thread handle;
  - on disconnect, timeout, ordinary failure, or task cancellation, set the request cancellation event;
  - await the active helper task to return before cleaning worker/shared-memory or releasing run-owned numerical state;
  - convert timeout and helper failure into ordinary quiet stream failure, not client data.
- Keep at most one active parent helper thread per request.
- Ensure a completed helper task is cleared before later work.
- On `asyncio.CancelledError`:
  - set cooperative cancellation;
  - wait for the active helper to finish;
  - complete required worker cleanup;
  - release the run slot last;
  - re-raise the same cancellation rather than logging it as an ordinary failure.
- Catch ordinary `Exception` only after `init`, log internally through the module logger, and terminate quietly.
- Add controlled tests for:
  - disconnect during startup polling;
  - disconnect during epoch polling;
  - disconnect before sample;
  - disconnect while a sample helper is active;
  - sample timeout;
  - final-evaluation timeout;
  - cancellation while a helper is active;
  - helper confirms cancellation and returns before memory cleanup;
  - no persistence or `done` on any case.

**Guardrails:**

- Do not use `asyncio.wait_for(asyncio.to_thread(...))` in a way that discards an uncancelled underlying thread after timeout.
- Do not pretend Python can force-stop a running thread.
- Do not release worker/shared-memory or run-owned numerical arrays while an active helper could still access them.
- Do not catch `BaseException`.
- Do not convert `asyncio.CancelledError` into `RequestScopedWorkerGroupError` or a generic stream failure.
- Do not add a new SSE `error` event.

**Expected result:**

- The event loop regains control during generation/evaluation.
- Disconnect, timeout, and cancellation stop later stages.
- Active helper threads are drained before numerical resource release.
- Cancellation propagates only after cleanup.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    -q `
    -k "disconnect or timeout or helper or cancellation or cancelled"
```

## Step 6 — Gate completion on final evaluation and fully successful worker cleanup

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - final-evaluation stage;
  - idempotent worker cleanup stage;
  - cleanup diagnostics logging;
  - forced-termination failure gate.
- `tests/test_train_transformer_route.py`
  - final-loss order, cleanup success/failure, forced termination, and stage-order tests.

**Purpose:**

Ensure `done` cannot be emitted and persistence cannot occur when final weights were not evaluated or the request's process resources did not close successfully.

**Actions:**

- After `training_run.is_complete` and the inclusive final Adam update:
  - verify the client is still connected;
  - run `evaluate_transformer_final_loss()` through the bounded helper wrapper using the same request cancellation event;
  - use the returned six-decimal final value, never `last_completed_loss` or the final epoch's pre-update loss.
- Complete `worker_group.cleanup()` before model persistence.
- Treat cleanup as idempotent and call it from the common lifecycle finalizer even if an earlier stage already requested cleanup.
- Preserve the original outcome:
  - an epoch/generation/evaluation failure remains the primary outcome;
  - cleanup failures are secondary diagnostics;
  - cleanup failure must not replace a previously observed exception in logs or tests.
- Log cleanup report categories without exposing process IDs, pipes, shared-memory names, paths, or numerical state.
- Require `worker_group.successful` before proceeding to model construction or persistence.
- Therefore, prevent persistence and `done` when cleanup reports any of:
  - cooperative shutdown incomplete;
  - terminate required;
  - kill required;
  - nonzero/unknown exit inconsistent with success;
  - secondary cleanup failure;
  - primary worker-group failure.
- This ordering is mandatory because Ticket 022 states that any forced process termination marks the run failed and prevents persistence and completion.
- Add tests where:
  - final epoch update succeeds but final evaluation fails;
  - final evaluation succeeds but cooperative cleanup requires terminate;
  - cleanup requires kill;
  - cleanup returns a secondary failure;
  - cleanup itself is cancelled and later finishes;
  - all such cases produce no model write and no `done`.
- Add an order test proving:

  ```text
  final Adam commit → final evaluation → worker cleanup success
  → model build → persistence → done
  ```

**Guardrails:**

- Do not emit `done` and then clean workers afterward.
- Do not persist before knowing whether forced termination was required.
- Do not infer success merely because all epoch events were emitted.
- Do not bypass public `worker_group.successful` with private cleanup internals.

**Expected result:**

- Completion means final weights were evaluated and all request-owned worker/shared-memory resources closed cooperatively without a primary or cleanup failure.
- Forced or failed cleanup cannot produce a false persisted model or `done` event.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    -q `
    -k "final_loss or cleanup or terminate or kill or forced or completion_gate"
```

## Step 7 — Persist the final model before exact `done` and harden all post-stream failures

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - Saved Transformer Model construction call;
  - persistence call;
  - exact architecture and `done` payload;
  - common ordinary-failure and cleanup finalizer.
- `tests/test_train_transformer_route.py`
  - persistence order, exact payloads, failure privacy, slot release, and isolation.
- `tests/test_train_transformer_persistence.py`
  - regression only; modify only if a real public compatibility defect is discovered.

**Purpose:**

Make the final event a trustworthy statement that the complete configuration-specific artifact exists and every request-owned resource has been safely resolved.

**Actions:**

- After final evaluation and successful worker-group cleanup:
  - check disconnect;
  - call `build_saved_transformer_model(training_run, preprocessing)`;
  - check disconnect;
  - persist through the existing `save_transformer_model()` boundary with the exact requested `epochs` and authoritative model `numLayers` configuration;
  - check disconnect;
  - only then format and yield `done`.
- Use thread offloading for blocking model serialization/filesystem persistence where appropriate, but do not invent a new public persistence API or timeout not approved by the ticket.
- The `done` payload must contain exactly:

  ```text
  architecture
  finalLoss
  samples
  ```

- Use the exact TypeScript-compatible architecture text:

  ```text
  Decoder-Only Transformer (<numLayers> layers, 32d, 2h, 128ff)
  ```

- Do not add cached, resumed, fine-tuned, checkpoint, worker, or persistence labels.
- Use the same request-owned ordered sample list already represented by all emitted epoch events; return fresh public containers if needed to prevent mutation aliasing.
- Yield exactly one `done` and return immediately with no later event or presentation delay.
- Handle ordinary failures after `init` from every stage:
  - initialization;
  - worker startup;
  - worker compute/protocol/timeout;
  - Ordered Gradient Reduction or Adam;
  - non-finite state;
  - generation;
  - final evaluation;
  - cleanup;
  - Saved Transformer Model construction;
  - JSON serialization;
  - directory/temp creation;
  - write/flush/file `fsync`/close;
  - replacement;
  - owned-temp cleanup.
- For every ordinary post-stream failure:
  - set cooperative cancellation;
  - drain any active helper;
  - attempt every cleanup stage;
  - log primary and secondary outcomes separately;
  - emit no `done` and no SSE `error`;
  - include no exception message, traceback, path, resource name, protocol value, or numerical state in SSE data.
- Release the module-local run slot as the final lifecycle action after:
  - active helper completion;
  - worker cleanup;
  - persistence temporary-file resolution;
  - all required logging.
- Prove slot release after:
  - success;
  - disconnect;
  - every controlled failure stage;
  - timeout;
  - forced termination;
  - cancellation after propagation is observed.
- Prove sequential requests receive fresh:
  - initial weights;
  - Training Run/Adam state;
  - worker group/processes/pipes/shared memory;
  - cancellation event;
  - samples;
  - temporary files.
- Prove only immutable preprocessing is reused.
- Prove controlled overlapping request data cannot leak into the owning run.

**Guardrails:**

- Do not load, parse, compare, or resume from an existing model.
- Do not write intermediate checkpoints.
- Do not expose the Saved Transformer Model, parameter arrays, destination path, or temporary path in `done`.
- Do not release the run slot before cleanup/persistence resolution.
- Do not allow cleanup failure to suppress later cleanup stages.

**Expected result:**

- A successful stream is exactly `init → epoch × N → done`.
- Persistence is complete before `done` is formatted/yielded.
- Every unsuccessful post-`init` path terminates quietly, cleans comprehensively, and permits a later run only after cleanup is complete.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    -q `
    -k "done or architecture or persist or failure or privacy or slot or sequential or isolation"
```

## Step 8 — Add one bounded real-public-boundary integration and completed-route regressions

**Files and symbols:**

- `tests/test_train_transformer_route.py`
  - one bounded integration case;
  - completed-route registration/regression checks.
- Existing public numerical, worker-group, and persistence modules.

**Purpose:**

Prevent a route suite made entirely of doubles from drifting away from the completed real Transformer interfaces, while keeping ordinary pytest practical.

**Actions:**

- Add at least one minimum bounded endpoint integration that reaches the real public shapes and ownership boundaries needed by the route.
- Keep it practical by controlling only expensive breadth, not by replacing every public boundary. A suitable integration should, after live inspection:
  - use the real canonical layout and deterministic initialization;
  - create a real `TransformerTrainingRun`;
  - advance it through canonical real `LogicalTrainingShardResult` objects or a real tiny worker-group runtime seam already approved by Ticket 020;
  - invoke real Generated Text Sample/final-evaluation/model boundaries against a deliberately bounded valid dataset where feasible;
  - persist through the real Ticket 021 boundary into `tmp_path`;
  - observe exact public SSE ordering.
- Do not run the full production corpus for 50 epochs merely to satisfy the integration label if it makes ordinary pytest impractical.
- Document in the test why the case is bounded and which real boundaries it covers.
- Keep all real spawn-process lifecycle coverage already present in `tests/test_transformer_worker.py` and `tests/test_transformer_worker_group.py`; do not duplicate maximum worker tests in the route suite.
- Add endpoint registration/regression checks for:
  - `GET /health`;
  - `POST /simple-chat`;
  - `POST /bpe-tokenize`;
  - `POST /neural-net`;
  - `POST /train-embed`;
  - `POST /train-transformer`.
- Reuse existing route tests for full observable behavior rather than copying them into the Transformer test module.
- Confirm the new request model does not change any earlier schema's validation.

**Guardrails:**

- Do not mark maximum `epochs=2000`, `numLayers=6` endurance as an ordinary route test.
- Do not use production `.data` in tests.
- Do not treat controlled route-unit tests as proof that real spawn cleanup still works; rerun the worker suites.
- Do not alter frontend code for backend integration.

**Expected result:**

- The endpoint suite proves exact orchestration quickly.
- At least one case catches interface drift across real public Transformer boundaries.
- Completed Learning Demo routes remain registered and behaviorally unchanged.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    tests/test_train_transformer_persistence.py `
    tests/test_transformer_training.py `
    tests/test_transformer_completion.py `
    tests/test_transformer_worker.py `
    tests/test_transformer_worker_group.py `
    -q

poetry run pytest `
    tests/test_simple_chat.py `
    tests/test_bpe_tokenize.py `
    tests/test_neural_net_route.py `
    tests/test_train_embed_route.py `
    tests/test_train_transformer_route.py `
    -q
```

## Focused verification plan

Run from the backend project root after implementation:

```powershell
poetry run pytest tests/test_train_transformer_route.py -q

poetry run pytest `
    tests/test_train_transformer_route.py `
    tests/test_train_transformer_persistence.py `
    tests/test_transformer_training.py `
    tests/test_transformer_completion.py `
    tests/test_transformer_worker.py `
    tests/test_transformer_worker_group.py `
    -q

poetry run pytest `
    tests/test_simple_chat.py `
    tests/test_bpe_tokenize.py `
    tests/test_neural_net_route.py `
    tests/test_train_embed_route.py `
    tests/test_train_transformer_route.py `
    -q

poetry run ruff format --check `
    src/how_llms_work/main.py `
    src/how_llms_work/schemas.py `
    src/how_llms_work/routes/train_transformer.py `
    tests/test_train_transformer_route.py

poetry run ruff check `
    src/how_llms_work/main.py `
    src/how_llms_work/schemas.py `
    src/how_llms_work/routes/train_transformer.py `
    tests/test_train_transformer_route.py

poetry run mypy src
```

If the live repository still treats Black as a separate check, also run:

```powershell
poetry run black --check `
    src/how_llms_work/main.py `
    src/how_llms_work/schemas.py `
    src/how_llms_work/routes/train_transformer.py `
    tests/test_train_transformer_route.py
```

Expected result:

- Exact request validation, `429` reservation, sanitized pre-stream failure, SSE field/order, report cadence, sample generation, delay requests, final-loss order, cleanup gate, persistence-before-`done`, disconnect/cancellation, privacy, slot release, and isolation tests pass.
- Completed Transformer numerical, worker, cleanup, and persistence suites remain green.
- Completed Learning Demo endpoint regressions remain green.
- Changed files satisfy formatting and lint checks.
- Strict mypy reports no issues in `src`.

## Full verification plan

Run after focused verification passes:

```powershell
poetry run pytest
poetry run ruff format --check .
poetry run ruff check .
poetry run mypy src
git diff --check
git diff
git status --short
```

Expected result:

- The complete suite passes, including bounded real spawn-process tests.
- Ruff formatting and lint checks pass.
- Strict mypy reports `Success: no issues found` or the current equivalent success output.
- `git diff --check` reports no whitespace errors.
- The final diff contains only planned Ticket 022 files and any narrowly justified conditional change.
- No generated model or temporary file appears in `.data` or the commit candidate.

## Manual two-server or Vite-proxy check

Record this separately from automated backend validation when practical.

Terminal 1:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"

poetry run uvicorn how_llms_work.main:app `
    --app-dir src `
    --reload `
    --host 127.0.0.1 `
    --port 8000
```

Terminal 2:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm dev
```

Practical checks:

- Open the Train Transformer page through the Vite app.
- Start the smallest practical valid run, for example `50 0.8 0.9 1 3`, understanding that real training may still be CPU-intensive.
- Confirm `init` renders before training progress.
- Confirm progress samples and losses appear in order.
- Confirm exactly one final architecture/final-loss result appears after the run completes.
- While one run is active, send a second valid direct request and confirm immediate HTTP `429`.
- Disconnect or cancel one practical run and confirm no completion is shown and a later request can start after cleanup.
- Confirm the expected configuration-specific model file is written only after successful completion.
- Record actual observations honestly; do not treat automated tests as proof of browser rendering or proxy behavior.

## Manual acceptance checklist

### Request and reservation

- [ ] Ticket 022 remains the only implementation scope.
- [ ] The endpoint is exactly `POST /train-transformer`.
- [ ] The request contains only `epochs`, `temperature`, `topP`, `numLayers`, and `maxTokens`.
- [ ] Defaults and inclusive bounds match the ticket exactly.
- [ ] Numeric strings, Booleans, fractional integer values, non-finite values, and out-of-range values return HTTP `422` before route/resource work.
- [ ] Unknown extra fields are ignored locally without changing earlier schemas.
- [ ] One process-local slot is acquired nonblockingly after validation.
- [ ] An overlapping valid request receives immediate `429`, no queue, and no SSE body.
- [ ] The slot covers preprocessing through final cleanup and is released last.

### Pre-stream behavior and `init`

- [ ] Preprocessing and complete `init` construction occur before returning `StreamingResponse`.
- [ ] Pre-stream failure returns sanitized `500` and releases the slot.
- [ ] No worker, pipe, process, helper thread, or shared memory is created on pre-stream failure.
- [ ] Shared `format_sse()` and `create_sse_response()` remain the only SSE transport boundary.
- [ ] Response media type and cache/proxy headers are exact.
- [ ] `init` is the first event and occurs before shared-memory allocation or worker startup.
- [ ] `init` contains exactly the eleven approved camelCase fields and no extras.

### Training and progress

- [ ] Every valid run creates fresh deterministic weights and never loads a model.
- [ ] One Request-Scoped Worker Group owns exactly five shared-memory blocks.
- [ ] Epochs advance inclusively from zero through the requested final epoch.
- [ ] The route sends no overlapping or pipelined epoch command.
- [ ] Every epoch uses committed results from all four Logical Training Shards.
- [ ] Ordered reduction and one parent Adam update occur before later work.
- [ ] Every report boundary generates exactly one independent epoch-seeded sample.
- [ ] Every `epoch` payload contains exactly `epoch`, six-decimal `loss`, and `sample`.
- [ ] Samples are collected as ordered `{epoch, text}` objects.
- [ ] One `0.02` presentation delay is requested after each epoch event and nowhere else.

### Finalization and persistence

- [ ] Final loss is recomputed from final post-Adam weights.
- [ ] Final evaluation uses bounded thread offloading and the request cancellation event.
- [ ] Worker cleanup completes before persistence is permitted.
- [ ] `worker_group.successful` is required before model build/persistence.
- [ ] Forced terminate, kill, nonzero exit, or cleanup failure prevents persistence and `done`.
- [ ] The complete Saved Transformer Model is built from the completed run.
- [ ] The exact configuration-specific destination is persisted atomically before `done`.
- [ ] `done` contains exactly `architecture`, `finalLoss`, and complete ordered `samples`.
- [ ] Architecture is exactly `Decoder-Only Transformer (N layers, 32d, 2h, 128ff)` for the selected `N`.
- [ ] Exactly one `done` is emitted and no later event or delay occurs.

### Failure, cancellation, and cleanup

- [ ] Worker polling remains off the event loop with existing `0.1` bounded waits.
- [ ] Browser disconnection is observed after every worker poll and before later blocking stages.
- [ ] Generation and final evaluation each use the approved five-minute deadline.
- [ ] One request-owned cooperative cancellation event controls parent helpers.
- [ ] An active helper thread is awaited before numerical/shared-memory release.
- [ ] Ordinary post-`init` failures emit no `done`, no SSE `error`, and no internal details.
- [ ] `asyncio.CancelledError` propagates after cleanup.
- [ ] Every cleanup stage is attempted despite earlier cleanup failure.
- [ ] Primary outcome is preserved and secondary cleanup failures are logged separately.
- [ ] Shared memory is released only after no worker remains alive.
- [ ] The run slot is the final resource released on every path.
- [ ] A failed or disconnected run cannot persist a model or appear complete.
- [ ] Sequential requests use fresh mutable state and reuse only immutable preprocessing.

### Verification and scope

- [ ] Focused endpoint tests use controlled seams rather than maximum training.
- [ ] At least one bounded integration reaches real public Transformer boundaries.
- [ ] Existing persistence, Transformer training/completion, worker, and worker-group tests pass.
- [ ] Health, Simple Chat, BPE, XOR, and Word2Vec route regressions pass.
- [ ] Complete pytest, Ruff formatting/lint, and strict mypy results are reported honestly.
- [ ] Browser/Vite-proxy result is recorded separately when practical.
- [ ] No frontend, dependency, lockfile, model-loading, checkpoint, queue, persistent-pool, or machine-wide-lock work enters the diff.

## Expected files changed

Likely changed:

```text
src/how_llms_work/schemas.py
src/how_llms_work/main.py
src/how_llms_work/routes/train_transformer.py
tests/test_train_transformer_route.py
```

Conditionally changed only if live implementation proves a genuine public-boundary defect that blocks Ticket 022:

```text
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
tests/test_transformer_completion.py
tests/test_transformer_worker_group.py
tests/test_train_transformer_persistence.py
```

Any conditional change must be minimal, directly documented, independently tested at the owning public seam, and must not move route orchestration into the ML/worker modules.

No new fixture is expected by default. Controlled route records can be defined locally in `tests/test_train_transformer_route.py`, while existing independent Transformer fixtures continue to own numerical evidence.

## Files not to change by default

```text
src/how_llms_work/__init__.py
src/how_llms_work/sse.py
src/how_llms_work/ml/__init__.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/word2vec.py
src/how_llms_work/routes/__init__.py
src/how_llms_work/routes/simple_chat.py
src/how_llms_work/routes/bpe_tokenize.py
src/how_llms_work/routes/neural_net.py
src/how_llms_work/routes/train_embed.py
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
022-stream-complete-transformer-training-runs-through-fastapi.md
llm_works_file_structure.md
```

## Risk notes and safeguards

1. **Risk:** Pydantic accepts numeric strings or Booleans through coercion.
   - **Safeguard:** Use model-local strict fields and parameterized HTTP `422` tests that prove the endpoint and run slot are untouched.
2. **Risk:** Strict float configuration rejects legitimate integer-valued JSON numbers for `temperature` or `topP`.
   - **Safeguard:** Protect the TypeScript `number` contract explicitly with valid JSON-number tests while still rejecting Booleans and strings.
3. **Risk:** The run slot is acquired before FastAPI validation.
   - **Safeguard:** Keep reservation inside the validated endpoint and assert no acquire attempt for every invalid request.
4. **Risk:** An async lock queues overlap instead of returning immediate `429`.
   - **Safeguard:** Use a nonblocking process-local primitive and a controlled concurrent endpoint test.
5. **Risk:** The slot leaks if pre-stream construction fails or a stream is never fully consumed.
   - **Safeguard:** Separate pre-stream ownership from stream ownership and test early disconnect/close paths plus a subsequent request.
6. **Risk:** Weight initialization or worker allocation occurs before the first `init` event.
   - **Safeguard:** Record call order and assert `init` yield precedes fresh numerical and worker-resource creation.
7. **Risk:** The route duplicates report scheduling, reduction, Adam, or final-loss logic.
   - **Safeguard:** Use only `TransformerTrainingRun`, its public observations, and existing completion operations.
8. **Risk:** The route pipelines epochs or commands while sample work is active.
   - **Safeguard:** One explicit sequential loop and tests that block one stage while asserting no later call begins.
9. **Risk:** `asyncio.wait_for(to_thread(...))` times out while the helper thread continues untracked.
   - **Safeguard:** Track and shield the task, set cooperative cancellation, and await it before resource release.
10. **Risk:** Browser disconnect is checked only between epochs, not during worker waits.
    - **Safeguard:** Pass the route's request-specific observer into the existing worker-group poll seam.
11. **Risk:** `asyncio.CancelledError` is swallowed by an ordinary failure handler.
    - **Safeguard:** Handle it in a dedicated branch, shield/drain cleanup, then re-raise.
12. **Risk:** A final epoch event is mistaken for successful completion.
    - **Safeguard:** Require final evaluation, successful group cleanup, model build, and persistence before `done`.
13. **Risk:** Worker cleanup runs after `done`, so forced termination becomes a false success.
    - **Safeguard:** Make `worker_group.successful` a pre-persistence and pre-`done` gate.
14. **Risk:** Cleanup failure replaces the original failure and hides the true outcome.
    - **Safeguard:** Preserve the first primary outcome and log immutable cleanup diagnostics separately.
15. **Risk:** Final loss is copied from the final pre-update shard loss.
    - **Safeguard:** Call only `evaluate_transformer_final_loss()` after `training_run.is_complete` and test deliberately different values.
16. **Risk:** The emitted sample list differs from epoch events through mutation or rebuilding.
    - **Safeguard:** Maintain one ordered request-owned sample collection and assert exact equality with `done.samples`.
17. **Risk:** Architecture text drifts or adds cached/resumed labels.
    - **Safeguard:** Protect the exact TypeScript string for representative layer counts.
18. **Risk:** Persistence is called with a layer value that disagrees with the model config.
    - **Safeguard:** Use the validated request and Ticket 021's authoritative consistency checks; add a controlled mismatch failure test.
19. **Risk:** Internal exceptions leak paths, shared-memory names, protocol details, or numerical values in SSE.
    - **Safeguard:** Strict SSE parser plus sentinel secret markers across every failure stage.
20. **Risk:** Tests mock every boundary and miss real signature or ownership drift.
    - **Safeguard:** Include one bounded real-public-boundary endpoint integration and rerun all owning suites.
21. **Risk:** A real minimum production run makes ordinary pytest unreasonably slow.
    - **Safeguard:** Bound data/work through approved seams, retain real public interfaces, and leave maximum endurance manual.
22. **Risk:** Test concurrency leaves the module run slot locked for later tests.
    - **Safeguard:** Autouse teardown restores slot state and every open stream/test task uses `try/finally`.
23. **Risk:** Registering the router removes or shadows completed endpoints.
    - **Safeguard:** Add only one import/include call and rerun all existing route suites.
24. **Risk:** Extending the persistence module accidentally changes Ticket 021 behavior.
    - **Safeguard:** Preserve public persistence functions and run the complete focused persistence suite unchanged.
25. **Risk:** Runtime models or temp files enter the working tree.
    - **Safeguard:** Redirect automated persistence to `tmp_path`, inspect `.data`, `git status`, and the final diff.
26. **Risk:** TypeScript cached-model or checkpoint behavior is copied during vertical integration.
    - **Safeguard:** Initialize fresh state unconditionally and prohibit all model reads/intermediate writes.
27. **Risk:** Route implementation expands into a task queue, persistent pool, cross-process lock, or dependency change.
    - **Safeguard:** Enforce the expected-file list and use only existing public Python/NumPy/FastAPI boundaries.
28. **Risk:** Broad formatting obscures the final integration change.
    - **Safeguard:** Format only changed files first, run `git diff --check`, and reject unrelated churn.
29. **Risk:** User-reported baseline is presented as verified.
    - **Safeguard:** Re-run commands during implementation and report exact observed output only.

## Commit guidance after all checks pass

Do not create a commit during `to-plan-prompt`.

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Stream Transformer training through FastAPI
```

The commit body should mention:

- exact strict five-field request validation and ignored extras;
- process-local nonblocking run reservation with immediate `429` overlap rejection;
- pre-stream immutable preprocessing and exact eleven-field `init` construction;
- first-event-before-worker/shared-memory ordering;
- fresh deterministic parameters and one Request-Scoped Worker Group per run;
- sequential inclusive epoch coordination through four Logical Training Shards, Ordered Gradient Reduction, and parent-side Adam;
- bounded cooperative per-report generation and final evaluation;
- exact `init → epoch × N → done` shared-SSE contract and presentation delays;
- final post-Adam evaluation, successful worker cleanup gate, configuration-specific persistence-before-`done`, and exact architecture text;
- disconnect, timeout, cancellation propagation, active-helper draining, quiet failure privacy, exhaustive cleanup, and run-slot release last;
- endpoint isolation, bounded real-public-boundary integration, and completed-route regressions;
- no frontend, dependency, model-loading, checkpoint, queue, pool, dynamic-shard, or machine-wide-lock changes;
- the exact focused, full, Ruff, mypy, and manual commands actually executed and their observed results.

## Handoff to `implement-prompt`

Run `implement-prompt` in a fresh conversation using:

- this `plan022.md`;
- `022-stream-complete-transformer-training-runs-through-fastapi.md`;
- completed Tickets 017, 018, 020, and 021, or their current public implementations/tests as equivalent evidence;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `py_llm_pipeline_explorer_file_structure(86).md`;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the live repository again, establish its own baseline before editing, preserve user changes, implement Ticket 022 only, reuse every completed public Transformer/worker/persistence boundary, create strict failure-first route evidence, keep the run slot and request resources correctly owned, emit `init` before allocation, coordinate one epoch at a time, drain active helper threads before releasing numerical memory, require fully successful worker cleanup before persistence, persist before exactly one `done`, release the run slot last, run focused and full verification, report actual outcomes honestly, inspect final scope, and create the implementation commit only after every required check passes.
