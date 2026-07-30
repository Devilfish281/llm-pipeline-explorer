---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "011"
source_work_item: 011-stream-complete-embedding-training-runs-through-fastapi.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(25).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 011: Stream complete Embedding Training Runs through FastAPI

## Initial checklist

- Confirm Ticket 011 is the only selected work item and that its Ticket 008 and Ticket 010 blockers are represented by the completed persistence and result-construction boundaries in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(25).md` as the source of truth for current Python code, tests, configuration, and established conventions.
- Use `llm_works_file_structure.md` only as the TypeScript Reference Implementation for the unchanged request and SSE behavior.
- Preserve the user-reported passing pytest, Ruff, and strict mypy baseline without describing it as tool-verified in this planning session.
- Limit production changes to the dedicated request model, Train Embed route orchestration, router registration, and the smallest endpoint-level test seam.
- Reuse the existing deterministic Word2Vec Training Run, exact result/model builders, atomic persistence boundary, and shared SSE transport without rewriting them.
- Finish with focused endpoint tests, affected Word2Vec and persistence regressions, the complete pytest suite, Ruff, strict mypy, and a practical two-server browser or Vite-proxy check.

## Source-of-truth hierarchy

1. The user's latest explicit direction to convert the selected TypeScript behavior to Python and treat the latest complete Python Backend export as current-code truth.
2. `011-stream-complete-embedding-training-runs-through-fastapi.md` for required behavior, acceptance criteria, approved test seam, blockers, and scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(25).md` for the current implementation, tests, fixtures, paths, dependencies, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and ADR 0001 as recorded by the approved Phase 4 handoff for durable request, deterministic-compatibility, cancellation, and persistence-before-`done` decisions.
5. `llm_works_file_structure.md`, especially the Train Embed request schema, route, generator, frontend hook, and result component, as behavior evidence only.
6. Older code exports, snippets, tickets, and implementation plans are non-authoritative when they conflict with the sources above.

## Work-item summary

Complete the Phase 4 vertical slice by exposing `POST /train-embed` through the Python FastAPI application.

The endpoint must validate the exact five-field frontend request, capture independent request-owned Query Words and hyperparameters, obtain the immutable reference preprocessing, create one deterministic `EmbeddingTrainingRun`, and return the existing shared SSE response. The stream must emit one exact `init` event before training, advance the Training Run one public reporting interval at a time through same-process thread offloading, emit every exact `epoch` update with a requested `0.02`-second presentation delay, and cooperatively stop before starting later work once a disconnect is observed.

After the terminal `CompletedEmbeddingTraining` value is reached, the route must construct the exact frontend `EmbeddingResult` and complete `SavedEmbeddingModel` through the existing Ticket 010 public boundaries. It must persist the complete model through Ticket 008's atomic save boundary before emitting exactly one `done` event. The complete model, output weights, raw matrices, paths, and other internal state must never appear in `done`.

Ordinary post-stream training, result-construction, serialization, write, or replacement failures must be logged internally and terminate the stream quietly. They must not emit `done`, invent an SSE `error` event, or expose exception text, tracebacks, paths, or numerical state. Cancellation must continue to propagate and must not be converted into an ordinary failure.

This ticket does not change the deterministic Word2Vec mathematics, preprocessing, public result formulas, persistence contract, shared SSE format, frontend code, dependencies, Transformer modules, or completed Learning Demo behavior.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, mypy, browser, or two-server command was executed while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and report the actual results honestly.

## Current code observations from the latest source

- `src/how_llms_work/ml/word2vec.py` already contains the completed reusable Phase 4 core:
  - immutable `Word2VecPreprocessing`;
  - `get_word2vec_preprocessing()`;
  - deterministic request-owned `EmbeddingTrainingRun`;
  - `EmbeddingEpochUpdate`;
  - finite `CompletedEmbeddingTraining`;
  - `create_embedding_training_run()`;
  - `build_embedding_result()`;
  - `build_saved_embedding_model()`.
- `EmbeddingTrainingRun.__next__()` already performs bounded advancement to the next public report boundary or terminal value. It does not require the route to implement epoch mathematics.
- Existing Word2Vec tests already prove the deterministic PRNG, numerical update order, inclusive epoch behavior, reporting schedule including non-divisible totals, six-decimal public loss, finite-state enforcement, and sequential/concurrent run isolation.
- Existing result tests already prove the exact five-field `EmbeddingResult`, complete five-field `SavedEmbeddingModel`, Query Word position handling, warnings, neighbors, similarities, analogies, no output-weight exposure, finite values, and repeated/concurrent construction isolation.
- `src/how_llms_work/routes/train_embed.py` already contains Ticket 008's complete atomic persistence boundary:
  - `get_embedding_model_directory()`;
  - `serialize_saved_embedding_model()`;
  - unique same-directory temporary-file creation;
  - complete UTF-8 writing;
  - atomic replacement;
  - temporary-file cleanup;
  - `save_embedding_model()`.
- Existing Train Embed persistence tests already prove exact document formatting, non-finite serialization rejection, prior-model preservation, cleanup, same-directory temporary files, controlled concurrency, and last-successful-finisher behavior.
- `src/how_llms_work/routes/train_embed.py` does not yet define an `APIRouter`, request handler, stream generator, disconnect seam, presentation delay, logging boundary, or integration with the Word2Vec public operations.
- `src/how_llms_work/schemas.py` contains only `ChatRequest` and `NeuralNetRequest`; no dedicated Train Embed request model exists.
- `src/how_llms_work/main.py` registers Simple Chat, BPE Tokenizer, and Neural Network routers, but not Train Embed.
- `src/how_llms_work/sse.py` already owns the exact shared `text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`, JSON framing, and response-construction behavior required by Ticket 011.
- `src/how_llms_work/routes/neural_net.py` provides close repository prior art for:
  - one-interval-at-a-time `asyncio.to_thread()` advancement;
  - route-level presentation sleep;
  - disconnect observation;
  - thread-offloaded persistence;
  - persistence-before-completion;
  - ordinary `Exception` logging without catching `BaseException`.
- `tests/test_neural_net_route.py` provides the closest endpoint-test pattern: strict SSE parsing, controlled iterators, validation seams, thread/process observations, disconnect behavior, failure privacy, persistence-before-`done`, and request isolation.
- No `tests/test_train_embed_route.py` currently exists.
- `pyproject.toml` already declares all required runtime and development dependencies and configures pytest, Ruff, and strict mypy. No dependency or lockfile change is expected.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - Immutable reference preprocessing, complete ordered Vocabulary, sentence count, and window-specific Training Pair sequences.
  - Deterministic request-owned Embedding Training Runs and bounded iterator advancement.
  - Inclusive epoch reporting, epoch zero, requested final epoch, non-divisible schedules, and six-decimal loss payloads.
  - Exact Embedding Result and Saved Embedding Model construction.
  - Complete atomic persistence, prior-model preservation, temporary-file isolation, and concurrent save behavior.
  - Shared SSE media type, headers, and JSON framing.
  - Existing Health, Simple Chat, BPE Tokenizer, and Neural Network behavior has dedicated regression coverage.
- **Behavior present but evidence incomplete:**
  - The Neural Network route establishes the required orchestration pattern, but Train Embed does not yet exercise it with `init`, terminal result/model construction, or the Train Embed payloads.
  - Pydantic supports strict fields, aliases, list constraints, and ignored extras, but no dedicated project model currently encodes or tests the Train Embed contract.
  - Existing numerical and persistence modules are independently tested, but no registered FastAPI integration proves that the real public boundaries are invoked in persistence-before-`done` order.
- **Partially implemented:**
  - `routes/train_embed.py` owns complete persistence but not HTTP or SSE orchestration.
  - `main.py` has the application and completed router registrations but not Train Embed registration.
  - `schemas.py` owns shared request models but not the Train Embed model.
- **Not implemented:**
  - Exact five-field Train Embed request validation, aliases, defaults, strictness, and bounds.
  - HTTP `422` evidence before Training Run creation.
  - `POST /train-embed` registration.
  - Exact `init → epoch × N → done` streaming.
  - Train Embed presentation-delay requests.
  - Disconnect handling between bounded stages.
  - Train Embed cancellation-propagation evidence.
  - Quiet route-level handling of numerical, result, model-construction, serialization, write, and replacement failures.
  - Sequential and controlled concurrent endpoint isolation.
  - A focused endpoint test module and a real minimum-size route integration through the public Word2Vec and persistence boundaries.
  - A recorded manual two-server browser or Vite-proxy check.
- **Evidence limitation:**
  - The baseline is user-reported rather than tool-verified in this planning session.
  - The standalone ADR file was not attached to this specific prompt, but its deterministic-compatibility and persistence-before-`done` decisions are repeated by the selected ticket and Phase 4 specification.
  - A real browser or Vite-proxy result cannot be inferred from backend tests and remains an implementation-session manual check.
  - Cooperative thread offloading cannot forcefully terminate an already-started blocking call; the implementation must prevent later stages once a disconnect is observed at a route boundary rather than claim impossible thread termination.

## Files to inspect before editing

1. `src/how_llms_work/schemas.py` — `ChatRequest`, `NeuralNetRequest`, and the destination for the dedicated Train Embed request model.
2. `src/how_llms_work/main.py` — `app`, current router imports, current `include_router()` calls, and `health`.
3. `src/how_llms_work/routes/train_embed.py` — existing persistence symbols and the destination for `router`, endpoint orchestration, logging, presentation delay, disconnect handling, and SSE streaming.
4. `src/how_llms_work/ml/word2vec.py` — `EmbeddingEpochUpdate`, `CompletedEmbeddingTraining`, `EmbeddingTrainingEvent`, `Word2VecPreprocessing`, `get_word2vec_preprocessing()`, `create_embedding_training_run()`, `build_embedding_result()`, and `build_saved_embedding_model()`.
5. `src/how_llms_work/sse.py` — `format_sse()` and `create_sse_response()`; reuse without duplication.
6. `src/how_llms_work/routes/neural_net.py` — `advance_training_run()`, `request_is_disconnected()`, `stream_neural_network()`, route registration, logging, delay, and persistence ordering as prior art.
7. `tests/test_neural_net_route.py` — controlled iterator, strict SSE parser, worker-thread assertions, disconnect tests, failure privacy, and request-isolation patterns.
8. `tests/test_train_embed_persistence.py` — exact save behavior, failure injection, temporary-directory use, and controlled concurrency.
9. `tests/test_word2vec_training.py` — public reporting schedule, bounded advancement, finite failures, and Training Run isolation.
10. `tests/test_word2vec_results.py` — exact result/model field sets, query positions, warnings, no internal-state exposure, and conversion isolation.
11. `tests/test_simple_chat.py`, `tests/test_bpe_tokenize.py`, and `tests/test_neural_net_route.py` — completed-route regression behavior to preserve.
12. `pyproject.toml` — current dependencies and configured pytest, Ruff, Black, and strict-mypy settings; no dependency change is expected.
13. `011-stream-complete-embedding-training-runs-through-fastapi.md`, `SPEC.md`, `CONTEXT.md`, and ADR 0001 as recorded in the handoff — approved route behavior, canonical terminology, and scope.
14. `llm_works_file_structure.md` — TypeScript Train Embed request, generator `init`, event order, and frontend event handling as compatibility evidence only.

## Step 1 — Reconfirm the implementation-session baseline and current diff

**Files and symbols:**
- `pyproject.toml` — pytest, Ruff, and strict-mypy configuration.
- `src/how_llms_work/schemas.py` — confirm no newer Train Embed request model exists.
- `src/how_llms_work/main.py` — confirm no newer Train Embed router registration exists.
- `src/how_llms_work/routes/train_embed.py` — confirm persistence-only ownership remains current.
- `tests/` — confirm no newer Train Embed route test supersedes this plan.

**Purpose:**
Establish tool-verified pre-edit evidence and protect any user changes made after the exported snapshot.

**Actions:**
- Work from the backend project root.
- Inspect `git status --short` and the relevant current files before editing.
- Run the complete existing pytest suite, Ruff, and strict mypy before editing.
- Record exact commands, exit codes, and relevant output.
- Separate any pre-existing environment or test failure from Ticket 011 work rather than repairing unrelated code.

**Guardrails:**
- Do not describe the user-reported baseline as tool-verified.
- Do not modify a file during this baseline step.
- Do not add dependencies, regenerate `poetry.lock`, or format unrelated files.
- Stop and reassess the plan if the live repository materially differs from the supplied latest export.

**Expected result:**
- A grounded pre-edit baseline and scope record exists.
- The implementation session knows whether the four expected production/test files remain the smallest complete change.

**Verification:**

```powershell
git status --short
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Step 2 — Establish the exact request-validation contract

**Files and symbols:**
- `tests/test_train_embed_route.py` — new validation and request-factory tests through the registered FastAPI application.
- `src/how_llms_work/schemas.py` — new dedicated Train Embed request model.
- `src/how_llms_work/routes/train_embed.py` — Training Run factory reference patched by invalid-request tests.

**Purpose:**
Prove every request field, alias, default, strict type, and inclusive bound before adding successful streaming behavior.

**Actions:**
- Add a dedicated Train Embed route test module with a strict local SSE parser modeled on the existing route tests.
- Assert the request model exposes only the public fields `words`, `epochs`, `dimensions`, `windowSize`, and `negativeSamples`, accounting for explicit Pydantic aliases when Python attributes use snake_case.
- Define `words` as a required list containing one through ten strict strings, each with at least one character.
- Preserve each string exactly as submitted; add cases proving whitespace-only entries remain structurally valid and are not trimmed or removed.
- Define strict integer constraints and defaults:
  - `epochs`: `10` through `10_000`, default `10_000`;
  - `dimensions`: `4` through `64`, default `32`;
  - `windowSize`: `1` through `5`, default `2`;
  - `negativeSamples`: `1` through `10`, default `5`.
- Use exact camelCase input aliases for `windowSize` and `negativeSamples`.
- Keep model-local extra-field behavior as ignore and add a valid request containing an unknown field to prove it does not reach the Training Run arguments.
- Parameterize missing `words`, zero words, eleven words, empty-string entries, numeric strings, booleans, fractional values, and below/above-bound values.
- For every invalid case, assert HTTP `422`, JSON response content, no SSE framing, and no Training Run factory call.
- Use controlled terminal events for minimum, maximum, and default request tests so schema tests do not execute production-size training.

**Guardrails:**
- Do not reuse `ChatRequest` or alter `ChatRequest` or `NeuralNetRequest`.
- Do not trim, split, lowercase, deduplicate, or normalize Query Words in the schema.
- Do not expose seed, corpus, learning rate, paths, persistence switches, report controls, or other fields.
- Do not implement custom HTTP error bodies or convert validation failures into SSE events.
- Do not make aliases globally affect existing request models.

**Expected result:**
- Invalid bodies fail before streaming or Training Run creation.
- Valid bodies reach the route with exact request-owned values and approved defaults.
- Existing request-model behavior remains unchanged.

**Verification:**

```powershell
poetry run pytest tests/test_train_embed_route.py -q -k "request or validation or default or boundary or extra"
```

## Step 3 — Register `POST /train-embed` and establish the exact `init` SSE shell

**Files and symbols:**
- `src/how_llms_work/routes/train_embed.py` — `router`, endpoint, stream entry point, immutable request capture, and exact `init` event.
- `src/how_llms_work/main.py` — Train Embed router import and `app.include_router(...)`.
- `src/how_llms_work/sse.py` — existing shared response and framing operations.
- `tests/test_train_embed_route.py` — registration, headers, `init`, and existing-route regression tests.

**Purpose:**
Expose the missing endpoint without disturbing completed Learning Demos and prove that the stream begins with the exact shared-SSE contract before any training interval starts.

**Actions:**
- Add an `APIRouter` to the existing Train Embed route module while preserving every persistence function and its public names.
- Add `POST /train-embed` accepting the dedicated request model and FastAPI/Starlette `Request`.
- Capture the submitted Query Words into a request-owned immutable sequence and capture each scalar hyperparameter before returning the streaming response.
- Obtain the immutable `Word2VecPreprocessing` through its existing public operation.
- Create one independent `EmbeddingTrainingRun` through `create_embedding_training_run()` using the validated hyperparameters.
- Build the `init` payload only from confirmed preprocessing and request values:
  - `vocabSize`;
  - `sentenceCount`;
  - `embeddingDim`;
  - `windowSize`;
  - `totalPairs`.
- Emit exactly one `init` before advancing the Training Run.
- Do not request a presentation sleep after `init`.
- Return the stream through `create_sse_response()` rather than constructing route-local headers or framing.
- Import and include the Train Embed router in `main.py` without removing or reordering existing observable routes.
- Add registration checks for `/health`, `/simple-chat`, `/bpe-tokenize`, `/neural-net`, and `/train-embed`.

**Guardrails:**
- Do not duplicate `format_sse()`, `create_sse_response()`, SSE constants, or headers.
- Do not derive `totalPairs` by mutating or rebuilding shared preprocessing.
- Do not begin a worker-thread interval before `init` is yielded.
- Do not write to the real `.data` directory in controlled route tests.
- Do not edit frontend or Vite configuration.

**Expected result:**
- A valid request returns HTTP `200` with the established SSE headers.
- The first and only first event is exact `init`.
- Existing Learning Demo routes remain registered and preserve their current validation or health behavior.

**Verification:**

```powershell
poetry run pytest tests/test_train_embed_route.py -q -k "registered or headers or init or preserved"
```

## Step 4 — Implement bounded threaded progress and persistence-before-`done`

**Files and symbols:**
- `src/how_llms_work/routes/train_embed.py` — bounded Training Run advancement, progress delay, terminal conversion, persistence, and `done`.
- `src/how_llms_work/ml/word2vec.py` — existing public events and builders, consumed without modification.
- `tests/test_train_embed_route.py` — controlled success stream, call order, delays, worker-thread, and real-boundary integration tests.

**Purpose:**
Deliver the successful tracer bullet while keeping CPU-bound work out of the async route thread and preserving exact public event order.

**Actions:**
- Add a narrow route-owned operation that advances one `EmbeddingTrainingEvent` by calling `next()` on the current request's iterator.
- Before starting each later blocking stage after `init`, observe the current request's disconnect state.
- Advance exactly one public reporting interval at a time with `asyncio.to_thread()`; never pass the whole iterator to `list()`, a complete training helper, or one uncancellable full-run operation.
- When the event is `EmbeddingEpochUpdate`:
  - emit exact `epoch` with only `epoch` and `loss`;
  - request `presentation_sleep(0.02)` exactly once;
  - return to the async orchestration loop before starting the next interval.
- When the event is `CompletedEmbeddingTraining`:
  - do not emit it directly;
  - construct the `EmbeddingResult` through `build_embedding_result()` using the captured Query Words and immutable preprocessing;
  - construct the complete `SavedEmbeddingModel` through `build_saved_embedding_model()`;
  - perform terminal CPU work through bounded same-process thread offloading where needed;
  - recheck disconnect state between completed bounded stages so an observed disconnect prevents starting later construction or persistence;
  - persist through `save_embedding_model()` in a worker thread;
  - emit one `done` containing only the exact Embedding Result after persistence returns successfully;
  - return immediately so no later event is possible.
- Build the frontend result and complete model before persistence so a result/model-construction failure cannot replace the previous saved model.
- Add a controlled successful stream test asserting exact `init → epoch × N → done`, exact field sets, one `done`, no later event, no model or internal state in the response, and no sleeps after `init` or `done`.
- Observe worker and route thread identifiers to prove interval advancement remains in the FastAPI process but outside the async orchestration thread.
- Add one focused minimum-size endpoint integration using the real public Word2Vec run, real result/model builders, and real save boundary redirected to `tmp_path`; patch only presentation sleep and the model-directory boundary.
- Use controlled iterators for default and maximum-contract cases so the suite does not execute unnecessary production-size runs.

**Guardrails:**
- Do not change `EmbeddingTrainingRun`, public result builders, or persistence semantics to simplify route tests.
- Do not expose `CompletedEmbeddingTraining`, output weights, raw matrices, Saved Embedding Model, destination path, or temporary path.
- Do not request presentation sleep based on wall-clock assertions; tests inspect requested values.
- Do not create an executor, alter the global thread pool, use multiprocessing, or add a queue, lock, semaphore, timeout, quota, or rate limiter.
- Do not persist before both public-object constructions succeed.
- Do not emit `done` before successful atomic replacement.

**Expected result:**
- Successful requests stream exact reference-compatible progress and completion.
- The event loop regains control between public reporting intervals.
- The complete model exists before the single `done` event is emitted.
- Automated tests remain fast by using controlled seams except for one minimum-size real-boundary integration.

**Verification:**

```powershell
poetry run pytest tests/test_train_embed_route.py -q -k "success or delay or worker or persists or integration"
```

## Step 5 — Enforce disconnect, cancellation, and failure privacy

**Files and symbols:**
- `src/how_llms_work/routes/train_embed.py` — disconnect boundary, ordinary failure logging, and cancellation propagation.
- `tests/test_train_embed_route.py` — disconnect, cancellation, training failure, result failure, model failure, and persistence failure tests.
- `tests/test_train_embed_persistence.py` — existing prior-model preservation evidence reused by route integration.

**Purpose:**
Prevent false completion, abandoned later work, prior-model damage, and client-visible internal details after streaming begins.

**Actions:**
- Check the current request's disconnect state after `init` and before starting each later Training Run interval or terminal blocking stage.
- Treat a completed already-started bounded call as the maximum unavoidable work before the next disconnect observation; once disconnected is observed, return without starting later work.
- Add a controlled disconnect test that:
  - emits `init` and any already-completed public update allowed by the selected boundary;
  - starts no later interval after disconnect is observed;
  - calls no result/model builder or persistence operation;
  - emits no `done` and no `error` event.
- Catch ordinary `Exception` around post-`init` stream work, log with the route logger, and return quietly.
- Do not catch `BaseException`.
- Add explicit cancellation evidence showing `asyncio.CancelledError` is propagated rather than logged or transformed into a normal training failure. Prefer the registered endpoint seam; if the in-process client cannot deterministically inject task cancellation, add one narrow async test against an intentionally public stream boundary rather than a private helper.
- Inject and test each ordinary post-stream failure class separately:
  - numerical/Training Run advancement failure;
  - Embedding Result construction failure;
  - Saved Embedding Model construction failure;
  - serialization failure;
  - temporary write failure;
  - atomic replacement failure.
- For result/model-construction failures, assert persistence is never called.
- For persistence failures, use `tmp_path` and the real persistence boundary where practical; assert a previous model remains byte-identical and owned temporary files are cleaned when cleanup succeeds.
- In every failure response, assert:
  - no `done`;
  - no SSE `error`;
  - no injected failure marker;
  - no traceback;
  - no filesystem path;
  - no model or numerical state.
- Assert the injected failure marker is present in captured internal logs.

**Guardrails:**
- Do not invent an SSE error payload or change an already-started stream into a different HTTP status.
- Do not swallow or log cancellation as an ordinary route failure.
- Do not catch `BaseException` or use a bare `except`.
- Do not attempt to forcefully terminate a Python worker thread.
- Do not delete or roll back a prior valid destination after a failed replacement; rely on the existing atomic save contract.
- Do not weaken existing persistence tests by replacing them with route mocks.

**Expected result:**
- Disconnects and cancellations stop safely at cooperative boundaries.
- Ordinary internal failures are observable in logs but not in the client stream.
- No failed run falsely completes or damages the previous model.

**Verification:**

```powershell
poetry run pytest tests/test_train_embed_route.py tests/test_train_embed_persistence.py -q -k "disconnect or cancel or failure or preserve or cleanup or privacy"
```

## Step 6 — Prove sequential and controlled concurrent request isolation

**Files and symbols:**
- `tests/test_train_embed_route.py` — sequential and controlled concurrent endpoint tests.
- `src/how_llms_work/routes/train_embed.py` — request-owned captured values and local Training Run state.
- `src/how_llms_work/ml/word2vec.py` — existing independent run and pure conversion boundaries, consumed unchanged.

**Purpose:**
Show that one request cannot mix Query Words, hyperparameters, disconnect state, Training Run progress, result payloads, or persistence temporary files with another request.

**Actions:**
- Add sequential requests with distinct Query Words and all four hyperparameters; assert each Training Run factory call receives only its own values and each `done` payload uses only its own captured Query Words and controlled completion.
- Add a controlled concurrent endpoint test using barriers or events at the public route seams so both requests overlap without relying on timing.
- Use separate controlled Training Run instances and separate request-scoped disconnect outcomes.
- Capture result-builder, model-builder, and save inputs and assert no cross-request object or value is mixed.
- Redirect saves to `tmp_path` and rely on the existing unique temporary-file and last-successful-finisher behavior; assert the final destination is one complete valid model and no temporary files remain.
- Assert both response streams retain their own exact `init`, `epoch`, and `done` data.
- Keep shared `Word2VecPreprocessing` read-only and prove no route operation attempts to mutate it.

**Guardrails:**
- Do not add a global training lock, semaphore, queue, shared mutable request registry, or request ID.
- Do not assert a nondeterministic concurrent completion order unless the test explicitly controls it.
- Do not require separate processes or a custom thread-pool implementation.
- Do not make private temporary-file names or internal object identities part of the endpoint contract.

**Expected result:**
- Sequential and overlapping valid requests remain independent.
- Concurrent persistence leaves one complete document under the already-approved last-successful-finisher rule.
- Shared immutable preprocessing remains safe for reuse.

**Verification:**

```powershell
poetry run pytest tests/test_train_embed_route.py tests/test_train_embed_persistence.py -q -k "isolated or concurrent or sequential"
```

## Step 7 — Run affected regressions, complete quality checks, and manual acceptance

**Files and symbols:**
- `tests/test_train_embed_route.py` — complete Ticket 011 endpoint coverage.
- Existing Word2Vec, persistence, SSE-route, and completed Learning Demo tests — regression coverage.
- `src/how_llms_work/schemas.py`, `src/how_llms_work/main.py`, and `src/how_llms_work/routes/train_embed.py` — final scope inspection.
- `.data/`, `frontend/`, `pyproject.toml`, and `poetry.lock` — confirm no unintended changes.

**Purpose:**
Close the implementation with complete automated evidence, scope inspection, and a real frontend/proxy observation that backend tests cannot provide.

**Actions:**
- Run the focused Train Embed route tests first.
- Run affected Word2Vec training/result and Train Embed persistence tests.
- Run completed-route regressions for Health, Simple Chat, BPE Tokenizer, and Neural Network.
- Run the complete pytest suite once after all focused tests pass.
- Run Ruff and strict mypy through Poetry.
- Run `git diff --check` and inspect `git status --short`.
- Confirm no runtime `.data/embedding-weights.json`, temporary file, cache, frontend edit, dependency change, or lockfile change is included unintentionally.
- When practical, start FastAPI and Vite in separate PowerShell terminals and submit a minimum valid Train Embed request through the Vite proxy or browser.
- Record the actual manual result separately from automated test results.
- If the manual check is impractical in the implementation environment, state that limitation explicitly rather than claiming browser compatibility was verified.

**Guardrails:**
- Do not treat `TestClient` as proof of browser rendering or Vite proxy configuration.
- Do not fix unrelated warnings or reformat the repository broadly.
- Do not commit generated model files, cache directories, or temporary files.
- Do not claim any command passed without its actual successful output.

**Expected result:**
- All focused and full backend checks pass.
- Existing Learning Demos remain unchanged.
- The final diff contains only Ticket 011 work.
- A practical browser/proxy result or an explicit limitation is recorded.

**Verification:**

```powershell
poetry run pytest tests/test_train_embed_route.py -q

poetry run pytest `
    tests/test_word2vec.py `
    tests/test_word2vec_training.py `
    tests/test_word2vec_results.py `
    tests/test_train_embed_persistence.py `
    tests/test_simple_chat.py `
    tests/test_bpe_tokenize.py `
    tests/test_neural_net_route.py `
    -q

poetry run pytest
poetry run ruff check .
poetry run mypy src
git diff --check
git status --short
```

## Focused verification plan

```powershell
poetry run pytest tests/test_train_embed_route.py -q

poetry run pytest `
    tests/test_word2vec_training.py `
    tests/test_word2vec_results.py `
    tests/test_train_embed_persistence.py `
    -q
```

Expected result:

- Exact validation, aliases, defaults, SSE headers, `init`, progress, delay, completion, bounded thread offloading, disconnect, cancellation, failure privacy, persistence ordering, and request isolation pass.
- Existing deterministic training, result/model conversion, and persistence behavior remains unchanged.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
git diff --check
```

Expected result:

- The complete pytest suite passes.
- Ruff reports no violations.
- Strict mypy reports no issues under `src`.
- The final diff contains no whitespace errors.

## Manual acceptance checklist

- [ ] Start the backend with Poetry/Uvicorn and confirm `GET /health` still returns `{"status":"healthy"}`.
- [ ] Start the unchanged Vite frontend in a second PowerShell terminal.
- [ ] Submit a minimum valid Train Embed request through the Vite proxy or browser.
- [ ] Confirm the response opens as SSE with one `init`, progress updates including epoch zero and the requested final epoch, and one final `done`.
- [ ] Confirm Train Embeddings renders embeddings, neighbors, similarities, analogies, and warnings without exposing the complete model or paths.
- [ ] Confirm `backend/.data/embedding-weights.json` is complete valid JSON only after a successful run.
- [ ] Confirm Simple Chat, Basic Tokenizer, and Neural Network still open and perform their existing behavior.
- [ ] Record the exact manual commands and observed outcome, or explicitly record why the browser/proxy check was not practical.

## Expected files changed

Likely changed:

```text
src/how_llms_work/schemas.py
src/how_llms_work/main.py
src/how_llms_work/routes/train_embed.py
tests/test_train_embed_route.py
```

Conditionally changed only if live repository inspection reveals a genuine acceptance-evidence gap that cannot be covered in the new route test module:

```text
tests/test_neural_net_route.py
```

No new fixture is expected because route tests can use existing public Word2Vec fixtures, controlled terminal values, and one minimum-size real-boundary integration.

## Files not to change

```text
src/how_llms_work/sse.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/word2vec.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/simple_chat.py
src/how_llms_work/routes/bpe_tokenize.py
src/how_llms_work/routes/neural_net.py
src/how_llms_work/routes/train_transformer.py
tests/fixtures/
tests/test_word2vec.py
tests/test_word2vec_training.py
tests/test_word2vec_results.py
tests/test_train_embed_persistence.py
tests/test_simple_chat.py
tests/test_bpe.py
tests/test_bpe_tokenize.py
tests/test_neural_net.py
tests/test_neural_net_persistence.py
pyproject.toml
poetry.lock
.data/
frontend/
SPEC.md
CONTEXT.md
docs/adr/
011-stream-complete-embedding-training-runs-through-fastapi.md
```

A listed file may be changed only if current live repository evidence proves the plan's source snapshot is stale or a focused regression correction is genuinely required. Such a change must be explained and kept within Ticket 011.

## Risk notes and safeguards

1. **Risk:** Pydantic coerces numeric strings, booleans, or fractions into accepted integers.
   - **Safeguard:** Use strict integer constraints and parameterized HTTP `422` tests for every numeric field.

2. **Risk:** CamelCase fields drift into snake_case request names or aliases become globally permissive.
   - **Safeguard:** Use explicit field aliases on the dedicated model and assert the exact accepted public names without changing existing models.

3. **Risk:** A validator trims, splits, removes, or deduplicates Query Words.
   - **Safeguard:** Keep validation declarative, copy entries unchanged, and test whitespace-only and duplicate positions.

4. **Risk:** Training starts before the browser receives corpus metadata.
   - **Safeguard:** Assert `init` is the first event and that the Training Run has not advanced before it is yielded.

5. **Risk:** The entire remaining run is offloaded as one operation, defeating cooperative disconnect checks.
   - **Safeguard:** Call `next()` once per `asyncio.to_thread()` invocation and assert the controlled iterator's advance count.

6. **Risk:** Disconnect is observed but later conversion or persistence still starts.
   - **Safeguard:** Check before each blocking stage and assert no later dependency call after the controlled disconnect boundary.

7. **Risk:** Cancellation is caught by an overly broad handler and reported as an ordinary failure.
   - **Safeguard:** Catch `Exception`, never `BaseException`, and add explicit `CancelledError` propagation evidence.

8. **Risk:** A result or model-construction failure replaces the prior successful model.
   - **Safeguard:** Construct both public objects before starting persistence and assert the save boundary is untouched on either failure.

9. **Risk:** Persistence fails after progress and the route emits a false `done`.
   - **Safeguard:** Await successful `save_embedding_model()` completion before formatting `done`; test serialization, write, and replacement failures.

10. **Risk:** Internal exception text, paths, raw matrices, output weights, or the full model leak through SSE.
    - **Safeguard:** Use exact field-set assertions and injected unique failure markers that must appear only in internal logs.

11. **Risk:** Tests become slow by executing default or maximum Word2Vec runs and real presentation delays.
    - **Safeguard:** Use controlled iterators and patched route-level sleep for contract tests, with only one minimum-size real-boundary integration.

12. **Risk:** Route mocks bypass the actual Ticket 008 and Ticket 010 boundaries entirely.
    - **Safeguard:** Keep controlled unit-style route tests, then include one `tmp_path` endpoint integration using the real run, builders, and persistence.

13. **Risk:** Concurrent requests mix Query Words, hyperparameters, disconnect state, or terminal payloads.
    - **Safeguard:** Capture all request inputs locally, use separate iterators, and add controlled overlapping endpoint calls.

14. **Risk:** Shared preprocessing is accidentally mutated by route logic.
    - **Safeguard:** Read only from the existing immutable preprocessing object and retain the current mutation-isolation regressions.

15. **Risk:** Registering Train Embed removes or changes a completed route.
    - **Safeguard:** Preserve every existing `include_router()` call and run endpoint-level regressions for all completed Learning Demos.

16. **Risk:** New transport helpers drift from established SSE formatting and headers.
    - **Safeguard:** Reuse `format_sse()` and `create_sse_response()` unchanged.

17. **Risk:** Runtime model or temporary files enter the implementation commit.
    - **Safeguard:** Redirect automated writes to `tmp_path`, inspect `.data/` and `git status --short`, and exclude generated artifacts.

18. **Risk:** Ticket 011 expands into frontend changes, Transformer work, process infrastructure, or operational controls.
    - **Safeguard:** Restrict production edits to schema, registration, and Train Embed HTTP orchestration around completed public boundaries.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

A suitable outcome-oriented subject would be:

```text
Stream embedding training through FastAPI
```

Commit body should mention:

- exact Train Embed request validation and camelCase aliases;
- registered shared-SSE `init → epoch × N → done` behavior;
- bounded same-process thread offloading and presentation delays;
- cooperative disconnect handling and cancellation propagation;
- exact result/model construction with persistence-before-`done`;
- quiet post-stream failures and no internal-state exposure;
- sequential/concurrent request isolation and completed-route regressions;
- the exact focused, full, Ruff, mypy, and manual commands actually executed.

Do not create the commit during `to-plan-prompt`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 011, Tickets 008 and 010 as completed blockers, `SPEC.md`, `CONTEXT.md`, ADR 0001 as recorded in the approved handoff, `py_llm_pipeline_explorer_file_structure(25).md`, and the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the live repository again, establish its own baseline, preserve user changes, implement only Ticket 011, verify the complete change, report actual command and manual-check results honestly, and create the implementation commit.
