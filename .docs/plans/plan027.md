---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "027"
source_work_item: 027-show-the-actual-transformer-worker-process-count-during-training.md
source_specification: SPEC.md
source_context: CONTEXT.md
architecture_decisions:
  - 0002-stabilize-python-transformer-training-and-process-lifecycle.md
  - 0003-load-saved-transformer-models-for-stateless-generation.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(138).md
frontend_code_reference: ts_llm_pipeline_explorer_file_structure.md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 027: Show the Actual Transformer Worker-Process Count During Training

## Initial checklist

- Confirm Ticket 027 is the only selected work item. It is marked ready, has no blockers, and is limited to one presentation-only addition to fresh-weight Transformer Training Runs.
- Treat `py_llm_pipeline_explorer_file_structure(138).md` as the latest Python Backend current-code source of truth. Re-inspect the live repository before implementation and do not let older exports, prior plans, pasted snippets, or the historical TypeScript backend override it.
- Preserve the completed Ticket 026 lifecycle behavior represented in the current export, including the shared process-local Transformer request slot, request-scoped cleanup, deadline handling for saved-model generation, and release-last behavior.
- Reuse the existing actual-worker-count calculation and the exact value retained by the current `RequestScopedWorkerGroup`; do not call `os.cpu_count()` again from the route and do not recalculate or infer the count from logical shards.
- Add the label only at the public `epoch.sample` presentation boundary. Keep every raw `GeneratedTextSample.text` value unchanged before that boundary.
- Keep the `epoch` payload field set exactly `epoch`, `loss`, and `sample`, and keep the `done` payload field set exactly `architecture`, `finalLoss`, and `samples`.
- Keep Saved Transformer Generation Runs label-free. They create no Request-Scoped Worker Group and must never synthesize a zero or placeholder count.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Finish implementation with focused worker-group and route tests, unchanged numerical/persistence/load regressions, the complete backend test suite once, formatting and lint checks, strict mypy, and a scope-only Git diff review.

## Source-of-truth hierarchy

1. The user's latest explicit direction: plan Ticket 027 only, use the supplied current Python Backend export as the source of truth, and preserve compatibility while converting the approved behavior into Python.
2. `027-show-the-actual-transformer-worker-process-count-during-training.md` for the immediate scope, exact label text, first-sample-only rule, approved test seams, acceptance criteria, and exclusions.
3. `py_llm_pipeline_explorer_file_structure(138).md` for current source, tests, dependencies, public symbols, typing conventions, completed Tickets 023 through 026, and the exact present training/load route behavior.
4. `SPEC.md` for the durable Phase 6 decision that fresh-weight training remains authoritative and only the first public training sample receives the presentation prefix.
5. `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for the continuing authority over fresh initialization, one-through-four actual worker processes, four Logical Training Shards, worker protocol, shared memory, cleanup, and persistence-before-`done`.
6. `0003-load-saved-transformer-models-for-stateless-generation.md` for the separation between Transformer Training Runs and Saved Transformer Generation Runs, including the rule that loading creates no Transformer worker processes and receives no worker label.
7. `CONTEXT.md` for the canonical meanings of Transformer Training Run, Generated Text Sample, Request-Scoped Worker Group, Logical Training Shard, Saved Transformer Model, and Saved Transformer Generation Run.
8. The current stable Python boundaries in:
   - `src/how_llms_work/ml/transformer_worker.py` — `calculate_actual_worker_count()`, `build_worker_shard_assignments()`, `RequestScopedWorkerGroup`, and `create_request_scoped_worker_group()`;
   - `src/how_llms_work/routes/train_transformer.py` — `stream_transformer_training()` and `stream_saved_transformer_generation()`;
   - `tests/test_transformer_worker_group.py` — bounded worker-count and one-observation tests;
   - `tests/test_train_transformer_route.py` — exact TestClient/SSE contract, controlled worker group, raw sample collection, lifecycle order, cleanup, and persistence-before-`done` tests;
   - `tests/test_load_transformer_route.py` — no-training-worker and no-worker-label regression;
   - `tests/test_transformer_completion.py` — independent deterministic Generated Text Sample fixtures and Saved Transformer Model construction evidence;
   - `tests/test_train_transformer_persistence.py` — complete model persistence evidence.
9. `llm_works_file_structure.md` only as historical TypeScript presentation and behavior evidence when it agrees with the accepted specification. It is not current-code authority and must not reintroduce browser-side count guesses, worker-thread assumptions, cached state, or a changed payload schema.
10. Official Python 3.12 documentation only as a technical cross-check that `os.cpu_count()` reports logical CPUs and may return `None`, and official FastAPI documentation only as a cross-check for the existing `TestClient` public HTTP seam.
11. Older Python exports, previous plans, production `.data` artifacts, generated caches, frontend redesign ideas, and unsupported assumptions are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 027 adds one learner-visible line to one place: the `sample` string in the first public `epoch` event of a fresh Transformer Training Run.

The required first sample has this exact presentation shape:

```text
Transformer worker processes: <actualWorkerCount>

<unchanged Generated Text Sample text>
```

The value must be the exact count already selected by the current Request-Scoped Worker Group for that run. The current group observes `os.cpu_count()` once, bounds the observation to one through four, retains the result in `_actual_worker_count`, builds static worker-to-shard assignments from it, and passes it into runtime startup. The route currently receives the successfully started group but has no public read-only way to retrieve that retained count.

The current training route already preserves the correct data separation needed by this ticket:

- `generate_transformer_text()` returns a raw `GeneratedTextSample`;
- `generated_sample.text` is appended to the request-local `samples` collection;
- the same raw text is currently emitted as the public `epoch.sample`;
- the final `done.samples` collection is copied from the raw request-local records;
- Saved Transformer Model construction and persistence occur later from the completed training state and preprocessing, not from public sample strings.

The smallest complete change is therefore:

1. expose the group’s already-retained actual worker count through a read-only public property;
2. capture that value from the successfully created group without another CPU observation;
3. create a presentation-only sample string for the first emitted `epoch` event;
4. continue storing and using the raw generated text everywhere else.

No request field, SSE event name, payload field, frontend control, numerical operation, worker assignment, random stream, model field, or persistence format changes.

## Baseline evidence

- **Status:** User-reported.
- **Commands reported:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Reported result:** The user states that pytest and Ruff passed and mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No tests, formatter, linter, typechecker, application server, or Git command was executed against the user's live repository while creating this plan. The supplied export was inspected as read-only planning evidence.
- **Implementation rule:** `implement-prompt` must re-inspect the live repository, preserve uncommitted user changes, establish or reconfirm its own baseline before editing, and report only commands actually executed during implementation.

## Current code observations from the latest source

### Existing actual-worker-count boundary

- `calculate_actual_worker_count(reported_cpu_count)` already implements the approved rule by bounding one supplied observation to one through four.
- Existing parameterized tests cover `None`, `0`, `1`, `2`, `4`, `5`, and `64`, with expected results `1`, `1`, `1`, `2`, `4`, `4`, and `4`.
- `RequestScopedWorkerGroup.__init__()` calls `os.cpu_count()` once, passes that single observation to `calculate_actual_worker_count()`, stores the result in `_actual_worker_count`, and builds static assignments from that result.
- `test_group_reads_cpu_count_once_and_retains_static_assignment` already proves one observation and retained assignment behavior through startup, one epoch, and cleanup.
- Runtime startup receives the retained `actual_worker_count`, so that value is the correct count to display after successful group creation.
- `RequestScopedWorkerGroup` currently exposes lifecycle properties such as `state`, `cleanup_report`, `primary_failure_code`, and `successful`, but it does not expose the retained worker count.

### Current training-route data flow

- `stream_transformer_training()` creates one fresh cancellation event and yields the existing `init` event before training work.
- It initializes fresh parameters from `Mulberry32(42)` and creates a fresh `TransformerTrainingRun` for every valid request.
- It creates one Request-Scoped Worker Group through `create_request_scoped_worker_group()` and then advances training through that group.
- The worker group continues to calculate exactly four Logical Training Shard results per epoch regardless of whether the actual process count is one, two, three, or four.
- For each public report epoch, the route obtains a raw `GeneratedTextSample`, validates its epoch, and appends exactly `generated_sample.text` to the request-local `samples` record.
- The route currently emits exactly `epoch`, `loss`, and `sample`, with `sample` equal to the raw generated text.
- After final evaluation and successful worker cleanup, it builds and persists the complete Saved Transformer Model before emitting `done`.
- The current `done.samples` collection is copied from the raw request-local `samples` records and therefore already has the correct boundary for remaining label-free.

### Current tests and reusable seams

- `tests/test_train_transformer_route.py` uses FastAPI `TestClient`, an exact SSE parser, and controlled public collaborators.
- `ControlledWorkerGroup` currently models compute and cleanup but has no controlled actual-worker-count property.
- `install_dependencies()` creates the controlled group and supplies deterministic sample strings such as `controlled sample 0` through `controlled sample 50`.
- `test_train_transformer_headers_payloads_and_lifecycle_order_are_exact` currently asserts all 51 `epoch` samples are raw strings and that the same strings appear in `done.samples`; this is the main exact contract test to update.
- Existing route tests already protect event order, payload key sets, one-at-a-time epochs, cleanup, persistence-before-`done`, slot release, isolation, and quiet failure behavior.
- `tests/test_load_transformer_route.py::test_load_transformer_creates_no_training_workers_resources_or_labels` already checks that Saved Transformer Generation Runs create no training workers and that the response contains no `Transformer worker processes` text.
- `tests/test_transformer_completion.py` already owns deterministic raw Generated Text Sample and Saved Transformer Model evidence. The presentation change should not require any fixture update.
- `tests/test_train_transformer_persistence.py` already owns the on-disk model contract. The presentation change should not require any persistence-test expectation or artifact change.

### Current public contract

- A training `epoch` payload contains exactly `epoch`, `loss`, and `sample`.
- A training `done` payload contains exactly `architecture`, `finalLoss`, and `samples`.
- Saved-model generation uses a separate `loaded → result → done` stream and creates no Request-Scoped Worker Group.
- The frontend already displays the `sample` string. Because the count is inserted into the existing string, no frontend payload parser or control change is required for Ticket 027.

## Acceptance criteria coverage

### Already satisfied and evidenced

- The actual count is calculated once per Request-Scoped Worker Group from one `os.cpu_count()` observation.
- The count is bounded to one through four.
- Unavailable and nonpositive observations map to `1`; observations `1`, `2`, and `4` remain unchanged; observations above four map to `4`.
- Every training request starts from fresh weights and creates its own Request-Scoped Worker Group.
- Worker assignment continues to cover exactly four Logical Training Shards.
- Raw Generated Text Samples are currently separate objects containing generated text only.
- `done.samples` is currently built from raw request-local sample records.
- Training `epoch` and `done` payload field sets already match the approved public schemas.
- Generated sample randomness, training mathematics, final parameters, model building, and persistence do not consume public presentation strings.
- Saved Transformer Generation Runs create no training workers and already have an explicit no-label regression.
- Cleanup, persistence-before-`done`, event order, and shared-slot regressions already exist.

### Behavior present but evidence incomplete

- The exact actual worker count is retained by `RequestScopedWorkerGroup` and supplied to the runtime, but no public read-only property currently lets route orchestration obtain it without reaching into private state.
- The route already has a clean separation between raw sample storage and public sample emission, but existing tests do not yet prove a one-time presentation prefix while raw records remain unchanged.

### Partially implemented

- The ingredients for presentation-only formatting already exist: a retained count, a first public report event, raw sample records, and an existing `sample` string field. They are not yet connected at the public presentation boundary.

### Not implemented

- The exact `Transformer worker processes: <actualWorkerCount>` prefix in the first public training sample.
- The required single blank line between the label and unchanged generated text.
- The first-public-sample-only occurrence rule.
- Route-level evidence that different actual worker counts appear dynamically without a second CPU observation.
- Exact evidence that later `epoch.sample` strings remain raw while `done.samples`, deterministic fixtures, and persisted models remain unchanged.

### Evidence limitation

- The live repository was not available as a writable checkout during planning, so current-code observations are grounded in `py_llm_pipeline_explorer_file_structure(138).md` rather than a live `git status` and direct file reads.
- The current frontend export was not attached in this handoff. This does not block Ticket 027 because the approved design changes only the value of the existing `sample` string and explicitly prohibits a new field or frontend control. The implementer should still confirm the live frontend makes no assumptions that samples are single-line text before declaring manual compatibility complete.
- No command was run during planning, so all future verification results remain unknown.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer_worker.py`
   - `calculate_actual_worker_count()`;
   - `build_worker_shard_assignments()`;
   - `RequestScopedWorkerGroup.__slots__` and constructor;
   - existing public lifecycle properties;
   - `RequestScopedWorkerGroup._start()`;
   - `create_request_scoped_worker_group()`.
2. `src/how_llms_work/routes/train_transformer.py`
   - `stream_transformer_training()` from worker-group creation through raw sample append and `epoch` emission;
   - final `done_samples` construction;
   - `stream_saved_transformer_generation()` only to confirm it stays label-free.
3. `tests/test_transformer_worker_group.py`
   - `test_actual_worker_count_is_bounded_from_reported_cpu_count`;
   - `test_group_reads_cpu_count_once_and_retains_static_assignment`;
   - `_StubRuntime` startup observations;
   - public contract and lifecycle tests that could be affected by a new read-only property.
4. `tests/test_train_transformer_route.py`
   - `ControlledWorkerGroup`;
   - `Dependencies`;
   - `install_dependencies()` and its controlled group factory;
   - `test_train_transformer_headers_payloads_and_lifecycle_order_are_exact`;
   - request isolation, cleanup, failure, and persistence-order regressions.
5. `tests/test_load_transformer_route.py`
   - `test_load_transformer_creates_no_training_workers_resources_or_labels`.
6. `tests/test_transformer_completion.py`
   - exact deterministic Generated Text Sample fixtures and Saved Transformer Model construction tests.
7. `tests/test_train_transformer_persistence.py`
   - exact persisted model shape, bytes, naming, replacement, and failure-preservation tests.
8. `pyproject.toml`
   - current Python, pytest, Ruff, Black, and strict-mypy configuration; no dependency change is expected.
9. `src/how_llms_work/schemas.py`, `src/how_llms_work/sse.py`, and `src/how_llms_work/main.py`
   - inspect only to confirm the request, SSE framing, and route registration remain unchanged.
10. The live frontend Transformer event consumer
    - inspect only for compatibility confirmation; no Ticket 027 frontend edit is expected.

## Step 1 — Expose the retained actual worker count through the stable worker-group boundary

**Files and symbols:**

- `tests/test_transformer_worker_group.py` — `test_group_reads_cpu_count_once_and_retains_static_assignment`, plus one focused public-property assertion or a narrowly named new test.
- `src/how_llms_work/ml/transformer_worker.py` — `RequestScopedWorkerGroup` public properties and `_actual_worker_count`.

**Purpose:**

Give route orchestration a safe, read-only way to obtain the exact count already selected for the current run. This serves the criteria requiring the displayed count to come from the existing one-observation, one-through-four boundary rather than a new calculation.

**Actions:**

- Re-inspect the live `RequestScopedWorkerGroup` implementation and verify the export still matches the repository.
- Add focused test evidence that the public group reports the same count passed to runtime startup.
- Extend the existing one-observation test so it proves reading the public value does not call `os.cpu_count()` again before, during, or after compute and cleanup.
- Add a read-only `actual_worker_count` property to `RequestScopedWorkerGroup` that returns the already-validated `_actual_worker_count`.
- Document the property as the number of spawned worker processes selected for this Transformer Training Run, not a statement about physical cores, affinity, or guaranteed hardware parallelism.
- Keep the calculation, constructor timing, assignment algorithm, runtime startup arguments, shared-memory layout, and cleanup state machine unchanged.

**Guardrails:**

- Do not add a setter or expose `_actual_worker_count` for mutation.
- Do not call `os.cpu_count()` from the new property.
- Do not recalculate the value from process handles, shard assignments, process affinity, or system topology.
- Do not rename or change `calculate_actual_worker_count()`.
- Do not change its boundary outputs, including the `None` and nonpositive cases.
- Do not add a module-global worker count.
- Do not add a count field to worker protocol messages unless live repository evidence proves the runtime no longer receives the existing value; the latest export already passes it directly.

**Expected result:**

- A successfully created `RequestScopedWorkerGroup` exposes one immutable integer in the range `1..4`.
- The exposed value equals the value used for runtime startup and static shard assignment.
- Exactly one `os.cpu_count()` observation still occurs per group.
- Existing worker lifecycle and real-spawn tests remain unchanged in behavior.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_worker_group.py -q
```

Expected focused evidence:

- all existing boundary cases still pass;
- the one-observation test still records exactly one call;
- the new public property reports the retained runtime count;
- no worker assignment or cleanup regression appears.

## Step 2 — Add route-level acceptance tests for the exact first-sample presentation contract

**Files and symbols:**

- `tests/test_train_transformer_route.py` — `ControlledWorkerGroup`, `Dependencies`, `install_dependencies()`, `test_train_transformer_headers_payloads_and_lifecycle_order_are_exact`, and the existing exact SSE parser/TestClient seam.

**Purpose:**

Create failure-first public evidence for the exact learner-visible behavior before changing route formatting. This covers the required prefix, blank line, dynamic count, one occurrence, unchanged later samples, unchanged payload keys, and unchanged `done.samples` collection.

**Actions:**

- Give `ControlledWorkerGroup` a read-only controlled actual-worker-count value matching the production property contract.
- Allow `install_dependencies()` to select a valid controlled count without changing its existing default behavior or unrelated lifecycle controls.
- Update the exact HTTP/SSE contract test so the first emitted `epoch.sample` is expected to be:
  - the exact label `Transformer worker processes: <count>`;
  - two newline characters after the count, yielding one blank line;
  - the unchanged raw first Generated Text Sample text.
- Keep every later `epoch.sample` expectation equal to the original raw generated text.
- Keep `expected_samples` for `done` entirely raw, including the first record.
- Assert the label occurs exactly once across all public training `epoch.sample` strings.
- Assert the full training response does not contain a second occurrence in `init`, later epochs, or `done`.
- Continue asserting every `epoch` payload has exactly the three approved keys and `done` has exactly the three approved keys.
- Add a narrowly parameterized public route test for representative actual counts `1`, `2`, and `4`, or parameterize the exact test when that remains readable. The test must obtain the value from the controlled group boundary, not patch a formatting helper or route local variable.
- Rely on the existing worker-count unit parameterization for `None`, `0`, and above-four observations; do not duplicate operating-system calculation behavior in the route test.
- Preserve the existing lifecycle-order assertion. The presentation change must not add a training, worker, persistence, cleanup, or delay stage.

**Guardrails:**

- Test through `POST /train-transformer` and parsed SSE, not through a private formatting helper.
- Do not assert a particular local string-concatenation implementation.
- Do not add `workerCount`, `actualWorkerCount`, `cores`, or another field to any payload.
- Do not couple the one-time rule to `epoch == 0`; couple it to the first public `epoch` event so the behavior remains correct if report scheduling changes later.
- Do not alter raw sample-generator return values to include the label.
- Do not weaken exact duplicate-field, duplicate-key, header, event-order, lifecycle-order, or slot-release assertions.

**Expected result:**

- Before the production route change, the new exact route assertions fail only because the first `sample` lacks the required prefix.
- The test suite clearly distinguishes the public first-sample string from raw `done.samples` records.
- Representative counts prove the public text is dynamic rather than hard-coded to `4` or another development-machine value.

**Verification:**

```powershell
poetry run pytest tests/test_train_transformer_route.py -q
```

Expected pre-implementation result:

- the newly added first-sample acceptance assertion is expected to fail against the current route;
- unrelated route tests should remain passing;
- do not manufacture failures in unrelated behavior.

## Step 3 — Prefix only the first public training sample at the route presentation boundary

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py` — `stream_transformer_training()` at Request-Scoped Worker Group creation, raw sample append, and `epoch` SSE emission.

**Purpose:**

Implement the learner-visible behavior with the smallest possible production change while preserving all raw numerical, deterministic, completion, persistence, and lifecycle state.

**Actions:**

- After successful Request-Scoped Worker Group creation, capture the group's public `actual_worker_count` value for this stream.
- Retain a simple request-local one-shot presentation state that identifies the first public report event. Do not use a process-global or model-global flag.
- Continue validating `generated_sample.epoch` exactly as now.
- Continue appending `generated_sample.text` unchanged to the raw `samples` collection before or independently of presentation formatting.
- For the first public `epoch` event only, provide a presentation string beginning exactly with `Transformer worker processes: <actualWorkerCount>`, followed by exactly one blank line, then the unchanged `generated_sample.text`.
- For every later `epoch` event, provide exactly `generated_sample.text` with no prefix or additional whitespace.
- Preserve all existing event names, field names, field ordering where asserted, loss values, presentation delays, disconnect checks, final evaluation, worker cleanup, model construction, persistence, `done` construction, and slot release.

**Guardrails:**

- Do not change `generate_transformer_text()` or `GeneratedTextSample`.
- Do not put the label into `samples`, `done_samples`, `SavedTransformerModel`, persistence metadata, preprocessing, model inputs, tokenization, random streams, loss calculation, gradients, optimizer state, final parameters, or logging.
- Do not read the private `_actual_worker_count` attribute from the route.
- Do not call `calculate_actual_worker_count()` or `os.cpu_count()` from the route.
- Do not infer the count from `len(logical_training_shards)`, which is always four, or from frontend input.
- Do not emit the label in `init`, `done`, an error event, or `stream_saved_transformer_generation()`.
- Do not use the wording `CPU cores`, `cores used`, `threads`, `hardware parallelism`, or `physical processors`.
- Do not add a reusable formatting helper unless live implementation evidence shows it is necessary; there is only one approved use site, and tests must remain at the public HTTP/SSE seam.

**Expected result:**

- The first public training `epoch.sample` has the exact required two-line presentation prefix and unchanged generated body.
- The label appears once per successful training stream.
- Later public samples remain byte-for-byte equal to their raw generated text.
- `done.samples` remains identical to the pre-ticket raw collection.
- Worker startup, numerical training, cleanup, model construction, persistence, and completion behavior are unchanged.

**Verification:**

```powershell
poetry run pytest tests/test_train_transformer_route.py -q
```

Expected result:

- the failure-first first-sample tests pass;
- all existing exact route, lifecycle, cleanup, cancellation, overlap, isolation, and persistence-order tests pass;
- payload key sets and event order remain exact.

## Step 4 — Prove the label cannot leak into numerical, completion, persistence, or saved-model-generation boundaries

**Files and symbols:**

- `tests/test_transformer_completion.py` — deterministic raw generation and Saved Transformer Model fixtures.
- `tests/test_train_transformer_persistence.py` — complete persisted model contract.
- `tests/test_load_transformer_route.py` — `test_load_transformer_creates_no_training_workers_resources_or_labels`.
- `tests/test_train_transformer_route.py` — exact `done.samples`, sample-generator calls, and request-isolation assertions.
- `src/how_llms_work/ml/transformer.py` — inspect only; no change expected.

**Purpose:**

Close the acceptance matrix by proving the presentation prefix remains outside all authoritative data and inference boundaries.

**Actions:**

- Run the existing completion fixture tests without updating their expected generated text.
- Run persistence tests without updating expected Saved Transformer Model content, metadata, key order, numeric values, or filenames.
- Run the saved-model generation route test that forbids training workers and worker-label text.
- Confirm the exact training route test still sees raw first text in `done.samples` and in the controlled sample generator's output/calls.
- Confirm request-isolation tests show one stream's one-shot presentation state does not suppress or duplicate the label in another request.
- Confirm unsuccessful runs do not persist models or emit `done`, preserving existing behavior; no special label-specific error path should be introduced.
- Strengthen an existing public test only if the live repository differs from the export and leaves a material criterion unproved. Prefer assertions in existing owning tests over a new fixture or broad snapshot.

**Guardrails:**

- Do not regenerate deterministic fixture files.
- Do not update `.data` files.
- Do not alter `build_saved_transformer_model()`, `save_transformer_model()`, `prepare_saved_transformer_prompt()`, or `generate_saved_transformer_text()`.
- Do not add a fake `Transformer worker processes: 0` value to saved-model generation.
- Do not treat the display line as generated model output.
- Do not add a cross-module presentation dependency to `ml/transformer.py` or persistence code.

**Expected result:**

- Existing deterministic raw Generated Text Sample fixtures remain unchanged.
- Existing losses, parameters, and Saved Transformer Model fixtures remain unchanged.
- Persisted JSON contains no worker-process label or worker-count metadata.
- `POST /load-transformer` remains label-free and training-worker-free.
- Every successful independent training request receives its own one-time first-sample prefix.

**Verification:**

```powershell
poetry run pytest `
    tests/test_transformer_completion.py `
    tests/test_train_transformer_persistence.py `
    tests/test_load_transformer_route.py `
    -q
```

Expected result:

- all tests pass without fixture or model-contract changes;
- the saved-model generation response contains no worker label;
- persistence remains byte- and structure-compatible with the existing contract.

## Step 5 — Complete repository-wide validation and inspect final scope

**Files and symbols:**

- All changed files.
- `pyproject.toml` — confirmed tool configuration.
- Git working tree and final diff.

**Purpose:**

Verify the complete ticket, protect adjacent endpoints and lifecycle behavior, and ensure the implementation contains no unrelated refactor or generated artifact.

**Actions:**

- Run focused tests first and resolve only Ticket 027 regressions.
- Run Black against the repository using the existing project configuration.
- Run Ruff against the complete repository.
- Run strict mypy against `src`.
- Run the complete pytest suite once at the end after focused checks are green.
- Inspect `git status --short` and `git diff --check`.
- Inspect the final diff and verify changes are limited to the worker-count exposure, route presentation boundary, and focused tests.
- Confirm no `.data` file, fixture, cache, lockfile, specification, ADR, frontend file, or unrelated backend module changed.
- Record actual command results in the implementation report; do not infer success from the user's pre-planning baseline.

**Guardrails:**

- Do not add or upgrade dependencies.
- Do not regenerate `poetry.lock`.
- Do not accept unrelated formatting churn.
- Do not weaken existing exact tests to make the new behavior pass.
- Do not create the implementation commit until every required check passes.

**Expected result:**

- Focused and complete tests pass.
- Formatting, linting, and strict typing pass.
- The final diff contains the smallest complete Ticket 027 change.
- No generated or production model artifacts are staged.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_worker_group.py -q
poetry run pytest tests/test_train_transformer_route.py -q
poetry run pytest `
    tests/test_transformer_completion.py `
    tests/test_train_transformer_persistence.py `
    tests/test_load_transformer_route.py `
    -q
poetry run black --check .
poetry run ruff check .
poetry run mypy src
poetry run pytest
git diff --check
git status --short
```

Expected result:

- every command succeeds;
- the final full suite passes once after focused verification;
- Git reports only intended source/test changes and no whitespace errors.

## Focused verification plan

Run these commands from the backend project directory in Windows PowerShell:

```powershell
poetry run pytest tests/test_transformer_worker_group.py -q
poetry run pytest tests/test_train_transformer_route.py -q
poetry run pytest `
    tests/test_transformer_completion.py `
    tests/test_train_transformer_persistence.py `
    tests/test_load_transformer_route.py `
    -q
```

Expected focused result:

- the worker count remains bounded and observed once;
- the public worker group exposes the same retained count used at startup;
- first training sample formatting is exact and dynamic;
- the label occurs once per training run;
- later epoch samples and `done.samples` are raw;
- deterministic generation and model fixtures are unchanged;
- persisted models are unchanged;
- saved-model generation emits no worker label and creates no training workers.

## Full verification plan

```powershell
poetry run black --check .
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

Expected result:

- Black reports no formatting changes required.
- Ruff reports no lint errors.
- Mypy reports `Success: no issues found`.
- All backend tests pass.

Do not state these outcomes as achieved until the implementation session actually runs the commands successfully.

## Manual acceptance checklist

- [ ] Start the backend from the live repository using the established Poetry/Uvicorn command.
- [ ] Submit one valid minimal Transformer training request through the current UI or registered training endpoint.
- [ ] Confirm the first displayed Generated Text Sample begins with the exact words `Transformer worker processes`.
- [ ] Confirm the displayed integer is between `1` and `4` and matches the worker count selected by the run rather than a hard-coded development value.
- [ ] Confirm there is exactly one blank line between the worker label and the generated text.
- [ ] Confirm the generated text beneath the label matches the raw sample expected for the same deterministic run.
- [ ] Confirm no later displayed training sample contains the worker label.
- [ ] Confirm the completed sample history contains raw generated text only and does not store the label.
- [ ] Confirm the final loss and expected sample text remain consistent with the pre-ticket deterministic reference for the same request.
- [ ] Confirm the model is persisted before `done` and the saved JSON contains no worker label or worker-count field.
- [ ] Submit a valid saved-model generation request and confirm its `loaded`, `result`, and `done` display contains no worker-process label and no synthetic zero count.
- [ ] Run a second independent training request and confirm it receives its own label exactly once.
- [ ] Confirm existing endpoints and the frontend remain functional without a new field or control.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/train_transformer.py
tests/test_transformer_worker_group.py
tests/test_train_transformer_route.py
```

Conditionally changed only if the live repository differs materially from the latest export and an acceptance criterion is otherwise unproved:

```text
tests/test_load_transformer_route.py
```

No fixture file is expected to change.

## Files not to change

```text
src/how_llms_work/__init__.py
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
src/how_llms_work/routes/__init__.py
src/how_llms_work/routes/simple_chat.py
src/how_llms_work/routes/bpe_tokenize.py
src/how_llms_work/routes/neural_net.py
src/how_llms_work/routes/train_embed.py
tests/fixtures/
.data/
frontend/
README.md
pyproject.toml
poetry.lock
poetry.toml
SPEC.md
CONTEXT.md
0002-stabilize-python-transformer-training-and-process-lifecycle.md
0003-load-saved-transformer-models-for-stateless-generation.md
027-show-the-actual-transformer-worker-process-count-during-training.md
llm_works_file_structure.md
```

A file from this list may be inspected, but editing it requires concrete live-repository evidence that Ticket 027 cannot be completed safely through the four likely files. Any such deviation must be minimal, explained, and independently verified.

## Risk notes and safeguards

1. **Risk:** The route calls `os.cpu_count()` a second time and displays a value different from the worker group actually created.
   - **Safeguard:** Expose and consume the group’s retained count; retain the one-observation test and add a public-property assertion.
2. **Risk:** The implementation displays four because there are four Logical Training Shards, even when fewer processes were spawned.
   - **Safeguard:** Read only `RequestScopedWorkerGroup.actual_worker_count`; never derive the display from shard count or result count.
3. **Risk:** The implementation reaches into `_actual_worker_count`, coupling route code to private state.
   - **Safeguard:** Add one narrow read-only public property at the owning worker-group boundary.
4. **Risk:** The label becomes part of `GeneratedTextSample.text` and changes deterministic fixtures.
   - **Safeguard:** Generate and store raw text first; create the prefixed string only for the first SSE payload.
5. **Risk:** The label is copied into `done.samples` because the route reuses the formatted public string.
   - **Safeguard:** Keep `samples` populated only from raw `generated_sample.text`; exact route tests must distinguish first public sample from first done record.
6. **Risk:** The label appears in every sample because formatting is applied inside the epoch loop without one-shot state.
   - **Safeguard:** Track first public emission in request-local state and assert exactly one occurrence across 51 report events.
7. **Risk:** The implementation assumes the first public sample is always epoch zero.
   - **Safeguard:** Base the rule on the first emitted report, not a hard-coded epoch number.
8. **Risk:** A global first-sample flag leaks across requests, causing later runs to omit or duplicate the label.
   - **Safeguard:** Keep state local to `stream_transformer_training()` and retain sequential/concurrent request-isolation tests.
9. **Risk:** The count is added as a new public field and breaks the frontend contract.
   - **Safeguard:** Preserve exactly `epoch`, `loss`, and `sample`; test exact key sets through TestClient.
10. **Risk:** The wording overclaims hardware behavior.
    - **Safeguard:** Use exactly `Transformer worker processes`; do not use `cores`, `CPU cores used`, affinity, or physical-processor terminology.
11. **Risk:** Saved-model generation synthesizes a zero count because it creates no worker group.
    - **Safeguard:** Leave `stream_saved_transformer_generation()` untouched and retain the existing no-worker/no-label route regression.
12. **Risk:** Model content changes because presentation text enters persistence metadata or model construction.
    - **Safeguard:** Leave `ml/transformer.py`, model builder, and persistence code untouched; run completion and persistence regressions without fixture changes.
13. **Risk:** Worker assignment, protocol, shared memory, or cleanup changes while exposing the property.
    - **Safeguard:** The property returns existing state only; retain the complete worker-group suite and avoid protocol edits.
14. **Risk:** Exact lifecycle order changes because formatting introduces a new asynchronous stage.
    - **Safeguard:** Keep formatting synchronous at the existing SSE emission point and preserve the exact call-order test.
15. **Risk:** Broad test rewrites hide regressions.
    - **Safeguard:** Update only the first public sample expectation, preserve raw done expectations, and retain all existing exact field/order assertions.
16. **Risk:** A frontend edit expands Ticket 027 into later command/display work.
    - **Safeguard:** Keep the existing sample field and page structure; inspect compatibility but make no frontend change.
17. **Risk:** Real `.data` models or deterministic fixtures are modified during testing.
    - **Safeguard:** Use existing temporary-directory persistence tests and verify `git status --short` before commit.

## Commit guidance after tests pass

Suggested outcome-oriented commit title:

```text
feat: show transformer worker process count
```

Commit body should mention:

- exposure of the existing retained Request-Scoped Worker Group count;
- exact first-training-sample presentation prefix and blank line;
- raw later samples, `done.samples`, deterministic generation, and persisted model preservation;
- saved-model generation remaining worker-label-free;
- the actual focused and full pytest, Black, Ruff, and mypy results.

Do not create the commit until all required checks pass. Do not include production `.data` files, generated caches, fixtures, frontend files, specifications, ADRs, or unrelated formatting changes.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- `plan027.md`;
- `027-show-the-actual-transformer-worker-process-count-during-training.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `0003-load-saved-transformer-models-for-stateless-generation.md`;
- `py_llm_pipeline_explorer_file_structure(138).md` as planning evidence;
- the live backend repository as implementation authority;
- `llm_works_file_structure.md` only as a compatible behavior reference.

`implement-prompt` must inspect the live repository again, establish its own baseline, preserve user changes, implement only Ticket 027, run focused verification before the complete suite, report actual command results honestly, inspect final scope, and create the implementation commit only after every required check passes.
