---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "021"
source_work_item: 021-persist-configuration-specific-saved-transformer-models-safely.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure.md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 021: Persist configuration-specific Saved Transformer Models safely

## Initial checklist

- Confirm Issue 021 is the only selected work item.
- Treat the latest `py_llm_pipeline_explorer_file_structure.md` export, dated internally `2026-07-28 21:53:25`, as the current Python Backend source of truth.
- Treat `SPEC.md`, `CONTEXT.md`, and ADR 0002 as the durable persistence authority.
- Treat `llm_works_file_structure.md` only as the TypeScript behavior reference for Saved Transformer Model shape, ordering, and historical serialization behavior.
- Confirm the Ticket 018 blocker is satisfied by the existing public `SavedTransformerModel` types and `build_saved_transformer_model()` boundary.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing it as independently verified in this planning session.
- Limit production changes to a route-owned, persistence-only boundary in `src/how_llms_work/routes/train_transformer.py`.
- Add focused filesystem tests under pytest-managed temporary directories.
- Do not implement `POST /train-transformer`, request validation, SSE streaming, route registration, worker orchestration, final evaluation, or persistence-before-`done` integration in this ticket.
- Finish implementation with focused persistence tests, Transformer completion regressions, the full pytest suite, Ruff, strict mypy, formatting verification, and a scope-only diff review.

## Source-of-truth hierarchy

1. The user's latest direction: convert the selected TypeScript behavior to Python and treat the latest complete Python Backend export as current-code truth.
2. `021-persist-configuration-specific-saved-transformer-models-safely.md` for the selected work item's exact acceptance wording, constraints, blockers, and approved test seam.
3. `py_llm_pipeline_explorer_file_structure.md`, internally dated `2026-07-28 21:53:25`, for current source, tests, fixtures, public symbols, dependencies, and repository conventions.
4. `SPEC.md` for the accepted Phase 5 persistence decisions:
   - one final model per unique `epochs` and `numLayers` configuration;
   - exact configuration-specific destination naming;
   - deterministic JSON;
   - complete serialization before filesystem mutation;
   - same-directory temporary files;
   - flush, file `fsync`, close, and atomic replacement;
   - prior-destination preservation;
   - no directory `fsync`;
   - no cross-process file lock;
   - no model loading, cache skipping, resume, or intermediate checkpoints.
5. `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for the exact filename, Windows-first durability boundary, model-order requirements, and module ownership.
6. `CONTEXT.md` for the canonical meaning of a Saved Transformer Model and configuration-specific retention.
7. Ticket 018 and `plan018.md` for the existing in-memory Saved Transformer Model construction boundary and its explicit exclusion of filesystem persistence.
8. Existing route-owned persistence code and tests:
   - `src/how_llms_work/routes/neural_net.py`;
   - `tests/test_neural_net_persistence.py`;
   - `src/how_llms_work/routes/train_embed.py`;
   - `tests/test_train_embed_persistence.py`.
9. Official Python documentation for `json.dumps()`, `tempfile.mkstemp()`, buffered-file `flush()` plus `os.fsync()`, and `os.replace()` as technical cross-checks only.
10. Older Python exports, prior implementation plans, historical direct-write TypeScript behavior, generated `.data` files, and prior assumptions are non-authoritative when they conflict with the sources above.

## Work-item summary

Issue 021 adds the independently testable filesystem boundary that persists one already-completed, already-validated `SavedTransformerModel`.

The persistence boundary must retain separate final files for separate Transformer training configurations. The approved destination format is:

```text
backend/.data/transformer-weights-e{epochs}-l{numLayers}-d32-h2-ff128-ctx32.json
```

For example:

```text
transformer-weights-e300-l2-d32-h2-ff128-ctx32.json
```

The boundary must:

1. accept a complete plain-Python `SavedTransformerModel`;
2. select the destination from the validated training configuration;
3. serialize the complete model in memory before creating or replacing any file;
4. preserve the model's established insertion order;
5. reject non-finite values with strict JSON serialization;
6. create the backend-owned `.data` directory when needed;
7. create one unique temporary file in the destination directory;
8. write the complete UTF-8 document using `\n`;
9. flush the Python writer;
10. call file `os.fsync()`;
11. close the writer;
12. replace the destination once with `os.replace()`;
13. remove only the failing save's temporary file where possible;
14. preserve the previous valid destination on every failed save;
15. permit safe whole-file concurrency without adding a global or cross-process lock;
16. return the final destination path only after successful replacement.

This ticket does not build the model, train a Transformer, evaluate loss, emit SSE, inspect browser disconnection, manage workers or shared memory, or emit `done`. A later orchestration ticket will call this synchronous persistence boundary only after successful final model construction.

## Baseline evidence

- **Status:** User-reported.
- **Command:** `poetry run pytest`
- **Reported result:** All tests passed.
- **Command:** `poetry run ruff check .`
- **Reported result:** Ruff passed.
- **Command:** `poetry run mypy src`
- **Reported result:** `Success: no issues found`.
- **Planning limitation:** These commands were not executed in this planning session.
- **Implementation rule:** The implementation session must establish or reconfirm its own baseline before editing and must report the actual results honestly.

## Current code observations

### Existing completed Transformer boundary

- `src/how_llms_work/ml/transformer.py` already publicly exports:
  - `SavedTransformerMerge`;
  - `SavedTransformerConfig`;
  - `SavedTransformerBlockWeights`;
  - `SavedTransformerWeights`;
  - `SavedTransformerModel`;
  - `build_saved_transformer_model()`.
- `build_saved_transformer_model()` is the intended source of the complete persistence-ready object.
- Ticket 018's focused tests already cover:
  - exact top-level order;
  - exact nested configuration and weight order;
  - complete Vocabulary and Merge Table;
  - canonical flattened parameter arrays;
  - six-decimal conversion;
  - positive-zero normalization;
  - finite ordinary Python values;
  - exclusion of transient training state;
  - fresh-container and mutation isolation.
- Ticket 018 explicitly stopped before JSON serialization and filesystem work. Issue 021 is the next boundary rather than a reason to modify the numerical builder.

### Current persistence ownership

- `src/how_llms_work/routes/train_transformer.py` exists as the correct route-owned destination for Transformer persistence.
- No focused `tests/test_train_transformer_persistence.py` module is listed in the latest test tree.
- No Transformer configuration-specific model file is listed under the current exported `.data` tree.
- Existing XOR and Word2Vec persistence code establishes local repository conventions for:
  - backend-root path resolution from `__file__`;
  - two-space JSON;
  - one final newline;
  - same-directory unique temporary paths;
  - `os.replace()`;
  - owned-temp cleanup;
  - controlled same-destination concurrency;
  - tests isolated under `tmp_path`.
- Those older boundaries are prior art, not exact templates. ADR 0002 adds a stronger Transformer-specific write contract requiring explicit writer flush and file `fsync` before close and replacement.

### Current dependencies and tooling

- Python 3.12 or newer is required.
- The standard library already supplies `json`, `os`, `pathlib`, `tempfile`, threading synchronization primitives, and executor support needed by the focused tests.
- `pytest`, Ruff, mypy, and Black are already declared.
- No dependency or lockfile change is expected.

### Evidence limitations

- The exact contents of `021-persist-configuration-specific-saved-transformer-models-safely.md` were not retrievable from the File Library during this planning session.
- The plan therefore uses the supplied work-item title plus the accepted persistence requirements in `SPEC.md`, `CONTEXT.md`, ADR 0002, Ticket 018, and the latest Python export.
- Before editing, the implementation session must read the actual Ticket 021 file and reconcile any more-specific acceptance wording. If the ticket conflicts with this plan, the ticket wins unless it conflicts with a later explicit user instruction or accepted ADR/specification decision.
- This planning session inspected exported source text rather than a live Git checkout. The implementation session must re-inspect the actual repository and preserve user changes.

## Acceptance criteria coverage

### Already satisfied and evidenced

- A complete, ordered, plain-Python `SavedTransformerModel` type exists.
- A public model builder exists and excludes optimizer, gradient, worker, process, shared-memory, path, timestamp, request, and checkpoint state.
- The model builder rejects incomplete, failed, invalid, or non-finite final Transformer state.
- Model values are rounded through the shared TypeScript-compatible six-decimal boundary before persistence.
- The complete model uses the approved top-level order:
  1. `type`;
  2. `config`;
  3. `vocab`;
  4. `merges`;
  5. `weights`.
- Transformer initialization and completion have no saved-model input, so persistence does not influence later training.
- The repository already has exact persistence and concurrency test patterns for other Learning Demos.

### Behavior present but not sufficient for Issue 021

- XOR and Word2Vec routes have route-owned serialization and atomic-replacement helpers.
- Their patterns demonstrate directory resolution, unique temporary ownership, replacement, cleanup, and concurrency.
- They do not establish the Transformer filename.
- They do not establish configuration isolation across `epochs` and `numLayers`.
- Their writer boundary does not prove the approved explicit flush-plus-file-`fsync` sequence.
- They must not be refactored merely to share code with Issue 021.

### Not implemented or not evidenced

- Exact Transformer configuration-specific filename construction.
- Runtime prevention of filename/model layer mismatch.
- Transformer production `.data` resolution.
- Exact Transformer JSON serializer.
- Strict pre-filesystem non-finite rejection at the persistence boundary.
- Unique same-directory Transformer temporary files.
- Explicit write, flush, `os.fsync`, close, replace ordering.
- Failure preservation for:
  - serialization;
  - directory creation;
  - temporary-file creation;
  - write;
  - flush;
  - file `fsync`;
  - close;
  - replacement.
- Best-effort cleanup that targets only the current save's temporary file.
- Internal preservation of both a persistence failure and a cleanup failure.
- Controlled same-configuration last-successful-replacement behavior.
- Controlled different-configuration isolation.
- A failed concurrent save's inability to damage another successful destination.
- Focused Transformer persistence tests under `tmp_path`.
- An explicit no-read/no-cache/no-resume persistence test.

## Files to inspect before editing

1. `021-persist-configuration-specific-saved-transformer-models-safely.md`
   - Read first.
   - Reconcile exact acceptance wording, blockers, public seam, and exclusions.

2. `src/how_llms_work/routes/train_transformer.py`
   - Confirm current contents.
   - Own the new synchronous Transformer persistence boundary here.
   - Do not add `APIRouter`, endpoint, stream, disconnect, or orchestration behavior.

3. `src/how_llms_work/ml/transformer.py`
   - Inspect only.
   - Reuse `SavedTransformerModel` and the established configuration field names.
   - Do not move filesystem behavior into the ML module.
   - Do not change numerical conversion unless a real public typing defect blocks the route-owned boundary.

4. `tests/test_transformer_completion.py`
   - Inspect the exact fixed model fixture and public builder tests.
   - Reuse or independently reproduce a small valid saved-model fixture without coupling persistence tests to expensive training.
   - Do not duplicate the entire completion test suite.

5. `tests/fixtures/transformer_completion_reference.json`
   - Inspect only if a small complete fixed model can be loaded cleanly.
   - Persistence expectations must remain independent of the production serializer.

6. `src/how_llms_work/routes/train_embed.py`
   - Inspect route-owned JSON, path, temporary-file, replacement, and cleanup conventions.
   - Do not modify.

7. `tests/test_train_embed_persistence.py`
   - Adapt exact-document, CWD independence, failure preservation, cleanup, and controlled concurrency patterns.
   - Add the Transformer-specific `fsync` cases rather than weakening them to the older boundary.

8. `src/how_llms_work/routes/neural_net.py`
   - Inspect mode-specific filename and atomic replacement prior art.
   - Do not modify.

9. `tests/test_neural_net_persistence.py`
   - Inspect different-destination and same-destination concurrency patterns.
   - Do not modify.

10. `SPEC.md`, `CONTEXT.md`, and ADR 0002
    - Reconfirm exact filename, serialization, durability, failure, and no-loading decisions.

11. `llm_works_file_structure.md`
    - Use only for TypeScript Saved Transformer Model structure and historical compatibility.
    - Do not copy direct-write, cache-skip, resume, checkpoint, or host-dependent behavior that the Python ADR intentionally replaces.

12. `pyproject.toml`
    - Confirm no dependency addition is needed.
    - Preserve current Python, test, lint, format, and strict typing configuration.

## Step 1 — Establish the Transformer persistence acceptance seam

**Files and symbols:**

- `tests/test_train_transformer_persistence.py` — new fixed model fixtures and public persistence-boundary tests.
- `src/how_llms_work/routes/train_transformer.py` — planned public persistence operations.
- `src/how_llms_work/ml/transformer.py` — existing public `SavedTransformerModel` type, inspected but unchanged by default.

**Purpose:**

Define the complete observable persistence contract before production implementation, without invoking real Transformer training, worker processes, SSE, or the production `.data` directory.

**Actions:**

- Create `tests/test_train_transformer_persistence.py`.
- Import only the intentionally public `SavedTransformerModel` type and the route-owned persistence boundary.
- Build at least two small, complete, ordinary-Python Saved Transformer Model fixtures:
  - one with `numLayers=1`;
  - one with `numLayers=2`;
  - complete top-level/config/vocab/merges/weights fields;
  - distinct sentinel weights so one complete document cannot be mistaken for another.
- Preserve exact top-level and nested insertion order in the fixtures.
- Avoid NumPy values in the persistence fixtures; Ticket 018 already proves conversion from final numerical state.
- Add filename cases for representative configurations, including:
  - `epochs=50`, `numLayers=1`;
  - `epochs=300`, `numLayers=2`;
  - `epochs=2000`, `numLayers=6`.
- Assert exact names:
  - `transformer-weights-e50-l1-d32-h2-ff128-ctx32.json`;
  - `transformer-weights-e300-l2-d32-h2-ff128-ctx32.json`;
  - `transformer-weights-e2000-l6-d32-h2-ff128-ctx32.json`.
- Establish one clear public persistence signature.
- Prefer a boundary that accepts:
  - the complete `SavedTransformerModel`;
  - the requested `epochs`;
  - an optional test-only destination directory.
- Derive `numLayers` from `model["config"]["numLayers"]` rather than accepting a second conflicting layer source.
- Validate `epochs` as a strict non-Boolean integer in the accepted request range before any filesystem operation.
- Validate `numLayers` as a strict non-Boolean integer in the accepted one-through-six range before any filesystem operation.
- Assert model configuration remains consistent with the fixed architecture encoded in the filename:
  - `contextLen=32`;
  - `embDim=32`;
  - `numHeads=2`;
  - `ffDim=128`.
- If the actual ticket requires explicit `num_layers`, require exact equality with `model["config"]["numLayers"]` and fail before filesystem work on mismatch.
- Add a production-path test that changes the current working directory and proves the destination still resolves to backend-owned `.data`.
- Add a missing-directory test using a nested `tmp_path`.
- Add an exact raw-document test comparing against an independently constructed expected string.
- Assert:
  - two-space indentation;
  - insertion-order keys;
  - no key sorting;
  - exactly one final `\n`;
  - no second trailing blank line;
  - parsed content equals the supplied model exactly;
  - no extra persistence metadata.
- Ensure every write test injects a pytest directory and never writes to the repository's real `.data`.
- Establish the focused suite as expected to fail before production implementation.

**Guardrails:**

- Do not call `build_saved_transformer_model()` in every persistence test.
- Do not run real training or spawn workers to produce fixtures.
- Do not generate expected JSON by calling the production serializer under test.
- Do not assert private helper names when observable ordering can be verified at the public save boundary.
- Do not add route, schema, SSE, main application, worker, or frontend behavior.
- Do not create a new general persistence module.
- Do not modify existing XOR or Word2Vec persistence implementations.

**Expected result:**

- One focused red suite defines exact naming, path resolution, document formatting, configuration consistency, and filesystem isolation.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_persistence.py `
    -q `
    -k "filename or path or directory or document or format or configuration"
```

Expected before implementation:

- Collection or assertions fail only because the Transformer persistence boundary is missing.

Expected after implementation:

- All naming, path, directory, exact-document, and configuration cases pass.

## Step 2 — Implement exact destination naming and complete in-memory serialization

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - destination filename construction;
  - backend `.data` resolver;
  - exact serializer;
  - typed public save boundary.
- `tests/test_train_transformer_persistence.py`
  - naming, validation, serialization-order, and pre-filesystem rejection tests.

**Purpose:**

Complete all deterministic, non-I/O work before directory or temporary-file creation.

**Actions:**

- Add constants only where they reduce duplication and remain internal to the route-owned persistence boundary.
- Construct the exact filename:

```text
transformer-weights-e{epochs}-l{numLayers}-d32-h2-ff128-ctx32.json
```

- Use decimal integer text without zero padding, signs, whitespace, locale formatting, or scientific notation.
- Derive `numLayers` from the model configuration unless the ticket explicitly mandates another typed configuration object.
- Reject:
  - Boolean epochs;
  - non-integer epochs;
  - epochs outside `50..2000`;
  - Boolean layer counts;
  - non-integer layer counts;
  - layer counts outside `1..6`;
  - fixed architecture fields that do not match the supported decoder architecture;
  - filename/model layer mismatches when an explicit layer argument is required.
- Perform these checks before directory creation.
- Resolve the production directory from `Path(__file__).resolve()` to the backend root, then append `.data`.
- Do not use `Path.cwd()`, a user-specific absolute path, environment-dependent shell location, or frontend directory.
- Serialize the supplied model directly with:
  - `json.dumps(model, indent=2, allow_nan=False)`;
  - default `sort_keys=False`;
  - one appended `"\n"`.
- Complete serialization before:
  - creating `.data`;
  - creating a temporary file;
  - opening a writer;
  - touching the destination.
- Preserve the supplied ordered plain-Python model exactly.
- Do not:
  - rebuild weights;
  - round again;
  - flatten again;
  - sort keys;
  - deduplicate Vocabulary or merges;
  - add epochs to the JSON;
  - add the destination name;
  - add timestamps, hashes, versions, request IDs, or filesystem metadata.
- Add parameterized `NaN`, positive-infinity, and negative-infinity cases at nested representative coordinates.
- Prove each non-finite case raises `ValueError` before directory or temporary creation.
- Add unsupported-object tests only where needed to prove the serializer reports failure before filesystem mutation.
- Keep public return and argument types compatible with strict mypy.

**Guardrails:**

- Do not duplicate Ticket 018's recursive model validation.
- Do not accept a `TransformerTrainingRun`, NumPy array, shared-memory view, worker group, route request, or frontend result in the save function.
- Do not inspect or parse an existing destination.
- Do not skip persistence because a matching filename already exists.
- Do not read a matching model for comparison.
- Do not add a cache, manifest, registry, history, rollback, or checkpoint file.
- Do not add a process-local or cross-process save lock.

**Expected result:**

- The complete model document and deterministic destination are known before any filesystem side effect.
- Invalid configuration and non-finite serialization cannot alter a previous destination.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_persistence.py `
    -q `
    -k "filename or configuration or serialize or finite or before_filesystem"
```

Expected result:

- Exact filenames and documents pass.
- Invalid configurations and non-finite values fail before filesystem work.
- Existing completion tests remain unchanged.

## Step 3 — Implement the Windows-first durable same-directory replacement boundary

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - unique temporary-file creation;
  - complete writer;
  - explicit flush;
  - file `fsync`;
  - close-before-replace;
  - atomic replacement;
  - owned-temp cleanup;
  - successful path return.
- `tests/test_train_transformer_persistence.py`
  - operation-order observation and normal-success tests.

**Purpose:**

Implement the stronger ADR 0002 durability sequence without exposing partial JSON or leaving the destination tied to an open temporary writer.

**Actions:**

- After successful configuration validation and complete serialization:
  1. create the destination directory with `parents=True` and `exist_ok=True`;
  2. calculate the final destination path;
  3. create one unique temporary file in the destination directory;
  4. ensure the creation descriptor is either used deliberately or closed exactly once;
  5. open or wrap the temporary writer as UTF-8 text with newline handling that produces `\n`;
  6. write the complete serialized document;
  7. call `flush()` on the buffered writer;
  8. call `os.fsync(writer.fileno())`;
  9. close the writer;
  10. call `os.replace(temporary_path, destination)`;
  11. return the final destination path.
- Use a temporary filename prefix tied to the final filename only for diagnosability; uniqueness is mandatory and exact spelling is not.
- Keep the temporary file in the destination directory so replacement does not cross filesystems.
- Ensure replacement is attempted exactly once per successful save.
- Ensure `os.replace()` is never called while the Python writer remains open.
- Do not require directory `fsync`.
- Add an observation seam that proves the order:

```text
write → flush → file fsync → close → replace
```

- Prove the temporary and destination paths:
  - share the same parent;
  - are not equal;
  - use unique temporary names across saves.
- Prove the raw destination bytes are the complete serialized document immediately after successful return.
- Prove no owned temporary path remains after successful replacement.
- Keep implementation synchronous. Later route orchestration can run it through `asyncio.to_thread()`.

**Guardrails:**

- Do not write directly to the final destination.
- Do not call `os.replace()` before flush, file `fsync`, and close.
- Do not call `os.fsync()` on the directory.
- Do not use one fixed `.tmp` filename.
- Do not rely on a writer's implicit close as the only durability step.
- Do not add retry loops, backoff, polling, or sleep-based coordination.
- Do not refactor the existing Learning Demo persistence code into a shared abstraction.
- Do not catch `BaseException`.

**Expected result:**

- A valid complete Saved Transformer Model is durably written to one unique same-directory temporary file and atomically replaces only its configuration-specific destination.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_persistence.py `
    -q `
    -k "temporary or write or flush or fsync or close or replace or success"
```

Expected result:

- The exact durability order, same-directory ownership, complete document, and successful cleanup cases pass.

## Step 4 — Cover every failure boundary and preserve the previous destination

**Files and symbols:**

- `tests/test_train_transformer_persistence.py`
  - parameterized stage failures;
  - prior-destination preservation;
  - owned-temp cleanup;
  - cleanup-failure reporting.
- `src/how_llms_work/routes/train_transformer.py`
  - only the minimum error-path corrections revealed by focused tests.

**Purpose:**

Prove that no failed persistence attempt can truncate, replace, delete, or silently supersede a prior valid configuration-specific model.

**Actions:**

- Prepopulate the target destination with known valid bytes for every failure case.
- Cover failures at:
  - configuration validation;
  - JSON serialization;
  - destination directory creation;
  - temporary-file creation;
  - complete-document write;
  - writer flush;
  - file `fsync`;
  - writer close;
  - `os.replace`.
- For every failure before replacement, assert:
  - the original exception remains observable;
  - the previous destination remains byte-for-byte identical;
  - replacement was not reported as successful;
  - the save function does not return a destination path;
  - only the failing save's temporary path is eligible for cleanup.
- After write, flush, `fsync`, close, or replace failure:
  - remove the owned temporary file when cleanup succeeds;
  - do not remove the final destination;
  - do not remove any unrelated temporary file.
- Add a cleanup-failure case after a primary persistence failure.
- Preserve both failures for internal handling, following the repository's established `ExceptionGroup` pattern unless the actual ticket mandates another representation.
- Keep the persistence failure first and the cleanup failure second.
- Do not claim cleanup succeeded when deletion is intentionally forced to fail.
- Assert the previous destination remains unchanged even when cleanup also fails.
- Add a close-stage test that proves replacement does not begin after close failure.
- Add an `fsync`-stage test that proves close is still attempted by normal context/finalization semantics while replacement remains prohibited.
- Use test doubles or route-owned seams rather than patching unrelated operating-system behavior globally.
- Keep every file under `tmp_path`.

**Guardrails:**

- Do not delete the destination in cleanup.
- Do not clean by wildcard, filename prefix scan, or whole-directory removal.
- Do not mask the primary failure with a successful cleanup.
- Do not report success after any stage failure.
- Do not retry replacement automatically.
- Do not use elapsed-time assertions.
- Do not require cleanup of a deliberately undeletable file in the cleanup-failure test.

**Expected result:**

- Validation, serialization, directory, temporary creation, write, flush, file-`fsync`, close, and replacement failures all preserve the previous valid model.
- Cleanup remains isolated to the owned temporary file.
- A secondary cleanup failure cannot hide the original persistence failure.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_persistence.py `
    -q `
    -k "failure or preserve or cleanup or serialization or creation or write or flush or fsync or close or replace"
```

Expected result:

- Every injected failure has the exact acceptance-aligned destination and cleanup outcome.

## Step 5 — Prove configuration isolation, controlled concurrency, and no model reuse

**Files and symbols:**

- `tests/test_train_transformer_persistence.py`
  - different-configuration isolation;
  - same-configuration controlled replacement order;
  - successful/failed concurrent save interaction;
  - no-read/no-cache evidence.
- `src/how_llms_work/routes/train_transformer.py`
  - only minimum corrections if concurrency exposes a real defect.

**Purpose:**

Prove that configuration-specific retention works without global locking and that whole-file atomic replacement prevents partial or interleaved documents.

**Actions:**

- Add two valid model/configuration fixtures targeting different filenames.
- Run controlled concurrent successful saves and assert:
  - both files exist;
  - each file equals its own complete expected document;
  - neither save removes or replaces the other's destination;
  - temporary paths are distinct.
- Add two distinct models targeting the same configuration-specific filename.
- Control replacement order with `Barrier`, `Event`, or a route-owned replacement seam.
- Parameterize which save replaces last.
- Assert the final destination equals the complete document from the controlled last successful replacement.
- Never infer order from thread start time or sleep duration.
- Add one same-configuration case where:
  - one save succeeds;
  - one save fails before replacement or during replacement;
  - the failed save cleans only its own temporary file;
  - the successful complete destination remains valid.
- Add one different-configuration case where a failure for one destination cannot alter the other configuration's successful file.
- Prepopulate a destination with:
  - valid old JSON;
  - invalid JSON;
  - arbitrary bytes.
- Prove a successful save replaces it without reading, parsing, comparing, merging, caching, or resuming from it.
- Inspect the implementation and tests to confirm:
  - no load function;
  - no destination `read_text()` or `open(..., "r")`;
  - no `exists()`-based skip;
  - no cache;
  - no manifest;
  - no intermediate checkpoint;
  - no application-level lock;
  - no cross-process file lock.
- Keep controlled concurrency tests deterministic through synchronization primitives, not timing.

**Guardrails:**

- Do not add a global lock to make tests deterministic.
- Do not promise process-wide ordering across independent FastAPI processes.
- Do not create a global latest-model file.
- Do not make one configuration delete or overwrite another.
- Do not mutate shared fixture dictionaries across threads.
- Do not run real training in persistence concurrency tests.

**Expected result:**

- Different configurations retain separate complete models.
- Same-configuration concurrent successes leave the complete model from the controlled last successful replacement.
- A failed concurrent save cannot corrupt or roll back another successful save.
- Existing model bytes never influence a new save or future training.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_persistence.py `
    -q `
    -k "configuration or concurrent or last or failed or no_read or no_cache"
```

Expected result:

- Configuration isolation, complete same-destination replacement, failed-save isolation, and no-loading cases pass without timing sleeps.

## Step 6 — Verify the persistence boundary and final implementation scope

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
- `tests/test_train_transformer_persistence.py`
- `src/how_llms_work/ml/transformer.py` — unchanged regression authority.
- `tests/test_transformer_completion.py` — unchanged builder regression authority.
- Existing persistence test modules — unchanged regression authority.
- Git diff and repository status.

**Purpose:**

Confirm Issue 021 is complete, strictly typed, formatting-clean, regression-safe, and limited to the approved persistence boundary.

**Actions:**

- Run the complete new persistence suite.
- Run Ticket 018 completion tests to prove the model builder remains unchanged.
- Run existing XOR and Word2Vec persistence tests to ensure no accidental shared behavior regression.
- Run all Transformer tests.
- Run the full test suite.
- Run Ruff.
- Run strict mypy.
- Run Black formatting verification or the repository's established formatter check.
- Inspect `git diff --check`.
- Inspect the final diff and reject:
  - endpoint code;
  - `APIRouter`;
  - Pydantic request changes;
  - SSE changes;
  - `main.py` registration;
  - worker changes;
  - numerical changes;
  - frontend changes;
  - dependencies;
  - lockfile changes;
  - generated `.data` files;
  - cache files;
  - temporary files.
- Confirm no production test wrote to the real `.data` directory.
- Confirm all actual command outcomes are reported honestly.
- Commit only after every required gate passes.

**Guardrails:**

- Do not fix unrelated lint, typing, formatting, or test issues inside Issue 021.
- Do not edit accepted requirement documents.
- Do not regenerate exact numerical fixtures.
- Do not weaken prior tests to make the new implementation pass.
- Do not claim any future verification result before it is actually executed.

**Expected result:**

- A two-file persistence-only implementation safely retains complete configuration-specific Saved Transformer Models and preserves all prior behavior.

**Verification:**

```powershell
poetry run pytest `
    tests/test_train_transformer_persistence.py `
    -q

poetry run pytest `
    tests/test_transformer_completion.py `
    tests/test_train_transformer_persistence.py `
    -q

poetry run pytest `
    tests/test_neural_net_persistence.py `
    tests/test_train_embed_persistence.py `
    tests/test_train_transformer_persistence.py `
    -q
```

Expected result:

- All new persistence tests pass.
- Ticket 018's exact model builder remains green.
- Existing persistence boundaries remain unchanged.

## Focused verification plan

Run from the backend directory:

```powershell
poetry run pytest `
    tests/test_train_transformer_persistence.py `
    -q
```

Expected result:

- Exact configuration-specific filenames pass.
- Production path resolution is independent of current working directory.
- The exact JSON document uses two-space indentation and one trailing newline.
- Non-finite values fail before filesystem mutation.
- Every save uses one unique same-directory temporary path.
- The observed durability order is write, flush, file `fsync`, close, replace.
- Every failure preserves the previous destination.
- Cleanup removes only the current save's temporary file when possible.
- Controlled concurrent replacements leave complete, non-interleaved documents.
- Different configurations remain isolated.
- Persistence never reads or reuses an existing model.

Then run the adjacent completion regression:

```powershell
poetry run pytest `
    tests/test_transformer_completion.py `
    tests/test_train_transformer_persistence.py `
    -q
```

Expected result:

- Saved Transformer Model construction and persistence compose through their public plain-Python contract without coupling numerical tests to filesystem behavior.

Then run all Transformer regressions:

```powershell
poetry run pytest `
    tests/test_transformer.py `
    tests/test_transformer_math.py `
    tests/test_transformer_training.py `
    tests/test_transformer_completion.py `
    tests/test_transformer_worker.py `
    tests/test_transformer_worker_group.py `
    tests/test_train_transformer_persistence.py `
    -q
```

Expected result:

- Preprocessing, layout, forward/backward, parent-side training, completion, worker protocol, worker lifecycle, and persistence remain green together.

## Full verification plan

Run from:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"
```

Then run:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
poetry run black --check .
```

Optional scope checks when the repository is a Git checkout:

```powershell
git diff --check
git status --short
git diff -- `
    src/how_llms_work/routes/train_transformer.py `
    tests/test_train_transformer_persistence.py
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.
- Black reports no formatting changes required.
- The diff contains only Issue 021 changes.
- No generated model, cache, or temporary file is present.

Do not claim these future commands passed until the implementation session runs them successfully.

## Manual acceptance checklist

- [ ] The actual Ticket 021 file was read before editing and any more-specific wording was reconciled.
- [ ] The persistence boundary accepts a complete `SavedTransformerModel`, not numerical training state.
- [ ] `numLayers` comes from one authoritative source or is checked against the model configuration.
- [ ] `epochs=50`, `numLayers=1` maps to `transformer-weights-e50-l1-d32-h2-ff128-ctx32.json`.
- [ ] `epochs=300`, `numLayers=2` maps to `transformer-weights-e300-l2-d32-h2-ff128-ctx32.json`.
- [ ] `epochs=2000`, `numLayers=6` maps to `transformer-weights-e2000-l6-d32-h2-ff128-ctx32.json`.
- [ ] Boolean, fractional, string, missing, and out-of-range configuration values cannot select a destination.
- [ ] Fixed architecture fields cannot disagree with the filename.
- [ ] Production `.data` is resolved from `train_transformer.py`, not the current working directory.
- [ ] A missing destination directory is created.
- [ ] The complete model is serialized before directory or temporary-file creation.
- [ ] JSON uses insertion order, two-space indentation, `allow_nan=False`, no key sorting, and exactly one trailing newline.
- [ ] Persistence does not round, flatten, rebuild, sort, deduplicate, or add metadata to the model.
- [ ] Every save owns one unique temporary path in the destination directory.
- [ ] The complete document is written before flush.
- [ ] The buffered writer is flushed before file `fsync`.
- [ ] File `fsync` succeeds before close and replacement.
- [ ] The writer is closed before `os.replace`.
- [ ] The destination changes through exactly one successful whole-file replacement.
- [ ] Directory `fsync` is not required or added.
- [ ] Serialization, directory creation, temporary creation, write, flush, file `fsync`, close, and replacement failures preserve the prior destination.
- [ ] Failed saves clean only their own temporary path when possible.
- [ ] Cleanup failure preserves the prior destination and leaves both failures available internally.
- [ ] Different `epochs` or `numLayers` configurations retain separate files.
- [ ] Controlled concurrent saves to one configuration leave one complete model from the last successful replacement.
- [ ] A failed concurrent save cannot corrupt, delete, truncate, or roll back a successful model.
- [ ] Persistence never reads, parses, loads, resumes, compares, caches, or skips because of an existing model.
- [ ] No intermediate checkpoint is written.
- [ ] Tests write only under pytest-managed temporary directories.
- [ ] No real backend `.data` file is added or modified.
- [ ] No `POST /train-transformer` route, request schema, SSE stream, disconnect check, deadline, run lock, worker orchestration, or `done` integration is added.
- [ ] No frontend, dependency, lockfile, numerical fixture, model-builder, worker, or shared-memory change is present.
- [ ] Actual pytest, Ruff, mypy, and formatting outcomes are recorded honestly.

## Expected files changed

Likely changed:

```text
src/how_llms_work/routes/train_transformer.py
tests/test_train_transformer_persistence.py
```

Conditionally changed only if live implementation evidence proves a narrow public typing defect blocks use of the already-completed Saved Transformer Model contract:

```text
src/how_llms_work/ml/transformer.py
tests/test_transformer_completion.py
```

The default expectation is no conditional change.

No package, lockfile, fixture, frontend, or generated-data change is expected.

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
src/how_llms_work/ml/transformer_worker.py
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
tests/test_transformer.py
tests/test_transformer_math.py
tests/test_transformer_training.py
tests/test_transformer_completion.py
tests/test_transformer_worker.py
tests/test_transformer_worker_group.py
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
021-persist-configuration-specific-saved-transformer-models-safely.md
llm_works_file_structure.md
```

A listed file may change only if the live repository proves the source export stale or a direct Issue 021 blocker exists. Any exception must be documented, minimal, and covered by focused tests.

## Risk notes and safeguards

1. **Risk:** The destination omits `epochs` or `numLayers`, causing valid configurations to overwrite one another.
   - **Safeguard:** Centralize exact filename construction and protect boundary examples plus different-configuration concurrent saves.

2. **Risk:** The filename's layer count disagrees with the saved model.
   - **Safeguard:** Derive layers from the model or require exact equality before any filesystem operation.

3. **Risk:** The wrong parent depth writes under `src`, the package directory, or the shell working directory.
   - **Safeguard:** Resolve backend `.data` from the route module and test after changing CWD.

4. **Risk:** Model serialization mutates order or values.
   - **Safeguard:** Serialize the supplied model directly, use exact raw-document assertions, and prohibit sorting or reconstruction.

5. **Risk:** NaN or infinity reaches disk as non-standard JSON.
   - **Safeguard:** Use `allow_nan=False` and finish serialization before creating a directory or temporary path.

6. **Risk:** Writing directly to the destination exposes a truncated model.
   - **Safeguard:** Write the complete document to one unique same-directory temporary file and replace only after durability steps succeed.

7. **Risk:** Buffered content has not reached the operating system before `fsync`.
   - **Safeguard:** Enforce and test `flush()` before `os.fsync()`.

8. **Risk:** Windows replacement fails because the temporary writer remains open.
   - **Safeguard:** Close the writer before `os.replace()` and test the ordering.

9. **Risk:** A fixed temporary filename lets concurrent saves overwrite or delete one another.
   - **Safeguard:** Use a unique path per save and clean only the path owned by that call.

10. **Risk:** Cleanup masks the persistence failure.
    - **Safeguard:** Preserve both failures through the established internal aggregation pattern, with the primary failure first.

11. **Risk:** A failed save removes the successful destination or another thread's temporary file.
    - **Safeguard:** Never target the destination in cleanup and never discover cleanup targets through directory scanning.

12. **Risk:** Same-configuration concurrency tests are timing-dependent.
    - **Safeguard:** Control replacement order with synchronization primitives around the observable replacement seam.

13. **Risk:** A global lock changes the accepted cross-process semantics.
    - **Safeguard:** Add no save lock; rely on complete same-directory temporary writes and whole-file replacement.

14. **Risk:** Persistence starts loading existing models for cache skipping or resume.
    - **Safeguard:** Add explicit no-read tests and inspect for destination reads, `exists()` skips, loader imports, and cache state.

15. **Risk:** The ticket expands into route orchestration because the file is route-owned.
    - **Safeguard:** Keep `APIRouter`, HTTP, SSE, disconnect, deadlines, run reservation, workers, cleanup orchestration, and `done` outside Issue 021.

16. **Risk:** Ticket 018's model builder is changed to simplify persistence.
    - **Safeguard:** Treat its public plain-Python object as the input contract and rerun completion tests unchanged.

17. **Risk:** A broad persistence refactor introduces regressions in XOR or Word2Vec.
    - **Safeguard:** Implement locally in `train_transformer.py`; run existing persistence tests but do not edit those modules.

18. **Risk:** The exact Ticket 021 text contains additional requirements not captured by the available sources.
    - **Safeguard:** Make reading and reconciling the actual ticket the first implementation step; stop and update the plan only if the ticket materially changes scope.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention.

Suggested commit subject:

```text
Persist configuration-specific Transformer models safely
```

Suggested commit body should mention:

- exact `epochs` and `numLayers` configuration-specific filenames;
- complete deterministic JSON serialization before filesystem mutation;
- unique same-directory temporary files;
- explicit write, flush, file `fsync`, close, and atomic replacement;
- previous-destination preservation for every failure stage;
- owned-temp cleanup and secondary cleanup-failure reporting;
- same-configuration and different-configuration concurrency evidence;
- no saved-model loading, cache skipping, resume, or intermediate checkpoints;
- no route, SSE, schema, worker, numerical, frontend, dependency, or generated-data changes;
- the exact verification commands actually executed.

Do not create the commit until all required tests and quality checks pass.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- `plan021.md`;
- `021-persist-configuration-specific-saved-transformer-models-safely.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- the latest `py_llm_pipeline_explorer_file_structure.md`;
- the latest `llm_works_file_structure.md`;
- Ticket 018 or `plan018.md` as the completed model-builder dependency;
- the live repository.

The implementation session must:

1. read the actual Ticket 021 file before editing;
2. inspect the live repository again;
3. establish or reconfirm the baseline;
4. preserve all user changes;
5. implement only Issue 021;
6. keep filesystem tests under `tmp_path`;
7. run focused and complete verification;
8. inspect the final diff;
9. report actual results honestly;
10. create the implementation commit only after every required gate passes.
