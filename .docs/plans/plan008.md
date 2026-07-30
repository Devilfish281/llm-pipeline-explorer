---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "008"
source_work_item: 008-persist-complete-saved-embedding-models-safely.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(19).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 008: Persist complete Saved Embedding Models safely

## Initial checklist

- Confirm Ticket 008 is the only selected work item and has no blockers.
- Treat `py_llm_pipeline_explorer_file_structure(19).md` as the latest current-code authority.
- Preserve the user-reported passing baseline for pytest, Ruff, and mypy without claiming current-session verification.
- Limit the change to a typed Saved Embedding Model contract, the route-owned persistence boundary, and focused persistence tests.
- Do not implement or register `POST /train-embed`.
- Finish with `poetry run pytest` after focused tests and quality checks.

## Source-of-truth hierarchy

1. The user's latest explicit direction that `py_llm_pipeline_explorer_file_structure(19).md` is the source of truth for the current Python Backend.
2. Ticket 008 for the required behavior, acceptance criteria, approved test seam, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(19).md` for the current implementation, test patterns, project configuration, and available dependencies.
4. `SPEC.md`, `CONTEXT.md`, ADR 0001 as summarized by the ticket/specification, and the TypeScript Reference Implementation for durable behavior and terminology.
5. Older code exports, snippets, tickets, or plans are non-authoritative when they conflict with the latest source.

## Work-item summary

Ticket 008 requires an independently testable persistence boundary for a complete Saved Embedding Model. The boundary must accept a typed, persistence-ready model; serialize the exact public model document; reject non-finite values before replacement; write through a unique same-directory temporary file; close the complete temporary document before one atomic replacement; preserve the previous destination on every failure; report cleanup failure together with the original failure; and support deterministic last-successful-finisher behavior under controlled concurrency.

This ticket is persistence-only. Fixed Saved Embedding Model fixtures must be used so implementation does not depend on Skip-gram training, Embedding Result construction, HTTP validation, SSE streaming, router registration, or Ticket 010.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**
  - `poetry run pytest`
  - `poetry run ruff check .`
  - `poetry run mypy src`
- **Result:** The user reports that pytest and Ruff passed, and mypy returned `Success: no issues found`.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing.

## Current code observations from the latest source

- `src/how_llms_work/routes/train_embed.py` is empty, so no Saved Embedding Model serialization or persistence boundary currently exists.
- `src/how_llms_work/ml/word2vec.py` already owns immutable reference-compatible corpus preprocessing through `Word2VecPreprocessing` and `get_word2vec_preprocessing()`.
- The current Word2Vec preprocessing exposes the complete ordered `vocabulary` and the complete ordered `merges`, but it does not define a typed complete Saved Embedding Model.
- `src/how_llms_work/routes/neural_net.py` provides the repository's relevant persistence prior art through `get_snapshot_directory()`, `serialize_saved_network()`, unique same-directory temporary-file creation, complete write/close, `os.replace()`, cleanup, and `ExceptionGroup` reporting.
- `tests/test_neural_net_persistence.py` already demonstrates pytest-managed temporary directories, exact JSON assertions, current-working-directory independence, failure preservation, cleanup checks, and controlled last-finisher concurrency.
- The TypeScript Reference Implementation constructs the five-field saved model, formats it with two-space indentation and one trailing newline, writes `embedding-weights.json`, and completes only afterward. Ticket 008 and `SPEC.md` strengthen that direct-write behavior into safe temporary-file and atomic-replacement behavior.
- `pyproject.toml` already contains the required runtime and test tooling. The standard library provides the needed JSON, temporary-file, path, and replacement facilities, so no dependency change is indicated.
- `main.py`, `schemas.py`, and the existing SSE routes are outside this ticket because the ticket explicitly excludes the `/train-embed` endpoint and router registration.

## Acceptance criteria coverage

- **Already satisfied and evidenced:** None at the embedding persistence boundary. Supporting ordered Vocabulary/Merge data and Neural Network persistence prior art already exist.
- **Behavior present but evidence incomplete:** None.
- **Partially implemented:** The Word2Vec module already exposes complete ordered Vocabulary and Merge Table data, but it does not expose the typed five-field Saved Embedding Model or complete embeddings mapping.
- **Not implemented:** Exact Saved Embedding Model serialization, non-finite rejection, backend-root destination resolution, directory creation, unique temporary files, close-before-replace, atomic replacement, failure preservation, cleanup aggregation, controlled concurrent replacement behavior, no-loading persistence behavior, and embedding-specific filesystem tests.
- **Evidence limitation:** Ticket 010 will later construct production Saved Embedding Models from completed training. Ticket 008 must therefore verify persistence with fixed typed fixtures rather than real training output. The standalone ADR file was not supplied in this planning handoff, but its persistence-before-completion and compatibility decisions are repeated in the selected ticket and `SPEC.md`; this does not block the ticket.

## Files to inspect before editing

1. `src/how_llms_work/routes/train_embed.py` — currently empty; this file must own the new persistence boundary.
2. `src/how_llms_work/ml/word2vec.py` — `Word2VecPreprocessing`, `get_word2vec_preprocessing()`, and the location for the public typed Saved Embedding Model contract.
3. `src/how_llms_work/routes/neural_net.py` — `get_snapshot_directory()`, `serialize_saved_network()`, `create_temporary_snapshot_path()`, `write_snapshot_document()`, `replace_snapshot_file()`, `remove_temporary_snapshot()`, and `save_network()` as repository prior art.
4. `tests/test_neural_net_persistence.py` — exact-document, failure, cleanup, and concurrency test patterns to adapt without coupling the new tests to Neural Network behavior.
5. `tests/test_word2vec.py` — current public Word2Vec test style and immutable preprocessing fixtures.
6. `pyproject.toml` — confirmed Python, pytest, Ruff, and strict-mypy configuration; verify that no dependency addition is necessary.
7. `llm_works_file_structure.md` — TypeScript `savedModel` shape and JSON formatting behavior only; it is not current backend code.

## Step 1 — Add the embedding-persistence acceptance test seam

**Files and symbols:**
- `tests/test_train_embed_persistence.py` — new fixed Saved Embedding Model fixtures and route-owned persistence acceptance tests.
- `tests/test_neural_net_persistence.py` — inspect existing fixture, failure-injection, and controlled-concurrency patterns; do not modify unless an actual regression requires it.

**Purpose:**
Create the approved pytest seam before production implementation. This step covers the exact model document, production path resolution, pytest-only filesystem isolation, directory creation, unique same-directory temporary files, and close-before-replace behavior.

**Actions:**
- Add one small fixed Saved Embedding Model fixture whose insertion order is `type`, `dimensions`, `vocab`, `merges`, and `embeddings`.
- Make the fixture contain multiple ordered Vocabulary Tokens, multiple ordered Merge records, and one complete public vector for every Vocabulary Token.
- Add a helper that detects temporary files only inside the supplied pytest temporary directory.
- Add a test proving that the production destination resolves to `backend/.data/embedding-weights.json` from the module location even after `monkeypatch.chdir(tmp_path)`.
- Add a test that saves into a missing nested pytest directory and compares the raw UTF-8 document against `json.dumps(model, indent=2, allow_nan=False) + "\n"`.
- Assert the parsed top-level field set, exact `type`, ordered Vocabulary, ordered Merge Table, complete embeddings mapping, exactly one final newline, and no leftover temporary file.
- Add an observation seam around replacement that proves the source temporary file and destination share a directory, have different paths, and the temporary file can be renamed immediately before final replacement, demonstrating that its writer is closed.
- Establish that these tests are expected to fail initially because the typed model and route-owned persistence boundary do not yet exist.

**Guardrails:**
- All writes must remain under `tmp_path`; never invoke the production destination in a write test.
- Test the public persistence boundary and observable filesystem behavior, not a required temporary-file library, local variable name, or exact internal loop structure.
- Do not add route requests, SSE assertions, training, result construction, or frontend behavior.

**Expected result:**
- A focused red acceptance suite defines the exact persistence contract without touching the real `.data` directory.

**Verification:**
- Run `poetry run pytest tests/test_train_embed_persistence.py -q`.
- Expected at this stage: focused failures caused only by the missing public model type and persistence boundary.

## Step 2 — Define the typed complete Saved Embedding Model contract

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — new public `SavedEmbeddingMerge` and `SavedEmbeddingModel` typed structures adjacent to the existing public Word2Vec data contracts.

**Purpose:**
Provide the ticket-required typed public input boundary while keeping construction and numerical training independent from persistence.

**Actions:**
- Define a plain-Python JSON-compatible type for one saved Merge record containing only `pair` and `merged`.
- Define a plain-Python JSON-compatible Saved Embedding Model type containing exactly:
  - literal `type` value `word2vec-skipgram`;
  - integer `dimensions`;
  - ordered `vocab`;
  - ordered `merges`;
  - token-to-public-vector `embeddings`.
- Use the repository's existing typed-dictionary style for persisted public structures unless current implementation inspection reveals a stricter established public pattern.
- Keep vectors, Vocabulary, Merge records, and the embeddings mapping representable by `json.dumps()` without NumPy scalars or arrays.
- Do not add a model builder, training result, route behavior, serialization, or filesystem access in this module.

**Guardrails:**
- Preserve `Word2VecPreprocessing`, `_WORD2VEC_PREPROCESSING`, and `get_word2vec_preprocessing()` unchanged.
- Do not expose internal training matrices or add Ticket 009/010 behavior.
- Do not validate or normalize a complete model in persistence; the typed boundary and exact serializer are the required seam.

**Expected result:**
- Fixed test fixtures and future Ticket 010 code can pass one complete, typed, persistence-ready Saved Embedding Model to the route boundary.

**Verification:**
- Run `poetry run pytest tests/test_word2vec.py tests/test_train_embed_persistence.py -q`.
- The existing Word2Vec preprocessing tests must remain green; persistence tests may remain red until Step 3.

## Step 3 — Implement exact serialization and atomic replacement

**Files and symbols:**
- `src/how_llms_work/routes/train_embed.py` — new embedding filename constant, backend-data path resolver, serializer, unique temporary-file creator, complete document writer, atomic replacer, temporary cleanup function, and public save function.
- `src/how_llms_work/ml/word2vec.py` — import the public `SavedEmbeddingModel` type only; no numerical changes.

**Purpose:**
Implement the smallest route-owned persistence boundary satisfying exact formatting, safe replacement, failure preservation, and future reuse by the Train Embed stream.

**Actions:**
- Add a constant for `embedding-weights.json`.
- Resolve the backend root from `Path(__file__).resolve()` using the same project-root convention already proven by the Neural Network route, then select the backend `.data` directory.
- Serialize the complete model before directory creation, temporary-file creation, or any possible replacement operation.
- Serialize with two-space indentation, `allow_nan=False`, UTF-8 text, and exactly one appended newline.
- Create the target directory with `parents=True` and `exist_ok=True`.
- Create one unique temporary file in the destination directory and close the creation descriptor before writing the model.
- Write the complete serialized document and ensure the writer is closed before replacement begins.
- Replace the destination exactly once with `os.replace()` only after the complete temporary write succeeds.
- On write, close, or replacement failure, remove only the temporary path owned by that save when cleanup succeeds, then re-raise the original failure.
- When cleanup also fails, raise one `ExceptionGroup` containing both the original persistence failure and the cleanup failure, preserving their order and information.
- Return the final destination path after successful replacement.
- Never read, parse, load, cache, resume from, or inspect an existing destination.
- Keep route registration, an `APIRouter`, HTTP handlers, logging, SSE behavior, and training orchestration out of this ticket.

**Guardrails:**
- Adapt the Neural Network persistence pattern; do not refactor both routes into a speculative shared persistence abstraction.
- Do not create a global lock, semaphore, queue, cross-process lock, manifest, history file, rollback file, or dependency.
- Do not touch `main.py`, `schemas.py`, `sse.py`, or `.data/embedding-weights.json`.
- Preserve the exact caller-provided ordering and values; persistence must not sort, deduplicate, round, or rebuild the model.

**Expected result:**
- One complete Saved Embedding Model can be safely persisted to a caller-supplied pytest directory or the backend-owned production destination.

**Verification:**
- Run `poetry run pytest tests/test_train_embed_persistence.py -q`.
- The exact-document, path, directory, temporary-file, close-before-replace, and normal-success tests should pass.

## Step 4 — Cover every failure boundary and cleanup outcome

**Files and symbols:**
- `tests/test_train_embed_persistence.py` — non-finite serialization and persistence-failure tests.
- `src/how_llms_work/routes/train_embed.py` — only the minimum corrections revealed by the focused tests.

**Purpose:**
Prove that every specified failure occurs before or without damaging the previous destination and that cleanup behavior is isolated to the failing save.

**Actions:**
- Parameterize fixed models containing `NaN`, positive infinity, and negative infinity in an embedding coordinate.
- Assert each non-finite model raises `ValueError` before temporary-file creation or replacement and leaves the previous destination byte for byte.
- Inject a temporary-file creation failure and assert the previous destination remains byte-identical and no unrelated path is removed.
- Inject a partial-write failure and a close-stage failure through the route-owned write seam; assert the prior destination remains byte-identical and the owned temporary file is removed when cleanup succeeds.
- Inject replacement failure after a complete temporary write; assert the previous destination remains byte-identical and the temporary file is removed.
- Inject cleanup failure after a persistence failure; assert the previous destination remains byte-identical and the raised `ExceptionGroup` contains both original and cleanup failures.
- Verify that a failed save never removes or rewrites a pre-existing valid destination.
- Keep all failure files under `tmp_path`.

**Guardrails:**
- Do not assert a particular `tempfile` API, operating-system wrapper implementation, or local helper name when the observable contract can be proven through the route-owned boundary.
- Do not manufacture elapsed-time assertions.
- Do not catch `BaseException`; ordinary persistence exceptions are sufficient.
- Do not swallow the original error when cleanup succeeds.

**Expected result:**
- Serialization, temporary creation, write, close, replacement, and cleanup failures all have deterministic, acceptance-aligned outcomes.

**Verification:**
- Run `poetry run pytest tests/test_train_embed_persistence.py -q`.
- Expected result: every failure case passes and no `.tmp` file remains except the intentionally retained file in the cleanup-failure case.

## Step 5 — Prove controlled concurrency and no model reuse

**Files and symbols:**
- `tests/test_train_embed_persistence.py` — controlled `ThreadPoolExecutor`, `Barrier`, `Event`, and lock-based concurrency cases.
- `src/how_llms_work/routes/train_embed.py` — only minimum corrections if concurrency tests expose a persistence defect.

**Purpose:**
Cover unique temporary ownership, complete last-successful-finisher replacement, and isolation between successful and failed concurrent saves without adding application-level locking.

**Actions:**
- Add two distinct complete Saved Embedding Model fixtures targeting the same filename.
- Control replacement order with synchronization primitives rather than elapsed timing.
- Assert two concurrent successful saves use distinct temporary paths in the same destination directory.
- Parameterize which save replaces last and assert the final raw document is exactly that complete model.
- Add a concurrent case in which one save succeeds and another fails during replacement or write; control the order so the failed save cannot truncate, corrupt, remove, or roll back the successful document.
- Assert the failed save cleans only its own temporary file when cleanup succeeds.
- Prepopulate the destination with invalid JSON and prove a successful save replaces it without attempting to parse or reuse it.
- Inspect the production code to confirm there is no load function, destination read, cache, resume path, global save lock, or training queue.

**Guardrails:**
- Do not add a global lock merely to make the test deterministic; synchronization belongs in the tests.
- Do not claim which request starts last; assert only the explicitly controlled successful replacement order.
- Do not share mutable model fixtures between worker calls if mutation could make the test ambiguous.

**Expected result:**
- Concurrent saves always leave one complete valid model, the controlled last successful replacer wins, and a failed concurrent save cannot damage another successful replacement.

**Verification:**
- Run `poetry run pytest tests/test_train_embed_persistence.py -q`.
- Repeat the focused command once only if a failure suggests nondeterministic test control; do not hide a race by adding timing sleeps.

## Step 6 — Run quality checks and inspect final scope

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — new public Saved Embedding Model typing only.
- `src/how_llms_work/routes/train_embed.py` — persistence-only implementation.
- `tests/test_train_embed_persistence.py` — complete persistence acceptance suite.
- Git diff — final scope and accidental-change inspection.

**Purpose:**
Verify the completed ticket against strict typing, linting, the complete regression suite, and the ticket's scope boundary.

**Actions:**
- Run the focused persistence tests first.
- Run Ruff across the project.
- Run strict mypy across `src`.
- Run the full pytest suite once at the end.
- Inspect the diff and confirm only the three expected files changed unless a documented, necessary correction was discovered.
- Confirm no generated `.data/embedding-weights.json`, temporary file, cache file, dependency file, frontend file, route registration, request schema, or unrelated formatting change is present.
- Confirm every Ticket 008 acceptance criterion has a direct test, code inspection, or manual checklist item.

**Guardrails:**
- Do not claim a command passed unless the implementation session actually executes it successfully.
- Do not commit until all checks pass and the diff remains limited to Ticket 008.
- Do not fold Ticket 010 model construction or Ticket 011 FastAPI streaming into this change.

**Expected result:**
- Ticket 008 is implementation-complete, regression-safe, and ready for the next work item.

**Verification:**
- Run the commands in the focused and full verification sections below and record their actual outputs.

## Focused verification plan

```powershell
poetry run pytest tests/test_train_embed_persistence.py -q
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All embedding-persistence tests pass.
- Ruff reports no errors.
- mypy reports no issues in `src`.

## Full verification plan

```powershell
poetry run pytest
```

Expected result:

- All tests pass.

## Manual acceptance checklist

- [ ] The persisted top-level fields are exactly `type`, `dimensions`, `vocab`, `merges`, and `embeddings`.
- [ ] `type` is exactly `word2vec-skipgram`.
- [ ] Vocabulary, Merge Table, embeddings keys, and vectors are preserved exactly as supplied.
- [ ] The JSON uses two-space indentation and exactly one trailing newline.
- [ ] `NaN`, positive infinity, and negative infinity fail before replacement.
- [ ] The default path resolves to `backend/.data/embedding-weights.json` independently of the current working directory.
- [ ] A missing backend data directory is created.
- [ ] Every save owns a unique same-directory temporary file.
- [ ] The complete temporary writer is closed before replacement.
- [ ] The destination changes only through one successful atomic replacement.
- [ ] Serialization, temporary creation, write, close, and replacement failures preserve the previous destination byte for byte.
- [ ] Failed saves remove only their own temporary files when cleanup succeeds.
- [ ] Cleanup failure preserves the destination and exposes both failures internally.
- [ ] Controlled concurrent successes leave the last successfully replaced complete model.
- [ ] A failed concurrent save cannot damage a successful model.
- [ ] Persistence never reads, loads, resumes, caches, or reuses an existing model.
- [ ] Tests write only under pytest-managed temporary directories.
- [ ] No `/train-embed` route, request schema, SSE behavior, or router registration was added.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/word2vec.py
src/how_llms_work/routes/train_embed.py
tests/test_train_embed_persistence.py
```

Conditionally changed:

```text
tests/test_word2vec.py
```

Only change `tests/test_word2vec.py` if the public typed contract cannot be fully and cleanly exercised from the dedicated persistence test module. Prefer no change.

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/routes/neural_net.py
tests/test_neural_net_persistence.py
pyproject.toml
poetry.lock
.data/embedding-weights.json
frontend/
src/how_llms_work/routes/train_transformer.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
```

## Risk notes and safeguards

1. **Risk:** Persistence could silently alter model ordering or values.
   - **Safeguard:** Serialize the supplied typed object directly and compare the exact raw document against an independent fixed fixture.

2. **Risk:** Non-finite data could create non-standard JSON or replace a valid model.
   - **Safeguard:** Complete `allow_nan=False` serialization before directory, temporary-file, or replacement work.

3. **Risk:** Temporary-file creation or writer-close failure could leave partial files or damage the destination.
   - **Safeguard:** Use one unique same-directory temporary path per save, close the creation descriptor, close the complete writer before replacement, and clean only the owned path.

4. **Risk:** Cleanup could hide the original persistence failure.
   - **Safeguard:** Preserve both failures in an `ExceptionGroup` and leave the previous destination untouched.

5. **Risk:** Concurrent saves could interleave data or make a failed save remove another save's file.
   - **Safeguard:** Never share temporary paths; use complete `os.replace()` operations and deterministic event-controlled concurrency tests.

6. **Risk:** Copying the Neural Network implementation could trigger an unnecessary refactor or shared abstraction.
   - **Safeguard:** Adapt its proven pattern locally in `train_embed.py`; defer generalization until multiple completed consumers justify it.

7. **Risk:** Ticket 008 could expand into training, result construction, or HTTP streaming.
   - **Safeguard:** Use fixed Saved Embedding Model fixtures and prohibit edits to `main.py`, `schemas.py`, SSE logic, and frontend files.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Commit body should mention:

- Added safe atomic persistence for complete Saved Embedding Models.
- Added exact formatting, failure-preservation, cleanup, and concurrency tests.
- Verification commands: `poetry run pytest`, `poetry run ruff check .`, and `poetry run mypy src`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 008, `SPEC.md`, `CONTEXT.md`, ADR 0001, and the latest current repository or `py_llm_pipeline_explorer_file_structure(19).md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 008, verify the complete change, and create the implementation commit.
