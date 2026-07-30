---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "005"
source_work_item: 005-persist-completed-xor-training-runs-safely.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(10).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 005: Persist completed XOR Training Runs safely

## Initial checklist

- Confirm Ticket 005 is the only work item in scope and Ticket 004 is satisfied by the current public Training Run implementation and tests.
- Treat `py_llm_pipeline_explorer_file_structure(10).md` as the latest current-code authority.
- Use `SPEC.md` and `CONTEXT.md` for the approved Saved Weight Snapshot rules and canonical terminology.
- Use `llm_works_file_structure.md` only as the TypeScript behavior reference; the Python ticket and specification govern the safer atomic persistence contract.
- Limit production changes to the route-owned persistence boundary and focused temporary-directory tests.
- Reconfirm the user-reported pytest, Ruff, and strict mypy baselines before editing.
- Finish with focused persistence tests, neural-network regression tests, the full suite, Ruff, mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript behavior to Python and treat the latest complete Python backend export as the current-code source of truth.
2. `005-persist-completed-xor-training-runs-safely.md` for required behavior, acceptance criteria, approved test seam, blockers, and scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(10).md` for the current Python implementation, tests, paths, dependencies, and repository conventions.
4. `SPEC.md` and `CONTEXT.md` for durable Phase 3 decisions and the canonical terms Training Run and Saved Weight Snapshot.
5. `llm_works_file_structure.md`, specifically the TypeScript `src/routes/neural-net/serialize.ts` behavior, as a compatibility reference only.
6. Older code exports, prior plans, and earlier assumptions are non-authoritative when they conflict with the sources above.

## Work-item summary

Add a fully typed persistence boundary owned by `backend/src/how_llms_work/routes/neural_net.py`. It must accept the completed Training Run's existing plain-Python `SavedNetwork` object, select the correct mode-specific filename, resolve `backend/.data/` independently of the shell's current working directory, create that directory when necessary, and write an exact two-space-indented JSON document with one final newline.

The final destination must be changed only through a successful same-directory atomic replacement after the unique temporary file is completely written and closed. Failures must leave the previous successful Saved Weight Snapshot unchanged and remove temporary files when cleanup succeeds. Concurrent saves must remain independent: different modes use different destinations, while same-mode saves produce one complete document and the last successful replacement wins.

This ticket does not add `POST /neural-net`, request validation, SSE events, presentation delays, disconnect handling, worker-thread orchestration, route registration, saved-weight loading, history, manifests, or frontend changes.

## Baseline evidence

- **Status:** User-reported.
- **Command:** `poetry run pytest`
- **Result:** The user reports that all tests passed before planning.
- **Command:** `poetry run ruff check .`
- **Result:** The user reports that Ruff passed before planning.
- **Command:** `poetry run mypy src`
- **Result:** The user reports `Success: no issues found` before planning.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing. These results were not tool-verified in this planning session.

## Current code observations from the latest source

- `backend/src/how_llms_work/ml/neural_net.py` now implements Ticket 004 and exposes the typed `SavedNetwork` union, `SingleLayerSnapshot`, `MultiLayerSnapshot`, and `TrainingResult.weights`.
- `SingleLayerState.to_snapshot()` already returns exactly `type`, `w1`, `w2`, and `bias` as ordinary Python scalar values.
- `MultiLayerState.to_snapshot()` already returns exactly `type`, `w1`, `b1`, `w2`, and `b2` as ordinary Python lists and scalar values.
- `backend/tests/test_neural_net.py` already proves the exact mode-specific key sets, plain JSON-compatible values, independent Training Runs, and isolated returned mutable state.
- `create_training_run()` initializes fresh numerical state and contains no filesystem read path, so completed Training Runs do not load or continue from a prior Saved Weight Snapshot.
- `backend/src/how_llms_work/routes/neural_net.py` exists but is empty. No destination resolver, serializer, temporary-file write, atomic replacement, cleanup, or persistence test seam exists.
- `backend/src/how_llms_work/main.py` does not register a neural-network router. Registration remains outside Ticket 005.
- `backend/pyproject.toml` already supplies Python 3.12, pytest, Ruff, strict mypy, and all standard runtime dependencies needed for this ticket. No package or lockfile change is required.
- The TypeScript serializer establishes the compatible two-space JSON formatting and final newline. The approved Python ticket and specification add same-directory unique temporary files, close-before-replace behavior, cleanup, failure preservation, project-root resolution, and concurrency guarantees.
- Existing route tests use `pytest.MonkeyPatch` to replace route-owned dependencies. Ticket 005 should preserve that style while using `tmp_path` for every filesystem write.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - The Single-Layer Saved Weight Snapshot object has exactly `type`, `w1`, `w2`, and `bias`.
  - The Multi-Layer Saved Weight Snapshot object has exactly `type`, `w1`, `b1`, `w2`, and `b2`.
  - Snapshot values produced by the numerical module are ordinary Python JSON-compatible numbers and lists rather than NumPy objects.
  - The producer adds no epochs, seed, timestamp, loss, verdict, architecture, dtype, shape metadata, request ID, manifest, or history.
  - Training Runs create fresh state and do not load or continue from saved snapshots.
  - The project already enables strict mypy for `src`.
- **Behavior present but evidence incomplete:**
  - `TrainingResult.weights` is the completed, typed input required by the persistence boundary, but it is not yet exercised by filesystem tests.
  - The repository has route-owned monkeypatching conventions, but no neural-network persistence dependency seam exists yet.
- **Partially implemented:**
  - The destination route module exists, but it is empty.
- **Not implemented:**
  - Mode-specific destination filenames.
  - Backend-project-root and `.data` resolution independent of the current working directory.
  - Directory creation.
  - Exact JSON file formatting.
  - Unique same-directory temporary files.
  - Close-before-replace behavior.
  - Atomic replacement after a complete successful write.
  - Previous-snapshot preservation on serialization, write, replacement, and cleanup failures.
  - Temporary-file cleanup.
  - Different-mode concurrent isolation.
  - Controlled same-mode last-successful-finisher-wins behavior.
  - Temporary-directory persistence tests.
- **Evidence limitation:**
  - Baseline commands are user-reported rather than tool-verified in this planning session.
  - The exact internal helper names and temporary-file primitive are intentionally left to implementation; tests must prove the observable contract rather than private implementation identity.
  - Cross-process locking and crash-durability guarantees beyond the confirmed same-process atomic replacement contract are out of scope.

## Files to inspect before editing

1. `backend/src/how_llms_work/routes/neural_net.py` — empty destination for the route-owned Saved Weight Snapshot persistence boundary.
2. `backend/src/how_llms_work/ml/neural_net.py` — `SavedNetwork`, `SingleLayerSnapshot`, `MultiLayerSnapshot`, `TrainingResult`, `SingleLayerState.to_snapshot()`, and `MultiLayerState.to_snapshot()`.
3. `backend/tests/test_neural_net.py` — exact snapshot-contract, plain-value, deterministic Training Run, and isolation prior art.
4. `backend/tests/test_bpe_tokenize.py` — route-owned monkeypatching and controlled failure-test prior art.
5. `backend/pyproject.toml` — Python version, pytest configuration, Ruff configuration, and strict mypy configuration.
6. `SPEC.md`, `CONTEXT.md`, and `005-persist-completed-xor-training-runs-safely.md` — approved persistence semantics, terminology, concurrency rule, test seam, and out-of-scope boundaries.
7. `llm_works_file_structure.md` — TypeScript `src/routes/neural-net/serialize.ts` formatting reference only.

## Step 1 — Establish the successful persistence contract through a focused public test seam

**Files and symbols:**
- `backend/tests/test_neural_net_persistence.py` — new persistence-focused test module.
- `backend/src/how_llms_work/routes/neural_net.py` — new route-owned Saved Weight Snapshot persistence boundary; exact helper names remain an implementation choice.

**Purpose:**
Create acceptance-relevant tests for destination selection, project-root resolution, directory creation, exact schemas, plain JSON output, and formatting before implementing filesystem behavior.

**Actions:**
- Add a dedicated persistence test module that imports only the intentionally callable route-owned persistence boundary and the existing public snapshot types or fixtures.
- Use fixed, explicitly typed Single-Layer and Multi-Layer snapshot fixtures; do not rerun long neural-network training merely to test filesystem behavior.
- Use `tmp_path` for every write and patch or inject only the route-owned snapshot-directory boundary.
- Add a resolver-only test that changes the process current working directory and proves the production default still derives `backend/.data/` from the installed source-file location rather than `Path.cwd()`.
- Add successful-save tests proving:
  - `single-layer` selects `single-layer-weights.json`;
  - `multi-layer` selects `multi-layer-weights.json`;
  - the missing `.data` directory is created;
  - parsed documents contain exactly the approved keys and no metadata;
  - all nested values are ordinary JSON-compatible values;
  - the raw document uses two-space indentation and exactly one final newline.
- Add a replacement-of-existing-snapshot test using deliberately invalid or stale prior content and prove the new snapshot is produced solely from the supplied completed Training Run weights.
- Keep assertions at the observable persistence boundary. Do not assert an exact temporary-file class, random filename pattern, private helper name, or lock identity.

**Guardrails:**
- Tests must never touch the repository's real `backend/.data/`.
- Do not add HTTP clients, request schemas, SSE parsing, route registration, delays, or disconnect fakes.
- Do not modify `ml/neural_net.py` to make persistence tests convenient; its public snapshot contract is already sufficient.
- Do not add key sorting, metadata, wrapper objects, or a manifest absent from the ticket.
- A new test is expected to fail initially because the route module is empty; do not manufacture unrelated failures.

**Expected result:**
- Focused tests precisely describe the successful mode-specific Saved Weight Snapshot contract and fail only because the persistence boundary is not implemented.

**Verification:**

```powershell
poetry run pytest tests/test_neural_net_persistence.py -q -k "path or directory or format or schema or existing"
```

- Expected before implementation: focused persistence tests fail at the missing boundary.
- Expected after Step 2: successful-path tests pass.

## Step 2 — Implement project-root resolution and exact successful snapshot replacement

**Files and symbols:**
- `backend/src/how_llms_work/routes/neural_net.py` — backend-root resolver, mode-to-filename mapping, JSON serialization boundary, and public persistence operation.
- `backend/tests/test_neural_net_persistence.py` — successful-path tests from Step 1.

**Purpose:**
Deliver the smallest typed production path that writes complete, exact Saved Weight Snapshots without depending on the shell's current working directory.

**Actions:**
- Derive the backend project root from `Path(__file__).resolve()` and the confirmed package layout, then append `.data`; do not use `Path.cwd()` or a user-specific absolute path.
- Provide a narrow patchable or injectable snapshot-directory seam so tests can redirect all writes to `tmp_path` while production uses the derived backend root.
- Map the snapshot `type` discriminant to exactly one destination:
  - `single-layer` → `single-layer-weights.json`;
  - `multi-layer` → `multi-layer-weights.json`.
- Create the destination directory with parent creation and existing-directory tolerance.
- Accept the existing `SavedNetwork` union rather than NumPy state, a `TrainingRun`, a request model, or the frontend result payload.
- Serialize one complete JSON document with two-space indentation and append exactly one newline.
- Preserve the supplied mode-specific snapshot fields without adding metadata or loading the existing destination.
- Write the complete serialized document to one unique temporary path located in the destination directory.
- Ensure the temporary writer is closed before replacement begins.
- Replace the final destination only after serialization and the complete temporary write succeed.
- Keep the boundary fully typed under the existing strict mypy configuration.

**Guardrails:**
- Do not expose `POST /neural-net` or create an `APIRouter` merely because the helper is route-owned.
- Do not include architecture, predictions, verdict, epochs, seed, timestamp, loss, request identifiers, dtype, shape metadata, history, or a manifest.
- Do not write directly to the final destination before the complete temporary document exists.
- Do not read or merge the previous destination.
- Do not add a new dependency; use `pathlib`, `json`, `tempfile`, `os`, or equivalent Python standard-library facilities.
- Do not require a fixed temporary-file name; uniqueness and same-directory placement are the contract.

**Expected result:**
- Both modes create exact complete documents beneath the correct `.data` directory, and an existing destination is replaced only after a closed temporary file contains the full new document.

**Verification:**

```powershell
poetry run pytest tests/test_neural_net_persistence.py -q -k "path or directory or format or schema or existing"
poetry run mypy src
```

- Expected result: successful persistence tests pass and the new source boundary satisfies strict mypy.

## Step 3 — Prove and implement failure preservation and temporary-file cleanup

**Files and symbols:**
- `backend/tests/test_neural_net_persistence.py` — controlled serialization, write, replacement, and cleanup failure cases.
- `backend/src/how_llms_work/routes/neural_net.py` — failure ordering, exception propagation, and best-effort temporary cleanup.

**Purpose:**
Guarantee that no failed save corrupts or replaces the previous successful Saved Weight Snapshot.

**Actions:**
- Prepopulate a destination with known complete bytes for each controlled failure case.
- Through narrow route-owned dependency patching, induce failures at these boundaries:
  - before or during serialization;
  - while writing the temporary document;
  - during final replacement;
  - while cleaning a leftover temporary file.
- For serialization, write, and replacement failures, assert:
  - the operation reports failure rather than claiming success;
  - the prior destination bytes are unchanged;
  - no partial document appears at the final destination;
  - temporary artifacts are removed when cleanup itself succeeds.
- Add a cleanup-failure case proving that a cleanup exception never triggers destination replacement and the prior successful snapshot remains unchanged. Do not falsely require deletion when the deletion operation itself is deliberately forced to fail.
- Structure production control flow so serialization and temporary writing complete before replacement is attempted.
- Track only the temporary path created by the current call and clean it in the failure path without deleting another concurrent save's file.
- Preserve the original failure context when cleanup also fails; avoid silently reporting success or masking the destination-preservation outcome.
- Use missing-file-tolerant cleanup after successful replacement so the moved temporary path does not cause a false failure.

**Guardrails:**
- Do not delete, truncate, or rewrite the prior destination as part of error recovery.
- Do not use one shared fixed temporary filename.
- Do not catch every exception and continue as though persistence succeeded.
- Do not expose internal exception details through a new SSE or HTTP contract; transport behavior belongs to a later ticket.
- Do not make tests depend on one particular `tempfile` API; induce failures through the narrow file-operation boundaries used by the route module.

**Expected result:**
- Every pre-replacement failure leaves the previous destination byte-for-byte unchanged; ordinary failure cleanup removes only the current call's temporary file; a forced cleanup failure still cannot replace the destination.

**Verification:**

```powershell
poetry run pytest tests/test_neural_net_persistence.py -q -k "failure or preserve or cleanup"
```

- Expected result: all controlled failure and cleanup tests pass.

## Step 4 — Prove concurrent mode isolation and last-successful-finisher-wins behavior

**Files and symbols:**
- `backend/tests/test_neural_net_persistence.py` — controlled concurrent-save tests using `threading.Event`, `Barrier`, or equivalent deterministic coordination.
- `backend/src/how_llms_work/routes/neural_net.py` — unique per-call temporary paths and atomic replacement behavior.

**Purpose:**
Verify that simultaneous Training Run completions cannot share temporary files, cross mode destinations, or expose corrupt JSON, and that same-mode replacement order determines the final Saved Weight Snapshot.

**Actions:**
- Add a concurrent different-mode test that overlaps one Single-Layer save and one Multi-Layer save under the same temporary `.data` directory.
- Prove the two operations target distinct final filenames and produce two complete parseable documents with their exact mode-specific schemas.
- Capture replacement source and destination paths through a wrapped route-owned replacement dependency and assert:
  - every source is distinct;
  - every source is in the same directory as its destination;
  - no source equals a final destination.
- Add a controlled same-mode test with two distinct complete snapshots.
- Coordinate the replacement boundary so one save is intentionally allowed to replace first and the other replaces last.
- Assert the final destination equals the complete document from the operation that successfully replaced last, never a mixture or truncated document.
- Run the controlled ordering in both directions when practical so the rule is tied to successful replacement order rather than thread creation order or fixture identity.
- Confirm no temporary artifacts remain after both successful saves.
- Preserve a lock-free or otherwise behaviorally equivalent design in which the observable final result is based on replacement completion. Do not add cross-process locking, history, version checks, or compare-and-swap semantics.

**Guardrails:**
- Do not rely on sleeps or nondeterministic scheduler timing to decide the winner.
- Do not assert a private lock object, executor identity, exact temporary filename, or implementation-specific call count beyond what proves the contract.
- Do not share mutable snapshot dictionaries between threads; give each save its own fixture.
- Do not expand the helper into a model registry or checkpoint manager.
- Same-process concurrency is the approved seam; cross-process locking is explicitly out of scope.

**Expected result:**
- Different modes persist independently, and controlled same-mode saves always leave one complete document from the last successful replacement.

**Verification:**

```powershell
poetry run pytest tests/test_neural_net_persistence.py -q -k "concurrent or simultaneous or last_successful"
```

- Expected result: all deterministic concurrency tests pass without timing-dependent flakiness.

## Step 5 — Run focused regressions, complete quality checks, and inspect scope

**Files and symbols:**
- `backend/src/how_llms_work/routes/neural_net.py` — completed persistence boundary.
- `backend/tests/test_neural_net_persistence.py` — complete Ticket 005 acceptance coverage.
- `backend/src/how_llms_work/ml/neural_net.py` and `backend/tests/test_neural_net.py` — unchanged Ticket 004 producer and regression suite.
- Existing Simple Chat, BPE, health, configuration, and future-phase modules — regression-only inspection.

**Purpose:**
Prove Ticket 005 integrates with the completed numerical boundary without changing algorithms, endpoints, frontend contracts, dependencies, or unrelated features.

**Actions:**
- Run the complete persistence test module.
- Run persistence and numerical neural-network tests together to prove the persisted schemas match the producer's existing `SavedNetwork` contract.
- Run the complete configured pytest suite once after focused tests pass.
- Run Ruff and strict mypy using the repository's confirmed commands.
- Inspect the final diff and confirm only the route persistence module and focused persistence test module changed.
- Confirm no test wrote into or changed the repository's real `.data` directory.
- Confirm `routes/neural_net.py` still does not expose or register `POST /neural-net`.
- Confirm no production path loads an existing snapshot.
- Record the exact implementation-session outputs; do not claim success without successful command results.

**Guardrails:**
- Do not fix unrelated lint, typing, test, or formatting findings within Ticket 005.
- Do not change Ticket 004 numerical formulas, snapshot conversion, deterministic fixtures, or verdict behavior.
- Do not add route registration, Pydantic models, SSE helpers, delays, disconnect checks, worker threads, or frontend changes.
- Do not create or commit generated snapshot or temporary files.
- Do not create a commit until all required checks pass and the diff is in scope.

**Expected result:**
- Ticket 005 persistence behavior and all existing backend behavior pass the configured test and quality gates with a two-file, persistence-only diff.

**Verification:**

```powershell
poetry run pytest tests/test_neural_net_persistence.py -q
poetry run pytest tests/test_neural_net_persistence.py tests/test_neural_net.py -q
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Focused verification plan

```powershell
poetry run pytest tests/test_neural_net_persistence.py -q
poetry run pytest tests/test_neural_net_persistence.py tests/test_neural_net.py -q
```

Expected result:

- All mode mapping, root resolution, directory creation, exact formatting, failure preservation, cleanup, isolation, and concurrency tests pass.
- Existing Training Run snapshot-contract and state-isolation tests remain green.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.

## Manual acceptance checklist

- [ ] Ticket 004 remains satisfied: `TrainingResult.weights` supplies the existing exact `SavedNetwork` object.
- [ ] Single-Layer Mode writes only `single-layer-weights.json`.
- [ ] Multi-Layer Mode writes only `multi-layer-weights.json`.
- [ ] Production `.data` resolution is based on the backend source location, not the current working directory.
- [ ] A missing destination directory is created.
- [ ] Raw JSON has two-space indentation and exactly one final newline.
- [ ] Parsed JSON contains only the exact approved mode-specific keys and ordinary JSON values.
- [ ] Every save uses a unique temporary path in the same directory as its destination.
- [ ] The temporary writer is closed before replacement.
- [ ] The final destination is replaced only after a complete successful write.
- [ ] Serialization, write, and replacement failures preserve the prior destination and clean temporary files when cleanup succeeds.
- [ ] A forced cleanup failure cannot replace the prior destination.
- [ ] Concurrent different-mode saves produce two independent complete snapshots.
- [ ] Controlled concurrent same-mode saves leave the complete snapshot from the last successful replacement.
- [ ] No implementation path reads or continues from a previous snapshot.
- [ ] Tests write only under pytest temporary directories and do not alter the repository's real `.data`.
- [ ] `POST /neural-net`, SSE behavior, request validation, disconnect handling, worker offloading, and router registration remain unimplemented in this ticket.
- [ ] No dependency, lockfile, frontend, BPE, Simple Chat, Word2Vec, transformer, matrix, or math-utility change is present.

## Expected files changed

Likely changed:

```text
backend/src/how_llms_work/routes/neural_net.py
backend/tests/test_neural_net_persistence.py
```

Conditionally changed:

```text
None expected.
```

## Files not to change

```text
backend/src/how_llms_work/main.py
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/sse.py
backend/src/how_llms_work/ml/neural_net.py
backend/src/how_llms_work/ml/bpe.py
backend/src/how_llms_work/ml/math_utils.py
backend/src/how_llms_work/ml/matrix.py
backend/src/how_llms_work/ml/word2vec.py
backend/src/how_llms_work/ml/transformer.py
backend/src/how_llms_work/ml/transformer_worker.py
backend/src/how_llms_work/routes/simple_chat.py
backend/src/how_llms_work/routes/bpe_tokenize.py
backend/src/how_llms_work/routes/train_embed.py
backend/src/how_llms_work/routes/train_transformer.py
backend/tests/test_neural_net.py
backend/tests/test_simple_chat.py
backend/tests/test_bpe.py
backend/tests/test_bpe_tokenize.py
backend/pyproject.toml
backend/poetry.lock
backend/.data/
frontend/
TypeScript reference source
```

## Risk notes and safeguards

1. **Risk:** Deriving the project root with the wrong parent depth writes snapshots under `src/`, the package directory, or the shell working directory.
   - **Safeguard:** Derive from the confirmed `routes/neural_net.py` source location and add a current-working-directory independence test.

2. **Risk:** Writing directly to the destination exposes truncated or invalid JSON after a write failure.
   - **Safeguard:** Serialize and write the complete document to a unique same-directory temporary file, close it, and only then replace the destination.

3. **Risk:** Windows file-sharing behavior prevents replacement while the temporary file is still open.
   - **Safeguard:** Exit the writer context and verify close-before-replace ordering before invoking the replacement operation.

4. **Risk:** A fixed temporary filename causes concurrent saves to overwrite or delete one another.
   - **Safeguard:** Generate a unique temporary path per call and clean only the path owned by that call.

5. **Risk:** Cleanup logic masks the original failure, deletes the prior destination, or reports success.
   - **Safeguard:** Never target the destination during cleanup, preserve failure context, and treat cleanup failure as a failed save with the prior destination unchanged.

6. **Risk:** A global lock, cache, or stale-read comparison changes the confirmed last-successful-finisher-wins rule.
   - **Safeguard:** Base the winner on controlled successful replacement order and add no history, versioning, or cross-process locking.

7. **Risk:** Tests accidentally write into the real project `.data` directory.
   - **Safeguard:** Patch or inject the route-owned directory boundary to `tmp_path` in every write test and inspect the repository tree after the suite.

8. **Risk:** Persistence tests duplicate or alter the completed neural-network algorithm.
   - **Safeguard:** Use fixed typed snapshot fixtures and retain `ml/neural_net.py` plus its current tests unchanged.

9. **Risk:** The serializer accepts or invents metadata not present in the Saved Weight Snapshot contract.
   - **Safeguard:** Keep the input typed as `SavedNetwork`, assert exact key sets after parsing, and assert exact raw fixture output without wrappers or metadata.

10. **Risk:** Concurrency tests become flaky because they rely on sleep duration or thread scheduling.
    - **Safeguard:** Use deterministic synchronization around the observable replacement boundary, not elapsed time.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Suggested outcome:

```text
Persist XOR weight snapshots atomically
```

Commit body should mention:

- route-owned mode-specific Saved Weight Snapshot persistence;
- project-root resolution, exact JSON formatting, same-directory atomic replacement, and failure cleanup;
- deterministic different-mode and last-successful-finisher-wins concurrency coverage;
- the exact focused and full verification commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, `005-persist-completed-xor-training-runs-safely.md`, `SPEC.md`, `CONTEXT.md`, `py_llm_pipeline_explorer_file_structure(10).md`, and `llm_works_file_structure.md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 005, verify the complete change, and create the implementation commit.
