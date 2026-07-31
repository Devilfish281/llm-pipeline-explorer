---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "025"
source_work_item: 025-load-the-newest-strictly-valid-saved-transformer-model.md
source_specification: SPEC.md
source_context: CONTEXT.md
architecture_decision: 0003-load-saved-transformer-models-for-stateless-generation.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(122).md
frontend_code_reference: ts_llm_pipeline_explorer_file_structure(6).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 025: Load the Newest Strictly Valid Saved Transformer Model

## Initial checklist

- Confirm Ticket 025 is the only selected work item and that its Ticket 024 blocker is represented by the completed named-model `POST /load-transformer` request, deterministic Saved Transformer Generation Run, and exact `loaded → result → done` stream in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(122).md` as the current-code source of truth. Older exports, previous plans, pasted snippets, real `.data` artifacts, and historical TypeScript loading behavior are non-authoritative when they conflict with it.
- Use Ticket 025 for immediate scope and acceptance behavior, with `SPEC.md`, `CONTEXT.md`, and ADR 0003 supplying the durable latest-selection, strict-validation, statelessness, safety, and terminology decisions.
- Reuse the completed Ticket 023 exact filename grammar, real model-directory boundary, link/junction rejection, resolved containment checks, duplicate-key-aware JSON parser, strict current-format validator, canonical parameter materialization, and request-owned snapshot construction.
- Reuse the completed Ticket 024 prompt preparation, deterministic generation, shared Transformer request slot, same-process helper offloading, SSE formatting, and safe named-load failure behavior.
- Make the smallest complete backend-only change: accept a required nullable `modelFile`, add deterministic newest-valid selection through the existing trust boundary, dispatch named versus latest loading without changing generation, and extend the existing focused loader and HTTP/SSE tests.
- Keep full deadline/disconnect/cancellation hardening, worker-count presentation, frontend `File:` command parsing/display, token streaming, caching, sessions, model repair, training-from-snapshot behavior, and model-management features outside Ticket 025.
- Preserve the user-reported pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Finish implementation with focused latest-loader tests, focused load-route tests, affected Transformer regressions, the complete backend suite once, Ruff, strict mypy, formatting verification, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: plan Ticket 025 only, convert the approved behavior to Python, and treat the latest supplied Python Backend export as current-code truth.
2. `025-load-the-newest-strictly-valid-saved-transformer-model.md` for the exact `modelFile: null` behavior, candidate filtering, deterministic ordering, invalid-candidate skipping, tie-break, safe no-valid-model message, approved test seam, blocker, and exclusions.
3. `py_llm_pipeline_explorer_file_structure(122).md` for current source, tests, dependencies, paths, typing conventions, public symbols, and the completed Tickets 023 and 024 implementation.
4. `SPEC.md` for the durable Phase 6 decisions concerning newest strictly valid selection, direct-entry safety, current-format validation, read-once request snapshots, no cache, no mutation, safe errors, deterministic generation, and compatibility.
5. `0003-load-saved-transformer-models-for-stateless-generation.md` for the binding distinction between exact named selection and automatic newest-valid selection, plus the rule that invalid candidates may be skipped only for the latter.
6. `CONTEXT.md` for the canonical meanings and relationships of Saved Transformer Model, Saved Transformer Generation Run, Saved Transformer Event Stream, Transformer Training Run, and Request-Scoped Worker Group.
7. The completed current public and stable boundaries in:
   - `src/how_llms_work/schemas.py` — `LoadTransformerRequest`;
   - `src/how_llms_work/routes/train_transformer.py` — `get_transformer_model_directory()`, `load_named_transformer_model()`, `LoadedTransformerModelSnapshot`, `stream_saved_transformer_generation()`, and `load_transformer()`;
   - `tests/test_transformer_loading.py` — temporary-directory strict-loading fixtures and public loader tests;
   - `tests/test_load_transformer_route.py` — request-validation probes, exact SSE parsing, controlled collaborators, and bounded real integration.
8. `src/how_llms_work/ml/transformer.py`, `src/how_llms_work/ml/math_utils.py`, and their existing tests only as unchanged generation and model-shape dependencies. Ticket 025 must not alter their mathematics.
9. `ts_llm_pipeline_explorer_file_structure(6).md` only as a frontend compatibility reference. Frontend command classification and display are reserved for later tickets, so no frontend edit is expected here.
10. `llm_works_file_structure.md` only as historical TypeScript behavior evidence when it agrees with the approved Python Phase 6 specification; it is not current-code authority and must not reintroduce cached or permissive loading.
11. Official Python 3.12 documentation as a technical cross-check that `Path.iterdir()` order is arbitrary, `Path.is_junction()` is available, and exact modification timestamps are exposed as `st_mtime_ns`; official Pydantic documentation as a cross-check for required nullable fields and strict validation; and official FastAPI documentation as a cross-check for `StreamingResponse` and in-process `TestClient` testing.
12. Older code exports, prior plans, generated caches, actual production `.data` contents, and adjacent Phase 6 tickets are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 025 completes the automatic selection branch of the existing Saved Transformer Generation Run.

A valid `POST /load-transformer` request must continue to contain exactly the public fields `modelFile`, `prompt`, `temperature`, `topP`, and `maxTokens`, but `modelFile` becomes a required nullable field:

- a nonempty string continues to mean “load only this exact canonical filename”;
- JSON `null` means “select the newest strictly valid Saved Transformer Model”;
- an omitted field, empty string, wrong type, or malformed generation setting remains invalid under the existing public request contract.

For `modelFile: null`, the backend must resolve the genuine backend `.data` directory, inspect only direct ordinary entries, ignore anything whose exact name does not satisfy the current canonical Transformer persistence grammar, reject symbolic links and Windows junctions, enforce resolved containment, obtain deterministic ordering metadata, and sort all approved candidates by:

1. descending modification time; then
2. descending exact filename.

The backend must attempt candidates in that exact order through the same read, duplicate-key parse, strict current-format validation, filename/configuration agreement, canonical `float32` materialization, and request-owned snapshot boundary used by named loading. Invalid candidates are skipped only in this automatic branch. The first strictly valid snapshot wins. If none wins, the Saved Transformer Event Stream emits exactly one safe `error` event whose message is `No valid saved Transformer model was found.`

After a latest snapshot is selected, the existing Ticket 024 flow remains authoritative: prepare the prompt with the selected model's Vocabulary and Merge Table, emit the exact selected filename and trimmed prompt in `loaded`, generate deterministically through the current generation boundary, emit one `result`, then one empty `done`. The selected snapshot is request-local, no model is cached, no candidate is repaired or rewritten, and a later request enumerates, reads, and validates again.

Named requests must remain exact-only. A named failure must keep the existing safe named-load message and must never trigger latest-selection fallback.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**
  - `poetry run pytest`
  - `poetry run ruff check .`
  - `poetry run mypy src`
- **Result:** The user reported that pytest and Ruff passed and mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** None of these commands was executed against the live repository in this planning session.
- **Planning rule:** The implementation run must inspect the live repository, preserve user changes, and establish or reconfirm its own baseline before editing.

## Current code observations from the latest source

- `src/how_llms_work/main.py` already includes the single `train_transformer_router`, and that router already owns both `POST /train-transformer` and `POST /load-transformer`. No route-registration change is needed for Ticket 025.
- `LoadTransformerRequest` already defines the exact five public fields and aliases, strict finite generation settings, and the existing `3..500` maximum-token range. Its `model_file` field is currently a required nonempty strict string, so JSON `null` is rejected before route work.
- `get_transformer_model_directory()` already derives `.data` from the backend project rather than `Path.cwd()`.
- `_resolve_transformer_model_directory()` already rejects a symbolic-link or junction model directory, requires an ordinary directory, resolves it strictly, and returns both the direct directory and resolved containment root.
- `_parse_transformer_model_filename()` already enforces the exact canonical persistence grammar, exact case, supported epoch/layer ranges, no separators, no parent traversal, no absolute path, no drive, and no leading-zero aliases.
- `_select_named_transformer_model_file()` already enumerates direct entries to enforce exact-case equality, rejects candidate symlinks and junctions, requires an ordinary file, resolves it strictly, and requires its resolved parent to equal the genuine resolved model directory.
- The current named loading path reads the selected file once with `Path.read_bytes()`, parses from captured bytes with duplicate-key rejection and nonfinite-constant rejection, strictly validates the entire current Phase 5 document, checks filename/configuration layer agreement, materializes independent canonical `float32` storage, and returns `LoadedTransformerModelSnapshot`.
- `load_named_transformer_model()` maps unsafe, unreadable, malformed, incompatible, and materialization failures to the stable safe message `The saved Transformer model could not be loaded.` It has no fallback behavior and no cache.
- Existing `tests/test_transformer_loading.py` already proves named exact-case selection, strict structure and parameter validation, one read per call, rereading after content changes, no cache, request-owned state, artifact immutability, link/junction rejection, containment, unsafe-name rejection, duplicate-key rejection, and missing-file no-fallback behavior.
- `stream_saved_transformer_generation()` currently accepts only `model_filename: str` and always calls `load_named_transformer_model()`. It already applies the shared prompt preparation and deterministic generation workflow and emits `loaded → result → done` on success.
- The current stream maps all named load failures to the existing named-load message. It has no latest-specific no-valid-model message or latest-loader dispatch.
- `load_transformer()` currently forwards `payload.model_file` unchanged to the stream and otherwise already preserves the shared nonblocking Transformer request slot and SSE response boundary.
- Existing `tests/test_load_transformer_route.py` already proves exact schema aliases and required fields, `422` before work, finite number handling, overlap behavior, safe named-load errors, prompt validation before `loaded`, exact success payloads and event order, no training workers, slot release, request isolation, and deterministic real named-model integration.
- The controlled route-test seam currently patches only `load_named_transformer_model`; it must be extended without coupling tests to private sorting helpers or private candidate containers.
- `Path.iterdir()` does not guarantee directory enumeration order, so Ticket 025 cannot rely on the order currently used by named exact lookup. Explicit sorting is required for the latest branch.
- No production dependency, lockfile, frontend file, generation function, model format, training route, or SSE utility change is required by the evidence currently supplied.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - The genuine backend model directory is derived from backend code rather than the shell working directory.
  - Canonical exact-name grammar, exact capitalization, direct-entry enumeration, non-link/non-junction checks, ordinary-file checks, resolved containment, duplicate-key parsing, strict current-format validation, filename/configuration agreement, and request-owned canonical parameter materialization already exist for named loading.
  - Named loading reads the selected file once per invocation, rereads and revalidates later invocations, uses no cache, and leaves the candidate unchanged.
  - Named requests fail without fallback and already expose only the stable safe named-load message.
  - The successful named Saved Transformer Generation Run already uses the selected model's prompt preparation and deterministic generation boundaries and emits exact `loaded → result → done` payloads.
- **Behavior present but evidence incomplete:**
  - None. Existing named behavior is extensively evidenced; the missing behavior is specifically the automatic latest branch.
- **Partially implemented:**
  - The five-field load request and route exist, but `modelFile` is string-only instead of required nullable.
  - The safety and strict-validation primitives needed for latest selection exist, but they are currently composed only for one exact named file.
  - The generation stream is reusable after selection, but it currently dispatches only to `load_named_transformer_model()` and has only the named-load error mapping.
  - The route-test infrastructure can exercise the new path, but its request-invalid cases and controlled loader seam currently treat `null` as invalid.
- **Not implemented:**
  - Automatic candidate enumeration for `modelFile: null`.
  - Candidate filtering to canonical ordinary direct files for latest selection.
  - Explicit descending `st_mtime_ns` and descending exact-filename ordering independent of raw directory enumeration.
  - Newest-to-oldest strict validation with per-candidate skipping.
  - Stable greatest-filename tie selection.
  - The exact `No valid saved Transformer model was found.` stream error.
  - Latest-specific route dispatch while preserving named no-fallback behavior.
  - Latest-path tests for filtering, ordering, invalid skipping, ties, no-valid behavior, read-once/no-cache, immutability, safe errors, exact selected filename, and inherited SSE success.
- **Evidence limitation:**
  - The repository was inspected through `py_llm_pipeline_explorer_file_structure(122).md`, not a live Git checkout. The implementation prompt must re-inspect live files before editing.
  - The user's baseline is reported rather than independently verified here.
  - Real symbolic-link and Windows-junction tests are platform-conditional and may skip when the implementation environment cannot create those filesystem objects; controlled classification tests must continue to provide deterministic coverage.
  - No current-code evidence shows a need to change frontend files, model mathematics, persistence, dependencies, or route registration.

## Files to inspect before editing

1. `src/how_llms_work/schemas.py` — `LoadTransformerRequest`; confirm the required five-field contract, strict aliases, and the smallest correct nullable annotation for `model_file`.
2. `src/how_llms_work/routes/train_transformer.py` — `_TRANSFORMER_MODEL_FILENAME_PATTERN`, `_parse_transformer_model_filename()`, `_path_is_transformer_model_indirection()`, `get_transformer_model_directory()`, `_resolve_transformer_model_directory()`, `_select_named_transformer_model_file()`, `_parse_saved_transformer_model_document()`, `_validate_loaded_transformer_model_document()`, `_build_loaded_transformer_model_snapshot()`, `load_named_transformer_model()`, `stream_saved_transformer_generation()`, and `load_transformer()`.
3. `tests/test_transformer_loading.py` — `model_directory`, `saved_transformer_model`, `_serialize_model()`, `_artifact_state()`, link/junction helpers, `_load_named_model()`, `_assert_public_load_rejection()`, named read/no-cache tests, named no-fallback test, and existing safety/immutability parametrization.
4. `tests/test_load_transformer_route.py` — request field constants, `_valid_request()`, `_invalid_request_parameters()`, `RequestValidationProbe`, `_install_controlled_dependencies()`, `_post_load_transformer()`, exact SSE helpers, named success/failure tests, request-isolation tests, and the real named integration test.
5. `src/how_llms_work/main.py` — confirm that no registration edit is necessary because `/load-transformer` remains on the already-included Transformer router.
6. `src/how_llms_work/ml/transformer.py` — inspect only the unchanged `prepare_saved_transformer_prompt()` and `generate_saved_transformer_text()` contracts to guard against accidental generation changes; do not edit unless live evidence contradicts the export.
7. `src/how_llms_work/sse.py` — inspect only to preserve the current SSE response and framing contract; no edit is expected.
8. `pyproject.toml` and `README.md` — reconfirm Poetry, pytest, Ruff, Black, and mypy commands before final verification; no dependency change is expected.
9. `ts_llm_pipeline_explorer_file_structure(6).md` — compatibility reference only; verify that Ticket 025 does not require frontend implementation.

## Step 1 — Establish the live baseline and add nullable-request contract coverage

**Files and symbols:**
- `src/how_llms_work/schemas.py` — `LoadTransformerRequest.model_file`.
- `tests/test_load_transformer_route.py` — `_invalid_request_parameters()`, `test_load_transformer_request_model_declares_exact_fields_and_aliases()`, request-validation probe tests, and a new focused `modelFile: null` acceptance case.

**Purpose:**
Prove and implement the first acceptance criterion: JSON `null` is a valid required selector value while the public field set, aliases, strict generation validation, missing-field behavior, and named-string behavior remain unchanged.

**Actions:**
- Before editing, inspect the live versions of the four likely changed files and run the implementation baseline commands from the backend root.
- Add a focused request-contract test showing that `modelFile: null` is accepted, preserved as Python `None`, and reaches the route seam with the same prompt and generation fields as a named request.
- Remove `None` from the invalid-value set for `modelFile`, but keep omission, empty string, lists, mappings, numbers, and Booleans invalid.
- Keep `modelFile` in the generated schema's required set; nullability must not turn it into an omitted/defaulted field.
- Change only the `LoadTransformerRequest.model_file` type/field declaration needed to accept either one strict nonempty string or `None`.
- Preserve exact public alias `modelFile`, internal name `model_file`, extra-field behavior, finite number validation, and existing bounds.
- Update the request-validation probe or its expected JSON only as needed to represent `None` without starting real filesystem or generation work.

**Guardrails:**
- Do not add a default of `None`; the field must remain required.
- Do not accept an empty string as latest selection. Only JSON `null` means latest.
- Do not introduce `useLatest`, a magic filename, or another request field.
- Do not relax strict validation for the other four fields or change aliases, defaults, ranges, or extra-field behavior.
- Do not touch route registration, frontend code, or generation behavior in this step.

**Expected result:**
- The same five-field public request contract accepts a nonempty exact filename or JSON `null` for `modelFile`.
- Missing `modelFile`, empty string, and wrong-type values still receive HTTP `422` before the shared slot, filesystem, prompt preparation, or generation boundary is touched.

**Verification:**
- Run the request-schema and request-validation subset in `tests/test_load_transformer_route.py`.
- Inspect `LoadTransformerRequest.model_json_schema(by_alias=True)` to confirm `modelFile` is both required and nullable without changing the other properties.

## Step 2 — Extract one shared selected-file validation path without weakening named loading

**Files and symbols:**
- `src/how_llms_work/routes/train_transformer.py` — `_SelectedTransformerModelFile`, `_select_named_transformer_model_file()`, `_parse_saved_transformer_model_document()`, `_validate_loaded_transformer_model_document()`, `_build_loaded_transformer_model_snapshot()`, and `load_named_transformer_model()`.
- `tests/test_transformer_loading.py` — existing named exact-selection, one-read, no-cache, immutability, safe-error, and no-fallback tests.

**Purpose:**
Create the minimum reusable composition point needed for named and latest selection to pass an already-selected direct file through exactly the same read/parse/validate/materialize trust boundary.

**Actions:**
- Preserve `_select_named_transformer_model_file()` as the exact-only selector for nonempty named requests.
- Extract or otherwise centralize the current sequence that reads one selected path once, parses captured bytes, validates the current Phase 5 document against the selected filename, and builds one request-owned snapshot.
- Keep `load_named_transformer_model()` responsible for exact-name selection followed by the shared selected-file validation path.
- Ensure the shared selected-file path performs no second directory enumeration and no second file read after selection.
- Preserve the current closed failure mapping for named loading, including exception suppression and the exact message `The saved Transformer model could not be loaded.`
- Rerun the complete existing named-loader test module before adding latest behavior so any regression in exact-only selection, read count, snapshot isolation, or safety is detected immediately.

**Guardrails:**
- This is a required narrow prefactor, not a new public model registry or general filesystem abstraction.
- Do not move strict model validation into the route or duplicate it in a latest-only validator.
- Do not let named loading call the future latest selector or catch a named failure and try another file.
- Do not add caching, shared mutable snapshots, file repair, file-size limits, new persistence behavior, or model format support.
- Do not expose the new private decomposition as the test seam; existing and new tests must use stable public loading behavior.

**Expected result:**
- Named loading remains behaviorally identical and continues to pass all Ticket 023 and Ticket 024 loader regressions.
- One selected-file trust boundary is available for safe reuse by the latest selector.

**Verification:**
- Run all of `tests/test_transformer_loading.py` after the prefactor.
- Confirm existing one-read, reread/no-cache, candidate immutability, path safety, strict validation, and named no-fallback tests still pass unchanged or with only evidence-preserving test organization changes.

## Step 3 — Implement deterministic newest-valid model selection through the public loading boundary

**Files and symbols:**
- `src/how_llms_work/routes/train_transformer.py` — directory-resolution and candidate-safety functions, selected-file trust boundary from Step 2, plus a new stable public latest-loading function such as `load_latest_transformer_model()`.
- `tests/test_transformer_loading.py` — temporary-directory model fixtures, `_artifact_state()`, controlled timestamps, read-count instrumentation, link/junction helpers, and new latest-loading tests.

**Purpose:**
Satisfy the core Ticket 025 selection criteria at the approved public model-selection seam without testing private sorting helpers or private implementation containers.

**Actions:**
- Add a stable public latest-loading boundary that returns the same `LoadedTransformerModelSnapshot` type as named loading.
- Resolve and validate the genuine model directory once per latest invocation using the existing directory trust boundary.
- Enumerate only direct entries from that directory.
- For each entry, classify it without opening model contents:
  - require its exact name to satisfy the existing canonical Transformer filename parser;
  - reject symbolic links and Windows junctions;
  - require an ordinary direct file;
  - resolve it strictly and require the resolved parent to equal the genuine resolved directory;
  - obtain modification metadata only after the candidate passes the filesystem safety checks.
- Skip entries that are noncanonical, path-indirected, not ordinary files, disappear during inspection, cannot be safely resolved/stat-ed, or otherwise fail candidate classification.
- Capture each approved candidate's exact filename, path, and integer modification timestamp, using `st_mtime_ns` rather than float `st_mtime` so equal-time behavior is explicit and testable.
- Sort the complete approved candidate collection independently of `Path.iterdir()` order by descending modification time and then descending exact filename.
- Attempt candidates in that exact sorted order through the shared selected-file read/parse/strict-validation/materialization path from Step 2.
- Return immediately on the first strictly valid snapshot.
- Catch only the established safe per-candidate load failures needed to skip an invalid candidate; do not catch `BaseException` or suppress cancellation/system-exit semantics.
- If every approved candidate fails or no approved candidate exists, raise the stable public load failure used by the route to map the latest-specific message.
- Keep all state local to the invocation and discard the candidate collection and rejected snapshots after return/failure.
- Add public-boundary tests using `tmp_path` that prove:
  - irrelevant names, wrong case, unsupported architecture names, directories, special files when available, links, junction-classified entries, and containment escapes are never selected;
  - newer valid beats older valid;
  - a newer malformed/unreadable/incompatible/filename-mismatched candidate is skipped and an older valid candidate wins;
  - equal modification times select the alphabetically greatest exact filename;
  - reversing or otherwise controlling raw enumeration order does not change the result;
  - repeated latest calls with unchanged files select the same model and re-enumerate/revalidate;
  - changing candidate metadata or contents between requests affects the next request rather than a cache;
  - the selected file is read exactly once per latest request;
  - invalid candidates are not deleted, renamed, rewritten, repaired, quarantined, or intentionally metadata-mutated;
  - no valid candidate produces only the stable public failure without private path, filename, exception, or model details.

**Guardrails:**
- Do not validate candidates in raw directory enumeration order.
- Do not sort by filename before modification time; the primary key is newest modification time.
- Do not use creation time, access time, epoch value encoded in the filename, file size, directory position, or training loss as “newest.”
- Do not accept arbitrary JSON files or old/alternate model formats.
- Do not open a link/junction target or follow a candidate outside the genuine directory.
- Do not repair or rewrite invalid candidates and do not update their modification time intentionally.
- Do not cache the selected filename, parsed document, validated snapshot, directory listing, or sort result across requests.
- Do not alter named selection or silently call latest selection after a named failure.
- Do not require a specific private tuple/dataclass/helper name in tests; assert public selected snapshots and observable reads/order instead.

**Expected result:**
- One public latest-loading call deterministically returns the newest strictly valid current-format Saved Transformer Model snapshot.
- A damaged newest candidate no longer blocks an older valid candidate.
- Equal modification times resolve to the greatest exact filename.
- Repeated requests are stable when disk state is unchanged and responsive to disk changes when it is changed.
- No candidate is mutated and no internal failure detail escapes the public loading boundary.

**Verification:**
- Run the latest-selection tests in `tests/test_transformer_loading.py` after each small behavior slice: filtering, primary ordering, tie ordering, invalid skipping, no-valid outcome, read/no-cache, and immutability.
- Then rerun the entire `tests/test_transformer_loading.py` module to prove named behavior remains intact.

## Step 4 — Dispatch nullable requests and map the latest-specific SSE failure

**Files and symbols:**
- `src/how_llms_work/routes/train_transformer.py` — latest-load failure constant, `stream_saved_transformer_generation()`, and `load_transformer()`.
- `tests/test_load_transformer_route.py` — controlled named/latest loader seam, exact error helper, success stream test, prompt-failure tests, request isolation, and deterministic integration.

**Purpose:**
Connect `modelFile: null` to latest selection while inheriting the existing Ticket 024 prompt, generation, slot, and SSE workflow and preserving exact named behavior.

**Actions:**
- Add one route-owned latest-selection client message with the exact text `No valid saved Transformer model was found.`
- Change the stream's selector parameter to support `str | None` and update its documentation from exact named-only wording to the full one-model Saved Transformer Generation Run.
- Dispatch once at the loading stage:
  - non-`None` selector calls `load_named_transformer_model()` with that exact string;
  - `None` calls the new public latest-loading boundary.
- Keep both branches inside the existing same-process helper-offloading boundary so file enumeration, parsing, validation, and materialization remain off the FastAPI event-loop thread without creating training workers.
- Map a latest-selection failure to exactly one `error` event containing `No valid saved Transformer model was found.`
- Keep a named-selection failure mapped to exactly one `error` event containing `The saved Transformer model could not be loaded.`
- After either branch returns a snapshot, reuse the exact existing prompt preparation, `loaded` payload, deterministic generation, `result`, and empty `done` behavior without branch-specific duplication.
- Continue to emit the snapshot's actual selected filename in `loaded`; never emit `null`, a requested placeholder, a rejected filename, or a path.
- Pass `payload.model_file` through the route without inventing a flag or converting `None` into a string.
- Extend the controlled route-test dependencies so named and latest loaders can be asserted independently without depending on private sorting internals.
- Add/adjust route tests to prove:
  - `modelFile: null` invokes only the latest loader;
  - a named string invokes only the named loader;
  - latest success emits the exact selected filename and the same `loaded → result → done` payload key sets as named success;
  - prompt preparation and generation receive the selected latest snapshot and the same validated generation values;
  - latest no-valid behavior emits one exact safe `error` and no `loaded`, `result`, or `done`;
  - latest loader private exceptions/details never appear in the response;
  - prompt failures after successful latest selection use the existing exact prompt messages and occur before `loaded`;
  - generation failures after latest `loaded` retain the existing safe terminal behavior;
  - the shared Transformer slot is released after latest success and latest error;
  - latest loading creates no Request-Scoped Worker Group, child process, shared memory, training event, or worker label;
  - named failure still has no fallback to the latest loader.

**Guardrails:**
- Do not create separate latest and named generation streams; selection is the only branch.
- Do not change `loaded`, `result`, or `done` payload keys or event order.
- Do not change prompt trimming, tokenization, maximum prompt length, temperature/top-p/max-token validation, deterministic seed behavior, generation mathematics, or worker behavior.
- Do not emit rejected candidate names, paths, exceptions, tracebacks, model values, or validation details.
- Do not complete Ticket 026 deadline/cancellation redesign or Ticket 028 frontend command parsing in this step.
- Do not change the shared slot's process-local, immediate-`429`, no-queue semantics.

**Expected result:**
- `POST /load-transformer` supports both exact named selection and automatic latest valid selection through one public contract and one generation workflow.
- Latest success is indistinguishable from named success after the `loaded` event identifies the actual selected file.
- Latest absence/failure has its exact dedicated safe message, while named failure retains its existing message and exact-only semantics.

**Verification:**
- Run the request, latest success, latest no-valid, named non-fallback, prompt failure, generation failure, slot release, and no-training-resource subsets in `tests/test_load_transformer_route.py`.
- Rerun the complete `tests/test_load_transformer_route.py` module.

## Step 5 — Add one bounded real latest-model HTTP/SSE integration and close regressions

**Files and symbols:**
- `tests/test_load_transformer_route.py` — existing real named-model integration fixture and a new bounded real `modelFile: null` integration.
- `tests/test_transformer_loading.py` — public latest selector and read-count tests.
- `tests/test_train_transformer_route.py` — existing route/slot regressions, inspection only unless a shared-contract failure requires a minimal test correction.
- `src/how_llms_work/main.py`, `src/how_llms_work/ml/transformer.py`, and `src/how_llms_work/sse.py` — regression inspection only.

**Purpose:**
Prove the complete public latest path reaches real directory enumeration, strict current-format validation, request-owned snapshot materialization, model-owned prompt handling, deterministic generation, and exact SSE output without relying solely on controlled mocks.

**Actions:**
- Reuse the bounded real Saved Transformer Model fixture pattern from the existing named integration test.
- Create at least two canonical candidates under `tmp_path` with controlled modification times, including a newer invalid candidate and an older valid candidate, or an equivalent compact arrangement that simultaneously proves real skip-and-select behavior.
- Send `modelFile: null` through FastAPI `TestClient` twice with the same prompt and generation settings.
- Assert the selected exact filename, trimmed prompt, complete deterministic result, empty `done`, no training events, and no worker creation.
- Record reads so the selected valid file is read once per request; account explicitly for any invalid candidate reads required before it.
- Assert repeated responses are identical while files and metadata are unchanged and that source artifacts remain unchanged after both requests.
- Keep the integration bounded to a very small generation length and the existing real generation boundary; do not run training or maximum-token endurance.
- Rerun affected training-route and Transformer tests to ensure nullable load selection has not changed training request validation, fresh-weight initialization, route registration, shared-slot behavior, persistence, or numerical results.
- Inspect the final diff for Ticket 025 scope only before the full suite.

**Guardrails:**
- Do not use real production `.data` files.
- Do not depend on wall-clock sleeps or the host filesystem's directory enumeration order.
- Use controlled `os.utime(..., ns=...)` or equivalent exact metadata fixtures rather than fragile elapsed-time assumptions.
- Do not add browser automation, frontend tests, maximum-size generation, or process-lifecycle behavior reserved for later tickets.
- Do not weaken or duplicate the existing named real integration test.

**Expected result:**
- One bounded public HTTP/SSE test proves that `modelFile: null` selects the correct real strict model and runs deterministic generation end to end.
- All affected named-loading, training, persistence, numerical, and route contracts remain unchanged.

**Verification:**
- Run the new real latest integration in isolation.
- Run `tests/test_transformer_loading.py`, `tests/test_load_transformer_route.py`, and `tests/test_train_transformer_route.py` together.
- Inspect route registration to confirm all completed endpoints remain present.

## Focused verification plan

Run from the backend project root in Windows PowerShell:

```powershell
poetry run pytest tests/test_transformer_loading.py -q
poetry run pytest tests/test_load_transformer_route.py -q
poetry run pytest `
    tests/test_transformer_loading.py `
    tests/test_load_transformer_route.py `
    tests/test_train_transformer_route.py
```

Expected result:

- Nullable request validation passes while missing/empty/wrong-type selectors remain `422`.
- Public latest selection filters safely, sorts deterministically, skips invalid candidates, selects stable ties, rereads without cache, reads the selected file once per request, and leaves artifacts unchanged.
- Latest HTTP/SSE success emits the selected exact filename followed by the inherited deterministic `loaded → result → done` contract.
- Latest no-valid behavior emits exactly `No valid saved Transformer model was found.` with no private details or successful events.
- Named selection remains exact-only with its existing safe error and no fallback.
- Shared slot, prompt, generation, no-worker, and training regressions remain green.

## Full verification plan

Run the complete backend verification once after all focused tests pass:

```powershell
poetry run pytest
poetry run black --check .
poetry run ruff check .
poetry run mypy src
```

If the live repository has standardized on Ruff formatting instead of Black, use the repository-confirmed formatter command rather than running both formatters speculatively. Do not change dependencies or regenerate `poetry.lock` for Ticket 025.

Expected result:

- The complete pytest suite passes.
- Formatting verification passes.
- Ruff reports no lint errors.
- Mypy reports no issues under the current strict configuration.
- No claim is made until each command is actually executed successfully during implementation.

## Manual acceptance checklist

- [ ] Sending a structured `POST /load-transformer` request with all five required fields and `"modelFile": null` is accepted rather than rejected with `422`.
- [ ] Omitting `modelFile`, using an empty string, or using a wrong type remains invalid; no `useLatest` field or magic filename exists.
- [ ] Automatic selection searches only the genuine backend `.data` directory and only direct canonical ordinary files.
- [ ] Symbolic links, Windows junctions, directories, special files, wrong-case names, unsupported filename configurations, traversal forms, and arbitrary JSON files are never selected.
- [ ] Candidates are ordered by descending modification time and then descending exact filename, independent of raw directory order.
- [ ] A damaged or incompatible newest candidate is skipped and the next valid candidate is selected.
- [ ] Equal modification times select the alphabetically greatest exact filename.
- [ ] No matching/valid candidate emits one `error` event with exactly `No valid saved Transformer model was found.`
- [ ] Latest success emits exactly one `loaded` containing the actual selected filename and trimmed prompt, then one `result`, then one empty `done`.
- [ ] Latest success uses the same prompt validation, model-owned Vocabulary and Merge Table, deterministic generation settings, and complete result behavior as named success.
- [ ] The selected latest file is read once for the request, and a later request re-enumerates, rereads, and revalidates rather than using cached state.
- [ ] Invalid candidates and the selected candidate remain byte-for-byte unchanged; no file is deleted, repaired, rewritten, renamed, quarantined, or intentionally touched.
- [ ] Latest failures expose no path, rejected filename, exception, traceback, model value, array, or validation detail.
- [ ] A named request still validates only its exact named file and never falls back to latest or another model.
- [ ] Saved-model generation still creates no Transformer worker process, pipe, queue, manager, shared-memory block, training event, or worker label.
- [ ] `POST /train-transformer` still starts fresh training and all existing endpoints remain registered.
- [ ] No frontend layout, input parsing, result display, model format, generation mathematics, training behavior, dependency, or lockfile changed in Ticket 025.

## Expected files changed

Likely changed:

```text
src/how_llms_work/schemas.py
src/how_llms_work/routes/train_transformer.py
tests/test_transformer_loading.py
tests/test_load_transformer_route.py
```

Conditionally changed only if live verification reveals a direct regression or established test organization requires it:

```text
tests/test_train_transformer_route.py
```

No new production module, route module, schema module, fixture file, dependency, or frontend file is expected.

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/sse.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/routes/simple_chat.py
src/how_llms_work/routes/bpe_tokenize.py
src/how_llms_work/routes/neural_net.py
src/how_llms_work/routes/train_embed.py
frontend/**
.data/**
pyproject.toml
poetry.lock
poetry.toml
SPEC.md
CONTEXT.md
0003-load-saved-transformer-models-for-stateless-generation.md
025-load-the-newest-strictly-valid-saved-transformer-model.md
llm_works_file_structure.md
```

A file listed above may be inspected, but it must not be edited unless the live repository contradicts the supplied export and the change is strictly necessary for Ticket 025. Any such deviation must be documented before editing and kept minimal.

## Risk notes and safeguards

1. **Risk:** Raw filesystem enumeration order changes which model is selected across Windows, Linux, or repeated runs.
   - **Safeguard:** Materialize approved candidates, then explicitly sort by `(st_mtime_ns, exact_filename)` descending before any model validation attempt.

2. **Risk:** Floating-point modification timestamps collapse distinguishable values or make ties unstable.
   - **Safeguard:** Use integer `st_mtime_ns` metadata and an explicit descending exact-filename secondary key; tests set controlled nanosecond timestamps.

3. **Risk:** A damaged newest model blocks all automatic loading.
   - **Safeguard:** Run each candidate through the same strict selected-file trust boundary and skip its safe load failure only in the latest branch.

4. **Risk:** Refactoring the loader accidentally gives named requests fallback behavior.
   - **Safeguard:** Keep exact named selection and latest enumeration as separate public entry points that share only the post-selection read/validate/materialize path; preserve explicit named no-fallback regression tests.

5. **Risk:** Latest selection accepts a path-indirected or misleading entry before strict validation.
   - **Safeguard:** Reuse canonical filename parsing, direct-entry checks, link/junction rejection, ordinary-file classification, strict resolution, and resolved-parent equality before recording ordering metadata or opening contents.

6. **Risk:** Candidate filesystem state changes between enumeration, metadata capture, and read.
   - **Safeguard:** Treat resolution/stat/read failures as safe candidate rejection in the latest branch, read the selected candidate once into an in-memory snapshot, and never emit `loaded` before complete strict validation/materialization.

7. **Risk:** Broad invalid-candidate skipping hides programmer defects or control-flow exceptions.
   - **Safeguard:** Skip only the established bounded load/OS/validation failure set; do not catch `BaseException`, cancellation, keyboard interrupts, or process-exit semantics.

8. **Risk:** Latest-specific error handling leaks a rejected filename, path, parser detail, or model value.
   - **Safeguard:** Map the exhausted latest boundary to the one exact closed message and assert private markers are absent through the HTTP/SSE seam.

9. **Risk:** Rejected candidates are modified while being inspected.
   - **Safeguard:** Latest selection performs read-only enumeration/stat/read operations, never persistence operations, and tests byte content, entry names, mode, inode, size, and modification/change metadata where stable.

10. **Risk:** A cache returns a model that is no longer newest or valid.
    - **Safeguard:** Keep candidates and snapshots request-local; repeated calls must enumerate, sort, read, parse, validate, and materialize again.

11. **Risk:** Nullable schema work accidentally makes `modelFile` optional or permits empty strings/coercion.
    - **Safeguard:** Keep the field required, use strict string validation for the non-null branch, and retain missing/empty/wrong-type `422` tests.

12. **Risk:** Latest loading forks a second generation path and drifts from Ticket 024.
    - **Safeguard:** Branch only during model selection, then reuse the existing prompt preparation, deterministic generation, SSE event construction, shared slot, and cleanup path.

13. **Risk:** Tests become coupled to a private sorting helper or implementation container.
    - **Safeguard:** Test only public selected snapshots, observable read/validation order, exact HTTP/SSE events, and stable artifact state through temporary directories.

14. **Risk:** Ticket 025 expands into adjacent frontend, lifecycle, worker-label, or model-management work.
    - **Safeguard:** Prohibit frontend and dependency changes; leave Tickets 026–029 behavior untouched and perform a final scope-only diff review.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention. A suitable subject is:

```text
Load the newest valid Transformer model
```

Commit body should mention:

- required nullable `modelFile` support;
- deterministic newest-valid candidate filtering and ordering;
- invalid-candidate skipping with exact safe no-valid-model behavior;
- preserved named no-fallback semantics and shared generation stream;
- focused loader/route tests plus the exact full verification commands executed.

Do not create the commit during planning. `implement-prompt` must inspect the live repository, establish its own baseline, preserve user changes, implement only Ticket 025, run the tests and quality gates, inspect the final diff, and then create the implementation commit.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 025, `SPEC.md`, `CONTEXT.md`, ADR 0003, `py_llm_pipeline_explorer_file_structure(122).md`, and the relevant current repository files.

`implement-prompt` must re-inspect the live repository before editing, establish its own baseline, preserve user changes, implement only this work item, verify the complete change, report actual command results, inspect the scope-only diff, and create the implementation commit.
