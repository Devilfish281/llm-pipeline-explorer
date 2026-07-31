---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "023"
source_work_item: 023-safely-load-exact-saved-transformer-model-snapshots.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(102).md
frontend_code_reference: ts_llm_pipeline_explorer_file_structure(5).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 023: Safely Load Exact Saved Transformer Model Snapshots

## Initial checklist

- Confirm Ticket 023 is the only selected work item and is marked ready with no blockers.
- Treat `py_llm_pipeline_explorer_file_structure(102).md` as the latest Python Backend source of truth, not older snippets, exports, plans, or assumptions.
- Use `023-safely-load-exact-saved-transformer-model-snapshots.md`, `SPEC.md`, `CONTEXT.md`, and ADR 0003 as the required-behavior authority.
- Reuse the completed Phase 5 canonical Transformer parameter layout, semantic views, Saved Transformer Model builder, filename convention, and backend `.data` resolver.
- Add one reusable, route-independent-in-behavior named-model loading boundary without implementing `POST /load-transformer`, generation, latest-model fallback, frontend parsing, worker labeling, or lifecycle orchestration.
- Prove the trust boundary with temporary directories and complete current-format model fixtures; never read or alter the real backend `.data` models in automated tests.
- Preserve the user-reported pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Finish with focused loading tests, existing Transformer regressions, the full backend suite once, Ruff lint and format checks, strict mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript behavior to Python while treating the latest supplied Python Backend export as current-code truth.
2. `023-safely-load-exact-saved-transformer-model-snapshots.md` for immediate scope, acceptance criteria, approved test seam, constraints, and exclusions.
3. `py_llm_pipeline_explorer_file_structure(102).md` for the current Python implementation, tests, dependencies, paths, typing conventions, and stable public Transformer boundaries.
4. `SPEC.md` for the complete Phase 6 safe-selection, exact-format, one-snapshot, no-cache, path-indirection, finite-number, and testing decisions.
5. `0003-load-saved-transformer-models-for-stateless-generation.md` for the architectural separation between fresh-weight Transformer Training Runs and stateless Saved Transformer Generation Runs.
6. `CONTEXT.md` for the canonical meanings of Saved Transformer Model, Saved Transformer Generation Run, Transformer Training Run, and Request-Scoped Worker Group.
7. The current Phase 5 persistence and numerical boundaries:
   - `src/how_llms_work/routes/train_transformer.py`;
   - `src/how_llms_work/ml/transformer.py`;
   - `tests/test_train_transformer_persistence.py`;
   - `tests/test_transformer.py`;
   - `tests/test_transformer_completion.py`.
8. `llm_works_file_structure.md` only as historical TypeScript model-shape and ordering evidence when it agrees with the accepted current Python Phase 5 format.
9. `ts_llm_pipeline_explorer_file_structure(5).md` only to confirm that frontend command routing and display changes are absent and explicitly outside Ticket 023.
10. Official Python 3.12 `pathlib` and `json` documentation and official NumPy documentation as technical cross-checks for link/junction classification, resolved paths, duplicate-key-aware parsing, strict finite-number handling, and independent `float32` materialization.
11. Older code exports, earlier plans, real `.data` file contents, and historical TypeScript cache/loading behavior are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 023 creates the reusable backend trust boundary for one specifically named current-format Saved Transformer Model.

The public operation must accept only one canonical configuration-specific filename, locate an exact-case ordinary entry inside the genuine backend model directory, reject path syntax and path indirection before loading, read the selected file exactly once, parse the in-memory bytes with duplicate-key detection, strictly validate the complete Phase 5 model document, and materialize all parameters into one independent canonical C-contiguous NumPy `float32` storage block with the existing semantic views.

The returned request-owned snapshot must preserve:

- the selected exact filename;
- the validated six-field Transformer configuration;
- the ordered Vocabulary;
- the ordered Merge Table;
- the exact layer count and canonical parameter layout;
- the independent parameter storage and semantic views.

The boundary must reread and revalidate on every invocation, retain no loaded-model cache, leave every candidate byte-for-byte untouched, and expose only one stable non-sensitive failure outcome. The epoch filename segment remains artifact-selection metadata and must not become optimizer, training, checkpoint, or resumed-run state.

This ticket does not add an HTTP request schema, register `POST /load-transformer`, tokenize a prompt, generate text, select the newest valid model, share or change the Transformer request slot, run work off the event loop, add deadline/disconnect handling, modify the frontend, or change training behavior. Those are reserved for later Phase 6 tickets.

## Baseline evidence

- **Status:** User-reported, not tool-verified in this planning session.
- **Command:** `poetry run pytest`
- **Reported result:** All tests passed before planning.
- **Command:** `poetry run ruff check .`
- **Reported result:** Ruff passed before planning.
- **Command:** `poetry run mypy src`
- **Reported result:** `Success: no issues found` before planning.
- **Planning rule:** `implement-prompt` must inspect the live repository again and establish its own baseline evidence before editing. The implementation must not convert these user-reported results into tool-verified claims.

## Current code observations from the latest source

- `src/how_llms_work/ml/transformer.py` already exposes the canonical numerical authorities required by the ticket:
  - `build_transformer_parameter_layout(num_layers)`;
  - `build_transformer_parameter_views(storage, layout)`;
  - `InitializedTransformerParameters`;
  - `TransformerParameterLayout`;
  - `TransformerParameterViews`;
  - `get_transformer_preprocessing()`;
  - `SavedTransformerModel` and its nested typed structures;
  - `build_saved_transformer_model(run, preprocessing)`.
- `build_transformer_parameter_layout()` derives the complete flat record order, block indices, offsets, lengths, shapes, total parameter count, and current fixed Vocabulary size. A loader must use this boundary rather than restating parameter dimensions or array lengths.
- `build_transformer_parameter_views()` maps one caller-owned flat `float32` storage block into canonical semantic views and validates the layout/storage relationship. It is the correct existing authority for producing usable loaded parameters.
- `build_saved_transformer_model()` already proves the exact current Python Phase 5 document shape and ordering: `type`, `config`, `vocab`, `merges`, and `weights`, with the canonical config, top-level weight, block, and flattened parameter ordering.
- The Transformer Preprocessing Snapshot is application-wide and reused by training. A loaded model may use it only as current-format validation evidence; the returned model snapshot must copy the validated Vocabulary and Merge Table and must not alias global preprocessing containers.
- `src/how_llms_work/routes/train_transformer.py` already owns:
  - the backend-root `.data` resolver `get_transformer_model_directory()`;
  - the canonical configuration-specific filename builder;
  - exact key-order tuples for the persisted model document;
  - persistence-oriented structural validation;
  - serialization and atomic persistence.
- The current persistence validator checks exact key ordering, model type, fixed architecture fields, layer bounds, Vocabulary length and string types, basic Merge Table containers, weight groups, block key order, and block count.
- The current persistence validator does not provide the Ticket 023 trust boundary. It does not safely select a browser-controlled exact filename, reject case mismatches on Windows, reject links/junctions, detect duplicate JSON keys, derive and validate every parameter length through the canonical layout, reject Boolean or non-finite numerical entries at every coordinate, materialize request-owned `float32` storage, prove BPE coherence, check filename/config agreement, enforce one read, or prevent cross-request caching.
- The persistence validator is used by existing serializer tests with persistence-focused fixtures. Ticket 023 should add a loader-specific strict boundary rather than silently changing unrelated persistence semantics or forcing a broad persistence refactor.
- `tests/test_transformer.py` already proves canonical layout records, offsets, lengths, shapes, storage ownership, semantic views, and isolation.
- `tests/test_transformer_completion.py` already demonstrates how to create a complete valid current-format one-layer Saved Transformer Model through public Phase 5 boundaries and how to verify exact field order, parameter lengths, finite plain values, and fresh nested containers.
- `tests/test_train_transformer_persistence.py` already demonstrates backend-root path resolution, temporary-directory filesystem tests, exact filename expectations, serialization behavior, failure preservation, and concurrency controls.
- `main.py` has no `POST /load-transformer` route, and `schemas.py` has no saved-model request model. Both omissions are expected at this ticket boundary and must remain unchanged.
- The latest frontend Transformer hook still handles only five-number training commands. Frontend command parsing and saved-model rendering are later-ticket work and must not enter this diff.
- `pyproject.toml` already provides Python 3.12+, NumPy, pytest, Ruff, and strict mypy. No dependency or lockfile change is required.

## Acceptance-criteria classification

### Already satisfied and evidenced as reusable prerequisites

- The canonical Transformer parameter layout exists and is independently tested.
- Canonical semantic parameter views over caller-owned `float32` storage exist and are independently tested.
- The current Python Phase 5 Saved Transformer Model builder defines the exact document field order and complete persisted contents.
- The exact production model-directory resolver and configuration-specific filename builder exist.
- Existing tests provide safe temporary-directory and complete-model fixture prior art.
- The project already has the required standard-library, NumPy, pytest, Ruff, and mypy toolchain.

### Behavior present but evidence or strictness is incomplete

- Persistence validation already checks much of the top-level and nested structure, but not the complete untrusted-file contract.
- The filename builder produces canonical names, but no public parser/validator accepts and verifies an untrusted name.
- The data-directory resolver identifies the intended destination, but no load boundary validates the directory and candidate against symlinks, junctions, exact-case enumeration, ordinary-file status, and resolved containment.
- Existing model-construction tests prove canonical lengths and finite values for produced models, but no loader proves those properties for untrusted JSON.
- Existing public views prove storage interpretation, but no loader reconstructs one independent request-owned storage block from a persisted document.

### Not implemented

- One stable public exact-name Saved Transformer Model load operation.
- One request-owned loaded-model snapshot value.
- Pre-open canonical filename grammar validation.
- Exact-case direct-entry selection on case-insensitive filesystems.
- Model-directory and candidate symlink/junction rejection.
- Ordinary-file and resolved-containment validation.
- One-read in-memory snapshot behavior.
- Duplicate JSON object-key rejection.
- Strict current-format validation for every container and key.
- Canonical-layout-derived parameter lengths and flattening positions.
- Strict numerical-coordinate type, finite-value, and `float32` overflow validation.
- Coherent current Phase 5 Vocabulary and Merge Table validation.
- Filename architecture/configuration agreement.
- No-cache reread behavior.
- Stable sanitized loading failure.
- Artifact immutability evidence.
- Windows junction execution or an honestly recorded platform skip plus deterministic path-classification evidence.

### Cannot be determined until implementation

- Whether the implementation environment permits creating a Windows junction without elevated privileges.
- Whether the live repository has user changes after export `(102)` that affect exact placement or symbol names.
- Whether stricter type-checking reveals a need for a small public snapshot type in `ml/transformer.py`; the preferred smallest change keeps the loader and snapshot at the existing Transformer route/storage boundary while reusing public numerical types.

## Acceptance mapping

| Ticket acceptance area | Current state | Planned coverage |
|---|---|---|
| Exact valid file produces one complete snapshot | Missing | Steps 1–5 |
| Config, Vocabulary, Merge Table, blocks, and canonical ordering preserved | Partial producer-side evidence only | Steps 3–4 |
| Independent canonical `float32` storage and semantic views | Numerical primitives exist; loading missing | Step 4 |
| Canonical filename grammar and path syntax rejection before open | Builder exists; parser missing | Step 2 |
| Exact-case direct-entry selection | Missing | Step 2 |
| Model-directory/candidate symlink and junction rejection | Missing | Steps 2 and 5 |
| Ordinary-file and resolved-containment checks | Missing | Step 2 |
| Duplicate, missing, unexpected, and wrong-type JSON rejection | Partial in-memory persistence validation | Step 3 |
| Exact current model type/config/weight/block validation | Partial | Step 3 |
| Parameter lengths and positions from canonical layout | Layout exists; loader use missing | Steps 3–4 |
| Strict finite ordinary numerical coordinates | Missing | Steps 3–4 |
| Vocabulary length and coherent current BPE artifacts | Basic length/type checks only | Step 3 |
| Filename/configuration agreement; epoch remains metadata | Missing | Steps 2–3 |
| One read and one in-memory parse snapshot | Missing | Steps 1 and 5 |
| Reread on later invocation; no cache | Missing | Steps 4–5 |
| No application file-size cap; sanitized concrete failures | Missing | Step 5 |
| Invalid artifact remains unchanged | Missing evidence | Steps 1 and 5 |
| Windows junction execution or honest skip | Missing | Step 5 |

## Files to inspect before editing

1. `src/how_llms_work/routes/train_transformer.py`
   - Existing model key-order constants.
   - `_validate_saved_transformer_model_structure()`.
   - `_build_transformer_model_filename()`.
   - `get_transformer_model_directory()`.
   - Existing persistence operations and public test seams.
   - Preferred location for the public exact-name load boundary and request-owned snapshot because later `/load-transformer` orchestration will share this Transformer route/storage module.
2. `src/how_llms_work/ml/transformer.py`
   - `InitializedTransformerParameters`.
   - `TransformerParameterLayout`.
   - `TransformerParameterLayoutRecord`.
   - `TransformerParameterViews`.
   - `build_transformer_parameter_layout()`.
   - `build_transformer_parameter_views()`.
   - `get_transformer_preprocessing()`.
   - `SavedTransformerModel` and nested types.
   - `build_saved_transformer_model()`.
3. `src/how_llms_work/ml/bpe.py`
   - `Merge` and `apply_merges()` only to understand current Merge Table semantics and future compatibility; no change is expected in this ticket.
4. `tests/test_transformer.py`
   - Canonical layout, storage, view, dtype, and isolation prior art.
5. `tests/test_transformer_completion.py`
   - Complete current-format model construction and exact model-shape prior art.
6. `tests/test_train_transformer_persistence.py`
   - Filename, model-directory, temporary-directory, serialization, and artifact-preservation prior art.
7. `tests/fixtures/transformer_completion_reference.json`
   - Existing model-order and selected-coordinate evidence; reuse rather than duplicating a giant model fixture where practical.
8. `pyproject.toml`
   - Python 3.12, NumPy, pytest, Ruff, formatting, and strict-mypy configuration.
9. `SPEC.md`, `CONTEXT.md`, ADR 0003, and Ticket 023
   - Path safety, one-read/no-cache, strict model validation, safe failure, and scope authority.
10. `llm_works_file_structure.md`
   - Model shape and ordering reference only; do not import historical cache or TypeScript loading behavior.
11. `ts_llm_pipeline_explorer_file_structure(5).md`
   - Confirm that no frontend file belongs in this ticket.

## Step 1 — Establish the public exact-model loading contract with focused tests

**Files and symbols:**

- `tests/test_transformer_loading.py` — new focused acceptance test module.
- `src/how_llms_work/routes/train_transformer.py` — new intentionally public exact-name loader and request-owned snapshot boundary; exact private helper decomposition remains an implementation choice.
- Existing public model-building symbols from `src/how_llms_work/ml/transformer.py`.

**Purpose:**

Create acceptance-relevant evidence at the approved stable reusable seam before production implementation. The tests must define what a successful loaded snapshot exposes and what every unsafe or malformed named request rejects, without binding the implementation to private helper names or a particular JSON/path library call sequence.

**Actions:**

- Add `tests/test_transformer_loading.py`.
- Build a complete valid one-layer Saved Transformer Model through the already-tested public Phase 5 model-construction boundary rather than hand-authoring or reading a production `.data` model.
- Serialize that valid model into a pytest-managed temporary directory under its exact canonical filename.
- Create focused public-boundary tests proving a successful load returns:
  - the exact selected filename;
  - an exact copied configuration;
  - the complete ordered Vocabulary;
  - the complete ordered Merge Table;
  - the canonical layer count and layout;
  - one C-contiguous writable NumPy `float32` storage block;
  - usable semantic views with expected shapes and canonical coordinates.
- Prove the loaded snapshot is independent:
  - it does not alias parsed JSON lists;
  - it does not alias the global Transformer Preprocessing Snapshot;
  - two separate load calls do not share storage, views, configuration containers, Vocabulary containers, or Merge Table containers;
  - mutating one loaded parameter coordinate or copied metadata does not affect another load or the file.
- Record source bytes and metadata before each rejection test so later steps can prove the candidate is not rewritten, deleted, renamed, or repaired.
- Add the first table-driven invalid cases for empty names, separators, parent references, drive-letter and absolute forms, wrong extension, malformed architecture segments, unsupported fixed dimensions, out-of-range epochs/layers, leading-zero or otherwise noncanonical spellings, and exact-case mismatch.
- Add a handcrafted raw JSON case with a duplicate object key; do not attempt to represent this case with a Python dictionary.
- Keep test fixtures entirely under `tmp_path`; never read, copy, or mutate the real backend `.data` files.
- Confirm the focused tests fail because the public loader/snapshot boundary is not yet implemented.

**Guardrails:**

- Test through the intentionally public load operation and snapshot fields, not private validation helpers.
- Do not assert a specific regular expression, `os.scandir()`/`Path.iterdir()` choice, JSON decoder class, sorting operation, local variable, or exception chaining implementation.
- Do not import or call `POST /load-transformer`; HTTP and SSE are Ticket 024.
- Do not create a latest-model selector or allow `None` as a filename in this ticket.
- Do not make a test depend on a production saved-model file, current working directory, browser input, or frontend code.
- Do not duplicate all numerical reference fixtures when the public model builder and existing completion fixture provide the approved current-format source.

**Expected result:**

- The new test module precisely describes Ticket 023's public trust boundary and initially fails only because the public exact-name loader and snapshot are absent.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_loading.py -q
```

Expected before implementation:

- Collection or focused assertions fail at the missing public loading boundary.
- Existing Transformer tests remain untouched.

## Step 2 — Implement canonical filename validation and safe exact-case filesystem selection

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - Existing `_build_transformer_model_filename()`.
  - Existing `get_transformer_model_directory()`.
  - New public exact-name loader.
  - New narrowly scoped filename parsing and candidate-classification support.
- `tests/test_transformer_loading.py`
  - Filename, direct-entry, exact-case, path syntax, directory, special-file, link, and containment cases.

**Purpose:**

Create the outer trust boundary that rejects unsafe input before opening a candidate and selects only one exact ordinary entry from the genuine model directory, including on Windows case-insensitive filesystems.

**Actions:**

- Validate the supplied model filename as an exact strict string before any directory enumeration or file open.
- Accept only the canonical Phase 5 grammar generated by the existing filename builder:
  - `transformer-weights-e<epochs>-l<numLayers>-d32-h2-ff128-ctx32.json`;
  - approved epoch bounds;
  - approved layer bounds;
  - no alternate capitalization, whitespace, leading-zero normalization, extra suffix, or alternate fixed architecture segment.
- Parse the epoch and layer fields only as filename metadata, then reconstruct the canonical filename through the existing builder and require exact string equality. Do not retain epochs as training or numerical state.
- Reject empty strings, absolute paths, rooted paths, drive-letter forms, UNC forms, parent references, `/`, `\`, and any non-plain filename before a candidate file is opened.
- Resolve the production model directory through the existing backend-root resolver when no test directory is supplied.
- Validate the model directory itself:
  - it exists;
  - it is an ordinary directory;
  - it is not a symbolic link;
  - it is not a Windows junction when `Path.is_junction()` is available;
  - its resolved path is the genuine root used for containment checks.
- Enumerate only direct directory entries and require exact `entry.name == requested_filename`; never ask the filesystem to resolve a differently capitalized user path directly.
- Reject a missing exact entry without falling back to another model.
- Reject candidate directories, symbolic links, junctions, and non-ordinary files.
- Resolve the selected candidate and require that its resolved parent remains the genuine resolved model directory.
- Preserve one selected `Path` only long enough for the subsequent single read; do not create a registry, manifest, cache, or retained selection.
- Add or complete focused cases for:
  - exact valid name;
  - differently capitalized name;
  - path traversal and separators;
  - absolute/drive/UNC forms;
  - missing file;
  - directory masquerading as a model;
  - candidate symlink;
  - model-directory symlink;
  - resolved candidate outside the genuine directory;
  - special file where the platform permits creating one.

**Guardrails:**

- A supplied named request must never fall back to any other model.
- Do not use `Path.cwd()` to locate production models.
- Do not create the model directory while loading; a missing directory is a safe load failure.
- Do not follow or normalize unsafe browser-controlled path text into an accepted candidate.
- Do not use case-folded comparison, glob matching, prefix matching, or filename sorting for named selection.
- Do not add latest-model ordering; Ticket 025 owns that behavior.
- Do not open, parse, or inspect a candidate until all input, directory, exact-entry, link/junction, ordinary-file, and containment checks pass.

**Expected result:**

- One exact canonical ordinary filename can be selected safely from one genuine model directory.
- All path syntax, case mismatch, path indirection, and non-file candidates fail through the stable public loading outcome before model parsing.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_loading.py -q -k "filename or path or case or exact or directory or symlink or junction or containment or ordinary"
```

Expected result:

- Safe exact-name selection tests pass.
- No invalid candidate is opened or altered.
- Junction cases may remain conditionally skipped until Step 5 records platform capability honestly.

## Step 3 — Parse one in-memory read snapshot and strictly validate the complete current Phase 5 document

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - Existing model/config/weight/block/Merge Table key-order constants.
  - Existing persistence-oriented structural validator as reusable prior art only.
  - New loader-specific duplicate-aware parse and strict current-format validation boundary.
- `src/how_llms_work/ml/transformer.py`
  - `get_transformer_preprocessing()`.
  - `build_transformer_parameter_layout()`.
  - Saved-model typed structures and architecture constants.
- `tests/test_transformer_loading.py`
  - Raw JSON, exact schema, configuration, BPE coherence, parameter-container, and filename/config agreement cases.

**Purpose:**

Turn one file read into one unambiguous validated model document before any successful snapshot is published. This step closes the gaps left by persistence-only validation and ensures malformed values cannot survive until generation.

**Actions:**

- Read the selected file exactly once into bytes or text and perform all decode, parse, validation, and materialization from that one in-memory value.
- Do not impose an application-level byte-size limit before reading.
- Decode as UTF-8 and parse JSON with duplicate-object-name rejection at every nesting level.
- Reject non-standard JSON constants such as NaN, positive infinity, and negative infinity during parsing rather than allowing the decoder to normalize them.
- Require exact current-format object order and fields:
  - top level: `type`, `config`, `vocab`, `merges`, `weights`;
  - type exactly `decoder-transformer`;
  - config exactly `vocabSize`, `contextLen`, `embDim`, `numHeads`, `ffDim`, `numLayers`;
  - weights exactly `tokEmb`, `posEmb`, `blocks`, `lnFGamma`, `lnFBeta`, `headW`, `headB`;
  - every block exactly the existing sixteen canonical block fields;
  - every merge exactly `pair`, `merged`.
- Reject missing keys, unexpected keys, duplicate keys, wrong key order where the current Phase 5 format requires canonical ordering, wrong mapping/list/scalar container types, Boolean integers, and unsupported values.
- Require the fixed architecture fields to match the current Python implementation and require the block count to equal `numLayers`.
- Build the canonical parameter layout from the validated `numLayers`; require its Vocabulary size and fixed architecture to agree with the validated config.
- Validate filename metadata against config:
  - filename layer count equals `config.numLayers`;
  - fixed `d32`, `h2`, `ff128`, and `ctx32` segments agree with config;
  - the epoch segment is validated selection metadata only and is not added to config, parameters, optimizer state, or the returned numerical model state.
- Validate the ordered Vocabulary:
  - exact length equals `vocabSize`;
  - entries are strict strings;
  - entries are nonambiguous and unique;
  - order agrees with the current Python Phase 5 Transformer preprocessing contract.
- Validate the ordered Merge Table:
  - each pair contains exactly two strict strings;
  - `merged` is a strict string and coherently represents the pair replacement;
  - pair/merged dependencies and order are usable without guessing or repair;
  - the complete table agrees with the current Python Phase 5 preprocessing contract.
- Compare against current preprocessing only as a format/coherence authority. Copy validated values into request-owned containers later; never return or mutate the global preprocessing containers.
- Validate weight containers and block arrays are lists in every exact required position before numerical conversion.
- Add focused invalid variants for:
  - duplicate keys at top level and nested levels;
  - reordered, missing, and extra keys;
  - wrong type/config;
  - wrong block count or block keys;
  - wrong Vocabulary length, order, duplicate token, or token type;
  - malformed, reordered, incoherent, or unsupported merges;
  - filename/config mismatch;
  - non-list parameter containers;
  - missing and extra parameter arrays.

**Guardrails:**

- Do not repair, reorder, deduplicate, fill, truncate, migrate, or rewrite malformed input.
- Do not accept old TypeScript, partial, alternate, or approximate formats.
- Do not make the loader depend on a JSON-library-specific exception type in its public contract.
- Do not replace the existing persistence validator indiscriminately; loader validation is stricter and must not create unrelated serializer churn.
- Do not emit or log raw model values through the public failure path.
- Do not publish a partially validated snapshot before every structure, BPE, filename, and parameter-container check succeeds.

**Expected result:**

- One selected file is decoded and parsed once into one unambiguous complete current-format model.
- Every structural, configuration, BPE, and filename inconsistency fails before numerical state or a successful snapshot is returned.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_loading.py -q -k "json or duplicate or schema or config or block or vocab or merge or filename_config or container"
```

Expected result:

- Valid exact documents pass structural validation.
- Every ambiguous, missing, extra, incompatible, or incoherent document fails with the same safe public loading outcome.

## Step 4 — Materialize canonical request-owned `float32` parameters and the complete loaded snapshot

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - New loaded-model snapshot value.
  - New canonical parameter materialization within the public loader.
- `src/how_llms_work/ml/transformer.py`
  - `build_transformer_parameter_layout()`.
  - `build_transformer_parameter_views()`.
  - `InitializedTransformerParameters`.
  - `TransformerParameterLayoutRecord`.
- `tests/test_transformer_loading.py`
  - Canonical lengths, numerical types, finite conversion, flattened order, views, ownership, and repeat-load isolation.

**Purpose:**

Convert validated plain JSON parameter arrays into the exact numerical state later inference can consume, without duplicating layout constants or aliasing parsed, global, training, or cross-request state.

**Actions:**

- Allocate one new owning C-contiguous NumPy array with:
  - dtype exactly `np.float32`;
  - shape exactly `(layout.total_float_count,)`;
  - writable request-local ownership.
- Traverse only the existing `layout.records` in canonical order.
- For each record:
  - locate the corresponding top-level or indexed block list using the record key and block index;
  - require exact list length `record.length`;
  - reject missing or extra coordinates;
  - accept only ordinary JSON integer/float numbers at each coordinate;
  - reject Boolean, string, `null`, nested container, NaN, infinity, and any value that becomes non-finite when represented as `float32`;
  - copy coordinates into the exact `float_offset : float_offset + length` slice.
- Do not calculate any parameter length, offset, shape, or block order from duplicated constants.
- After the complete storage block is populated and proven finite, call `build_transformer_parameter_views(storage, layout)` to construct the semantic views.
- Wrap the layout, storage, and views in the existing `InitializedTransformerParameters` owner or an equally narrow public loaded-parameter owner that preserves the same canonical guarantees.
- Return one request-owned snapshot containing:
  - exact selected filename;
  - a fresh validated config container or immutable config value;
  - fresh ordered Vocabulary data;
  - fresh ordered Merge Table data;
  - canonical initialized parameters.
- Ensure no snapshot field aliases:
  - JSON parser lists/dictionaries;
  - global preprocessing lists/tuples/mappings;
  - a Transformer Training Run;
  - another loader call;
  - module-level mutable state.
- Preserve exact validated numerical values as `float32`; do not rerandomize, initialize defaults, normalize, round again, repair signed values, or rebuild weights from training.
- Add exact tests that compare representative top-level and block flattened coordinates with the source model and verify every semantic view shape.
- Add wrong-length tests for every top-level and block parameter family using the canonical layout records.
- Add numerical-type cases for Boolean, string, `null`, nested values, oversized finite JSON numbers that overflow `float32`, NaN, and infinities.
- Add repeated-load and mutation tests:
  - equal file bytes produce equal numerical contents;
  - storage/view objects are distinct per request;
  - changing one loaded storage coordinate changes only its own views;
  - changing a source JSON object after serialization cannot change a loaded snapshot;
  - changing the file between calls is observed by the second call.

**Guardrails:**

- Do not call Weight Initialization or create a Transformer Training Run.
- Do not allocate optimizer moments, gradients, shared memory, pipes, workers, caches, or generation state.
- Do not flatten arrays in ad hoc key order.
- Do not use NumPy conversion in a way that silently accepts Boolean values or overflows to infinity without detection.
- Do not return parsed parameter lists as the numerical model.
- Do not copy semantic views into separate unrelated arrays; all views must map the one canonical request-owned storage block.
- Do not cache layout-dependent loaded storage or the completed snapshot across requests.

**Expected result:**

- A valid model produces one complete request-owned inference snapshot with canonical finite `float32` storage and usable semantic views.
- Every invalid numerical shape or coordinate fails before the snapshot is returned.
- Independent calls are deterministic in content but isolated in identity and mutation.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_loading.py -q -k "layout or length or canonical or float32 or finite or material or view or storage or isolation or reread"
```

Expected result:

- Canonical length, dtype, flattening, semantic-view, and isolation tests pass.
- No untrusted parsed container remains part of the numerical snapshot.

## Step 5 — Complete one-read, no-cache, sanitized-failure, immutability, and platform safety evidence

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py`
  - Public load failure boundary.
  - Final exact-name loader lifecycle.
  - Model-directory/candidate classification.
- `tests/test_transformer_loading.py`
  - Read-count, changed-between-requests, failure sanitization, file-size-policy, artifact immutability, symlink, junction, and platform-limit reporting cases.

**Purpose:**

Prove the complete trust boundary remains stateless and non-destructive across success and every failure, and accurately handles Windows-specific path indirection without claiming tests that could not run.

**Actions:**

- Map expected filename, filesystem, decode, parse, schema, BPE, numerical conversion, and memory failures to one stable non-sensitive public loading failure.
- Use the specification-approved generic load failure wording or a stable public failure value that Ticket 024 can map directly to `The saved Transformer model could not be loaded.`.
- Suppress raw path, filename internals beyond the caller's submitted public value, exception text, traceback, JSON fragment, parameter value, array shape detail, and numerical state from the public failure.
- Keep internal exception chaining only when it cannot escape through the public boundary; do not catch process-control exceptions.
- Instrument the selected candidate read at the public seam and prove:
  - one successful invocation reads the selected file exactly once;
  - validation and materialization use that one in-memory snapshot;
  - the file is not reopened during the invocation;
  - a second invocation rereads the file;
  - changed valid contents are observed on the second invocation;
  - no loaded model, parsed object, path selection, or numerical storage is cached.
- Prove there is no application-level file-size rejection before the actual read. Use a valid representative padded JSON document or another bounded observable case without inventing a new supported maximum.
- For every failure stage, capture candidate bytes and existence before and after and prove the loader does not delete, truncate, rewrite, repair, rename, replace, chmod, or intentionally update metadata.
- Exercise model-directory and candidate symbolic links through real temporary paths where the platform permits.
- On Windows, attempt a real junction test using a temporary directory and the repository's documented PowerShell/test capability:
  - if junction creation is permitted, assert the public loader rejects it;
  - if privileges or platform support prevent creation, mark only that real-junction case skipped with the concrete reason and retain deterministic public path-classification tests using controlled `Path.is_junction()` behavior.
- On non-Windows platforms, skip only the Windows-junction integration case while still testing the shared path-classification seam.
- Rerun all invalid cases and ensure they expose the same stable public failure rather than raw lower-level differences.

**Guardrails:**

- Do not add a model file-size constant.
- Do not delete or quarantine invalid models.
- Do not cache successful or failed selection results.
- Do not expose a list of directory candidates or validation reasons.
- Do not claim real junction coverage passed when it was skipped or simulated.
- Do not use a broad `except BaseException`.
- Do not add logging that includes model contents, arrays, or unsafe path details.
- Do not add latest-model skipping; named failure remains terminal.

**Expected result:**

- Every invocation is one independent read/validate/materialize operation.
- Every failure is stable and non-sensitive.
- Every candidate remains untouched.
- Symlink and junction protections are evidenced honestly for the implementation platform.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_loading.py -q -k "read_once or reread or cache or changed or failure or unchanged or size or symlink or junction or classification"
```

Expected result:

- One-read/no-cache, safe-failure, artifact-preservation, and path-indirection tests pass.
- Any platform skip states exactly what could not be created or exercised.

## Step 6 — Run focused regressions, quality gates, and final scope inspection

**Files and symbols:**

- `src/how_llms_work/routes/train_transformer.py` — completed exact-model loading boundary.
- `tests/test_transformer_loading.py` — complete Ticket 023 acceptance suite.
- `src/how_llms_work/ml/transformer.py` — conditional only if the smallest safe implementation required a public snapshot/materialization type.
- Existing Transformer source/tests — regression verification only.
- Git working tree — scope and generated-file inspection.

**Purpose:**

Prove Ticket 023 is complete, regression-safe, typed, formatted, lint-clean, and limited to the reusable named-file loading trust boundary.

**Actions:**

- Run the complete new focused loading suite.
- Run existing canonical layout, completion/model-construction, and persistence tests unchanged.
- Run the complete backend pytest suite once after focused tests pass.
- Run Ruff lint and Ruff formatting verification.
- Run strict mypy over `src`.
- Run `git diff --check` and inspect `git status --short`.
- Confirm no test wrote to the real `.data` directory and no saved model, temporary file, cache, generated fixture, or test artifact entered the diff.
- Inspect the final diff against every Ticket 023 criterion.
- Confirm no HTTP route, schema, SSE, generation, latest-selection, request-slot, worker, frontend, documentation, or dependency behavior entered the change.
- Record all actually executed results honestly, including any Windows junction skip.

**Guardrails:**

- Do not weaken existing tests, strict mypy, Ruff rules, or numerical tolerances.
- Do not fix unrelated failures by modifying unrelated source files.
- Do not claim full platform coverage when the environment prevented a real junction test.
- Do not claim success unless every listed command was actually executed and its result recorded.
- Do not commit during `to-plan-prompt`; commit creation belongs to `implement-prompt` after successful verification.

**Expected result:**

- Ticket 023 has one stable reusable exact-model loading seam with complete safety and numerical evidence.
- Completed Phase 5 training, model construction, persistence, endpoints, and frontend behavior remain unchanged.
- The final diff is minimal and ready for the later named-generation route ticket.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_loading.py -q
poetry run pytest tests/test_transformer.py tests/test_transformer_completion.py tests/test_train_transformer_persistence.py -q
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
git diff --check
git status --short
```

Expected result:

- All focused and regression tests pass.
- The complete backend suite passes.
- Ruff reports no lint or formatting violations.
- Strict mypy reports no issues.
- `git diff --check` reports no whitespace errors.
- `git status --short` lists only the approved Ticket 023 files and no generated artifacts.

## Focused verification plan

Run before the complete suite:

```powershell
poetry run pytest tests/test_transformer_loading.py -q
poetry run pytest tests/test_transformer.py tests/test_transformer_completion.py tests/test_train_transformer_persistence.py -q
```

The focused evidence must cover:

- exact canonical filename acceptance;
- pre-open path syntax rejection;
- exact-case direct-entry selection;
- ordinary-file, symlink, junction, and containment rules;
- one read per invocation and no cache;
- duplicate JSON key rejection;
- exact current-format key and container validation;
- current architecture and filename/config agreement;
- complete ordered Vocabulary and Merge Table coherence;
- canonical layout-derived lengths and flattening;
- strict ordinary finite numerical values;
- independent writable C-contiguous `float32` storage;
- usable semantic views;
- cross-request and parsed/global-state isolation;
- stable sanitized failures;
- artifact immutability;
- honest Windows junction evidence.

## Full verification plan

Run once after all focused tests pass:

```powershell
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
```

Expected result:

- All backend tests pass.
- Ruff lint and format checks pass.
- Strict mypy reports no issues.

## Manual acceptance checklist

- [ ] Ticket 023 is the only implemented work item.
- [ ] The implementation uses `py_llm_pipeline_explorer_file_structure(102).md` only as planning evidence and re-inspects the live repository before editing.
- [ ] A valid exact canonical filename inside a pytest temporary model directory loads successfully.
- [ ] Unsafe names are rejected before a candidate is opened.
- [ ] Differently capitalized names fail even on a case-insensitive filesystem.
- [ ] Named selection enumerates direct entries and never falls back to another model.
- [ ] The production directory derives from the backend project, not `Path.cwd()`.
- [ ] The model directory and candidate are rejected when symbolic links or junctions.
- [ ] The candidate is an ordinary direct file whose resolved parent is the genuine resolved model directory.
- [ ] The selected file is read exactly once per invocation.
- [ ] Duplicate JSON object keys are rejected at every nesting level.
- [ ] Missing, extra, reordered, wrong-type, partial, old, or alternate model structures are rejected.
- [ ] `type` is exactly `decoder-transformer`.
- [ ] Config fields and block count match the current architecture.
- [ ] Filename architecture segments agree with validated config.
- [ ] The filename epoch segment remains artifact metadata only.
- [ ] Vocabulary length and order match `vocabSize` and the current Phase 5 BPE contract.
- [ ] The complete Merge Table is coherent, ordered, and copied.
- [ ] Every parameter array length and flat position comes from `build_transformer_parameter_layout()`.
- [ ] Boolean, string, `null`, nested, NaN, infinity, and `float32`-overflow coordinates are rejected.
- [ ] A valid model produces one finite writable C-contiguous `float32` storage block.
- [ ] Semantic views are built through `build_transformer_parameter_views()`.
- [ ] The returned snapshot owns fresh config, Vocabulary, Merge Table, storage, and views.
- [ ] The snapshot does not alias global preprocessing, parsed JSON containers, training state, or another request.
- [ ] A second invocation rereads changed valid file contents.
- [ ] No successful or failed model load is cached.
- [ ] No application file-size cap was introduced.
- [ ] Public failures expose no path, exception, traceback, model value, or numerical state.
- [ ] Invalid candidates remain byte-for-byte unchanged and are never repaired, deleted, rewritten, or renamed.
- [ ] A real Windows junction test ran when permitted, or the limitation and skip reason are recorded honestly while the public classification seam remains covered.
- [ ] No `POST /load-transformer` route, request schema, SSE event, generation, latest-model fallback, frontend parser, worker label, or lifecycle change was added.
- [ ] No production dependency or lockfile changed.
- [ ] Automated tests never touched the real `.data` directory.
- [ ] Focused tests, Transformer regressions, the complete suite, Ruff, Ruff formatting, and mypy were actually run before completion is claimed.

## Expected files changed

Likely changed:

```text
backend/src/how_llms_work/routes/train_transformer.py
backend/tests/test_transformer_loading.py
```

Conditionally changed only if live-code inspection proves a separate public numerical snapshot type is the smallest safe boundary:

```text
backend/src/how_llms_work/ml/transformer.py
```

No fixture file is expected if the new tests can build a complete valid model through the existing public `build_saved_transformer_model()` boundary. A small fixed raw-JSON fixture is acceptable only if it materially improves duplicate-key or malformed-document evidence without duplicating a full model.

## Files not to change

```text
backend/src/how_llms_work/main.py
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/sse.py
backend/src/how_llms_work/routes/__init__.py
backend/src/how_llms_work/routes/simple_chat.py
backend/src/how_llms_work/routes/bpe_tokenize.py
backend/src/how_llms_work/routes/neural_net.py
backend/src/how_llms_work/routes/train_embed.py
backend/src/how_llms_work/ml/bpe.py
backend/src/how_llms_work/ml/math_utils.py
backend/src/how_llms_work/ml/matrix.py
backend/src/how_llms_work/ml/neural_net.py
backend/src/how_llms_work/ml/transformer_worker.py
backend/tests/test_train_transformer_route.py
backend/tests/test_transformer_worker.py
backend/tests/test_transformer_worker_group.py
backend/pyproject.toml
backend/poetry.lock
backend/.data/
frontend/
SPEC.md
CONTEXT.md
0003-load-saved-transformer-models-for-stateless-generation.md
```

Existing Transformer tests may be read and run but should remain unchanged unless live repository evidence reveals a direct contradiction that blocks the approved public test seam. Any such change must be narrowly justified in the implementation record.

## Risk notes and safeguards

1. **Risk:** Directly joining the user filename to `.data` could accept traversal or differently capitalized files on Windows.
   - **Safeguard:** Validate canonical plain-name grammar first, enumerate direct entries, require exact string equality, reject links/junctions, and prove resolved containment.
2. **Risk:** Checking only the candidate while the model directory itself is a symlink or junction could bypass the storage boundary.
   - **Safeguard:** Classify and resolve the directory before enumeration and reject directory indirection.
3. **Risk:** Python's default JSON decoder accepts repeated object names and retains one value, making an ambiguous model appear valid.
   - **Safeguard:** Use duplicate-aware object-pair validation at every nesting level and handcrafted duplicate-key tests.
4. **Risk:** Reusing the persistence validator alone could accept wrong parameter lengths or invalid numerical coordinates.
   - **Safeguard:** Add a loader-specific strict validation/materialization boundary driven by the canonical layout.
5. **Risk:** `bool` is a subclass of `int`, so broad numeric checks could accept `true`/`false` as weights or configuration values.
   - **Safeguard:** Require exact ordinary numeric types and test Boolean values independently.
6. **Risk:** A large finite JSON number can overflow when converted to NumPy `float32`.
   - **Safeguard:** Validate source finiteness and completed `float32` storage finiteness before publishing views.
7. **Risk:** Ad hoc parameter iteration could drift from the canonical Phase 5 flat layout.
   - **Safeguard:** Traverse `TransformerParameterLayout.records` and use each record's exact offset, length, block index, and shape.
8. **Risk:** Comparing only `vocabSize` could accept a reordered or incoherent BPE model.
   - **Safeguard:** Validate the complete ordered Vocabulary and Merge Table against the fixed current Phase 5 preprocessing contract, then copy them into request-owned containers.
9. **Risk:** Returning parsed lists or the global preprocessing objects could leak mutable state between requests.
   - **Safeguard:** Allocate fresh metadata containers and one fresh owning numerical block for every public load invocation; add mutation-based isolation tests.
10. **Risk:** Reading a candidate more than once could combine different filesystem snapshots if the file changes mid-request.
    - **Safeguard:** Perform one read, then decode, parse, validate, and materialize exclusively from the captured in-memory value.
11. **Risk:** Caching could make later requests ignore file changes or share mutable parameters.
    - **Safeguard:** Keep no module-level loaded model, parsed document, path result, or storage cache; prove changed-between-request behavior.
12. **Risk:** Broad error reporting could reveal internal paths, model values, or validation details.
    - **Safeguard:** Collapse expected load failures to one stable public outcome and keep raw details out of the public boundary.
13. **Risk:** Strengthening the existing persistence validator could break unrelated persistence fixtures and expand scope.
    - **Safeguard:** Keep persistence behavior stable and add the stricter loader-specific boundary unless live evidence proves safe consolidation is smaller.
14. **Risk:** Tests could accidentally alter real trained models.
    - **Safeguard:** Require `tmp_path` for every file operation and assert the production `.data` resolver is tested without writing there.
15. **Risk:** Junction creation can require platform support or privileges.
    - **Safeguard:** Attempt the real Windows case when possible, skip it transparently when not, and retain deterministic public classification coverage.
16. **Risk:** Ticket 023 could drift into route, generation, or frontend work because those are the visible learner outcome.
    - **Safeguard:** Stop at the reusable named-model snapshot boundary; leave HTTP/SSE to Ticket 024, latest selection to Ticket 025, lifecycle hardening to Ticket 026, worker display to Ticket 027, and frontend work to Tickets 028–029.

## Commit guidance after tests pass

Suggested outcome-oriented commit title:

```text
feat: safely load exact saved transformer models
```

The commit body should mention:

- exact-case safe model-file selection;
- duplicate-aware current-format validation;
- canonical request-owned `float32` materialization;
- one-read/no-cache behavior and non-destructive safe failures;
- focused test coverage and actual pytest/Ruff/mypy results.

Do not create the commit until all implementation verification passes. Do not include unrelated files, production `.data` artifacts, caches, or temporary calibration files.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- `plan023.md`;
- `023-safely-load-exact-saved-transformer-model-snapshots.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0003-load-saved-transformer-models-for-stateless-generation.md`;
- `py_llm_pipeline_explorer_file_structure(102).md`;
- `ts_llm_pipeline_explorer_file_structure(5).md`;
- `llm_works_file_structure.md`.

`implement-prompt` must inspect the live repository again, establish its own baseline, preserve user changes, implement only Ticket 023, run focused verification before the complete suite, report actual results honestly, inspect final scope, and create the implementation commit only after all required checks pass.
