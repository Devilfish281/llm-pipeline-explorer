---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "015"
source_work_item: 015-build-canonical-transformer-parameter-layouts-and-initialization.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(39).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 015: Build canonical Transformer parameter layouts and initialization

## Initial checklist

- Confirm Ticket 015 is the only selected work item and that its Ticket 012 blocker is represented by the completed shared `Mulberry32` implementation and tests in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(39).md` as the current-code source of truth; older exports, snippets, and plans must not override it.
- Use `SPEC.md`, `CONTEXT.md`, and ADR 0002 for the fixed Transformer architecture, canonical flat order, mixed-precision rules, and Weight Initialization terminology.
- Use `llm_works_file_structure.md` only as the TypeScript Reference Implementation for array shapes, Xavier fan-in/fan-out values, and random-consumption order.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing those commands as tool-verified in this planning session.
- Limit production work to the stable public Transformer layout, flat-view, parameter-count, and fresh Weight Initialization boundary in `src/how_llms_work/ml/transformer.py`.
- Finish with focused Transformer tests, deterministic utility and matrix regressions, Black checking for changed Python files, the full pytest suite, Ruff, strict mypy, and final scope inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript behavior to Python and treat `py_llm_pipeline_explorer_file_structure(39).md` as the source of truth for current Python code.
2. `015-build-canonical-transformer-parameter-layouts-and-initialization.md` for required behavior, acceptance criteria, approved test seam, blocker, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(39).md` for current source, tests, fixture style, dependency configuration, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for durable Phase 5 architecture, canonical-layout, initialization, dtype, and process-boundary decisions.
5. `llm_works_file_structure.md`, especially TypeScript `src/routes/train-transformer/transformer.ts`, `weight-layout.ts`, and related model initialization code, as compatibility evidence only.
6. Older exports, prior plans, snippets, and specification statements about current file contents are non-authoritative when they conflict with export `(39)`.

## Work-item summary

Ticket 015 adds the sole canonical authority for interpreting every Transformer parameter coordinate. The public Transformer boundary must produce one stable ordered layout for each supported layer count from one through six. Every record must identify the semantic parameter key, optional block index, flat float offset, length, shape, and canonical total size. The resulting regions must begin at zero, remain contiguous and non-overlapping, contain no gaps, and end exactly at the reported total float count.

The same layout must construct exact C-order NumPy `float32` views over a caller-owned flat storage array. The view boundary must reject the wrong dtype, wrong rank, non-contiguous storage, insufficient capacity, inconsistent or non-canonical records, and any region outside the canonical range. Extra trailing capacity is valid, but no returned view may extend into it.

Fresh Weight Initialization must allocate independent flat storage, build views only through the canonical layout boundary, and consume one caller-owned `Mulberry32` stream seeded by the run owner. Random matrices must be filled in the approved traversal order, which is intentionally different from flat layout order: each block's `wQ`, `wK`, `wV`, `wO`, `ff1W`, and `ff2W`, followed by `tokEmb`, `posEmb`, and `headW`. Every coordinate uses Xavier uniform initialization, consumes exactly one draw, and is materialized immediately as `float32`. Biases and Layer Normalization beta arrays are zero; Layer Normalization gamma arrays are one; deterministic fills consume no draws.

With the current immutable preprocessing snapshot, the fixed Vocabulary has `392` tokens. Therefore the canonical planning anchors are:

```text
fixed non-block floats = 26,568
floats per block       = 12,704
total floats           = 26,568 + 12,704 × numLayers

fixed random draws     = 26,112
random draws per block = 12,288
total random draws     = 26,112 + 12,288 × numLayers
```

Representative totals are:

| Layers | Layout records | Total floats | Total bytes | Random draws | Final Mulberry32 state from seed 42 |
|---:|---:|---:|---:|---:|---:|
| 1 | 22 | 39,272 | 157,088 | 38,400 | 2,037,747,242 |
| 2 | 38 | 51,976 | 207,904 | 50,688 | 2,689,826,346 |
| 3 | 54 | 64,680 | 258,720 | 62,976 | 3,341,905,450 |
| 4 | 70 | 77,384 | 309,536 | 75,264 | 3,993,984,554 |
| 5 | 86 | 90,088 | 360,352 | 87,552 | 351,096,362 |
| 6 | 102 | 102,792 | 411,168 | 99,840 | 1,003,175,466 |

These values are planning anchors, not a substitute for the independent fixture required during implementation. The fixture must still preserve complete offsets, lengths, shapes, selected coordinate bit patterns, deterministic fills, full-array checksums, draw counts, and final generator states without generating expected values through the production Transformer helper.

This ticket does not add forward or backward Transformer mathematics, Adam updates, gradient buffers, worker processes, shared-memory allocation, request validation, FastAPI routing, SSE behavior, model conversion, persistence, cache loading, resuming, or frontend changes.

## Baseline evidence

- **Status:** User-reported.
- **Command:** `poetry run pytest`
- **Result:** The user reports that all tests passed before planning.
- **Command:** `poetry run ruff check .`
- **Result:** The user reports that Ruff passed before planning.
- **Command:** `poetry run mypy src`
- **Result:** The user reports `Success: no issues found` before planning.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing. None of these commands was tool-verified in this planning session.

## Current code observations from the latest source

- `src/how_llms_work/ml/transformer.py` is no longer empty. It owns the completed immutable Transformer preprocessing boundary from Ticket 014, including the exact corpus, BPE artifacts, `392`-token ordered Vocabulary, `3,195` token IDs, `3,179` Training Sequences, generation seed IDs, and four Logical Training Shards.
- The current `transformer.py` module imports no NumPy symbols and defines no architecture constants for context length, embedding dimension, head count, head dimension, or feed-forward dimension. It has no parameter-layout record, layout builder, total-count function, flat-view builder, or Weight Initialization function.
- The current `transformer.py.__all__` exports only preprocessing constants, records, shard construction, and `get_transformer_preprocessing()`. Ticket 015 must extend that stable public seam without removing or renaming existing exports.
- `src/how_llms_work/ml/math_utils.py` already provides the canonical request-owned `Mulberry32` class with exact unsigned state, `random()`, `state`, and `draw_count`. No new generator or module-global random state is needed.
- `tests/test_math_utils.py` and `tests/fixtures/math_utils_reference.json` already provide independent fixture provenance, exact random streams, final states, draw counts, same-seed isolation, interleaving, sequential reuse, and concurrent reuse patterns.
- `src/how_llms_work/ml/matrix.py` already provides strict NumPy `float32` validation and test conventions for dtype, C-contiguity, finiteness, memory sharing, exact structure, and explicit failure types. Ticket 015 should reuse those conventions but must not move layout ownership into `matrix.py`.
- `tests/test_transformer.py` currently verifies only preprocessing, shard construction, immutability, reuse, concurrent first access, and failure retry. It contains no layout, view, initialization, parameter-count, or storage-isolation tests.
- `tests/fixtures/transformer_preprocessing_reference.json` is a large independent preprocessing fixture. It should remain unchanged; layout and initialization evidence belongs in a separate focused fixture.
- `src/how_llms_work/ml/transformer_worker.py` and `src/how_llms_work/routes/train_transformer.py` remain empty. There is no current Saved Transformer Model read path, cache, resume path, shared-memory allocator, worker attachment logic, optimizer, or route orchestration to modify.
- `pyproject.toml` already supplies Python 3.12, NumPy, pytest, Ruff, Black, and strict mypy. No dependency or lockfile change is expected.

## Acceptance criteria coverage

### Already satisfied and evidenced

- The fixed architecture's Vocabulary input is available from the completed immutable preprocessing snapshot and currently contains exactly `392` tokens.
- The shared JavaScript-compatible `Mulberry32` owner, state, draw count, exact output behavior, and request-stream isolation are implemented and independently tested.
- NumPy is already a production dependency, and project tests already use exact dtype, C-contiguity, finiteness, and memory-sharing assertions.
- Python 3.12, pytest, Ruff, Black, and strict mypy are configured.
- No Saved Transformer Model read, load, cache, resume, or skip-training path exists in the current Python source.
- The TypeScript Reference Implementation and ADR 0002 establish the sixteen block keys, top-level keys, shapes, Xavier formula, and random-consumption order.

### Behavior present but evidence incomplete

- The immutable preprocessing snapshot supplies the canonical Vocabulary size, but no layout boundary currently consumes it.
- The TypeScript Reference Implementation contains compatible model structures and initialization behavior, but no independent Python fixture currently freezes complete offsets, selected initialized coordinates, final PRNG states, or flat-array checksums.
- Existing matrix tests demonstrate suitable validation and memory-alias assertions, but they do not exercise parameter storage or semantic Transformer views.

### Partially implemented

- `src/how_llms_work/ml/transformer.py` is the correct existing production owner and `tests/test_transformer.py` is the correct existing public test seam, but both currently stop at preprocessing.
- The canonical random generator exists, but Transformer Weight Initialization does not yet use it.

### Not implemented

- Fixed Transformer model-dimension constants for context length `32`, embedding dimension `32`, two heads, head dimension `16`, and feed-forward dimension `128` at the Transformer boundary.
- Validation of requested layer count `1..6`, including explicit rejection of Booleans and non-integers.
- One immutable canonical parameter-layout representation.
- Exact top-level and sixteen-per-block flat ordering.
- Stable semantic key and block identity for every record.
- Exact float offsets, byte offsets, lengths, shapes, record counts, total float counts, and total byte counts.
- Gap-free, overlap-free, in-range structural validation.
- Exact layout fixtures for one, two, and six layers plus all-depth invariants.
- A public total-parameter-count boundary derived from the canonical layout rather than a parallel formula table.
- A strict one-dimensional C-contiguous `float32` backing-storage contract.
- Exact C-order NumPy views using `byte_offset = float_offset × 4`.
- Rejection of undersized, wrong-dtype, wrong-rank, non-contiguous, inconsistent, and out-of-range storage/layout combinations.
- Acceptance of extra storage capacity while keeping every view inside the logical canonical range.
- Fresh independent `float32` storage for each Weight Initialization.
- Separate random traversal order distinct from flat layout order.
- Coordinate-by-coordinate Xavier initialization with immediate `float32` materialization.
- Exact zero fills for all biases and beta arrays.
- Exact one fills for all Layer Normalization gamma arrays.
- Exact draw-count and final-state verification.
- Selected exact float32 coordinate fixtures and complete flat-storage checksums.
- Full-array finiteness validation before successful return.
- Same-seed equivalence with no memory aliasing across runs.
- Different-depth isolation and exact depth-specific layouts.

### Evidence limitations

- The latest export is a source snapshot rather than an executable repository checkout, so no baseline command or focused test was run during planning.
- Exact initialized coordinate bit patterns and full-array checksums are not present in the current Python source. They must be independently captured from the TypeScript Reference Implementation or calculated by a scalar reference that imports no production Transformer module.
- Tests must not call the new production layout builder or initializer to create their expected fixture content.
- The ticket intentionally leaves private record decomposition and container implementation flexible. Tests must exercise the stable public semantic boundary rather than requiring a particular dictionary type or private class name.
- Worker read-only weight views, shared-memory names, gradient storage, optimizer storage, and persistence conversion are later-ticket consumers. Ticket 015 must make its boundary suitable for those consumers without implementing them now.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer.py` — existing `__all__`, preprocessing constants and records, `TransformerPreprocessingSnapshot`, `get_transformer_preprocessing()`, and destination for the new public layout, view, total-count, and Weight Initialization boundary.
2. `src/how_llms_work/ml/math_utils.py` — `Mulberry32`, `state`, `draw_count`, and canonical generator ownership; import and reuse without modification.
3. `src/how_llms_work/ml/matrix.py` — strict dtype, rank, contiguity, finiteness, and type-alias conventions to follow; do not move parameter-layout ownership here.
4. `tests/test_transformer.py` — existing public Transformer seam and fixture-loading style; extend without weakening preprocessing tests.
5. `tests/test_math_utils.py` — independent fixture provenance, exact PRNG, state, draw-count, same-seed, interleaving, and concurrent-isolation prior art.
6. `tests/test_matrix.py` — public-symbol, dtype, C-contiguity, invalid-storage, `np.shares_memory()`, and failure-preservation assertion patterns.
7. `tests/fixtures/transformer_preprocessing_reference.json` — source for the fixed `392`-token Vocabulary evidence only; do not append layout or initialization data to this fixture.
8. `tests/fixtures/math_utils_reference.json` — fixture organization and generator-state evidence only; do not change existing utility expectations.
9. `pyproject.toml` — current NumPy, pytest, Ruff, Black, and strict-mypy configuration; no dependency change is expected.
10. `015-build-canonical-transformer-parameter-layouts-and-initialization.md` — immediate acceptance and scope authority.
11. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` — fixed architecture, canonical order, mixed precision, random ownership, and initialization rules.
12. `llm_works_file_structure.md` — TypeScript `BLOCK_KEYS`, weight structures, `xavierInit()`, `initBlockWeights()`, `initWeights()`, and `weight-layout.ts` evidence.

## Step 1 — Establish independent layout and initialization evidence at the public Transformer seam

**Files and symbols:**

- `tests/fixtures/transformer_layout_initialization_reference.json` — new independent fixed fixture.
- `tests/test_transformer.py` — new layout, view, initialization, and isolation tests through the public Transformer boundary.
- `llm_works_file_structure.md` — TypeScript behavior used to capture expected values; never imported by Python tests.

**Purpose:**

Create failure-first acceptance evidence before production implementation. This protects exact flat ordering, offset arithmetic, Xavier draw order, immediate float32 materialization, deterministic fills, and run isolation from being inferred by the same code under test.

**Actions:**

- Add provenance metadata stating that the fixture was captured from the TypeScript Reference Implementation or an independent scalar reference that imports no `how_llms_work.ml.transformer` production operation.
- Record the fixed architecture values:
  - Vocabulary size `392`;
  - context length `32`;
  - embedding dimension `32`;
  - head count `2`;
  - head dimension `16`;
  - feed-forward dimension `128`;
  - supported layer counts `1..6`;
  - initialization seed `42`.
- Record the exact top-level key order and exact sixteen-key block order.
- Store complete layout records for representative one-, two-, and six-layer configurations. Each record must include semantic key, block index or explicit top-level identity, float offset, byte offset, length, shape, and canonical total float count.
- Store summary totals for every layer count from one through six, including record count, total floats, total bytes, random draw count, and final generator state.
- Include independent landmarks that catch offset drift, including:
  - `tokEmb` at float offset `0`, length `12,544`, shape `(392, 32)`;
  - `posEmb` at float offset `12,544`, length `1,024`, shape `(32, 32)`;
  - block zero beginning at float offset `13,568`;
  - one-layer final normalization beginning at `26,272`;
  - two-layer final normalization beginning at `38,976`;
  - six-layer final normalization beginning at `89,792`;
  - one-layer `headB` beginning at `38,880` and ending at `39,272`;
  - two-layer `headB` beginning at `51,584` and ending at `51,976`;
  - six-layer `headB` beginning at `102,400` and ending at `102,792`.
- Store selected initialized coordinates across every random-array family and multiple traversal boundaries:
  - first and last coordinate of block-zero `wQ`;
  - transition from `wQ` to `wK`;
  - at least one coordinate from `wV`, `wO`, `ff1W`, and `ff2W`;
  - first random coordinate in a later block;
  - first and last coordinate of `tokEmb`;
  - first and last coordinate of `posEmb`;
  - first and last coordinate of `headW`.
- Store selected float32 coordinates as exact 32-bit representations or another unambiguous exact form rather than relying only on JSON decimal rendering.
- Store deterministic fill evidence for every bias, beta, and gamma family.
- Store one documented byte-level checksum per representative complete flat array. Prefer a stable test-only checksum such as SHA-256 over canonical C-order little-endian float32 bytes; production code does not need a checksum API.
- Add public-boundary tests expected to fail before implementation for:
  - exact complete representative layouts;
  - all-depth structural invariants;
  - public total-count results;
  - exact selected initialization coordinates;
  - exact draw counts and final generator states;
  - deterministic fills;
  - complete checksums;
  - finiteness and non-aliasing.
- Keep tests independent of private record class names, dictionary implementation, local loops, and helper decomposition.

**Guardrails:**

- Do not generate expected fixture values by calling the new Python layout builder, view builder, or initializer.
- Do not modify `transformer_preprocessing_reference.json` or existing deterministic utility fixtures.
- Do not assert a private class identity, private helper name, local iteration variable, or exact internal container implementation.
- Do not add production checksum, serialization, shared-memory, worker, route, or persistence behavior merely to make tests convenient.

**Expected result:**

- The repository contains one reviewable independent fixture and focused acceptance tests that precisely describe Ticket 015 before production code is added.
- The new tests fail only because layout, view, and initialization behavior is missing, while all existing preprocessing tests remain unchanged.

**Verification:**

- Run `poetry run pytest tests/test_transformer.py -k "layout or parameter or view or initialization or weight" -q` and confirm the new acceptance tests fail for missing public behavior rather than fixture or import errors.
- Inspect the fixture-generation notes and confirm they import no production Transformer operation.

## Step 2 — Add the single canonical Transformer parameter layout and total-count boundary

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — existing `__all__`; new fixed architecture constants; new stable public immutable layout record/collection; new public layout and parameter-count boundary.
- `tests/test_transformer.py` — exact order, shape, offset, total, validation, and public-symbol tests.

**Purpose:**

Create one authoritative interpretation of Transformer parameter storage that later parent, worker, optimizer, conversion, and persistence code can reuse without maintaining parallel offset tables.

**Actions:**

- Add fixed architecture constants for context length `32`, embedding dimension `32`, head count `2`, head dimension `16`, and feed-forward dimension `128` in the Transformer module.
- Derive the canonical Vocabulary size from the immutable Transformer preprocessing snapshot. Do not make Vocabulary size or any fixed model dimension request-configurable in this ticket.
- Define one stable immutable public layout representation whose observable boundary exposes:
  - semantic parameter key;
  - block index for block-owned arrays and an explicit top-level identity for non-block arrays;
  - float offset;
  - byte offset or enough data to derive it exactly as `float_offset × 4`;
  - element length;
  - exact shape;
  - canonical total float count;
  - canonical total byte count through the layout boundary.
- Keep private class and container decomposition an implementation choice, but make semantic iteration and lookup stable enough for later parent and worker use.
- Validate `num_layers` as an actual Python integer from `1` through `6`; reject `bool`, floats, strings, zero, and values above six.
- Emit records in exactly this flat order:
  1. `tokEmb`;
  2. `posEmb`;
  3. for each block in ascending index: `ln1Gamma`, `ln1Beta`, `wQ`, `bQ`, `wK`, `bK`, `wV`, `bV`, `wO`, `bO`, `ln2Gamma`, `ln2Beta`, `ff1W`, `ff1B`, `ff2W`, `ff2B`;
  4. `lnFGamma`;
  5. `lnFBeta`;
  6. `headW`;
  7. `headB`.
- Use exact shapes:
  - `tokEmb`: `(V, D)`;
  - `posEmb`: `(C, D)`;
  - gamma, beta, and attention biases: `(D,)`;
  - `wQ`, `wK`, `wV`, `wO`: `(D, D)`;
  - `ff1W`: `(D, F)`;
  - `ff1B`: `(F,)`;
  - `ff2W`: `(F, D)`;
  - `ff2B`: `(D,)`;
  - `lnFGamma`, `lnFBeta`: `(D,)`;
  - `headW`: `(D, V)`;
  - `headB`: `(V,)`.
- Calculate each length from the shape rather than maintaining an unrelated manual length table.
- Advance one float offset monotonically from zero. Build the canonical total from the final end offset rather than maintaining a second authoritative total formula.
- Ensure each returned layout is immutable or otherwise safe from caller mutation changing future layouts.
- Ensure repeated calls for the same layer count return equivalent layout values. Object caching is not required by this ticket and must not create mutable shared state.
- Extend `transformer.py.__all__` with only the deliberately stable public layout and initialization boundary while preserving every current preprocessing export.
- Add exact tests for:
  - representative one-, two-, and six-layer fixtures;
  - all layer counts `1..6`;
  - record count `16 × layers + 6`;
  - first offset zero;
  - every `current.offset + current.length == next.offset`;
  - `product(shape) == length`;
  - block indices ascending and complete;
  - no duplicate semantic `(block, key)` identity;
  - final end equals total float count;
  - total byte count equals total floats multiplied by four;
  - public parameter-count output comes from the same canonical layout;
  - invalid layer-count rejection.

**Guardrails:**

- Do not create separate parent, worker, optimizer, or persistence offset tables.
- Do not make context length, embedding dimension, heads, head dimension, feed-forward dimension, Vocabulary, or seed configurable.
- Do not add gradient layouts, Adam buffers, model serialization, worker records, shared-memory names, or route configuration.
- Do not copy TypeScript's mutable array objects or global random state.
- Do not remove or alter completed preprocessing behavior.

**Expected result:**

- One stable public builder is the sole source for parameter order, semantic ownership, shapes, offsets, and total counts for every supported depth.
- Exact fixture and all-depth structural tests pass.

**Verification:**

- Run `poetry run pytest tests/test_transformer.py -k "layout or parameter_count or layer_count" -q`.
- Confirm the one-, two-, and six-layer complete records match the independent fixture exactly and all six depths satisfy gap-free structural invariants.

## Step 3 — Build strict exact-range NumPy views over caller-owned flat storage

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — new public flat-storage view-construction boundary and canonical layout consistency validation.
- `tests/test_transformer.py` — exact view shape, offset, aliasing, capacity, and rejection tests.

**Purpose:**

Allow later parent, spawned worker, optimizer, and model-conversion code to interpret one flat `float32` allocation identically while preventing silent NumPy coercion, shape drift, or access outside the canonical logical range.

**Actions:**

- Define the narrow backing-storage contract as one actual one-dimensional NumPy `ndarray` with dtype exactly `np.float32` and C-contiguous memory.
- Do not coerce Python lists, `float64`, integer arrays, multidimensional arrays, or non-contiguous slices into accepted storage.
- Accept capacity equal to or greater than the canonical total float count.
- Reject storage whose logical element capacity or byte capacity is smaller than the canonical requirement.
- Validate the supplied layout before creating any views:
  - expected record count;
  - exact semantic order;
  - exact shapes and lengths;
  - first offset zero;
  - contiguous boundaries;
  - no overlap or gaps;
  - every region end at or before the canonical total;
  - consistent total count on every observable record/layout value;
  - final region end exactly equal to the canonical total;
  - requested layer count and block ownership consistent with the canonical builder.
- Construct each NumPy view from the original backing buffer with:
  - dtype `np.float32`;
  - exact canonical shape;
  - byte offset equal to `float_offset × np.dtype(np.float32).itemsize`;
  - C order.
- Return a stable semantic view collection addressable by top-level key and block index without requiring tests or later consumers to depend on a particular dictionary implementation.
- Preserve the backing storage's writeability state. The general view builder must support later read-only worker weight views; the initializer must separately require writable storage before filling.
- Ensure every returned array:
  - has dtype exactly `float32`;
  - has the exact canonical shape;
  - is C-contiguous;
  - shares memory with the caller-owned flat storage;
  - begins at the exact expected byte address;
  - remains entirely within the canonical logical range.
- Add success tests using exact-size and oversized storage.
- For oversized storage, fill the tail with a sentinel and prove view writes do not alter any element at or after `total_float_count`.
- Add mutation-direction tests proving:
  - writing a semantic view changes the correct flat storage coordinates;
  - writing a flat coordinate changes the corresponding semantic view;
  - adjacent views do not overlap.
- Add rejection tests for:
  - non-NumPy input;
  - wrong dtype;
  - rank two or other wrong rank;
  - non-C-contiguous slice;
  - undersized storage;
  - forged offset gap;
  - forged overlap;
  - wrong shape/length product;
  - wrong semantic order;
  - incorrect block index;
  - inconsistent total count;
  - region outside canonical range.
- Ensure validation finishes before returning a partial view collection.

**Guardrails:**

- Do not allocate shared memory or import `multiprocessing.shared_memory` in this ticket.
- Do not copy storage to make invalid inputs acceptable.
- Do not flatten or reshape an incompatible input silently.
- Do not extend any view into valid-but-extra trailing capacity.
- Do not make worker weights read-only here by policy; preserve caller-owned writeability and leave worker ownership enforcement to the worker ticket.
- Do not test private validation-helper names or a particular mapping class.

**Expected result:**

- Exact-size and oversized one-dimensional C-contiguous `float32` storage produce semantic C-order views with byte-accurate offsets and shared memory.
- Every invalid storage or inconsistent-layout case fails before partial output can escape.

**Verification:**

- Run `poetry run pytest tests/test_transformer.py -k "view or storage or capacity or offset" -q`.
- Confirm exact-size and oversized cases pass and sentinel tail bytes remain unchanged.

## Step 4 — Implement deterministic fresh Weight Initialization through the canonical layout and shared Mulberry32 stream

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — new public fresh Weight Initialization boundary, Xavier scalar fill behavior, deterministic zero/one fills, and complete finite-state validation.
- `src/how_llms_work/ml/math_utils.py` — existing `Mulberry32` imported and reused unchanged.
- `tests/test_transformer.py` — selected-coordinate, draw-order, final-state, deterministic-fill, checksum, and finiteness tests.
- `tests/fixtures/transformer_layout_initialization_reference.json` — independent exact initialization evidence.

**Purpose:**

Create reproducible fresh model weights without hidden global state, NumPy-native randomness, vectorized draw reordering, saved-model loading, or uninitialized coordinates.

**Actions:**

- Make the numerical core accept one caller-owned `Mulberry32` instance plus one canonical layout, or an equivalently explicit public boundary that preserves caller-owned stream state and exposes final `state` and `draw_count` through the same generator.
- Allocate a new one-dimensional C-contiguous `np.float32` storage array of exactly the canonical total size for every initialization call.
- Build all semantic arrays through the canonical view builder; do not reconstruct offsets or shapes inside initialization.
- Initialize all deterministic regions without consuming random values:
  - every bias array to exact `0.0`;
  - every Layer Normalization beta array to exact `0.0`;
  - every Layer Normalization gamma array to exact `1.0`.
- Fill random matrices in this exact traversal order, independent of their flat storage order:
  1. blocks in ascending layer index;
  2. within each block: `wQ`, `wK`, `wV`, `wO`, `ff1W`, `ff2W`;
  3. after all blocks: `tokEmb`, `posEmb`, `headW`.
- Use exact fan-in and fan-out pairs:
  - `wQ`, `wK`, `wV`, `wO`: `(D, D)`;
  - `ff1W`: `(D, F)`;
  - `ff2W`: `(F, D)`;
  - `tokEmb`: `(V, D)`;
  - `posEmb`: `(C, D)`;
  - `headW`: `(D, V)`.
- For each random array, calculate one scalar Xavier limit:

  ```text
  limit = sqrt(6 / (fan_in + fan_out))
  ```

- Traverse every coordinate explicitly in C order. For each coordinate:
  1. consume exactly one `generator.random()` value;
  2. calculate `(random × 2 - 1) × limit` in Python float;
  3. convert and store that completed coordinate immediately as `np.float32`;
  4. only then request the next random value.
- Do not use `numpy.random`, list comprehensions that obscure materialization order, bulk vectorized random filling, or delayed whole-array casting.
- Check the complete flat storage for dtype, C-contiguity, exact size, and finiteness before returning successful initialized weights.
- Return the fresh flat storage and stable semantic views or an equivalent public initialized-weight value that preserves both the canonical flat boundary and semantic access without copying.
- Compare selected coordinates exactly against independent float32 bit-pattern fixtures.
- Assert exact draw counts and final generator states for all six depths.
- Assert the representative one-, two-, and six-layer complete byte checksums.
- Assert every bias and beta coordinate is exact zero and every gamma coordinate is exact one.
- Assert deterministic fills do not change generator state or draw count.
- Assert no coordinate remains uninitialized and the complete flat array is finite.

**Guardrails:**

- Do not create or use a module-global generator.
- Do not instantiate a second generator inside the random-fill traversal.
- Do not consume random values for biases, beta arrays, gamma arrays, storage allocation, layout construction, view construction, validation, or checksums.
- Do not replace the caller's generator or reset its state.
- Do not use saved weights, `.data/`, filesystem reads, model caches, resume logic, or checkpoint logic.
- Do not add forward passes, gradients, Adam state, generation, workers, routes, or persistence.

**Expected result:**

- Identical layer count and seed produce exact equivalent initialized values and generator end state.
- Every initialized coordinate follows the approved TypeScript-compatible random traversal and immediate float32 materialization rule.
- Deterministic fills are exact and consume zero draws.
- The returned model is completely finite and ready for later forward or shared-memory work.

**Verification:**

- Run `poetry run pytest tests/test_transformer.py -k "initialization or xavier or draw or checksum or finite" -q`.
- Confirm exact selected coordinate representations, complete checksums, draw counts, final states, and deterministic fills match the independent fixture.

## Step 5 — Prove run isolation, depth isolation, public contract stability, and no saved-model influence

**Files and symbols:**

- `tests/test_transformer.py` — same-seed non-aliasing, different-depth behavior, mutation isolation, public exports, and no-load acceptance tests.
- `src/how_llms_work/ml/transformer.py` — only the smallest corrections revealed by the focused tests.

**Purpose:**

Complete the remaining acceptance criteria without expanding into later Transformer tickets or coupling tests to private implementation details.

**Actions:**

- Create two independent initializations with separate `Mulberry32(42)` instances and the same layer count.
- Assert:
  - equivalent canonical layouts;
  - exact equal flat values;
  - exact equal selected semantic views;
  - identical draw counts and final states;
  - different flat storage objects;
  - `np.shares_memory()` is false across runs;
  - mutating one run does not change the other.
- Initialize at least one one-layer and one six-layer run in the same test process and prove each uses its own exact layout, total size, views, generator state, and storage.
- Re-run initialization sequentially and through `ThreadPoolExecutor` tasks where each task owns its own generator and layout; assert exact results and no shared mutable numerical state.
- Assert the public `transformer.py.__all__` retains all existing preprocessing exports and adds only the intended layout/view/count/initialization boundary and fixed architecture constants.
- Assert callers can obtain the total parameter count without creating weights and that the returned count agrees with the layout's final end.
- Confirm initialization has no path, file, cache, Saved Transformer Model, resume, or checkpoint input. Prefer public signature and behavior checks plus final scope inspection rather than private source-string tests.
- Keep any filesystem monkeypatch test narrow and only if implementation accidentally introduces a file dependency; the preferred implementation has no filesystem call to patch.
- Assert all existing preprocessing, shard, utility, and matrix tests continue to pass unchanged.

**Guardrails:**

- Do not require object identity or caching for repeated equivalent layouts.
- Do not assert a particular private dataclass, tuple, dictionary, or lookup implementation.
- Do not test shared-memory names, worker attachment, gradient ownership, optimizer traversal, HTTP configuration parsing, or persistence paths.
- Do not add production concurrency locks; independent local objects are sufficient for this ticket.
- Do not alter current preprocessing fixture values or completed BPE, Word2Vec, XOR, route, or persistence behavior.

**Expected result:**

- Weight Initialization is deterministic but not aliased.
- Layer-depth configurations remain isolated and exact.
- The public Transformer seam is complete for later numerical and worker tickets while remaining free of filesystem and process infrastructure.

**Verification:**

- Run `poetry run pytest tests/test_transformer.py -k "isolation or alias or concurrent or public or saved" -q`.
- Run `poetry run pytest tests/test_math_utils.py tests/test_matrix.py tests/test_transformer.py -q`.

## Step 6 — Finalize formatting, typing, regression verification, and scope-only diff review

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — complete Ticket 015 production boundary.
- `tests/test_transformer.py` — complete focused acceptance suite.
- `tests/fixtures/transformer_layout_initialization_reference.json` — complete independent evidence.
- Repository status and diff — scope validation only.

**Purpose:**

Prove the implementation is complete, typed, formatted, regression-safe, and restricted to Ticket 015 before handoff or commit.

**Actions:**

- Format only changed Python files with Black if needed.
- Run focused Transformer, utility, and matrix tests before the full suite.
- Run the complete pytest suite exactly once after focused tests are green.
- Run Ruff and strict mypy using the repository commands.
- Run Black in check mode for the changed Python files.
- Inspect `git diff --check`, `git status --short`, and the final diff.
- Confirm no dependency, lockfile, route, schema, worker, persistence, frontend, `.data`, specification, context, ADR, or existing fixture content changed.
- Confirm the fixture contains independent provenance and no generated runtime path, timestamp, shared-memory name, or machine-specific data.
- Record actual command results honestly; do not claim success for any command not executed.

**Guardrails:**

- Do not broaden formatting to unrelated files.
- Do not accept snapshot updates that change existing preprocessing, utility, or matrix fixtures.
- Do not commit generated `.data` files, cache files, temporary files, or fixture-generation scripts unless a small reviewed independent script is deliberately retained by repository convention.
- Do not create the implementation commit until every required command passes and final scope is correct.

**Expected result:**

- Ticket 015 is ready for implementation review with only the canonical Transformer layout/initialization source, focused tests, and one independent fixture changed.

**Verification:**

- Execute every command in the focused and full verification sections below and inspect the final diff manually.

## Focused verification plan

Run from the backend project root:

```powershell
poetry run pytest tests/test_transformer.py -q
poetry run pytest tests/test_math_utils.py tests/test_matrix.py tests/test_transformer.py -q
poetry run black --check src/how_llms_work/ml/transformer.py tests/test_transformer.py
poetry run ruff check src/how_llms_work/ml/transformer.py tests/test_transformer.py
poetry run mypy src/how_llms_work/ml/transformer.py
```

Expected result:

- Existing preprocessing tests remain green.
- Exact representative layouts, all-depth invariants, strict view validation, Xavier coordinates, deterministic fills, checksums, draw counts, final states, finiteness, and run isolation all pass.
- Shared utility and matrix regressions remain green.
- Changed Python files pass Black, Ruff, and strict mypy checks.

## Full verification plan

Run from the backend project root:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
poetry run black --check src/how_llms_work/ml/transformer.py tests/test_transformer.py
git diff --check
git status --short
```

Expected result:

- All tests pass.
- Ruff reports no errors.
- mypy reports `Success: no issues found`.
- Changed Python files require no Black formatting changes.
- `git diff --check` reports no whitespace errors.
- Repository status contains only the expected Ticket 015 files and any pre-existing user changes identified before editing.

## Manual acceptance checklist

- [ ] Ticket 015 is the only implemented work item.
- [ ] Ticket 012's existing `Mulberry32` is reused directly; no copied or global generator exists.
- [ ] Layer counts `1..6` are supported and invalid values, including Booleans, are rejected.
- [ ] Fixed architecture values remain `C=32`, `D=32`, `heads=2`, `headDim=16`, and `F=128`.
- [ ] Vocabulary size comes from the fixed immutable preprocessing snapshot and remains `392` for the current fixture.
- [ ] The flat order is exactly `tokEmb`, `posEmb`, all sixteen arrays for each block in ascending order, `lnFGamma`, `lnFBeta`, `headW`, `headB`.
- [ ] Every record exposes stable semantic identity, block identity where applicable, offset, length, shape, and canonical total information.
- [ ] Every region is contiguous, non-overlapping, gap-free, and inside the exact canonical total.
- [ ] Total parameter count is derived from the canonical layout rather than a second authoritative offset table.
- [ ] One-, two-, and six-layer complete fixtures match exactly.
- [ ] All six depths satisfy complete structural invariants.
- [ ] Backing storage must be an actual one-dimensional C-contiguous NumPy `float32` array.
- [ ] Exact-size and oversized capacity are accepted.
- [ ] Wrong dtype, wrong rank, non-contiguous, undersized, inconsistent, and out-of-range cases are rejected before partial output escapes.
- [ ] Every semantic view has exact shape, dtype, C order, byte offset, and shared-memory relationship to the flat array.
- [ ] Extra trailing storage remains outside every canonical view and its sentinel data is unchanged.
- [ ] Every initialization call allocates fresh non-aliased storage.
- [ ] One caller-owned `Mulberry32` stream is used for all random weights.
- [ ] Random traversal is block `wQ → wK → wV → wO → ff1W → ff2W`, then `tokEmb → posEmb → headW`.
- [ ] Each coordinate consumes one draw and is stored immediately as `float32` before the next draw.
- [ ] Xavier uses `sqrt(6 / (fan_in + fan_out))` with the correct fan pairs.
- [ ] Biases and beta arrays are exact zero.
- [ ] Layer Normalization gamma arrays are exact one.
- [ ] Deterministic fills consume no random draws.
- [ ] Selected coordinate bit patterns, draw counts, final generator states, and complete flat-array checksums match independent evidence.
- [ ] All initialized weights are finite.
- [ ] Same configuration and seed produce equal values but no memory aliasing.
- [ ] Different layer counts produce their own exact sizes and layouts.
- [ ] Initialization performs no Saved Transformer Model read, load, cache, resume, or skip behavior.
- [ ] Existing Transformer preprocessing, BPE, Word2Vec, XOR, route, and persistence behavior remains unchanged.
- [ ] No worker, shared-memory allocation, forward/backward, Adam, route, SSE, persistence, or frontend work was added.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer.py
tests/test_transformer.py
tests/fixtures/transformer_layout_initialization_reference.json
```

Conditionally changed:

```text
None expected.
```

A separate test module such as `tests/test_transformer_layout.py` is acceptable only if `tests/test_transformer.py` would become materially harder to review; do not duplicate the same public acceptance tests across both files.

No package or lockfile change is expected.

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
src/how_llms_work/routes/
tests/test_math_utils.py
tests/test_matrix.py
tests/fixtures/math_utils_reference.json
tests/fixtures/matrix_reference.json
tests/fixtures/transformer_preprocessing_reference.json
tests/fixtures/word2vec_preprocessing_reference.json
tests/fixtures/word2vec_training_reference.json
tests/fixtures/word2vec_results_reference.json
.data/
README.md
pyproject.toml
poetry.lock
poetry.toml
frontend/
SPEC.md
CONTEXT.md
0002-stabilize-python-transformer-training-and-process-lifecycle.md
015-build-canonical-transformer-parameter-layouts-and-initialization.md
```

## Risk notes and safeguards

1. **Risk:** Parent code and future workers interpret the same flat array with different offset tables.
   - **Safeguard:** Build every record, total, view, and later traversal from one canonical public layout; prohibit duplicate offset tables.

2. **Risk:** The initializer follows flat layout order instead of the approved TypeScript random-consumption order.
   - **Safeguard:** Keep layout traversal and random traversal explicitly separate and protect transitions with selected exact coordinate fixtures, draw counts, final states, and full checksums.

3. **Risk:** NumPy vectorization changes the number or order of random draws.
   - **Safeguard:** Use one explicit coordinate loop and one `Mulberry32.random()` call per coordinate; forbid `numpy.random` and bulk random fills.

4. **Risk:** Values are accumulated as Python floats and cast only after a whole matrix is complete, changing later compatibility evidence.
   - **Safeguard:** Cast and store each completed Xavier coordinate as `np.float32` before consuming the next draw.

5. **Risk:** Bias, beta, or gamma fills accidentally consume random values.
   - **Safeguard:** Separate deterministic fills from the random traversal and assert unchanged state/draw count around those fills.

6. **Risk:** An `np.empty()` allocation leaves forgotten coordinates uninitialized.
   - **Safeguard:** Initialize every deterministic region explicitly, initialize every random region explicitly, and require whole-array finiteness plus complete checksum equality before return.

7. **Risk:** A parallel length or total formula drifts from actual offsets.
   - **Safeguard:** Derive lengths from shapes and total count from the final canonical end offset; tests compare the public count boundary to the layout.

8. **Risk:** A forged or stale layout maps valid storage incorrectly.
   - **Safeguard:** Validate semantic order, keys, blocks, shapes, lengths, offsets, totals, and canonical range before constructing any views.

9. **Risk:** NumPy silently accepts wrong dtype, wrong rank, or non-contiguous storage through coercion or reshape.
   - **Safeguard:** Require an actual one-dimensional C-contiguous `np.ndarray` with dtype exactly `np.float32`; never coerce invalid storage.

10. **Risk:** Extra capacity causes the final view to include or overwrite the storage tail.
    - **Safeguard:** Use canonical lengths and byte offsets for every view and protect the tail with sentinel tests.

11. **Risk:** View tests verify values but miss aliasing or wrong byte placement.
    - **Safeguard:** Check `np.shares_memory()`, exact data-pointer differences, bidirectional mutation, adjacent non-overlap, and byte-offset landmarks.

12. **Risk:** Returning mutable layout metadata lets one caller corrupt later calls.
    - **Safeguard:** Use immutable values or defensive reconstruction and add mutation-attempt/equivalence tests at the public seam.

13. **Risk:** Same-seed tests pass because both results alias the same storage.
    - **Safeguard:** Assert exact equality and `np.shares_memory() is False`, then mutate one run and verify the other remains unchanged.

14. **Risk:** The fixture reproduces the same bug as production code.
    - **Safeguard:** Record independent provenance, prohibit production imports during expected-value generation, and preserve selected human-reviewable landmarks alongside checksums.

15. **Risk:** Byte checksums vary by endian representation or test method.
    - **Safeguard:** Document one canonical test-only byte representation, preferably C-order little-endian float32 bytes, before recording expected hashes.

16. **Risk:** Layout construction triggers filesystem or saved-model behavior to obtain configuration.
    - **Safeguard:** Derive only from fixed constants and the immutable preprocessing snapshot; do not import routes or persistence modules.

17. **Risk:** The ticket expands into gradient layout, Adam, workers, shared memory, forward/backward math, or routing because those are future consumers.
    - **Safeguard:** Enforce the expected-file list and leave `matrix.py`, `transformer_worker.py`, routes, schemas, `main.py`, `.data`, and frontend files unchanged.

18. **Risk:** New public symbols remove or shadow completed preprocessing exports.
    - **Safeguard:** Add an exact public-symbol regression that preserves every current `transformer.py.__all__` entry.

19. **Risk:** A broad formatter or refactor changes completed preprocessing code.
    - **Safeguard:** Format only changed Python files, inspect `git diff --check`, and reject unrelated churn before commit.

20. **Risk:** User-reported baseline is mistaken for current-session verification.
    - **Safeguard:** Re-run the baseline in `implement-prompt` before editing and report actual output honestly.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Build canonical Transformer layouts and initialization
```

Commit body should mention:

- one immutable canonical parameter order for every supported one-through-six-layer Transformer;
- exact semantic records, float/byte offsets, shapes, total counts, gap-free invariants, and strict layer validation;
- exact C-order `float32` views over caller-owned flat storage with capacity and consistency validation;
- request-owned `Mulberry32` Xavier initialization in the approved random traversal with immediate float32 materialization;
- zero biases/betas, one-valued Layer Normalization gammas, exact draw counts, final states, selected coordinates, checksums, finiteness, and non-aliased run isolation;
- independent fixture provenance;
- no forward/backward math, Adam, worker, shared-memory allocation, route, persistence, frontend, dependency, or lockfile changes;
- the exact focused, full pytest, Ruff, Black, mypy, and diff commands actually executed and their observed results.

Do not create the commit during `to-plan-prompt`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan015.md`;
- `015-build-canonical-transformer-parameter-layouts-and-initialization.md`;
- completed blocker `012-centralize-typescript-compatible-randomness-and-rounding.md` or the current `math_utils.py` and its tests as equivalent evidence;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `py_llm_pipeline_explorer_file_structure(39).md`;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the real repository again, establish its own baseline before editing, preserve user changes, implement only Ticket 015, create independent failure-first layout/initialization evidence, reuse the completed `Mulberry32` and preprocessing boundaries, run focused and full verification, report actual command results honestly, inspect final scope, and create the implementation commit only after all required checks pass.
