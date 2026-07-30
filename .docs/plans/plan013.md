---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "013"
source_work_item: 013-provide-strict-float32-matrix-primitives.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(35).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 013: Provide strict float32 matrix primitives

## Initial checklist

- Confirm Ticket 013 is the only selected work item and has no blocker.
- Treat `py_llm_pipeline_explorer_file_structure(35).md` as the current Python Backend source of truth, even where the older specification's current-code notes are stale.
- Use `llm_works_file_structure.md` only for Transformer matrix formulas, required operation inventory, and shape evidence; do not copy its mutable global random state into Python.
- Preserve the user-reported passing pytest, Ruff, and strict mypy baseline without describing it as verified in this planning session.
- Implement one stateless, strict NumPy `float32` boundary in `src/how_llms_work/ml/matrix.py` with explicit ranks, shapes, contiguity, finiteness, purity, overlap, and transactional-mutation rules.
- Add independently calculated small fixtures and focused tests at the approved public matrix seam.
- Finish with focused matrix tests, affected numerical regressions, the full pytest suite, Ruff, strict mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest direction to convert the selected TypeScript behavior to Python and treat the latest complete Python Backend export as current-code truth.
2. `013-provide-strict-float32-matrix-primitives.md` for the required behavior, acceptance criteria, approved test seam, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(35).md` for the current implementation, tests, dependencies, paths, typing style, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and ADR 0002 for the stable Phase 5 numerical contract, canonical terminology, mixed-precision rules, and stateless matrix ownership.
5. `llm_works_file_structure.md`, especially TypeScript `src/routes/train-transformer/matrix.ts` and its Transformer call sites, as formula and shape evidence only.
6. Older source exports, plans, snippets, and specification statements about current file contents are non-authoritative when they conflict with export `(35)`.

## Work-item summary

Ticket 013 creates the reusable numerical boundary that later Transformer tickets will rely on. The Python Backend currently has an empty `src/how_llms_work/ml/matrix.py`; this ticket fills that file with only the matrix operations already required by the approved decoder-only Transformer formulas.

The boundary must be deliberately stricter than ordinary NumPy behavior. Public operations must reject unsupported ranks, incompatible shapes, wrong dtypes, non-C-contiguous arrays, accidental broadcasting, NaN, infinity, and invalid causal-mask rows rather than allowing NumPy to reinterpret them silently. Pure operations must never mutate or alias an input and must return independent C-contiguous `float32` arrays.

Approved reductions use wider scratch precision: matrix multiplication and column sums accumulate in `float64`, then materialize the completed result once as `float32`. The single approved broadcasting operation accepts one exact row-bias shape. The explicitly named in-place addition is the only operation allowed to mutate caller-owned storage; it must reject overlapping memory, calculate a complete candidate in separate `float64` scratch state, validate the complete `float32` candidate, and commit only after every value is valid. Any failure must leave the destination byte-for-byte unchanged.

The stable row softmax must accept finite scores plus intentional negative-infinity causal-mask entries, reject NaN, positive infinity, and rows with no finite selectable value, leave the input untouched, return a separate finite probability matrix, keep masked positions exactly zero, and normalize each valid row to one within a named tight tolerance.

This ticket does not implement Transformer layers, forward or backward propagation, parameter layouts, preprocessing, workers, shared memory, training, Adam, generation, HTTP routing, SSE, persistence, request validation, or frontend changes.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Reported result:** All pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, mypy, formatter, browser, or runtime command was executed while creating this plan.
- **Implementation rule:** The implementation run must establish or reconfirm its own baseline before editing and report only command results it actually observes.

## Current code observations from the latest source

- `src/how_llms_work/ml/matrix.py` exists but is empty.
- `src/how_llms_work/ml/transformer.py`, `src/how_llms_work/ml/transformer_worker.py`, and `src/how_llms_work/routes/train_transformer.py` remain empty and are outside this ticket.
- `src/how_llms_work/ml/math_utils.py` is no longer empty. Ticket 012 has already centralized request-owned `Mulberry32` and TypeScript-compatible rounding there.
- The Phase 5 `SPEC.md` contains an older current-code observation saying `math_utils.py` is empty. Export `(35)` supersedes that observation; this ticket must not remove, duplicate, or relocate the completed Ticket 012 utilities.
- The TypeScript reference places `mulberry32`, mutable `rand`, and `resetRand` in `matrix.ts`. ADR 0002 and the completed Ticket 012 intentionally replace that ownership. Python `matrix.py` must remain stateless and must not port those globals.
- The TypeScript reference confirms row-major Transformer matrix formulas such as ordinary matrix multiplication and output bias addition. Its flat `Float32Array` representation is behavior evidence, not a requirement to reproduce the same function signatures or global state line for line.
- `pyproject.toml` already declares Python 3.12+, NumPy 2.5+, pytest, Ruff, Black, and strict mypy. No dependency or lockfile change is expected.
- Existing numerical tests provide repository prior art:
  - `tests/test_neural_net.py` uses named `float32` tolerances, exact dtype assertions, `numpy.testing.assert_allclose()`, and independently calculated scalar references.
  - Word2Vec tests use fixed JSON fixtures, exact structural assertions, tolerance-aware hidden numerical checks, mutation isolation, and `np.shares_memory()`.
  - `tests/test_math_utils.py` records provenance proving expected fixture values were captured without production imports.
- There is no current `tests/test_matrix.py` and no matrix reference fixture.
- The current empty matrix module exposes no compatibility burden from existing Python call sites, so the public seam can be defined cleanly from Ticket 013, ADR 0002, the specification, and the exact Transformer operation inventory.

## Acceptance criteria coverage

### Already satisfied and evidenced

- NumPy is already a declared production dependency.
- Python 3.12, pytest, Ruff, Black, and strict mypy are already configured.
- The destination module exists and is isolated from route and process infrastructure.
- The current tests already establish project conventions for independent fixtures, exact-versus-tolerance assertions, dtype checks, mutation checks, and memory-sharing checks.
- Randomness and rounding already have a separate canonical owner in `math_utils.py`, allowing `matrix.py` to remain stateless.

### Behavior present but evidence incomplete

- The TypeScript reference provides formulas and operation-use evidence, but it does not provide the stricter Python-owned validation, transactionality, overlap, and output-contiguity contract.
- Existing tests demonstrate suitable assertion techniques, but no dedicated matrix fixture or public matrix test seam exists.

### Partially implemented

- `matrix.py` exists only as an empty destination.
- Phase 5 source material identifies the required numerical behavior, but none of the matrix operations or their validation rules are implemented in Python.

### Not implemented

- A documented public matrix-array contract.
- Exact rank, shape, dtype, contiguity, and finiteness validation.
- Ordinary matrix multiplication with `float64` accumulation and one final `float32` materialization.
- Required left- or right-transposed multiplication variants, if confirmed by the approved Transformer call-site inventory.
- Column sums with `float64` accumulation and a stable output shape.
- Exact row-bias addition with no other broadcasting.
- Pure elementwise addition, subtraction, multiplication, scalar multiplication, transpose, row concatenation, and row slicing needed by approved Transformer formulas.
- Transactional non-overlapping in-place addition.
- Stable causal-mask-aware row softmax.
- Independent exact/tolerance fixtures and focused tests for every public primitive.
- Failure tests proving no partial destination mutation.

### Evidence limitations

- The final public symbol list must be confirmed against all exported TypeScript matrix operations and every `matrix.ts` call site in `transformer.ts`, `train.ts`, and `train-worker.ts` before implementation. The implementation must expose only operations actually required by Phase 5.
- The TypeScript reference uses flat arrays plus explicit dimensions, while the accepted Python contract uses shape-checked NumPy arrays. The implementation should preserve formulas and observable numerical intent, not copy flat-array signatures mechanically.
- Exact unrounded fixture values and the smallest useful per-operation tolerances are not present in the current Python export. They must be calculated independently or captured from the reference behavior; production helpers must not generate their own expected values.
- No baseline command or numerical experiment was run during planning.

## Files to inspect before editing

1. `src/how_llms_work/ml/matrix.py` — empty destination for the complete public boundary.
2. `src/how_llms_work/ml/math_utils.py` — completed Ticket 012 ownership boundary; inspect to avoid duplicate PRNG or rounding state, but do not modify.
3. `src/how_llms_work/ml/transformer.py` — empty future consumer; do not implement it in this ticket.
4. `src/how_llms_work/ml/transformer_worker.py` — empty future worker consumer; do not implement it in this ticket.
5. `tests/test_neural_net.py` — named float32 tolerances, independent scalar calculations, dtype checks, and `assert_allclose()` prior art.
6. `tests/test_word2vec_training.py` and `tests/test_word2vec_results.py` — fixture provenance, non-finite rejection, memory-isolation, and exact-versus-tolerance prior art.
7. `tests/test_math_utils.py` and `tests/fixtures/math_utils_reference.json` — independent-fixture provenance and public-seam test organization.
8. `pyproject.toml` — current dependency, Ruff, Black, pytest, and strict-mypy configuration.
9. `013-provide-strict-float32-matrix-primitives.md` — direct acceptance authority.
10. `SPEC.md`, `CONTEXT.md`, and ADR 0002 — mixed-precision, statelessness, masking, compatibility, and numerical-test decisions.
11. `llm_works_file_structure.md` — TypeScript `matrix.ts` exports and every Transformer call site needed to inventory formulas and required operations.

## Step 1 — Inventory the exact Phase 5 operation surface and freeze the public contract

**Files and symbols:**

- `llm_works_file_structure.md` — TypeScript `matrix.ts`, `transformer.ts`, `train.ts`, and `train-worker.ts` operation-use evidence.
- `src/how_llms_work/ml/matrix.py` — public type aliases, module documentation, exported operation names, and validation contract.
- `tests/test_matrix.py` — public symbol and signature-level contract tests.

**Purpose:**

Prevent both under-implementation and framework-like scope growth. The Python module should contain every matrix primitive later approved Transformer formulas require and nothing unrelated.

**Actions:**

- Enumerate every TypeScript matrix export and every direct matrix-helper use in the Transformer forward pass, backward pass, loss evaluation, generation, and gradient assembly.
- Separate operations into:
  - ordinary multiplication;
  - multiplication against a logically transposed operand;
  - pure same-shape elementwise operations;
  - exact one-row bias addition;
  - scalar operations;
  - transpose;
  - column reduction;
  - row concatenation and row slicing;
  - explicitly in-place addition;
  - stable row softmax.
- Exclude TypeScript `mulberry32`, mutable `rand`, and `resetRand`; their Python owner is already `math_utils.py`.
- Prefer clear public names that retain the reference formula's intent, including `stable_row_softmax()` and an explicitly `_in_place`-named mutating operation.
- Define one NumPy typing alias for rank-2 `float32` arrays and precise return types compatible with strict mypy.
- Document the common public contract:
  - inputs are NumPy arrays, not arbitrary sequences;
  - supported numerical arrays have dtype exactly `np.float32`;
  - required arrays have exact supported rank and C-contiguous layout;
  - pure outputs are newly allocated, independent, C-contiguous `float32` arrays;
  - ordinary operations require only finite values;
  - only `stable_row_softmax()` may accept intentional `-np.inf` mask entries;
  - no operation silently broadens shapes through NumPy broadcasting;
  - wrong Python type or dtype is a type-contract failure;
  - invalid rank, shape, contiguity, overlap, or mask is a value-contract failure;
  - non-finite arithmetic or materialization is a numerical failure.
- Define stable empty-dimension policy from the approved Transformer formulas rather than inheriting whatever NumPy happens to permit. Reject any matrix shape that would make a required reduction, normalization, or multiplication semantically empty unless a confirmed Phase 5 consumer requires it.
- Add a public-symbol test that prevents accidental export of random state, general tensor objects, optimizer behavior, or unrelated helpers.

**Guardrails:**

- Do not introduce a Matrix class, Tensor class, hidden gradient state, automatic differentiation, operator overloading, GPU support, sparse support, batching abstractions, or a general numerical framework.
- Do not add request, route, worker, persistence, or Transformer architecture behavior.
- Do not require private helper names or exact internal vectorization in tests.
- Do not loosen input handling to lists, Python nested sequences, integers, or arbitrary floating dtypes merely because `np.asarray()` could convert them.

**Expected result:**

- The operation inventory is complete for the approved Transformer formulas.
- The public seam has explicit, reviewable behavior before numerical implementation begins.
- The module remains stateless and dependency-light.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py -q -k "public or contract or dtype or rank or contiguous"
```

Expected result:

- Public names, supported input categories, and common rejection rules are fixed without testing private implementation details.

## Step 2 — Add independent fixtures and reusable output-contract assertions

**Files and symbols:**

- `tests/fixtures/matrix_reference.json` — new independently calculated fixture data and provenance.
- `tests/test_matrix.py` — fixture loading, reusable assertions, and failure-first tests.

**Purpose:**

Create evidence before implementation so the tests do not simply repeat production formulas or miss structural requirements such as aliasing and contiguity.

**Actions:**

- Add provenance metadata stating that expected values were hand-calculated, calculated by a small independent scalar reference, or captured from the supplied TypeScript formulas without importing `how_llms_work.ml.matrix`.
- Store small non-symmetric matrices whose values reveal transpose-order, row/column, and broadcasting mistakes.
- Include independent cases for every confirmed public primitive.
- Include at least one matrix-multiplication fixture deliberately selected so `float32` stepwise accumulation differs from `float64` accumulation followed by one `float32` cast.
- Include a column-sum fixture with cancellation or magnitude differences that detects accidental `float32` reduction.
- Include row-bias data whose compatible shape is exactly `(1, columns)` and nearby invalid shapes such as `(columns,)`, `(rows, 1)`, and `(rows, columns)`.
- Include softmax rows with:
  - ordinary finite scores;
  - large positive and negative finite scores that require max subtraction;
  - one valid causal prefix followed by `-np.inf` future positions;
  - a single selectable value;
  - all `-np.inf`;
  - NaN;
  - positive infinity.
- Define reusable test assertions for every successful pure operation:
  - exact expected shape;
  - dtype exactly `np.float32`;
  - C-contiguous output;
  - finite output;
  - no memory sharing with any input;
  - every input equal to a pre-call copy;
  - mutation of the returned array does not alter an input or a later call.
- Use exact equality for dtype, shape, contiguity, masked zeros, input bytes, and simple exactly representable outputs.
- Use named, explicit, per-boundary tolerances for nontrivial floating-point results. Select the smallest verified tolerances that pass representative supported-environment results and still expose formula or precision defects.

**Guardrails:**

- Do not generate fixture expectations by calling the production operation under test.
- Do not make tests depend on BLAS implementation identity, temporary-array identity, private helper decomposition, or exact elapsed time.
- Do not use one overly broad tolerance for every primitive.
- Do not overwrite existing fixtures.

**Expected result:**

- A focused red test seam defines both numerical and structural behavior.
- Every later operation can be tested consistently for purity, independence, dtype, and contiguity.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py -q -k "fixture or provenance or output_contract"
```

Expected result before implementation:

- Fixture/provenance checks pass.
- Operation tests fail only because public matrix behavior is not yet implemented.

## Step 3 — Implement shared validation and pure output materialization rules

**Files and symbols:**

- `src/how_llms_work/ml/matrix.py` — internal validation/materialization support and public operations that reuse it.
- `tests/test_matrix.py` — dtype, rank, shape, contiguity, non-finite, purity, and alias tests.

**Purpose:**

Create one consistent contract so individual primitives do not drift into different dtype, memory, or failure behavior.

**Actions:**

- Validate that every array argument is an actual `np.ndarray` with dtype exactly `np.float32`.
- Validate exact rank for each operation. The default matrix boundary is rank two; permit another rank only when a confirmed public operation explicitly requires it.
- Require C-contiguous inputs at the public boundary. Reject transposed, stepped, reversed, or otherwise non-C-contiguous views instead of silently copying and hiding a caller-layout defect.
- Validate dimensions and exact shape relationships before numerical work.
- For ordinary operations, reject NaN, positive infinity, and negative infinity before calculation.
- For `stable_row_softmax()`, use a separate validator that permits only finite values and intentional `-np.inf` entries.
- Materialize every pure result through one deliberate path that returns an independent C-contiguous `float32` array and validates the completed result.
- Treat a float64 scratch result that becomes non-finite when cast to `float32` as failure rather than returning infinity.
- Keep validation helpers private; test their effects only through public operations.

**Guardrails:**

- Do not use `np.asarray(..., dtype=np.float32)` at the public boundary to silently accept or convert unsupported caller values.
- Do not let an apparently pure transpose or slice return a NumPy view.
- Do not rely on `np.isfinite()` alone for softmax input because intentional `-np.inf` masks are valid only there.
- Do not mutate array flags, writeability, shape, strides, or contents of caller-owned inputs.

**Expected result:**

- All public operations share one strict and predictable validation model.
- Pure results cannot alias input buffers or leak a non-contiguous view.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py -q -k "dtype or rank or shape or contiguous or finite or pure or alias"
```

Expected result:

- Wrong dtypes, ranks, shapes, non-contiguous arrays, and invalid values fail at the public boundary.
- Successful pure operations leave inputs unchanged and return independent C-contiguous `float32` arrays.

## Step 4 — Implement multiplication and column-reduction primitives with explicit wider accumulation

**Files and symbols:**

- `src/how_llms_work/ml/matrix.py` — ordinary matrix multiplication, confirmed transposed multiplication variants, and column sums.
- `tests/test_matrix.py` — independent numerical, shape, precision, and output-contract tests.
- `tests/fixtures/matrix_reference.json` — multiplication and reduction fixtures.

**Purpose:**

Provide the high-risk numerical operations with explicit `float64` accumulation rather than relying on NumPy defaults for `float32` arrays.

**Actions:**

- Implement ordinary `(M, K) × (K, N) → (M, N)` multiplication.
- Implement only the logically transposed multiplication variants confirmed by the TypeScript Transformer formulas, while preserving exact shape validation and avoiding caller-visible transpose views.
- Promote operands or accumulation state explicitly to `float64` for the complete multiplication calculation.
- Materialize the completed matrix once as a new C-contiguous `float32` array.
- Validate the completed `float32` result before returning it.
- Implement column sums with an explicit `float64` accumulator.
- Preserve the exact output shape required by later Transformer formulas. If the confirmed bias-gradient consumer expects one row, return `(1, columns)` rather than a rank-one vector.
- Materialize the completed reduction once as `float32` and validate it.
- Add incompatible inner-dimension, wrong-rank, zero-dimension, non-finite, and overflow-materialization failures.
- Prove precision intent with fixtures that would not pass under accidental `float32` accumulation.

**Guardrails:**

- Do not rely on default `np.matmul()` or `np.sum()` accumulator dtype for `float32` inputs.
- Do not cast partial sums repeatedly to `float32`.
- Do not expose a view into scratch state.
- Do not test or depend on a particular BLAS library or loop/vectorization implementation.

**Expected result:**

- Every supported multiplication and column sum has exact shape behavior, explicit wider accumulation, one final `float32` materialization, and the common purity contract.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py -q -k "matmul or multiply or transposed or column_sum or accumulation"
```

Expected result:

- Independent fixtures pass.
- Precision-sensitive cases distinguish the approved mixed-precision contract from default `float32` reduction behavior.

## Step 5 — Implement pure elementwise, row-bias, scalar, transpose, concatenation, and slicing operations

**Files and symbols:**

- `src/how_llms_work/ml/matrix.py` — confirmed pure non-reduction primitives.
- `tests/test_matrix.py` — exact structural, numerical, rejection, purity, and aliasing tests.
- `tests/fixtures/matrix_reference.json` — small exact cases.

**Purpose:**

Complete the ordinary building blocks required by Transformer residual paths, gradient formulas, bias handling, and shape assembly without permitting accidental broadcasting or view leakage.

**Actions:**

- Implement pure same-shape elementwise addition and every other confirmed same-shape elementwise operation.
- Require exact shape equality for ordinary elementwise operations; reject all NumPy-compatible but unapproved broadcasting combinations.
- Implement approved row-bias addition only for matrix shape `(rows, columns)` and bias shape exactly `(1, columns)`.
- Reject rank-one bias vectors, column vectors, full matrices, empty rows, or any other broadcasting shape.
- Calculate row-bias output in independent scratch state and return a fresh C-contiguous `float32` result.
- Implement scalar multiplication or other confirmed scalar operations with finite scalar validation and completed-output finiteness validation.
- Implement transpose as a pure operation that returns a copied C-contiguous result, not `array.T` or another shared-memory view.
- Implement row concatenation only along the approved row axis and only when all non-concatenated dimensions match exactly.
- Implement row slicing with explicit integral bounds and return an independent C-contiguous copy.
- Reject Python negative-index convenience, out-of-range bounds, reversed ranges, or empty results unless an approved Transformer formula explicitly requires one of them.
- Add mutation-isolation tests for transpose, concatenation, and slice results because those are the operations most likely to accidentally return views.

**Guardrails:**

- Do not use unrestricted `np.add()` broadcasting.
- Do not expose a general `axis` parameter when Phase 5 requires only one axis.
- Do not accept arbitrary scalar arrays as substitutes for a scalar.
- Do not broaden slicing into a general indexing API.

**Expected result:**

- Every confirmed pure primitive follows the same exact shape, dtype, contiguity, finiteness, purity, and independence contract.
- Row bias is the only supported broadcasting behavior.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py -q -k "elementwise or bias or scalar or transpose or concatenate or slice or broadcast"
```

Expected result:

- Approved operations match independent fixtures.
- Every unapproved broadcast or view-producing shortcut is rejected or exposed by the tests.

## Step 6 — Implement transactional non-overlapping in-place addition

**Files and symbols:**

- `src/how_llms_work/ml/matrix.py` — explicitly named in-place addition operation.
- `tests/test_matrix.py` — successful commit, overlap, partial-overlap, overflow, exception, and byte-preservation tests.

**Purpose:**

Provide the one approved mutation operation without allowing partial writes, self-aliasing, or silent corruption.

**Actions:**

- Require an explicitly writable destination and a separate source with exact matching shape, rank, dtype, and C-contiguous layout.
- Reject any actual memory overlap between source and destination, including:
  - the same object;
  - two views of the same base with identical ranges;
  - partially overlapping slices;
  - differently shaped views that touch common bytes.
- Use an actual memory-overlap check rather than object identity or base-object equality.
- Copy or promote the complete destination and source into non-overlapping `float64` scratch state.
- Calculate the complete candidate without writing to the destination.
- Materialize the complete candidate once as an independent `float32` array.
- Validate candidate dtype, shape, contiguity, and every value before commit.
- Commit with one final destination assignment only after all checks pass.
- Return no independent result unless the selected public contract deliberately returns the destination; preserve one clear mutating convention.
- Add a successful test proving the source is unchanged and only the destination changes.
- Add a failure case using values whose finite `float64` sum overflows to `float32`; compare `destination.tobytes()` before and after the exception.
- Add parameterized failures before and after candidate calculation and prove byte-for-byte destination preservation in every case.

**Guardrails:**

- Do not mutate while iterating or validate one element at a time after writing it.
- Do not allow `destination += source`, `np.add(..., out=destination)`, or any equivalent operation before complete candidate validation.
- Do not weaken overlap detection to `destination is source`.
- Do not silently copy a read-only or non-contiguous destination and then pretend the caller's array was updated.

**Expected result:**

- Successful in-place addition commits one complete finite candidate.
- Every failure leaves the destination exactly unchanged.
- Any actual source/destination overlap is rejected before mutation.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py -q -k "in_place or transactional or overlap or partial or unchanged or overflow"
```

Expected result:

- Successful commits pass.
- Same-buffer and partial-view overlap tests fail safely.
- Overflow and validation failures preserve destination bytes exactly.

## Step 7 — Implement stable row softmax with explicit causal-mask semantics

**Files and symbols:**

- `src/how_llms_work/ml/matrix.py` — `stable_row_softmax()`.
- `tests/test_matrix.py` — finite-score, stability, mask, all-masked, NaN, positive-infinity, normalization, purity, and exact-zero tests.
- `tests/fixtures/matrix_reference.json` — independent probability fixtures.

**Purpose:**

Provide the attention normalization primitive with stable arithmetic and unambiguous intentional-mask handling.

**Actions:**

- Accept one rank-two C-contiguous `float32` score matrix.
- Permit each entry to be finite or exactly negative infinity.
- Reject NaN and positive infinity before calculation.
- Require every row to contain at least one finite selectable score.
- For each row:
  - locate the maximum among finite entries only;
  - subtract that finite maximum;
  - exponentiate selectable entries in wider scratch precision;
  - assign exactly zero contribution to every `-np.inf` entry;
  - accumulate the denominator in wider precision;
  - reject a non-positive or non-finite denominator;
  - normalize selectable entries;
  - explicitly restore masked positions to exact `0.0` before final materialization.
- Materialize the complete probability matrix once as independent C-contiguous `float32`.
- Validate that all returned values are finite and non-negative.
- Validate each row sum against `1.0` with a named tight tolerance appropriate for `float32` probabilities.
- Leave the input score matrix byte-for-byte unchanged.
- Add a causal triangular fixture proving all future positions are exactly zero, not merely close to zero.
- Add a single-selectable-entry row proving probability exactly one at that position and zero elsewhere.
- Add large finite-score fixtures that would overflow without max subtraction.
- Add failure tests for an all-masked row, NaN, positive infinity, wrong dtype, wrong rank, and non-contiguous input.

**Guardrails:**

- Do not replace intentional `-np.inf` with a large finite sentinel.
- Do not apply max over an all-masked row and allow NaN propagation.
- Do not normalize masked entries and merely rely on underflow to make them small.
- Do not mutate the score matrix in place while subtracting maxima.

**Expected result:**

- Every valid row returns stable finite probabilities summing to one.
- Causal future positions are exactly zero.
- Invalid rows fail before any probability matrix is returned.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py -q -k "softmax or mask or causal or selectable or row_sum or infinity or nan"
```

Expected result:

- Ordinary and extreme finite rows match independent fixtures within explicit tolerances.
- Masked positions are exactly zero.
- Invalid rows fail predictably without input mutation.

## Step 8 — Complete public-seam regression, isolation, and scope verification

**Files and symbols:**

- `src/how_llms_work/ml/matrix.py` — final public boundary.
- `tests/test_matrix.py` and `tests/fixtures/matrix_reference.json` — complete focused evidence.
- Existing numerical and completed-route tests — regression only.

**Purpose:**

Prove that the complete boundary is consistent, safe to call repeatedly, and isolated from completed application behavior.

**Actions:**

- Parameterize the common output contract across every pure public primitive.
- Add repeated-call tests proving one returned array can be mutated without changing a later result.
- Add read-only-source tests where appropriate: pure operations may read valid read-only inputs, while the in-place destination must be writable.
- Add sequential and threaded read-only calls using independent arrays to prove there is no module-level mutable state.
- Add exact exception-path tests proving failed operations do not alter any input.
- Run existing math, XOR, Word2Vec, route, and persistence regressions to catch accidental shared-module or dependency changes.
- Inspect the final public module for accidental global arrays, caches, random generators, reusable scratch buffers, or mutable singleton state.
- Confirm no production import from `matrix.py` reaches a route or worker yet; later tickets own integration.
- Run `git diff --check` and inspect `git status --short` so generated caches, data files, or unrelated edits are not included.

**Guardrails:**

- Do not add a temporary integration into empty Transformer modules merely to demonstrate usage.
- Do not weaken existing tests or fixture tolerances to make the new implementation pass.
- Do not run or write to the real `.data` directory.
- Do not add benchmarks as acceptance criteria.

**Expected result:**

- Ticket 013 is complete at the approved public matrix seam.
- Completed Learning Demos remain unchanged.
- The final diff contains only the strict matrix boundary and its focused tests/fixture.

**Verification:**

```powershell
poetry run pytest tests/test_matrix.py
poetry run pytest tests/test_math_utils.py tests/test_neural_net.py tests/test_word2vec_training.py tests/test_word2vec_results.py
poetry run ruff check src/how_llms_work/ml/matrix.py tests/test_matrix.py
poetry run mypy src
```

Expected result:

- All focused matrix behavior passes.
- Existing deterministic utilities and numerical Learning Demo tests remain green.
- Ruff and strict mypy report no issues.

## Focused verification plan

Run from the backend project root:

```powershell
poetry run pytest tests/test_matrix.py
poetry run pytest tests/test_math_utils.py tests/test_neural_net.py tests/test_word2vec_training.py tests/test_word2vec_results.py
poetry run ruff check src/how_llms_work/ml/matrix.py tests/test_matrix.py
poetry run mypy src
```

Expected result:

- Every public primitive has exact rank and shape behavior.
- Unsupported broadcasting, wrong dtype, non-contiguous input, aliasing, NaN, and infinity fail.
- Pure outputs are independent C-contiguous `float32` arrays and inputs remain unchanged.
- Matrix multiplication and column sums match independent wider-accumulation fixtures.
- Row-bias addition accepts only one exact compatible row.
- In-place addition is non-overlapping and transactional.
- Stable row softmax preserves exact causal zeros and normalized finite rows.
- Existing shared numerical utilities and completed numerical Learning Demos do not regress.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
git diff --check
git status --short
```

Expected result:

- All tests pass.
- Ruff reports no issues.
- Strict mypy reports no issues in `src`.
- The diff has no whitespace errors.
- Only intended source, test, and fixture files are modified or added.

No manual browser or Vite-proxy check is required for this numerical-only ticket because it adds no endpoint, SSE, schema, route registration, or frontend behavior.

## Manual acceptance checklist

- [ ] Ticket 013 is the only implemented work item and remains unblocked.
- [ ] `matrix.py` has no mutable module-level numerical state.
- [ ] `matrix.py` does not duplicate `Mulberry32`, TypeScript rounding, or Ticket 012 behavior.
- [ ] Every public operation validates its exact supported rank and shape.
- [ ] Ordinary elementwise operations reject all accidental broadcasting.
- [ ] Row-bias addition accepts only shape `(1, columns)` against a compatible matrix.
- [ ] Unsupported input dtypes are rejected at the documented public boundary.
- [ ] Supported array inputs use dtype exactly `np.float32` and C-contiguous layout.
- [ ] Pure operations return new independent C-contiguous `np.float32` arrays.
- [ ] Pure operations leave every input unchanged.
- [ ] Ordinary operations reject NaN, positive infinity, and negative infinity.
- [ ] Matrix multiplication accumulates in `float64` and casts the completed result once to `float32`.
- [ ] Every confirmed transposed multiplication variant preserves the same precision and output contract.
- [ ] Column sums accumulate in `float64` and cast the completed result once to `float32`.
- [ ] A result that overflows during `float32` materialization is rejected.
- [ ] The in-place operation is explicitly named as mutating.
- [ ] The in-place source and destination must have exact matching shape, dtype, rank, and contiguity.
- [ ] Same-buffer and partially overlapping source/destination memory are rejected.
- [ ] In-place candidate calculation uses separate complete `float64` scratch state.
- [ ] The complete `float32` candidate is validated before commit.
- [ ] Every in-place failure leaves the destination byte-for-byte unchanged.
- [ ] `stable_row_softmax()` accepts finite scores plus intentional `-np.inf` masks only.
- [ ] `stable_row_softmax()` rejects NaN and positive infinity.
- [ ] An all-masked row fails before probabilities are returned.
- [ ] Softmax leaves its score matrix unchanged.
- [ ] Softmax returns a separate finite C-contiguous `float32` probability matrix.
- [ ] Every valid softmax row sums to one within a named tight tolerance.
- [ ] Every masked future position is exactly `0.0`.
- [ ] Transpose returns a copy rather than a shared-memory view.
- [ ] Concatenation and slicing return independent C-contiguous arrays.
- [ ] Every public primitive has exact structural tests and independent numerical fixtures.
- [ ] Precision-sensitive fixtures distinguish approved `float64` accumulation from accidental `float32` accumulation.
- [ ] Tests prove transactionality rather than only checking the raised exception.
- [ ] Tests do not depend on private helper names, BLAS internals, temporary-array identity, or elapsed timing.
- [ ] `transformer.py`, `transformer_worker.py`, routes, schemas, main application, SSE, persistence, and frontend code remain unchanged.
- [ ] `pyproject.toml` and `poetry.lock` remain unchanged.
- [ ] The full pytest, Ruff, and strict-mypy commands are run and their actual outcomes are reported honestly.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/matrix.py
tests/test_matrix.py
tests/fixtures/matrix_reference.json
```

Conditionally changed only if the repository's explicit export convention requires it after inspecting current imports:

```text
src/how_llms_work/ml/__init__.py
```

The default expectation is to leave `ml/__init__.py` unchanged because it is currently empty and existing modules are imported directly.

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/word2vec.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/
tests/test_simple_chat.py
tests/test_bpe.py
tests/test_bpe_tokenize.py
tests/test_math_utils.py
tests/test_neural_net.py
tests/test_neural_net_persistence.py
tests/test_neural_net_route.py
tests/test_train_embed_persistence.py
tests/test_train_embed_route.py
tests/test_word2vec.py
tests/test_word2vec_training.py
tests/test_word2vec_results.py
.data/
pyproject.toml
poetry.lock
poetry.toml
README.md
frontend/
SPEC.md
CONTEXT.md
0002-stabilize-python-transformer-training-and-process-lifecycle.md
013-provide-strict-float32-matrix-primitives.md
```

## Risk notes and safeguards

1. **Risk:** NumPy silently broadcasts incompatible shapes.
   - **Safeguard:** Validate exact shape relationships before every calculation and reserve broadcasting exclusively for exact `(1, columns)` row bias.

2. **Risk:** Public functions silently cast `float64`, integer, list, or non-contiguous inputs.
   - **Safeguard:** Require actual C-contiguous `np.ndarray` inputs with dtype exactly `np.float32`.

3. **Risk:** A pure transpose or slice returns a view and later mutation corrupts an input.
   - **Safeguard:** Apply the common independent C-contiguous output materialization path and assert `np.shares_memory()` is false.

4. **Risk:** Default NumPy multiplication or reduction accumulates at the wrong precision.
   - **Safeguard:** Request or create explicit `float64` accumulation and protect it with precision-sensitive independent fixtures.

5. **Risk:** Repeated casting to `float32` recreates TypeScript scalar behavior accidentally but violates the accepted Python contract.
   - **Safeguard:** Accumulate the complete approved reduction in `float64` and materialize exactly once.

6. **Risk:** A finite `float64` candidate overflows to `float32` infinity.
   - **Safeguard:** Validate the completed materialized `float32` result before return or commit.

7. **Risk:** In-place addition writes some elements before a later overflow or invalid value is detected.
   - **Safeguard:** Calculate and validate the complete candidate in separate scratch state, then perform one final commit.

8. **Risk:** Object-identity checks miss partially overlapping views.
   - **Safeguard:** Use actual memory-overlap detection and include crafted partial-view tests.

9. **Risk:** Softmax treats an all-masked row as valid and returns NaN.
   - **Safeguard:** Require at least one finite selectable value in every row before max subtraction.

10. **Risk:** Masked positions become tiny positive values rather than exact zero.
    - **Safeguard:** Exclude masks from exponentiation and explicitly set masked probabilities to `0.0` before final validation.

11. **Risk:** Softmax overflows because the row maximum is not subtracted.
    - **Safeguard:** Protect with large finite-score fixtures and calculate exponentials only after stable shifting.

12. **Risk:** One broad tolerance hides shape-order, accumulation, or formula errors.
    - **Safeguard:** Use exact assertions for discrete/structural behavior and named smallest-verified tolerances for each numerical boundary.

13. **Risk:** Fixtures reproduce the same bug as production code.
    - **Safeguard:** Record provenance and prohibit imports of the production matrix module during expected-value generation.

14. **Risk:** The TypeScript reference's global random state is copied into `matrix.py`.
    - **Safeguard:** Treat Ticket 012 and ADR 0002 as ownership authority; leave all randomness in request-owned `Mulberry32` instances outside this module.

15. **Risk:** The ticket expands into a general Tensor API or early Transformer implementation.
    - **Safeguard:** Inventory exact Phase 5 call-site needs first, expose only those primitives, and enforce the expected-file list during final diff review.

16. **Risk:** Tests require private vectorization or BLAS behavior and become platform fragile.
    - **Safeguard:** Assert public values, precision policy, memory contract, and transactionality rather than implementation mechanics.

17. **Risk:** A formatter or broad cleanup changes completed code.
    - **Safeguard:** Limit formatting to changed files, inspect `git diff --check`, and reject unrelated changes before commit.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Provide strict float32 matrix primitives
```

The commit body should mention:

- the stateless, shape-checked NumPy matrix boundary;
- exact `float32` inputs and independent C-contiguous pure outputs;
- explicit `float64` accumulation for multiplication and column sums;
- exact row-bias-only broadcasting;
- transactional non-overlapping in-place addition;
- stable causal-mask-aware row softmax with exact masked zeros;
- independent exact/tolerance fixtures and byte-preservation failure tests;
- no Transformer architecture, worker, route, persistence, frontend, or dependency changes;
- the exact pytest, Ruff, and mypy commands actually executed and their observed results.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan013.md`;
- `013-provide-strict-float32-matrix-primitives.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `py_llm_pipeline_explorer_file_structure(35).md`;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the repository again, establish its own baseline before editing, preserve user changes, implement only Ticket 013, add independent failure-first matrix evidence, run focused and full verification, report actual command results honestly, inspect the final scope, and create the implementation commit only after all required checks pass.
