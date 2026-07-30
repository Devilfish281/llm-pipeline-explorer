---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "012"
source_work_item: 012-centralize-typescript-compatible-randomness-and-rounding.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(32).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 012: Centralize TypeScript-compatible randomness and rounding

## Initial checklist

- Confirm Ticket 012 is the only selected work item and has no blockers.
- Treat `py_llm_pipeline_explorer_file_structure(32).md` as the current Python Backend source of truth.
- Use `llm_works_file_structure.md` only as the TypeScript Reference Implementation for independently fixed random and rounding behavior.
- Preserve the user-reported passing pytest, Ruff, and strict mypy baseline without describing it as tool-verified in this planning session.
- Limit production changes to the shared deterministic utility module and compatibility imports/wrappers in the completed Word2Vec and XOR numerical modules.
- Add independent exact utility fixtures before or alongside extraction; never generate expected values through the production Python operation under test.
- Finish with focused deterministic-utility and completed-demo regressions, then the full pytest suite, Ruff, and strict mypy.

## Source-of-truth hierarchy

1. The user's latest explicit direction to convert the selected TypeScript behavior to Python and to treat the latest complete Python Backend export as current-code truth.
2. `012-centralize-typescript-compatible-randomness-and-rounding.md` for required behavior, acceptance criteria, the approved public test seam, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(32).md` for the current implementation, tests, fixtures, paths, dependencies, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and ADR 0002 for the durable shared-utility ownership, Sample Random Stream, Weight Initialization, signed-zero, exact-fixture, and regression decisions.
5. `llm_works_file_structure.md`, especially the TypeScript Word2Vec Mulberry32 implementation and `Math.round(...)` call sites, as independent compatibility evidence only.
6. Older exports, backup modules, prior plans, snippets, and earlier assumptions are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 012 creates the reusable deterministic utility boundary required before Transformer numerical work begins.

The current live Python Backend already contains a TypeScript-compatible `Mulberry32` implementation and a finite decimal-rounding helper inside `src/how_llms_work/ml/word2vec.py`. The XOR module independently contains `round_like_typescript()` with substantially the same decimal-rounding formula. In contrast, `src/how_llms_work/ml/math_utils.py` is empty.

The implementation must move the canonical Mulberry32 state machine and TypeScript-compatible decimal-rounding behavior into `math_utils.py`, without changing the completed Word2Vec or XOR Learning Demo contracts. `word2vec.py` must import and re-export the exact shared `Mulberry32` class and its established rounding name rather than retaining local implementations. `neural_net.py` must keep the existing `round_like_typescript()` name and signature as a compatibility wrapper that delegates to the shared boundary.

The shared random generator must remain instance-owned: no module-global generator, cached stream, or shared draw counter may be introduced. Separate Weight Initialization and Sample Random Stream instances must be provably independent, including when their draws are interleaved or used concurrently.

The rounding boundary must distinguish two related requirements:

1. reproduce JavaScript `Math.round(value × scale) / scale`, including negative half ties and raw signed-zero behavior;
2. provide the explicit public-number normalization required by current Word2Vec/XOR outputs and future Saved Transformer Model values, converting either sign of zero to ordinary positive `0.0`.

Non-finite values must be rejected before public output or persistence. This ticket does not implement Transformer preprocessing, initialization traversal, matrix operations, forward/backward mathematics, generation, workers, routes, SSE, or persistence.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, mypy, Node, browser, or two-server command was executed while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and report the actual results honestly.

## Current code observations from the latest source

- `src/how_llms_work/ml/math_utils.py` is empty, so the ADR-owned shared deterministic utility boundary does not yet exist.
- `src/how_llms_work/ml/word2vec.py` currently owns:
  - `UINT32_MASK`;
  - `MULBERRY32_INCREMENT`;
  - `MULBERRY32_DIVISOR`;
  - the request-owned `Mulberry32` class;
  - `state` and `draw_count` properties;
  - `round_typescript_decimal()`;
  - `round_embedding_loss()`.
- The current `Mulberry32.__init__()` normalizes its seed with `seed & 0xFFFFFFFF`, and `random()`:
  - advances state by the established increment modulo `2^32`;
  - applies the established unsigned 32-bit multiply/XOR/shift sequence;
  - returns a value divided by `2^32`;
  - increments `draw_count` exactly once.
- `tests/fixtures/word2vec_training_reference.json` already contains independent exact Mulberry32 evidence for seed `42` and seed `4_294_967_295`, including output sequences and final state.
- `tests/test_word2vec_training.py` already proves:
  - the seed-42 and maximum-seed sequences;
  - exact state and draw counts;
  - deterministic Word2Vec initialization, shuffling, and sampling;
  - no NumPy random use in the Word2Vec production path;
  - sequential and concurrent Embedding Training Run isolation.
- The current Word2Vec fixture does not independently cover seed zero, positive seed overflow such as `2^32 + 42`, negative wraparound such as `-1`, or direct interleaving of two standalone generator instances.
- `round_typescript_decimal()` currently:
  - rejects negative digit counts;
  - rejects non-finite input, non-finite scaled values, and non-finite rounded values;
  - uses `floor(scaled + 0.5) / scale`;
  - normalizes any zero result to positive `0.0`.
- `tests/test_word2vec_results.py` already covers ordinary six-decimal values, positive and negative half ties, `-1.5`, a very small negative magnitude that becomes `0.0`, invalid digits, and NaN rejection.
- The current Word2Vec helper does not expose a separately testable raw signed-zero-preserving path. That distinction is required because JavaScript rounding semantics and public persisted zero normalization are related but not identical contracts.
- `src/how_llms_work/ml/neural_net.py` independently defines `round_like_typescript()`, using the same scale-plus-floor formula and positive-zero normalization, but it does not explicitly reject non-finite values before rounding.
- `tests/test_neural_net.py` already protects the XOR wrapper name and representative outputs, including ordinary values and negative half ties.
- The live Word2Vec code calls its rounding boundary for public losses, six-decimal vectors, two-decimal similarity scores, and analogy scores. Therefore extraction must be behavior-preserving across both `tests/test_word2vec_training.py` and `tests/test_word2vec_results.py`.
- `src/how_llms_work/ml/transformer.py`, `src/how_llms_work/ml/transformer_worker.py`, and `src/how_llms_work/routes/train_transformer.py` are empty. Ticket 012 should make the shared boundary importable by later Transformer tickets but must not add Transformer call sites now.
- `src/how_llms_work/ml/__init__.py` is empty. A package-level re-export is not required by the ticket; the stable seam can remain `how_llms_work.ml.math_utils`.
- `src/how_llms_work/ml/word2vec_OLD.py` is a duplicate archival module in the supplied export and has no shown import references. It must not become an alternate utility source. The implementation should confirm repository references before deciding whether to remove that stale tracked backup.
- `pyproject.toml` already provides Python 3.12, NumPy, pytest, Ruff, strict mypy, and Black. No dependency or lockfile change is expected.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - The live Word2Vec `Mulberry32` implementation reproduces independent JavaScript-reference sequences for seed `42` and maximum unsigned 32-bit seed.
  - Current state is normalized with an unsigned 32-bit mask.
  - Each current `random()` call increments `draw_count` once.
  - Existing Word2Vec training, result, and persistence-ready output fixtures are deterministic.
  - Existing Word2Vec public rounding rejects non-finite values.
  - Existing Word2Vec and XOR public wrappers normalize rounded zero to positive `0.0`.
  - Existing XOR wrapper name `round_like_typescript` is present and tested.
  - Existing completed Word2Vec runs are sequentially and concurrently isolated.
- **Behavior present but evidence incomplete:**
  - Seed zero and explicit positive/negative wraparound behavior follow from the current mask but lack independent fixed utility fixtures.
  - Same-seed standalone generators are instance-owned by implementation shape, but direct lockstep/interleaving tests are absent.
  - Separate conceptual Weight Initialization and Sample Random Stream instances can already be created, but their mutual non-interference is not directly tested at the approved shared utility seam.
  - Current decimal rounding covers several ties and small magnitudes, but raw negative-zero preservation is not separated from public zero normalization.
  - Current XOR non-finite values fail indirectly rather than through the approved shared finite-input boundary.
- **Partially implemented:**
  - The required algorithms exist, but ownership is split between `word2vec.py` and `neural_net.py` rather than centralized in `math_utils.py`.
  - Existing Word2Vec fixtures can protect regression behavior, but no dedicated shared-utility fixture/test module exists.
- **Not implemented:**
  - Canonical `Mulberry32` ownership in `ml/math_utils.py`.
  - Canonical TypeScript-compatible decimal-rounding ownership in `ml/math_utils.py`.
  - Raw signed-zero-preserving rounding evidence plus explicit public positive-zero normalization.
  - Word2Vec re-export identity proving that `word2vec.Mulberry32` is the shared class rather than a copy.
  - XOR delegation to the shared rounding implementation while preserving `round_like_typescript()`.
  - Independent utility fixtures for seed zero and explicit wraparound equivalence cases.
  - Direct same-seed, interleaved-stream, Weight Initialization versus Sample Random Stream, and concurrent utility-instance isolation tests.
- **Evidence limitation:**
  - Baseline commands are user-reported rather than tool-verified in this planning session.
  - The existing Word2Vec reference fixture is strong regression evidence but was created for Word2Vec training rather than the new shared utility seam.
  - Exact new fixture values must be independently captured from the TypeScript Reference Implementation or calculated by a separate reference script that imports no production Python utility.
  - The implementation should not broaden the public API beyond the smallest stable symbols needed by Word2Vec, XOR, and later Transformer tickets.

## Files to inspect before editing

1. `src/how_llms_work/ml/math_utils.py` — empty destination for the canonical Mulberry32 and decimal-rounding boundary.
2. `src/how_llms_work/ml/word2vec.py` — current Mulberry32 constants/class, `round_typescript_decimal()`, `round_embedding_loss()`, all utility call sites, and the compatibility names that must remain importable.
3. `src/how_llms_work/ml/neural_net.py` — current `round_like_typescript()` wrapper and every XOR loss/prediction call site.
4. `src/how_llms_work/ml/word2vec_OLD.py` — stale duplicate utility definitions; confirm it is unreferenced before conditionally removing it from the tracked source tree.
5. `src/how_llms_work/ml/__init__.py` — confirm that no package-level export convention needs changing; leave empty unless current repository usage proves otherwise.
6. `tests/test_word2vec_training.py` — current exact Mulberry32, draw-count, training-sequence, and concurrent-run regression tests.
7. `tests/fixtures/word2vec_training_reference.json` — existing independent seed-42 and maximum-seed evidence that must remain unchanged unless a strictly additive correction is unavoidable.
8. `tests/test_word2vec_results.py` — current decimal-rounding and complete public-output regression tests.
9. `tests/test_neural_net.py` — current XOR wrapper and completed numerical-result regression tests.
10. `tests/test_train_embed_route.py` and `tests/test_train_embed_persistence.py` — completed endpoint/model regressions to include in broader verification without modifying route behavior.
11. `pyproject.toml` — current Python, pytest, Ruff, strict mypy, NumPy, and no-new-dependency constraints.
12. `012-centralize-typescript-compatible-randomness-and-rounding.md`, `SPEC.md`, `CONTEXT.md`, and ADR 0002 — acceptance authority, canonical terminology, signed-zero requirement, module ownership, and exact fixture policy.
13. `llm_works_file_structure.md` — TypeScript Mulberry32 arithmetic and `Math.round(...)` call sites used only to capture independent expected evidence.

## Step 1 — Establish independent shared-utility fixtures and public-seam tests

**Files and symbols:**
- `tests/fixtures/math_utils_reference.json` — new fixed independent reference data.
- `tests/test_math_utils.py` — new tests for the approved public deterministic-utility seam.
- `llm_works_file_structure.md` — TypeScript source used to capture expected values, not imported by the Python test suite.

**Purpose:**

Create failure-first evidence for the complete shared contract before extraction. This prevents a move-only refactor from silently preserving existing gaps, especially seed zero, explicit wraparound equivalence, interleaved streams, raw signed zero, and public zero normalization.

**Actions:**

- Add one committed JSON fixture with source/provenance metadata stating that expected values were captured from the supplied TypeScript Reference Implementation or an independent scalar reference that imports no `how_llms_work` production module.
- Include representative Mulberry32 cases:
  - seed `42`;
  - seed `0`;
  - seed `4_294_967_295`;
  - seed `4_294_967_296 + 42`, proving positive modulo-`2^32` normalization;
  - seed `-1`, proving negative wraparound equivalence to maximum unsigned 32-bit state.
- For every random case, store:
  - the original seed;
  - the normalized initial state;
  - an exact output sequence long enough to catch shift/multiply/masking mistakes;
  - final state;
  - final draw count.
- Include rounding cases for:
  - ordinary values;
  - values beyond six decimals;
  - positive half ties;
  - negative half ties;
  - values immediately around half ties;
  - very small positive and negative magnitudes;
  - positive zero;
  - negative zero input;
  - a negative value that JavaScript rounds to negative zero;
  - six-decimal public normalization to positive `0.0`.
- Store signed-zero expectations separately from numeric equality because `-0.0 == 0.0` in Python. Tests must use `math.copysign()` or an equivalent sign-sensitive assertion.
- Add parameterized exact tests against the new fixture for shared `Mulberry32`.
- Add a same-seed lockstep test proving two instances produce identical values while maintaining separate state and draw counts.
- Add an interleaving test:
  - build uninterrupted control instances;
  - interleave draws from two new instances;
  - prove each interleaved sequence still matches its own control sequence exactly.
- Add a conceptual Weight Initialization versus Sample Random Stream test using independent instances and the approved sample seed calculation `(42 + epoch) & 0xFFFFFFFF`; consuming either stream must not alter the other.
- Add sequential and `ThreadPoolExecutor` tests in which each task creates its own generator. Assert exact outputs, states, and draw counts.
- Do not assert thread-safe concurrent mutation of one deliberately shared generator instance; the contract permits shared state only when the caller explicitly passes the same instance.
- Add finite-input rejection tests for NaN, positive infinity, negative infinity, and any scaling overflow case supported by the public signature.
- Add negative-digit rejection tests.
- Keep expected data static. No test helper may call the production operation to construct expected sequences, expected rounded values, or expected zero signs.

**Guardrails:**

- Do not invoke Node or TypeScript during ordinary pytest.
- Do not add a package dependency for fixture generation.
- Do not inspect private bit-operation helper names or internal storage layout.
- Do not make tests depend on Python's `random` module or NumPy randomness.
- Do not weaken exact equality for random streams or rounded public numbers.
- Keep the fixture small and purpose-specific; do not duplicate complete Word2Vec training fixtures.

**Expected result:**

- The shared utility contract is completely specified by independent exact evidence.
- The tests initially fail while `math_utils.py` remains empty, establishing the intended implementation target.
- Random stream ownership, signed-zero behavior, public zero normalization, and non-finite rejection are observable without any Transformer implementation.

**Verification:**

```powershell
poetry run pytest tests/test_math_utils.py -q
```

Expected before implementation:

- Collection succeeds.
- Tests fail only because the new shared utility symbols are not yet implemented.

Expected after implementation:

- All shared deterministic-utility tests pass exactly.

## Step 2 — Implement the canonical deterministic utility boundary

**Files and symbols:**
- `src/how_llms_work/ml/math_utils.py` — shared `Mulberry32`, TypeScript-compatible decimal rounding, and public zero normalization.
- `tests/test_math_utils.py` — direct approved-seam verification.
- `tests/fixtures/math_utils_reference.json` — fixed expected values.

**Purpose:**

Make `math_utils.py` the single live owner of JavaScript-compatible random streams and decimal rounding, with no global mutable generator state and no dependency on Word2Vec, XOR, Transformer, NumPy randomness, or route code.

**Actions:**

- Move the established unsigned 32-bit Mulberry32 constants and state machine into `math_utils.py`.
- Expose one stable `Mulberry32` class with:
  - per-instance state;
  - per-instance draw count;
  - constructor seed normalization modulo `2^32`;
  - exact JavaScript-compatible unsigned masking after the required arithmetic stages;
  - one state advance and one draw-count increment per successful `random()` call;
  - output in `[0.0, 1.0)`.
- Keep the implementation standard-library-only unless the current signature must accept an existing NumPy scalar through ordinary float conversion.
- Do not create a module-global generator, singleton, cached stream, hidden seed counter, or lock-shared state.
- Implement the TypeScript-compatible decimal-rounding primitive with:
  - non-negative digit validation;
  - finite input validation before scaling;
  - finite scaled-value validation;
  - JavaScript-compatible half-tie direction;
  - sign-sensitive handling of the negative-zero region;
  - finite final-result validation.
- Provide the smallest explicit public normalization path required by completed demos and future Saved Transformer Model conversion. This may be:
  - a narrowly named second shared helper; or
  - an explicit keyword on the shared operation.
- The raw rounding path must permit tests to observe JavaScript negative zero.
- The public-normalizing path must convert either `-0.0` or `0.0` to ordinary positive `0.0`.
- Use clear docstrings describing raw TypeScript semantics versus public JSON/model normalization.
- Keep symbol names stable and simple enough for later direct imports by `transformer.py`.
- Add an explicit module export list only if it improves the stable boundary without requiring changes to `ml/__init__.py`.

**Guardrails:**

- Do not use Python built-in `round()` because its tie behavior differs.
- Do not use `decimal.Decimal` as a substitute for the JavaScript binary floating-point operation.
- Do not use NumPy, Python `random`, cryptographic randomness, or a third-party PRNG.
- Do not add Transformer-specific seed calculations or Xavier traversal to `math_utils.py`; callers own stream creation and draw order.
- Do not suppress non-finite inputs by converting them to zero.
- Do not catch broad exceptions.
- Do not annotate or expose private intermediate arithmetic merely for tests.

**Expected result:**

- `how_llms_work.ml.math_utils` is the stable public deterministic-utility seam.
- All fixture-backed random, wraparound, draw-count, stream-independence, rounding, signed-zero, normalization, and non-finite tests pass.
- The module remains independent of the completed Learning Demos and future Transformer internals.

**Verification:**

```powershell
poetry run pytest tests/test_math_utils.py -q
```

Expected result:

- All shared utility tests pass.
- Exact random outputs, states, draw counts, and rounded results match the committed independent fixture.

## Step 3 — Make Word2Vec re-export and use the shared implementation without behavior changes

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — current local Mulberry32 constants/class, `round_typescript_decimal()`, `round_embedding_loss()`, `EmbeddingTrainingRun`, and public result builders.
- `tests/test_word2vec_training.py` — re-export identity and exact training regression.
- `tests/test_word2vec_results.py` — existing rounding and public-output regression.
- `tests/fixtures/word2vec_training_reference.json` — existing exact random/training evidence.
- `tests/fixtures/word2vec_results_reference.json` — existing exact public result/model evidence.

**Purpose:**

Remove the second live implementation while preserving every established Word2Vec import, random draw, training fixture, public vector, score, result, and Saved Embedding Model value.

**Actions:**

- Import `Mulberry32` from `how_llms_work.ml.math_utils` into `word2vec.py` under the existing public name.
- Remove the local Mulberry32 constants and class implementation from `word2vec.py`.
- Import the shared public-normalizing decimal helper under the existing `round_typescript_decimal` name, or retain a one-line compatibility wrapper only if required to preserve the exact public signature and exception contract.
- Keep `round_embedding_loss()` available and make it delegate through the shared Word2Vec-compatible rounding name.
- Preserve all Word2Vec-specific constants such as seed `42`, public digits, learning rate, sigmoid limits, and corpus data in `word2vec.py`; only the generic deterministic algorithms move.
- Do not alter any call order inside:
  - weight initialization;
  - Fisher–Yates shuffling;
  - negative sampling;
  - positive/negative coordinate updates;
  - public vector conversion;
  - neighbor/similarity/analogy scoring.
- Add a direct compatibility assertion proving:

  ```python
  how_llms_work.ml.word2vec.Mulberry32 is how_llms_work.ml.math_utils.Mulberry32
  ```

- Add the equivalent identity/delegation assertion for the Word2Vec rounding name when direct re-export is used.
- Retain the current independent Word2Vec fixture unchanged whenever possible. It must remain a regression oracle for seed-42 random consumption and complete short-run outputs.
- Run existing exact Word2Vec training and result tests without loosening equality or tolerances.

**Guardrails:**

- Do not create a subclass or copied wrapper class around `Mulberry32`; Word2Vec must expose the same class object.
- Do not change the production seed or draw order.
- Do not replace the deterministic generator with NumPy or Python randomness.
- Do not change public zero normalization for existing Word2Vec losses, vectors, or scores.
- Do not regenerate existing expected Word2Vec fixtures from the moved production helper.
- Do not change route, SSE, persistence, request validation, corpus, or BPE behavior.

**Expected result:**

- Existing imports such as `from how_llms_work.ml.word2vec import Mulberry32` continue to work.
- They resolve to the canonical shared class.
- Existing Word2Vec exact random sequences, training fixtures, rounded public outputs, result ordering, and model contents remain unchanged.

**Verification:**

```powershell
poetry run pytest `
    tests/test_math_utils.py `
    tests/test_word2vec_training.py `
    tests/test_word2vec_results.py `
    -q
```

Expected result:

- All tests pass.
- No existing Word2Vec fixture or expected public value changes merely because ownership moved.

## Step 4 — Preserve the XOR rounding wrapper while delegating to the shared boundary

**Files and symbols:**
- `src/how_llms_work/ml/neural_net.py` — `round_like_typescript()` and its loss/prediction call sites.
- `tests/test_neural_net.py` — wrapper-name, rounding, full Training Run, prediction, and Saved Weight Snapshot regressions.
- `src/how_llms_work/ml/math_utils.py` — shared public-normalizing decimal helper.

**Purpose:**

Eliminate the second rounding implementation without changing the completed XOR module's public name, call sites, rounded losses, predictions, verdict behavior, or snapshot output.

**Actions:**

- Keep `round_like_typescript(value, digits)` defined in `neural_net.py`.
- Replace its local arithmetic with delegation to the shared public-normalizing rounding boundary.
- Preserve acceptance of the existing `float | np.float32` inputs through explicit safe conversion at the wrapper or shared boundary.
- Preserve the existing negative-digit rejection behavior.
- Ensure NaN and infinities now fail at the shared finite-input boundary before becoming an Epoch Update or Prediction.
- Extend the existing parameterized wrapper test only as needed to prove:
  - the name remains importable;
  - ordinary values are unchanged;
  - positive and negative half ties are unchanged;
  - very small rounded values normalize to positive `0.0`;
  - non-finite values are rejected.
- Keep all XOR numerical initialization, training, reporting, prediction ordering, verdict calculations, route behavior, and persistence untouched.
- Run the entire existing neural numerical suite, not only the wrapper test, because the helper is used in Epoch Updates and final Predictions.

**Guardrails:**

- Do not rename or remove `round_like_typescript()`.
- Do not make XOR import Word2Vec to reach the helper.
- Do not change prediction precision, loss precision, threshold logic, architecture labels, or verdict text.
- Do not alter NumPy random initialization; Ticket 012 centralizes JavaScript-compatible randomness, not the existing Python-owned XOR initialization scheme.
- Do not change route or persistence modules.

**Expected result:**

- Existing XOR imports and callers remain compatible.
- XOR uses the same shared decimal-rounding implementation as Word2Vec and future Transformer code.
- Completed XOR public behavior remains exactly unchanged for valid finite training values.

**Verification:**

```powershell
poetry run pytest tests/test_neural_net.py -q
```

Expected result:

- All XOR numerical and wrapper tests pass.
- Exact rounded losses, predictions, verdicts, and snapshots remain unchanged.

## Step 5 — Remove or neutralize stale duplicate utility ownership and prove repository-wide regression safety

**Files and symbols:**
- `src/how_llms_work/ml/word2vec_OLD.py` — conditionally remove only after confirming it is tracked, unreferenced, and not intentionally required by repository instructions.
- `src/how_llms_work/ml/math_utils.py` — canonical live owner.
- `src/how_llms_work/ml/word2vec.py` — canonical compatibility re-export.
- `src/how_llms_work/ml/neural_net.py` — canonical compatibility wrapper.
- Completed Word2Vec, XOR, route, and persistence tests.

**Purpose:**

Ensure the repository contains one live deterministic implementation and that the extraction does not regress completed Learning Demos or drift into Transformer implementation.

**Actions:**

- Search the real repository before editing:

  ```powershell
  rg -n "word2vec_OLD|from .*word2vec_OLD|import .*word2vec_OLD" .
  ```

- If `word2vec_OLD.py` is tracked and unreferenced exactly as the latest export indicates, remove it from `src` so it cannot remain a second package-level Mulberry32/rounding implementation or mislead later Transformer work.
- If repository instructions establish that the backup must remain, do not import or modify it as production truth; document the reason in the implementation report and confirm no runtime/test imports reference it.
- Search live source for duplicate algorithm definitions after the change:

  ```powershell
  rg -n "class Mulberry32|MULBERRY32_INCREMENT|def round_typescript_decimal|def round_like_typescript" src
  ```

- Expected live arrangement:
  - one `Mulberry32` class in `math_utils.py`;
  - one canonical generic decimal implementation in `math_utils.py`;
  - a Word2Vec import/re-export or thin compatibility wrapper;
  - the required thin XOR wrapper.
- Run focused completed-demo route and persistence regressions to prove the utility move did not change HTTP/SSE or saved-model behavior.
- Inspect the diff and confirm no Transformer, route, schema, frontend, dependency, or generated `.data` change is present.

**Guardrails:**

- Do not delete a file without first confirming repository references and instructions.
- Do not rewrite archival code as a second shared implementation.
- Do not add utility imports to empty Transformer modules merely to demonstrate future use.
- Do not change existing JSON fixtures unless an independently proven fixture defect exists.
- Do not commit generated `.data` files, caches, or local test artifacts.

**Expected result:**

- The active Python Backend has one canonical deterministic utility implementation.
- Completed Word2Vec and XOR Learning Demos remain exact.
- Later Transformer tickets can import `Mulberry32` and the public rounding boundary directly from `math_utils.py`.
- Ticket 012 introduces no Transformer implementation and no frontend/backend contract changes.

**Verification:**

```powershell
poetry run pytest `
    tests/test_math_utils.py `
    tests/test_neural_net.py `
    tests/test_neural_net_persistence.py `
    tests/test_neural_net_route.py `
    tests/test_word2vec.py `
    tests/test_word2vec_training.py `
    tests/test_word2vec_results.py `
    tests/test_train_embed_persistence.py `
    tests/test_train_embed_route.py `
    -q
```

Expected result:

- All focused deterministic utility, XOR, Word2Vec, route, and persistence regressions pass.

## Focused verification plan

Run from the backend directory:

```powershell
poetry run pytest `
    tests/test_math_utils.py `
    tests/test_word2vec_training.py `
    tests/test_word2vec_results.py `
    tests/test_neural_net.py `
    -q
```

Expected result:

- Shared random and rounding fixtures pass exactly.
- Word2Vec exposes the shared class and preserves every exact training/public result fixture.
- XOR preserves its established rounding wrapper and complete numerical behavior.

Then run the wider completed-demo regression set:

```powershell
poetry run pytest `
    tests/test_neural_net_persistence.py `
    tests/test_neural_net_route.py `
    tests/test_train_embed_persistence.py `
    tests/test_train_embed_route.py `
    -q
```

Expected result:

- Existing persistence and route contracts remain unchanged.

## Full verification plan

Run from:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"
```

Then:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.

Do not claim these future commands passed until the implementation session actually runs them successfully.

## Manual acceptance checklist

- [ ] `src/how_llms_work/ml/math_utils.py` is the only live owner of the Mulberry32 algorithm.
- [ ] Seed `42` matches the independent JavaScript fixture exactly.
- [ ] Seed `0` matches the independent JavaScript fixture exactly.
- [ ] Seed `4_294_967_295` matches the independent JavaScript fixture exactly.
- [ ] Positive overflow and negative seeds normalize modulo `2^32` and match their equivalent normalized seeds.
- [ ] Every successful `random()` call consumes exactly one draw.
- [ ] Two same-seed instances advance independently.
- [ ] Interleaving two instances does not alter either sequence.
- [ ] Conceptual Weight Initialization and Sample Random Stream instances do not alter one another.
- [ ] Sequential and concurrent callers using separate instances produce identical expected results without shared state.
- [ ] Raw TypeScript-compatible rounding preserves the expected sign of zero.
- [ ] Public six-decimal normalization returns positive `0.0`.
- [ ] Ordinary values, values beyond six decimals, positive half ties, negative half ties, and small magnitudes match fixed expected values.
- [ ] NaN and positive/negative infinity are rejected.
- [ ] `word2vec.Mulberry32 is math_utils.Mulberry32`.
- [ ] Existing `word2vec.round_typescript_decimal` imports remain valid.
- [ ] Existing Word2Vec random sequences, training fixtures, losses, vectors, rankings, analogies, and Saved Embedding Model outputs are unchanged.
- [ ] Existing `neural_net.round_like_typescript` imports remain valid.
- [ ] Existing XOR losses, predictions, verdicts, and snapshots are unchanged.
- [ ] No HTTP request gains a seed field.
- [ ] No Transformer preprocessing, mathematics, initialization traversal, generation, worker, route, SSE, or persistence code is added.
- [ ] No dependency or lockfile change is present.
- [ ] No generated `.data`, cache, or temporary file is included in the diff.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/word2vec.py
src/how_llms_work/ml/neural_net.py
tests/test_math_utils.py
tests/fixtures/math_utils_reference.json
tests/test_word2vec_training.py
tests/test_neural_net.py
```

Conditionally changed:

```text
src/how_llms_work/ml/word2vec_OLD.py
tests/test_word2vec_results.py
```

Conditions:

- Delete `word2vec_OLD.py` only after confirming it is tracked, unreferenced, and not required by repository instructions.
- Change `test_word2vec_results.py` only if the cleanest compatibility assertion belongs beside its existing rounding tests; its existing fixture values should not change.

Expected unchanged fixtures:

```text
tests/fixtures/word2vec_training_reference.json
tests/fixtures/word2vec_results_reference.json
tests/fixtures/word2vec_preprocessing_reference.json
```

No package or lockfile change is expected.

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/
tests/test_simple_chat.py
tests/test_bpe.py
tests/test_bpe_tokenize.py
frontend/
.data/
README.md
pyproject.toml
poetry.lock
poetry.toml
SPEC.md
CONTEXT.md
0002-stabilize-python-transformer-training-and-process-lifecycle.md
012-centralize-typescript-compatible-randomness-and-rounding.md
```

## Risk notes and safeguards

1. **Risk:** A mechanical move changes unsigned arithmetic because Python integers do not overflow automatically.
   - **Safeguard:** Preserve explicit `0xFFFFFFFF` masking at the established stages and protect seeds, outputs, states, and draw counts with exact independent fixtures.

2. **Risk:** One `random()` call increments the draw counter at the wrong point or more than once.
   - **Safeguard:** Assert output, final state, and draw count together for every fixture case, including interrupted/interleaved consumption.

3. **Risk:** A module-global generator accidentally couples Word2Vec, Weight Initialization, or Generated Text Samples.
   - **Safeguard:** Keep all mutable state on `Mulberry32` instances and test same-seed, interleaved, conceptual stream-role, sequential, and concurrent isolation.

4. **Risk:** Python built-in `round()` or a simplified floor formula mishandles negative half ties or negative zero.
   - **Safeguard:** Use one explicit JavaScript-compatible primitive, sign-sensitive zero tests, and a separate explicit public-zero normalization path.

5. **Risk:** Normalizing zero inside the raw primitive makes it impossible to reproduce JavaScript signed-zero behavior.
   - **Safeguard:** Keep raw rounding semantics observable and apply positive-zero normalization only at the public/model boundary.

6. **Risk:** Preserving raw negative zero changes current Word2Vec or XOR JSON-facing outputs.
   - **Safeguard:** Route both completed demos through the explicit public-normalizing path and retain their exact regression fixtures.

7. **Risk:** Word2Vec imports still work but resolve to a copied class or subclass.
   - **Safeguard:** Assert object identity between `word2vec.Mulberry32` and `math_utils.Mulberry32`.

8. **Risk:** XOR's wrapper is removed during deduplication.
   - **Safeguard:** Keep `round_like_typescript()` as a thin compatibility wrapper and test its existing name and outputs directly.

9. **Risk:** Expected test values are accidentally generated by the production helper, allowing the same defect in implementation and fixture.
   - **Safeguard:** Commit fixture provenance, capture values independently, and prohibit production imports in fixture-generation logic.

10. **Risk:** Moving the helper changes Word2Vec random-call order or public values even when the generator sequence itself is correct.
    - **Safeguard:** Run the complete existing Word2Vec training and result fixtures unchanged.

11. **Risk:** `word2vec_OLD.py` remains an attractive but stale second source of truth.
    - **Safeguard:** Confirm references and remove the tracked archival duplicate when safe; otherwise explicitly leave it unused and outside the live ownership boundary.

12. **Risk:** The shared module imports completed demos and creates a circular dependency.
    - **Safeguard:** Keep `math_utils.py` dependency-light and one-directional; Word2Vec and XOR import it, never the reverse.

13. **Risk:** The ticket expands into early Transformer implementation.
    - **Safeguard:** Make the utility importable by future Transformer code but leave all Transformer modules unchanged.

14. **Risk:** A formatter or broad refactor creates unrelated churn.
    - **Safeguard:** Restrict edits to the expected files, run `git diff --check`, inspect `git status --short`, and reject unrelated changes before commit.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Suggested outcome:

```text
Centralize deterministic random and rounding utilities
```

Commit body should mention:

- canonical JavaScript-compatible Mulberry32 ownership in `ml/math_utils.py`;
- exact seed, wraparound, draw-count, signed-zero, public-normalization, and stream-isolation fixtures;
- Word2Vec same-class re-export and unchanged deterministic fixtures;
- preserved XOR `round_like_typescript()` compatibility wrapper;
- no Transformer implementation, frontend change, request seed control, or dependency change;
- the exact pytest, Ruff, and mypy commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan012.md`;
- `012-centralize-typescript-compatible-randomness-and-rounding.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `py_llm_pipeline_explorer_file_structure(32).md`;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the real repository again, establish its own baseline, preserve user changes, implement only Ticket 012, create independent fixed utility evidence, verify completed Word2Vec and XOR behavior, report actual command results honestly, and create the implementation commit only after all required checks pass.
