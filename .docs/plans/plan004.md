---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "004"
source_work_item: 004-provide-reference-compatible-xor-training-runs.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(9).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 004: Provide reference-compatible XOR Training Runs

## Initial checklist

- Confirm Ticket 004 is the only work item in scope and has no blockers.
- Treat `py_llm_pipeline_explorer_file_structure(9).md` as the latest current-code authority.
- Use the TypeScript neural-network files only as the observable behavior reference.
- Limit the change to reusable numerical behavior and its focused Python tests.
- Reconfirm the user-reported pytest, Ruff, and mypy baselines before editing.
- Finish with focused neural-network tests, the full suite, Ruff, mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript XOR numerical behavior to Python, while treating the latest Python code export as the current-code source of truth.
2. `004-provide-reference-compatible-xor-training-runs.md` for required behavior, acceptance criteria, test seam, and scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(9).md` for the current Python implementation and repository conventions.
4. `SPEC.md` and `CONTEXT.md` for durable Phase 3 decisions and canonical domain language.
5. `llm_works_file_structure(4).md`, specifically the TypeScript `src/routes/neural-net/train.ts` and `serialize.ts` sections, as the behavior reference only.
6. Older snippets, plans, or assumptions are non-authoritative when they conflict with the sources above.

## Work-item summary

Implement the reusable XOR Neural Network Demo mathematics in `backend/src/how_llms_work/ml/neural_net.py` and prove it through a new public-module test file. The Python boundary must support independent Single-Layer and Multi-Layer Training Runs, preserve the TypeScript reference formulas and update order, use NumPy `float32` state, emit reference-compatible Epoch Updates, calculate ordered rounded XOR Predictions and exact Training Verdict strings, and return exact JSON-compatible Saved Weight Snapshot objects.

This ticket does not add the FastAPI request model, `/neural-net` router, SSE framing, delays, disconnect handling, worker-thread orchestration, filesystem persistence, or router registration. Those belong to later tickets.

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

- `backend/src/how_llms_work/ml/neural_net.py` exists but is empty; none of Ticket 004's numerical behavior is currently implemented.
- `backend/src/how_llms_work/routes/neural_net.py` also exists but is empty. It is explicitly outside this ticket and must remain unchanged.
- There is no `backend/tests/test_neural_net.py`; the current test suite covers Simple Chat and BPE only.
- `backend/src/how_llms_work/ml/bpe.py` demonstrates the current reusable-module style: focused module-level constants, typed public records, small pure helpers, deterministic behavior, and no framework coupling.
- `backend/tests/test_bpe.py` demonstrates the approved testing style: exact assertions for discrete contracts, focused public-module tests, parameterized cases, and request-state isolation checks.
- `backend/pyproject.toml` already declares NumPy and configures Python 3.12, pytest, Ruff, and strict mypy. Ticket 004 requires no dependency change.
- The TypeScript reference defines the fixed XOR order, sigmoid and derivative formulas, learning rate `1.0`, online per-example updates, mean-squared loss, epoch range `0..epochs`, approximately fifty reports, final predictions, exact labels and verdicts, and exact Saved Weight Snapshot keys.
- The TypeScript reference uses JavaScript `Math.round(value * scale) / scale`; Python's built-in rounding must not be assumed compatible at halfway values without focused tests.

## Acceptance criteria coverage

- **Already satisfied and evidenced:** None of the ticket's runtime or test acceptance criteria. The required NumPy dependency and strict mypy configuration are present as prerequisites only.
- **Behavior present but evidence incomplete:** None.
- **Partially implemented:** None.
- **Not implemented:** All Single-Layer and Multi-Layer state, initialization, training, reporting, prediction, verdict, snapshot-conversion, deterministic-seed, tolerance, isolation, and typing criteria.
- **Evidence limitation:** The exact deterministic seed, approved test epoch count, and explicit numerical tolerances cannot be selected until the Python reference formulas execute. They must be calibrated during implementation and then committed as fixed test data; exploratory seed-search code must not remain in production or the final tests.

## Files to inspect before editing

1. `backend/src/how_llms_work/ml/neural_net.py` — empty destination module; establish the public numerical boundary here.
2. `backend/tests/test_bpe.py` — public-module testing and isolation prior art.
3. `backend/src/how_llms_work/ml/bpe.py` — typing, constants, records, and module-organization prior art.
4. `backend/pyproject.toml` — existing NumPy dependency and strict pytest/Ruff/mypy configuration.
5. `llm_works_file_structure(4).md` — behavior-reference sections for `src/routes/neural-net/train.ts` and `src/routes/neural-net/serialize.ts`.
6. `SPEC.md`, `CONTEXT.md`, and `004-provide-reference-compatible-xor-training-runs.md` — exact architecture labels, verdicts, terminology, test seam, and out-of-scope rules.

## Step 1 — Establish the public numerical contract with focused failing tests

**Files and symbols:**
- `backend/tests/test_neural_net.py` — new public-module acceptance tests.
- `backend/src/how_llms_work/ml/neural_net.py` — new public Training Run boundary; exact symbol names remain an implementation choice under the ticket.

**Purpose:**
Lock the ticket's observable numerical contract before production implementation and create a route-independent seam that a later SSE route can advance one report at a time.

**Actions:**
- Add `backend/tests/test_neural_net.py` and import only intentionally public symbols from `how_llms_work.ml.neural_net`.
- Define tests around one synchronous public Training Run iterator or equivalent bounded state-machine boundary that:
  - accepts `single-layer` or `multi-layer` mode;
  - accepts an epoch count;
  - creates a fresh production `numpy.random.Generator` when none is supplied;
  - accepts an explicitly supplied `Generator` for deterministic tests;
  - yields Epoch Updates and then one final result containing predictions, verdict, and snapshot data.
- Add exact contract tests for the four XOR examples and targets in the required order.
- Add initialization tests for state shapes, `float32` dtypes, zero biases, independent samples in `[-1, 1)`, and fresh mutable state per Training Run.
- Exercise sigmoid and derivative behavior through the stable public numerical seam. Expose activation helpers only when they are deliberately part of that public seam; do not reach into private local variables merely to satisfy a test.
- Add initial reporting-boundary expectations for `epochs=100`, `epochs=101`, and `epochs=5000`.
- Confirm these tests fail because the public numerical behavior is not yet implemented.

**Guardrails:**
- Do not import FastAPI, Pydantic, SSE helpers, `asyncio`, filesystem APIs, or route modules into the numerical module or its tests.
- Do not freeze a private class, dataclass, local-variable, or helper identity in the tests.
- Do not add a seed to any HTTP schema; this ticket has no HTTP changes.
- A numerical-only `epochs=0` test case may be used to isolate the first complete XOR pass, but it must not alter the later HTTP minimum of `100`.

**Expected result:**
- The new test file precisely describes the public numerical contract and initially fails for missing implementation rather than unrelated setup problems.

**Verification:**
```powershell
poetry run pytest tests/test_neural_net.py -q
```
- Expected at this stage: focused failures identify the missing numerical API and behavior.

## Step 2 — Implement shared XOR data, typed result records, initialization, and rounding compatibility

**Files and symbols:**
- `backend/src/how_llms_work/ml/neural_net.py` — module constants, public result/state records, generator handling, initialization, and serialized-rounding boundary.
- `backend/tests/test_neural_net.py` — initialization, typing, range, order, and rounding cases.

**Purpose:**
Create the smallest typed foundation shared by both network modes and satisfy the acceptance criteria that do not depend on the training loops.

**Actions:**
- Define the four XOR inputs and targets once in exact `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]` order using NumPy-compatible `float32` data for calculations and ordinary integers/lists for public serialized predictions.
- Define fully typed public representations for:
  - an Epoch Update containing only `epoch` and `loss`;
  - an XOR Prediction containing `input`, `expected`, and rounded `actual`;
  - the final Training Run result containing architecture, predictions, verdict, and Saved Weight Snapshot data;
  - mode-specific numerical state when needed by the bounded Training Run design.
- Make every Training Run own its generator and numerical state. When no generator is supplied, create a new `numpy.random.default_rng()` for that run; when one is supplied, use only that generator.
- Initialize every weight independently from `[-1, 1)` and explicitly store numerical state as `float32`; initialize all biases to `float32` zero.
- Preserve the exact Single-Layer state of two weights plus one bias and Multi-Layer shapes `(2, 4)`, `(4,)`, `(4,)`, and one scalar output bias.
- Add a narrow serialized rounding operation that matches the TypeScript `Math.round(value * scale) / scale` contract for six-decimal losses and two-decimal predictions. First encode fixed halfway and ordinary-value tests; use Python's built-in `round` only if those tests prove it compatible for every required case.
- Keep snapshot conversion separate from internal NumPy state so returned snapshot objects contain only ordinary Python numbers and lists.

**Guardrails:**
- Do not use NumPy's global random state.
- Do not use `float64` arrays as the stored network state, even if intermediate NumPy operations naturally promote values.
- Do not add general matrix, optimizer, activation-registry, or model-framework abstractions.
- Do not add dependency or configuration changes.

**Expected result:**
- Public initialization creates correctly shaped, isolated `float32` state for each mode, generator injection is deterministic, and serialized rounding has explicit compatibility evidence.

**Verification:**
```powershell
poetry run pytest tests/test_neural_net.py -q -k "initial or dtype or shape or generator or rounding or xor_order"
```
- Expected result: the shared-contract and initialization subset passes.

## Step 3 — Implement the reference-compatible Single-Layer Training Run

**Files and symbols:**
- `backend/src/how_llms_work/ml/neural_net.py` — Single-Layer Mode initialization, one-epoch advancement, reporting, prediction, verdict, and snapshot conversion through the public boundary.
- `backend/tests/test_neural_net.py` — Single-Layer reference and educational-outcome tests.

**Purpose:**
Implement the `2 → 1` sigmoid network exactly enough to preserve the TypeScript learning procedure and deterministic educational failure.

**Actions:**
- Add a focused first-pass test using controlled initialization and a numerical-only short run to prove, through public outputs, the following order for each XOR example:
  1. calculate the sigmoid output from the current weights and bias;
  2. calculate `error = output - target`;
  3. accumulate squared error;
  4. calculate the sigmoid derivative from the already-computed output;
  5. immediately update both weights and the bias before processing the next example.
- Calculate epoch loss as total squared error divided by four.
- Run epochs inclusively from `0` through the requested epoch value.
- Produce Epoch Updates at `step = max(1, floor(epochs / 50))` boundaries and always at the final requested epoch, without duplicate final updates.
- Serialize loss to six decimal places only at the public Epoch Update boundary; retain unrounded numerical state for continued training.
- Build final predictions in the fixed XOR order, round each actual output to two decimals, and calculate success only from the rounded values using strict `< 0.1` comparisons.
- Preserve exactly:
  - `Single-Layer Perceptron (2 → 1)`;
  - `SUCCESS — network learned XOR`;
  - `FAILED — loss stuck, predictions are random guesses`.
- Convert the final state to a snapshot object with exactly `type`, `w1`, `w2`, and `bias` and ordinary Python scalar values.

**Guardrails:**
- Do not batch or shuffle examples.
- Do not vectorize into a batch-gradient update that changes immediate per-example behavior.
- Do not add early stopping, convergence forcing, alternate initialization, or a special case that guarantees failure.
- Do not include weights in the frontend-facing result shape later used by the route; keep the numerical result's snapshot data as a separate field or typed component that the route can remove before SSE serialization.

**Expected result:**
- Single-Layer Mode follows the reference update procedure, reports at the correct boundaries, returns exact labels and snapshot keys, and produces the required deterministic failure for the eventually selected seed.

**Verification:**
```powershell
poetry run pytest tests/test_neural_net.py -q -k "single_layer or reporting"
```
- Expected result: Single-Layer and shared reporting tests pass.

## Step 4 — Implement the reference-compatible Multi-Layer Training Run

**Files and symbols:**
- `backend/src/how_llms_work/ml/neural_net.py` — Multi-Layer Mode forward pass, backpropagation, updates, reporting, prediction, verdict, and snapshot conversion.
- `backend/tests/test_neural_net.py` — Multi-Layer one-pass numerical reference and educational-outcome tests.

**Purpose:**
Implement the `2 → 4 → 1` network while preserving the update order that is most likely to diverge during a TypeScript-to-Python conversion.

**Actions:**
- Add a controlled first-pass test that independently calculates the expected public state/result for one complete XOR pass and uses explicit `numpy.testing.assert_allclose()` tolerances for unrounded values.
- For each example, preserve this exact sequence:
  1. compute four hidden pre-activations from the current input-to-hidden weights and hidden biases;
  2. apply sigmoid to form hidden activations;
  3. compute the output pre-activation from the current hidden-to-output weights and output bias;
  4. apply sigmoid to form the prediction;
  5. calculate output error, squared loss, and output delta;
  6. calculate every hidden delta using the current, pre-update output weights;
  7. update hidden-to-output weights and output bias;
  8. update input-to-hidden weights and hidden biases;
  9. continue immediately to the next fixed-order XOR example.
- Reuse the shared inclusive epoch and report-boundary logic rather than duplicating a subtly different schedule.
- Build final predictions and success from rounded values using the same shared contract as Single-Layer Mode.
- Preserve exactly:
  - `Multi-Layer Network (2 → 4 → 1)`;
  - `SUCCESS — network learned XOR via backpropagation`;
  - `FAILED — network did not converge, try more epochs`.
- Convert final state to a snapshot object with exactly `type`, `w1`, `b1`, `w2`, and `b2`, using nested ordinary Python lists and scalars.

**Guardrails:**
- Hidden deltas must not use already-updated output weights.
- Do not replace the reference loops with batch matrix training when that would change floating-point or update order.
- Do not add a configurable hidden size, optimizer, learning rate, or activation.
- Keep every Training Run's arrays and result collections independent.

**Expected result:**
- Multi-Layer Mode matches the reference forward/backward procedure, reports correctly, returns exact labels and snapshot structure, and can reproduce a deterministic successful XOR run.

**Verification:**
```powershell
poetry run pytest tests/test_neural_net.py -q -k "multi_layer or backprop or reporting"
```
- Expected result: Multi-Layer, backpropagation-order, and shared reporting tests pass.

## Step 5 — Complete final-result, snapshot, reporting, and isolation coverage

**Files and symbols:**
- `backend/tests/test_neural_net.py` — full acceptance matrix across both modes.
- `backend/src/how_llms_work/ml/neural_net.py` — only the minimum corrections exposed by the completed acceptance tests.

**Purpose:**
Cover every discrete contract that is easy to break even when the core loss decreases.

**Actions:**
- Assert report epoch sequences exactly for:
  - `epochs=100`, producing `0, 2, 4, …, 100`;
  - `epochs=101`, producing regular step-2 boundaries plus the final `101`;
  - `epochs=5000`, producing `0, 100, 200, …, 5000`.
- Assert every Epoch Update exposes exactly the epoch and six-decimal loss values through the public record or serialized representation.
- Assert final prediction order, exact integer inputs/expected outputs, two-decimal actual values, strict success threshold, architecture strings, and all success/failure verdict strings.
- Assert the single-layer snapshot has exactly `type`, `w1`, `w2`, and `bias`; assert the multi-layer snapshot has exactly `type`, `w1`, `b1`, `w2`, and `b2`.
- Recursively assert snapshot values are JSON-compatible Python lists, integers/floats, and strings, with no NumPy arrays or NumPy scalar objects.
- Start separate Training Runs with separate controlled generators and prove they share no mutable weight arrays, epoch state, predictions, snapshots, or generator progress.
- Mutate a returned prediction or snapshot collection in one test instance and verify another completed run remains unchanged.

**Guardrails:**
- Do not test private helper names or the exact internal container type.
- Do not require bit-for-bit TypeScript floating-point equality.
- Use exact assertions only after rounding/serialization; use explicit tolerances for unrounded numerical values.
- Do not widen scope into route payloads, JSON file formatting, or filesystem writes.

**Expected result:**
- Every non-calibration acceptance criterion has direct public-boundary evidence for both modes.

**Verification:**
```powershell
poetry run pytest tests/test_neural_net.py -q
```
- Expected result: all focused numerical-module tests pass except any temporary seed-calibration placeholder that is intentionally completed in Step 6.

## Step 6 — Calibrate and freeze one deterministic educational seed and tolerances

**Files and symbols:**
- `backend/tests/test_neural_net.py` — fixed seed, approved epoch count, exact educational verdicts, and explicit tolerances.
- `backend/src/how_llms_work/ml/neural_net.py` — conditional corrections only if actual calibration reveals a reference-formula, dtype, update-order, or rounding gap.

**Purpose:**
Turn the specification's intentionally deferred seed and tolerance decisions into reproducible test evidence based on the completed Python implementation.

**Actions:**
- Use a temporary, uncommitted bounded calibration loop or temporary parameterization to evaluate candidate NumPy `Generator` seeds against both modes using the same approved epoch count.
- Select the first documented seed that reliably demonstrates:
  - Single-Layer Mode returns `FAILED — loss stuck, predictions are random guesses`;
  - Multi-Layer Mode returns `SUCCESS — network learned XOR via backpropagation`;
  - all four rounded Multi-Layer predictions differ from their expected targets by strictly less than `0.1`.
- Record the selected seed and epoch count as named test data with a short comment explaining that it proves the educational contrast and is not part of production or HTTP behavior.
- Remove all seed-search scaffolding from the final diff; the committed test must execute only the chosen seed.
- Derive narrow explicit `rtol` and `atol` values from actual `float32` comparisons for unrounded activations, losses, and weights. Record those tolerances in the relevant assertions rather than relying on NumPy defaults.
- Rerun the deterministic test repeatedly in the same process and in isolated test invocations to confirm reproducibility and state isolation.

**Guardrails:**
- Do not search indefinitely or add a production fallback that retries random initializations until convergence.
- Do not expose the seed in public request models or production defaults.
- Do not choose broad tolerances merely to make an incorrect update order pass.
- If no reasonable bounded seed demonstrates the required contrast, stop and inspect formula order, `float32` preservation, and rounding before changing the ticket's requirements.

**Expected result:**
- The final focused suite contains one fixed, reproducible educational seed and explicit numerical tolerances derived from actual Python execution.

**Verification:**
```powershell
poetry run pytest tests/test_neural_net.py -q -k "deterministic or educational or isolation"
poetry run pytest tests/test_neural_net.py -q -k "deterministic or educational or isolation"
```
- Expected result: both consecutive invocations pass with identical discrete outcomes.

## Step 7 — Run quality gates and inspect the final scope

**Files and symbols:**
- `backend/src/how_llms_work/ml/neural_net.py` — completed numerical implementation.
- `backend/tests/test_neural_net.py` — completed public-boundary tests.
- Existing backend source and tests — regression verification only.

**Purpose:**
Prove the ticket is complete, typed, lint-clean, regression-safe, and restricted to the approved numerical slice.

**Actions:**
- Run the complete focused neural-network tests.
- Run the full pytest suite once after all focused work passes.
- Run Ruff and strict mypy using the repository commands.
- Inspect the final diff and confirm only the two approved files changed.
- Confirm no HTTP schema, route, SSE, persistence, dependency, frontend, BPE, Simple Chat, Word2Vec, transformer, matrix, or math utility work entered the diff.
- Confirm no temporary seed-search code, debug output, generated files, cache files, or Saved Weight Snapshots are present.

**Guardrails:**
- Do not weaken strict mypy, exclude the module, add `Any` solely to silence errors, or suppress legitimate NumPy typing issues without a narrow justification.
- Do not fix unrelated failures by changing unrelated production files; report any independently existing failure separately.
- Do not claim success unless every command is actually executed and passes.

**Expected result:**
- Ticket 004 is implementation-ready for handoff with a minimal two-file change and complete public numerical evidence.

**Verification:**
```powershell
poetry run pytest tests/test_neural_net.py -q
poetry run pytest
poetry run ruff check .
poetry run mypy src
```
- Expected result: all commands pass.

## Focused verification plan

```powershell
poetry run pytest tests/test_neural_net.py -q
```

Expected result:

- Single-Layer and Multi-Layer public numerical tests pass, including initialization, `float32` state, update order, report boundaries, rounding, prediction order, verdicts, snapshots, deterministic educational outcomes, and request-state isolation.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues under `src`.

## Manual acceptance checklist

- [ ] `backend/src/how_llms_work/ml/neural_net.py` contains no FastAPI, SSE, filesystem, or presentation-delay logic.
- [ ] Production calls create a fresh NumPy `Generator`; tests can inject a generator without global random state.
- [ ] Single-Layer state is two `float32` weights plus one zero-initialized `float32` bias.
- [ ] Multi-Layer state has exact `(2, 4)`, `(4,)`, `(4,)`, and scalar `float32` shapes with zero biases.
- [ ] Every epoch uses the fixed XOR order and immediate per-example updates.
- [ ] Multi-Layer hidden deltas use pre-update output weights and output-layer updates occur before input-layer updates.
- [ ] Epochs run from zero through the requested value inclusively and reporting includes both zero and the final epoch.
- [ ] Loss and prediction rounding matches the TypeScript serialized contract.
- [ ] Prediction order, architecture labels, verdict strings, and strict success threshold are exact.
- [ ] Snapshot objects have only the required keys and ordinary JSON-compatible values.
- [ ] The fixed deterministic seed proves Single-Layer failure and Multi-Layer success.
- [ ] Separate Training Runs share no mutable state.
- [ ] No route, schema, SSE, persistence, frontend, dependency, or future-phase file changed.

## Expected files changed

Likely changed:

```text
backend/src/how_llms_work/ml/neural_net.py
backend/tests/test_neural_net.py
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
backend/src/how_llms_work/routes/neural_net.py
backend/src/how_llms_work/routes/simple_chat.py
backend/src/how_llms_work/routes/bpe_tokenize.py
backend/src/how_llms_work/ml/bpe.py
backend/src/how_llms_work/ml/math_utils.py
backend/src/how_llms_work/ml/matrix.py
backend/src/how_llms_work/ml/word2vec.py
backend/src/how_llms_work/ml/transformer.py
backend/src/how_llms_work/ml/transformer_worker.py
backend/tests/test_simple_chat.py
backend/tests/test_bpe.py
backend/tests/test_bpe_tokenize.py
backend/pyproject.toml
backend/poetry.lock
frontend/
```

## Risk notes and safeguards

1. **Risk:** Repeated NumPy operations may silently promote values to `float64`, changing the intended `float32` state and deterministic curve.
   - **Safeguard:** Assert dtypes after initialization and representative updates; explicitly preserve `float32` state at mutation boundaries.
2. **Risk:** Python's built-in rounding uses ties-to-even and may differ from JavaScript `Math.round` at halfway values.
   - **Safeguard:** Add fixed compatibility cases first and use one narrow serialized-rounding boundary for six- and two-decimal outputs.
3. **Risk:** Computing Multi-Layer hidden deltas after updating output weights produces a plausible but reference-incompatible algorithm.
   - **Safeguard:** Use an independently calculated one-pass public-state test that specifically distinguishes pre-update from post-update output weights.
4. **Risk:** Batch vectorization or sample shuffling changes online update order and learning behavior.
   - **Safeguard:** Keep the four-example loop explicit and test a controlled first pass plus final reference outputs.
5. **Risk:** A seed may be chosen that passes accidentally because tolerances are too broad or only one mode is checked.
   - **Safeguard:** Select one seed against both modes, assert exact verdicts and rounded predictions, and keep unrounded tolerances narrow and explicit.
6. **Risk:** Mutable arrays or returned nested lists may leak across Training Runs.
   - **Safeguard:** Create all state per call and add mutation-based isolation tests for internal state, predictions, and snapshot copies.
7. **Risk:** The public API may expose too many implementation details merely to make tests convenient.
   - **Safeguard:** Test observable Training Run transitions and outputs; expose activation or state helpers only when they are a deliberate reusable boundary for later route orchestration.
8. **Risk:** The implementation drifts into the route/persistence ticket because the TypeScript reference colocates those concerns.
   - **Safeguard:** Restrict this diff to `ml/neural_net.py` and `test_neural_net.py`; leave route, schema, SSE, filesystem, and registration files untouched.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Commit body should mention:

- reference-compatible Single-Layer and Multi-Layer XOR Training Runs;
- deterministic generator seam, exact result/snapshot contracts, and state isolation;
- verification with `poetry run pytest`, `poetry run ruff check .`, and `poetry run mypy src`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, `004-provide-reference-compatible-xor-training-runs.md`, `SPEC.md`, `CONTEXT.md`, `py_llm_pipeline_explorer_file_structure(9).md`, and `llm_works_file_structure(4).md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 004, verify the complete change, and create the implementation commit.
