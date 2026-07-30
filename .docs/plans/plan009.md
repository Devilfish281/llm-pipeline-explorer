---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "009"
source_work_item: 009-run-deterministic-reference-compatible-skip-gram-training.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(21).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 009: Run deterministic reference-compatible Skip-gram training

## Initial checklist

- Confirm Ticket 009 is the only selected work item and that its Ticket 007 blocker is satisfied by the current immutable preprocessing implementation and fixtures.
- Treat `py_llm_pipeline_explorer_file_structure(21).md` as the latest current-code authority.
- Use `llm_works_file_structure.md` only as the TypeScript Reference Implementation for deterministic random consumption, numerical formulas, operation order, and reporting behavior.
- Preserve the user-reported passing pytest, Ruff, and strict mypy baseline without claiming current-session verification.
- Limit production changes to reusable Word2Vec numerical training in `src/how_llms_work/ml/word2vec.py`.
- Add fixed independent numerical evidence; never derive expected values by calling the Python production operation under test.
- Finish with focused Word2Vec tests, affected-area regression tests, Ruff, strict mypy, and the complete pytest suite.

## Source-of-truth hierarchy

1. The user's latest explicit direction to convert the TypeScript behavior to Python and treat `py_llm_pipeline_explorer_file_structure(21).md` as the source of truth for the current Python Backend.
2. `009-run-deterministic-reference-compatible-skip-gram-training.md` for required behavior, acceptance criteria, approved test seam, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(21).md` for the current implementation, tests, paths, dependencies, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and ADR 0001 as recorded by the approved Phase 4 handoff for durable deterministic-compatibility decisions and canonical terminology.
5. `llm_works_file_structure.md`, especially TypeScript `src/routes/train-embed/train.ts`, as behavior evidence only.
6. Older exports, snippets, tickets, and plans are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 009 adds the reusable numerical core for one deterministic Embedding Training Run. The run must consume immutable preprocessing from Ticket 007, copy the selected ordered Training Pairs into request-owned mutable state, create one request-owned Mulberry32 generator seeded with `42`, initialize separate NumPy `float64` input and output matrices in the exact reference random-call order, construct the frequency-to-the-`0.75` negative-sampling distribution, and execute inclusive Skip-gram epochs with Mulberry32-driven Fisher–Yates shuffling.

Each Training Pair must perform its positive update before drawing negative candidates. Every requested negative candidate consumes exactly one random draw; a draw equal to the true context is skipped without replacement. Positive and effective negative updates must be immediate and coordinate ordered, with output coordinates using the saved pre-update input coordinate. The run must preserve the reference sigmoid clipping, loss formulas, `1e-10` protection, pair-count loss normalization, linear learning-rate schedule, report boundaries, and six-decimal public Embedding Epoch Updates.

The public Word2Vec seam must support bounded advancement and expose a completed numerical training state for Ticket 010 without constructing Query Word results, Nearest Neighbors, Similarity Pairs, Vector Analogies, a Saved Embedding Model, HTTP/SSE behavior, or persistence. Any non-finite weight, gradient, score, or loss must prevent a successful terminal training value.

## Baseline evidence

- **Status:** User-reported.
- **Command:** `poetry run pytest`
- **Result:** The user reports that all tests passed before planning.
- **Command:** `poetry run ruff check .`
- **Result:** The user reports that Ruff passed before planning.
- **Command:** `poetry run mypy src`
- **Result:** The user reports `Success: no issues found` before planning.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing. None of these commands were tool-verified in this planning session.

## Current code observations from the latest source

- `src/how_llms_work/ml/word2vec.py` currently implements immutable corpus preprocessing only. It defines the exact corpus, the requested 500-Merge limit, the verified natural 423-Merge result, ordered Vocabulary data, token indices, and immutable Training Pairs for window sizes 1 through 5.
- `Word2VecPreprocessing.training_pairs` stores immutable tuples. Ticket 009 therefore must create a mutable list owned by each Embedding Training Run before Fisher–Yates shuffling; it must not mutate the shared tuple or replace shared preprocessing.
- `get_word2vec_preprocessing()` exposes the stable public Word2Vec boundary and accepts no Query Word input, preserving the rule that Query Words do not alter training data.
- `tests/test_word2vec.py` and `tests/fixtures/word2vec_preprocessing_reference.json` already provide independent exact evidence for the corpus, Merge Table, tokenization, frequencies, Vocabulary, indices, every supported Training Pair sequence, immutability, and sequential/concurrent preprocessing equivalence.
- The current Word2Vec module has no Mulberry32 implementation, NumPy numerical state, negative-sampling distribution, weight initialization, shuffle operation, positive or negative update, loss calculation, learning-rate schedule, bounded training iterator, Embedding Epoch Update, completed numerical state, or non-finite enforcement.
- `src/how_llms_work/ml/neural_net.py` provides useful local prior art for a bounded iterator that advances to reporting boundaries, emits rounded epoch updates, and keeps mutable run state independent. Ticket 009 must preserve that architectural pattern without coupling Word2Vec behavior to XOR-specific types or `float32` calculations.
- `src/how_llms_work/routes/train_embed.py` already contains persistence-only behavior from Ticket 008. It is outside Ticket 009 and must not be connected to the numerical trainer in this work item.
- `pyproject.toml` already declares NumPy, pytest, Ruff, and strict mypy. No dependency or lockfile change is required.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - Ticket 007's fixed Embedding Training Corpus, ordered Vocabulary, token frequencies, indices, immutable preprocessing, and exact ordered Training Pairs for every supported window size.
  - Shared preprocessing cannot be mutated through its public boundary, and repeated/concurrent preprocessing calls are equivalent.
  - NumPy and the required test/quality tools are already declared.
  - The repository contains bounded Training Run and exact/tolerance test-pattern prior art in the XOR module.
- **Behavior present but evidence incomplete:** None for Ticket 009's numerical training behavior.
- **Partially implemented:**
  - The immutable inputs needed by training are complete, but no request-owned mutable numerical run consumes them yet.
  - A typed `SavedEmbeddingModel` contract exists for Ticket 008, but Ticket 009 must not construct or persist that model.
- **Not implemented:**
  - Request-owned Mulberry32 seed `42` with exact 32-bit wraparound and output sequence.
  - Reference-order `float64` input/output matrix initialization and matrix/run memory separation.
  - Negative-sampling cumulative distribution and deterministic candidate selection.
  - Mulberry32 Fisher–Yates shuffling of a run-owned Training Pair list.
  - Inclusive epochs, linear learning-rate schedule, and report schedule.
  - Positive-before-negative online coordinate updates, skip-without-redraw behavior, and saved pre-update input-coordinate use.
  - Sigmoid clipping, positive/negative loss formulas, `1e-10` safeguard, and positive-pair loss normalization.
  - Bounded Embedding Training Run advancement, rounded Embedding Epoch Updates, and a terminal completed numerical state.
  - Non-finite-state rejection.
  - Fixed independent numerical fixtures and sequential/concurrent mutable-state isolation tests.
- **Evidence limitation:**
  - The current repository contains no independent Ticket 009 fixture with Mulberry32 sequences, shuffles, cumulative distributions, initialized matrices, one-step updates, or short-run losses.
  - Exact unrounded expected values and the smallest useful `rtol` and `atol` must be captured or independently calculated during implementation from the TypeScript Reference Implementation; they must not be generated by the production Python helper under test.
  - ADR 0001 was not supplied as a standalone file in this handoff, but its binding decision is recorded consistently in Ticket 009, `SPEC.md`, and the approved Phase 4 handoff: deterministic random-call and operation order take priority over optimization, with exact rounded comparisons and tight tolerance-aware hidden checks.

## Files to inspect before editing

1. `src/how_llms_work/ml/word2vec.py` — `TrainingPair`, `Word2VecPreprocessing`, `get_word2vec_preprocessing()`, ordered Vocabulary/frequency data, and the destination for the new stable numerical training boundary.
2. `tests/test_word2vec.py` — existing independent-fixture loading, exact structure checks, mutation tests, and concurrent-call prior art that must remain passing.
3. `tests/fixtures/word2vec_preprocessing_reference.json` — existing preprocessing evidence; do not overwrite or repurpose it as a numerical fixture.
4. `src/how_llms_work/ml/neural_net.py` — `EpochUpdate`, `TrainingRun`, `create_training_run()`, reporting-boundary advancement, TypeScript-style rounding, and state-isolation prior art only.
5. `tests/test_neural_net.py` — exact-versus-tolerance assertions, deterministic first-update fixtures, report schedule checks, and returned-state isolation prior art only.
6. `src/how_llms_work/routes/train_embed.py` and `tests/test_train_embed_persistence.py` — existing Ticket 008 persistence boundary to preserve unchanged and keep disconnected from this numerical-only ticket.
7. `pyproject.toml` — Python 3.12, NumPy, pytest, Ruff, and strict mypy configuration; no dependency change is expected.
8. `009-run-deterministic-reference-compatible-skip-gram-training.md`, `SPEC.md`, and `CONTEXT.md` — approved behavior, comparison policy, terminology, and scope boundary.
9. TypeScript Reference Implementation `src/routes/train-embed/train.ts` in `llm_works_file_structure.md` — exact Mulberry32 arithmetic, random-call order, initialization, cumulative sampling, shuffle, positive/negative formulas, coordinate-update order, inclusive epoch loop, learning rate, loss normalization, and reporting behavior.

## Step 1 — Add independent deterministic fixtures and public-seam tests for random and initialization behavior

**Files and symbols:**
- `tests/fixtures/word2vec_training_reference.json` — new fixed evidence captured independently from the TypeScript Reference Implementation or an independent reference calculation.
- `tests/test_word2vec_training.py` — new tests through the stable public Word2Vec numerical boundary.
- `src/how_llms_work/ml/word2vec.py` — existing `TrainingPair`, `Word2VecPreprocessing`, and `get_word2vec_preprocessing()` plus the new public deterministic-randomness and training-state boundary whose exact symbol names remain an implementation choice under `SPEC.md`.

**Purpose:**
Establish non-circular evidence for the first deterministic layer before production numerical behavior is written. This covers the Mulberry32, matrix initialization, matrix separation, negative-sampling distribution, candidate selection, and Fisher–Yates acceptance criteria.

**Actions:**
- Create a dedicated numerical fixture rather than adding implementation-derived values to the preprocessing fixture.
- Record fixed Mulberry32 outputs that exercise ordinary progression and 32-bit wraparound, including enough consecutive values to detect altered constants, signed/unsigned shifts, multiplication wrapping, or division scaling.
- Record a small Vocabulary-frequency example with its exact frequency-to-the-`0.75` cumulative distribution and fixed random draws mapped to candidate indices.
- Record one small in-place Fisher–Yates shuffle result and the consumed random sequence.
- Record small initialized input/output matrices for a fixed Vocabulary size and dimensions, preserving the exact reference random-call order and formula `scale = 0.5 / dimensions`, `weight = (random - 0.5) × scale`.
- Add tests that assert NumPy `float64` dtype, exact shapes, distinct input/output buffers, and no memory sharing between matrices or separately created runs.
- Add tests proving production creation always owns a fresh seed-`42` generator and does not accept or call NumPy-native randomness.
- Keep exact fixture generation outside the Python production module and document the fixture source in the JSON.

**Guardrails:**
- Do not call the production Mulberry32, initializer, shuffle, sampler, or trainer while constructing expected fixture values.
- Do not expose a configurable production seed.
- Do not use `numpy.random`, Python `random`, or a third-party PRNG anywhere in the production Word2Vec path.
- Do not mutate `Word2VecPreprocessing.training_pairs` or any other shared preprocessing value.
- Do not implement training updates, result construction, routing, or persistence in this step.

**Expected result:**
- Focused tests define the exact deterministic primitive behavior and initially fail only because the public numerical boundary is not implemented.
- Independent fixture values are available for production implementation and future regression protection.

**Verification:**
- Run `poetry run pytest tests/test_word2vec_training.py -k "mulberry or initialization or distribution or shuffle"`.
- Confirm failures point to missing numerical behavior rather than fixture-loading, preprocessing, or import errors.

## Step 2 — Implement request-owned deterministic random, pair-order, matrix, and sampling state

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — existing `TrainingPair`, `Word2VecPreprocessing`, and `get_word2vec_preprocessing()` plus new stable public creation/inspection behavior for one Embedding Training Run.
- `tests/test_word2vec_training.py` — deterministic primitive, shape, dtype, memory, and run-ownership tests from Step 1.

**Purpose:**
Create the independent mutable state required before any gradient update can occur, while preserving immutable preprocessing and the TypeScript reference random-consumption sequence.

**Actions:**
- Add a typed Mulberry32 implementation that explicitly reproduces JavaScript 32-bit signed/unsigned coercion, wrapped multiplication, shifts, state addition, and `[0, 1)` output scaling.
- Add a stable public way to create one Embedding Training Run from `get_word2vec_preprocessing()`, selected `dimensions`, `window_size`, `epochs`, and `negative_samples`, with production seed fixed internally to `42`.
- Copy the selected immutable Training Pair tuple into a run-owned mutable list in the original order.
- Build the negative-sampling cumulative distribution once per run from ordered Vocabulary frequencies raised to `0.75`, preserving Vocabulary index order.
- Allocate separate two-dimensional NumPy `float64` input and output matrices with shape `(vocab_size, dimensions)`.
- Initialize every coordinate in the exact TypeScript random-call order, using one Mulberry32 draw per coordinate and no intervening random consumption.
- Keep generator state, mutable pair order, cumulative distribution, matrices, epoch cursor, report cursor, gradients, and losses owned by one run.
- Validate impossible internal configurations early enough to avoid divide-by-zero, empty-pair, invalid-dimension, or invalid-negative-sample behavior; do not broaden the public HTTP contract in this ticket.

**Guardrails:**
- Preserve the shared preprocessing object and its immutable nested values.
- Do not vectorize initialization if doing so obscures or changes random-call order.
- Do not share matrix views, pair lists, generator objects, or epoch/loss state between runs.
- Do not introduce a general matrix framework or edit `ml/matrix.py` or `ml/math_utils.py`.
- Do not construct public Word Embeddings or any Ticket 010 result.

**Expected result:**
- Every run begins from the same deterministic reference state for identical inputs while owning distinct mutable objects.
- Mulberry32, distribution, shuffle, initialization, dtype, shape, and memory-separation tests pass exactly.

**Verification:**
- Run `poetry run pytest tests/test_word2vec_training.py -k "mulberry or initialization or distribution or shuffle or separation"`.
- Confirm `np.shares_memory()` is false for input versus output matrices and across runs.

## Step 3 — Implement exact positive and negative online coordinate transitions with finite-state enforcement

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — new stable public numerical transition behavior or equivalent public Training Run state methods; exact private helper decomposition remains an implementation choice.
- `tests/test_word2vec_training.py` — fixed one-positive-update, one-negative-update, collision-skip, clipping, loss, and non-finite tests.
- `tests/fixtures/word2vec_training_reference.json` — independent pre-state and expected post-state values with explicit tolerances for unrounded results.

**Purpose:**
Implement the core Skip-gram with negative sampling mathematics in the precise operation order required by Deterministic Embedding Compatibility.

**Actions:**
- Add independent fixtures for one positive transition and one negative transition using small explicit matrices, target/context indices, learning rate, and expected score, gradient, loss, and post-update coordinates.
- Implement dot products by iterating dimensions in reference order.
- Apply the reference sigmoid clipping before exponentiation and preserve the positive score and negative score formulas.
- For a positive sample, calculate the score and gradient, then update input and output coordinates immediately and dimension by dimension.
- Save each pre-update input coordinate before changing it and use that saved value for the corresponding output-coordinate update.
- Add the positive loss as `-log(score + 1e-10)` after the positive update sequence.
- Draw each negative candidate only after the positive update is complete.
- Consume exactly `negative_samples` candidate draws. When a candidate equals the true context, skip its update and loss without drawing a replacement.
- For each effective negative candidate, calculate score and gradient from the current already-updated state, then apply immediate coordinate subtraction in reference order using the saved pre-update input coordinate.
- Add the negative loss as `-log(1 - score + 1e-10)`.
- Check scores, gradients, losses, and changed coordinates for finiteness at a deterministic boundary. Raise or otherwise mark the run failed before any successful terminal value can be produced.
- Use exact equality for discrete draw/order/skip behavior and the fixture's explicit tight `rtol`/`atol` for unrounded `float64` calculations.

**Guardrails:**
- Do not accumulate a vector gradient for later application.
- Do not replace coordinate loops with matrix-wide or batched operations.
- Do not draw negatives before the positive update.
- Do not redraw a true-context collision.
- Do not use the already-updated input coordinate to update the matching output coordinate.
- Do not add gradient clipping, subsampling, early stopping, hierarchical softmax, or a redesigned optimizer.
- Do not allow NaN or infinity to remain latent until JSON serialization.

**Expected result:**
- One positive update and one negative update match independently calculated unrounded states within explicit tight tolerances.
- Random consumption and skip-without-redraw behavior match exact fixtures.
- Injected or naturally encountered non-finite state prevents successful completion.

**Verification:**
- Run `poetry run pytest tests/test_word2vec_training.py -k "positive or negative or collision or clipping or finite"`.
- Confirm the tests detect a deliberate operation-order mutation such as using an updated input coordinate for the output update.

## Step 4 — Implement inclusive epoch advancement, learning-rate decay, loss normalization, and reporting

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — new bounded Embedding Training Run iterator/advancement boundary and Embedding Epoch Update representation; exact public symbol names remain an implementation choice.
- `tests/test_word2vec_training.py` — epoch, learning-rate, reporting, rounded-loss, bounded-advancement, and terminal-state tests.
- `tests/fixtures/word2vec_training_reference.json` — short complete run inputs, epoch updates, unrounded matrices/losses, and terminal-state evidence.

**Purpose:**
Compose deterministic primitives and one-pair transitions into a bounded Embedding Training Run suitable for later same-process thread advancement by the route ticket, without implementing HTTP or SSE orchestration now.

**Actions:**
- Process epochs inclusively from `0` through the requested final epoch.
- At the start of each epoch, calculate the reference linear learning rate from `0.025` to `0.001` at the confirmed epoch boundaries.
- Shuffle the run-owned Training Pair list in place once per epoch using the run-owned Mulberry32 generator.
- Process every shuffled Training Pair with the exact positive-then-negative sequence from Step 3.
- Divide epoch total loss by the number of positive Training Pairs, regardless of skipped or effective negative updates.
- Calculate `report_step = max(1, floor(epochs / 50))`.
- Emit or return an Embedding Epoch Update at epoch zero, every report boundary, and the requested final epoch, with loss rounded to six decimals using TypeScript-compatible rounding.
- Structure advancement so one call performs only the work needed to reach the next public report boundary or terminal completed-training value, following the bounded XOR prior art.
- After the final Epoch Update, expose exactly one completed numerical state suitable for Ticket 010. It may contain run-owned matrices and required metadata, but it must not construct selected embeddings, neighbors, similarities, analogies, warnings, a Saved Embedding Model, filesystem output, or an SSE payload.
- Prevent a terminal completed-training value when any finite-state check has failed.
- Ensure subsequent advancement after the one terminal value stops cleanly.

**Guardrails:**
- Do not emit an `init` or `done` SSE event in the numerical module.
- Do not sleep, offload to threads, check request disconnection, log route failures, or persist files.
- Do not round internal matrices or internal loss before the public six-decimal Epoch Update boundary.
- Do not omit epoch zero or the requested final epoch when it is not divisible by `report_step`.
- Do not normalize by positive-plus-negative example count.
- Do not mutate the immutable preprocessing Training Pair tuple while shuffling the run-owned list.

**Expected result:**
- A caller can advance one Embedding Training Run from report to report and then receive one completed finite numerical state.
- Inclusive epoch count, learning-rate endpoints, report boundaries, and rounded public losses match exact independent fixtures.

**Verification:**
- Run `poetry run pytest tests/test_word2vec_training.py -k "epoch or learning_rate or report or bounded or completion"`.
- Confirm the expected Epoch Update sequences include `0` and the exact requested final epoch for divisible and non-divisible report intervals.

## Step 5 — Prove complete short-run compatibility and sequential/concurrent isolation

**Files and symbols:**
- `tests/test_word2vec_training.py` — short complete run, repeated-run, concurrent-run, mutation-isolation, and no-shared-state tests.
- `tests/fixtures/word2vec_training_reference.json` — independent short-run exact rounded updates and tolerance-based unrounded terminal state.
- `src/how_llms_work/ml/word2vec.py` — completed public numerical boundary from Steps 2 through 4.

**Purpose:**
Verify the complete Ticket 009 vertical slice through the approved public Word2Vec module seam and prove that determinism does not rely on shared mutable state.

**Actions:**
- Add a small complete deterministic configuration that is fast enough for routine tests and captures every reported loss, final mutable Training Pair order, final generator progression evidence where publicly observable, and representative or complete final input/output matrices.
- Compare exact Epoch Update epochs and six-decimal losses to fixed fixtures.
- Compare unrounded terminal matrices and losses with the smallest verified explicit `rtol` and `atol` values that distinguish formula/order errors from cross-runtime transcendental differences.
- Run two identical configurations sequentially and assert exact observable equality while proving their generators, pair lists, matrices, gradients, losses, and completion objects are distinct mutable instances.
- Run identical configurations concurrently through `ThreadPoolExecutor` or equivalent test-only coordination and assert the same observable result without shared memory or ordering interference.
- Mutate one completed run's pair list or matrix after completion and prove another completed run and the shared preprocessing remain unchanged.
- Add a controlled non-finite case proving that a failed run does not expose a successful terminal state.
- Keep fixtures independent and include source/provenance metadata.

**Guardrails:**
- Do not make full default training part of every test.
- Do not assert universal bit-for-bit equality for all unrounded transcendental intermediates.
- Do not test private loop variables, helper identity, a specific iterator class design, or a particular thread-pool implementation.
- Do not use concurrency to share or synchronize production mutable state.
- Do not expand into Ticket 010 result construction or later HTTP/SSE orchestration.

**Expected result:**
- The complete short run is reference compatible at exact rounded boundaries and tightly compatible for unrounded `float64` state.
- Repeated and concurrent identical runs are deterministic because each run owns its mutable state, not because training is serialized or globally locked.

**Verification:**
- Run `poetry run pytest tests/test_word2vec_training.py`.
- Run `poetry run pytest tests/test_word2vec.py tests/test_word2vec_training.py`.

## Focused verification plan

```powershell
poetry run pytest tests/test_word2vec_training.py
poetry run pytest tests/test_word2vec.py tests/test_word2vec_training.py
```

Expected result:

- All deterministic primitive, numerical transition, bounded run, finite-state, preprocessing-regression, and isolation tests pass.
- Exact comparisons pass for random outputs, draw/order behavior, shuffle results, report epochs, and six-decimal public losses.
- Explicit tight tolerance comparisons pass for independently captured unrounded `float64` matrices, scores, gradients, and losses.

## Full verification plan

```powershell
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

Expected result:

- Ruff reports no issues.
- Strict mypy reports no issues in `src`.
- All tests pass.

## Manual acceptance checklist

- [ ] Ticket 007 remains satisfied: corpus-derived preprocessing is unchanged, immutable, and shared safely.
- [ ] Two identical Embedding Training Runs start with separate Mulberry32 objects, mutable Training Pair lists, and `float64` matrices.
- [ ] Input and output matrices have the requested shape and do not share memory.
- [ ] No production Word2Vec path imports or calls NumPy-native randomness or Python `random`.
- [ ] The random sequence is consumed only by initialization, per-epoch Fisher–Yates shuffling, and requested negative candidate draws in the confirmed order.
- [ ] Every Training Pair applies its positive update before any negative draw.
- [ ] A true-context negative collision is skipped without replacement.
- [ ] Coordinate updates use the saved pre-update input value for the corresponding output coordinate.
- [ ] Epochs run from `0` through the requested final epoch and use the `0.025` to `0.001` linear schedule.
- [ ] Reported loss is divided by positive Training Pair count and rounded to six decimals only at the public update boundary.
- [ ] Epoch zero and the requested final epoch are always present in the public update sequence.
- [ ] Non-finite state cannot produce a successful terminal numerical value.
- [ ] Sequential and concurrent identical runs produce the same observable updates and terminal numerical state without sharing mutable state.
- [ ] No Query Word selection, result ranking, analogy, route, SSE, persistence, frontend, Transformer, dependency, or general-framework change is present.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/word2vec.py
tests/test_word2vec_training.py
tests/fixtures/word2vec_training_reference.json
```

Conditionally changed only if a small import or shared fixture-loading adjustment is genuinely required:

```text
tests/test_word2vec.py
```

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/
tests/test_bpe.py
tests/test_bpe_tokenize.py
tests/test_neural_net.py
tests/test_neural_net_route.py
tests/test_neural_net_persistence.py
tests/test_train_embed_persistence.py
tests/fixtures/word2vec_preprocessing_reference.json
.data/
pyproject.toml
poetry.lock
frontend/
SPEC.md
CONTEXT.md
009-run-deterministic-reference-compatible-skip-gram-training.md
```

## Risk notes and safeguards

1. **Risk:** Python integer arithmetic fails to reproduce JavaScript's signed/unsigned 32-bit Mulberry32 behavior.
   - **Safeguard:** Implement explicit masking/coercion at every required operation and protect ordinary progression plus wraparound with fixed exact outputs.

2. **Risk:** A harmless extra random draw shifts every later matrix value, shuffle, negative candidate, and loss.
   - **Safeguard:** Separate exact fixtures for initialization, shuffle, candidate draws, and a complete short run; prohibit any production randomness outside the confirmed sequence.

3. **Risk:** NumPy vectorization changes operation order or creates temporary accumulated gradients.
   - **Safeguard:** Preserve scalar coordinate loops for dot products and immediate updates, even when a shorter vectorized expression exists.

4. **Risk:** The output update uses an already-updated input coordinate.
   - **Safeguard:** Store the pre-update input coordinate per dimension and verify a fixture designed to produce a detectably different result when this order is wrong.

5. **Risk:** A true-context negative collision is redrawn, changing random consumption and effective training examples.
   - **Safeguard:** Assert the exact draw count and following random output after a collision, not only the final matrix.

6. **Risk:** Epoch loss is normalized by all positive and negative examples rather than positive Training Pair count.
   - **Safeguard:** Use a fixture with effective negatives and a skipped collision whose expected denominator is unambiguous.

7. **Risk:** Shared immutable Training Pairs are converted to one module-global mutable list and shuffled across requests.
   - **Safeguard:** Copy the selected tuple for every run and test sequential/concurrent order and mutation isolation.

8. **Risk:** Input and output matrices or separate runs share memory through views or reused arrays.
   - **Safeguard:** Assert exact shapes/dtypes, `np.shares_memory(...) is False`, and mutation independence across all matrix pairs.

9. **Risk:** Public rounded losses match while hidden numerical state has drifted.
   - **Safeguard:** Combine exact public updates with tight independent unrounded matrix, score, gradient, and loss checks.

10. **Risk:** Tolerances are chosen too loosely and hide a formula or operation-order defect.
    - **Safeguard:** Measure TypeScript/Python differences on fixed representative cases and select the smallest useful explicit `rtol`/`atol`; include a mutation test or reviewed calculation showing the tolerance fails for known wrong ordering.

11. **Risk:** NaN or infinity survives until Ticket 010 or persistence and appears to be a completed run.
    - **Safeguard:** Check finite scores, gradients, losses, and changed coordinates during bounded advancement and prohibit terminal completion after failure.

12. **Risk:** The completed numerical state exposes or constructs future-ticket behavior.
    - **Safeguard:** Return only the data Ticket 010 needs to construct results later; do not select Query Words, round public vectors, rank neighbors, evaluate analogies, create a Saved Embedding Model, or serialize anything.

13. **Risk:** Full-size deterministic tests make the suite impractically slow.
    - **Safeguard:** Use independent small configurations for primitive and short-run tests, retaining the existing complete preprocessing fixtures without routinely training the full default model.

14. **Risk:** Ticket 009 expands into HTTP/SSE orchestration or modifies the completed persistence work.
    - **Safeguard:** Keep all likely production edits in `ml/word2vec.py`, prohibit route/schema/application edits, and inspect the final diff against the expected file list.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

A suitable outcome-oriented subject would describe deterministic reference-compatible Skip-gram training without claiming result construction, HTTP streaming, or persistence integration.

Commit body should mention:

- request-owned Mulberry32, pair order, `float64` matrices, negative-sampling distribution, epochs, gradients, and losses;
- exact positive-before-negative coordinate update order and collision behavior;
- bounded inclusive training, learning-rate and reporting schedules, finite-state enforcement, and completed numerical state;
- independent fixtures plus sequential/concurrent isolation coverage;
- the exact focused and full verification commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 009, Ticket 007, `SPEC.md`, `CONTEXT.md`, ADR 0001 as recorded in the approved handoff, `py_llm_pipeline_explorer_file_structure(21).md`, and the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 009, verify the complete change, report actual command results honestly, and create the implementation commit.
