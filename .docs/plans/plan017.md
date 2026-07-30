---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "017"
source_work_item: 017-advance-transformer-epochs-with-ordered-reduction-and-adam.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure.md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 017: Advance Transformer epochs with Ordered Gradient Reduction and Adam

## Initial checklist

- Confirm Ticket 017 is the only selected work item and that its Ticket 016 blocker is represented by the completed Transformer forward, backward, Training Sequence, and Logical Training Shard calculations in the latest Python Backend export.
- Treat the latest supplied `py_llm_pipeline_explorer_file_structure.md` export, created after Ticket 016, as the source of truth for current Python code; do not let older snippets, exports, plans, or assumptions override it.
- Preserve the existing canonical flat Transformer parameter layout and `LogicalTrainingShardResult` boundary instead of creating a second parameter order, gradient structure, or shard representation.
- Add only the parent-owned numerical state, Ordered Gradient Reduction, Adam transition, inclusive epoch progression, and report-schedule behavior required by Ticket 017.
- Create independent synthetic reduction and optimizer evidence before production implementation, including order-sensitive values, selected Adam coordinates, exact report schedules, and simulated one-through-four-worker completion orders.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Finish with focused Transformer training tests, existing Transformer regressions, Black checking for changed Python files, the complete pytest suite, Ruff, strict mypy, and a final scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript behavior to Python and treat the latest complete Python Backend source export as current-code truth.
2. `017-advance-transformer-epochs-with-ordered-reduction-and-adam.md` for immediate scope, acceptance criteria, approved test seam, blocker, constraints, and out-of-scope boundaries.
3. The latest supplied `py_llm_pipeline_explorer_file_structure.md` export, created `2026-07-28T02:37:24Z`, for the current implementation, tests, fixtures, dependencies, paths, typing style, and public Transformer boundary.
4. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for durable Phase 5 terminology, parent ownership, mixed precision, inclusive epoch, report, finite-state, and optimizer decisions.
5. The completed Ticket 016 implementation and tests in the current source, plus `plan016.md` only as historical implementation context where it agrees with the current source.
6. The latest `llm_works_file_structure.md`, especially the TypeScript `adamUpdateParam()`, `adamUpdate()`, and training-loop/reporting behavior, as formula and compatibility evidence only.
7. The original Adam paper and official NumPy documentation as technical cross-checks only; repository decisions and accepted Phase 5 documents remain authoritative.
8. Older Python exports, prior plans, pasted snippets, and earlier specification observations are non-authoritative when they conflict with the latest source export.

## Work-item summary

Ticket 017 adds the first parent-owned Transformer training state transition on top of the completed Ticket 016 mathematical boundary.

The current Python Backend can already:

- build the immutable Transformer preprocessing snapshot;
- construct exactly four deterministic Logical Training Shards;
- construct one canonical flat parameter layout and semantic views;
- initialize fresh finite `float32` weights in deterministic reference order;
- run forward and analytical backward passes;
- calculate one Training Sequence loss and complete canonical gradient;
- calculate one Logical Training Shard result in fixed Training Sequence order;
- return exact zero loss and an all-zero canonical gradient for an empty shard.

What is still missing is the parent-side operation that accepts the four shard results for one epoch, validates and canonicalizes them by shard ID, reduces their gradients in the required order, accumulates their losses in that same order, performs exactly one transactional Adam update, advances inclusive epoch state, and exposes exact report boundaries for later SSE orchestration.

The implementation must keep one parent-owned mutable model state per fresh Transformer Training Run:

- canonical flat `float32` weights;
- canonical flat `float32` first moments;
- canonical flat `float32` second moments;
- one reusable canonical flat `float32` Ordered Gradient Reduction workspace;
- exactly two reusable parent-local `float64` Adam scratch arrays;
- epoch cursor, requested final epoch, report schedule, and failure/completion state.

The two `float64` scratch arrays are calculation workspaces, not persisted model state, worker-visible state, SSE output, or Saved Transformer Model content. Completed first moments, second moments, and parameter candidates must be materialized as `float32`. The update must remain transactional: the parent must validate all four shard results, the completed reduced gradient and loss, all completed moment candidates, and the completed weight candidate before committing any new persistent weight or optimizer value.

The stable public seam should support both:

1. direct deterministic testing with synthetic `LogicalTrainingShardResult` values; and
2. later process orchestration feeding worker-produced results into the same parent epoch transition.

No operating-system process, shared-memory allocator, pipe protocol, request schema, FastAPI route, SSE event formatter, text sample, final evaluation, Saved Transformer Model, filesystem persistence, or frontend behavior belongs in this ticket.

## Baseline evidence

- **Status:** User-reported.
- **Commands reported by the user:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Reported result:** All pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, Black, mypy, numerical fixture generator, TypeScript runtime, browser, or backend command was executed while creating this plan.
- **Implementation rule:** `implement-prompt` must establish or reconfirm its own tool-verified baseline before editing and must report only command outcomes it actually observes.

## Current code observations from the latest source

### Completed foundations that Ticket 017 must reuse

- `src/how_llms_work/ml/transformer.py` is implemented through Ticket 016 rather than empty.
- The module already owns the complete canonical Transformer parameter layout and view boundary. Every future weight, moment, reduction, candidate, and gradient array must use that same flat length and order.
- `InitializedTransformerParameters` owns a finite C-contiguous `float32` flat storage array plus semantic views.
- `TransformerGradientBuffer` owns a canonical flat `float32` gradient storage array plus semantic views built from the same layout.
- `LogicalTrainingShardResult` already carries:
  - the `LogicalTrainingShard` metadata;
  - `processed_sequence_count`;
  - accumulated Python `float` loss;
  - one complete canonical `TransformerGradientBuffer`.
- `calculate_logical_training_shard()` already processes the shard’s Training Sequences in ascending fixed order, accumulates sequence losses without shard averaging, accumulates complete gradients without shard averaging, and returns an exact zero result for an empty shard.
- `build_logical_training_shards()` already creates exactly four ordered contiguous boundaries with shard indices `0`, `1`, `2`, and `3`.
- Existing forward/backward tests prove parameter purity, independent gradient storage, finite values, repeated-token accumulation, causal behavior, and empty-shard behavior.
- `src/how_llms_work/ml/math_utils.py` already exports the shared compatibility boundary:
  - `round_typescript_decimal_raw`;
  - `normalize_public_number`;
  - `round_typescript_decimal`;
  - `Mulberry32`.
- Ticket 017 should import and use `round_typescript_decimal(loss, 6)` for report-ready loss values. It must not duplicate the JavaScript rounding formula or use Python’s built-in `round()`.
- `tests/test_transformer.py` asserts the exact ordered contents of `transformer.py.__all__`. Any new stable public records or operations require a deliberate update that preserves every existing export.
- `tests/test_transformer_math.py` and `tests/fixtures/transformer_forward_backward_reference.json` cover the completed Ticket 016 seam. They should remain the forward/backward authority rather than being rewritten for optimizer behavior.
- `pyproject.toml` already supplies Python 3.12, NumPy, pytest, Ruff, Black, and strict mypy. No dependency or lockfile change is expected.

### Missing Ticket 017 behavior

- There is no parent-owned Transformer Training Run or equivalent parent state object.
- There are no first- or second-moment arrays.
- There is no reusable Ordered Gradient Reduction workspace.
- There are no two reusable parent-local `float64` Adam scratch arrays.
- There is no operation that validates a complete four-shard epoch result set.
- There is no reduction operation that canonicalizes arrival-order results and combines them strictly in shard order.
- There is no parent-side loss reduction across shards.
- There is no Adam update operation.
- There is no transactional candidate-and-commit boundary for weights and moments.
- There is no inclusive epoch cursor or exact requested-final-epoch behavior.
- There is no pure report-schedule boundary.
- There is no public Transformer Epoch Update containing exact epoch and six-decimal report loss.
- There is no failed-run lifecycle preventing later advancement after a numerical failure.
- There is no independent reduction/Adam/report fixture.
- There is no one-through-four simulated completion-order test.
- There is no test proving that fresh runs own non-overlapping weights, moments, reduction storage, Adam scratch storage, and report state.

### Current out-of-scope files

- `src/how_llms_work/ml/transformer_worker.py` remains the future owner of spawned worker records, shared-memory attachment, worker execution, and worker-group supervision. It must remain unchanged in Ticket 017.
- `src/how_llms_work/routes/train_transformer.py` remains the future HTTP/SSE orchestration owner. It must remain unchanged in Ticket 017.
- No Train Transformer request model or route integration is needed for the parent numerical seam.
- No Saved Transformer Model load path exists. Ticket 017 must not add one.

### TypeScript compatibility evidence

The current TypeScript reference updates each parameter coordinate using:

```text
m = beta1 × m + (1 - beta1) × gradient
v = beta2 × v + (1 - beta2) × gradient²
m_hat = m / (1 - beta1^step)
v_hat = v / (1 - beta2^step)
parameter = parameter - learning_rate × m_hat / (sqrt(v_hat) + epsilon)
```

The accepted Python contract intentionally applies this behavior over one canonical flat array rather than repeating the TypeScript semantic-object traversal. The canonical layout remains the sole order authority.

A precision-sensitive implementation detail must be protected: first and second moment candidates are completed and stored as `float32` before their values are promoted into the approved `float64` bias-correction and parameter-update stages. This matches the accepted mixed-precision contract and the behavior of TypeScript `Float32Array` moment buffers.

## Acceptance criteria coverage

### Already satisfied and evidenced

- Exactly four fixed Logical Training Shard boundaries exist with stable IDs `0..3`.
- Training Sequence order within each shard is fixed and ascending.
- Shard gradients are unaveraged.
- Shard losses are accumulated without averaging across Training Sequences.
- Empty shards return exact zero loss and complete all-zero canonical gradients.
- Canonical flat parameter and gradient order is implemented and independently tested.
- Fresh deterministic Weight Initialization is implemented without Saved Transformer Model input.
- Complete forward, backward, Training Sequence, and Logical Training Shard calculations are finite, isolated, and side-effect-free.
- The shared TypeScript-compatible decimal-rounding helper exists in `math_utils.py`.
- NumPy, pytest, Ruff, Black, and strict mypy are already configured.
- The approved public Transformer mathematical seam exists in `transformer.py`.

### Behavior present but evidence incomplete

- The parent-only ownership rule is established in the specification and ADR, but no parent numerical state currently enforces or proves it.
- Existing shard result records contain enough metadata to validate a four-shard epoch, but no current operation performs complete-set validation.
- Existing gradient buffers use canonical order, but no current test distinguishes canonical shard-order reduction from arrival-order reduction.
- Existing finite checks protect shard calculations, but there is no finite checkpoint across reduced gradients, losses, moments, or update candidates.
- Existing initialization tests prove independent weight arrays, but no optimizer-state or report-state isolation evidence exists.
- The TypeScript reference provides Adam formulas, but no independent Python fixture freezes selected mixed-precision moment and parameter outcomes.

### Partially implemented

- The complete inputs required by Ordered Gradient Reduction exist, but the parent reducer is absent.
- The complete inputs required by Adam exist, but optimizer state and update behavior are absent.
- Inclusive-loop and report-step rules are fixed in the specification and demonstrated by earlier Learning Demo patterns, but no Transformer-specific reporting boundary exists.
- Failure handling exists at lower mathematical boundaries, but no atomic parent epoch transition protects previously completed optimizer state.

### Not implemented

- Exact four-result validation and canonicalization.
- Ordered loss accumulation in shard order.
- One reusable `float32` reduction workspace.
- Unaveraged fixed-order cross-shard gradient reduction.
- Parent-owned first and second Adam moments.
- Exactly two reusable parent-local `float64` Adam scratch arrays.
- Transactional first-moment, second-moment, and weight candidates.
- Exact Adam constants and `epoch + 1` optimizer step.
- Inclusive epoch progression from zero through the requested final epoch.
- Exact report schedule and no-duplicate final report behavior.
- Public six-decimal Transformer Epoch Update loss.
- Failed-run state or equivalent rejection of later advancement after failure.
- Worker-completion-order independence across simulated one-through-four-worker schedules.
- Independent selected optimizer fixtures and explicit tight tolerances.
- Fresh-run non-aliasing across weights, moments, reduction workspace, scratch arrays, and report state.

### Evidence limitations

- The latest source is a complete code export rather than an executable repository checkout, so current code was inspected but not run in this planning session.
- Exact selected Adam fixture values and the smallest useful tolerances are not present in the supplied Python tests. They must be independently calculated or captured during implementation.
- Expected values must not be generated by importing or calling the new production reduction or Adam operations.
- The current Ticket 016 nonempty-shard test derives part of its expected gradient through production sequence calculations. That test remains useful Ticket 016 regression evidence but is not sufficient independent evidence for Ticket 017.
- Internal symbol names and record decomposition remain implementation choices except where a deliberately stable public seam is needed. Tests must target behavior rather than private optimizer helper names or a particular iterator implementation.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer.py`
   - current `__all__`;
   - architecture and Adam-adjacent constants;
   - `InitializedTransformerParameters`;
   - `TransformerParameterLayout`;
   - `TransformerParameterViews`;
   - `TransformerGradientBuffer`;
   - `LogicalTrainingShard`;
   - `LogicalTrainingShardResult`;
   - `build_transformer_parameter_views()`;
   - `create_transformer_gradient_buffer()`;
   - `initialize_transformer_parameters()`;
   - `build_logical_training_shards()`;
   - `calculate_logical_training_shard()`.
2. `src/how_llms_work/ml/math_utils.py`
   - `round_typescript_decimal()`;
   - `normalize_public_number()`;
   - exact public export and finite-value behavior.
3. `src/how_llms_work/ml/matrix.py`
   - inspect only for reusable validation and transactionality conventions;
   - do not move optimizer ownership into this stateless module.
4. `tests/test_transformer.py`
   - exact `__all__` expectation;
   - canonical layout and storage validation;
   - initialization ownership and no-saved-model signature test;
   - fixture and concurrency conventions.
5. `tests/test_transformer_math.py`
   - `LogicalTrainingShardResult` construction;
   - fixed-order shard accumulation;
   - empty-shard behavior;
   - dtype, contiguity, finite-state, purity, and isolation patterns.
6. `tests/fixtures/transformer_layout_initialization_reference.json`
   - canonical layout identity, selected initial coordinates, checksums, and fixture provenance.
7. `tests/fixtures/transformer_forward_backward_reference.json`
   - existing tolerance encoding, selected-coordinate style, and independent-evidence conventions.
8. `tests/test_math_utils.py` and `tests/fixtures/math_utils_reference.json`
   - exact TypeScript-compatible six-decimal rounding and positive-zero normalization evidence.
9. `tests/test_word2vec_training.py` and the XOR Training Run tests
   - prior art for inclusive epochs, bounded report intervals, run-owned mutable state, failure state, and sequential/concurrent isolation.
10. `pyproject.toml`
    - pytest, Ruff, Black, and strict-mypy settings;
    - confirm no package change is needed.
11. `017-advance-transformer-epochs-with-ordered-reduction-and-adam.md`
    - direct acceptance and scope authority.
12. `016-execute-reference-compatible-transformer-forward-and-backward-passes.md` or the completed current implementation
    - blocker behavior and approved shard seam.
13. `SPEC.md`, `CONTEXT.md`, and ADR 0002
    - canonical terms and binding numerical decisions.
14. `llm_works_file_structure.md`
    - `adamUpdateParam()`, `adamUpdate()`, report-step calculation, inclusive epoch evidence, and reference update intent.

## Step 1 — Reconfirm the implementation-session baseline and current scope

**Files and symbols:**

- `pyproject.toml`
- `src/how_llms_work/ml/transformer.py`
- `src/how_llms_work/ml/math_utils.py`
- `tests/test_transformer.py`
- `tests/test_transformer_math.py`
- current `tests/fixtures/` entries
- repository status and diff

**Purpose:**

Establish tool-verified pre-edit evidence and protect changes made after the supplied export.

**Actions:**

- Work from the backend project root.
- Inspect `git status --short` before editing.
- Confirm that `transformer.py` still contains the completed Ticket 016 public types and calculations.
- Confirm that `transformer_worker.py` and `routes/train_transformer.py` have not become current owners of parent epoch behavior in a newer live checkout.
- Confirm that `math_utils.py` still exports `round_typescript_decimal()`.
- Confirm that no `tests/test_transformer_training.py` or `transformer_training_reference.json` already supersedes this plan.
- Run the user-reported baseline commands before editing.
- Record exact commands, exit codes, and relevant output.
- Treat pre-existing failures as baseline evidence; do not silently repair unrelated code.

**Guardrails:**

- Do not describe the user-reported baseline as current-session verification.
- Do not modify files during this step.
- Do not regenerate the lockfile, format unrelated files, or remove user work.
- If the live repository materially differs from the latest export, update the implementation approach to the live code while preserving Ticket 017 scope.

**Expected result:**

- The Ticket 016 blocker is tool-confirmed in the live checkout.
- The smallest expected production change remains `transformer.py`.
- A clean and honest pre-edit baseline is recorded.

**Verification:**

```powershell
git status --short
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Step 2 — Create independent failure-first reduction, Adam, and reporting evidence

**Files and symbols:**

- New `tests/fixtures/transformer_training_reference.json`
- New `tests/test_transformer_training.py`
- Existing canonical layout fixture for offset identity
- TypeScript reference or a separate test-only scalar calculation with no production optimizer import

**Purpose:**

Freeze reviewable Ticket 017 expectations before adding the production reducer and optimizer.

**Actions:**

- Create one focused JSON fixture with explicit provenance stating that expected values were captured from the TypeScript reference or calculated by an independent scalar script that imports no production Transformer reduction, optimizer, epoch, or report helper.
- Use a one-layer canonical layout to keep the flat state deterministic and to reuse the completed layout identity.
- Keep the fixture compact by recording:
  - layout identity and total flat length;
  - initial selected weight coordinates;
  - four shard IDs and exact metadata;
  - selected nonzero gradient coordinates per shard;
  - shard losses;
  - ordered reduced selected gradient coordinates;
  - ordered total loss;
  - first-epoch selected first moments;
  - first-epoch selected second moments;
  - first-epoch selected updated weights;
  - second-epoch selected moments and weights where needed to prove step `2` bias correction;
  - explicit `rtol` and `atol` per hidden numerical boundary;
  - exact six-decimal report losses;
  - exact report epoch schedules for representative requested epoch counts.
- Choose at least one loss fixture whose floating-point result changes under a different summation order. This proves loss accumulation is `0 → 1 → 2 → 3`, not arrival order.
- Choose at least one gradient coordinate whose `float32` result changes under a different summation order. This proves the reduction workspace is used in canonical shard order.
- Include a coordinate where all four shard gradients are ordinary small values so the expected result clearly distinguishes an unaveraged sum from division by four.
- Include one empty shard with exact zero loss and an all-zero gradient so neutrality is explicit.
- Include positive, negative, and zero gradient coordinates so both Adam moments and update direction are covered.
- Include a second epoch with a different gradient pattern so the fixture distinguishes:
  - `epoch + 1` from `epoch`;
  - fresh moments from carried moments;
  - completed `float32` moment materialization from a pure all-`float64` implementation.
- Add exact schedule fixtures for at least:
  - requested epochs `0`;
  - `1`;
  - `49`;
  - `50`;
  - `51`;
  - `99`;
  - `100`;
  - `5000`.
- Include the expected default `5000` schedule `0, 100, 200, …, 5000`, totaling `51` unique reports.
- Define four representative arrival orders corresponding to simulated one-through-four physical workers. The exact scheduling simulation is test data only; production reduction must not accept or inspect worker count.
- Add failure-first tests through the planned stable public parent-side seam for:
  - missing shard;
  - duplicate shard;
  - shard ID below zero or above three;
  - wrong shard metadata;
  - wrong processed sequence count;
  - wrong gradient layout;
  - wrong gradient length;
  - wrong dtype;
  - non-C-contiguous gradient;
  - non-finite shard loss;
  - non-finite shard gradient;
  - reduced-gradient overflow or non-finite result;
  - invalid requested final epoch;
  - out-of-order or excess epoch advancement;
  - advancement after failure;
  - state-isolation failures.
- Keep exact structural assertions separate from tolerance-based hidden numerical assertions.
- Do not serialize complete full-size weight and moment arrays into the fixture. Store selected coordinates, exact structural metadata, and full-array checksums only where a checksum materially improves drift detection.

**Guardrails:**

- Do not call the new Python production reducer, Adam transition, schedule helper, or training state to generate expected values.
- Do not copy values from a production test result into the fixture without independent provenance.
- Do not use broad tolerances to hide order, dtype, bias-correction, or step errors.
- Do not start processes, allocate shared memory, test pipe timing, or create route mocks.
- Do not make tests depend on private helper names, a private loop variable, or a specific iterator class.

**Expected result:**

- Ticket 017 has independent failure-first evidence for every material discrete and numerical behavior.
- The tests fail only because the parent reduction/optimizer/report boundary is not yet implemented.

**Verification:**

```powershell
Get-Content tests\fixtures\transformer_training_reference.json |
    ConvertFrom-Json |
    Out-Null

poetry run pytest tests/test_transformer_training.py -q
```

Expected before implementation:

- The JSON fixture parses.
- Focused tests fail at the intended missing public boundary rather than from invalid test data.

## Step 3 — Define one parent-owned Transformer training state and stable public epoch boundary

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py`
- `tests/test_transformer_training.py`
- `tests/test_transformer.py` exact public export list

**Purpose:**

Create one clear owner for mutable weights, optimizer state, reduction storage, Adam scratch storage, epoch progression, report state, and failure state.

**Actions:**

- Define the smallest stable public parent-side contracts needed by later worker and route tickets. Exact class and function names remain an implementation choice, but the public seam must expose behavior for:
  - creating a fresh independent parent training state;
  - inspecting completed canonical weights and moments for deterministic tests;
  - accepting exactly one complete four-shard result set for the current epoch;
  - returning one completed epoch result;
  - identifying report boundaries and report-ready loss;
  - knowing whether the run is active, failed, or complete.
- Keep the public report-facing value deliberately small, such as a frozen slotted Transformer Epoch Update containing only:
  - `epoch`;
  - six-decimal `loss`.
- Keep the internal completed epoch loss separately available at Python floating-point precision for approved mathematical tests and later orchestration. Do not replace internal loss with the rounded public loss.
- Create fresh run-owned canonical flat `float32` arrays for:
  - weights;
  - first moments initialized to exact zero;
  - second moments initialized to exact zero;
  - Ordered Gradient Reduction workspace initialized to exact zero.
- Create exactly two run-owned reusable C-contiguous `float64` Adam scratch arrays with the canonical flat length.
- Do not include the scratch arrays in the public report object, Saved Transformer Model candidates, serialization records, worker startup data, or any future event-ready dictionary.
- Provide a stable way for the approved parent-side test seam to verify that exactly two `float64` scratch arrays exist without exposing them as mutable model output. Prefer immutable metadata or a narrow diagnostic property over returning writable scratch arrays.
- Ensure all arrays are exact one-dimensional canonical flat storage with expected dtype, C-contiguity, writable parent ownership, and finite initialization.
- Build semantic weight views only through the existing canonical layout/view builder.
- Do not build semantic moment or reduction offset tables separately. Flat canonical storage is sufficient unless existing view infrastructure can safely interpret those arrays without creating another authority.
- Validate the requested final epoch as an actual Python integer, not `bool`, and require a non-negative value. Request-layer maximum bounds belong to a later route ticket.
- Track the next expected epoch internally so each epoch is processed exactly once and in increasing order.
- A final epoch of `0` must still create one real epoch transition with Adam step `1`.
- Decide ownership explicitly:
  - either copy the supplied initialized flat weights into run-owned storage; or
  - accept ownership only through a constructor whose contract guarantees exclusive mutable storage.
- Prefer copying at the direct public factory unless the live repository already has a clear exclusive-ownership convention. Tests must prove that two fresh runs created from equivalent initialization do not share weight storage.
- Extend `transformer.py.__all__` only with deliberately stable public Ticket 017 records and operations. Preserve every completed export and its existing order unless the repository has an established grouped-order rule.

**Guardrails:**

- Do not expose a general configurable optimizer.
- Do not add learning-rate, beta, epsilon, shard-count, worker-count, batch-size, clipping, weight-decay, schedule, or resume parameters.
- Do not create a general training framework or base optimizer class.
- Do not store moments in `float64`.
- Do not store scratch arrays in dataclass equality, public serialization, or worker-visible records.
- Do not make a module-global mutable Training Run, reduction workspace, moment array, or scratch array.
- Do not read a Saved Transformer Model or `.data` file.
- Do not add a request seed or route-facing configuration.

**Expected result:**

- Every fresh run has one independent parent-owned numerical state.
- The public seam is sufficient for direct four-shard epoch tests and later process orchestration without exposing transient optimizer internals as model output.
- Existing preprocessing, layout, initialization, forward, backward, and shard APIs remain intact.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_training.py -q -k `
    "create or state or ownership or dtype or contiguous or zero or scratch or isolation"

poetry run pytest tests/test_transformer.py -q -k "public_contract"
```

## Step 4 — Implement strict four-shard validation and Ordered Gradient Reduction

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py`
- `tests/test_transformer_training.py`
- `tests/fixtures/transformer_training_reference.json`

**Purpose:**

Convert an arbitrary arrival-order collection of exactly four completed shard results into one deterministic unaveraged canonical gradient and one deterministic loss.

**Actions:**

- Accept a finite collection containing exactly four `LogicalTrainingShardResult` values for one epoch.
- Do not accept worker count, worker index, CPU count, assignment order, completion time, or process metadata at the reducer boundary.
- Validate the complete result set before reduction:
  - collection cardinality is exactly four;
  - every item is the expected public result type;
  - shard index is an actual integer and one of `0`, `1`, `2`, `3`;
  - every shard ID occurs exactly once;
  - each shard boundary matches the run’s expected immutable shard metadata;
  - `start_index <= stop_index`;
  - `processed_sequence_count` equals the represented range length;
  - empty ranges have exact zero processed count, zero loss, and all-zero gradient;
  - result loss is a finite Python numeric value;
  - gradient layout is exactly compatible with the run layout;
  - gradient storage is rank one, canonical length, `float32`, C-contiguous, finite, and independent of parent-owned weights, moments, reduction workspace, and Adam scratch storage.
- Canonicalize validated results by shard ID. Never rely on input collection order or sort by completion metadata.
- Zero the complete reusable `float32` reduction workspace immediately before each reduction.
- Accumulate shard losses separately into one Python `float` in the exact order `0`, `1`, `2`, `3`.
- Check loss finiteness after each completed addition and again after the full sum.
- Add each canonical gradient into the reusable reduction workspace in the exact order `0`, `1`, `2`, `3`, preserving `float32` materialization at every completed workspace stage.
- Use an explicit floating-point error boundary so overflow or invalid arithmetic cannot silently produce a successful reduced gradient.
- Check the complete workspace for finiteness after each shard addition and after the full reduction.
- Do not divide the workspace or loss by:
  - four shards;
  - processed sequence count;
  - total Training Sequence count;
  - actual worker count;
  - any batch size.
- Ensure an empty shard’s zero state is exactly neutral.
- Return or expose one stable completed reduction observation containing the ordered loss and canonical reduced gradient needed by the epoch transition. Do not allocate a second persistent reduced-gradient array when the run-owned workspace itself is the approved reusable reduction storage.
- If validation fails before reduction, leave weights and moments untouched.
- If reduction fails after the workspace was zeroed or partially accumulated, treat the workspace as invalid scratch state, fail the run or transition, and leave previously completed weights and moments untouched.

**Guardrails:**

- Do not reduce in arrival order.
- Do not use a dictionary iteration order as an implicit numerical contract; explicitly traverse shard IDs `0..3`.
- Do not use `sum()` over gradient arrays.
- Do not use `np.stack(...).sum(...)`, tree reduction, pairwise reduction, multiprocessing reduction, or a wider temporary reduction whose operation order differs.
- Do not create one workspace per worker or shard.
- Do not average gradients.
- Do not mutate any shard gradient.
- Do not use the Adam `float64` scratch arrays as the reduction workspace.
- Do not allow malformed empty-shard metadata to pass merely because its gradient is zero.

**Expected result:**

- Every valid arrival order produces the same exact ordered loss and reduced `float32` gradient.
- Order-sensitive fixture coordinates prove canonical shard order rather than accidental associativity.
- Empty shards are neutral.
- Invalid or non-finite result sets cannot reach Adam.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_training.py -q -k `
    "shard and (validate or missing or duplicate or range or malformed or finite)"

poetry run pytest tests/test_transformer_training.py -q -k `
    "ordered_reduction or arrival_order or worker_count or unaveraged or empty_shard"
```

## Step 5 — Implement transactional mixed-precision Adam with exactly two reusable float64 scratches

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py`
- `tests/test_transformer_training.py`
- `tests/fixtures/transformer_training_reference.json`

**Purpose:**

Apply one deterministic parent-owned Adam update while preserving completed state on every failure.

**Actions:**

- Keep exact fixed constants at the Transformer boundary:
  - learning rate `0.001`;
  - beta1 `0.9`;
  - beta2 `0.999`;
  - epsilon `1e-8`.
- Use optimizer step `epoch + 1`; epoch `0` therefore uses step `1`.
- Reject an invalid epoch or step before calculation.
- Validate current weights, first moments, second moments, and reduced gradient for:
  - canonical length;
  - expected dtype;
  - C-contiguity;
  - finiteness;
  - non-overlap where distinct ownership is required.
- Calculate the complete first-moment candidate using the first reusable `float64` scratch array.
- Materialize the complete first-moment candidate once as a separate C-contiguous `float32` candidate and validate it fully.
- Calculate the complete second-moment candidate using the second reusable `float64` scratch array.
- Materialize the complete second-moment candidate once as a separate C-contiguous `float32` candidate and validate it fully.
- Preserve the accepted mixed-precision staging:
  - bias correction and parameter update consume the completed `float32` moment candidates, promoted into the reusable `float64` scratches;
  - do not continue using unmaterialized higher-precision moment intermediates.
- Calculate bias-correction denominators exactly from `step`:
  - `1 - beta1**step`;
  - `1 - beta2**step`.
- Validate both denominators as finite and strictly positive.
- Reuse the same two `float64` scratch arrays to calculate:
  - bias-corrected first moments;
  - bias-corrected second moments;
  - square-root denominator plus epsilon;
  - complete parameter delta;
  - complete parameter candidate.
- Materialize the complete parameter candidate once as a separate C-contiguous `float32` candidate and validate it fully.
- Check all candidate arrays for finiteness before committing any persistent state.
- Commit first moments, second moments, and weights only after every complete candidate has passed validation.
- Use exact-size final copies into existing parent-owned arrays so object ownership and semantic views remain stable across epochs.
- Do not mutate the reduced-gradient workspace during Adam.
- Preserve the previously completed weight and moment bytes if:
  - a current input is non-finite;
  - a moment candidate overflows;
  - a bias-correction stage is invalid;
  - a square root or denominator is invalid;
  - a parameter candidate overflows;
  - candidate materialization is non-finite;
  - any validation fails.
- If the enclosing Training Run has lifecycle state, mark it failed after such a failure and reject later advancement. The atomic state-preservation rule remains mandatory even when failure status is recorded.
- Add a controlled failure seam that proves transactionality without making tests depend on private optimizer helper names. Prefer injecting non-finite public input state or a deliberately extreme finite fixture over monkeypatching private locals.
- Assert that no more than two reusable `float64` arrays are owned by the run for Adam calculation and that their identities remain stable across multiple epochs.
- Assert that moment and weight array identities also remain stable across successful commits.
- Compare selected coordinates after step `1` and step `2` against independent fixtures with named tight tolerances.

**Guardrails:**

- Do not apply Adam once per shard; apply it exactly once after the complete four-shard reduction.
- Do not use step `epoch`, a zero-based bias correction, or a global update counter unrelated to the run’s epoch cursor.
- Do not average or clip the gradient.
- Do not add weight decay, AdamW, AMSGrad, warmup, learning-rate decay, gradient scaling, or epsilon configuration.
- Do not keep moments or weights as `float64`.
- Do not allocate a new pair of `float64` scratch arrays per parameter group or per epoch.
- Do not mutate current moments before the weight candidate is known to be valid.
- Do not commit coordinates incrementally.
- Do not traverse a second semantic parameter-key table; operate over the one canonical flat order.

**Expected result:**

- One successful epoch performs one exact deterministic Adam update.
- Step `1` and step `2` selected moments and weights match independent evidence.
- All persisted optimizer and model state is `float32`.
- Exactly two reusable parent-local `float64` scratch arrays support the update.
- Every failure preserves the last completed weights and moments byte-for-byte.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_training.py -q -k `
    "adam or moment or bias_correction or step or float32 or scratch"

poetry run pytest tests/test_transformer_training.py -q -k `
    "transaction or preserve or overflow or nonfinite or failed"
```

## Step 6 — Implement one atomic epoch transition and inclusive Training Run progression

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py`
- `tests/test_transformer_training.py`

**Purpose:**

Combine validation, Ordered Gradient Reduction, Adam, epoch progression, and lifecycle state into one parent-owned transition that later worker orchestration can call safely.

**Actions:**

- Define one stable public operation that advances the current epoch using one complete four-shard result set.
- The operation must derive the current epoch from run-owned state or validate an explicit epoch against the run-owned expected cursor. It must not permit:
  - skipping an epoch;
  - repeating an epoch;
  - processing epochs after the requested final epoch;
  - processing after failure;
  - processing after completion.
- For each epoch, perform the stages in this order:
  1. validate active run state and current epoch;
  2. validate and canonicalize all four shard results;
  3. perform Ordered Gradient Reduction;
  4. validate reduced loss and gradient;
  5. calculate all Adam candidates with step `epoch + 1`;
  6. validate all candidates;
  7. atomically commit weights and moments;
  8. record the completed internal loss;
  9. produce a report update only when the epoch is a report boundary;
  10. increment the epoch cursor or mark the run complete.
- Treat the reduced shard loss as the loss associated with the gradients used for that epoch’s update.
- Do not round the loss before reduction, Adam, failure checks, or internal completion storage.
- If the transition fails before commit, retain the prior completed weights, moments, epoch cursor, and report history.
- If a failure status is recorded, record no successful report for that epoch and prevent later sampling, evaluation, persistence, or advancement.
- Process the requested range inclusively:
  - first epoch is `0`;
  - final epoch is exactly the requested value;
  - total Adam update count is `requested_epochs + 1`.
- Keep the low-level epoch transition independent of how shard results were produced. It must accept:
  - direct `calculate_logical_training_shard()` results;
  - synthetic fixture results;
  - future validated worker results.
- Add one focused integration test using the existing direct shard calculation seam on a small controlled Training Sequence collection. Keep the expensive full corpus out of ordinary Ticket 017 tests.
- Add a controlled multi-epoch synthetic run that records every requested epoch and proves:
  - exact inclusive count;
  - one reduction and one update per epoch;
  - carried moments;
  - no sequence shuffling input;
  - no worker-count input;
  - exact completion state.

**Guardrails:**

- Do not calculate all epochs in one uninterruptible route-facing method; later orchestration must retain a boundary between completed epochs and reports.
- Do not add asynchronous code, threads, processes, pipes, shared memory, deadlines, disconnect checks, or presentation sleeps.
- Do not generate text at report boundaries.
- Do not run final evaluation.
- Do not persist intermediate or final state.
- Do not expose raw gradients, moments, scratch arrays, or unrounded loss in a future SSE-ready report object.
- Do not reload weights from disk.

**Expected result:**

- A fresh parent training state advances deterministically from epoch `0` through the exact requested final epoch.
- Each epoch has one four-shard reduction and one Adam commit.
- Failure leaves the last successful epoch state intact and prevents later successful work.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_training.py -q -k `
    "epoch_transition or inclusive or epoch_zero or final_epoch or update_count or lifecycle"
```

## Step 7 — Implement the exact report schedule and six-decimal Transformer Epoch Updates

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py`
- `src/how_llms_work/ml/math_utils.py` as an unchanged dependency
- `tests/test_transformer_training.py`
- `tests/fixtures/transformer_training_reference.json`

**Purpose:**

Expose deterministic report boundaries for later SSE orchestration without adding HTTP or sample generation.

**Actions:**

- Implement one pure stable report-schedule boundary using:

  ```text
  report_step = max(1, floor(requested_epochs / 50))
  ```

- Since requested epochs are non-negative integers, use exact integer division semantics equivalent to `floor`.
- Include an epoch when:
  - `epoch % report_step == 0`; or
  - `epoch == requested_epochs`.
- Traverse each epoch only once, so a final epoch that is already divisible appears once rather than being appended twice.
- Include epoch `0` for every valid request.
- For requested epochs `0`, return exactly one report epoch: `0`.
- For requested epochs `5000`, return exactly `51` report epochs:
  - `0`;
  - every `100`;
  - exact final `5000`.
- Produce a small frozen public Transformer Epoch Update with:
  - exact integer `epoch`;
  - `loss` calculated through `round_typescript_decimal(internal_loss, 6)`.
- Preserve positive public zero normalization through the shared helper.
- Reject a non-finite internal loss before report construction.
- Keep the internal unrounded loss in the completed epoch observation or parent state only; do not overwrite it with the public rounded value.
- Do not include sample text, duration, learning rate, worker count, shard count, gradient norm, moments, weights, or process metadata.
- Add exact fixture comparisons for all representative schedules and six-decimal losses.
- Add a case where the requested final epoch is not divisible by `report_step`, proving the exact final epoch is included once.
- Add a case where the final epoch is divisible, proving no duplicate.
- Add a small negative-zero rounding case to prove the public report contains ordinary positive `0.0`.

**Guardrails:**

- Do not use Python built-in `round()`.
- Do not create SSE text or event names.
- Do not sleep after reports.
- Do not generate a sample.
- Do not make report scheduling depend on actual worker count or completion order.
- Do not round internal loss before finite checks.

**Expected result:**

- Report epochs match exact fixtures.
- Public losses match six-decimal TypeScript-compatible fixtures exactly.
- Internal loss remains available at approved hidden precision.
- Later route work can consume the stable updates without reimplementing scheduling.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_training.py -q -k `
    "report or schedule or final or duplicate or rounding or public_loss"
```

## Step 8 — Prove completion-order independence, finite-state enforcement, and fresh-run isolation

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py`
- `tests/test_transformer_training.py`
- `tests/fixtures/transformer_training_reference.json`

**Purpose:**

Close the deterministic and ownership guarantees that make later multiprocessing safe.

**Actions:**

- Feed the same four validated shard results in each independently defined simulated arrival order for one-through-four physical workers.
- For every simulation, start from byte-identical but non-aliasing fresh parent state.
- Assert identical:
  - ordered reduced loss;
  - reduced gradient;
  - first moments;
  - second moments;
  - updated weights;
  - completed epoch;
  - report epochs;
  - rounded report losses.
- Use exact equality for:
  - shard IDs and metadata;
  - report epoch tuples;
  - public six-decimal losses;
  - dtypes, shapes, contiguity, and array ownership;
  - all-zero empty-shard coordinates.
- Use explicit tight `assert_allclose()` tolerances for:
  - selected unrounded moments;
  - selected updated weights;
  - any direct-shard integration loss whose accepted NumPy path is tolerance-based.
- Inject non-finite values independently into:
  - shard loss;
  - shard gradient;
  - current weight;
  - first moment;
  - second moment;
  - a completed moment candidate;
  - a completed weight candidate.
- For each failure, record and compare before/after:
  - weight bytes;
  - first-moment bytes;
  - second-moment bytes;
  - epoch cursor;
  - completed report history.
- Assert no successful epoch update is returned from a failed transition.
- Assert a failed Training Run rejects later advancement.
- Create at least two same-configuration fresh runs and prove no shared memory between any mutable arrays:
  - weights;
  - first moments;
  - second moments;
  - reduction workspaces;
  - both Adam scratch arrays;
  - candidate arrays if retained;
  - report-history containers.
- Run equivalent fresh runs sequentially and through a small `ThreadPoolExecutor` using synthetic shard fixtures. Assert deterministic equal values and no cross-run aliasing.
- Mutate one completed run after comparison and prove another run is unchanged.
- Assert no constructor or factory accepts:
  - Saved Transformer Model;
  - path;
  - file handle;
  - resume flag;
  - checkpoint;
  - worker count;
  - process group.
- Search the production diff for forbidden behavior:
  - shuffling;
  - clipping;
  - weight decay;
  - learning-rate schedule;
  - early stopping;
  - averaging;
  - persistence;
  - sampling;
  - multiprocessing.

**Guardrails:**

- Do not assert universal bit-for-bit equality for every hidden floating-point intermediate where the specification permits tolerance-based comparison.
- Do not use real processes for this ticket.
- Do not make physical worker simulation part of production logic.
- Do not treat scratch-workspace mutation after failure as model-state corruption; the mandatory preserved state is the previously completed weights, moments, epoch cursor, and successful report state.
- Do not expose scratch arrays through serialization merely to test isolation.

**Expected result:**

- Worker completion order and simulated worker count affect neither numerical state nor reporting.
- Every non-finite boundary fails before successful completion.
- Fresh runs are deterministic but own completely independent mutable state.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_training.py -q -k `
    "worker or completion_order or deterministic or isolation or concurrent"

poetry run pytest tests/test_transformer_training.py -q -k `
    "finite or transaction or failed or preserve or no_saved_model"
```

## Step 9 — Run focused regressions, full verification, and scope inspection

**Files and symbols:**

- all changed Ticket 017 files
- completed Transformer, matrix, and deterministic utility tests
- repository status and diff

**Purpose:**

Prove Ticket 017 is complete without regressing prior phases or expanding into future tickets.

**Actions:**

- Run the complete new parent-side Transformer training test module.
- Run existing Transformer preprocessing/layout/initialization and forward/backward tests.
- Run matrix and deterministic utility tests because the new code relies on their contracts.
- Run Black in check mode only over changed Python files first.
- Run the complete pytest suite.
- Run Ruff and strict mypy over the configured project.
- Run `git diff --check`.
- Inspect `git status --short` and `git diff --stat`.
- Review the final diff against the expected-file list.
- Confirm no generated `.data`, cache, temporary fixture generator, compiled file, dependency, lockfile, route, worker, or frontend change is present.
- Record exact command results honestly.
- Create the implementation commit only in `implement-prompt`, after every required check passes.

**Guardrails:**

- Do not weaken or delete existing tests to make the new suite pass.
- Do not update existing independent fixtures unless implementation evidence proves an existing fixture defect unrelated to new expected values; such a defect must be reported separately.
- Do not run a broad formatter that creates unrelated churn.
- Do not claim a manual process, browser, worker, route, or persistence test was performed in this ticket.

**Expected result:**

- Ticket 017 focused tests pass.
- Completed Transformer and utility regressions pass.
- The full suite, Ruff, Black check, strict mypy, and diff checks pass.
- The final diff contains only the smallest complete Ticket 017 change.

## Focused verification plan

Run from:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"
```

Then run:

```powershell
poetry run pytest tests/test_transformer_training.py -q

poetry run pytest `
    tests/test_transformer.py `
    tests/test_transformer_math.py `
    tests/test_matrix.py `
    tests/test_math_utils.py `
    -q
```

Expected result:

- Four-shard validation, Ordered Gradient Reduction, unaveraged gradients, empty-shard neutrality, Adam steps, transactionality, inclusive epochs, exact reports, finite-state enforcement, completion-order independence, and fresh-run isolation pass.
- Existing preprocessing, layout, initialization, forward/backward, shard, matrix, and rounding behavior remains unchanged.

Run formatting checks for the expected changed Python files:

```powershell
poetry run black --check `
    src/how_llms_work/ml/transformer.py `
    tests/test_transformer.py `
    tests/test_transformer_training.py
```

If implementation does not change one of the listed files, remove that file from the command rather than creating a formatting-only edit.

## Full verification plan

Run:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
git diff --check
git status --short
git diff --stat
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.
- `git diff --check` reports no whitespace errors.
- Repository status contains only intended Ticket 017 source, test, and fixture changes.
- No command result should be described as successful until `implement-prompt` actually observes it.

## Manual acceptance checklist

- [ ] Ticket 016’s completed forward, backward, sequence, and shard APIs remain intact.
- [ ] Exactly four shard results are required.
- [ ] Shard IDs are exactly `0`, `1`, `2`, and `3`.
- [ ] Missing, duplicate, out-of-range, malformed, layout-incompatible, and non-finite shard results are rejected.
- [ ] Shard results are canonicalized by shard ID rather than trusted in arrival order.
- [ ] Shard losses are accumulated as Python floats in exact order `0 → 1 → 2 → 3`.
- [ ] One reusable canonical flat `float32` reduction workspace is used.
- [ ] Gradient addition occurs in exact order `0 → 1 → 2 → 3`.
- [ ] The reduced gradient is not averaged.
- [ ] The reduced loss is not averaged.
- [ ] Empty-shard zero state is exactly neutral.
- [ ] Shard gradients remain unchanged.
- [ ] Parent weights, moments, and workspaces do not alias shard gradient inputs.
- [ ] The parent is the only owner that mutates weights and optimizer state.
- [ ] First moments are persistent C-contiguous `float32`.
- [ ] Second moments are persistent C-contiguous `float32`.
- [ ] Weights remain persistent C-contiguous `float32`.
- [ ] Exactly two reusable parent-local C-contiguous `float64` Adam scratch arrays exist.
- [ ] No Adam scratch array appears in a public report, worker record, model, or serialized object.
- [ ] Adam learning rate is exactly `0.001`.
- [ ] Adam beta1 is exactly `0.9`.
- [ ] Adam beta2 is exactly `0.999`.
- [ ] Adam epsilon is exactly `1e-8`.
- [ ] Optimizer step is exactly `epoch + 1`.
- [ ] Epoch `0` performs Adam step `1`.
- [ ] First and second moment candidates are completed as `float32` before bias-corrected update stages consume them.
- [ ] Bias correction uses the current completed moment candidates.
- [ ] Exactly one Adam update occurs per complete epoch.
- [ ] No Adam update occurs per shard.
- [ ] No clipping, weight decay, learning-rate schedule, warmup, early stopping, shuffling, batching change, or gradient averaging exists.
- [ ] Weights and moments are committed only after all complete candidates are finite.
- [ ] Any failed reduction or update preserves prior completed weight and moment bytes.
- [ ] Failure does not advance the epoch cursor.
- [ ] Failure does not add a successful report.
- [ ] A failed run cannot later sample, persist, or continue advancing.
- [ ] Training processes every epoch from `0` through the requested final epoch inclusive.
- [ ] Requested epoch `0` performs one update and produces one report.
- [ ] The report step is exactly `max(1, requested_epochs // 50)`.
- [ ] Epoch `0` is always reported.
- [ ] Every divisible report boundary is included.
- [ ] The exact requested final epoch is included.
- [ ] The requested final epoch is never duplicated.
- [ ] Requested epoch `5000` produces exactly `51` report epochs from `0` through `5000` by `100`.
- [ ] Public report loss uses `round_typescript_decimal(loss, 6)`.
- [ ] Public rounded zero is positive `0.0`.
- [ ] Internal loss is retained unrounded at approved Python floating-point precision.
- [ ] One-through-four simulated worker completion schedules produce identical reduced losses and gradients.
- [ ] Those schedules produce identical moments, weights, report epochs, and public losses.
- [ ] Selected step-1 and step-2 moments and parameters match independent fixtures within named tight tolerances.
- [ ] Exact structural and report assertions do not use broad numerical tolerances.
- [ ] Fresh runs never load a Saved Transformer Model.
- [ ] Fresh runs do not share weights.
- [ ] Fresh runs do not share first moments.
- [ ] Fresh runs do not share second moments.
- [ ] Fresh runs do not share reduction workspaces.
- [ ] Fresh runs do not share either Adam scratch array.
- [ ] Fresh runs do not share report-history state.
- [ ] Sequential and concurrent independent runs remain deterministic.
- [ ] No real process, pipe, shared-memory block, route, schema, SSE, sample, final evaluation, persistence, or frontend code is added.
- [ ] No dependency or lockfile change is present.
- [ ] No generated `.data`, cache, or temporary file is included in the diff.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer.py
tests/test_transformer.py
tests/test_transformer_training.py
tests/fixtures/transformer_training_reference.json
```

### Conditions

- `tests/test_transformer.py` changes only to preserve and extend the exact approved public export contract or to add a narrow creation-signature/no-saved-model assertion.
- `tests/test_transformer_training.py` is preferred as a new focused module because parent reduction, Adam, lifecycle, and reporting are a distinct approved seam and `tests/test_transformer_math.py` is already the forward/backward authority.
- `tests/fixtures/transformer_training_reference.json` contains only new independent Ticket 017 evidence.

Conditionally changed only if live repository inspection proves a small existing public test helper must be shared without duplicating fixtures:

```text
tests/test_transformer_math.py
```

The default expectation is to leave `tests/test_transformer_math.py` and `tests/fixtures/transformer_forward_backward_reference.json` unchanged.

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
tests/fixtures/math_utils_reference.json
tests/fixtures/matrix_reference.json
tests/fixtures/transformer_preprocessing_reference.json
tests/fixtures/transformer_layout_initialization_reference.json
tests/fixtures/transformer_forward_backward_reference.json
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
017-advance-transformer-epochs-with-ordered-reduction-and-adam.md
```

A listed file may change only if live repository evidence proves the supplied source export is stale and the change is strictly required for Ticket 017. Such a change must be explained in the implementation report and kept within the accepted scope.

## Risk notes and safeguards

1. **Risk:** Arrival order silently becomes numerical reduction order.
   - **Safeguard:** Validate into four fixed shard-ID slots, then traverse explicit IDs `0..3`; protect with order-sensitive loss and gradient fixtures.

2. **Risk:** A dictionary or set accepts duplicates or obscures missing shard IDs.
   - **Safeguard:** Track explicit occupancy for all four IDs and reject a second record before reduction.

3. **Risk:** A result has the right shard ID but the wrong range or sequence count.
   - **Safeguard:** Compare complete shard metadata against the run’s expected immutable boundaries and require exact processed-count agreement.

4. **Risk:** A malformed empty shard hides stale values.
   - **Safeguard:** Require exact zero loss and a complete all-zero gradient for an empty range.

5. **Risk:** The reducer creates a new array each epoch or one array per worker.
   - **Safeguard:** Allocate one run-owned canonical `float32` reduction workspace and assert stable identity across epochs.

6. **Risk:** NumPy performs a tree, pairwise, or wider reduction whose order differs.
   - **Safeguard:** Apply one explicit `float32` workspace addition per shard in canonical order and use order-sensitive fixtures.

7. **Risk:** Gradients are averaged because that is common in data-parallel training.
   - **Safeguard:** Include a simple coordinate whose expected unaveraged sum is unmistakable and assert no division path exists.

8. **Risk:** Loss is averaged or accumulated in arrival order.
   - **Safeguard:** Use separate Python-float ordered accumulation with a non-associative fixture.

9. **Risk:** A shard gradient aliases parent state and can mutate weights or moments.
   - **Safeguard:** Validate memory independence and add crafted alias-rejection tests.

10. **Risk:** Adam is applied once per shard.
    - **Safeguard:** Expose one complete epoch transition, record one update count per epoch, and assert no weight mutation during shard validation/reduction.

11. **Risk:** Epoch zero uses optimizer step zero and divides by zero in bias correction.
    - **Safeguard:** Derive step only as `epoch + 1` and protect epoch-zero coordinates with the independent fixture.

12. **Risk:** A later epoch resets moments.
    - **Safeguard:** Include step-2 fixture values that cannot match a fresh-moment calculation.

13. **Risk:** The implementation uses all-`float64` moments and only casts after the parameter update.
    - **Safeguard:** Materialize each completed moment candidate as `float32`, then promote that completed value for bias correction; protect with precision-sensitive step-2 coordinates.

14. **Risk:** The implementation allocates more than two persistent or reusable `float64` Adam arrays.
    - **Safeguard:** Centralize exactly two run-owned scratch allocations, reuse their identities for every epoch, and expose immutable test metadata rather than mutable arrays.

15. **Risk:** Temporary `float32` candidate arrays are confused with persistent state or serialized.
    - **Safeguard:** Keep candidates parent-local and transient; serialize or report only approved stable values.

16. **Risk:** Moments are committed before a later weight candidate failure.
    - **Safeguard:** Calculate and validate all three complete candidates before any final copy into persistent state.

17. **Risk:** Final commit writes coordinates incrementally and fails halfway.
    - **Safeguard:** Prevalidate exact shapes, dtypes, writability, and finiteness, then perform exact-size final copies only after all failure-prone calculation is complete.

18. **Risk:** The reduction workspace remains partially accumulated after failure and is mistaken for valid state.
    - **Safeguard:** Mark the transition/run failed, never expose a successful reduction, and zero the workspace before any future valid reduction path.

19. **Risk:** Non-finite current weights or moments are detected only after mutation.
    - **Safeguard:** Validate all persistent inputs before candidate calculation.

20. **Risk:** A finite `float64` candidate overflows when materialized as `float32`.
    - **Safeguard:** Validate every completed `float32` moment and weight candidate before commit.

21. **Risk:** Report loss is rounded before finite checks or used as internal loss.
    - **Safeguard:** Keep ordered internal loss separate and call the shared rounding helper only when constructing the public update.

22. **Risk:** Python built-in rounding changes negative half ties or signed zero.
    - **Safeguard:** Import `round_typescript_decimal()` from `math_utils.py` and retain exact shared utility regressions.

23. **Risk:** The final report appears twice when divisible by the report step.
    - **Safeguard:** Evaluate one report predicate during a single inclusive traversal; assert exact tuple uniqueness.

24. **Risk:** Requested epoch zero produces no update.
    - **Safeguard:** Treat the requested value as the inclusive final epoch and test one update/report at zero.

25. **Risk:** A broad Training Run method performs all epochs and prevents later cooperative orchestration.
    - **Safeguard:** Keep one stable epoch transition and report boundary; route and process tickets decide asynchronous scheduling later.

26. **Risk:** A particular iterator implementation becomes the test contract.
    - **Safeguard:** Assert public state transitions, returned values, and lifecycle behavior, not `__next__()` or private cursor decomposition.

27. **Risk:** Physical worker count leaks into numerical behavior.
    - **Safeguard:** Exclude worker count from production signatures and treat one-through-four completion schedules as test-only permutations.

28. **Risk:** The implementation starts real processes to prove order independence.
    - **Safeguard:** Use synthetic result permutations and direct shard calculations only; reserve spawn protocol for its later ticket.

29. **Risk:** Fresh runs share initialized storage because the same source object is reused.
    - **Safeguard:** Establish exclusive ownership or copy initial flat weights; prove `np.shares_memory()` is false across runs.

30. **Risk:** Concurrent runs share moments, workspaces, scratch arrays, epoch cursors, or report history.
    - **Safeguard:** Keep every mutable object run-owned and add sequential and threaded isolation tests.

31. **Risk:** A failed run can later emit a report or be persisted by future orchestration.
    - **Safeguard:** Record terminal failure state and reject later advancement; no successful update is returned for the failed epoch.

32. **Risk:** The ticket expands into sample generation, final evaluation, worker protocol, shared memory, HTTP, SSE, or persistence.
    - **Safeguard:** Enforce the expected-file list and inspect the final diff for forbidden imports and behavior.

33. **Risk:** The new fixture is circular.
    - **Safeguard:** Record independent provenance and prohibit production optimizer/reducer imports during expected-value generation.

34. **Risk:** Tolerances are broad enough to accept the wrong Adam staging.
    - **Safeguard:** Use selected precision-sensitive coordinates, exact dtype/stage assertions, and the smallest independently justified `rtol`/`atol`.

35. **Risk:** A formatter or refactor creates unrelated Transformer churn.
    - **Safeguard:** Format only changed files, run `git diff --check`, and reject unrelated changes before commit.

36. **Risk:** User-reported baseline is mistaken for current-session verification.
    - **Safeguard:** Re-run the baseline in `implement-prompt` before editing and report actual output honestly.

## Commit guidance after tests pass

Use the repository’s established outcome-oriented convention.

Suggested subject:

```text
Advance Transformer epochs with ordered Adam updates
```

The commit body should mention:

- strict validation of exactly four Logical Training Shard results;
- canonical `0 → 1 → 2 → 3` loss and gradient reduction independent of completion order;
- one reusable `float32` reduction workspace and unaveraged gradients;
- parent-owned `float32` weights and Adam moments;
- exactly two reusable parent-local `float64` Adam scratch arrays;
- transactional candidate validation and atomic state commit;
- exact Adam constants and `epoch + 1` step;
- inclusive epoch zero-through-final progression;
- exact report schedules and shared six-decimal TypeScript-compatible loss rounding;
- finite-state failure preservation and failed-run lifecycle;
- one-through-four simulated worker completion-order equivalence;
- independent selected moment/weight fixture provenance and explicit tolerances;
- fresh-run sequential/concurrent isolation;
- no worker process, shared memory, generation, final evaluation, route, SSE, persistence, frontend, dependency, or lockfile work;
- the exact focused and full verification commands actually executed and their observed results.

Do not create a commit during `to-plan-prompt`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan017.md`;
- `017-advance-transformer-epochs-with-ordered-reduction-and-adam.md`;
- completed blocker Ticket 016 or the current completed forward/backward/shard implementation and tests as equivalent evidence;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- the latest `py_llm_pipeline_explorer_file_structure.md` source export created after Ticket 016;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the live repository again, establish its own baseline before editing, preserve user changes, implement only Ticket 017, create independent failure-first reduction/Adam/report evidence, reuse the completed canonical layout, shard result, and shared rounding boundaries, run focused and full verification, report actual command outcomes honestly, inspect final scope, and create the implementation commit only after all required checks pass.
