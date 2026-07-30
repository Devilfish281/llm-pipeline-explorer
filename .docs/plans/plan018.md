---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "018"
source_work_item: 018-generate-deterministic-text-and-construct-saved-transformer-models.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(49).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 018: Generate deterministic text and construct Saved Transformer Models

## Initial checklist

- Confirm Ticket 018 is the only selected work item and that its Ticket 016 blocker is satisfied by the completed public Transformer forward, cross-entropy, backward, sequence, and Logical Training Shard boundaries in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(49).md` as the source of truth for current Python code, tests, fixtures, configuration, typing style, and established public seams.
- Use the selected ticket, `SPEC.md`, `CONTEXT.md`, and ADR 0002 as the required-behavior authority; use `llm_works_file_structure.md` only as the TypeScript Reference Implementation for generation, text reconstruction, configuration order, and saved-model shape.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Limit production work to the parent-side public generation, final-evaluation, and Saved Transformer Model construction boundary in `ml/transformer.py`.
- Add independent exact generation/model fixtures and focused tests without testing private sampling helpers, a specific sorting implementation, filesystem persistence, route formatting, process timing, or optimizer internals.
- Finish with focused completion tests, all Transformer regressions, formatting, the complete pytest suite, Ruff, strict mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: convert the selected TypeScript behavior to Python and treat the latest complete Python Backend export as current-code truth.
2. `018-generate-deterministic-text-and-construct-saved-transformer-models.md` for immediate scope, acceptance criteria, approved test seam, blocker, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(49).md` for the current implementation, tests, fixtures, dependencies, file paths, type conventions, and public Transformer boundary.
4. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for durable Phase 5 decisions and the canonical terms Generated Text Sample, Sample Random Stream, Transformer Training Sequence, and Saved Transformer Model.
5. The current Ticket 016 forward/backward implementation and Ticket 017 parent-owned training state as represented in export `(49)`.
6. The latest `llm_works_file_structure.md`, especially TypeScript `generateText()`, `serializeModel()`, `TransformerConfig`, `TransformerWeights`, and `BLOCK_KEYS`, as behavior and insertion-order evidence only.
7. Official Python and NumPy documentation only as technical cross-checks for `threading.Event`, stable ordering, and JSON insertion-order preservation.
8. Older Python exports, prior plans, pasted snippets, cached model behavior, and historical TypeScript worker choices are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 018 adds the parent-side numerical completion boundary needed after deterministic Transformer epoch advancement.

For every reported epoch, the Python Backend must generate one deterministic Generated Text Sample from the fixed first three corpus-derived token IDs. The operation must use only the latest sixteen IDs as forward input, apply finite temperature scaling and stable softmax, build the minimum stable top-p nucleus, and sample exclusively from that nucleus with a fresh `Mulberry32` Sample Random Stream seeded by `(42 + epoch) modulo 2^32`. The returned text must reconstruct the complete seed-plus-generated token sequence by concatenating the ordered Vocabulary tokens exactly as the TypeScript reference does. Cancellation must be checked between generated tokens and must prevent a successful return.

After the inclusive final epoch's Adam update has committed, a separate public final-evaluation operation must recompute average loss from the final weights over every immutable Transformer Training Sequence in fixed order. It must check cancellation between sequences, reject incomplete or failed runs and non-finite state, and return the shared six-decimal public value rather than reusing `last_completed_loss`, which describes the pre-update shard computation.

A third public operation must convert the completed finite state into one fresh plain-Python Saved Transformer Model with exact insertion order:

```text
type → config → vocab → merges → weights
```

The model type is `decoder-transformer`. Configuration order is:

```text
vocabSize → contextLen → embDim → numHeads → ffDim → numLayers
```

Weight order is:

```text
tokEmb → posEmb → blocks → lnFGamma → lnFBeta → headW → headB
```

Every block must follow the canonical sixteen-key order already defined by the current layout. Every parameter array must be flattened in canonical C order and converted coordinate-by-coordinate to ordinary finite six-decimal Python numbers with signed zero normalized to positive `0.0`. The result must contain the complete ordered Vocabulary and Merge Table but no optimizer, gradient, cache, shared-memory, process, path, timestamp, request, or checkpoint state.

This ticket stops at in-memory construction. Route orchestration, worker supervision, SSE, deadlines, atomic persistence, and frontend changes remain later work.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, mypy, Black, fixture-generation, Node, browser, route, or persistence command was executed while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and report the actual results honestly.

## Readiness assessment

- **Selected work item:** Ticket 018 only.
- **Formal blocker:** Ticket 016.
- **Blocker status in current code:** Satisfied. The latest export contains public forward, cross-entropy, backward, complete-sequence, and Logical Training Shard calculations with exact fixture coverage.
- **Required preceding state:** Also present. Ticket 017's `TransformerTrainingRun.advance_epoch()` performs Ordered Gradient Reduction and the Adam update before marking the inclusive final epoch complete.
- **Product or architecture decisions remaining:** None that block implementation.
- **Approved test seam:** The stable public generation, final-evaluation, and Saved Transformer Model construction boundary.
- **Planning status:** Ready for implementation.

## Current code observations from the latest source

- `src/how_llms_work/ml/transformer.py` is the live owner of immutable preprocessing, the canonical parameter layout, deterministic initialization, forward/backward mathematics, Logical Training Shard calculation, Ordered Gradient Reduction, Adam, and `TransformerTrainingRun`.
- The module currently exports no Generated Text Sample type, no generation operation, no final-evaluation operation, and no Saved Transformer Model type or builder.
- `TransformerPreprocessingSnapshot` already provides immutable ordered `vocabulary`, public `merges`, `training_sequences`, and the exact three `generation_seed_ids` required by this ticket.
- The current constants already establish:
  - generation seed length `3`;
  - training/generation window length `16`;
  - context length `32`;
  - embedding dimension `32`;
  - attention heads `2`;
  - feed-forward dimension `128`;
  - supported layer count `1..6`.
- `calculate_transformer_forward()` already returns finite `float32` logits and probabilities while preserving learned token embeddings, absolute positional embeddings, causal attention, and final Vocabulary projection.
- `calculate_transformer_cross_entropy()` already implements the approved average next-token loss with `1e-10` protection and finite validation.
- `TransformerTrainingRun.advance_epoch()` currently builds the public loss observation before applying Adam, then applies Adam and marks the final inclusive epoch complete. Consequently, `last_completed_loss` is intentionally not the final post-update evaluation value.
- `TransformerTrainingRun` already exposes `parameters`, `weights`, `requested_epochs`, `last_completed_epoch`, `is_failed`, and `is_complete`, allowing completion operations to validate lifecycle state without reaching into worker or route objects.
- `build_transformer_parameter_layout()` already defines the sole flat order:
  - `tokEmb`;
  - `posEmb`;
  - each block in ascending index and canonical sixteen-key order;
  - `lnFGamma`;
  - `lnFBeta`;
  - `headW`;
  - `headB`.
- `_TRANSFORMER_BLOCK_PARAMETER_KEYS` already matches TypeScript `BLOCK_KEYS` exactly.
- `src/how_llms_work/ml/math_utils.py` already owns the instance-local JavaScript-compatible `Mulberry32` and `round_typescript_decimal()`. The latter rejects non-finite values and normalizes either sign of zero to positive `0.0`.
- Transformer tokenization already stores decoded Vocabulary tokens, including their leading spaces. The TypeScript reconstruction behavior is therefore equivalent to concatenating the complete seed-plus-generated token strings with no inserted separator.
- `tests/test_transformer.py`, `tests/test_transformer_math.py`, and `tests/test_transformer_training.py` already establish exact-fixture provenance, public-seam testing, finite-state rejection, mutation protection, concurrency isolation, canonical layout order, forward/backward behavior, and parent-side optimizer behavior.
- `tests/test_word2vec_results.py` and `build_saved_embedding_model()` provide useful prior art for complete ordered public model conversion, plain-JSON recursion checks, non-finite rejection, fresh-container isolation, and independent exact fixtures.
- `src/how_llms_work/routes/train_transformer.py` and `src/how_llms_work/ml/transformer_worker.py` remain empty and are outside Ticket 018.
- `schemas.py` has no Train Transformer request model and `main.py` does not register the Transformer route; those omissions are expected at this stage and must not be pulled into this ticket.
- `pyproject.toml` already contains all required runtime and development dependencies. No dependency or lockfile change is needed.

## Acceptance criteria coverage

### Already satisfied and evidenced

- Immutable preprocessing supplies the exact first three corpus-derived generation seed IDs.
- Immutable preprocessing supplies the complete ordered Vocabulary, complete ordered public Merge Table, and fixed-order Transformer Training Sequences.
- The current forward pass preserves learned absolute positions and causal behavior and supports input lengths up to the fixed context length.
- The public matrix softmax boundary provides stable finite row-softmax behavior.
- `Mulberry32` is instance-owned and supports unsigned 32-bit seed normalization.
- `round_typescript_decimal()` provides finite six-decimal public rounding and positive-zero normalization.
- The canonical flat parameter layout and exact block-key order are already implemented and independently fixture-tested.
- `TransformerTrainingRun` applies Adam before committing completion of the final inclusive epoch.
- Current parameter, optimizer, gradient, and scratch state are separately owned, allowing the saved-model builder to include only final parameter data.

### Behavior present but evidence incomplete

- The existing decoded Vocabulary representation contains everything needed for exact token-to-text reconstruction, but no public Generated Text Sample operation proves the full seed-plus-sampled result.
- `TransformerTrainingRun.parameters` exposes the current/final weights, but no public completion boundary currently rejects incomplete or failed runs.
- Existing public-model conversion tests for Word2Vec demonstrate the correct testing pattern, but Transformer-specific complete field/key/value fixtures do not yet exist.

### Partially implemented

- All numerical primitives required for generation and final evaluation exist, but they are not composed into Ticket 018's public parent-side operations.
- All layout and preprocessing data required for Saved Transformer Model construction exist, but there is no exact typed plain-Python model builder.

### Not implemented

- A public Generated Text Sample value and generation operation.
- Fresh epoch-seeded Sample Random Stream ownership.
- Latest-sixteen context truncation at generation time.
- Temperature validation and temperature-scaled stable softmax for generation.
- Stable descending top-p nucleus construction and deterministic sampling.
- Cancellation checks between generated tokens and cancellation-before-success handling.
- Independent exact generated-text fixtures.
- A public final post-update loss evaluation operation.
- Complete-run gating and fixed-order evaluation cancellation checks.
- A typed Saved Transformer Model contract and builder.
- Exact top-level, configuration, weight, and block insertion order.
- Complete C-order parameter flattening and six-decimal plain-number conversion.
- Transformer-specific NaN/infinity rejection, transient-state exclusion, fresh-container isolation, and complete serialization-ready fixture evidence.

### Evidence limitations

- Baseline results are user-reported rather than tool-verified in this planning session.
- The latest Python source is a complete export rather than direct repository access; the implementation run must inspect the live repository again.
- The current TypeScript implementation uses one historical mutable random stream. Ticket 018 and ADR 0002 deliberately supersede that ownership with one fresh epoch-seeded Sample Random Stream while preserving the reference sampling math and text reconstruction.
- Exact new generated texts, final losses, and complete saved-model values are not present in the current Python fixtures. They must be captured from the TypeScript behavior where applicable or calculated by an independent scalar/reference implementation that imports no production Ticket 018 operation.
- The exact public function and type names do not yet exist. The implementation should select clear stable names once and export them through `transformer.__all__`; tests must target that public seam rather than private helper names.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer.py` — module exports, constants, `TransformerPreprocessingSnapshot`, `TransformerParameterLayout`, parameter views, `InitializedTransformerParameters`, `TransformerTrainingRun`, `calculate_transformer_forward()`, `calculate_transformer_cross_entropy()`, `_TRANSFORMER_BLOCK_PARAMETER_KEYS`, and token reconstruction support.
2. `src/how_llms_work/ml/math_utils.py` — `Mulberry32`, `round_typescript_decimal()`, and positive-zero normalization; inspect and reuse without duplicating state or rounding logic.
3. `src/how_llms_work/ml/matrix.py` — `stable_row_softmax()` input contract and finite/mask behavior; inspect and reuse for temperature-scaled probabilities.
4. `src/how_llms_work/ml/bpe.py` — `Merge` field semantics; inspect only to construct plain persistence merges without adding BPE behavior.
5. `tests/test_transformer.py` — preprocessing, canonical layout, initialization, immutability, exact-fixture provenance, and concurrency prior art.
6. `tests/test_transformer_math.py` — public forward/cross-entropy fixtures, finite checks, mutation protection, and deterministic numerical assertions.
7. `tests/test_transformer_training.py` — `TransformerTrainingRun` completion ordering, Adam, public rounding, synthetic shard-result helpers, and independent fixture conventions.
8. `tests/test_word2vec_results.py` — complete plain-model recursion, key-order, non-finite rejection, mutation, repeat-call, and concurrency-isolation prior art.
9. `tests/fixtures/transformer_preprocessing_reference.json` — complete Vocabulary, Merge Table, Training Sequence, and generation-seed evidence.
10. `tests/fixtures/transformer_layout_initialization_reference.json` — canonical layout, complete parameter count, and initialization evidence.
11. `tests/fixtures/transformer_forward_backward_reference.json` — selected exact logits/probabilities/loss values for generation and final-evaluation cross-checks.
12. `tests/fixtures/transformer_training_reference.json` — final-update and `TransformerTrainingRun` state-transition evidence.
13. `pyproject.toml` — Python 3.12, NumPy, pytest, Ruff, Black, and strict-mypy configuration; no dependency change is expected.
14. `018-generate-deterministic-text-and-construct-saved-transformer-models.md` — direct acceptance authority.
15. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` — deterministic generation, Sample Random Stream, final-loss, cancellation, and Saved Transformer Model decisions.
16. `llm_works_file_structure.md` — TypeScript `generateText()`, `serializeModel()`, `TransformerConfig`, `TransformerWeights`, and `BLOCK_KEYS` evidence only.

## Step 1 — Establish independent completion fixtures and freeze the public contract

**Files and symbols:**

- `tests/fixtures/transformer_completion_reference.json` — new independent generation, final-evaluation, and complete model fixture.
- `tests/test_transformer_completion.py` — new focused tests through the approved public completion seam.
- `src/how_llms_work/ml/transformer.py` — new public exports and stable completion value contracts; exact private helper names remain an implementation choice.

**Purpose:**

Create failure-first acceptance evidence for all three Ticket 018 outcomes before production implementation. This prevents the implementation from generating its own expected values and keeps tests focused on stable public behavior.

**Actions:**

- Add fixture provenance stating that expected values were captured from the TypeScript Reference Implementation where behavior is unchanged or from an independent scalar/reference program that imports no Ticket 018 production operation.
- Include exact generation cases using deliberately controlled finite one-layer states:
  - a state with a non-uniform final-position distribution;
  - a tie-sensitive state that proves stable Vocabulary-index tie order;
  - `topP=1.0`;
  - a narrow top-p nucleus;
  - supported temperature boundary values;
  - multiple report epochs whose Sample Random Stream seeds are `(42 + epoch) & 0xFFFFFFFF`;
  - a sample long enough to prove context truncation from seventeen or more accumulated IDs to the latest sixteen;
  - `maxTokens` at the validated minimum and one larger representative value.
- Record for each exact generation case:
  - epoch and normalized seed;
  - fixed generation seed IDs;
  - temperature, top-p, and maximum generated-token count;
  - expected sampled IDs;
  - expected complete reconstructed text;
  - selected nucleus membership and/or a behavioral landmark sufficient to catch out-of-nucleus selection without testing a private helper.
- Include an independent final-evaluation case whose `TransformerTrainingRun` has completed its final Adam update. Record the exact unrounded or tightly tolerance-protected loss and the exact six-decimal public loss.
- Include one complete representative one-layer Saved Transformer Model fixture containing every field, every Vocabulary token, every merge, every block field, and every rounded parameter coordinate.
- Use a deterministic independent coordinate pattern that includes:
  - ordinary positive and negative values;
  - more than six fractional digits;
  - raw positive and negative zero;
  - values immediately around six-decimal boundaries.
- Record the exact top-level order, six-field config order, weight order, block order, flattened lengths, and complete serialization-ready contents.
- Add public-seam tests that initially fail because the generation, final-evaluation, and Saved Transformer Model symbols are absent.
- Test public results, exact ordered fields, and complete values. Do not assert private helper names, a specific `argsort`/`sorted` call, an internal nucleus container type, or a private cancellation exception name.

**Guardrails:**

- Fixtures must not be generated by importing the production generation, final-evaluation, or model-builder operation under test.
- Do not add filesystem writes, route clients, SSE parsing, multiprocessing, deadlines, or presentation-delay controls.
- Do not make tests depend on `.data`, current working directory, process count, wall-clock timing, or the historical TypeScript cache path.
- Do not duplicate the entire existing preprocessing, initialization, forward/backward, or Adam fixture when a referenced fixture plus Ticket 018-specific values is sufficient.
- Keep exact complete model evidence representative rather than creating redundant giant fixtures for every layer count.

**Expected result:**

- The new focused tests precisely define the complete Ticket 018 public contract and fail only because the three public completion boundaries have not yet been implemented.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_completion.py -q
```

- Expected before production implementation: collection or assertions fail at the missing Ticket 018 public symbols.
- Expected after Steps 2 through 4: the focused completion suite passes.

## Step 2 — Implement deterministic epoch-seeded Generated Text Samples

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — new public Generated Text Sample value, public generation operation, export list, and narrowly scoped private validation/sampling support.
- `tests/test_transformer_completion.py` — generation, cancellation, finiteness, stream-isolation, and exact-text tests.
- `tests/fixtures/transformer_completion_reference.json` — independent generation evidence from Step 1.

**Purpose:**

Deliver the report-boundary generation behavior while keeping Sample Random Stream ownership independent from Weight Initialization, other report epochs, workers, routes, and global state.

**Actions:**

- Add one immutable public Generated Text Sample value containing exactly the reported `epoch` and reconstructed `text`.
- Add one stable public generation operation that accepts:
  - the current initialized Transformer parameters or their approved public parameter owner;
  - the immutable preprocessing snapshot;
  - the report epoch;
  - validated `temperature`;
  - validated `topP`;
  - validated `maxTokens`;
  - a request-owned `threading.Event`-compatible cancellation signal.
- Validate exact Python types and ranges at the numerical boundary:
  - epoch is a non-negative strict integer;
  - temperature is finite and within `0.1..2.0`;
  - top-p is finite and within `0.1..1.0`;
  - maxTokens is a strict integer within `3..500`;
  - preprocessing contains exactly three valid generation seed IDs;
  - parameter Vocabulary size and layer/layout data match preprocessing and the fixed architecture.
- Reject failed/non-finite parameter state before sampling.
- Copy the three immutable seed IDs into request-local generation state; never mutate preprocessing.
- Create exactly one fresh `Mulberry32((42 + epoch) & 0xFFFFFFFF)` inside each public generation call.
- For each generated token:
  - check cancellation before starting the token computation;
  - pass only the latest sixteen accumulated IDs to `calculate_transformer_forward()`;
  - use the final row of the returned logits;
  - apply temperature division using finite intermediate checks;
  - materialize a separate finite `float32` row and call the existing stable softmax boundary;
  - rank probabilities in stable descending order so equal probabilities preserve ascending original Vocabulary index;
  - include the minimum prefix whose cumulative probability is greater than or equal to top-p;
  - scale one Mulberry32 draw by the nucleus probability sum;
  - select only from that nucleus using the approved strict cumulative comparison;
  - append exactly one sampled token ID.
- Check cancellation after the loop and before constructing the successful public value so a signal arriving after the last token cannot produce a successful sample.
- Reconstruct text from the full seed-plus-generated sequence by mapping IDs through the ordered Vocabulary and concatenating the token strings with no extra separator.
- Return no more than `maxTokens` newly sampled IDs. The fixed three seed IDs are context and are included in the reconstructed text, not counted as generated tokens.
- Add exact tests for:
  - same state/epoch/options producing identical text and sampled behavior;
  - different epochs using their own exact streams;
  - extra Weight Initialization draws having no effect;
  - generating an earlier epoch having no effect on a later epoch;
  - sequential, interleaved, and threaded calls remaining independent;
  - stable tie behavior and no out-of-nucleus result;
  - latest-sixteen context behavior;
  - lower/upper validation boundaries and strict rejection cases;
  - cancellation before the first token, between tokens, and immediately before successful return;
  - no mutation or aliasing of parameters or preprocessing.

**Guardrails:**

- Do not introduce a module-global generator, cached sample stream, shared draw counter, or caller-supplied random seed.
- Do not reuse the initialization generator or store a generator on `TransformerTrainingRun`.
- Do not call the forward pass with more than sixteen IDs merely because the architectural context length is thirty-two.
- Do not sample from the forward result's pre-temperature probabilities.
- Do not use an unstable descending ordering or an unordered mapping for tie-sensitive nucleus membership.
- Do not expose logits, probabilities, nucleus entries, sampled IDs, generator state, or draw count in the public Generated Text Sample.
- Do not add an EOS rule, early stopping, token filtering, repetition penalty, KV cache, batch generation, or any sampling option absent from the ticket.
- Tests must prove observable stable ordering and selection; they must not mandate one sorting implementation.

**Expected result:**

- Every report epoch can produce one exact deterministic Generated Text Sample from current weights with independent random ownership, bounded latest context, approved top-p behavior, and cooperative cancellation.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_completion.py -q -k "generation or sample or temperature or top_p or context or cancel or stream"
```

- Expected result: all generation-focused tests pass.

## Step 3 — Recompute final public loss from final updated weights

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — new public final-evaluation operation and complete-run/cancellation validation.
- `tests/test_transformer_completion.py` — final-update ordering, fixed-sequence order, cancellation, finite-state, and exact-loss tests.
- `tests/fixtures/transformer_completion_reference.json` — independent final-evaluation evidence.

**Purpose:**

Ensure completion reports a loss calculated from the parameters after the last Adam update rather than the pre-update loss stored during the final epoch transition.

**Actions:**

- Add one stable public final-evaluation operation that accepts a `TransformerTrainingRun`, the immutable preprocessing snapshot, and a request-owned cancellation signal.
- Require:
  - exact `TransformerTrainingRun` and snapshot types;
  - `run.is_complete` is true;
  - `run.is_failed` is false;
  - `last_completed_epoch == requested_epochs`;
  - layout Vocabulary size, layer count, and parameter storage agree with preprocessing and fixed configuration;
  - all final weights are finite.
- Do not read `last_completed_loss` as the result.
- Iterate `preprocessing.training_sequences` in their stored order without shuffling, partitioning, or parallel reduction.
- Before each sequence:
  - check cancellation;
  - call `calculate_transformer_forward()` using the sequence input and final parameter views;
  - call `calculate_transformer_cross_entropy()` using the sequence targets;
  - accumulate loss in Python floating-point order.
- Require a non-empty Training Sequence collection and finite per-sequence and accumulated values.
- Divide by the exact Training Sequence count after the complete fixed-order traversal.
- Check cancellation once more before returning a successful value.
- Convert the finite final average through `round_typescript_decimal(value, 6)` and return the ordinary public float.
- Add tests proving:
  - incomplete and failed runs are rejected;
  - evaluation occurs after the final Adam update;
  - changing the final parameter storage changes final loss even when `last_completed_loss` is unchanged;
  - sequence order is the immutable preprocessing order;
  - cancellation before evaluation, between sequences, and before return prevents success;
  - the run, parameters, preprocessing, optimizer buffers, and existing updates remain unchanged;
  - repeated and concurrent evaluations produce the same exact public value.

**Guardrails:**

- Do not call `advance_epoch()`, apply Adam, alter optimizer state, or mark a run complete inside evaluation.
- Do not reuse the final epoch's shard loss, `last_completed_loss`, a prior public update, or a cached evaluation.
- Do not route evaluation through worker processes or Logical Training Shards.
- Do not average already-rounded per-sequence values; round only the complete final average.
- Do not add route timeouts, thread-pool ownership, or async behavior. Later orchestration will run this synchronous cooperative helper through `asyncio.to_thread()`.

**Expected result:**

- A completed run yields one exact finite six-decimal loss that describes its final post-update parameters, while cancellation and incomplete state cannot be mistaken for successful completion.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_completion.py -q -k "final_loss or evaluation or complete or sequence or cancel"
```

- Expected result: all final-evaluation tests pass.

## Step 4 — Construct the exact fresh plain-Python Saved Transformer Model

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — Saved Transformer Model `TypedDict` hierarchy or equivalent precise plain-container types, public builder, canonical flattening, and finite validation.
- `tests/test_transformer_completion.py` — complete fixture, field/order, plain-JSON, signed-zero, isolation, mutation, and non-finite tests.
- `tests/fixtures/transformer_completion_reference.json` — complete deterministic serialization-ready model.

**Purpose:**

Convert only the durable final model state into the exact TypeScript-compatible in-memory object required by later atomic persistence work.

**Actions:**

- Define precise public plain-container types for:
  - saved merge entries;
  - the six-field Transformer configuration;
  - each block's sixteen ordered parameter arrays;
  - the ordered top-level Transformer weights;
  - the complete Saved Transformer Model.
- Add one stable public builder accepting a completed `TransformerTrainingRun` and immutable `TransformerPreprocessingSnapshot`.
- Apply the same complete-run, failure, layout, Vocabulary-size, layer-count, storage, and finiteness validation used by final evaluation.
- Construct a new ordinary `dict` in exact top-level insertion order:
  1. `type`;
  2. `config`;
  3. `vocab`;
  4. `merges`;
  5. `weights`.
- Set `type` to `decoder-transformer`.
- Construct `config` in exact order:
  1. `vocabSize`;
  2. `contextLen`;
  3. `embDim`;
  4. `numHeads`;
  5. `ffDim`;
  6. `numLayers`.
- Derive fixed values from the approved constants and derive `vocabSize` and `numLayers` from validated preprocessing/layout state; do not accept duplicate caller-supplied configuration values.
- Copy the complete Vocabulary into a new list without reordering.
- Convert every public merge into a fresh object containing only:
  - `pair`: a fresh two-element list;
  - `merged`: the decoded merged token.
- Omit merge frequency because the TypeScript Saved Transformer Model contains only `pair` and `merged`.
- Construct `weights` in exact order:
  1. `tokEmb`;
  2. `posEmb`;
  3. `blocks`;
  4. `lnFGamma`;
  5. `lnFBeta`;
  6. `headW`;
  7. `headB`.
- Construct blocks in ascending layer order. Within each block, iterate the existing canonical sixteen-key tuple exactly once.
- For every parameter:
  - take the validated semantic view;
  - iterate coordinates in C order;
  - convert each NumPy scalar to Python `float`;
  - call `round_typescript_decimal(value, 6)`;
  - append the result to a new flat Python list.
- Validate expected flattened length against the canonical layout record before publishing each array.
- Ensure the completed object recursively contains only:
  - `dict`;
  - `list`;
  - `str`;
  - strict Python `int`;
  - finite Python `float`;
  - no Boolean values.
- Ensure the result is accepted by strict JSON serialization with non-finite values disallowed, but do not write a file in this ticket.
- Add exact tests proving:
  - complete top-level/config/weight/block key order;
  - full exact Vocabulary and merges;
  - every flat array length and complete representative contents;
  - six-decimal values and positive zero sign;
  - no NumPy scalar/array, memoryview, mapping proxy, tuple, dataclass, path, process, pipe, optimizer, gradient, scratch, cache, timestamp, request ID, or checkpoint field;
  - non-finite values anywhere in final parameter storage prevent construction;
  - construction does not mutate the run, parameters, optimizer state, or preprocessing;
  - two calls return equal but non-identical nested containers;
  - mutating one returned model does not affect the source or another returned model;
  - sequential and threaded construction is deterministic and isolated.

**Guardrails:**

- Do not serialize, create directories, open temporary files, call `fsync()`, replace a destination, or resolve `.data` paths.
- Do not load or resume from a Saved Transformer Model.
- Do not include epochs, loss, samples, seed, generator state, optimizer moments, gradients, process state, shared-memory names, shapes, dtypes, offsets, paths, timestamps, request IDs, or checkpoint metadata.
- Do not sort keys alphabetically; preserve the explicit reference and canonical insertion order.
- Do not preserve NumPy shapes in the JSON object; TypeScript stores every parameter array as one flat list.
- Do not return read-only proxies or tuples merely to enforce immutability; the contract requires fresh plain-Python JSON-compatible containers.

**Expected result:**

- Any completed finite run can be converted repeatedly into the exact deterministic, ordered, complete, plain-Python Saved Transformer Model without exposing or mutating transient training state.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_completion.py -q -k "saved_model or model or order or plain or zero or finite or isolation"
```

- Expected result: all Saved Transformer Model tests pass.

## Step 5 — Verify the integrated public completion boundary and preserve prior Transformer behavior

**Files and symbols:**

- `tests/test_transformer_completion.py` — complete acceptance suite.
- `src/how_llms_work/ml/transformer.py` — final public export list and integrated completion contracts.
- Existing Transformer test modules and fixtures — unchanged regression authority.

**Purpose:**

Prove that generation, final evaluation, and model construction compose safely with Tickets 014 through 017 without weakening preprocessing, layout, forward/backward, reduction, Adam, or run-state behavior.

**Actions:**

- Add one end-to-end public numerical test that:
  - obtains immutable preprocessing;
  - initializes a supported one-layer state;
  - creates a `TransformerTrainingRun`;
  - advances it through its requested inclusive final epoch using controlled valid Logical Training Shard results;
  - generates exact samples at selected public report epochs from the post-update current parameters;
  - evaluates final loss only after completion;
  - builds the Saved Transformer Model;
  - proves final loss and model weights correspond to the same final parameter storage.
- Confirm public generation can also operate at intermediate reported epochs without requiring the run to be complete.
- Confirm final evaluation and model construction reject the same run before completion and after a failed transition.
- Confirm generated samples, final evaluation, and model construction leave:
  - parameter bytes;
  - first and second moments;
  - reduced gradient workspace;
  - Adam scratch arrays;
  - preprocessing;
  - existing epoch updates;
  unchanged.
- Confirm `transformer.__all__` exposes only the intended new stable public values and operations; do not export private nucleus, sorting, flattening, or validation helpers.
- Run the existing preprocessing/layout, forward/backward, and training suites unchanged.
- Inspect the final diff to confirm no route, worker, schema, persistence, frontend, dependency, lockfile, or runtime model artifact was changed.

**Guardrails:**

- Keep the integrated test numerically small enough for routine local execution while still using the real public forward, training-run, generation, evaluation, and model-construction boundaries.
- Do not convert the completion test into a full 50-to-2000-epoch training benchmark.
- Do not monkeypatch the operations under test with fake generation, fake forward, fake cross-entropy, or fake model conversion.
- Do not add broad refactors or formatting churn to completed Transformer mathematics.

**Expected result:**

- Ticket 018's three public completion operations work together on one deterministic final state, and all existing Phase 5 numerical behavior remains unchanged.

**Verification:**

```powershell
poetry run pytest `
    tests/test_transformer.py `
    tests/test_transformer_math.py `
    tests/test_transformer_training.py `
    tests/test_transformer_completion.py `
    -q
```

- Expected result: all Transformer preprocessing, layout, mathematics, training, generation, final-evaluation, and model-construction tests pass.

## Focused verification plan

Run from the backend project root:

```powershell
poetry run pytest tests/test_transformer_completion.py -q
```

Expected result:

- All Ticket 018 public generation, final-evaluation, Saved Transformer Model, cancellation, exact-fixture, finite-state, and isolation tests pass.

Then run the complete affected Transformer area:

```powershell
poetry run pytest `
    tests/test_math_utils.py `
    tests/test_transformer.py `
    tests/test_transformer_math.py `
    tests/test_transformer_training.py `
    tests/test_transformer_completion.py `
    -q
```

Expected result:

- Shared random/rounding utilities and all Transformer numerical regressions pass unchanged.

Check formatting for the likely changed Python files:

```powershell
poetry run black --check `
    src/how_llms_work/ml/transformer.py `
    tests/test_transformer_completion.py
```

Expected result:

- Black reports no formatting changes required.

## Full verification plan

Run once after the focused suites pass:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
poetry run black --check .
```

Expected result:

- All tests pass.
- Ruff reports no issues.
- Strict mypy reports `Success: no issues found`.
- Black reports no formatting changes required.

Perform final scope and whitespace inspection:

```powershell
git diff --check
git status --short
git diff -- `
    src/how_llms_work/ml/transformer.py `
    tests/test_transformer_completion.py `
    tests/fixtures/transformer_completion_reference.json
```

Expected result:

- No whitespace errors.
- Only Ticket 018 files and intentional user changes appear.
- No generated runtime model, temporary file, cache, dependency, lockfile, route, worker, schema, frontend, or persistence change is included.

## Manual acceptance checklist

- [ ] Ticket 018 is the only implemented work item.
- [ ] The latest export `(49)` was rechecked against the live repository before editing.
- [ ] Each sample begins with `preprocessing.generation_seed_ids`.
- [ ] Each sample call creates a fresh `Mulberry32((42 + epoch) & 0xFFFFFFFF)`.
- [ ] Sampling one epoch cannot affect initialization or any other epoch.
- [ ] Forward generation input never exceeds the latest sixteen IDs.
- [ ] Temperature and top-p enforce the approved strict finite ranges.
- [ ] Stable descending ties preserve original Vocabulary index order.
- [ ] The nucleus is the minimum prefix reaching top-p, and selection cannot escape it.
- [ ] At most `maxTokens` new IDs are appended.
- [ ] Text is the no-separator concatenation of seed-plus-generated Vocabulary tokens.
- [ ] Cancellation between tokens prevents a successful Generated Text Sample.
- [ ] Final evaluation rejects incomplete and failed runs.
- [ ] Final loss is recomputed from final post-update weights in fixed Training Sequence order.
- [ ] Final loss is finite and rounded once through the shared six-decimal helper.
- [ ] Saved model top-level order is `type`, `config`, `vocab`, `merges`, `weights`.
- [ ] Saved config, weight, and block key order exactly matches the reference and canonical layout.
- [ ] Type is exactly `decoder-transformer`.
- [ ] Vocabulary and Merge Table are complete and ordered.
- [ ] Every weight array is flat C-order and every coordinate is a finite ordinary six-decimal Python number.
- [ ] Both signs of zero become positive `0.0`.
- [ ] No NumPy, process, shared-memory, optimizer, gradient, cache, path, timestamp, request, or checkpoint object/metadata appears in the saved model.
- [ ] Repeated model construction returns fresh isolated containers and mutates no source state.
- [ ] Exact independent generation, final-loss, and complete saved-model fixtures pass.
- [ ] Existing Transformer preprocessing, layout, forward/backward, and Adam tests remain unchanged and pass.
- [ ] No filesystem persistence, HTTP/SSE, worker supervision, frontend, dependency, or lockfile work was added.
- [ ] The actual pytest, Ruff, mypy, and Black results are reported honestly.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer.py
tests/test_transformer_completion.py
tests/fixtures/transformer_completion_reference.json
```

Conditionally changed:

```text
None expected.
```

A conditional change to another live module is justified only if repository inspection proves a required public operation is absent and cannot be reused without violating its current owner. Such a gap must be documented before editing; no dependency or lockfile change is allowed.

## Files not to change

```text
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/train_transformer.py
src/how_llms_work/schemas.py
src/how_llms_work/main.py
src/how_llms_work/sse.py
pyproject.toml
poetry.lock
.data/
frontend/
```

Existing Transformer fixtures should remain unchanged unless independent evidence proves they are incorrect. Ticket 018 should add its own fixture rather than rewriting earlier-ticket authority.

## Risk notes and safeguards

1. **Risk:** Generation accidentally reuses the forward pass's pre-temperature probabilities.
   - **Safeguard:** Extract final-position logits, scale logits first, then run the existing stable softmax and assert temperature-sensitive exact samples.

2. **Risk:** Equal-probability tokens enter the nucleus in host- or implementation-dependent order.
   - **Safeguard:** Require stable descending behavior with original Vocabulary index as the preserved tie order and verify it through exact public samples rather than a private sort implementation.

3. **Risk:** The cumulative threshold includes too few or too many tokens.
   - **Safeguard:** Test values immediately below, at, and above a cumulative boundary and require the minimum prefix whose sum reaches top-p.

4. **Risk:** Sampling consumes Weight Initialization randomness or earlier samples alter later samples.
   - **Safeguard:** Instantiate one local Mulberry32 per call from the epoch formula and compare against uninterrupted, interleaved, and threaded control streams.

5. **Risk:** Generation uses all thirty-two context positions or resets learned positions incorrectly.
   - **Safeguard:** Capture the latest sixteen accumulated IDs exactly, pass them unchanged to the public forward calculation, and include a context-sensitive fixture after more than sixteen IDs have accumulated.

6. **Risk:** Cancellation arrives after the last token check and a successful sample still escapes.
   - **Safeguard:** Check before every token and once immediately before public result construction.

7. **Risk:** Final completion reports the final epoch's pre-update shard loss.
   - **Safeguard:** Require a completed run and recompute every sequence from current final parameter views; include a test where `last_completed_loss` and recomputed loss differ.

8. **Risk:** Evaluation order or partial rounding changes the public loss.
   - **Safeguard:** Traverse the immutable sequence tuple in order, accumulate unrounded Python floats, divide once, and apply the shared six-decimal helper only at the end.

9. **Risk:** Saved-model order drifts because ordinary mappings are filled indirectly or keys are alphabetically sorted.
   - **Safeguard:** Build each mapping explicitly in reference order and assert `list(mapping)` at every nested level.

10. **Risk:** Multidimensional NumPy views serialize with nested shapes rather than the TypeScript flat arrays.
    - **Safeguard:** Flatten each validated view in C order and verify exact lengths against canonical layout records.

11. **Risk:** NumPy scalars, arrays, tuples, proxies, optimizer state, or other transient objects leak into the artifact.
    - **Safeguard:** Use a recursive plain-JSON assertion, strict non-finite serialization check, exact field-set checks, and forbidden-type/forbidden-key tests.

12. **Risk:** Signed negative zero survives persistence conversion.
    - **Safeguard:** Convert every coordinate through `round_typescript_decimal(value, 6)` and use sign-sensitive assertions with `math.copysign()`.

13. **Risk:** Returned containers alias final weights, preprocessing, or another model.
    - **Safeguard:** Allocate every dict/list anew and mutate one returned model in tests while checking source bytes and a second model.

14. **Risk:** A complete model fixture becomes self-referential or generated by production code.
    - **Safeguard:** Require independent provenance and prohibit production Ticket 018 imports in fixture-generation logic.

15. **Risk:** The ticket expands into persistence or route orchestration because the resulting object is serialization-ready.
    - **Safeguard:** Stop at the in-memory object. Keep `.data`, path resolution, JSON file formatting, temporary files, `fsync()`, atomic replacement, deadlines, SSE, and `done` behavior out of scope.

16. **Risk:** Large exact fixtures make focused tests unnecessarily slow or unreadable.
    - **Safeguard:** Keep one complete representative model fixture, use selected exact generation/final-loss cases, and test additional layer counts through key order and canonical length assertions rather than duplicate full artifacts.

17. **Risk:** Broad formatting or refactoring obscures the numerical change.
    - **Safeguard:** Restrict edits to the three expected files, run scope-only diffs, and reject unrelated churn before commit.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Suggested outcome:

```text
Generate deterministic Transformer samples and models
```

Commit body should mention:

- independent epoch-seeded Generated Text Samples with latest-sixteen context and stable top-p behavior;
- cooperative cancellation between generated tokens;
- final post-Adam fixed-order loss evaluation;
- exact ordered plain-Python Saved Transformer Model construction;
- six-decimal finite coordinates, signed-zero normalization, and transient-state exclusion;
- independent exact fixtures and isolation tests;
- no persistence, route, worker, frontend, dependency, or lockfile changes;
- the exact focused and full verification commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan018.md`;
- `018-generate-deterministic-text-and-construct-saved-transformer-models.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `py_llm_pipeline_explorer_file_structure(49).md`;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the live repository again, establish its own baseline, preserve user changes, implement only Ticket 018, verify the complete change, report actual command results honestly, and create the implementation commit.
