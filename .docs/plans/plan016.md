---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "016"
source_work_item: 016-execute-reference-compatible-transformer-forward-and-backward-passes.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(40).md
behavior_reference: llm_works_file_structure(10).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 016: Execute reference-compatible Transformer forward and backward passes

## Initial checklist

- Confirm Ticket 016 is the only selected work item and that completed Tickets 013, 014, and 015 are present in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(40).md` as the source of truth for current Python source, tests, fixtures, exports, dependencies, and repository conventions.
- Use `SPEC.md`, `CONTEXT.md`, and ADR 0002 for the fixed architecture, mixed-precision, causal-mask, finite-state, gradient, and Logical Training Shard contracts.
- Use `llm_works_file_structure(10).md`, especially TypeScript `src/routes/train-transformer/transformer.ts`, only as formula and analytical-backpropagation evidence.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing those commands as tool-verified in this planning session.
- Limit production changes to the reusable Transformer mathematics in `src/how_llms_work/ml/transformer.py`; do not begin worker, Adam, generation, route, SSE, or persistence work.
- Finish with focused Transformer-math tests, matrix/layout/preprocessing regressions, the full pytest suite, Ruff, Black checking for changed Python files, strict mypy, and final scope inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: Tickets 013, 014, and 015 are complete, and `py_llm_pipeline_explorer_file_structure(40).md` is the current Python Backend source of truth.
2. `016-execute-reference-compatible-transformer-forward-and-backward-passes.md` for required behavior, acceptance criteria, approved test seam, blockers, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(40).md` for the current implementations of matrix primitives, preprocessing, layouts, parameter views, initialization, tests, fixtures, and package configuration.
4. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` for durable Phase 5 numerical, terminology, dtype, mask, shard, and compatibility decisions.
5. `llm_works_file_structure(10).md`, especially TypeScript `src/routes/train-transformer/transformer.ts` and `matrix.ts`, as reference evidence for formulas, residual order, cache requirements, and analytical gradients.
6. Older Python exports, plans, snippets, and assumptions are non-authoritative when they conflict with export `(40)`.

## Work-item summary

Ticket 016 adds the complete framework-free decoder-only Transformer mathematical boundary used by direct deterministic tests and later spawned workers. The implementation must evaluate one ordered token sequence through learned token and absolute positional embeddings, one through six pre-normalized Transformer blocks, final Layer Normalization, the Vocabulary projection head, and stable softmax. It must calculate average next-token cross-entropy and perform the full analytical backward pass into one flat gradient array interpreted by the already-completed canonical parameter layout.

The block architecture is fixed: embedding dimension `32`, two attention heads, head dimension `16`, attention scale `0.25`, feed-forward dimension `128`, population-variance Layer Normalization with epsilon `1e-5`, ReLU, causal masking, and residual connections around both sublayers. Future attention probabilities and attention-score gradients must be exactly zero. All materialized arrays must be C-contiguous `float32`; scalar statistics and approved reductions use Python `float` or NumPy `float64` and are materialized once into completed `float32` tensors.

The same module must calculate one Logical Training Shard by processing its assigned Transformer Training Sequences in fixed order, accumulating unaveraged canonical gradients and Python-float shard loss. An empty shard returns exact zero loss and an all-zero canonical gradient. The ticket stops at this mathematical seam: it does not reduce multiple shards, update weights, create shared memory, start processes, generate text, stream events, validate HTTP requests, or persist a model.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Reported result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, Black, mypy, fixture-extraction, TypeScript execution, or repository command was run while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and report only command results it actually observes.

## Current code observations from the latest source

- `src/how_llms_work/ml/transformer.py` is no longer empty. It currently owns the completed Ticket 014 and 015 boundaries:
  - the fixed Transformer Training Corpus and immutable `TransformerPreprocessingSnapshot`;
  - `TransformerTrainingSequence` and `LogicalTrainingShard`;
  - exact four-shard construction;
  - fixed architecture constants for context `32`, embedding `32`, heads `2`, head dimension `16`, feed-forward `128`, and layer counts `1..6`;
  - the canonical `TransformerParameterLayout` and exact semantic `TransformerParameterViews`;
  - deterministic `Mulberry32` Xavier initialization through `initialize_transformer_parameters()`.
- `TransformerParameterViews` already exposes the exact parameter shapes required by Ticket 016: `tok_emb`, `pos_emb`, block Layer Normalization vectors, query/key/value/output projections, `32 × 128 × 32` feed-forward projections, final normalization vectors, `head_w`, and `head_b`.
- The current flat layout is the sole offset and shape authority. Ticket 016 must represent every parameter gradient in the same flat order and must not define a second gradient-offset table.
- `src/how_llms_work/ml/matrix.py` already provides the strict reusable numerical operations required by the formulas: `matmul`, transposed-left and transposed-right matrix multiplication, `sum_columns`, exact-shape elementwise operations, explicit row-bias addition, scalar multiplication, transpose/copy operations, row slicing/concatenation, transactional `add_in_place`, and causal-mask-aware `stable_row_softmax`.
- Matrix primitives require actual rank-two, C-contiguous `float32` arrays, reject implicit dtype coercion and invalid broadcasting, use approved `float64` accumulation, return independent C-contiguous `float32` outputs, and reject non-finite results. Transformer-specific vector and cache handling must preserve those contracts rather than weakening `matrix.py`.
- `tests/test_transformer.py` currently covers preprocessing, shard-boundary construction, canonical layouts/views, deterministic initialization, public exports, immutability, and sequential/concurrent initialization isolation. It contains no forward, loss, backward, activation-gradient, or shard-gradient tests.
- `tests/test_transformer.py` asserts the complete exact order of `transformer.py.__all__`; any new public mathematical records or functions require a deliberate update that preserves every completed export.
- Existing Transformer fixtures are limited to `transformer_preprocessing_reference.json` and `transformer_layout_initialization_reference.json`. There is no independent forward/backward fixture.
- `src/how_llms_work/ml/transformer_worker.py` remains empty. `src/how_llms_work/routes/train_transformer.py` remains outside the current mathematical boundary. They must stay unchanged for Ticket 016.
- `pyproject.toml` already contains NumPy, pytest, Ruff, Black, and strict mypy. No dependency or lockfile change is needed.
- The TypeScript Reference Implementation provides the approved formula order and cache evidence: per-position Layer Normalization, Q/K/V projections, per-head causal attention, output projection, residual, second Layer Normalization, ReLU feed-forward, second residual, final Layer Normalization, Vocabulary head, stable softmax, cross-entropy, and the complete reverse analytical path.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - Fixed architecture dimensions and supported layer counts are public and exact.
  - Learned token, positional, block, final-normalization, and output-head parameter views exist in canonical shapes.
  - Canonical flat parameter order, offsets, total counts, C-order views, and deterministic finite initialization are implemented and independently tested.
  - Strict `float32` matrix primitives, `float64` accumulation, exact row-bias behavior, stable causal-mask-aware softmax, transactionality, finiteness checks, and independent outputs are implemented and tested.
  - Immutable ordered Transformer Training Sequences and four deterministic Logical Training Shard boundaries exist and are tested.
- **Behavior present but evidence incomplete:**
  - `stable_row_softmax()` already guarantees exact zero probabilities for negative-infinity masks, but no Transformer-level test yet proves the attention code constructs the correct causal mask for both heads at every position.
  - Canonical parameter views can also interpret a flat gradient buffer, but there is no public gradient result type, zero-gradient constructor, or test proving canonical gradient order and isolation.
- **Partially implemented:**
  - The parameter, preprocessing, and matrix foundations required by forward/backward are complete, but `transformer.py` has no Transformer mathematical operations that consume them.
  - Logical shard boundaries exist, but there is no operation that evaluates the sequences within a shard or accumulates shard loss and gradients.
- **Not implemented:**
  - Public forward-pass result/cache records.
  - Token-plus-position embedding lookup and validation.
  - Layer Normalization forward and backward mathematics.
  - Pre-normalized multi-head causal self-attention forward and backward mathematics.
  - Feed-forward and residual forward and backward mathematics.
  - Final normalization, Vocabulary logits, probabilities, and loss.
  - Complete canonical analytical parameter gradients, activation gradients, and repeated-token embedding accumulation.
  - One-sequence and Logical Training Shard public calculation boundaries.
  - Transformer-level non-finite enforcement and input-purity checks.
  - Independent forward/backward/shard fixtures and sequential/concurrent state-isolation tests.
- **Evidence limitation:**
  - The supplied source export is code text rather than an executable repository checkout, so current behavior was inspected but not executed in this planning session.
  - Exact hidden numerical tolerances and selected fixture coordinates must be calibrated during implementation from independent TypeScript or independently calculated evidence; they must not be guessed or generated by the new Python production functions.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer.py` — current `__all__`, architecture constants, `TransformerTrainingSequence`, `LogicalTrainingShard`, `TransformerParameterLayout`, `TransformerParameterViews`, `build_transformer_parameter_views()`, and `initialize_transformer_parameters()`; this is the only expected production file.
2. `src/how_llms_work/ml/matrix.py` — public strict matrix operations and their exact shape, dtype, contiguity, non-overlap, finiteness, and softmax behavior; reuse without weakening.
3. `tests/test_transformer.py` — existing public-export contract, preprocessing/layout/initialization fixtures, naming style, exact-versus-tolerance patterns, and completed regression coverage.
4. `tests/test_matrix.py` — established independent numerical fixture style, transactionality assertions, exact causal-mask zeros, and common output-contract tests.
5. `tests/fixtures/transformer_layout_initialization_reference.json` — selected initialized coordinates, checksums, architecture values, and fixture-provenance conventions used to construct deterministic weights for the new fixture.
6. `tests/fixtures/matrix_reference.json` — approved numeric encoding and tolerance conventions for float32 matrix evidence.
7. `llm_works_file_structure(10).md` — TypeScript `src/routes/train-transformer/transformer.ts`, especially `layerNormForward`, `layerNormBackward`, `blockForward`, `blockBackward`, `forward`, `crossEntropyLoss`, and `backward`.
8. `SPEC.md`, `CONTEXT.md`, and ADR 0002 — confirmed mixed precision, public seam, exact/tolerance policy, causal mask, repeated token, fixed-order shard, and empty-shard decisions.
9. `pyproject.toml` — current pytest, Ruff, Black, NumPy, and strict-mypy configuration; no package change is expected.

## Step 1 — Create independent failure-first Transformer mathematics evidence

**Files and symbols:**

- New `tests/fixtures/transformer_forward_backward_reference.json` — independently captured selected forward, backward, repeated-token, and shard results plus explicit tolerance metadata.
- New `tests/test_transformer_math.py` — focused public-boundary tests for Ticket 016.
- `tests/test_transformer.py` — exact `__all__` expectation updated only after the public seam is defined.

**Purpose:**

Define reviewable acceptance evidence before production mathematics is added, while keeping expected values independent from the Python functions under test.

**Actions:**

- Create one compact deterministic one-layer reference case using the completed canonical Vocabulary size and the completed seed-42 initialized parameter layout.
- Use one ordinary ordered sequence and one controlled repeated-token sequence. Keep all IDs valid and include enough repeated positions to distinguish correct accumulation from advanced-index overwrite behavior.
- Capture or independently calculate selected values from the supplied TypeScript Reference Implementation or a separate test-only scalar derivation that imports no production Transformer forward/backward function.
- Record fixture provenance, architecture, sequence IDs, target IDs, layer count, initialized-weight fixture identity/checksum, and explicit `rtol`/`atol` values per numerical boundary.
- Include selected independent evidence for:
  - token-plus-position embeddings;
  - Layer Normalization means, population variances, normalized values, gamma/beta outputs;
  - Q, K, and V coordinates;
  - both heads' selected attention scores and probabilities;
  - exact upper-triangular future-mask probabilities;
  - attention output, first residual, second normalization, feed-forward pre-activation, ReLU, feed-forward output, and second residual;
  - final normalization, selected logits, selected probabilities, and row sums;
  - average cross-entropy loss;
  - selected output-head, final-normalization, block, positional-embedding, and token-embedding gradient coordinates;
  - selected input-activation gradients and per-head attention-score gradients;
  - exact zero future-position attention-score gradients;
  - repeated-token embedding-row accumulation;
  - a small ordered non-empty shard's accumulated loss and selected canonical gradient coordinates;
  - exact empty-shard zero loss and all-zero gradient behavior.
- Store only selected coordinates and structural metadata needed to catch formula drift; do not serialize every full cache and gradient array into an unreviewable fixture.
- Add failure-first tests for the intended stable public mathematical seam. The test names should describe behavior, not private helper decomposition.
- Use `numpy.testing.assert_allclose(..., strict=True)` or equivalent explicit shape/dtype checks plus named per-boundary tolerances for unrounded arrays. Use exact equality for shapes, dtypes, masks, zero future values, IDs, shard metadata, and canonical flat ordering.

**Guardrails:**

- Do not generate expected values by calling the new Python forward, backward, loss, or shard operation.
- Do not copy expected values from a later Python run after implementation without preserving independent TypeScript or derivation provenance.
- Do not use broad universal tolerances; calibrate the smallest useful explicit tolerance for each boundary.
- Do not assert private helper names, loop order not required by the formulas, BLAS behavior, temporary array identity, or every hidden coordinate.
- Do not modify existing preprocessing, layout, initialization, matrix, Word2Vec, XOR, route, or persistence fixtures.
- Do not add a production fixture-generation API or checksum API.

**Expected result:**

- Ticket 016 has one independent, reviewable numerical fixture and focused public-seam tests.
- The new tests fail only because the public Transformer forward/backward/shard behavior is absent, not because of malformed fixture data or stale imports.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_math.py -q
```

Expected initial result:

- New acceptance tests fail at the missing public mathematical symbols or behavior boundaries while existing Ticket 014–015 tests remain unchanged.

## Step 2 — Define the minimal stable public mathematical records and canonical gradient boundary

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — existing `__all__`; new public forward-pass, backward-pass/canonical-gradient, and Logical Training Shard result records; new public zero-gradient and mathematical operations.
- `tests/test_transformer.py` — exact public-export regression.
- `tests/test_transformer_math.py` — signature, validation, layout, dtype, contiguity, and isolation tests.

**Purpose:**

Create a typed public seam that direct tests and future workers can call without exposing process, route, optimizer, or persistence concerns and without creating a second parameter layout.

**Actions:**

- Extend `transformer.py.__all__` while preserving every current export in its current order. Append only the stable Ticket 016 records and functions.
- Introduce the smallest public result decomposition needed by the approved seam. A suitable boundary should provide:
  - a forward-pass result containing input IDs, stable selected cache arrays required by analytical backpropagation, final logits, and probabilities;
  - one per-block forward record containing the public activation and attention evidence needed by the approved tests;
  - a backward-pass result containing one owning flat canonical gradient array, semantic views interpreted through the existing layout, the input activation gradient, and per-block attention-score gradients required to prove exact future zeros;
  - a Logical Training Shard result containing shard identity, processed sequence count, Python-float accumulated loss, and one canonical gradient result.
- Use frozen, slotted records where consistent with current style, but ensure their NumPy arrays are fresh per call and are never aliases of another call's cache or gradient storage.
- Reuse `TransformerParameterViews` for semantic interpretation of the gradient storage only if the public result clearly documents that those views address gradient coordinates. Do not create a duplicate key, offset, or shape table.
- Allocate each fresh gradient as an exact-length one-dimensional C-contiguous `float32` array initialized to zero, then build semantic views with `build_transformer_parameter_views()`.
- Validate public inputs before calculation:
  - canonical layout and parameter views are structurally consistent;
  - all weight arrays are `float32`, C-contiguous, correct shape, and finite;
  - token and target IDs are non-Boolean Python integers within the canonical Vocabulary range;
  - sequence length is nonzero, does not exceed context length `32`, and target count equals the forward count;
  - a Transformer Training Sequence used by a shard has the approved fixed length `16` for both input and targets;
  - a Logical Training Shard has a valid index and half-open range within the supplied ordered sequence collection.
- Keep validation side-effect-free and reject malformed or non-finite inputs before returning a successful result.

**Guardrails:**

- Do not define a second `TransformerConfig`; derive dimensions and Vocabulary size from existing constants and the existing canonical layout/views.
- Do not add worker IDs, epochs, shared-memory names, request fields, optimizer state, generation controls, filesystem paths, or SSE payload fields to mathematical records.
- Do not expose private cache implementation names as the test seam. Public records should represent stable mathematical observations and dependencies, not local loop details.
- Do not make caches or gradients module-level singletons or reusable global scratch buffers.
- Do not mutate caller-owned weight storage during validation or calculation.
- Do not change current layout, initialization, or preprocessing semantics merely to simplify the new records.

**Expected result:**

- The module has one stable typed numerical seam for one forward pass, one backward pass, and one Logical Training Shard.
- Every gradient coordinate is interpreted by the completed canonical layout, and separate calls own separate caches and gradients.

**Verification:**

```powershell
poetry run pytest tests/test_transformer.py tests/test_transformer_math.py -q -k "public or signature or gradient or layout or validation or dtype or contiguous"
```

Expected result:

- Public exports preserve Tickets 014–015 and include only the deliberate Ticket 016 seam.
- Invalid IDs, lengths, layouts, shapes, dtypes, non-contiguous arrays, and non-finite weights fail predictably.
- Fresh zero gradients have exact canonical length, shape views, `float32` dtype, C contiguity, and no cross-call aliasing.

## Step 3 — Implement per-position Layer Normalization and embedding input construction

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — private checked Layer Normalization forward/backward helpers; public forward/backward operations using those helpers.
- `tests/test_transformer_math.py` — embedding, normalization, mixed-precision, purity, and finite-state tests.
- `tests/fixtures/transformer_forward_backward_reference.json` — selected expected values and tolerances.

**Purpose:**

Establish the first and final normalization primitives and the exact learned token-plus-position input used by every later Transformer stage.

**Actions:**

- Build the initial activation by reading the same token-embedding row for every occurrence of a Vocabulary Token and adding the learned absolute positional row for each sequence position.
- Materialize the complete initial activation once as a fresh C-contiguous `float32` matrix with shape `(sequence_length, 32)`.
- Implement Layer Normalization independently per position across exactly 32 features:
  - calculate the mean in Python float or NumPy `float64`;
  - calculate population variance with divisor `32`, not sample variance;
  - use epsilon `1e-5` inside the inverse standard deviation;
  - materialize means, variances, normalized values, and affine output as finite C-contiguous `float32` arrays;
  - apply gamma and beta exactly without unapproved hidden broadcasting.
- Implement the analytical Layer Normalization backward formula using the cached normalized values and variance. Accumulate `dGamma` and `dBeta` into their canonical gradient views and return a fresh finite C-contiguous `float32` input gradient.
- Ensure final Layer Normalization uses the last block output, or the initial embedding activation when a controlled zero-block helper case is used internally for testing only; production layouts remain one through six layers.
- Add exact shape/dtype/contiguity checks and selected fixture comparisons for first-block and final normalization.
- Snapshot caller weight bytes and input ID values before calls and prove they remain unchanged.

**Guardrails:**

- Do not use `ddof=1`, batch statistics, running statistics, or normalization across positions.
- Do not keep means or variances as materialized `float64` arrays; completed cache tensors are `float32`.
- Do not rely on implicit NumPy broadcasting that bypasses the project's shape rules. Use explicit row handling or the approved row-bias boundary.
- Do not mutate gamma, beta, source activation, cached normalized values, token IDs, or positional embeddings.
- Do not suppress non-finite intermediate results; reject the complete operation before success.

**Expected result:**

- Learned token and absolute position embeddings produce the reference initial activation.
- Layer Normalization forward and backward match independent selected fixtures, use the exact population-variance and epsilon rules, and preserve all inputs.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_math.py -q -k "embedding or layer_norm or normalization or population or epsilon or gamma or beta or purity"
```

Expected result:

- Selected embedding and normalization values match within explicit tight tolerances.
- Means, variances, outputs, and gradients have exact shapes, `float32` dtype, C contiguity, finiteness, and no input mutation.

## Step 4 — Implement pre-normalized two-head causal self-attention forward behavior

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — block-forward attention calculations and cache population.
- `src/how_llms_work/ml/matrix.py` — reuse only; no change expected.
- `tests/test_transformer_math.py` — Q/K/V, scores, causal mask, probabilities, values, output projection, residual, and state-isolation tests.

**Purpose:**

Complete the reference-compatible attention sublayer with exact causal semantics and reviewable educational intermediate state.

**Actions:**

- Project first-normalized activations through the approved `wQ/bQ`, `wK/bK`, and `wV/bV` parameters using existing strict matrix operations.
- Split the 32 features into exactly two heads of 16 features each without changing feature order.
- For each head, calculate scaled dot-product scores with scale exactly `0.25`.
- Materialize an exact upper-triangular causal mask using intentional negative infinity for every future key position `j > i`.
- Call the existing `stable_row_softmax()` on each head's score matrix. Preserve the score matrix unchanged and materialize probabilities as finite C-contiguous `float32`.
- Store attention scores and attention probabilities in a stable per-block representation with shape `(2, sequence_length, sequence_length)` or an equivalently explicit public structure. Future probabilities must be exactly `0.0`, not merely small.
- Calculate each head's weighted values only from allowed positions, concatenate heads in original feature order, apply `wO/bO`, and add the first residual connection to the block input.
- Validate Q, K, V, scores excluding intentional negative infinity, probabilities, weighted values, projected attention output, and residual output at the earliest complete boundary.
- Add tests that change a future token while holding an earlier prefix fixed and prove earlier-position attention results remain unchanged, in addition to exact mask assertions.

**Guardrails:**

- Do not replace negative infinity with a large finite sentinel.
- Do not normalize a row containing future positions before masking.
- Do not permit an all-masked row; every position must attend to itself or earlier positions.
- Do not apply scale twice or omit it.
- Do not reorder heads or interleave their feature slices incorrectly.
- Do not mutate Layer Normalization output, score matrices, parameter arrays, or another call's cache.
- Do not add optimized fused attention, dropout, flash attention, rotary embeddings, or framework operations.

**Expected result:**

- Both attention heads match independent score/probability fixtures.
- Every future probability is exactly zero, every row sums to one within the named tolerance, and the projected attention residual matches the reference architecture.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_math.py -q -k "attention or causal or mask or score or probability or qkv or residual"
poetry run pytest tests/test_matrix.py -q -k "softmax or mask or causal"
```

Expected result:

- Transformer-level attention evidence and the existing strict softmax regressions pass without a `matrix.py` behavior change.

## Step 5 — Complete the feed-forward blocks, final head, probabilities, and cross-entropy

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — block feed-forward forward path, complete multi-block forward pass, final Layer Normalization, Vocabulary head, stable probabilities, and cross-entropy operation.
- `tests/test_transformer_math.py` — block order, ReLU, residual, multi-layer, logits, probabilities, loss, purity, and non-finite tests.

**Purpose:**

Finish the complete decoder-only forward pass and its training objective before implementing the reverse gradient path.

**Actions:**

- Apply the second pre-normalization to the first residual output.
- Calculate the position-wise feed-forward network exactly as `32 → 128 → 32`:
  - first projection and bias;
  - ReLU with zero for values not greater than zero;
  - second projection and bias;
  - second residual addition to the first residual output.
- Repeat the exact block structure for every block in ascending layout order. Only `num_layers` may change the architecture.
- Apply final Layer Normalization to the last block output.
- Calculate Vocabulary logits with `head_w/head_b`, producing shape `(sequence_length, vocabulary_size)`.
- Use `stable_row_softmax()` to produce a finite probability distribution for every sequence position without mutating logits.
- Implement average next-token cross-entropy:
  - select the probability for each target ID;
  - add exactly `1e-10` inside the logarithm;
  - accumulate the scalar loss in Python float/NumPy `float64`;
  - divide exactly once by the represented target-position count;
  - reject non-finite probabilities or loss.
- Preserve a reusable low-level forward boundary for token-ID sequences up to context length `32`; shard processing continues to require the fixed 16-position Transformer Training Sequence.
- Add representative one-layer and multi-layer structural tests and selected fixture comparisons for residuals, feed-forward values, final normalization, logits, probabilities, and loss.

**Guardrails:**

- Do not use GELU, tanh, dropout, a hidden bias broadcast, weight tying, or a different feed-forward width.
- Do not apply Layer Normalization after a sublayer instead of before it.
- Do not average loss in both the sequence and shard boundaries.
- Do not round logits, probabilities, or hidden loss values to six decimals; public event rounding belongs to a later ticket.
- Do not mutate logits while computing probabilities or mutate probabilities while computing loss.
- Do not add optimizer, generation, sampling, or final-evaluation orchestration.

**Expected result:**

- The full forward pass matches selected independent activations and logits.
- Every probability is finite and non-negative, each row sums to one, and average cross-entropy matches the independent fixture.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_math.py -q -k "feed_forward or relu or block or logits or probability or cross_entropy or loss or multi_layer"
```

Expected result:

- Block order, ReLU behavior, both residual paths, final normalization, output head, stable probabilities, and exact loss semantics pass.

## Step 6 — Implement the complete analytical backward pass in canonical parameter order

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — public backward operation and private block/attention/normalization reverse helpers.
- `tests/test_transformer_math.py` — complete gradient, finite-difference spot checks, selected fixture coordinates, residual-path, mask-gradient, and repeated-token tests.
- `tests/fixtures/transformer_forward_backward_reference.json` — selected independent backward values and tolerances.

**Purpose:**

Produce one complete unaveraged analytical gradient for a sequence without automatic differentiation and with exact causal and repeated-token behavior.

**Actions:**

- Start from softmax plus average cross-entropy using the reference derivative:
  - copy probabilities divided by sequence length;
  - subtract `1 / sequence_length` at each target coordinate;
  - materialize `dLogits` as finite C-contiguous `float32`.
- Accumulate output-head bias and weight gradients into the canonical gradient views and propagate the activation gradient through `head_w`.
- Backpropagate through final Layer Normalization into the final block output.
- Traverse blocks in reverse order and preserve both residual paths:
  - split the output gradient into the direct residual and feed-forward branch;
  - calculate `ff2` bias/weight gradients and propagate through `ff2_w`;
  - apply the exact ReLU derivative from cached `ff1` pre-activations;
  - calculate `ff1` bias/weight gradients and propagate through `ff1_w`;
  - backpropagate through second Layer Normalization and combine with the direct residual;
  - calculate output-projection bias/weight gradients and propagate through `wO`;
  - backpropagate attention weighted values into attention probabilities and V;
  - apply the row-wise softmax Jacobian only over allowed `j <= i` entries;
  - include scale `0.25` exactly in score gradients;
  - leave every future `dScore` coordinate exactly zero;
  - propagate score gradients into Q and K;
  - calculate Q/K/V biases and weights and combine their activation gradients;
  - backpropagate through first Layer Normalization and add the block-input residual gradient.
- Accumulate positional-embedding gradients by exact sequence position.
- Accumulate token-embedding gradients for every position into the row addressed by its token ID. Use explicit ordered accumulation or another operation that provably accumulates repeated indices; do not use advanced-index assignment that overwrites duplicate rows.
- Return the complete owning flat canonical gradient and the minimum public activation-gradient evidence required by the approved seam, including input gradients and per-block attention-score gradients.
- Validate every completed activation gradient and the complete flat parameter gradient for dtype, C contiguity, shape, and finiteness before success.
- Add independent finite-difference spot checks for a small selected set of parameter coordinates across different parameter families. Keep finite differences as test evidence only and use an explicit tolerance suitable for float32 analytical gradients.

**Guardrails:**

- Do not use PyTorch, TensorFlow, JAX, autograd, numerical differentiation in production, or a hidden machine-learning framework.
- Do not average, clip, normalize, regularize, or update gradients in this ticket.
- Do not calculate gradients in a separate semantic structure with a different ordering from the canonical flat layout.
- Do not mutate forward caches, weights, target IDs, input IDs, or the preprocessing snapshot.
- Do not accidentally reuse a gradient from another sequence or call.
- Do not rely only on finite differences; preserve independent TypeScript/analytical fixture coordinates and exact structural assertions.
- Do not weaken exact future-gradient zeros into an `allclose` assertion.

**Expected result:**

- The analytical gradient covers every canonical parameter family and matches selected independent fixtures and finite-difference checks.
- Repeated token IDs accumulate all contributions into one token-embedding row.
- Every future attention-score gradient is exactly zero.
- Both residual paths and every normalization, projection, embedding, and output-head derivative are exercised.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_math.py -q -k "backward or gradient or repeated or finite_difference or residual or future or canonical"
```

Expected result:

- Selected gradient coordinates match explicit tolerances.
- Gradient layout, dtype, contiguity, exact mask zeros, repeated-token accumulation, input purity, and finite-state checks pass.

## Step 7 — Implement fixed-order Logical Training Shard calculation

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — public Logical Training Shard calculation over an ordered sequence collection.
- `tests/test_transformer_math.py` — ordered accumulation, unaveraged gradients, loss, empty shard, validation, and isolation tests.
- `tests/fixtures/transformer_forward_backward_reference.json` — selected small-shard independent evidence.

**Purpose:**

Provide the exact worker-independent mathematical unit that later spawned workers will execute.

**Actions:**

- Accept an immutable ordered collection of `TransformerTrainingSequence` values, one validated `LogicalTrainingShard`, and canonical weight views.
- Allocate one fresh all-zero canonical gradient for the call.
- Process sequence indices from `start_index` through `stop_index - 1` in ascending order with no shuffling, batching, parallelism, or completion-order behavior.
- For each sequence:
  - run the same public-compatible forward calculation;
  - calculate its average next-token cross-entropy;
  - run the same analytical backward calculation into the shard-owned gradient accumulator.
- Accumulate each sequence loss in Python floating-point order. Return the sum, not the average across sequences.
- Preserve each sequence's internally averaged position loss while leaving the complete shard gradient unaveraged across sequences.
- For an empty shard, return:
  - exact Python float `0.0` loss;
  - processed sequence count `0`;
  - a complete canonical all-zero `float32` gradient;
  - no forward cache retained from another call.
- Validate the complete shard loss and complete gradient before returning success.
- Test a small controlled non-empty shard against independent selected loss/gradient evidence and against exact fixed-order metadata.
- Add a test that reversing the supplied sequence collection changes the fixture outcome, proving order is not silently discarded, while the production path preserves the provided order.

**Guardrails:**

- Do not perform Ordered Gradient Reduction across the four shards; that is a later parent-side ticket.
- Do not divide shard gradients or shard loss by sequence count.
- Do not start a process, allocate shared memory, send protocol messages, or depend on `transformer_worker.py`.
- Do not read the global preprocessing snapshot inside the low-level shard operation when an ordered sequence collection is supplied; workers and direct tests must call the same calculation boundary.
- Do not permit a shard range outside the supplied collection or a mismatched shard index.
- Do not retain per-sequence caches after the sequence has been accumulated into the shard result.

**Expected result:**

- One Logical Training Shard produces a deterministic unaveraged canonical gradient and accumulated loss from fixed-order Training Sequences.
- Empty shards are exact, finite, and side-effect-free.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_math.py -q -k "shard or empty or fixed_order or unaveraged or sequence_count"
```

Expected result:

- Non-empty shard selected values match independent fixtures.
- Empty shard behavior is exact.
- No worker count, CPU count, process, shared-memory, optimizer, or route behavior affects the result.

## Step 8 — Enforce finite-state failure, purity, and sequential/concurrent isolation

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — complete Ticket 016 validation and local state ownership.
- `tests/test_transformer_math.py` — non-finite injection, byte-preservation, aliasing, repeated-call, and threaded-call tests.
- `tests/test_transformer.py` and `tests/test_matrix.py` — completed foundation regressions.

**Purpose:**

Prove no successful mathematical result can contain corrupted numerical state and no call can mutate or contaminate another call.

**Actions:**

- Add public-boundary failure tests for NaN and positive/negative infinity in representative weight families.
- Use narrow monkeypatching of an existing public matrix boundary only when needed to simulate a non-finite activation, probability, or gradient that valid finite weights cannot naturally produce. Assert the Transformer boundary rejects the result before success.
- Preserve byte snapshots of complete weight storage, input tuples, target tuples, and supplied Training Sequence collections across successful and failed calls.
- Verify every cache, activation-gradient array, and canonical gradient is independent of all weight storage and of corresponding arrays returned by another call.
- Run two sequential identical calls and verify equivalent numerical values but no shared mutable array memory.
- Run concurrent read-only calls against the same finite weight views. Verify each call receives independent caches and gradients and that the shared weights remain byte-for-byte unchanged.
- Mutate a returned cache or gradient from one completed call and prove it cannot change another completed result or a later recomputation.
- Verify failed operations return no partially committed public result and leave all supplied inputs unchanged.
- Re-run existing preprocessing, layout, initialization, and matrix tests to ensure the mathematical implementation does not weaken completed contracts.

**Guardrails:**

- Do not add a production lock around pure numerical calls; local allocations and read-only weight use should provide isolation.
- Do not make caller-owned weight arrays globally read-only as a side effect of a direct numerical call; later worker code owns its own read-only shared-memory views.
- Do not test elapsed timing or rely on thread scheduling order.
- Do not patch private Transformer helpers merely to assert their names or decomposition.
- Do not accept NaN equality in `assert_allclose`; successful results must contain no NaN or infinity.

**Expected result:**

- Every successful result is completely finite, local to its call, and non-aliased across sequential and concurrent calls.
- Every failure is transactional with respect to caller-owned inputs.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_math.py -q -k "finite or nan or infinity or purity or mutation or isolation or concurrent or alias"
poetry run pytest tests/test_matrix.py tests/test_transformer.py tests/test_transformer_math.py -q
```

Expected result:

- Finite-state, byte-preservation, non-aliasing, sequential isolation, concurrent isolation, and all completed foundation regressions pass.

## Step 9 — Finalize formatting, typing, regression verification, and scope-only diff review

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — completed Ticket 016 production mathematical boundary.
- `tests/test_transformer.py` — preserved and extended public-export regression.
- `tests/test_transformer_math.py` — focused Ticket 016 public acceptance suite.
- `tests/fixtures/transformer_forward_backward_reference.json` — independent selected numerical evidence.
- Repository status and diff — scope validation only.

**Purpose:**

Prove Ticket 016 is complete, typed, formatted, regression-safe, and restricted to reusable forward/backward/shard mathematics.

**Actions:**

- Format only changed Python files with Black if needed.
- Run the focused Transformer mathematics suite first.
- Run matrix, preprocessing, layout, and initialization regressions together with the new suite.
- Run the complete pytest suite exactly once after focused tests are green.
- Run Ruff, Black check mode for changed Python files, and strict mypy.
- Inspect `git diff --check`, `git diff`, and `git status --short`.
- Confirm no dependency manifest, lockfile, worker, route, schema, application registration, SSE, persistence, `.data`, frontend, specification, context, ADR, or completed fixture changed.
- Confirm the new fixture contains independent provenance and no timestamp, local absolute path, CPU count, shared-memory name, process ID, or machine-specific data.
- Confirm the final public export retains every completed preprocessing/layout/initialization symbol and adds only the deliberate Ticket 016 seam.
- Record actual command outputs honestly; do not claim success for any command not executed.

**Guardrails:**

- Do not fix unrelated failures or reformat unrelated files.
- Do not weaken explicit fixture tolerances or exact mask assertions to make the suite pass.
- Do not add optimizer, generation, worker, shared-memory, request, route, SSE, persistence, or frontend behavior.
- Do not modify `matrix.py` unless a concrete missing general primitive is proven during implementation. If such a change is genuinely required, keep it stateless and general, update `tests/test_matrix.py` and `matrix_reference.json`, and document why Ticket 013's existing boundary was insufficient.
- Do not create the implementation commit until every required command passes and the final scope is correct.

**Expected result:**

- Ticket 016 is ready for implementation review with one production module, one preserved export regression, one focused Transformer-math test module, and one independent numerical fixture changed.

**Verification:**

- Execute every command in the focused and full verification sections below and inspect the final diff manually.

## Focused verification plan

Run from the backend project root:

```powershell
poetry run pytest tests/test_transformer_math.py -q

poetry run pytest `
    tests/test_matrix.py `
    tests/test_transformer.py `
    tests/test_transformer_math.py `
    -q

poetry run black --check `
    src/how_llms_work/ml/transformer.py `
    tests/test_transformer.py `
    tests/test_transformer_math.py

poetry run mypy src
```

Expected result:

- The complete public forward, loss, backward, repeated-token, causal-mask, gradient-layout, Logical Training Shard, finite-state, purity, and isolation tests pass.
- Completed matrix, preprocessing, layout, and initialization tests remain green.
- Changed Python files satisfy Black.
- Strict mypy reports no issues in `src`.

## Full verification plan

Run once after focused verification passes:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
git diff --check
git diff
git status --short
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.
- `git diff --check` reports no whitespace errors.
- The final diff contains only planned Ticket 016 files and any narrowly justified conditional matrix test/source change.

## Manual acceptance checklist

- [ ] The latest Python export `(40)` remains the current-code authority throughout implementation.
- [ ] Tickets 013, 014, and 015 remain intact and their tests pass unchanged except for the deliberate public-export extension.
- [ ] Forward input uses learned token embeddings and learned absolute positional embeddings.
- [ ] Repeated token IDs read the same embedding row and accumulate every positional gradient contribution into that row.
- [ ] Every block uses `LayerNorm → causal attention → residual → LayerNorm → 32→128→32 ReLU feed-forward → residual`.
- [ ] Layer Normalization is per position across 32 features, uses population variance and epsilon `1e-5`, and applies gamma and beta exactly.
- [ ] Attention uses exactly two heads, 16 features per head, and scale `0.25`.
- [ ] Every future attention score is masked with negative infinity before softmax.
- [ ] Every future attention probability is exactly zero.
- [ ] Every future attention-score gradient is exactly zero.
- [ ] Attention score matrices remain unchanged by softmax.
- [ ] The feed-forward width and ReLU derivative match the reference.
- [ ] Final Layer Normalization, Vocabulary logits, and probabilities have exact expected shapes.
- [ ] Every probability row is finite, non-negative, and sums to one within the named tolerance.
- [ ] Cross-entropy uses `1e-10` and divides exactly once by target-position count.
- [ ] Backward returns one complete flat canonical gradient in the Ticket 015 order.
- [ ] Output head, final normalization, every block, position embeddings, and token embeddings receive analytical gradients.
- [ ] All materialized weights, activations, caches, logits, probabilities, attention matrices, activation gradients, and parameter gradients are C-contiguous `float32` where represented as arrays.
- [ ] Scalar statistics and approved reductions use Python float or NumPy `float64` before completed `float32` materialization.
- [ ] Forward, loss, backward, and shard calculations leave weights, input IDs, target IDs, Training Sequences, and preprocessing unchanged.
- [ ] A Logical Training Shard processes sequences in ascending fixed order.
- [ ] Shard loss is accumulated but not averaged across sequences.
- [ ] Shard gradients are unaveraged across sequences.
- [ ] An empty shard returns exact `0.0` loss and a complete all-zero canonical gradient.
- [ ] NaN or infinity in weights, activations, probabilities, loss, or gradients prevents a successful result.
- [ ] Independent fixture values cover selected activations, normalization statistics, attention scores/probabilities, residuals, logits, loss, gradients, repeated-token behavior, and empty-shard behavior.
- [ ] Exact assertions protect shapes, dtypes, masks, future zeros, canonical ordering, IDs, and shard metadata.
- [ ] Explicit per-boundary tolerances protect only approved unrounded floating-point comparisons.
- [ ] Sequential and concurrent calls return independent caches and gradients and cannot mutate another call's state.
- [ ] No Adam, gradient reduction across shards, generation, process, shared memory, HTTP, SSE, persistence, frontend, dependency, or lockfile work is included.
- [ ] Actual pytest, Ruff, Black, mypy, and diff results are recorded honestly.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer.py
tests/test_transformer.py
tests/test_transformer_math.py
tests/fixtures/transformer_forward_backward_reference.json
```

Conditionally changed only if implementation proves the completed Ticket 013 matrix boundary lacks one genuinely reusable primitive that cannot safely remain Transformer-private:

```text
src/how_llms_work/ml/matrix.py
tests/test_matrix.py
tests/fixtures/matrix_reference.json
```

The default expectation is no `matrix.py` change.

Optional only if the repository deliberately re-exports public ML symbols there; the current export does not, so leave unchanged by default:

```text
src/how_llms_work/ml/__init__.py
```

No package or lockfile change is expected.

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/word2vec.py
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
tests/fixtures/math_utils_reference.json
tests/fixtures/transformer_preprocessing_reference.json
tests/fixtures/transformer_layout_initialization_reference.json
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
016-execute-reference-compatible-transformer-forward-and-backward-passes.md
```

## Risk notes and safeguards

1. **Risk:** The implementation creates a separate gradient offset table that drifts from Ticket 015.
   - **Safeguard:** Allocate one flat gradient block and interpret it exclusively with the existing canonical layout and view builder.

2. **Risk:** NumPy silently broadcasts gamma, beta, or biases in ways that bypass explicit shape rules.
   - **Safeguard:** Validate exact shapes and use explicit row operations or the approved row-bias primitive; add wrong-shape tests.

3. **Risk:** Completed caches become `float64` because NumPy reductions or expressions promote dtype.
   - **Safeguard:** Use `float64` only for approved scalar/reduction stages and explicitly materialize each completed tensor as C-contiguous `float32` with dtype assertions.

4. **Risk:** Layer Normalization uses sample variance or normalizes across positions.
   - **Safeguard:** Protect per-position means, population variances, epsilon behavior, and selected outputs with independent fixtures.

5. **Risk:** The causal mask is applied after softmax or represented by a finite sentinel.
   - **Safeguard:** Construct negative-infinity future entries before `stable_row_softmax()` and assert exact upper-triangular zeros.

6. **Risk:** The attention scale is applied twice, omitted, or applied in the wrong backward location.
   - **Safeguard:** Include selected score and `dScore` fixtures and exact architecture constant checks for scale `0.25`.

7. **Risk:** Head reshaping reorders features or mixes head outputs.
   - **Safeguard:** Use explicit stable head slices and fixture coordinates from both heads and concatenation boundaries.

8. **Risk:** Softmax mutates score matrices needed by backward or fixture inspection.
   - **Safeguard:** Reuse the completed pure softmax boundary and assert score bytes remain unchanged.

9. **Risk:** Future gradients become tiny nonzero values through a dense softmax Jacobian.
   - **Safeguard:** Iterate or mask only allowed `j <= i` coordinates and assert exact zero future `dScore` values.

10. **Risk:** Residual gradients are dropped, duplicated, or attached to the wrong branch.
    - **Safeguard:** Add controlled residual-path fixtures and selected finite-difference checks across both attention and feed-forward branches.

11. **Risk:** ReLU backward uses the post-ReLU value incorrectly at zero or a different activation.
    - **Safeguard:** Cache the approved pre-activation and test positive, negative, and exact-zero coordinates.

12. **Risk:** Cross-entropy is averaged twice or uses shard sequence count instead of target-position count.
    - **Safeguard:** Separate sequence average loss from shard sum and protect both with independent one-sequence and multi-sequence fixtures.

13. **Risk:** Backward overwrites repeated token rows through advanced indexing.
    - **Safeguard:** Use explicit ordered accumulation or a provably repeated-index-safe operation and include a repeated-token fixture that fails under overwrite semantics.

14. **Risk:** Positional gradients are incorrectly combined by token ID.
    - **Safeguard:** Assert each position row receives its own input-gradient contribution while token rows combine by token ID.

15. **Risk:** A shard clears gradients per sequence or returns only the final sequence gradient.
    - **Safeguard:** Allocate once per shard, accumulate every sequence, and compare selected shard coordinates against independent multi-sequence evidence.

16. **Risk:** Empty shards accidentally reuse stale gradient memory.
    - **Safeguard:** Allocate and explicitly zero the complete canonical block before processing; assert exact all-zero bytes for an empty shard.

17. **Risk:** Non-finite intermediate arrays are detected only after partial public state is returned.
    - **Safeguard:** Validate every completed boundary and the complete result before return; failure tests preserve all inputs.

18. **Risk:** Forward or backward mutates weight views because gradients and weights share storage.
    - **Safeguard:** Allocate separate owning gradient storage, assert no memory overlap with weights, and compare weight bytes before and after every success/failure case.

19. **Risk:** Public cache records expose excessive private implementation detail and become brittle.
    - **Safeguard:** Expose only stable mathematical observations and dependencies required by analytical backward and approved tests; avoid local loop or temporary names.

20. **Risk:** Concurrent calls share a mutable cache, gradient, or scratch workspace.
    - **Safeguard:** Keep all arrays call-local and add threaded read-only calls plus cross-result memory-overlap checks.

21. **Risk:** Fixture expectations are circular because they are generated by the Python implementation under test.
    - **Safeguard:** Preserve TypeScript or independent scalar-calculation provenance and prohibit production forward/backward imports in fixture generation.

22. **Risk:** Tolerances are broad enough to hide wrong formulas.
    - **Safeguard:** Calibrate explicit per-boundary tolerances, combine them with exact masks/shapes/dtypes, and use selected finite-difference checks.

23. **Risk:** Full actual-corpus shard tests make the ordinary suite prohibitively slow.
    - **Safeguard:** Use small controlled ordered sequence collections for detailed fixtures, retain one bounded public integration case, and avoid benchmarking or full-epoch training in Ticket 016.

24. **Risk:** The implementation drifts into Adam, Ordered Gradient Reduction, workers, generation, routing, or persistence because the TypeScript file colocates those concerns.
    - **Safeguard:** Enforce the expected-file list and leave `transformer_worker.py`, routes, schemas, application registration, SSE, `.data`, and frontend files unchanged.

25. **Risk:** New public symbols remove or reorder completed Ticket 014–015 exports.
    - **Safeguard:** Update the exact `__all__` regression by appending only deliberate Ticket 016 symbols and retaining every existing entry.

26. **Risk:** Broad formatting or unrelated cleanup obscures the mathematical change.
    - **Safeguard:** Format only changed Python files, inspect `git diff --check`, and reject unrelated churn before commit.

27. **Risk:** User-reported baseline is mistaken for current-session verification.
    - **Safeguard:** Re-run the baseline in `implement-prompt` before editing and report actual output honestly.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Execute Transformer forward and backward passes
```

Commit body should mention:

- learned token and absolute positional embeddings;
- per-position population-variance Layer Normalization with epsilon `1e-5`;
- two-head causal attention with head dimension `16`, scale `0.25`, exact future probability and gradient zeros;
- pre-normalized residual attention and `32 → 128 → 32` ReLU feed-forward blocks;
- final normalization, Vocabulary logits, stable probabilities, and average cross-entropy with `1e-10`;
- complete analytical canonical gradients, residual paths, and repeated-token accumulation;
- fixed-order unaveraged Logical Training Shard loss and gradients, including exact empty-shard behavior;
- `float32` C-contiguous materialized state, `float64` approved reductions, finite-state enforcement, purity, and sequential/concurrent isolation;
- independent selected forward/backward/shard fixtures with exact structural assertions and explicit tight tolerances;
- no Adam, generation, worker, shared-memory, route, SSE, persistence, frontend, dependency, or lockfile changes;
- the exact focused, full pytest, Ruff, Black, mypy, and diff commands actually executed and their observed results.

Do not create the commit during `to-plan-prompt`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan016.md`;
- `016-execute-reference-compatible-transformer-forward-and-backward-passes.md`;
- completed blocker Tickets 013, 014, and 015, or their current `matrix.py`, `transformer.py`, tests, and fixtures as equivalent evidence;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `py_llm_pipeline_explorer_file_structure(40).md`;
- `llm_works_file_structure(10).md`.

`implement-prompt` must inspect the real repository again, establish its own baseline before editing, preserve user changes, implement only Ticket 016, create independent failure-first forward/backward/shard evidence, reuse the completed matrix/preprocessing/layout/initialization boundaries, run focused and full verification, report actual command results honestly, inspect final scope, and create the implementation commit only after all required checks pass.
