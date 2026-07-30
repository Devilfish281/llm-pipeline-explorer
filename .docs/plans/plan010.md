---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "010"
source_work_item: 010-construct-exact-embedding-results-and-saved-embedding-models.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(22).md
behavior_reference: llm_works_file_structure(8).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 010: Construct exact Embedding Results and Saved Embedding Models

## Initial checklist

- Confirm Ticket 010 is the only selected work item and that its Ticket 009 blocker is satisfied by the completed deterministic Skip-gram implementation and tests in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(22).md` as the source of truth for the current Python Backend; do not let older snippets, exports, or plans override it.
- Use `llm_works_file_structure(8).md`, especially TypeScript `src/routes/train-embed/train.ts` and `src/server/lib/math.ts`, only as behavior evidence for result construction.
- Preserve the user-reported passing pytest, Ruff, and strict mypy baseline without describing it as verified in this planning session.
- Limit production changes to reusable Word2Vec result and Saved Embedding Model construction in `src/how_llms_work/ml/word2vec.py`.
- Add independent exact fixtures and focused tests at the approved public Word2Vec module boundary.
- Finish with focused result tests, affected Word2Vec and persistence regressions, Ruff, strict mypy, and the complete pytest suite.

## Source-of-truth hierarchy

1. The user's latest explicit direction to convert the TypeScript behavior to Python and treat `py_llm_pipeline_explorer_file_structure(22).md` as the current-code source of truth.
2. `010-construct-exact-embedding-results-and-saved-embedding-models.md` for the required behavior, acceptance criteria, approved test seam, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(22).md` for the current implementation, tests, fixtures, paths, dependencies, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and ADR 0001 as recorded by the approved Phase 4 handoff for durable compatibility decisions and canonical terminology.
5. `llm_works_file_structure(8).md`, especially TypeScript `src/routes/train-embed/train.ts` and `src/server/lib/math.ts`, as reference behavior only.
6. Older exports, snippets, tickets, and plans are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 010 converts one finite `CompletedEmbeddingTraining` value into two distinct public objects:

1. an **Embedding Result** containing only the Query Word-selected display data required by the unchanged frontend; and
2. a complete **Saved Embedding Model** suitable for the already-implemented Ticket 008 persistence boundary.

The implementation must derive every public Word Embedding only from the corresponding input-weight row, round coordinates to six decimals with TypeScript-compatible semantics, preserve Query Word positions and duplicates, lower-case only for exact Vocabulary lookup and BPE warning analysis, and preserve original submitted text wherever the ticket requires query-facing display or warning text.

Recognized positions produce selected embeddings and Nearest Neighbor groups. Every pair of recognized positions produces one Similarity Pair, including repeated positions that resolve to the same Vocabulary Token. Unrecognized and multi-token positions each produce exactly one ordered warning and are omitted from embedding-derived collections without failing the completed run. An all-unrecognized request still evaluates the predefined Vector Analogies.

Nearest Neighbor and Similarity Pair calculations use six-decimal public vectors. Neighbor scores are rounded to two decimals before ranking; descending ties retain Vocabulary order. Vector Analogies preserve the confirmed mixed-precision behavior: raw input-weight rows form `a - b + c`, six-decimal public vectors are the candidates, the three source tokens are excluded, and equal unrounded candidate scores retain the first Vocabulary candidate encountered.

The Saved Embedding Model must contain exactly `type`, `dimensions`, `vocab`, `merges`, and `embeddings`, preserve the complete ordered Vocabulary and Merge Table, and include a six-decimal public vector for every Vocabulary Token. Neither public object may contain NaN or infinity, expose output weights or unrounded matrices, or share mutable construction state with a later call.

This ticket does not implement request validation, FastAPI routing, SSE framing, presentation delays, disconnect handling, persistence, model loading, frontend changes, or Transformer behavior.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, or mypy command was executed while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing.

## Current code observations from the latest source

- `src/how_llms_work/ml/word2vec.py` now contains the completed Ticket 007 and Ticket 009 foundations:
  - immutable `Word2VecPreprocessing`;
  - the complete ordered `merges`, `vocabulary`, and `token_indices`;
  - `get_word2vec_preprocessing()`;
  - deterministic `EmbeddingTrainingRun`;
  - finite `CompletedEmbeddingTraining` values containing `dimensions`, ordered `vocabulary`, copied `input_weights`, copied `output_weights`, and final numerical metadata.
- `CompletedEmbeddingTraining` copies both matrices and the run-owned Training Pair list when terminal state is built, so Ticket 010 can consume a completed snapshot without touching the still-running iterator.
- `EmbeddingTrainingRun._build_completion()` already rejects non-finite input weights, output weights, and final loss before returning a successful completion.
- `round_embedding_loss()` already demonstrates JavaScript `Math.round(value * scale) / scale` behavior for public loss values, but result construction has no general public-value rounding boundary for negative vector coordinates or two-decimal scores.
- `SavedEmbeddingMerge` and `SavedEmbeddingModel` already define the exact persistence-facing shape required by Ticket 008:
  - `type`;
  - `dimensions`;
  - `vocab`;
  - `merges`;
  - `embeddings`.
- No public Embedding Result type or result-construction operation currently exists.
- No current Python operation:
  - converts input-weight rows to public six-decimal vectors;
  - resolves Query Word positions;
  - creates warnings;
  - calculates cosine similarity;
  - ranks Nearest Neighbors;
  - creates Similarity Pairs;
  - evaluates the seven Vector Analogies;
  - converts a completion into a complete Saved Embedding Model.
- `tests/test_word2vec.py` verifies exact immutable preprocessing and complete ordered fixtures.
- `tests/test_word2vec_training.py` verifies deterministic training primitives, exact public epoch updates, tightly compared unrounded terminal matrices, finite-state enforcement, and sequential/concurrent isolation.
- `tests/test_train_embed_persistence.py` verifies the independent persistence boundary using fixed complete Saved Embedding Model fixtures; it deliberately does not construct models from training state.
- No dedicated result-construction test module or independent result fixture exists.
- `src/how_llms_work/routes/train_embed.py` already owns only atomic Saved Embedding Model persistence. It does not yet orchestrate training or result construction, and Ticket 010 must not add that orchestration.
- `src/how_llms_work/ml/math_utils.py` remains empty. The selected ticket and specification assign reusable Word2Vec result mathematics to `ml/word2vec.py`, so a general math-module refactor is unnecessary.
- The TypeScript reference constructs:
  - public vectors from `wIn` only;
  - ordered warnings and recognized index positions;
  - neighbors from six-decimal vectors with two-decimal pre-ranking scores;
  - pairwise similarities by recognized position;
  - the seven established analogies using raw-source and rounded-candidate mixed precision;
  - the complete Saved Embedding Model before yielding the final result.
- The ticket's newer wording requires original submitted text to remain available for query-facing display and warnings. Where this is more specific than the older TypeScript implementation's canonical Vocabulary labels, the selected ticket is the immediate acceptance authority.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - Ticket 009's blocker is complete: the current module produces deterministic, finite, copied terminal input and output matrices.
  - Immutable ordered preprocessing already exposes the complete Vocabulary, token indices, and Merge Table needed by result and model construction.
  - The exact `SavedEmbeddingModel` top-level type exists and the Ticket 008 persistence boundary already accepts and safely serializes it.
  - Existing preprocessing and training tests already prove request-owned numerical state and shared-preprocessing isolation.
- **Behavior present but evidence incomplete:**
  - TypeScript-compatible rounding logic exists only as the loss-specific `round_embedding_loss()` operation; it is not yet established as the reusable boundary for signed vector coordinates and two-decimal public scores.
  - `CompletedEmbeddingTraining` contains all raw numerical data required by Ticket 010, but no public conversion API currently consumes it.
  - `SavedEmbeddingModel` is typed and persistence-tested with fixed fixtures, but no production conversion from a completed run exists.
- **Partially implemented:**
  - The public Saved Embedding Model contract is present, but complete Vocabulary embedding construction, Merge conversion, six-decimal vectors, and finite-value checks are missing.
  - Finite-state enforcement exists at training completion, but result construction must still prevent non-finite cosine or derived values from entering either public object.
- **Not implemented:**
  - Public Embedding Result contracts and exact top-level field set.
  - Public vector extraction and six-decimal signed rounding.
  - Query Word positional lookup, duplicate preservation, original-text handling, BPE warning analysis, and exact warning strings.
  - Selected embeddings, Nearest Neighbor groups, Similarity Pairs, and all-unrecognized behavior.
  - Stable two-decimal neighbor ranking and Vocabulary-order ties.
  - The seven mixed-precision Vector Analogies, exclusions, tie behavior, and ordering.
  - Complete Saved Embedding Model conversion from `CompletedEmbeddingTraining`.
  - Dedicated independent fixtures and tests for exact rounded outputs, field sets, tie outcomes, mixed precision, finite failures, and repeated-call isolation.
- **Evidence limitation:**
  - No standalone ADR 0001 file was supplied in this handoff, but its exact-public/tolerance-hidden compatibility decision is repeated in the selected ticket and specification.
  - Exact Ticket 010 fixture values are not present in the latest Python export. They must be captured independently from the TypeScript behavior or hand-calculated synthetic states during implementation; they must not be generated by calling the new Python production operations under test.
  - The planning session did not execute the baseline or calculate production result fixtures.

## Files to inspect before editing

1. `src/how_llms_work/ml/word2vec.py` — `SavedEmbeddingMerge`, `SavedEmbeddingModel`, `Word2VecPreprocessing`, `get_word2vec_preprocessing()`, `CompletedEmbeddingTraining`, `round_embedding_loss()`, and `create_embedding_training_run()`.
2. `tests/test_word2vec.py` — immutable preprocessing fixtures, exact-order assertions, fixture-loading style, and public-module test conventions.
3. `tests/test_word2vec_training.py` — small independent preprocessing construction, `CompletedEmbeddingTraining` collection, finite-state tests, tight `float64` comparisons, and isolation patterns.
4. `tests/fixtures/word2vec_preprocessing_reference.json` — fixed ordered Vocabulary and Merge Table evidence that must remain unchanged.
5. `tests/fixtures/word2vec_training_reference.json` — independent deterministic training state used as input evidence, not as a substitute for new result fixtures.
6. `src/how_llms_work/routes/train_embed.py` — existing `save_embedding_model()` boundary and exact `SavedEmbeddingModel` consumer; inspect for compatibility but do not add orchestration.
7. `tests/test_train_embed_persistence.py` — exact Saved Embedding Model field order, complete embedding mapping, no-NaN serialization, and filesystem-isolation prior art.
8. `src/how_llms_work/ml/bpe.py` — `Merge` and `apply_merges()` used for warning tokenization; no BPE behavior change is expected.
9. `pyproject.toml` — Python 3.12, pytest, Ruff, and strict-mypy configuration; no dependency addition should be necessary.
10. `llm_works_file_structure(8).md` — TypeScript `src/routes/train-embed/train.ts` and `src/server/lib/math.ts` for exact output formulas, operation order, and field shapes only.

## Step 1 — Establish the independent public result-construction fixture and test seam

**Files and symbols:**
- `tests/test_word2vec_results.py` — new public-boundary tests for Embedding Result and Saved Embedding Model construction.
- `tests/fixtures/word2vec_results_reference.json` — new independent exact public fixtures.
- `src/how_llms_work/ml/word2vec.py` — planned stable public result/model conversion operations; exact private helper decomposition remains an implementation choice.

**Purpose:**
Define Ticket 010's observable contract before production implementation, using small deterministic inputs that isolate result behavior from expensive Skip-gram training, HTTP, SSE, and filesystem persistence.

**Actions:**
- Add a dedicated result-construction test module that imports only intentionally public Word2Vec types and conversion operations.
- Build a small immutable `Word2VecPreprocessing` fixture directly from explicit corpus-derived values rather than invoking production preprocessing.
- Build one or more explicit `CompletedEmbeddingTraining` fixtures with:
  - ordered Vocabulary Tokens;
  - dimensions and matrix shapes that are easy to inspect;
  - raw input rows containing positive, negative, tie-boundary, and more-than-six-decimal coordinates;
  - output rows filled with clearly different finite sentinel values so accidental output-weight exposure is immediately visible.
- Include all Vocabulary Tokens required by the seven predefined analogies in the analogy-specific fixture, plus enough candidate tokens to prove exclusion and tie behavior.
- Store exact expected public objects in `word2vec_results_reference.json`, including:
  - signed six-decimal vectors;
  - exact selected-embedding order;
  - exact warning text and order;
  - exact Nearest Neighbor candidates and scores;
  - every expected Similarity Pair;
  - all seven ordered analogy records;
  - the exact complete Saved Embedding Model.
- Calculate fixture expectations independently from the Python production code under test. Use the supplied TypeScript formulas, a reviewed standalone calculation, or hand-calculated synthetic vectors.
- Add exact top-level and nested field-set assertions so extra internal fields fail immediately.
- Establish tests that are expected to fail initially because the public result/model conversion operations and Embedding Result contracts do not yet exist.

**Guardrails:**
- Do not run full default Word2Vec training to produce result fixtures.
- Do not derive expected values by importing and calling the new production conversion functions.
- Do not test private helper names, local sorting syntax, dataclass identity, or internal cache/container identity.
- Do not use route serialization, `TestClient`, the real `.data` directory, or persistence helpers in this test module.
- Do not edit TypeScript or execute it as part of the normal Python test suite.

**Expected result:**
- A focused red test suite defines every Ticket 010 public field, rounded value, order, warning, tie outcome, mixed-precision analogy result, and complete Saved Embedding Model.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec_results.py -q
```

- Expected before implementation: collection or assertions fail only because Ticket 010 public behavior is missing.
- Expected after all production steps: every exact fixture assertion passes.

## Step 2 — Add typed public result contracts and six-decimal Saved Embedding Model conversion

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — `SavedEmbeddingMerge`, `SavedEmbeddingModel`, `CompletedEmbeddingTraining`, `round_embedding_loss()`, and new public result/model conversion boundary.
- `tests/test_word2vec_results.py` — public-vector, field-set, input-only, complete-model, and finite-value tests.
- `tests/fixtures/word2vec_results_reference.json` — exact public vectors and complete Saved Embedding Model.

**Purpose:**
Create one reusable, typed source of public vectors and convert a completed run into the exact complete model required by Ticket 008, without exposing or using output weights.

**Actions:**
- Define typed plain-Python contracts for:
  - Word Embedding;
  - Nearest Neighbor candidate and group;
  - Similarity Pair;
  - Vector Analogy;
  - Embedding Result.
- Keep `SavedEmbeddingModel` as the persistence-facing contract already consumed by `save_embedding_model()`.
- Generalize or wrap the existing TypeScript-compatible rounding behavior so it safely handles signed coordinates and two-decimal scores while preserving the existing six-decimal loss behavior.
- Validate the minimum structural assumptions needed for safe conversion:
  - dimensions are positive;
  - ordered Vocabulary length matches the input-weight row count;
  - each public input row has exactly `dimensions` coordinates;
  - completion Vocabulary and supplied preprocessing Vocabulary align in exact order when both are required by the conversion boundary.
- Build a fresh six-decimal public vector for every ordered Vocabulary index from `input_weights` only.
- Never read output-weight coordinates for public vector construction.
- Reject a non-finite public coordinate or derived public value before returning a successful object.
- Convert every preprocessing `Merge` to exactly:
  - `pair`: a two-item list preserving pair order;
  - `merged`: the merged Token;
  - no `frequency` field.
- Build the Saved Embedding Model in exact insertion order:
  1. `type`;
  2. `dimensions`;
  3. `vocab`;
  4. `merges`;
  5. `embeddings`.
- Preserve complete ordered Vocabulary and Merge Table values.
- Build the complete embeddings mapping in Vocabulary order with a fresh six-decimal vector for every Token.
- Ensure the conversion returns ordinary JSON-compatible Python values and no NumPy scalar, array, matrix, raw row, view, output weight, training metadata, or persistence metadata.

**Guardrails:**
- Do not change Skip-gram training, PRNG consumption, weight updates, preprocessing, or terminal-state construction unless a focused result test proves a genuine integration defect.
- Do not include `window_size`, `epochs`, `negative_samples`, `final_loss`, Training Pairs, paths, timestamps, or request identifiers in the Saved Embedding Model.
- Do not call `save_embedding_model()` from the Word2Vec module.
- Do not move general vector behavior into `ml/math_utils.py`; Ticket 010's approved ownership is the reusable Word2Vec module.
- Do not use Python's built-in `round()` when its tie semantics would differ from JavaScript `Math.round()`.

**Expected result:**
- The public Word2Vec boundary can convert one completed numerical state into an exact finite complete Saved Embedding Model.
- Every saved vector comes from `input_weights`, is six-decimal and JSON-compatible, and covers the complete ordered Vocabulary.
- Existing Ticket 008 persistence fixtures remain valid without changing the persistence boundary.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec_results.py -q -k "vector or rounding or input_weight or saved_model or merge or field or finite"
poetry run pytest tests/test_train_embed_persistence.py -q
```

Expected result:

- Public-vector and Saved Embedding Model tests pass exactly.
- Ticket 008 persistence tests still pass unchanged.

## Step 3 — Resolve Query Word positions and construct exact selected embeddings and warnings

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — new public Embedding Result construction operation using `Word2VecPreprocessing.token_indices`, `apply_merges()`, and public vectors.
- `tests/test_word2vec_results.py` — uppercase, duplicates, leading/trailing space, whitespace-only, unknown, multi-token, partial-recognition, and all-unrecognized cases.
- `tests/fixtures/word2vec_results_reference.json` — exact positional selections and warning strings.

**Purpose:**
Preserve Query Word input meaning exactly while separating request validation from Vocabulary recognition and BPE warning analysis.

**Actions:**
- Accept the submitted Query Word sequence as read-only positional input.
- For each position, retain the original submitted string without trimming, splitting, filtering, deduplicating, or mutating it.
- Create a lowercased copy of the complete submitted entry only for:
  - exact `token_indices` lookup;
  - `apply_merges()` warning analysis.
- Treat an entry as recognized only when the complete lowercased entry is exactly one known Vocabulary key.
- Preserve recognized positions in submission order, including repeated positions that resolve to the same Vocabulary index.
- For each unrecognized or multi-token position:
  - apply the complete ordered Merge Table to the lowercased but otherwise unmodified entry;
  - append exactly one warning;
  - preserve the original submitted entry inside the quoted warning;
  - format the BPE split as comma-and-space-separated Tokens inside brackets.
- Use this exact warning format:

  ```text
  "<submitted word>" is not a single BPE token — it splits into [<comma-separated tokens>]
  ```

- Construct one selected Word Embedding for each recognized position and no selected embedding for unrecognized positions.
- Follow Ticket 010's latest original-display requirement for query-facing labels while retaining canonical Vocabulary Tokens for candidate Vocabulary identities.
- Ensure duplicate recognized positions produce repeated selected embeddings rather than shared or deduplicated entries.
- Preserve warning order independently of recognized result order.
- Permit all Query Words to be unrecognized; return empty selected embeddings while leaving analogy construction available to the later step.

**Guardrails:**
- Do not alter preprocessing, the completed matrices, or the caller's Query Word list.
- Do not call `.strip()`, split on whitespace, normalize internal whitespace, filter falsy strings, or convert the caller's sequence in place.
- Do not reject structurally valid non-empty values at this boundary; request validation belongs to Ticket 011.
- Do not add warnings for recognized uppercase entries.
- Do not fail the completed run merely because one or all Query Words are unrecognized.

**Expected result:**
- Selected embeddings and warnings exactly reflect submitted positions and original text.
- Uppercase recognized Query Words resolve case-insensitively.
- Leading-space, trailing-space, whitespace-only, unknown, and multi-token entries each preserve their confirmed unrecognized behavior.
- Duplicate recognized positions remain repeated output positions.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec_results.py -q -k "query or warning or uppercase or duplicate or whitespace or unknown or multi_token or all_unrecognized"
```

Expected result:

- Every recognition, positional-order, duplicate, omission, and exact-warning assertion passes.

## Step 4 — Construct stable Nearest Neighbor groups and positional Similarity Pairs

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — public cosine behavior, Nearest Neighbor ranking, and Similarity Pair construction.
- `tests/test_word2vec_results.py` — six-decimal source precision, two-decimal rounding, stable ties, self-exclusion, five-result limit, duplicate positions, and pair-order tests.
- `tests/fixtures/word2vec_results_reference.json` — exact neighbor groups and Similarity Pairs.

**Purpose:**
Reproduce the frontend-visible geometry and ordering rules from public six-decimal Word Embeddings.

**Actions:**
- Calculate cosine similarity through explicit ordered coordinate accumulation compatible with the TypeScript reference.
- Use only six-decimal public vectors for both Nearest Neighbor and Similarity Pair calculations.
- Detect zero magnitude or any other condition that would produce a non-finite public score and fail result construction rather than emitting NaN or infinity.
- For each recognized Query Word position:
  - traverse candidate Vocabulary Tokens in exact Vocabulary order;
  - exclude only the recognized Query Word's own Vocabulary index;
  - calculate cosine similarity from public vectors;
  - round each score to two decimals before ranking;
  - retain candidate Vocabulary order for equal rounded scores;
  - return at most the first five ranked candidates.
- Implement stable descending ranking without an alphabetical, raw-score, or index-based secondary tie-breaker that changes the required tie outcome.
- Produce one Nearest Neighbor group per recognized Query Word position, including repeated groups for duplicate recognized positions.
- Construct Similarity Pairs by positional nested traversal:
  - first recognized position from left to right;
  - every later recognized position from left to right;
  - no self-position pair;
  - duplicate positions that resolve to the same Vocabulary Token remain valid and normally produce a same-vector score.
- Round each Similarity Pair score to two decimals with the same TypeScript-compatible public rounding operation.
- Preserve query-facing labels according to the ticket's original-display requirement and preserve candidate labels as ordered Vocabulary Tokens.
- Do not sort the Similarity Pair list during backend construction; retain recognized positional pair order.

**Guardrails:**
- Do not calculate neighbors or similarities from raw input rows.
- Do not use output weights.
- Do not rank on an unrounded score and only round after slicing.
- Do not deduplicate repeated Query Word indices before building groups or pairs.
- Do not mutate the public-vector table or any vector stored in a returned result.
- Do not import a third-party vector or machine-learning library.

**Expected result:**
- Every recognized Query Word position has one stable top-five-or-smaller Nearest Neighbor group.
- Equal displayed neighbor scores retain ordered Vocabulary position.
- Similarity Pairs cover every recognized positional combination exactly once and in deterministic order.
- All scores are finite and two-decimal TypeScript-compatible values.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec_results.py -q -k "neighbor or similarity or cosine or tie or exclude or limit or positional"
```

Expected result:

- Exact neighbor rankings, tie outcomes, self-exclusion, top-five limits, pair order, duplicate-position pairs, and finite-score tests pass.

## Step 5 — Evaluate the seven mixed-precision Vector Analogies and assemble the exact Embedding Result

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — predefined analogy definitions, mixed-precision candidate search, and final `EmbeddingResult`.
- `tests/test_word2vec_results.py` — analogy order, raw-source precision, rounded-candidate precision, exclusions, first-candidate ties, score rounding, all-unrecognized behavior, and exact result field set.
- `tests/fixtures/word2vec_results_reference.json` — exact seven analogy records and complete Embedding Result.

**Purpose:**
Complete the public frontend payload while preserving the reference's deliberately mixed numerical precision and exact field boundary.

**Actions:**
- Define the seven analogy triples in this exact order:

  ```text
  king - man + woman
  queen - woman + man
  prince - boy + girl
  kitten - cat + dog
  puppy - dog + cat
  he - man + woman
  his - man + woman
  ```

- Resolve each source Token through the ordered preprocessing index mapping.
- Build each analogy query coordinate from raw input-weight rows in exact `a - b + c` order.
- Compare the raw query vector against each candidate's six-decimal public vector.
- Traverse candidates in ordered Vocabulary order.
- Exclude all three source indices even when an implementation could otherwise choose a source Token as the closest result.
- Update the selected candidate only on a strictly greater unrounded cosine score so equal scores retain the first eligible Vocabulary candidate.
- Reject non-finite query coordinates, magnitudes, candidate scores, or selected scores before a successful result is returned.
- Round only the final selected analogy score to two decimals using TypeScript-compatible semantics.
- Create analogy `query` text in exact lowercased source-token form shown above.
- Evaluate analogies independently of Query Word recognition so an all-unrecognized Query Word request still returns all available predefined analogies.
- Assemble the Embedding Result with exactly these insertion-ordered top-level fields:
  1. `embeddings`;
  2. `neighbors`;
  3. `similarities`;
  4. `analogies`;
  5. `warnings`.
- Exclude the complete Saved Embedding Model, Merge Table, raw matrices, output weights, dimensions, training settings, final loss, persistence data, and internal state from the Embedding Result.
- Return fresh JSON-compatible lists and dictionaries so mutation of one returned object cannot change a later conversion.

**Guardrails:**
- Do not replace the mixed-precision analogy behavior with all-raw or all-rounded calculations.
- Do not sort analogy candidates or analogy records.
- Do not add optional, request-defined, or dynamically generated analogies.
- Do not skip established analogies merely because the submitted Query Words are unrecognized.
- Do not expose any Saved Embedding Model data through the Embedding Result.

**Expected result:**
- The public Word2Vec boundary returns the exact five-field Embedding Result.
- All seven analogies appear in established order with correct mixed-precision candidates, exclusions, tie behavior, and two-decimal scores.
- An all-unrecognized request returns empty selected embeddings, neighbors, and similarities, ordered warnings, and the normal analogy list.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec_results.py -q -k "analogy or mixed_precision or exclusion or first_candidate or all_unrecognized or result_field"
```

Expected result:

- Every analogy and exact Embedding Result assertion passes.

## Step 6 — Prove non-finite failure boundaries, mutation isolation, and Ticket 007–009 regressions

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — complete Ticket 010 public conversion boundary.
- `tests/test_word2vec_results.py` — non-finite, repeated-call, mutation, and no-output-weight-exposure tests.
- `tests/test_word2vec.py` — immutable preprocessing regressions.
- `tests/test_word2vec_training.py` — deterministic training and terminal-state regressions.
- `tests/test_train_embed_persistence.py` — complete-model persistence compatibility regressions.

**Purpose:**
Prove that exact public behavior is safe to call repeatedly and composes with the completed preprocessing, training, and persistence tickets without expanding into route work.

**Actions:**
- Add public-boundary failure tests for:
  - NaN input coordinates;
  - positive and negative infinity input coordinates;
  - zero-magnitude public vectors that would make cosine non-finite;
  - malformed dimensions, matrix row count, row width, or Vocabulary/preprocessing alignment when such invalid inputs can reach the public conversion API.
- Assert failed construction returns no partial successful Embedding Result or Saved Embedding Model.
- Use distinctive finite output-weight sentinels and assert they appear nowhere in either public object.
- Snapshot the completed input matrix, output matrix, ordered Vocabulary, preprocessing values, and Query Word sequence before construction; compare them after construction.
- Call each public conversion operation repeatedly and assert equivalent exact values.
- Mutate a first returned Embedding Result and Saved Embedding Model, then call construction again and prove the second outputs and source state are unchanged.
- Exercise sequential and controlled concurrent result/model construction against the same read-only completion and preprocessing values; assert equivalent outputs and no mutation.
- Run the preprocessing, deterministic training, result construction, and persistence test modules together.
- Inspect the final diff and confirm no route, schema, SSE, frontend, persistence implementation, dependency, cache, generated `.data` file, or Transformer change was introduced.
- Report actual command outcomes honestly.

**Guardrails:**
- Test observable source and cross-call isolation, not internal object identity or a required private caching strategy.
- Do not weaken Ticket 009's finite terminal-state checks merely because Ticket 010 also validates its public boundary.
- Do not write to the real `backend/.data/`.
- Do not fix unrelated failures by expanding the ticket.
- Do not claim full success unless focused tests, Ruff, strict mypy, and complete pytest all pass in the implementation session.

**Expected result:**
- Ticket 010 public conversions are exact, finite, input-only, non-mutating, repeatable, and safe for concurrent read-only use.
- Tickets 007, 008, and 009 remain green.
- The final implementation remains isolated to reusable Word2Vec result/model construction and its independent tests and fixtures.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec_results.py -q -k "finite or nan or infinity or zero or mutation or isolation or concurrent or output_weight"
poetry run pytest tests/test_word2vec.py tests/test_word2vec_training.py tests/test_word2vec_results.py tests/test_train_embed_persistence.py
```

Expected result:

- All focused failure, isolation, and affected-area regression tests pass.

## Focused verification plan

Run from the backend project root:

```powershell
poetry run pytest tests/test_word2vec_results.py
poetry run pytest tests/test_word2vec.py tests/test_word2vec_training.py tests/test_word2vec_results.py tests/test_train_embed_persistence.py
```

Expected result:

- Exact six-decimal vectors, Query Word position handling, warning text, neighbor rankings, Similarity Pairs, mixed-precision analogies, field sets, and complete Saved Embedding Model fixtures pass.
- Non-finite values cannot produce a successful public object.
- Repeated and concurrent conversion does not mutate the completion, preprocessing, Query Word sequence, or later outputs.
- Existing preprocessing, deterministic training, and atomic persistence tests remain green.

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

- [ ] Ticket 009 remains satisfied and one finite `CompletedEmbeddingTraining` can be converted without retraining.
- [ ] Every public Word Embedding uses only its input-weight row.
- [ ] Every public coordinate uses TypeScript-compatible six-decimal rounding, including negative values and half-boundary cases.
- [ ] Output weights, raw input matrices, NumPy values, and unrounded rows are absent from both public objects.
- [ ] Query Words are not trimmed, split, filtered, deduplicated, or mutated.
- [ ] Exact lookup lowercases the complete Query Word only for recognition.
- [ ] Original submitted text is retained wherever Ticket 010 requires query-facing display and in every warning.
- [ ] Uppercase recognized Query Words resolve without a warning.
- [ ] Leading-space, trailing-space, whitespace-only, unknown, and multi-token entries preserve their expected unrecognized behavior.
- [ ] Every unrecognized position produces exactly one warning with the exact punctuation and BPE split.
- [ ] Duplicate recognized positions produce repeated selected embeddings and Nearest Neighbor groups.
- [ ] Unrecognized positions are absent from selected embeddings, neighbors, and Similarity Pairs.
- [ ] Similarity Pairs cover every pair of recognized positions, including duplicate same-token positions.
- [ ] An all-unrecognized request produces empty selected collections and ordered warnings while still evaluating analogies.
- [ ] Neighbor scores are calculated from six-decimal vectors, rounded before ranking, sorted descending, and stable by Vocabulary order for ties.
- [ ] Each neighbor group excludes only its own Vocabulary index and contains at most five candidates.
- [ ] Similarity scores use six-decimal vectors and TypeScript-compatible two-decimal rounding.
- [ ] The seven analogies appear in the established order.
- [ ] Analogy query vectors use raw input rows; candidate vectors use six-decimal public rows.
- [ ] Every analogy excludes all three source Tokens and retains the first Vocabulary candidate on equal unrounded scores.
- [ ] The Embedding Result contains exactly `embeddings`, `neighbors`, `similarities`, `analogies`, and `warnings`.
- [ ] The Saved Embedding Model contains exactly `type`, `dimensions`, `vocab`, `merges`, and `embeddings`.
- [ ] The Saved Embedding Model preserves the complete ordered Vocabulary and Merge Table and includes every Vocabulary Token.
- [ ] Neither public object contains NaN or positive/negative infinity.
- [ ] Mutating one returned result or model cannot change the completion, preprocessing, Query Words, or a later conversion.
- [ ] No route, schema, SSE, persistence implementation, frontend, Transformer, dependency, or `.data` change is present.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/word2vec.py
tests/test_word2vec_results.py
tests/fixtures/word2vec_results_reference.json
```

Conditionally changed only if focused integration evidence proves an existing public completion or fixture helper must be adjusted without changing Ticket 009 behavior:

```text
tests/test_word2vec_training.py
tests/fixtures/word2vec_training_reference.json
```

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/
tests/test_simple_chat.py
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
poetry.toml
README.md
frontend/
SPEC.md
CONTEXT.md
010-construct-exact-embedding-results-and-saved-embedding-models.md
```

## Risk notes and safeguards

1. **Risk:** Python's built-in `round()` applies different half-tie behavior from JavaScript `Math.round()`, especially for negative values.
   - **Safeguard:** Reuse one explicit TypeScript-compatible rounding operation and protect positive, negative, and half-boundary cases with exact fixtures.

2. **Risk:** A public vector is accidentally built from output weights or a combination of input and output rows.
   - **Safeguard:** Use deliberately different output-weight sentinel values and assert neither object contains them.

3. **Risk:** Query Words are normalized with `.strip()`, split, filtered, or deduplicated before lookup.
   - **Safeguard:** Include uppercase, duplicate, leading-space, trailing-space, whitespace-only, unknown, and multi-token positions in one exact ordered fixture.

4. **Risk:** An index-only recognized list loses the original submitted display text required by the ticket.
   - **Safeguard:** Retain a positional recognition record containing both original text and Vocabulary index, and assert uppercase/duplicate query-facing labels explicitly.

5. **Risk:** Warning analysis trims or otherwise alters the submitted entry before BPE tokenization.
   - **Safeguard:** Apply merges to the lowercased but otherwise unchanged entry and assert exact split output for leading, trailing, internal, and whitespace-only cases.

6. **Risk:** Duplicate recognized Query Words collapse because results are keyed by Token or Vocabulary index.
   - **Safeguard:** Build selected collections from an ordered positional sequence, never a set or dictionary keyed by the recognized Token.

7. **Risk:** Neighbor candidates are ranked by raw cosine and rounded only after selection.
   - **Safeguard:** Use a fixture where raw scores differ but round to the same two-decimal value; assert Vocabulary-order tie preservation.

8. **Risk:** A secondary alphabetical or numeric key silently changes equal-score neighbor ordering.
   - **Safeguard:** Preserve candidate construction in Vocabulary order and sort only by descending rounded score.

9. **Risk:** Similarity Pairs are deduplicated by Token value and omit repeated same-token positions.
   - **Safeguard:** Generate pairs by recognized positional indices and assert the complete pair count and order.

10. **Risk:** Analogy code simplifies the mixed-precision contract by using either raw vectors everywhere or rounded vectors everywhere.
    - **Safeguard:** Use a crafted fixture whose selected candidate changes under the wrong precision mode and assert the required raw-query/rounded-candidate result.

11. **Risk:** Equal analogy scores select a later candidate due to sorting or `>=`.
    - **Safeguard:** Traverse Vocabulary order and replace the current best only on strict `>`; protect with a first-candidate tie fixture.

12. **Risk:** A zero vector or malformed state yields NaN or infinity that survives into a public object.
    - **Safeguard:** Check coordinate, magnitude, cosine, and final public-value finiteness and fail construction before returning a successful object.

13. **Risk:** The Saved Embedding Model contains only selected Query Words or includes Merge frequencies and training metadata.
    - **Safeguard:** Compare the entire model against an exact fixture and assert one vector per ordered Vocabulary Token plus exact top-level and Merge field sets.

14. **Risk:** Public lists or vectors alias source arrays or prior returned objects.
    - **Safeguard:** Return fresh plain-Python containers and prove mutation of one result cannot affect source state or a later conversion.

15. **Risk:** Result tests become slow or circular by training the full production model.
    - **Safeguard:** Use small explicit synthetic completions and independently calculated expected public fixtures.

16. **Risk:** Ticket 010 expands into HTTP/SSE orchestration or changes the completed persistence boundary.
    - **Safeguard:** Keep likely production edits in `ml/word2vec.py`, leave `routes/train_embed.py` unchanged, and inspect the final diff against the expected file list.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

A suitable outcome-oriented subject would describe exact reference-compatible Embedding Result and complete Saved Embedding Model construction without claiming HTTP streaming or persistence integration.

Commit body should mention:

- input-weight-only six-decimal public vectors and TypeScript-compatible score rounding;
- positional Query Word lookup, original-text warnings/display, duplicates, and all-unrecognized behavior;
- stable Nearest Neighbor rankings and positional Similarity Pairs;
- seven mixed-precision Vector Analogies with exclusions and first-candidate ties;
- exact five-field Embedding Result and complete five-field Saved Embedding Model conversion;
- non-finite rejection, no internal-weight exposure, and repeated/concurrent construction isolation;
- the exact focused and full verification commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 010, Ticket 009, `SPEC.md`, `CONTEXT.md`, ADR 0001 as recorded in the approved handoff, `py_llm_pipeline_explorer_file_structure(22).md`, and `llm_works_file_structure(8).md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 010, verify the complete change, report actual command results honestly, and create the implementation commit.
