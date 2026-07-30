---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "014"
source_work_item: 014-produce-immutable-transformer-preprocessing-snapshot.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(37).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 014: Produce an immutable Transformer Preprocessing Snapshot

## Initial checklist

- Confirm Ticket 014 is the only selected work item and has no blocker.
- Treat `py_llm_pipeline_explorer_file_structure(37).md` as the current Python Backend source of truth.
- Use `llm_works_file_structure.md` only to capture the exact TypeScript Transformer corpus and preprocessing behavior.
- Preserve the user-reported passing pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Implement only the fixed-corpus Transformer Preprocessing Snapshot and its public preprocessing tests.
- Reuse the established BPE boundary; do not add a second tokenizer, request-owned preprocessing, training mathematics, workers, persistence, or HTTP behavior.
- Finish with focused Transformer preprocessing tests, affected BPE and Word2Vec regressions, the full suite, Ruff, strict mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest direction to convert the selected TypeScript behavior to Python and treat the latest complete Python Backend export as current-code truth.
2. `014-produce-immutable-transformer-preprocessing-snapshot.md` for the required behavior, acceptance criteria, approved test seam, constraints, and out-of-scope boundaries.
3. `py_llm_pipeline_explorer_file_structure(37).md` for current Python code, tests, dependencies, paths, and repository conventions.
4. `SPEC.md`, `CONTEXT.md`, and ADR 0002 for the durable Phase 5 preprocessing, immutability, publication, sequence, and Logical Training Shard decisions.
5. `llm_works_file_structure.md`, especially TypeScript `src/routes/train-embed/corpus.ts`, `src/routes/train-transformer/train.ts`, and `src/server/lib/bpe.ts`, as exact behavior evidence only.
6. Older exports, snippets, plans, and specification statements about current file contents are non-authoritative when they conflict with export `(37)`.

## Work-item summary

Ticket 014 creates the fixed data foundation used by every later Transformer Training Run. The implementation must process the exact ordered Transformer Training Corpus into one immutable application-wide `TransformerPreprocessingSnapshot`.

The snapshot must preserve the TypeScript Reference Implementation's exact Transformer-specific BPE training text, Pre-token boundaries, ordered Merge Table, per-story tokenization, ordered Vocabulary, token indices, complete token-ID stream, fixed-length next-token Training Sequences, first-three generation seed IDs, and exactly four deterministic Logical Training Shard boundaries.

Construction must be lazy and all-or-nothing. A module-local thread lock must protect double-checked initialization. The module must publish the snapshot only after all derived values have been constructed, validated, and converted to immutable or read-only containers. A failed attempt must leave the global value unpublished, release the initialization path, and permit a later caller to retry successfully.

This ticket does not initialize weights, perform Transformer forward or backward mathematics, run Adam, create processes or shared memory, expose `POST /train-transformer`, stream SSE, persist models, accept request text, or change the frontend.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Reported result:** The user reported that all pytest tests passed, Ruff passed, and strict mypy returned `Success: no issues found` before planning.
- **Planning-session limitation:** No pytest, Ruff, mypy, formatter, TypeScript extraction, browser, or runtime command was executed while creating this plan.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and report only command results it actually observes.

## Current code observations from the latest source

- `src/how_llms_work/ml/transformer.py` exists but is empty. There is no current Transformer preprocessing type, constructor, getter, corpus constant, Training Sequence type, or Logical Training Shard boundary type.
- `src/how_llms_work/ml/transformer_worker.py` and `src/how_llms_work/routes/train_transformer.py` remain empty and are outside Ticket 014.
- `src/how_llms_work/ml/bpe.py` already provides the reusable reference-compatible BPE seam:
  - frozen, slotted `Merge` values;
  - first-encounter ordered `count_words()`;
  - deterministic `train_bpe()` tie handling;
  - non-overlapping left-to-right pair replacement;
  - `apply_merges()` that preserves Pre-token boundaries.
- `src/how_llms_work/ml/word2vec.py` provides useful immutability prior art through frozen dataclasses, nested tuples, and `MappingProxyType`, but its preprocessing is built eagerly at import time. Ticket 014 must adapt the immutable-container pattern without copying the eager publication pattern.
- `src/how_llms_work/ml/math_utils.py` and `src/how_llms_work/ml/matrix.py` are now implemented by earlier Phase 5 tickets. They are not needed for fixed preprocessing and must remain unchanged.
- Existing BPE and Word2Vec tests already demonstrate exact fixture comparisons, ordered-container assertions, mutation-isolation checks, and sequential/concurrent reuse tests.
- The current test tree has no `tests/test_transformer.py` and no Transformer preprocessing reference fixture.
- `pyproject.toml` already provides Python 3.12+, NumPy, pytest, Ruff, Black, and strict mypy. No dependency or lockfile change is expected.
- The TypeScript Transformer trainer imports the fixed stories from the reference corpus module and composes the shared BPE operations. The implementation must inspect the exact reference join delimiter, normalization, merge limit, Vocabulary traversal, token-ID flattening, and sequence traversal instead of inferring them from Word2Vec.
- No current Python request model or route calls the empty Transformer module, so Ticket 014 can define a clean public preprocessing boundary without preserving an accidental Python compatibility surface.

## Acceptance criteria coverage

### Already satisfied and evidenced

- The reusable Python BPE implementation already prevents merges across Pre-token boundaries and preserves ordered merge application.
- The `Merge` record is already immutable.
- Python 3.12, pytest, Ruff, Black, strict mypy, and all required production dependencies are already configured.
- The destination Transformer module exists and is isolated from route, persistence, and process code.
- Existing tests provide established repository patterns for independent JSON fixtures, exact structural assertions, mapping immutability, mutation isolation, and thread-based concurrency checks.

### Behavior present but evidence incomplete

- The general BPE operations appear suitable for Transformer preprocessing, but there is no Transformer-specific fixture proving the exact training text, Merge Table, tokenized stories, or ordered Vocabulary.
- The Word2Vec preprocessing object demonstrates nested immutable containers, but it does not prove lazy double-checked publication or failure-and-retry behavior.
- The TypeScript Reference Implementation contains the exact corpus and construction traversal, but those values have not yet been captured as independent Python test evidence.

### Partially implemented

- `transformer.py` exists only as an empty destination.
- General reusable BPE behavior exists, but no Transformer-specific composition or public snapshot exists.

### Not implemented

- The exact fixed ordered Transformer Training Corpus in Python.
- Exact Transformer BPE training-text construction.
- The Transformer-specific ordered Merge Table.
- Exact tokenized-story preservation.
- Exact ordered Vocabulary and stable token-index mapping.
- The complete token-ID stream.
- Fixed-length `16` input and next-token target Training Sequences.
- Generation seed IDs equal to the first three complete-stream IDs.
- Exactly four contiguous Logical Training Shard boundaries calculated with `ceil(sequence_count / 4)`.
- Deterministic empty trailing shard representation for small sequence counts.
- An immutable public snapshot containing all required artifacts and no mutable request or training state.
- Lazy double-checked initialization guarded by one module-local thread lock.
- All-or-nothing publication and successful retry after controlled construction failure.
- Exact independent Transformer preprocessing fixtures.
- Public-boundary tests for exact artifacts, immutability, request independence, sequential reuse, concurrent initialization, failure retry, and shard edge cases.

### Evidence limitations

- The exact corpus, Transformer BPE training-text delimiter, merge limit, complete Merge Table, tokenized stories, Vocabulary order, complete token-ID stream, exact sequence count, and exact shard boundaries are not present in the current Python code. They must be captured from the supplied TypeScript Reference Implementation.
- The implementation must not assume the Transformer uses the same merge limit, Vocabulary ordering, story-boundary behavior, or sequence traversal as Word2Vec merely because both reuse BPE and the story corpus.
- The exact rule for whether adjacent Training Sequences may span story boundaries must be taken from the TypeScript Transformer trainer and protected by the fixed fixture.
- Independent expected values must not be generated by importing or calling `how_llms_work.ml.transformer`.
- No current-session command verified the supplied baseline or generated reference data.

## Files to inspect before editing

1. `src/how_llms_work/ml/transformer.py` — empty destination for snapshot records, fixed constants, deterministic construction, shard-boundary construction, and the public lazy getter.
2. `src/how_llms_work/ml/bpe.py` — existing `Merge`, `count_words()`, `train_bpe()`, and `apply_merges()` ownership boundary to reuse.
3. `src/how_llms_work/ml/word2vec.py` — immutable nested-container and read-only mapping prior art; do not copy its eager module-level construction.
4. `tests/test_bpe.py` — exact Pre-token, tie-order, merge-order, edge-case, and isolation evidence.
5. `tests/test_word2vec.py` — public preprocessing fixture, ordered artifact, immutability, and concurrent-reuse prior art.
6. `tests/fixtures/word2vec_preprocessing_reference.json` — fixture organization and provenance example only; do not derive Transformer expectations from it.
7. `tests/test_math_utils.py` — `ThreadPoolExecutor`, independent-state, and controlled deterministic fixture patterns.
8. `pyproject.toml` — current test, lint, formatting, typing, Python, and dependency configuration.
9. `014-produce-immutable-transformer-preprocessing-snapshot.md` — direct acceptance authority.
10. `SPEC.md`, `CONTEXT.md`, and `0002-stabilize-python-transformer-training-and-process-lifecycle.md` — canonical terminology and binding lazy-publication, immutability, sequence-length, and four-shard rules.
11. TypeScript `src/routes/train-embed/corpus.ts` in `llm_works_file_structure.md` — exact story values and order.
12. TypeScript `src/routes/train-transformer/train.ts` in `llm_works_file_structure.md` — exact BPE training text, merge-learning invocation, story tokenization, Vocabulary construction, token-ID flattening, Training Sequence traversal, seed extraction, and reference data partitioning.
13. TypeScript `src/server/lib/bpe.ts` in `llm_works_file_structure.md` — Transformer-used BPE semantics and stopping behavior.

## Step 1 — Capture independent Transformer preprocessing evidence

**Files and symbols:**

- `tests/fixtures/transformer_preprocessing_reference.json` — new fixed reference evidence.
- `tests/test_transformer.py` — fixture loader and initial public-contract tests.
- TypeScript reference corpus, Transformer trainer, and BPE modules — evidence source only.

**Purpose:**

Create non-circular evidence for the entire fixed preprocessing contract before implementing the Python composition. The fixture must be capable of detecting a wrong corpus sentence, delimiter, merge tie, token order, Vocabulary order, ID assignment, sequence boundary, seed, or shard split.

**Actions:**

- Capture the exact ordered story corpus from the TypeScript corpus module.
- Capture explicit provenance in the fixture, including the TypeScript source files and a statement that no Python production Transformer module was imported to create expected values.
- Record the exact BPE training text or enough exact metadata to prove its construction:
  - delimiter and story order;
  - total text length;
  - Pre-token count or other independently useful boundary evidence;
  - configured merge limit;
  - actual learned merge count.
- Store the complete ordered Merge Table, including each pair, merged token, and reference frequency when the reference exposes it.
- Store every tokenized story as an ordered token list.
- Store the complete ordered Vocabulary and exact token-index mapping.
- Store the complete token-ID stream, not only a hash or a few examples.
- Store every Training Sequence input and target when fixture size remains practical. At minimum, store the exact total count plus independently selected first, boundary, middle, and last sequences, but prefer the complete list because this ticket explicitly requires an exact complete token-ID stream and exact sequence construction.
- Store the exact generation seed IDs.
- Store all four shard records with `shard_id`, inclusive `start`, and exclusive `stop` boundaries.
- Add a separate small explicit expected table for sequence counts `0`, `1`, `2`, `3`, `4`, `5`, and at least one non-multiple of four. This small table must be calculated independently and must prove deterministic empty trailing shards.
- Keep expected data plain JSON. Do not serialize production dataclass objects into the fixture during tests.

**Guardrails:**

- Do not run the Python production constructor to generate its own expectations.
- Do not copy the Word2Vec fixture and edit values speculatively.
- Do not shorten the fixture in a way that allows a wrong internal sequence or Merge entry to pass.
- Do not add generated weights, request values, model configuration, worker counts, losses, samples, or persistence data.
- Do not commit a Node or TypeScript backend dependency to the Python Backend. Any one-time extraction command is evidence-generation tooling only.

**Expected result:**

- One fixed, reviewable fixture independently describes every externally relevant Transformer preprocessing artifact.
- Initial tests can fail against the empty Python module for the intended missing public boundary rather than because expected values are incomplete.

**Verification:**

```powershell
Get-Content tests\fixtures\transformer_preprocessing_reference.json |
    ConvertFrom-Json |
    Out-Null
```

Expected result:

- PowerShell parses the committed JSON fixture successfully.

## Step 2 — Define the immutable public preprocessing model

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — public constants and immutable records.
- `tests/test_transformer.py` — public shape, type, and immutability tests.

**Purpose:**

Define one narrow public representation that later Transformer training, generation, layout, and worker tickets can consume without reconstructing corpus data or receiving mutable state.

**Actions:**

- Define typed fixed constants for:
  - the exact ordered Transformer Training Corpus;
  - Training Sequence length `16`;
  - Logical Training Shard count `4`;
  - the exact Transformer BPE merge limit confirmed from the TypeScript reference.
- Define a frozen, slotted `TransformerTrainingSequence` record with:
  - `input_ids: tuple[int, ...]`;
  - `target_ids: tuple[int, ...]`.
- Define a frozen, slotted `LogicalTrainingShard` boundary record with:
  - stable `shard_id`;
  - inclusive `start`;
  - exclusive `stop`.
- Define a frozen, slotted `TransformerPreprocessingSnapshot` containing exactly:
  - `corpus`;
  - BPE training text or the approved stable BPE training artifact needed by later consumers;
  - ordered `merges`;
  - `tokenized_stories`;
  - ordered `vocabulary`;
  - read-only `token_indices`;
  - complete `token_ids`;
  - ordered `training_sequences`;
  - `generation_seed_ids`;
  - exactly four `shards`.
- Use immutable nested tuples for every ordered collection.
- Build token indices in a fresh private dictionary, wrap it in a read-only mapping only after construction, and retain no mutable external reference to that dictionary.
- Keep the snapshot free of request values, weights, gradients, random generators, processes, locks, NumPy arrays, paths, model outputs, and persistence state.
- Provide one deliberately public pure shard-boundary operation, such as `build_logical_training_shards(sequence_count)`, only if needed to give the approved small-count edge cases a stable public test seam. It must always return four ordered boundaries and must not accept worker count or shard count.
- Keep all construction-only helpers private. Tests must call the public snapshot getter and any deliberately public shard-boundary operation rather than private builders or cache variables.

**Guardrails:**

- Do not expose a configurable corpus, configurable merge count, configurable sequence length, or configurable shard count.
- Do not return lists, mutable dictionaries, mutable dataclasses, iterators, generators, or writable NumPy arrays.
- Do not introduce a generic preprocessing framework or a base cache class.
- Do not re-export unrelated Transformer mathematics or matrix symbols.
- Do not make lock identity or cache-variable names part of the public API.

**Expected result:**

- The public type system makes the snapshot contents and fixed four-shard model explicit.
- Mutation through one caller cannot change the data observed by another caller.

**Verification:**

```powershell
poetry run pytest tests/test_transformer.py -k "public or immutable or shard" -q
```

Expected result:

- Public structure, nested immutability, and small-count shard tests pass after implementation.

## Step 3 — Implement exact deterministic fixed-corpus construction

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — private complete-construction path.
- `src/how_llms_work/ml/bpe.py` — reused public BPE operations.
- `tests/test_transformer.py` — exact fixture comparison.

**Purpose:**

Translate the TypeScript Transformer preprocessing traversal into Python without duplicating the tokenizer or changing any observable fixed artifact.

**Actions:**

- Construct the exact BPE training text from the fixed stories using the delimiter, normalization, and order confirmed in the TypeScript Transformer trainer.
- Call the existing `count_words()` and `train_bpe()` operations with the confirmed Transformer merge limit.
- Apply the learned ordered Merge Table to each story through existing `apply_merges()`.
- Preserve story boundaries in `tokenized_stories` even if the reference later flattens tokens for sequence creation.
- Build the ordered Vocabulary using the exact TypeScript traversal and tie/first-encounter rule. Do not sort unless the reference explicitly sorts.
- Build one stable token-index mapping directly from that ordered Vocabulary.
- Flatten token IDs in the exact reference order and preserve every ID in the public complete stream.
- Build fixed-length `16` inputs and their one-token-shifted targets using the exact TypeScript start/stop traversal. Protect whether sequences may cross story boundaries through fixture evidence rather than assumption.
- Set `generation_seed_ids` to exactly `token_ids[:3]`.
- Calculate `shard_size` as `ceil(sequence_count / 4)` without consulting CPU count, worker count, operating system, request fields, or completion order.
- Produce exactly four contiguous boundaries. Clamp each boundary to the sequence count so trailing shards become deterministic empty ranges where necessary.
- Validate all invariants before returning:
  - non-empty fixed corpus;
  - Merge Table and tokenized stories match expected shapes;
  - every Vocabulary token has one unique ID;
  - every tokenized token resolves through the index;
  - all token IDs are in range;
  - every Training Sequence input and target has length `16`;
  - targets are the approved next-token shift;
  - the three seed IDs equal the first three stream IDs;
  - four shard IDs are `0`, `1`, `2`, `3`;
  - shards are contiguous, ordered, non-overlapping, cover every sequence exactly once, and contain no out-of-range boundary.
- Convert every nested value to its final immutable representation only after successful local construction and validation.

**Guardrails:**

- Do not mutate or extend the shared BPE module's state.
- Do not merge across Pre-token boundaries.
- Do not use a set or unordered traversal where order determines Vocabulary or IDs.
- Do not split data according to `os.cpu_count()`.
- Do not construct request-specific copies.
- Do not use generated samples or submitted text as training data.
- Do not silently continue after a missing token, duplicate Vocabulary entry, malformed sequence, or invalid boundary.

**Expected result:**

- One complete locally constructed snapshot matches the independent TypeScript-derived fixture exactly.
- Construction has no externally visible partial state.

**Verification:**

```powershell
poetry run pytest tests/test_transformer.py -k "reference or sequence or seed or vocabulary" -q
```

Expected result:

- Exact corpus, BPE, tokenization, Vocabulary, IDs, sequence, seed, and shard comparisons pass.

## Step 4 — Add lazy double-checked all-or-nothing publication

**Files and symbols:**

- `src/how_llms_work/ml/transformer.py` — module-local lock, unpublished optional value, and public getter.
- `tests/test_transformer.py` — sequential, concurrent, and failure-retry tests through the public getter.

**Purpose:**

Ensure expensive fixed preprocessing is performed once per application process, safely reused by concurrent callers, and never poisoned by a failed construction attempt.

**Actions:**

- Create exactly one module-local `threading.Lock`.
- Hold one module-local optional snapshot reference initialized to `None`.
- Implement `get_transformer_preprocessing()` with double-checked access:
  1. return the already published snapshot without acquiring the lock;
  2. otherwise acquire the lock;
  3. check again for a value;
  4. construct and validate a complete local snapshot;
  5. assign the global reference once, only after complete success;
  6. return the published snapshot.
- Let construction exceptions propagate to the caller without assigning any partial value.
- Rely on lock context management so exceptions release the initialization path.
- Keep all temporary lists and dictionaries local to the construction attempt.
- Do not catch an exception merely to publish a placeholder, empty snapshot, or failure sentinel.
- Through a freshly imported module boundary, patch a stable external construction dependency for one controlled first failure, restore it, and call the public getter again. Assert the retry returns the complete fixture-matching snapshot.
- Use `ThreadPoolExecutor` and a synchronization barrier to make multiple callers enter the uninitialized public getter concurrently.
- Assert through returned public values that every caller receives the one completely published snapshot and no caller observes a partial object.
- Do not inspect or assert private lock identity, lock acquisition counts, private cache-variable names, or private helper names.

**Guardrails:**

- Do not initialize the snapshot eagerly at module import.
- Do not use a mutable dictionary cache, `lru_cache`, disk cache, per-request cache, machine-wide lock, process-shared lock, or worker-owned cache.
- Do not publish before validation or immutable conversion.
- Do not swallow the controlled failure.
- Do not make tests order-dependent. Use an isolated fresh module boundary for initialization-lifecycle tests and restore patched dependencies automatically.

**Expected result:**

- Sequential calls reuse the published snapshot.
- Concurrent first calls publish one complete snapshot.
- A failed first construction publishes nothing, releases the lock path, and permits a successful retry.

**Verification:**

```powershell
poetry run pytest tests/test_transformer.py -k "lazy or concurrent or retry or publication" -q
```

Expected result:

- Public lazy-initialization, concurrency, and failure-retry tests pass without private-cache assertions.

## Step 5 — Prove request independence and complete immutability

**Files and symbols:**

- `tests/test_transformer.py` — mutation, caller-isolation, and request-independence coverage.
- `src/how_llms_work/ml/transformer.py` — only if a test exposes a public mutability leak.

**Purpose:**

Demonstrate that no caller, training configuration, or returned nested value can alter the fixed preprocessing used by a later or concurrent Transformer Training Run.

**Actions:**

- Attempt representative prohibited mutations:
  - replace or append a corpus story;
  - alter a Merge record;
  - replace a token inside one tokenized story;
  - assign a Vocabulary position;
  - assign or delete a token-index entry;
  - alter the token-ID stream;
  - alter a Training Sequence input or target;
  - alter seed IDs;
  - alter a shard boundary.
- Assert each mutation is rejected and that a later public getter call still exactly matches the reference fixture.
- Copy public data into caller-owned mutable containers, mutate those copies, and prove the shared snapshot is unaffected.
- Run sequential and concurrent read-only consumers that calculate summaries from the snapshot and prove stable identical results.
- Simulate different valid request configurations in caller-owned data and prove they are never passed into or reflected by the preprocessing getter. The snapshot must remain exact for differing `epochs`, `temperature`, `topP`, `numLayers`, and `maxTokens` values.
- Assert the snapshot exposes no request, weight, optimizer, process, shared-memory, persistence, generated-sample, or route-owned fields.
- Keep tests at the public snapshot and public shard-boundary seams.

**Guardrails:**

- Do not weaken immutability tests to simple equality checks.
- Do not depend on a particular frozen-container implementation where the ticket permits equivalent read-only containers.
- Do not add request parameters to the public preprocessing getter merely to test that they are ignored.
- Do not inspect object internals that later tickets should be free to refactor.

**Expected result:**

- Mutation by one caller cannot affect any later or concurrent caller.
- Preprocessing artifacts are structurally independent of all request fields.

**Verification:**

```powershell
poetry run pytest tests/test_transformer.py -k "immutable or isolation or request" -q
```

Expected result:

- Nested mutation, copied-container isolation, concurrent reads, and request-independence tests pass.

## Step 6 — Run regressions and enforce the ticket boundary

**Files and symbols:**

- `tests/test_transformer.py`
- `tests/test_bpe.py`
- `tests/test_word2vec.py`
- all existing backend tests and source files.

**Purpose:**

Confirm that adding Transformer preprocessing reuses rather than changes completed BPE and Word2Vec behavior and does not drift into later Phase 5 tickets.

**Actions:**

- Run the focused Transformer preprocessing suite.
- Run BPE and Word2Vec preprocessing regressions together with the new tests.
- Run the complete pytest suite.
- Run Ruff and strict mypy.
- Run `git diff --check`.
- Inspect `git status --short` and the final diff.
- Confirm no runtime `.data` artifact, cache file, TypeScript output, temporary fixture generator, or unrelated formatting change is included.
- Confirm `transformer_worker.py`, route modules, schemas, application registration, frontend files, persistence code, matrix code, and shared random utilities remain unchanged.
- Confirm no dependency or lockfile change occurred.

**Guardrails:**

- Do not fix unrelated failures by broad refactoring.
- Do not format untouched files.
- Do not begin Ticket 015 or later Transformer mathematics, layout, process, route, or persistence work.
- If the existing BPE seam fails an independently captured Transformer reference case, document the exact general BPE incompatibility and make only the smallest reusable correction; do not fork a Transformer-only tokenizer.

**Expected result:**

- Ticket 014 is fully covered at the stable public preprocessing seam.
- All completed Learning Demo behavior remains unchanged.
- The final diff is limited to the snapshot, its tests, and independent fixture evidence.

**Verification:**

```powershell
poetry run pytest tests/test_transformer.py -q
poetry run pytest tests/test_bpe.py tests/test_word2vec.py tests/test_transformer.py -q
poetry run pytest
poetry run ruff check .
poetry run mypy src
git diff --check
git status --short
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.
- `git diff --check` reports no whitespace errors.
- `git status --short` contains only the expected Ticket 014 files and any narrowly justified BPE correction.

## Focused verification plan

```powershell
poetry run pytest tests/test_transformer.py -q
poetry run pytest tests/test_bpe.py tests/test_word2vec.py tests/test_transformer.py -q
```

Expected result:

- Exact Transformer preprocessing, immutability, shard edge cases, lazy publication, concurrency, failure retry, and affected reusable-BPE regressions pass.

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
- Ruff passes.
- Strict mypy returns no issues.
- No whitespace error or out-of-scope file change remains.

## Manual acceptance checklist

- [ ] The Python corpus matches every reference story and its exact order.
- [ ] The BPE training text uses the exact reference delimiter, normalization, and merge limit.
- [ ] The complete ordered Merge Table matches independent reference evidence.
- [ ] Every tokenized story matches exactly and no merge crosses a Pre-token boundary.
- [ ] The ordered Vocabulary and every token index match the fixture.
- [ ] The complete token-ID stream matches exactly.
- [ ] Every Training Sequence has `16` input IDs and `16` next-token target IDs in exact order.
- [ ] Generation seed IDs equal the first three IDs of the complete stream.
- [ ] Exactly four shard boundaries are present.
- [ ] Shards use `ceil(sequence_count / 4)`, are contiguous, cover every sequence once, and are independent of CPU or worker count.
- [ ] Small sequence counts produce deterministic empty trailing shards.
- [ ] The public snapshot contains all required artifacts and no request-owned or mutable training state.
- [ ] Every nested public collection is immutable or read-only.
- [ ] Sequential and concurrent callers observe the one complete published snapshot.
- [ ] A controlled first construction failure publishes nothing and a later retry succeeds.
- [ ] Changing `epochs`, `temperature`, `topP`, `numLayers`, or `maxTokens` outside the preprocessing boundary changes no preprocessing artifact.
- [ ] Expected values come from independent reference evidence rather than the production operation under test.
- [ ] No route, frontend, worker, numerical training, persistence, dependency, or lockfile change is included.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/transformer.py
tests/test_transformer.py
tests/fixtures/transformer_preprocessing_reference.json
```

Conditionally changed only if independent reference evidence proves a real reusable-BPE incompatibility:

```text
src/how_llms_work/ml/bpe.py
tests/test_bpe.py
```

Optional only when the repository deliberately re-exports public ML symbols there; otherwise leave unchanged:

```text
src/how_llms_work/ml/__init__.py
```

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/ml/word2vec.py
src/how_llms_work/routes/
tests/test_matrix.py
tests/test_math_utils.py
tests/test_neural_net.py
tests/test_train_embed_route.py
pyproject.toml
poetry.lock
.data/
frontend/
TypeScript reference source
```

## Risk notes and safeguards

1. **Risk:** The Python corpus is copied from an older or Word2Vec-specific source instead of the current Transformer reference.
   - **Safeguard:** Capture story values and order from the latest supplied TypeScript corpus import used by `train-transformer/train.ts`, and compare the entire tuple exactly.

2. **Risk:** Joining stories with the wrong delimiter changes Pre-token counts, merge frequencies, and the complete Merge Table.
   - **Safeguard:** Record and test the exact BPE training text construction and independent metadata before implementing Python composition.

3. **Risk:** The developer assumes Word2Vec and Transformer preprocessing are identical because they share stories and BPE helpers.
   - **Safeguard:** Maintain a dedicated Transformer fixture and inspect every Transformer-specific merge, Vocabulary, ID, sequence, and seed step.

4. **Risk:** A Python `set`, sorting operation, or reordered dictionary changes Vocabulary and IDs.
   - **Safeguard:** Preserve the exact reference traversal, use insertion-ordered containers during construction, and compare the complete Vocabulary and index mapping.

5. **Risk:** Story tokens are flattened too early and BPE merges cross story or Pre-token boundaries.
   - **Safeguard:** Apply merges story by story through `apply_merges()`, retain tokenized stories separately, and protect flattening behavior with exact fixtures.

6. **Risk:** Sequence creation has an off-by-one error or incorrectly resets at story boundaries.
   - **Safeguard:** Store exact sequence count plus complete or representative boundary sequences and explicitly test the first and last valid start positions.

7. **Risk:** Seed IDs are derived from a selected story, Vocabulary prefix, or request text.
   - **Safeguard:** Assert exact equality with `snapshot.token_ids[:3]`.

8. **Risk:** Shards depend on host CPU count or physical workers.
   - **Safeguard:** Make the shard function accept only sequence count, fix the shard count at four, and test the same results under simulated CPU-count changes without consulting those values.

9. **Risk:** `ceil` and clamping produce overlapping, skipped, or negative ranges for small counts.
   - **Safeguard:** Add independent tables for counts below, equal to, and above four and assert contiguous full coverage plus deterministic empty ranges.

10. **Risk:** A frozen outer dataclass still exposes mutable inner lists or dictionaries.
    - **Safeguard:** Convert every nested sequence to tuples and expose indices through a read-only mapping that has no retained mutable alias.

11. **Risk:** A `MappingProxyType` remains dynamically tied to a dictionary that later construction code mutates.
    - **Safeguard:** Finish and validate the private dictionary, create the proxy during final immutable conversion, and never retain or mutate the backing dictionary afterward.

12. **Risk:** Double-checked access publishes a partially filled snapshot.
    - **Safeguard:** Construct and validate into local variables, create one final immutable object, then perform the sole global assignment.

13. **Risk:** A failed constructor leaves a placeholder or poisoned sentinel that blocks retries.
    - **Safeguard:** Keep the global reference `None` until successful completion and add a controlled failure-then-success public-getter test.

14. **Risk:** Concurrent tests pass accidentally because preprocessing was initialized by an earlier test.
    - **Safeguard:** Isolate lifecycle tests through a fresh module boundary and synchronize first callers before invoking the public getter.

15. **Risk:** Lifecycle tests become coupled to private cache, lock, or helper names.
    - **Safeguard:** Assert only public getter outcomes, returned snapshot completeness, public object reuse, propagated failure, and successful retry; patch a stable external dependency and let pytest restore it.

16. **Risk:** A complete fixture is generated by the Python code under test, reproducing the same bug in expected and actual values.
    - **Safeguard:** Record fixture provenance and prohibit imports from `how_llms_work.ml.transformer` in extraction or reference calculations.

17. **Risk:** The large complete fixture becomes hard to review.
    - **Safeguard:** Keep deterministic formatting, descriptive top-level sections, summary counts, and representative human-readable metadata alongside complete exact arrays.

18. **Risk:** Request fields leak into preprocessing because later Transformer tickets need them.
    - **Safeguard:** Give the getter no request parameters and keep request validation, architecture selection, sampling, and training outside the snapshot.

19. **Risk:** Ticket 014 expands into model initialization, Transformer mathematics, worker processes, HTTP streaming, or persistence.
    - **Safeguard:** Enforce the expected-file list and leave `matrix.py`, `transformer_worker.py`, routes, schemas, `main.py`, `.data`, and frontend files unchanged.

20. **Risk:** A formatter or broad cleanup changes completed code.
    - **Safeguard:** Format only changed Python files, run `git diff --check`, inspect `git status --short`, and reject unrelated changes before commit.

## Commit guidance after tests pass

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Produce immutable Transformer preprocessing snapshot
```

Commit body should mention:

- the exact fixed ordered Transformer Training Corpus and Transformer-specific BPE artifacts;
- exact tokenized stories, ordered Vocabulary, stable IDs, fixed-length Training Sequences, and first-three generation seed IDs;
- exactly four deterministic contiguous Logical Training Shard boundaries, including empty trailing shard behavior;
- nested immutable/read-only public values;
- lazy double-checked all-or-nothing publication and successful failure retry;
- independent exact fixtures and public-boundary sequential/concurrent tests;
- no Transformer mathematics, workers, route, persistence, frontend, request, dependency, or lockfile changes;
- the exact pytest, Ruff, mypy, and diff commands actually executed and their observed results.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- this `plan014.md`;
- `014-produce-immutable-transformer-preprocessing-snapshot.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`;
- `py_llm_pipeline_explorer_file_structure(37).md`;
- the latest `llm_works_file_structure.md`.

`implement-prompt` must inspect the repository again, establish its own baseline before editing, preserve user changes, implement only Ticket 014, create independent reference evidence before production construction, run focused and full verification, report actual command results honestly, inspect the final scope, and create the implementation commit only after all required checks pass.
