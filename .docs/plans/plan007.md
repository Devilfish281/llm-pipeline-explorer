---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "007"
source_work_item: 007-produce-immutable-reference-compatible-word2vec-training-data.md
source_specification: SPEC.md, amended by the user's approved 423-merge compatibility decision
code_source_of_truth: py_llm_pipeline_explorer_file_structure(17).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 007: Produce immutable reference-compatible Word2Vec training data

## Initial checklist

- Confirm Ticket 007 is the only work item in scope.
- Apply the user's latest correction: request **up to 500** BPE Merges and preserve the complete **423-rule** Merge Table produced by the current TypeScript Reference Implementation.
- Treat `py_llm_pipeline_explorer_file_structure(17).md` as the current Python code source of truth and `llm_works_file_structure(5).md` as behavior evidence only.
- Preserve the existing public BPE boundary and add only the missing Word2Vec preprocessing boundary, independent fixtures, and focused tests.
- Keep routes, schemas, SSE behavior, training mathematics, persistence, frontend code, and dependencies out of this ticket.
- Reconfirm the user-reported baseline before editing and finish with `poetry run pytest`, `poetry run ruff check .`, and `poetry run mypy src`.

## Source-of-truth hierarchy

1. The user's latest explicit correction: request up to 500 BPE Merges; the fixed reference corpus exhausts all candidate pairs after exactly 423 learned merges.
2. Ticket 007 for required preprocessing behavior, testing seam, constraints, and out-of-scope boundaries, with its former “exactly 500” wording superseded only by the correction above.
3. `py_llm_pipeline_explorer_file_structure(17).md` for the current Python implementation.
4. `llm_works_file_structure(5).md` for the exact fixed corpus and TypeScript reference behavior.
5. `SPEC.md`, `CONTEXT.md`, and ADR 0001 decisions for durable compatibility and canonical terminology.
6. Older snippets, prior plans, and assumptions are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 007 introduces the first public Word2Vec module boundary in the Python Backend. It must derive deterministic, reusable preprocessing from the fixed 107-sentence Embedding Training Corpus: the exact lowercased, single-space-joined BPE Training Text; the complete ordered Merge Table produced when the reference BPE is asked for up to 500 merges; tokenized corpus sentences; first-encounter token frequencies; a stable frequency-sorted Vocabulary and deterministic token indices; and ordered Training Pairs for context-window sizes 1 through 5.

The result must be safe to reuse across sequential and concurrent Embedding Training Runs. External callers must not be able to mutate shared corpus-derived data. Query Words are not an input to this preprocessing boundary and must not influence any derived structure.

This ticket does not implement Skip-gram weights, randomness, negative sampling, loss, Embedding Results, HTTP/SSE behavior, persistence, request schemas, or route registration.

## Baseline evidence

- **Status:** User-reported.
- **Commands:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Result:** The user reported that all tests passed, Ruff reported no errors, and strict mypy reported no issues before planning.
- **Planning-session limitation:** No repository test, lint, or typecheck command was executed while creating this plan. A read-only reference calculation was used only to confirm the TypeScript preprocessing outputs described below.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing.

## Current code observations from the latest source

- `src/how_llms_work/ml/word2vec.py` is empty. No public Word2Vec preprocessing API or Word2Vec-specific test file currently exists.
- `src/how_llms_work/ml/bpe.py` already provides the reusable public BPE seam required by this ticket:
  - `count_words()` preserves Pre-token first-encounter order;
  - `train_bpe()` selects frequency ties by first encounter, returns an ordered tuple of frozen `Merge` records, honors an upper merge bound, and stops when no pairs remain;
  - `apply_merges()` replays rules in order within each Pre-token and never merges across Pre-token boundaries.
- `tests/test_bpe.py` already verifies deterministic merge ordering, tie behavior, non-overlapping replacements, Pre-token boundaries, requested merge limits, and isolation of mutable return values.
- The TypeScript Reference Implementation defines `CORPUS` as 107 sentences, builds the BPE Training Text with `CORPUS.join(" ").toLowerCase()`, requests 500 merges, and stops when its pair-frequency map becomes empty.
- A read-only execution of the supplied TypeScript algorithm and exact corpus confirms these reference facts:
  - requested merge limit: `500`;
  - actual learned Merge Table length: `423`;
  - first merge: `("h", "e") -> "he"`, frequency `141`;
  - last merge: `("grow", "s") -> "grows"`, frequency `1`;
  - ordered Vocabulary size: `192`;
  - first ordered Vocabulary Tokens: space, `the`, `a`, `is`, `and`, `cat`, `in`, `dog`, `man`, `woman`, `king`, `queen`, `are`, `of`, `boy`, `girl`, `on`, `at`, `prince`, `princess`;
  - Training Pair counts are `2,092`, `3,970`, `5,634`, `7,084`, and `8,320` for window sizes 1 through 5 respectively.
- The TypeScript `buildVocab()` behavior is first-encounter frequency counting followed by stable descending-frequency sorting. Equal-frequency Tokens therefore remain in first-encounter order.
- TypeScript generates Training Pairs by corpus sentence order, target-token position, and ascending context position from the left edge of the window to the right edge, excluding the target position.
- Query Words appear only in later result selection and are not read while building the corpus, Merge Table, Vocabulary, indices, or Training Pairs.

## Acceptance criteria coverage

- **Already satisfied and evidenced:**
  - The reusable Python BPE seam applies merges only within Pre-token boundaries and in learned order.
  - The BPE seam preserves deterministic pair-frequency tie behavior and stops when no pairs remain.
  - Existing BPE tests protect its general merge, boundary, ordering, and isolation behavior.
- **Behavior present but evidence incomplete:**
  - The existing BPE operations appear compatible with the fixed embedding corpus, but the repository does not yet contain the complete independent 423-merge fixture or corpus-specific tokenization evidence.
- **Partially implemented:**
  - General BPE preprocessing exists, but no Word2Vec module composes it with the fixed corpus, Vocabulary construction, indices, Training Pair generation, or an immutable public result.
- **Not implemented:**
  - The exact 107-sentence Python corpus constant and joined training text.
  - The public immutable Word2Vec preprocessing boundary.
  - Corpus-specific 423-rule Merge Table evidence.
  - Tokenized-corpus, frequency, Vocabulary, and index construction.
  - Ordered Training Pairs for window sizes 1 through 5.
  - Word2Vec preprocessing fixtures, exact public-seam tests, and sequential/concurrent mutation-isolation tests.
- **Evidence limitation:**
  - The complete 423-entry Merge Table and complete pair sequences are not currently checked into the Python Backend. During implementation they must be captured as fixed independent evidence from the supplied TypeScript Reference Implementation, not generated by calling the Python production operation under test.
  - ADR 0001 is referenced through the approved specification and ticket but was not supplied as a separate file in this planning handoff.

## Files to inspect before editing

1. `src/how_llms_work/ml/word2vec.py` — currently empty; this will own the new stable public preprocessing boundary.
2. `src/how_llms_work/ml/bpe.py` — `Merge`, `count_words()`, `train_bpe()`, and `apply_merges()` must be reused rather than duplicated.
3. `tests/test_bpe.py` — existing exact-assertion, boundary, tie, and mutation-isolation test patterns to preserve.
4. `src/routes/train-embed/corpus.ts` in the supplied TypeScript Reference Implementation — exact `CORPUS`, `BPE_MERGE_COUNT`, `BPE_MERGES`, `tokenize()`, and `buildVocab()` behavior evidence only.
5. `src/routes/train-embed/train.ts` in the supplied TypeScript Reference Implementation — exact Training Pair traversal order evidence only.
6. `src/server/lib/bpe.ts` in the supplied TypeScript Reference Implementation — early-stop, tie-order, Pre-token, and ordered-merge behavior evidence only.
7. `pyproject.toml` — confirm the existing pytest, Ruff, and strict mypy workflow; no dependency change is expected.

## Step 1 — Add independent reference fixtures and public-seam acceptance tests

**Files and symbols:**
- `tests/fixtures/word2vec_preprocessing_reference.json` — new fixed reference data captured independently from the supplied TypeScript implementation.
- `tests/test_word2vec.py` — new tests through the stable public Word2Vec module boundary.

**Purpose:**
Create non-circular evidence for every Ticket 007 acceptance criterion before implementing production behavior. This establishes the approved public test seam and protects against reproducing the same mistake in both implementation and expectation code.

**Actions:**
- Capture the exact fixed corpus and reference outputs without importing or calling `how_llms_work.ml.word2vec`.
- Store fixed evidence sufficient to assert:
  - all 107 corpus sentences and their order;
  - the exact lowercased single-space-joined BPE Training Text;
  - the requested merge limit of 500 and the actual complete 423-entry ordered Merge Table, including each pair, merged Token, frequency, and order;
  - representative boundary-sensitive sentence tokenizations;
  - complete ordered token frequencies, Vocabulary, and index assignments;
  - complete ordered Training Pair sequences, or equivalently complete fixed pair fixtures with exact ordered comparison, for every window size from 1 through 5;
  - the confirmed aggregate counts: Vocabulary `192`; pair counts `2,092`, `3,970`, `5,634`, `7,084`, and `8,320`.
- Add public-seam tests that compare production results exactly against the fixed fixture.
- Add explicit assertions for the first and last Merge and the first ordered Vocabulary Tokens so fixture corruption is easy to diagnose.
- Add tests proving Query Words are not accepted by, stored in, or used by the preprocessing boundary.

**Guardrails:**
- Do not generate expected data with the Python production helper being tested.
- Do not execute or retain TypeScript as part of the Python Backend's normal test suite; the committed fixture is static evidence.
- Do not test private helper names, local loops, cache identity, or a specific immutable-container implementation.
- Do not include Skip-gram training, randomness, route, SSE, schema, result, or persistence assertions.

**Expected result:**
- The new tests precisely describe the intended public behavior and initially fail because `word2vec.py` has no public preprocessing implementation.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec.py -q
```

Expected at this stage: focused failures identify the missing public Word2Vec preprocessing boundary, not fixture-generation errors.

## Step 2 — Implement the fixed corpus and immutable BPE-derived preprocessing boundary

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — new public corpus data, immutable preprocessing result, and public construction/access boundary.
- `src/how_llms_work/ml/bpe.py` — inspect and reuse `Merge`, `count_words()`, `train_bpe()`, and `apply_merges()`; edit only if the independent fixture proves a real compatibility gap.

**Purpose:**
Deliver the exact Embedding Training Corpus, BPE Training Text, complete reference Merge Table, and tokenized sentences while preserving the existing educational BPE implementation.

**Actions:**
- Represent the 107 corpus sentences in exact TypeScript order using an immutable sequence.
- Build the BPE Training Text by lowercasing the complete corpus and joining sentences with exactly one ordinary space.
- Request up to 500 merges through the existing `count_words()` and `train_bpe()` boundary.
- Preserve the natural early stop when no pairs remain; assert or otherwise expose the complete reference result of exactly 423 learned merges rather than fabricating 77 additional rules.
- Tokenize each lowercased corpus sentence with `apply_merges()` and the complete learned Merge Table.
- Expose the corpus, training text, Merge Table, and tokenized sentences through a stable public Word2Vec result or accessor whose nested values are read-only to callers.
- Use one clear immutable representation, preferably frozen records plus tuples and read-only mappings, without exposing mutable backing containers.

**Guardrails:**
- Do not copy BPE pair counting, merge selection, or merge application into `word2vec.py`.
- Do not alter the corpus, normalize sentence whitespace beyond the confirmed join/lowercase behavior, or add Query Words.
- Do not force exactly 500 output rules.
- Do not introduce a configurable corpus, merge count, or cache policy.
- Preserve all existing BPE endpoint behavior and tests.

**Expected result:**
- The public Word2Vec boundary returns the exact 107 sentences, joined training text, 423 ordered Merges, and reference tokenized sentences without exposing mutable shared lists or dictionaries.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec.py -q -k "corpus or training_text or merge or token"
poetry run pytest tests/test_bpe.py -q
```

Expected result: corpus, Merge Table, and tokenization assertions pass, and all existing BPE tests remain green.

## Step 3 — Build deterministic frequencies, Vocabulary, and token indices

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — public preprocessing result and Vocabulary construction behavior.
- `tests/test_word2vec.py` — exact frequencies, ordering, stable-tie, and index assertions.

**Purpose:**
Satisfy the Vocabulary and token-index acceptance criteria while preserving first-encounter order for equal frequencies.

**Actions:**
- Traverse tokenized sentences in corpus order and Tokens in sentence order.
- Count each Token while preserving the first position at which it is encountered.
- Produce the ordered Vocabulary by sorting only on descending frequency and relying on stable ordering for ties.
- Assign each Token's integer index from its position in the ordered Vocabulary.
- Expose frequencies, ordered Vocabulary, and token-to-index data through the same immutable public preprocessing boundary.
- Compare the complete production outputs against the independent fixture, including the confirmed Vocabulary size of 192 and representative equal-frequency tie groups.

**Guardrails:**
- Do not add a secondary alphabetical tie-breaker.
- Do not use a set or another unordered intermediate that discards first-encounter information.
- Do not derive expected order from production output in tests.
- Do not expose a caller-mutable token-to-index dictionary.

**Expected result:**
- Every Vocabulary Token has one deterministic index, and equal-frequency Tokens remain in their TypeScript first-encounter order.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec.py -q -k "frequency or vocab or index or tie"
```

Expected result: complete frequency, Vocabulary, stable-tie, and index fixtures pass exactly.

## Step 4 — Generate immutable ordered Training Pairs for window sizes 1 through 5

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — public Training Pair generation through the preprocessing boundary.
- `tests/test_word2vec.py` — exact ordered-pair fixtures for all supported window sizes.

**Purpose:**
Complete the reusable preprocessing data needed by later deterministic Skip-gram training tickets.

**Actions:**
- Convert each tokenized sentence to its deterministic Vocabulary indices.
- For each supported window size, traverse:
  1. corpus sentences in original order;
  2. target positions from left to right;
  3. context positions in ascending order from the bounded left edge through the bounded right edge;
  4. omit only the target position itself.
- Return immutable ordered target-context pairs.
- Compare complete pair order against independent fixtures for window sizes 1, 2, 3, 4, and 5.
- Assert the confirmed pair counts:
  - window 1: `2,092`;
  - window 2: `3,970`;
  - window 3: `5,634`;
  - window 4: `7,084`;
  - window 5: `8,320`.

**Guardrails:**
- Do not shuffle, deduplicate, batch, sort, or otherwise reorder Training Pairs.
- Do not cross sentence boundaries.
- Do not include Query Words or any request-owned data.
- Do not add route-level validation or behavior outside the supported 1-through-5 seam required by this ticket.

**Expected result:**
- The public Word2Vec boundary returns exact reference-compatible ordered Training Pairs for all five supported context-window sizes.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec.py -q -k "training_pair or window"
```

Expected result: every complete pair fixture and aggregate count passes exactly.

## Step 5 — Prove mutation isolation and finish the focused vertical slice

**Files and symbols:**
- `src/how_llms_work/ml/word2vec.py` — immutable public result and any internal shared preprocessing state.
- `tests/test_word2vec.py` — repeated-call, mutation-attempt, sequential-isolation, and concurrent-isolation tests.

**Purpose:**
Ensure corpus-derived data can be reused safely by later Embedding Training Runs without allowing one caller to change another caller's observations.

**Actions:**
- Add tests that attempt to mutate every publicly reachable nested structure: corpus, Merge Table, tokenized sentences, frequencies, Vocabulary, indices, and Training Pairs.
- Verify mutation is rejected or affects only a caller-owned copy and never changes a later result.
- Call the public boundary repeatedly and assert equivalent values.
- Exercise concurrent reads/construction through the public seam and assert every result is equivalent to the fixed reference fixture.
- Verify the public API has no Query Word parameter and that changing a separate Query Word list has no effect on preprocessing outputs.
- Review the final diff to ensure only preprocessing, tests, and fixed reference evidence were added.

**Guardrails:**
- Test externally observable isolation, not object identity or a particular cache implementation.
- Do not add global mutable state, a configurable cache, locks, semaphores, queues, or request orchestration.
- Do not broaden the implementation into numerical Word2Vec training or HTTP behavior.

**Expected result:**
- Repeated and concurrent use produces equivalent deterministic values, and no caller can mutate shared preprocessing observed by another run.

**Verification:**

```powershell
poetry run pytest tests/test_word2vec.py -q -k "immutable or mutation or isolation or concurrent or query"
```

Expected result: all isolation and Query Word independence assertions pass.

## Focused verification plan

Run from the backend project root:

```powershell
poetry run pytest tests/test_word2vec.py tests/test_bpe.py
```

Expected result:

- The independent fixed fixture matches the public Word2Vec preprocessing boundary exactly.
- The corpus has 107 sentences.
- Asking for up to 500 BPE Merges produces the complete 423-rule reference Merge Table.
- The ordered Vocabulary contains 192 Tokens.
- All five exact Training Pair fixtures and counts pass.
- Existing BPE behavior remains unchanged.

Then run affected quality checks:

```powershell
poetry run ruff check src/how_llms_work/ml/word2vec.py tests/test_word2vec.py
poetry run mypy src
```

Expected result:

- Ruff reports no errors in the affected Python files.
- Strict mypy reports no issues in `src`.

## Full verification plan

Run once after all focused checks pass:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no errors.
- Strict mypy reports no issues.

## Manual acceptance checklist

- [ ] The Python corpus contains exactly 107 sentences in exact TypeScript order.
- [ ] The BPE Training Text is exactly `" ".join(corpus).lower()` in observable content.
- [ ] The merge request limit is 500, but the complete natural result contains exactly 423 rules.
- [ ] The first Merge is `("h", "e") -> "he"` with frequency 141.
- [ ] The last Merge is `("grow", "s") -> "grows"` with frequency 1.
- [ ] Representative sentence tokenization never crosses Pre-token boundaries and matches fixed evidence.
- [ ] Vocabulary frequencies preserve first encounter, and descending sorting preserves ties.
- [ ] The Vocabulary size is 192 and every Token has one deterministic index.
- [ ] Training Pair order is sentence, target position, then ascending context position.
- [ ] Pair counts for windows 1 through 5 are 2,092; 3,970; 5,634; 7,084; and 8,320.
- [ ] Publicly reachable preprocessing data cannot be mutated in a way that changes later or concurrent results.
- [ ] Query Words are absent from the preprocessing API and do not alter any derived structure.
- [ ] No route, schema, SSE, persistence, frontend, numerical training, Transformer, or dependency change is present.

## Expected files changed

Likely changed:

```text
src/how_llms_work/ml/word2vec.py
tests/test_word2vec.py
tests/fixtures/word2vec_preprocessing_reference.json
```

Conditionally changed only if independent evidence reveals a genuine reusable-BPE mismatch:

```text
src/how_llms_work/ml/bpe.py
tests/test_bpe.py
```

## Files not to change

```text
src/how_llms_work/main.py
src/how_llms_work/schemas.py
src/how_llms_work/sse.py
src/how_llms_work/routes/
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/matrix.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/ml/transformer_worker.py
tests/test_simple_chat.py
tests/test_bpe_tokenize.py
tests/test_neural_net.py
tests/test_neural_net_route.py
tests/test_neural_net_persistence.py
.data/
pyproject.toml
poetry.lock
frontend/
SPEC.md
CONTEXT.md
007-produce-immutable-reference-compatible-word2vec-training-data.md
```

## Risk notes and safeguards

1. **Risk:** The 107-sentence corpus is copied with a missing sentence, changed punctuation, reordered entry, or altered whitespace.
   - **Safeguard:** Compare the complete immutable corpus and joined training text to fixed independent fixtures.

2. **Risk:** An implementer treats 500 as a required output count and invents extra Merges after all candidate pairs are exhausted.
   - **Safeguard:** Assert the requested limit separately from the exact natural output of 423 rules, including first and last Merge fixtures.

3. **Risk:** Vocabulary ties drift because an alphabetical or unordered tie-breaker is introduced.
   - **Safeguard:** Preserve first-encounter frequency insertion order, sort only by descending count, and assert representative and complete tie ordering.

4. **Risk:** Training Pair order changes through set use, sorting, vectorization, or traversal refactoring.
   - **Safeguard:** Compare complete ordered fixtures for every supported window and retain exact traversal-order assertions.

5. **Risk:** A frozen outer object still exposes mutable nested lists or dictionaries.
   - **Safeguard:** Use immutable nested values or read-only mappings and explicitly attempt mutation at every public level.

6. **Risk:** Tests become circular by generating expectations with the production Python implementation.
   - **Safeguard:** Commit fixed fixtures captured independently from the supplied TypeScript reference and ensure fixture-loading tests do not import production code during expectation construction.

7. **Risk:** Editing shared BPE code regresses the working BPE Tokenizer Learning Demo.
   - **Safeguard:** Expect no BPE edit; if a proven compatibility correction is necessary, add the smallest focused regression and run both `tests/test_bpe.py` and the full suite.

8. **Risk:** The ticket expands into later Word2Vec phases.
   - **Safeguard:** Prohibit PRNG, weights, negative sampling, training epochs, result construction, routes, SSE, schemas, persistence, and frontend changes in the final diff.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

A suitable outcome-oriented subject would describe immutable reference-compatible Word2Vec preprocessing without claiming numerical training or route completion.

Commit body should mention:

- the exact fixed corpus, 423-rule Merge Table, deterministic Vocabulary, and ordered Training Pairs;
- independent reference fixtures and mutation-isolation coverage;
- verification with `poetry run pytest`, `poetry run ruff check .`, and `poetry run mypy src`.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 007, the corrected source requirement, `SPEC.md`, `CONTEXT.md`, the latest Python source export, and the supplied TypeScript Reference Implementation.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only this work item, verify the complete change, and create the implementation commit.
