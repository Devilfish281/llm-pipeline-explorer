---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: 002
source_work_item: 02-provide-deterministic-reference-compatible-bpe-tokenizer.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(5).md
behavior_reference: llm_works_file_structure(1).md
baseline_test_status: not-supplied
recommended_next_prompt: implement-prompt
---

# Plan for Issue 02: Provide a deterministic reference-compatible BPE Tokenizer

## Initial checklist

- Confirm Ticket 02 is the only work item in scope and has no blockers.
- Treat `py_llm_pipeline_explorer_file_structure(5).md` as the latest current Python code source.
- Use `llm_works_file_structure(1).md` only as the TypeScript behavioral reference.
- Record that no baseline command result was supplied; establish a fresh baseline before editing.
- Limit production work to the reusable BPE module and its public-interface tests.
- Finish with focused pytest, the complete pytest suite, Ruff, and strict mypy.

## Source-of-truth hierarchy

1. The user's latest correction and current architecture: the backend is Python-only; TypeScript server files are behavioral reference material, not current backend code.
2. Ticket 02 defines the immediate required behavior, approved test seam, constraints, and out-of-scope boundary.
3. `py_llm_pipeline_explorer_file_structure(5).md` is the current-code authority.
4. `SPEC.md`, `CONTEXT.md`, and `llm_works_file_structure(1).md` provide durable decisions, canonical terminology, and reference behavior.
5. Older snapshots, prior plans, and stale specification observations are non-authoritative when they conflict with the latest Python export.

## Work-item summary

Implement the reusable educational BPE Tokenizer in `backend/src/how_llms_work/ml/bpe.py` through typed public operations equivalent to `count_words()`, `train_bpe()`, and `apply_merges()`. The module must reproduce the TypeScript reference behavior for Pre-token boundaries, ASCII-style word classification, frequency weighting, deterministic tie handling, non-overlapping merge replacement, ordered Merge Table replay, edge cases, and the 1,000-merge ceiling.

This ticket does not expose or register `POST /bpe-tokenize`. HTTP validation, SSE payloads, delays, headers, route registration, and frontend integration belong to Ticket 03.

## Baseline evidence

- **Status:** Not supplied
- **Command:** `Not supplied`
- **Result:** No current-session or user-reported baseline result is available. Cache directories in the source export are not proof that the current suite passes.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing.

## Current code observations from the latest source

- `backend/src/how_llms_work/ml/bpe.py` exists but is empty; none of Ticket 02's reusable public operations or types are implemented.
- `backend/tests/test_bpe.py` does not exist.
- `backend/tests/test_simple_chat.py` now exists and exercises the completed Simple Chat HTTP/SSE contract.
- `backend/src/how_llms_work/schemas.py` and `backend/src/how_llms_work/sse.py` are already populated, and `routes/simple_chat.py` already imports their shared seams. This supersedes older specification notes that described those files as empty.
- `backend/pyproject.toml` targets Python 3.12, configures pytest to discover `tests`, configures Ruff for Python 3.12, and enables strict mypy with `mypy_path = "src"`.
- The TypeScript reference exposes `countWords()`, `trainBpe()`, and `applyMerges()`; it also contains `mergeTokens()` and `trainBpeOnText()`. Ticket 02 permits the sequence-rewrite operation to remain private and explicitly defers `train_bpe_on_text()` and custom regex support.
- Reference Pre-tokenization groups ASCII word characters, emits each whitespace character separately, and emits each non-word/non-whitespace character separately.
- The reference preserves first encounter order with insertion-ordered maps and changes the winning pair only when a later frequency is strictly greater, not equal.
- The reference counts adjacent pair candidates with repeated Pre-token frequency weights, then replaces selected pair occurrences non-overlappingly from left to right.
- The reference records each learned merge in order and replays that order independently inside each newly Pre-tokenized input unit.
- Compatibility requires ASCII-style word classification without accidentally narrowing Unicode whitespace behavior. A blanket ASCII regex mode would be too broad because it also changes whitespace classification.
- No current code requires changes to routes, schemas, SSE helpers, FastAPI registration, dependencies, or frontend files for Ticket 02.

## Acceptance criteria coverage

- **Already satisfied and evidenced:** Ticket 01 prerequisites are present in the current source; the reusable BPE ticket itself has no implemented acceptance behavior.
- **Behavior present but evidence incomplete:** None.
- **Partially implemented:** None; only the empty destination module exists.
- **Not implemented:** All Ticket 02 acceptance criteria, including typed public operations, Pre-token counting, deterministic training, ordered merge application, edge handling, state isolation, and BPE-specific tests.
- **Evidence limitation:** No baseline commands were run during planning. The TypeScript reference supplies expected behavior, but no Python parity tests or implementation currently exist.

## Files to inspect before editing

1. `backend/pyproject.toml` — pytest discovery, Ruff target, strict mypy settings, and existing dependencies.
2. `backend/src/how_llms_work/ml/bpe.py` — empty production destination for all Ticket 02 public symbols.
3. `backend/tests/test_simple_chat.py` — current pytest style, import conventions, and assertion conventions; do not reuse its HTTP-only seam for this algorithm ticket.
4. `src/server/lib/bpe.ts` within `llm_works_file_structure(1).md` — `MAX_MERGES`, `PRE_TOKEN_RE`, `Merge`, `countWords()`, `mergeTokens()`, `trainBpe()`, and `applyMerges()` reference behavior.
5. `SPEC.md` and `CONTEXT.md` — approved BPE ownership, canonical language, hard scope boundary, and testing decisions.

## Step 1 — Establish and record the pre-edit backend baseline

**Files and symbols:**
- `backend/pyproject.toml` — pytest, Ruff, and mypy configuration.
- `backend/src/how_llms_work/ml/bpe.py` — confirm it is still empty before editing.
- `backend/tests/` — confirm the current tests and any user changes made after the export.

**Purpose:**
Create implementation-session evidence before changing Ticket 02 files and separate pre-existing environment or quality failures from BPE work.

**Actions:**
- Work from the `backend` directory.
- Run the complete current pytest suite and record its exact exit status and output.
- Run Ruff and strict mypy and record any pre-existing findings.
- Confirm the BPE module imports as an empty module before symbols are added.
- Do not edit files during this step.

**Guardrails:**
- Do not claim a passing baseline unless the implementation session receives successful output.
- Do not repair unrelated Simple Chat, dependency, packaging, or environment findings under Ticket 02.
- Do not add or upgrade dependencies or regenerate `poetry.lock`.

**Expected result:**
- A grounded baseline record exists for pytest, Ruff, and mypy.
- Any pre-existing failure is clearly distinguished from later Ticket 02 results.

**Verification:**

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Step 2 — Add public-interface acceptance tests from fixed reference cases

**Files and symbols:**
- `backend/tests/test_bpe.py` — new tests importing only the public reusable BPE interface from `how_llms_work.ml.bpe`.

**Purpose:**
Encode every Ticket 02 acceptance criterion through the approved public seam before production implementation. These tests should initially fail because the required public symbols do not exist.

**Actions:**
- Create focused tests for `count_words()`, `train_bpe()`, `apply_merges()`, and the public Merge record.
- Assert representative fixed reference cases, including:
  - `the cat and the cat` counts `the` twice, the individual space four times, `cat` twice, and `and` once.
  - `café café` classifies `caf` as an ASCII word Pre-token, each `é` as its own non-word Pre-token, and the separating space independently.
  - `cat cat car` selects `c + a` first with frequency `3`, then learns `ca + t` with frequency `2`, then `ca + r` with frequency `1`.
  - `ab ac` resolves the initial frequency tie in favor of `a + b`, the first encountered pair.
  - `aaaa` discovers the overlapping adjacent pair frequency but performs the selected replacement non-overlappingly from left to right.
  - A synthetic ordered Merge Table applies `c + a` before `ca + t`, producing `cat`; reversing the table must not be treated as equivalent.
  - Single-character, whitespace-only, and punctuation-only inputs finish with no invalid cross-boundary merge.
  - A configurable small merge bound stops training at that bound, and the production ceiling remains 1,000.
  - Separate training calls remain independent; training or using one Merge Table cannot alter a prior or later call.
- Assert returned values and observable merge records, not private helper names, local data structures, or loop organization.
- Keep tests Python-only; do not execute Node, pnpm, Hono, or TypeScript.

**Guardrails:**
- Do not test HTTP, FastAPI, SSE, delays, payload names, route registration, or frontend rendering in this file.
- Do not import or assert a private merge helper.
- Do not require `train_bpe_on_text()`, custom Pre-token patterns, NumPy, persistence, or future-phase behavior.
- Preserve deterministic expected values derived from the supplied reference rather than generating expectations from the Python implementation under test.

**Expected result:**
- `backend/tests/test_bpe.py` comprehensively expresses Ticket 02 through the approved public seam.
- The first focused run fails for missing BPE symbols or behavior, providing legitimate red evidence.

**Verification:**

```powershell
poetry run pytest tests/test_bpe.py
```

Expected at this point:

- Failure is attributable to the intentionally missing BPE implementation, not test collection, import path, or unrelated project behavior.

## Step 3 — Implement the typed Pre-token and Merge foundations

**Files and symbols:**
- `backend/src/how_llms_work/ml/bpe.py` — `MAX_MERGES`, the compiled Pre-token pattern, the public Merge record, and `count_words()`.

**Purpose:**
Satisfy the classification, counting, typing, and stateless-foundation criteria before adding iterative training.

**Actions:**
- Define the 1,000-merge ceiling as a typed module constant.
- Define a fully typed, value-comparable, preferably immutable Merge record carrying:
  - the two-token BPE Pair;
  - the merged token;
  - the selected frequency.
- Define one compiled Pre-token pattern that:
  - groups ASCII letters, digits, and underscore into word sequences;
  - emits each Unicode whitespace character as an individual Pre-token;
  - emits each remaining character as an individual non-word Pre-token.
- Avoid applying a blanket ASCII regex flag that would also narrow whitespace classification.
- Implement `count_words(text)` with fresh per-call state and insertion-preserving counts.
- Do not accept a custom pattern argument in this focused ticket.
- Keep all public and internal definitions fully annotated for strict mypy.

**Guardrails:**
- Do not add module-level mutable counters, caches, learned merges, or word-split state.
- Do not normalize, trim, lowercase, or otherwise transform input text.
- Do not add HTTP, Pydantic, SSE, NumPy, or route concerns.
- Do not expose future-phase convenience APIs.

**Expected result:**
- Pre-token and counting tests pass, including the accented Unicode parity case.
- The public Merge record and constants are available for training and application steps.
- No Tokenization Run state survives a call.

**Verification:**

```powershell
poetry run pytest tests/test_bpe.py -k "count or pretoken or unicode"
poetry run mypy src
```

## Step 4 — Implement deterministic BPE training through the ordered Merge Table

**Files and symbols:**
- `backend/src/how_llms_work/ml/bpe.py` — private non-overlapping sequence rewrite helper and public `train_bpe()`.

**Purpose:**
Satisfy frequency weighting, deterministic tie selection, non-overlapping replacement, learned order, early termination, merge ceiling, and training-state isolation.

**Actions:**
- Build fresh character-token splits for each counted Pre-token at the start of every call.
- Traverse Pre-tokens in first encounter order from `count_words()`.
- Accumulate adjacent BPE Pair frequencies in first encounter order and add each Pre-token's occurrence count as the weight.
- Select the first highest-frequency pair by preserving encounter order and replacing the current winner only for a strictly larger frequency.
- Use a private helper to replace selected pair occurrences non-overlappingly from left to right within each Pre-token.
- Apply each selected pair separately to every Pre-token split so a merge never crosses a boundary.
- Append a new Merge record after each selection, preserving learned order.
- Stop when no adjacent pair remains or the effective merge bound reaches the 1,000 ceiling.
- Return only the ordered Merge Table required by the route; keep evolving word splits private.
- Keep the returned Merge Table independent from all later calls.

**Guardrails:**
- Do not use unordered selection such as a set or a tie-breaking `max()` that changes encounter semantics.
- Do not use a delimiter-encoded string as the Python BPE Pair when a typed two-item tuple can represent it directly.
- Do not count pairs across Pre-token boundaries.
- Do not merge overlapping occurrences during replacement.
- Do not persist or cache trained state.

**Expected result:**
- Repeated Pre-token weighting, deterministic ties, non-overlapping replacement, early termination, bounded training, and state-isolation tests pass.
- Learned Merge records match the reference-compatible order and frequencies.

**Verification:**

```powershell
poetry run pytest tests/test_bpe.py -k "train or frequency or tie or overlap or limit or independent"
poetry run mypy src
```

## Step 5 — Implement ordered Merge Table application

**Files and symbols:**
- `backend/src/how_llms_work/ml/bpe.py` — public `apply_merges()` and reuse of the private sequence rewrite helper.

**Purpose:**
Complete the public reusable interface by applying learned behavior to original BPE Training Text without crossing Pre-token boundaries.

**Actions:**
- Pre-tokenize the supplied text using the same fixed compatibility pattern as training.
- Split each Pre-token into character tokens.
- Replay every Merge record in learned order within that Pre-token.
- Append the resulting tokens to one output sequence while preserving the original Pre-token order.
- Return a fresh typed token sequence for every call.
- Ensure single-character, whitespace-only, punctuation-only, repeated, tied, and Unicode reference cases produce the fixed expected tokens.

**Guardrails:**
- Do not train implicitly inside `apply_merges()`.
- Do not reorder, deduplicate, or optimize away Merge records.
- Do not merge across Pre-token boundaries.
- Do not add `train_bpe_on_text()` or custom patterns.
- Do not retain output or intermediate state between calls.

**Expected result:**
- All `backend/tests/test_bpe.py` tests pass through the public BPE seam.
- The module provides the minimum reusable behavior required by the later route ticket.

**Verification:**

```powershell
poetry run pytest tests/test_bpe.py
poetry run mypy src
```

## Step 6 — Run complete validation and inspect scope

**Files and symbols:**
- `backend/src/how_llms_work/ml/bpe.py` — complete Ticket 02 production diff.
- `backend/tests/test_bpe.py` — complete Ticket 02 acceptance evidence.
- Entire backend tree — regression and scope inspection only.

**Purpose:**
Prove the complete reusable tokenizer change works with the existing backend and contains no Ticket 03 or future-phase work.

**Actions:**
- Run the focused BPE test file once more.
- Run the complete pytest suite once at the end.
- Run Ruff and strict mypy using the repository configuration.
- Inspect the final diff and confirm only the expected two files changed.
- Confirm no dependencies, routes, shared SSE/schema files, frontend files, caches, or generated artifacts changed.
- Record exact command results honestly.

**Guardrails:**
- Do not fix unrelated failures or formatting outside the Ticket 02 files.
- Do not implement or register `/bpe-tokenize`.
- Do not add Node/TypeScript execution to backend validation.
- Do not claim success if any configured check fails.

**Expected result:**
- Focused BPE tests pass.
- The complete existing backend test suite passes.
- Ruff and strict mypy pass, or any pre-existing unrelated failure is reported separately without scope expansion.
- The final diff contains only the reusable BPE implementation and its public-interface tests.

**Verification:**

```powershell
poetry run pytest tests/test_bpe.py
poetry run pytest
poetry run ruff check .
poetry run mypy src
git diff -- backend/src/how_llms_work/ml/bpe.py backend/tests/test_bpe.py
git status --short
```

## Focused verification plan

```powershell
poetry run pytest tests/test_bpe.py
```

Expected result:

- All Ticket 02 public-interface parity tests pass without Node, TypeScript, FastAPI, SSE, or real network execution.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no type errors in `src`.

## Manual acceptance checklist

- [ ] `count_words("the cat and the cat")` preserves reference Pre-token counts and encounter order.
- [ ] ASCII word sequences are grouped, while representative accented or non-Latin characters follow the confirmed reference classification.
- [ ] Each whitespace and punctuation character remains an independent Pre-token.
- [ ] BPE Pair discovery and replacement never cross a Pre-token boundary.
- [ ] Repeated Pre-tokens weight pair frequencies by occurrence count.
- [ ] Equal-frequency pairs select the first pair encountered in reference traversal order.
- [ ] Selected pairs replace non-overlapping occurrences from left to right.
- [ ] Learned Merge records preserve selection order and exact selected frequency.
- [ ] Training stops when no pair remains and never learns more than 1,000 merges.
- [ ] `apply_merges()` replays the Merge Table in learned order and returns reference-compatible tokens.
- [ ] Single-character, whitespace-only, and punctuation-only inputs complete successfully.
- [ ] Separate calls do not share mutable Tokenization Run state.
- [ ] Public BPE symbols pass strict mypy.
- [ ] No HTTP, SSE, route, frontend, dependency, persistence, caching, or future-phase work was added.

## Expected files changed

Likely changed:

```text
backend/src/how_llms_work/ml/bpe.py
backend/tests/test_bpe.py
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
backend/src/how_llms_work/routes/
backend/src/how_llms_work/ml/__init__.py
backend/src/how_llms_work/ml/word2vec.py
backend/src/how_llms_work/ml/transformer.py
backend/pyproject.toml
backend/poetry.lock
frontend/
```

## Risk notes and safeguards

1. **Risk:** A direct Python `\w` pattern can classify Unicode letters as words, diverging from the TypeScript reference.
   - **Safeguard:** Use an explicit ASCII word-character branch and a fixed accented-text parity test.
2. **Risk:** Applying ASCII mode to the entire Python pattern can also narrow `\s`, even though the JavaScript reference recognizes Unicode whitespace.
   - **Safeguard:** Separate ASCII word classification from whitespace classification and include at least one non-ASCII whitespace case when practical.
3. **Risk:** Tie resolution can drift if pair frequencies are collected or selected through unordered data.
   - **Safeguard:** Preserve insertion order end to end and update the winner only on a strictly greater frequency.
4. **Risk:** Counting overlapping pair candidates can be confused with replacing overlapping occurrences.
   - **Safeguard:** Assert the `aaaa` reference case: candidate frequency reflects adjacent positions, while replacement remains non-overlapping and left-to-right.
5. **Risk:** An implementation may accidentally merge across words, spaces, or punctuation by flattening text before training or application.
   - **Safeguard:** Keep per-Pre-token token sequences and assert exact boundary-preserving outputs.
6. **Risk:** Returning internal evolving word splits would enlarge the public API and expose mutable state.
   - **Safeguard:** Return only the ordered Merge Table required by the route and keep training containers local.
7. **Risk:** The 1,000 limit may be treated as a suggestion rather than a ceiling.
   - **Safeguard:** Define one typed ceiling, apply it in the loop bound, and test a smaller explicit bound plus the production constant.
8. **Risk:** Scope may drift into Ticket 03 because the route destination already exists.
   - **Safeguard:** Do not edit routes, `main.py`, shared SSE/schema modules, or frontend files.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Suggested outcome:

```text
Implement deterministic reference-compatible BPE tokenizer
```

Commit body should mention:

- the typed reusable `count_words()`, `train_bpe()`, and `apply_merges()` behavior;
- fixed reference-compatible parity tests;
- the focused and full verification commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 02, `SPEC.md`, `CONTEXT.md`, `py_llm_pipeline_explorer_file_structure(5).md`, and `llm_works_file_structure(1).md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 02, verify the complete change, and create the implementation commit.
