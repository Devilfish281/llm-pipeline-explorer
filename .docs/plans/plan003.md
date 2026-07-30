---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: 003
source_work_item: 03-stream-complete-bpe-tokenization-runs-through-fastapi(2).md
source_specification: SPEC.md
source_context: CONTEXT.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(7).md
behavior_reference: llm_works_file_structure(3).md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 03: Stream complete BPE Tokenization Runs through FastAPI

## Initial checklist

- Confirm Ticket 03 is the only work item in scope and that Tickets 01 and 02 are present in the latest Python source.
- Treat `py_llm_pipeline_explorer_file_structure(7).md` as the current-code source of truth.
- Use `llm_works_file_structure(3).md` only as the TypeScript behavioral reference for the unchanged frontend contract.
- Record the three supplied quality-check results as user-reported, not tool-verified in this planning session.
- Limit production changes to the BPE route and FastAPI router registration; add only HTTP/SSE acceptance coverage needed for this ticket.
- Finish with focused BPE tests, regression tests, the complete pytest suite, Ruff, and strict mypy.

## Source-of-truth hierarchy

1. The user's latest explicit direction: the backend is Python/FastAPI only, the latest complete Python export is current-code authority, and the supplied TypeScript backend is behavioral reference material only.
2. Ticket 03 defines the immediate required behavior, approved test seams, blockers, constraints, and out-of-scope boundary.
3. `py_llm_pipeline_explorer_file_structure(7).md` defines the current Python implementation and supersedes stale source-state observations in older artifacts.
4. `SPEC.md` and `CONTEXT.md` define durable behavior decisions and canonical BPE terminology.
5. `llm_works_file_structure(3).md` defines the observable TypeScript reference contract for `POST /bpe-tokenize` but is not runtime code to retain or execute.
6. Older source exports, snippets, plans, and assumptions are non-authoritative when they conflict with the latest Python export.

## Work-item summary

Complete the Phase 2 vertical slice by implementing and registering `POST /bpe-tokenize` in FastAPI. Each valid request must use the shared `ChatRequest`, train a fresh in-memory Merge Table through the existing reusable BPE API, and return a shared SSE response containing exactly `init`, zero or more ordered `merge` events, and `result`. Payload keys, values, ordering, initialization truncation, merge statistics, one-decimal Compression Ratio, headers, and the 800-millisecond production initialization delay must preserve the frontend contract.

The latest source already contains the Ticket 01 shared schema/SSE infrastructure and the Ticket 02 deterministic BPE implementation with public-interface tests. Therefore, this plan does not rewrite those completed prerequisites. It adds the missing HTTP route, registers it, and proves the endpoint contract through FastAPI `TestClient` while preserving existing Simple Chat and health behavior.

## Baseline evidence

- **Status:** User-reported
- **Commands and results:**
  - `poetry run pytest` — user reported that all tests passed before planning.
  - `poetry run ruff check .` — user reported that all checks passed before planning.
  - `poetry run mypy src` — user reported `Success: no issues found` before planning.
- **Planning rule:** These results are not tool-verified in this planning session. The implementation run must establish or reconfirm its own baseline before editing and report the actual output honestly.

## Current code observations from the latest source

- `backend/src/how_llms_work/routes/bpe_tokenize.py` exists but is empty, so no BPE FastAPI router, stream generator, feature payload construction, or production delay currently exists.
- `backend/src/how_llms_work/main.py` registers only `simple_chat_router`; `POST /bpe-tokenize` is not currently available from the application.
- `backend/src/how_llms_work/schemas.py` already defines the shared `ChatRequest` with `message: str = Field(min_length=1)`. It does not trim, normalize, reject whitespace-only text, or impose a maximum length.
- `backend/src/how_llms_work/sse.py` already provides `format_sse()` and `create_sse_response()` with `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- `backend/src/how_llms_work/ml/bpe.py` already provides the typed public `Merge`, `count_words()`, `train_bpe()`, and `apply_merges()` seams required by Ticket 03. The implementation is deterministic, uses reference-compatible Pre-token boundaries, keeps state local to each call, and enforces the 1,000-merge ceiling.
- `backend/tests/test_bpe.py` already covers repeated Pre-token weighting, deterministic ties, non-overlapping replacement, boundary preservation, minimal inputs, Unicode behavior, merge limits, ordered replay, and mutable-state isolation.
- `backend/tests/test_simple_chat.py` already supplies a strict SSE parser and HTTP regression coverage for the shared media type, headers, framing, request validation, Simple Chat event order, delays, and `GET /health`.
- The TypeScript reference route computes BPE state before creating the stream, emits `init`, waits 800 milliseconds, emits each learned merge without an additional delay, then emits `result` and completes.
- The TypeScript reference initializes `vocab` from submitted characters and updates merge `tokenCount` by subtracting each learned merge's recorded `frequency`. This displayed statistic must be preserved rather than replaced with a newly invented calculation.
- `SPEC.md` and Ticket 03 contain stale evidence notes stating that the BPE/shared modules and backend tests were empty. The latest Python export resolves that conflict: Tickets 01 and 02 are implemented, and only Ticket 03 route/registration/HTTP coverage remains.
- `backend/pyproject.toml` already declares the required runtime and development dependencies and configures pytest, Ruff, and strict mypy. No dependency or lockfile change is required.

## Acceptance criteria coverage

- **Already satisfied and evidenced:** Shared `ChatRequest`; shared SSE formatting, media type, and headers; deterministic reusable BPE operations; public BPE parity tests; no shared mutable BPE state; empty-message validation semantics; whitespace-only acceptance semantics; unchanged Simple Chat and health regression tests; existing Poetry/pytest/Ruff/mypy configuration and dependency set.
- **Behavior present but evidence incomplete:** The shared SSE and request seams are proven through Simple Chat but are not yet exercised by a BPE HTTP route. The reusable BPE interface is proven independently but is not yet orchestrated into frontend-compatible BPE events.
- **Partially implemented:** The destination `routes/bpe_tokenize.py` file exists but contains no route. The FastAPI application exists but does not include a BPE router.
- **Not implemented:** BPE router registration; BPE HTTP 200 response; BPE `init → merge × N → result` stream; exact camelCase feature payloads; 200-character initialization truncation; merge display statistics; final compression formatting; 800-millisecond route delay; BPE HTTP edge-case, isolation, and non-leakage tests.
- **Evidence limitation:** The baseline is user-reported rather than tool-verified here. Backend tests cannot prove browser rendering or Vite proxy behavior, and manual frontend verification remains optional for this ticket.

### Acceptance mapping

| Ticket acceptance area | Current state | Planned coverage |
|---|---|---|
| Router registration while preserving `/health` and `/simple-chat` | Missing BPE registration; regression routes already tested | Steps 3–5 |
| HTTP 200, SSE media type, cache/buffering headers | Shared transport exists; BPE seam missing | Steps 2–4 |
| Valid SSE framing and JSON | Shared formatter exists; BPE seam missing | Steps 2–4 |
| Exact `init → merge × N → result` order and completion | Missing | Steps 2–4 |
| Exact `init` keys, character truncation, complete counts | Missing | Steps 2–4 |
| Exact ordered `merge` keys, numbering, vocabulary and token statistics | Missing | Steps 2–4 |
| Exact `result` keys, token consistency, original count, one-decimal ratio | Missing | Steps 2–4 |
| Empty message returns 422 | Shared model exists; BPE route missing | Steps 2–4 |
| Whitespace, punctuation, and single-character validity | Algorithm exists; BPE route missing | Steps 2–4 |
| No trimming or normalization | Shared model and algorithm preserve input; route missing | Steps 2–4 |
| Per-request Merge Table isolation | Public API test exists; HTTP orchestration missing | Steps 2–4 |
| 800-millisecond production delay and test replacement | Missing | Steps 2–4 |
| No intentional internal-detail serialization | Shared transport does not invent errors; route missing | Steps 2–4 |
| Full pytest, Ruff, strict mypy using existing dependencies | Configured and user-reported passing | Steps 1 and 5 |

## Files to inspect before editing

1. `backend/pyproject.toml` — existing dependencies, pytest discovery, Ruff configuration, and strict mypy settings.
2. `backend/src/how_llms_work/main.py` — `app`, current Simple Chat router import, and router registration order.
3. `backend/src/how_llms_work/routes/bpe_tokenize.py` — empty production destination for the BPE router and stream generator.
4. `backend/src/how_llms_work/schemas.py` — shared `ChatRequest`; confirm it remains unchanged.
5. `backend/src/how_llms_work/sse.py` — `format_sse()` and `create_sse_response()`; confirm the route reuses them unchanged.
6. `backend/src/how_llms_work/ml/bpe.py` — `Merge`, `count_words()`, `train_bpe()`, and `apply_merges()` public seams.
7. `backend/tests/test_bpe.py` — completed public-algorithm parity and state-isolation coverage.
8. `backend/tests/test_simple_chat.py` — existing `parse_sse_events()` pattern and regression coverage for Simple Chat and health.
9. `src/routes/bpe-tokenize.ts` inside `llm_works_file_structure(3).md` — exact event sequence, payload calculations, character truncation, vocabulary tracking, merge token statistic, and Compression Ratio behavior.
10. `src/client/hooks/use-bpe-tokenize-chat.tsx` and `src/client/components/bpe-tokenize-result/index.tsx` inside `llm_works_file_structure(3).md` — frontend event discrimination and exact expected camelCase fields.
11. `SPEC.md`, `CONTEXT.md`, and Ticket 03 — durable behavior, terminology, testing seams, and scope boundaries.

## Step 1 — Reconfirm the pre-edit backend baseline

**Files and symbols:**
- `backend/pyproject.toml` — configured test and quality commands.
- `backend/src/how_llms_work/routes/bpe_tokenize.py` — confirm the route destination is still empty.
- `backend/src/how_llms_work/main.py` — confirm the BPE router is still unregistered.
- `backend/tests/` — confirm no newer user changes supersede this plan.

**Purpose:**
Create implementation-session evidence and separate any pre-existing environment or quality failure from Ticket 03 changes.

**Actions:**
- Work from the `backend` directory.
- Run the complete existing pytest suite before editing.
- Run Ruff and strict mypy before editing.
- Record exact commands, exit codes, and relevant output.
- Confirm the current files still match the latest export before making changes.

**Guardrails:**
- Do not describe the user-reported baseline as tool-verified.
- Do not repair unrelated failures under Ticket 03.
- Do not add or upgrade packages or regenerate `poetry.lock`.

**Expected result:**
- A grounded pre-edit baseline is recorded, or any pre-existing blocker is clearly distinguished from Ticket 03.

**Verification:**

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Step 2 — Add BPE HTTP/SSE acceptance tests through `TestClient`

**Files and symbols:**
- `backend/tests/test_bpe_tokenize.py` — new HTTP/SSE acceptance test module.
- Planned private test helper: `parse_sse_events()` or an equivalently strict local parser.
- Planned tests:
  - `test_bpe_tokenize_streams_reference_compatible_contract`
  - `test_bpe_tokenize_truncates_init_characters_without_changing_counts`
  - `test_bpe_tokenize_rejects_empty_message`
  - `test_bpe_tokenize_accepts_minimal_unmodified_inputs`
  - `test_bpe_tokenize_requests_are_isolated`
  - `test_bpe_tokenize_does_not_expose_internal_failure_details`

**Purpose:**
Encode every missing observable Ticket 03 behavior at the approved FastAPI application seam before production changes.

**Actions:**
- Add a strict SSE parser that requires a terminating blank line, exactly one `event:` line, exactly one `data:` line, a non-empty event name, and valid JSON for every event block.
- Use `TestClient(app)` for successful requests and `TestClient(app, raise_server_exceptions=False)` only for the controlled unexpected-failure case.
- Use a fixed multi-merge input such as `cat cat car` and assert the exact event sequence and exact payloads derived from the current public BPE implementation and TypeScript route contract.
- Assert the successful response status, `text/event-stream` content type, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- Assert exact camelCase key sets for every event type; do not accept extra or snake_case fields.
- Assert the `init` event contains the unchanged submitted text, the expected characters, complete `charCount`, and distinct Pre-token `wordCount`.
- Assert every `merge` event uses one-based `step`, the learned pair order, recorded frequency, `newToken`, reference-compatible `vocabSize`, and the displayed `tokenCount` calculation inherited from the TypeScript route.
- Assert the final `inputTokens`, `tokenCount`, `originalCharCount`, and one-decimal `compressionRatio` are internally consistent and exact for the fixed case.
- Patch only `how_llms_work.routes.bpe_tokenize.asyncio.sleep` with an immediate async mock and assert that production requested exactly `0.8` seconds; do not assert elapsed wall-clock time.
- Add a longer-than-200-character request and assert `characters` is exactly the first 200 submitted characters while `charCount` and `originalCharCount` report the complete Python input length.
- Parameterize whitespace-only, punctuation-only, and single-character requests; assert each produces `init`, no invalid merge event, and a final result without trimming or normalization.
- Send two distinct requests through one client and prove the second response contains no merge or token state learned by the first.
- Induce a controlled pre-stream failure by narrowly replacing the route's referenced BPE training operation; assert HTTP 500 does not contain the exception marker, traceback text, environment data, or an invented SSE error payload.
- Expect the new endpoint tests to fail initially because the route is not registered; this is the intended red evidence.

**Guardrails:**
- Test only through the HTTP response and the existing public BPE seam; do not assert private helper identity, internal containers, or generator structure.
- Do not execute Node, pnpm, Hono, TypeScript, Vite, or browser tests.
- Do not duplicate the full algorithm test matrix already present in `tests/test_bpe.py`.
- Do not weaken the existing strict SSE assertions to accommodate incomplete output.

**Expected result:**
- The new focused test module precisely describes the missing endpoint contract and fails for the expected missing-route reason before implementation.

**Verification:**

```powershell
poetry run pytest tests/test_bpe_tokenize.py -q
```

Expected pre-implementation result:

- Endpoint acceptance tests fail because `POST /bpe-tokenize` is not registered; unrelated existing tests remain untouched.

## Step 3 — Implement the thin BPE route and frontend-compatible stream

**Files and symbols:**
- `backend/src/how_llms_work/routes/bpe_tokenize.py` — `router`, planned stream generator, and `POST /bpe-tokenize` handler.
- Existing public dependencies: `ChatRequest`, `Merge`, `count_words()`, `train_bpe()`, `apply_merges()`, `format_sse()`, and `create_sse_response()`.

**Purpose:**
Add the smallest production boundary that orchestrates the already-complete BPE algorithm into the exact frontend SSE contract.

**Actions:**
- Define one `APIRouter` in the existing route module.
- Import and use the shared `ChatRequest`; do not introduce a route-local request model or additional validation.
- For each request, preserve `request.message` exactly, count Pre-tokens, create the submitted-character list, and train a fresh Merge Table before returning the streaming response.
- Keep all counted Pre-tokens, characters, merges, vocabulary, token statistics, and final tokens local to the request; do not add module-level mutable state, persistence, or caching.
- Return the stream through `create_sse_response()` and format every event through `format_sse()`.
- Emit one `init` event first with exactly `corpus`, `characters`, `charCount`, and `wordCount`.
- Limit `characters` to the first 200 submitted characters while retaining the complete character count.
- After yielding `init`, await `asyncio.sleep(0.8)` through the route module's stable reference so tests can replace it narrowly.
- Initialize the displayed Vocabulary from the submitted characters and initialize the displayed token total from the complete submitted-character count.
- For each learned `Merge` in order, add its merged token to the Vocabulary, subtract its recorded frequency from the displayed token total, and emit exactly `step`, `pair`, `frequency`, `newToken`, `vocabSize`, and `tokenCount` using one-based numbering.
- Apply the complete ordered Merge Table to the original unmodified message and emit exactly one final `result` event with `inputTokens`, `tokenCount`, `originalCharCount`, and a one-decimal multiplier string for `compressionRatio`.
- Allow the async iterator to end immediately after `result`.
- Keep training outside broad exception handling so unexpected pre-stream failures use FastAPI's existing server-error behavior rather than an invented SSE error contract.
- Use modern Python types compatible with strict mypy, including an async string iterator and the existing immutable `Merge` sequence.

**Guardrails:**
- Do not move reusable BPE behavior into the route or HTTP/SSE behavior into `ml/bpe.py`.
- Do not alter `schemas.py`, `sse.py`, or `ml/bpe.py` unless verification proves a concrete Ticket 03 defect; no such defect is visible in the latest source.
- Preserve the TypeScript route's displayed merge `tokenCount` behavior even for overlapping candidate frequencies; do not silently replace it with a different “corrected” statistic.
- Do not add merge delays, a final delay, an SSE error event, normalization, maximum length, disconnect redesign, persistence, caching, or future-phase APIs.
- Do not serialize dataclass objects directly; construct explicit frontend payload dictionaries with exact camelCase keys.

**Expected result:**
- The route module exposes one FastAPI router capable of producing a complete, deterministic, frontend-compatible BPE Event Stream for each valid request.

**Verification:**

```powershell
poetry run python -c "from how_llms_work.routes.bpe_tokenize import router; print(len(router.routes))"
poetry run mypy src/how_llms_work/routes/bpe_tokenize.py
```

Expected result:

- The route module imports successfully, reports one route, and passes strict type checking for the focused file.

## Step 4 — Register the BPE router and turn the HTTP tests green

**Files and symbols:**
- `backend/src/how_llms_work/main.py` — BPE router import and `app.include_router(...)` call.
- `backend/tests/test_bpe_tokenize.py` — focused endpoint acceptance suite.
- `backend/tests/test_simple_chat.py` — existing Simple Chat and health regression suite.

**Purpose:**
Expose the completed route from the application while proving that registration does not disturb the established endpoints.

**Actions:**
- Import the BPE router from `how_llms_work.routes.bpe_tokenize` using the same alias pattern as the Simple Chat router.
- Include the BPE router on the existing `FastAPI` application without changing the health handler or Simple Chat registration.
- Run the focused BPE endpoint tests and correct only Ticket 03 contract mismatches.
- Run the existing Simple Chat/health tests after registration.
- Run the public BPE tests to confirm route integration did not require or cause algorithm changes.

**Guardrails:**
- Do not reorder or redesign unrelated application setup.
- Do not modify the frontend or create an `/api` prefix in FastAPI; Vite owns proxy rewriting outside this ticket.
- Do not weaken existing Simple Chat or BPE assertions to obtain a passing suite.

**Expected result:**
- `POST /bpe-tokenize`, `POST /simple-chat`, and `GET /health` are all available from the same FastAPI application, and all affected focused tests pass.

**Verification:**

```powershell
poetry run pytest tests/test_bpe_tokenize.py tests/test_bpe.py tests/test_simple_chat.py -q
```

Expected result:

- All BPE route, reusable BPE, Simple Chat, and health tests pass without real animation waiting.

## Step 5 — Complete quality, scope, and acceptance verification

**Files and symbols:**
- `backend/src/how_llms_work/routes/bpe_tokenize.py` — final route behavior.
- `backend/src/how_llms_work/main.py` — final registration.
- `backend/tests/test_bpe_tokenize.py` — final endpoint acceptance evidence.
- Entire backend tree — final quality and scope inspection.

**Purpose:**
Prove the complete ticket, ensure no regressions or type/lint problems remain, and prevent unrelated changes from entering the implementation handoff.

**Actions:**
- Run the complete configured pytest suite once after all focused tests pass.
- Run Ruff and strict mypy using the repository commands.
- Inspect the final diff and confirm only the three expected files changed unless a documented acceptance failure required a narrowly justified conditional edit.
- Confirm no dependency, lockfile, frontend, future-phase ML, persistence, cache, worker, or generated-file changes were introduced.
- Perform the manual HTTP/SSE checklist below when the backend can be started locally; do not make browser/Vite verification a blocker.
- Record the exact final command outputs for the implementation completion report and commit body.

**Guardrails:**
- Do not claim a check passed unless its implementation-session output is successful.
- Do not fix unrelated lint, typing, test, or formatting findings as part of Ticket 03.
- Do not create a commit until all required checks pass and the diff is in scope.

**Expected result:**
- The complete backend suite and quality checks pass, the final diff is limited to Ticket 03, and the work is ready for implementation review and commit.

**Verification:**

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Focused verification plan

```powershell
poetry run pytest tests/test_bpe_tokenize.py -q
poetry run pytest tests/test_bpe_tokenize.py tests/test_bpe.py tests/test_simple_chat.py -q
```

Expected result:

- The BPE endpoint tests pass with the route-level sleep replaced.
- Public BPE parity tests remain green.
- Simple Chat and health regression tests remain green.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Expected result:

- All tests pass.
- Ruff reports no violations.
- Strict mypy reports no issues.

## Manual acceptance checklist

- [ ] Start Uvicorn from the backend directory with the repository's existing Poetry command.
- [ ] Confirm `GET http://127.0.0.1:8000/health` still returns `{"status":"healthy"}`.
- [ ] Confirm a deterministic `POST /simple-chat` request still streams `start → word × N → done`.
- [ ] Send `POST /bpe-tokenize` with `{"message":"cat cat car"}` and confirm HTTP 200 plus `text/event-stream`.
- [ ] Confirm the BPE stream has exactly `init → merge → merge → merge → result` for that fixed input.
- [ ] Confirm every SSE block has one `event:` line, one JSON `data:` line, and a terminating blank line.
- [ ] Confirm all BPE payload keys are camelCase and exactly match the frontend contract.
- [ ] Confirm the final tokens are `cat`, space, `cat`, space, `car`; `tokenCount` is 5; `originalCharCount` is 11; and `compressionRatio` is `2.2x`.
- [ ] Confirm a single-character request emits `init` followed by `result` with no `merge` event.
- [ ] Confirm an empty-message request returns HTTP 422 rather than an SSE stream.
- [ ] Confirm no frontend, Node, pnpm, Hono, persistence, caching, or future-phase process is required for backend tests.

## Expected files changed

Likely changed:

```text
backend/src/how_llms_work/routes/bpe_tokenize.py
backend/src/how_llms_work/main.py
backend/tests/test_bpe_tokenize.py
```

Conditionally changed:

```text
None expected.

Only if a focused acceptance test proves a current public-BPE regression not already covered may
backend/tests/test_bpe.py or backend/src/how_llms_work/ml/bpe.py be changed, and that change must be
narrowly documented before proceeding.
```

## Files not to change

```text
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/sse.py
backend/src/how_llms_work/routes/simple_chat.py
backend/tests/test_simple_chat.py
backend/pyproject.toml
backend/poetry.lock
backend/src/how_llms_work/ml/neural_net.py
backend/src/how_llms_work/ml/word2vec.py
backend/src/how_llms_work/ml/transformer.py
backend/src/how_llms_work/ml/transformer_worker.py
frontend/
TypeScript reference source
```

## Risk notes and safeguards

1. **Risk:** The route may emit valid BPE results but wrong frontend field names or extra fields.
   - **Safeguard:** Construct explicit camelCase dictionaries and assert exact key sets at the HTTP seam.
2. **Risk:** A developer may “correct” the merge-step `tokenCount` statistic and diverge from the TypeScript reference, especially for overlapping pair candidates.
   - **Safeguard:** Preserve the reference calculation of subtracting each recorded merge frequency and encode a fixed exact multi-merge contract case.
3. **Risk:** Placing the 800-millisecond sleep before the `init` yield or adding per-merge delays would change animation behavior.
   - **Safeguard:** Yield `init`, then await exactly `0.8`; patch and assert the route-level sleep call without measuring elapsed time.
4. **Risk:** Module-level vocabulary, merge, or token state could leak across requests.
   - **Safeguard:** Allocate every training and display structure inside the request/stream lifecycle and test two sequential requests through one client.
5. **Risk:** Broad exception handling could turn internal exceptions into client-visible stack traces, secrets, or a new SSE error contract.
   - **Safeguard:** Avoid broad catches and verify a controlled pre-stream failure produces a generic HTTP 500 body without the exception marker.
6. **Risk:** Router registration could accidentally remove or alter existing routes.
   - **Safeguard:** Add only the BPE router import/include call and rerun the existing Simple Chat and health tests.
7. **Risk:** Reimplementing BPE logic in the route could drift from the completed Ticket 02 module.
   - **Safeguard:** Route orchestration must call only `count_words()`, `train_bpe()`, and `apply_merges()` for reusable algorithm work.
8. **Risk:** Scope may expand into future tokenizer consumers or frontend changes.
   - **Safeguard:** Keep the final diff to the route, registration, and endpoint tests; prohibit dependencies, `train_bpe_on_text()`, Word2Vec, transformer, Node, and frontend edits.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Suggested outcome:

```text
Stream BPE tokenization through FastAPI
```

Commit body should mention:

- registration of `POST /bpe-tokenize` through the shared request and SSE infrastructure;
- exact `init → merge × N → result` frontend-contract coverage and route-level delay replacement;
- preservation of reusable BPE, Simple Chat, and health behavior;
- the exact focused and full verification commands actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 03, `SPEC.md`, `CONTEXT.md`, `py_llm_pipeline_explorer_file_structure(7).md`, and `llm_works_file_structure(3).md`.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only this work item, verify the complete change, and create the implementation commit.
