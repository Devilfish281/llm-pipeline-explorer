---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: 001
source_work_item: 01-preserve-simple-chat-through-shared-request-and-sse-infrastructure.md
source_specification: SPEC.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure.md (uploaded as py_llm_pipeline_explorer_file_structure(3).md)
baseline_test_status: not-supplied
recommended_next_prompt: implement-prompt
---

# Plan for Issue 01: Preserve Simple Chat through shared request and SSE infrastructure

## Initial checklist

- Confirm Ticket 01 is the only work item in scope and has no blockers.
- Treat the latest supplied `py_llm_pipeline_explorer_file_structure(3).md` export as current-code authority.
- Record that no pre-planning baseline test result was supplied or tool-verified.
- Preserve the existing `/simple-chat` and `/health` behavior while moving only shared validation and SSE transport concerns.
- Add regression evidence through FastAPI's `TestClient` before changing the working route.
- Finish with `poetry run pytest`, `poetry run ruff check .`, and `poetry run mypy src`.

## Source-of-truth hierarchy

1. The latest explicit project direction: the current server is the Python/FastAPI backend, and Ticket 01 must not add BPE or frontend work.
2. Ticket 01 is the immediate scope, acceptance, test-seam, and out-of-scope authority.
3. `py_llm_pipeline_explorer_file_structure(3).md` is the latest user-identified complete code export and is the source of truth for current implementation details.
4. `SPEC(2).md`, `CONTEXT(6).md`, `backend/README.md`, and `backend/pyproject.toml` provide durable behavior, terminology, commands, and repository conventions.
5. Older snippets, earlier plans, and assumptions are non-authoritative when they conflict with the selected ticket or latest code export.

## Work-item summary

Ticket 01 is a behavior-preserving infrastructure refactor. Move the existing `ChatRequest` validation contract from the Simple Chat route into the shared schema module, move the existing SSE formatting and response configuration into the shared SSE module, and update `POST /simple-chat` to use those shared seams. Establish the first HTTP-level regression tests for Simple Chat and health through FastAPI's `TestClient`. Do not implement BPE, modify the frontend, add dependencies, or change any observable Simple Chat behavior.

## Baseline evidence

- **Status:** Not supplied
- **Command:** `Not supplied`
- **Result:** No user-reported or tool-verified pre-planning baseline exists. The latest snapshot shows an empty `backend/tests/` directory, so the implementation run may initially report that no tests were collected.
- **Planning rule:** The implementation run must establish or reconfirm its own baseline before editing and record the exact result without treating an empty baseline suite as proof that application behavior is correct.

## Current code observations from the latest source

- `backend/src/how_llms_work/schemas.py` is empty; no shared `ChatRequest` currently exists.
- `backend/src/how_llms_work/sse.py` is empty; no shared SSE formatter or response factory currently exists.
- `backend/src/how_llms_work/routes/simple_chat.py` locally defines `ChatRequest` with `message: str = Field(min_length=1)`. No trimming, normalization, maximum length, or whitespace rejection is configured.
- `backend/src/how_llms_work/routes/simple_chat.py` locally defines `format_sse()`, using `json.dumps()` and returning one `event:` line, one `data:` line, and a terminating blank line.
- `stream_chat()` emits `start`, waits 1 second, emits one `word` event per response word with a 0.2-second wait after each, and emits `done` last.
- A greeting input such as `hello` deterministically produces `Hello! How can I help you today?`; the fallback branch uses `random.choice()`, so exact-payload regression tests should use a deterministic input.
- `simple_chat()` directly constructs `StreamingResponse` with `media_type="text/event-stream"`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- `backend/src/how_llms_work/main.py` includes the Simple Chat router and defines `GET /health`, returning `{"status": "healthy"}`.
- `backend/tests/` contains no visible tests in the supplied snapshot.
- `backend/pyproject.toml` already declares FastAPI, Pydantic, pytest, HTTPX, Ruff, and mypy, configures pytest to use `tests`, configures Ruff for Python 3.12, and enables strict mypy checking with `src` on the import path.
- The present runtime behavior is largely already correct; the missing work is shared ownership and durable regression evidence, not a rewrite of chat logic.

## Acceptance criteria coverage

- **Already satisfied and evidenced:** The current route locally requires a non-empty string, leaves whitespace untrimmed, formats named SSE events, uses the required media type and headers, emits `start → word × N → done`, preserves the 1-second and 0.2-second delays, and leaves `GET /health` unchanged.
- **Behavior present but evidence incomplete:** Empty-message `422`, whitespace-only acceptance, no application-level maximum length, exact SSE framing and JSON payloads, response headers, event order, and health behavior have current-code support but no automated regression tests in the supplied snapshot.
- **Partially implemented:** Shared request validation and shared SSE transport exist only as route-local implementations; their intended destination modules are empty.
- **Not implemented:** `POST /simple-chat` does not import shared `ChatRequest` or shared SSE utilities, and no `TestClient` regression suite covers Ticket 01.
- **Evidence limitation:** No baseline commands were run during planning, and the snapshot cannot prove runtime test results or exact installed-package behavior on the user's machine.

## Files to inspect before editing

1. `backend/pyproject.toml` — `[tool.pytest.ini_options]`, `[tool.ruff]`, `[tool.mypy]`, and declared test/runtime dependencies.
2. `backend/src/how_llms_work/main.py` — `app`, `simple_chat_router`, and `health`; verify these remain unchanged.
3. `backend/src/how_llms_work/routes/simple_chat.py` — `ChatRequest`, `format_sse`, `get_simple_chat_response`, `stream_chat`, and `simple_chat`.
4. `backend/src/how_llms_work/schemas.py` — empty shared schema destination.
5. `backend/src/how_llms_work/sse.py` — empty shared SSE transport destination.
6. `backend/tests/` — confirm it remains empty or identify any user changes made after the supplied export before selecting the test filename and helpers.

## Step 1 — Establish and record the pre-edit backend baseline

**Files and symbols:**
- `backend/pyproject.toml` — pytest, Ruff, and mypy configuration.
- `backend/src/how_llms_work/main.py` — `app` and `health`.
- `backend/src/how_llms_work/routes/simple_chat.py` — `router`, `stream_chat`, and `simple_chat`.

**Purpose:**
Establish the implementation session's own evidence before editing, as required because no baseline result was supplied. This protects against attributing pre-existing environment, import, lint, or typing failures to Ticket 01.

**Actions:**
- Work from the `backend` directory and run `poetry install` only when the Poetry environment is not already synchronized.
- Run the current configured pytest command and record its exact exit status and output; an empty suite may report no tests collected.
- Run the configured Ruff and strict mypy checks and record all pre-existing findings.
- Confirm that importing `how_llms_work.main:app` succeeds before adding tests.
- Do not modify any file during this baseline step.

**Guardrails:**
- Do not describe a command as passing unless the current implementation session receives a successful result.
- Do not fix unrelated baseline findings as part of Ticket 01.
- Do not add dependencies or regenerate `poetry.lock` merely to perform the refactor.

**Expected result:**
- A grounded pre-edit record exists for pytest, Ruff, mypy, and application import behavior.
- Any no-tests-collected condition or pre-existing failure is clearly separated from later Ticket 01 results.

**Verification:**

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
poetry run python -c "from how_llms_work.main import app; print(app.title)"
```

## Step 2 — Add HTTP-level regression evidence for the existing Simple Chat contract

**Files and symbols:**
- `backend/tests/test_simple_chat.py` — new `TestClient` regression tests and a test-local SSE parsing helper.
- `backend/src/how_llms_work/main.py` — `app` used only through the public HTTP test seam.
- `backend/src/how_llms_work/routes/simple_chat.py` — route-level sleep reference patched only to bypass real waiting while exercising HTTP behavior.

**Purpose:**
Protect every currently observable Ticket 01 behavior before moving ownership. These are evidence-alignment tests: the current implementation already appears to provide the expected behavior, so the focused tests should pass immediately unless the source snapshot or runtime differs.

**Actions:**
- Create one focused test module using `fastapi.testclient.TestClient` with the application from `how_llms_work.main`.
- Add a small test-local SSE parser that splits complete event blocks, verifies the `event:` and `data:` lines, rejects malformed blocks, and JSON-decodes every data value.
- Patch the sleep reference used by the Simple Chat route with an immediate async replacement so tests remain fast; capture requested delay values rather than measuring wall-clock time.
- Add a deterministic `hello` request test that asserts HTTP `200`, a `text/event-stream` content type, both required response headers, exact event order, exact semantic payloads, valid SSE framing, and the unchanged requested delay sequence.
- Assert the deterministic event sequence is `start`, seven `word` events for `Hello! How can I help you today?`, then `done`; assert `start` and `done` payloads are empty objects and each word payload contains the established `word` field.
- Add an empty-string request test asserting the standard HTTP `422` response.
- Add a whitespace-only request test asserting HTTP `200`, valid SSE data, `start` first, and `done` last without requiring one random fallback sentence.
- Add a long single-token request test that is substantially longer than ordinary input and still returns HTTP `200`, proving no application-level maximum message length was introduced.
- Add a health regression test asserting HTTP `200` and exactly `{"status": "healthy"}`.

**Guardrails:**
- Exercise behavior through `TestClient`; do not test private helper identity, local variable names, or generator implementation details.
- Do not assert exact elapsed time.
- Use the deterministic greeting path for exact word payload assertions; do not make tests depend on the random fallback selection.
- Keep parsing support local to this test file unless another existing test proves a shared fixture is needed.
- Do not change production code merely to manufacture a red test when the current behavior is already correct.

**Expected result:**
- The first backend regression suite proves the existing `/simple-chat` Frontend Contract and `/health` response before the refactor.
- Tests run without real 1-second or 0.2-second waits.

**Verification:**

```powershell
poetry run pytest tests/test_simple_chat.py -q
```

## Step 3 — Move `ChatRequest` to the shared schema module

**Files and symbols:**
- `backend/src/how_llms_work/schemas.py` — new public `ChatRequest` model.
- `backend/src/how_llms_work/routes/simple_chat.py` — remove the local `ChatRequest` definition and import the shared model.

**Purpose:**
Satisfy the shared request-validation criteria without changing the JSON body, validation behavior, or Simple Chat response behavior.

**Actions:**
- Define the single shared `ChatRequest` model in `schemas.py` using the current `message` string constraint of minimum length one.
- Do not add validators, configuration, whitespace stripping, normalization, coercion policy changes, or a maximum length.
- Update the Simple Chat route to import `ChatRequest` from `how_llms_work.schemas`.
- Remove only the now-duplicated local Pydantic model and its unused Pydantic imports from `simple_chat.py`.
- Run the focused tests immediately after this move.

**Guardrails:**
- Preserve the request field name `message` and standard FastAPI/Pydantic error handling.
- Whitespace-only values must remain valid.
- Do not add BPE imports or edit `bpe_tokenize.py` in this ticket.
- Do not create aliases or additional request models.

**Expected result:**
- `ChatRequest` has one owner in `schemas.py`.
- `POST /simple-chat` consumes that shared model with unchanged HTTP validation behavior.

**Verification:**

```powershell
poetry run pytest tests/test_simple_chat.py -q
poetry run ruff check src/how_llms_work/schemas.py src/how_llms_work/routes/simple_chat.py
poetry run mypy src
```

## Step 4 — Move SSE formatting and response configuration to the shared transport module

**Files and symbols:**
- `backend/src/how_llms_work/sse.py` — shared `format_sse` formatter and one shared SSE response-construction seam.
- `backend/src/how_llms_work/routes/simple_chat.py` — `stream_chat` and `simple_chat` updated to use the shared transport functions.

**Purpose:**
Centralize the established SSE wire format, media type, and headers while leaving feature-specific message selection, event sequence, payloads, and animation delays in the Simple Chat route.

**Actions:**
- Move the current `format_sse()` behavior into `sse.py` without changing JSON serialization semantics or the `event:`/`data:`/blank-line framing.
- Add one narrowly typed shared response-construction function in `sse.py` that accepts the route's stream iterable and returns `StreamingResponse` configured with `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- Update `stream_chat()` to call the shared formatter while preserving the exact `start → word × N → done` order and existing payload dictionaries.
- Update `simple_chat()` to call the shared response-construction seam instead of duplicating media type and headers.
- Remove only imports and route-local transport code made obsolete by the move.
- Keep `get_simple_chat_response()` and its current behavior unchanged.
- Run the focused regression tests, then inspect the diff to confirm that the production sleep calls still request 1 second before words and 0.2 seconds after each word.

**Guardrails:**
- Do not move feature-specific event sequencing or delay logic into `sse.py`.
- Do not rename Simple Chat events or payload keys.
- Do not introduce an SSE error-event contract, exception wrapper, buffering redesign, or disconnect-handling change.
- Do not edit `main.py`; the existing router registration and health endpoint require no production change for Ticket 01.
- Do not add dependencies, modify `pyproject.toml`, or update `poetry.lock`.

**Expected result:**
- Shared SSE formatting and response configuration have one owner in `sse.py`.
- Simple Chat uses the shared transport seam while remaining observably identical through the HTTP boundary.
- The focused Ticket 01 tests pass after both ownership moves.

**Verification:**

```powershell
poetry run pytest tests/test_simple_chat.py -q
poetry run ruff check src/how_llms_work/sse.py src/how_llms_work/routes/simple_chat.py tests/test_simple_chat.py
poetry run mypy src
```

## Focused verification plan

```powershell
poetry run pytest tests/test_simple_chat.py -q
```

Expected result:

- All Ticket 01 tests pass.
- The deterministic greeting request emits `start`, the seven expected `word` events, and `done` with unchanged payloads.
- Empty input returns `422`; whitespace-only and long non-empty input remain valid.
- The response uses SSE media type and the required cache and buffering headers.
- `GET /health` remains exactly healthy.
- Tests bypass real waiting while confirming the route still requests the established production delays.

## Affected quality-check plan

```powershell
poetry run ruff check .
poetry run mypy src
```

Expected result:

- Ruff reports no violations introduced by Ticket 01.
- Strict mypy checking succeeds for the shared schema, shared SSE seam, and updated route.

## Full verification plan

```powershell
poetry run pytest
```

Expected result:

- All tests pass.

## Manual acceptance checklist

- [ ] `POST /simple-chat` still accepts `{"message":"hello"}` and returns an SSE stream.
- [ ] The stream begins with `start`, emits `word` events containing `Hello! How can I help you today?`, and ends with `done`.
- [ ] Each event has one `event:` line, one JSON `data:` line, and a terminating blank line.
- [ ] The response content type is `text/event-stream` and includes `Cache-Control: no-cache` and `X-Accel-Buffering: no`.
- [ ] The production route still requests a 1-second initial delay and a 0.2-second delay after each word; no wall-clock timing assertion is required.
- [ ] `{"message":""}` returns HTTP `422` through standard FastAPI/Pydantic validation.
- [ ] A whitespace-only non-empty `message` returns HTTP `200` and a complete Simple Chat stream.
- [ ] A long non-empty message is not rejected by a newly introduced application-level maximum.
- [ ] `GET /health` still returns HTTP `200` with `{"status":"healthy"}`.
- [ ] No BPE, frontend, dependency, persistence, caching, multiprocessing, or SSE error-contract work appears in the final diff.

## Expected files changed

Likely changed:

```text
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/sse.py
backend/src/how_llms_work/routes/simple_chat.py
backend/tests/test_simple_chat.py
```

Conditionally changed:

```text
None. Create backend/tests/conftest.py only if the current repository, when re-inspected, already uses it for shared TestClient fixtures; the supplied snapshot does not justify adding it.
```

## Files not to change

```text
backend/src/how_llms_work/main.py
backend/src/how_llms_work/ml/bpe.py
backend/src/how_llms_work/routes/bpe_tokenize.py
backend/src/how_llms_work/routes/neural_net.py
backend/src/how_llms_work/routes/train_embed.py
backend/src/how_llms_work/routes/train_transformer.py
backend/pyproject.toml
backend/poetry.lock
backend/README.md
frontend/
```

## Risk notes and safeguards

1. **Risk:** Moving the model could accidentally introduce trimming, normalization, or a maximum length.
   - **Safeguard:** Copy only the existing minimum-length constraint and prove empty, whitespace-only, and long input behavior through HTTP tests.
2. **Risk:** The shared response helper could alter media type or headers.
   - **Safeguard:** Assert the response content type and both established headers through `TestClient` before and after the move.
3. **Risk:** SSE extraction could change event framing or JSON serialization.
   - **Safeguard:** Preserve the current formatter semantics and parse every complete event block in the regression suite.
4. **Risk:** The refactor could move feature-specific sequencing or delays into generic infrastructure.
   - **Safeguard:** Keep `stream_chat()` responsible for `start → word × N → done` and its sleep calls; shared code owns transport only.
5. **Risk:** Exact response assertions could become flaky because fallback chat responses are random.
   - **Safeguard:** Use the deterministic `hello` path for exact payload assertions and use contract-only assertions for whitespace and long-input cases.
6. **Risk:** Delay testing could make the suite slow or accidentally patch unrelated asynchronous behavior.
   - **Safeguard:** Patch the narrow route-used sleep reference, restore it automatically through pytest, assert requested values rather than elapsed time, and keep the patch scoped to each test.
7. **Risk:** Scope may drift into the BPE endpoint because these seams are intended for later reuse.
   - **Safeguard:** Do not edit BPE modules or register routes; Ticket 01 ends after Simple Chat adopts the shared seams and its behavior is proven.
8. **Risk:** An empty pre-existing test suite may be misreported as a passing baseline.
   - **Safeguard:** Record the exact pytest exit code and output; distinguish no tests collected from a successful test run.

## Commit guidance after tests pass

```text
Use the repository's established outcome-oriented convention.
```

Commit body should mention:

- the shared `ChatRequest` and SSE transport ownership move;
- preservation of the Simple Chat and health HTTP contracts;
- the focused and full pytest commands, Ruff check, and strict mypy check actually executed.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using this plan, Ticket 01, `SPEC.md`, `CONTEXT.md`, and the current repository.

`implement-prompt` must inspect the repository again, establish its own baseline, preserve user changes, implement only Ticket 01, verify the complete change, and create the implementation commit.
