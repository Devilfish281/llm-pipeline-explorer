---
workflow: engineering-prompt-chain
document_type: grill_with_docs_result
prompt_name: grill-with-docs-prompt
status: confirmed
version: 1
recommended_next_prompt: to-spec-prompt
---

# Grill With Docs Result: Phase 2 BPE Tokenizer Python Backend

## Original idea

Convert the existing BPE tokenizer behavior from the supplied TypeScript reference implementation into the Python-only FastAPI backend for the `llm-pipeline-explorer` project.

Phase 1—the frontend/server foundation and the Python `/simple-chat` endpoint—is already complete. Phase 2 should create the smallest working Python implementation of the Basic Tokenizer demo by converting the behavior represented by:

```text
src/server/lib/bpe.ts
src/routes/bpe-tokenize.ts
```

into:

```text
backend/src/how_llms_work/ml/bpe.py
backend/src/how_llms_work/routes/bpe_tokenize.py
```

The TypeScript/Vite browser interface must remain unchanged and continue communicating with the backend through HTTP and Server-Sent Events.

## Problem

The Python backend has placeholders for the reusable BPE logic and the `/bpe-tokenize` route, but those files are empty and the BPE router is not registered in the FastAPI application.

The existing frontend already expects a specific request shape, SSE event sequence, camelCase payload structure, and BPE behavior. A Python conversion that changes any of those observable details could break the Basic Tokenizer page even if the underlying algorithm appears generally correct.

The work must remain narrowly focused on producing one functioning BPE learning demo rather than prematurely implementing tokenizer features needed only by later Word2Vec or transformer phases.

## Desired outcome

The Python FastAPI backend accepts a non-empty `message` at `POST /bpe-tokenize`, trains an educational BPE tokenizer on that message in memory, and streams the same meaningful events and payloads expected by the existing frontend:

```text
init → merge × N → result
```

The implementation is registered in FastAPI, reuses shared request and SSE infrastructure, and is covered by self-contained Python tests that do not require Node, pnpm, Hono, or a TypeScript backend.

## Primary users or stakeholders

- The project developer maintaining and extending the Python backend.
- Learners using the Basic Tokenizer page to observe BPE merge steps.
- The existing TypeScript/Vite frontend, which is the direct consumer of the HTTP and SSE contract.
- Future specification and implementation prompts that will use this confirmed result as source material.

## Confirmed scope

- Implement the BPE behavior required by the current `/bpe-tokenize` route.
- Add reusable BPE logic to `backend/src/how_llms_work/ml/bpe.py`.
- Add the FastAPI route to `backend/src/how_llms_work/routes/bpe_tokenize.py`.
- Register the BPE router in `backend/src/how_llms_work/main.py`.
- Move the shared `ChatRequest` Pydantic model into `backend/src/how_llms_work/schemas.py`.
- Update both `/simple-chat` and `/bpe-tokenize` to use the shared `ChatRequest`.
- Move reusable SSE formatting and standard response configuration into `backend/src/how_llms_work/sse.py`.
- Update `/simple-chat` and `/bpe-tokenize` to use the shared SSE utility without changing Simple Chat behavior.
- Preserve strict observable compatibility with the supplied TypeScript reference behavior.
- Add Python-only BPE algorithm tests.
- Add Python-only FastAPI route and SSE contract tests.
- Keep the production BPE animation delay while allowing automated tests to bypass real waiting.
- Train BPE independently in memory for every request.

Expected Phase 2 files:

```text
backend/src/how_llms_work/ml/bpe.py
backend/src/how_llms_work/routes/bpe_tokenize.py
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/sse.py
backend/src/how_llms_work/main.py
backend/src/how_llms_work/routes/simple_chat.py
backend/tests/test_bpe.py
backend/tests/test_bpe_tokenize.py
```

## Out of scope

- `train_bpe_on_text()` or other APIs used only by later phases.
- Custom pre-token regular-expression support.
- Word2Vec-specific tokenization.
- Transformer-specific tokenization.
- Phase 3 XOR neural-network work.
- Phase 4 Word2Vec embedding work.
- Phase 5 transformer work.
- Multiprocessing, worker pools, or shared memory.
- Persisted BPE models or tokenizer files.
- Cross-request BPE caching.
- Frontend changes.
- Input trimming or normalization.
- A new application-level maximum message length.
- Replacing the educational algorithm with a tokenizer library, LangChain, LangGraph, or a hosted AI service.
- Running or retaining a TypeScript backend.
- Executing Node, pnpm, Hono, or TypeScript as part of Python backend tests.
- Exact wall-clock timing assertions for SSE animation delays.
- Unrelated cleanup or redesign.

## Confirmed decisions

1. **Python-only backend:** The current program uses FastAPI/Python as its only backend. The supplied Hono/TypeScript server code is reference material for behavior, not a runtime component to retain or execute.

2. **Strict frontend-contract compatibility:** Preserve the HTTP method, endpoint, request field, validation behavior, SSE names, SSE order, payload shapes, camelCase field names, completion behavior, and production animation behavior expected by the existing frontend.

3. **Strict BPE behavioral compatibility:** Preserve the reference pre-token boundaries, ASCII-style word classification, deterministic pair-selection behavior, non-overlapping merge behavior, ordered merge application, and the current maximum merge count.

4. **Focused vertical slice:** Implement only the reusable BPE behavior necessary to make `/bpe-tokenize` work now. Defer generalized APIs needed only by Word2Vec and transformer phases.

5. **Thin route boundary:** Put reusable tokenization logic in `ml/bpe.py`; keep validation, event construction, streaming, and HTTP response creation in the route and shared transport modules.

6. **Shared request model:** Define one `ChatRequest` Pydantic model in `schemas.py` and import it from both Simple Chat and BPE.

7. **Shared SSE infrastructure:** Define the reusable SSE wire-format and response configuration in `sse.py`. Keep each feature’s event order and delays in its feature stream generator.

8. **Router registration is part of Phase 2:** Include the BPE router in `main.py` so the endpoint is available in the running application.

9. **Python-only parity tests:** Encode representative TypeScript reference behavior as fixed Python test expectations. Do not invoke the reference implementation during normal tests.

10. **No real test delays:** Keep the production delay, but make the sleep operation patchable or injectable so tests run immediately and verify event order rather than elapsed time.

11. **Existing request validation remains unchanged:** Require `message` to have at least one character. Do not strip whitespace, reject whitespace-only strings, normalize text, or add a maximum length in this phase.

12. **Per-request in-memory execution:** Train BPE from the submitted message for every request, stream the result, and discard temporary state afterward.

13. **No ADR:** The confirmed choices are focused, expected, and inexpensive to revisit. None passes all three ADR gates of being hard to reverse, surprising without context, and involving a substantial architectural tradeoff.

## Current behavior verified from files or tools

- The Python project targets Python 3.12 or newer and already declares FastAPI, Pydantic, NumPy, pytest, pytest-asyncio, HTTPX, Ruff, and mypy in `pyproject.toml`.
- `backend/src/how_llms_work/main.py` currently registers only the Simple Chat router and defines `GET /health`.
- `backend/src/how_llms_work/schemas.py`, `backend/src/how_llms_work/sse.py`, `backend/src/how_llms_work/ml/bpe.py`, and `backend/src/how_llms_work/routes/bpe_tokenize.py` are currently empty.
- The existing Python Simple Chat route currently defines `ChatRequest` and SSE formatting locally.
- The TypeScript frontend posts to `/bpe-tokenize` with a JSON body containing `message`.
- The frontend distinguishes BPE event payloads by their fields and expects initialization data, merge-step data, and a final result.
- The reference BPE route sends one `init` event, one `merge` event for each learned merge, and one `result` event.
- The reference `init` payload contains `corpus`, `characters`, `charCount`, and `wordCount`.
- The reference limits the `characters` array in the `init` payload to the first 200 characters while reporting the complete `charCount`.
- Each reference `merge` payload contains `step`, `pair`, `frequency`, `newToken`, `vocabSize`, and `tokenCount`.
- The reference `result` payload contains `inputTokens`, `tokenCount`, `originalCharCount`, and `compressionRatio`.
- The reference request schema accepts a string with at least one character and does not trim it.
- The reference pre-token pattern separates word sequences, individual whitespace characters, and individual punctuation characters so merges remain inside pre-token boundaries.
- The reference merge helper replaces non-overlapping adjacent pair occurrences from left to right.
- The updated root `CONTEXT.md` defines the canonical BPE, compatibility, and migration terminology for this work.

## Desired behavior

- `POST /bpe-tokenize` is available from the FastAPI application.
- A valid request uses this shape:

```json
{
  "message": "the cat"
}
```

- The endpoint returns `text/event-stream`.
- Standard SSE response headers prevent caching and unwanted proxy buffering.
- Each event is encoded as valid SSE text:

```text
event: <event-name>
data: <valid-json>

```

- Events are emitted in this order:

```text
init
merge
merge
...
result
```

- The production `init` event retains the reference animation delay.
- Each learned merge is streamed in order.
- The final result applies the learned merge table in order to the original message.
- Internal Python names may use snake_case, but serialized field names must remain the exact camelCase names consumed by the frontend.
- The route does not persist per-request BPE data.
- The implementation does not require NumPy because this BPE slice is string- and collection-based rather than numerical-array-based.
- The completed Python route works without any TypeScript backend process.

## Domain model

### Terms created or changed

- **LLM Pipeline Explorer:** The interactive educational application that demonstrates major language-model pipeline stages.
- **Learning Demo:** A user-facing experience that exposes a pipeline concept’s inputs, intermediate steps, and result.
- **BPE Tokenizer:** The educational Byte Pair Encoding algorithm that begins with character-level tokens inside pre-token boundaries and repeatedly merges the most frequent adjacent pair.
- **Basic Tokenizer:** The frontend display name of the BPE Tokenizer learning demo, not the canonical name of the reusable algorithm.
- **BPE Training Text:** The user-provided text from which one tokenization run learns pair frequencies and merge operations.
- **Pre-token:** An initial word, whitespace character, or punctuation character that defines a boundary BPE merges may not cross.
- **Token:** A string unit that starts as a character and may grow through merges.
- **BPE Pair:** Two adjacent tokens considered as a merge candidate.
- **BPE Merge:** An operation replacing each non-overlapping occurrence of one adjacent pair with one combined token.
- **Merge Table:** The ordered sequence of learned BPE merges.
- **Vocabulary:** The set of distinct character and merged tokens known during a run.
- **Tokenization Run:** One execution using one BPE Training Text and producing initialization, merge, and result events.
- **Compression Ratio:** Original character count divided by final token count and displayed as a multiplier.
- **Frontend Contract:** The HTTP, validation, SSE, payload, ordering, completion, and error behavior consumed by the browser.
- **Strict TypeScript Compatibility:** The requirement that Python reproduce the observable behavior of the supplied TypeScript reference.
- **BPE Event Stream:** One `init` event, zero or more ordered `merge` events, and one `result` event.
- **Python Backend:** The only current server-side implementation, built with FastAPI under `backend/src/how_llms_work/`.
- **TypeScript Reference Implementation:** Historical server code used only to establish behavior that Python must reproduce.
- **Phase Migration:** A focused conversion that implements and registers the Python route and proves compatibility with Python tests.

### Important relationships

- One Basic Tokenizer user submission creates one Tokenization Run.
- One Tokenization Run uses one BPE Training Text.
- One Tokenization Run learns one ordered Merge Table.
- One Merge Table contains zero or more BPE Merges.
- Each BPE Merge is selected from one BPE Pair.
- One Tokenization Run produces exactly one BPE Event Stream.
- One BPE Event Stream contains exactly one `init` event, zero or more `merge` events, and exactly one `result` event.
- The TypeScript/Vite frontend consumes the Frontend Contract produced by the Python Backend.
- The TypeScript Reference Implementation informs parity expectations but has no runtime relationship with the current application.
- The Basic Tokenizer route depends on the reusable BPE Tokenizer but owns its HTTP and event-stream presentation.

### Domain artifacts

- [CONTEXT.md](CONTEXT.md)

## Architectural decisions

None. No confirmed decision passed all three ADR gates.

## Constraints

- Use Python 3.12 or newer.
- Use Poetry for the Python environment and dependencies.
- Use FastAPI routers and Pydantic request validation.
- Keep reusable BPE logic separate from route logic.
- Use Windows PowerShell commands in later implementation instructions.
- Preserve the existing TypeScript/Vite frontend.
- Preserve `/health` and `/simple-chat`.
- Preserve `/bpe-tokenize` and its current browser contract.
- Keep serialized API fields in camelCase even when internal Python names use snake_case.
- Use only dependencies already declared in `pyproject.toml`; no new dependency is currently justified.
- Keep the BPE implementation educational and visible rather than delegating to an opaque tokenizer package.
- Do not claim tests passed unless they are actually executed successfully during implementation.
- Keep Phase 2 focused on a working vertical slice.
- Do not introduce future-phase abstractions solely for anticipated reuse.

## Edge cases and failure behavior

- **Empty message:** Reject with FastAPI/Pydantic validation and HTTP `422`.
- **Whitespace-only message:** Accept because it has at least one character; do not strip it.
- **Punctuation-only message:** Accept and tokenize according to the reference pre-token boundaries.
- **Repeated words or pre-tokens:** Weight internal pair frequencies by occurrence count, matching the reference behavior.
- **No adjacent pairs remain:** Produce zero additional merge events and still send the final result.
- **Single-character input:** Produce `init` followed by `result`; no merge is required.
- **More than 200 input characters:** Include only the first 200 entries in the `characters` array while preserving the complete `charCount`.
- **Pair-frequency tie:** Use deterministic selection matching the reference implementation’s encounter order.
- **Overlapping pair candidates:** Merge non-overlapping occurrences from left to right.
- **Unicode input:** Apply the confirmed ASCII-style word classification needed for reference parity rather than Python’s default Unicode `\w` behavior.
- **Compression calculation:** Use the reference formatting and return `"N/A"` only when the reference behavior requires it; valid Phase 2 requests are non-empty.
- **Client disconnect:** No additional cancellation design was confirmed for this focused phase; implementation should not add unrelated behavior that changes the contract.
- **Unexpected internal failure:** Do not expose stack traces or environment data. The exact new error-event design is outside the confirmed scope and must not be invented in the specification without further evidence.

## Testing expectations

### BPE algorithm tests

Tests should cover the minimum reusable behavior required by the route:

- Pre-token counting for words, whitespace, punctuation, and repeated values.
- ASCII-style word classification.
- Non-overlapping left-to-right pair merging.
- Pair frequencies weighted by pre-token occurrence counts.
- Deterministic selection when pair frequencies tie.
- Early termination when no adjacent pair remains.
- The current maximum merge count.
- Applying learned merges in their original order.
- Representative fixed outputs derived from the TypeScript Reference Implementation.
- Whitespace-only and punctuation-only inputs.
- A representative Unicode input proving deliberate ASCII-compatibility behavior.

### FastAPI and SSE contract tests

Tests should verify:

- `POST /bpe-tokenize` is registered.
- A normal request returns HTTP `200`.
- The response media type is `text/event-stream`.
- The response includes the expected cache and proxy-buffering headers.
- An empty message returns HTTP `422`.
- A whitespace-only message remains valid.
- Event order is exactly `init → merge × N → result`.
- Every `data:` value is valid JSON.
- The `init` payload uses the exact expected camelCase fields.
- The `characters` array is truncated at 200 while `charCount` reports the full length.
- Each `merge` payload uses the exact expected fields and step numbering.
- The final payload uses the exact expected fields.
- The final `inputTokens` match representative reference outputs.
- `tokenCount` matches the length of `inputTokens`.
- `originalCharCount` matches the submitted message length.
- `compressionRatio` uses the reference one-decimal multiplier format.
- Real animation waiting is bypassed through monkeypatching or an equivalent narrow test seam.
- Simple Chat continues to work after moving `ChatRequest` and SSE utilities.
- Tests run through Poetry and pytest without Node or TypeScript.

### Later implementation validation

The implementation prompt should run, at minimum, the focused and configured checks:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

It should also start Uvicorn and manually confirm the endpoint through the Vite proxy when practical. Test outcomes must be reported honestly.

## Risks and safeguards

- **Risk:** Python regex defaults could classify Unicode word characters differently from JavaScript.
  - **Safeguard:** Explicitly implement the confirmed ASCII-style pre-token behavior and add a parity test.

- **Risk:** Pair-frequency ties could select a different merge and cause all later tokens to diverge.
  - **Safeguard:** Preserve deterministic encounter-order selection and cover ties with a fixed test.

- **Risk:** Internal snake_case naming could leak into the API response and break the frontend.
  - **Safeguard:** Construct explicit camelCase payload dictionaries and assert exact field names.

- **Risk:** Moving shared validation or SSE code could accidentally change the completed Simple Chat route.
  - **Safeguard:** Keep the refactor behavior-preserving and add or retain Simple Chat contract coverage.

- **Risk:** Real animation sleeps could make tests slow and flaky.
  - **Safeguard:** Patch or inject the sleep operation in tests and assert order instead of wall-clock timing.

- **Risk:** Expanding `bpe.py` for future phases could delay the first working result.
  - **Safeguard:** Implement only the functions exercised by `/bpe-tokenize` and defer generalized APIs.

- **Risk:** Large inputs could consume significant CPU because work runs synchronously in the application process.
  - **Safeguard:** Accept this limitation for the focused first slice; evaluate limits or offloading only after measured need.

- **Risk:** Duplicate transport helpers could drift across routes.
  - **Safeguard:** Centralize SSE formatting and standard headers in `sse.py`.

- **Risk:** Tests could accidentally depend on the historical TypeScript backend.
  - **Safeguard:** Store fixed expected values in Python tests and run with Poetry/pytest only.

- **Risk:** Developers could mistake the TypeScript reference files for a current backend.
  - **Safeguard:** Use the canonical terms Python Backend and TypeScript Reference Implementation from `CONTEXT.md`.

## Open questions

- None that block writing the specification.
- Exact implementation signatures, class choices, and test fixtures should be defined by `to-spec-prompt` from the confirmed behavior above.
- No implementation has been performed and no tests have been run as part of this grilling workflow.

## Source material consulted

- `GRILL_WITH_DOCS_PROMPT.md`
- Updated root `CONTEXT.md`
- `py_llm_pipeline_explorer_file_structure.md`
- `llm_works_file_structure.md`
- `backend/pyproject.toml`
- `backend/src/how_llms_work/main.py`
- `backend/src/how_llms_work/routes/simple_chat.py`
- `backend/src/how_llms_work/schemas.py`
- `backend/src/how_llms_work/sse.py`
- `backend/src/how_llms_work/ml/bpe.py`
- `backend/src/how_llms_work/routes/bpe_tokenize.py`
- TypeScript reference `src/server/lib/bpe.ts`
- TypeScript reference `src/routes/bpe-tokenize.ts`
- TypeScript reference `src/schemas/chat-request.ts`
- Frontend `src/client/hooks/use-bpe-tokenize-chat.tsx`
- Frontend `src/client/components/bpe-tokenize-result/index.tsx`
- Frontend `src/client/lib/sse.ts`
- Official FastAPI documentation for `APIRouter`, `include_router()`, streaming responses, and testing.
- Official Pydantic documentation for field constraints.
- Official pytest documentation for `monkeypatch`.

## Recommended next step

Run `to-spec-prompt` using this file, the updated `CONTEXT.md`, and the original supporting files as inputs.

The specification should remain limited to Phase 2 and must not introduce Word2Vec, transformer, multiprocessing, persistence, caching, frontend changes, or other work outside the confirmed scope.

`to-spec-prompt` is not included in the current prompt pack.
