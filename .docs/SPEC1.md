---
workflow: engineering-prompt-chain
document_type: specification
prompt_name: to-spec-prompt
status: ready-for-agent
triage_label: ready-for-agent
version: 1
source_document: GRILL_WITH_DOCS_RESULT.md
recommended_next_prompt: to-tickets-prompt
---

# Specification: Deliver the Phase 2 BPE Tokenizer Through the Python Backend

## Problem

Learners can use the completed Simple Chat demonstration, but the Basic Tokenizer learning demo does not yet have a working Python backend. The Python Backend contains empty BPE and route modules, and the BPE router is not registered with the FastAPI application.

The existing TypeScript/Vite frontend already depends on a precise Frontend Contract for `POST /bpe-tokenize`: a non-empty `message`, a Server-Sent Events response, an ordered sequence of initialization, merge, and result events, and exact camelCase payload fields. A Python implementation that is merely similar—but not observably compatible—can cause the browser experience or displayed BPE steps to diverge.

The project developer also needs the implementation to remain small, educational, and independently testable with the existing Poetry/Python toolchain. Introducing future Word2Vec or transformer abstractions, persistent models, multiprocessing, or a TypeScript backend would delay the first working result and expand Phase 2 beyond its confirmed purpose.

## Solution

Implement one focused Phase Migration that makes the Basic Tokenizer learning demo work through the existing Python Backend while leaving the TypeScript/Vite frontend unchanged.

For each valid request, the Python Backend will train the educational BPE Tokenizer in memory from the submitted BPE Training Text, emit one `init` event, emit each learned `merge` event in order, and finish with one `result` event. The implementation will preserve Strict TypeScript Compatibility for pre-token boundaries, ASCII-style word classification, deterministic pair selection, non-overlapping merge behavior, ordered merge application, merge limits, validation, payload fields, SSE framing, and production animation behavior.

Reusable tokenization behavior will be owned by the BPE module. HTTP validation, event construction, streaming, and response creation will be owned by the route and shared transport modules. The shared `ChatRequest` model and shared SSE formatting will also be used by Simple Chat without changing its observable behavior.

The completed vertical slice will be registered in FastAPI and verified through self-contained Python tests. Tests will exercise the full HTTP/SSE boundary and the public reusable BPE interface, while avoiding private implementation details, real animation waits, Node, pnpm, Hono, and TypeScript execution.

## User Stories

1. As a learner, I want to submit non-empty text to the Basic Tokenizer demo, so that I can observe how BPE learns tokens from my text.
2. As a learner, I want the demo to begin with initialization information, so that I can understand the original training text before merges occur.
3. As a learner, I want to see every learned BPE Merge in order, so that I can follow how larger tokens are formed.
4. As a learner, I want to receive a final tokenization result, so that I can compare the original text with the learned tokens.
5. As a learner, I want a single-character input to complete successfully without a merge event, so that valid minimal inputs still produce a result.
6. As a learner, I want punctuation-only input to remain valid, so that punctuation behavior is visible rather than rejected.
7. As a learner, I want whitespace-only input to remain valid, so that the demo preserves the established request behavior.
8. As a learner, I want repeated words or pre-tokens to influence pair frequencies by their occurrence counts, so that the displayed learning process reflects the full submitted text.
9. As a learner, I want overlapping pair candidates to be merged non-overlappingly from left to right, so that merge results are predictable.
10. As a learner, I want pair-frequency ties to resolve deterministically, so that the same input produces the same sequence every time.
11. As a learner, I want BPE Merges to remain within Pre-token boundaries, so that words, individual whitespace characters, and individual punctuation characters do not merge across their original boundaries.
12. As a learner, I want the final tokenization to apply the learned Merge Table in its learned order, so that the result matches the demonstrated training process.
13. As a learner, I want a run with no remaining adjacent pair to finish with a result, so that early termination is handled cleanly.
14. As a learner, I want Unicode text to follow the established ASCII-style classification behavior, so that the Python conversion does not silently change the current demonstration.
15. As a learner, I want long input initialization data to show at most the first 200 characters while retaining the complete character count, so that the stream remains manageable without reporting an incorrect total.
16. As a learner, I want the final token count to equal the number of returned input tokens, so that the displayed metrics are internally consistent.
17. As a learner, I want the original character count to equal the submitted message length, so that the compression calculation is understandable.
18. As a learner, I want the Compression Ratio to use the established one-decimal multiplier format, so that the browser displays it consistently.
19. As the TypeScript/Vite frontend, I want to send `POST /bpe-tokenize` with a JSON `message` field, so that the existing client code works without modification.
20. As the TypeScript/Vite frontend, I want an HTTP `200` response with the `text/event-stream` media type for a valid request, so that the existing SSE reader can consume it.
21. As the TypeScript/Vite frontend, I want events in the exact order `init`, zero or more `merge` events, then `result`, so that the Basic Tokenizer display updates correctly.
22. As the TypeScript/Vite frontend, I want every event to use valid SSE framing and JSON data, so that events can be parsed reliably.
23. As the TypeScript/Vite frontend, I want the `init` payload to contain `corpus`, `characters`, `charCount`, and `wordCount`, so that the existing initialization display continues to work.
24. As the TypeScript/Vite frontend, I want each `merge` payload to contain `step`, `pair`, `frequency`, `newToken`, `vocabSize`, and `tokenCount`, so that each merge step can be rendered without adaptation.
25. As the TypeScript/Vite frontend, I want the `result` payload to contain `inputTokens`, `tokenCount`, `originalCharCount`, and `compressionRatio`, so that the final display remains compatible.
26. As the TypeScript/Vite frontend, I want serialized API field names to remain camelCase, so that internal Python naming choices do not leak into the Frontend Contract.
27. As the TypeScript/Vite frontend, I want cache prevention and proxy-buffering headers on the SSE response, so that updates are delivered progressively rather than cached or buffered.
28. As the TypeScript/Vite frontend, I want the production initialization animation delay preserved, so that the learning-demo pacing does not change during migration.
29. As an API client, I want an empty `message` to return the existing FastAPI/Pydantic HTTP `422` validation response, so that invalid input handling remains consistent.
30. As an API client, I want submitted text left untrimmed and unnormalized, so that the backend processes exactly what I sent.
31. As an API client, I want separate requests to train independently, so that one Tokenization Run cannot alter another.
32. As a project developer, I want the BPE route registered in the FastAPI application, so that the endpoint is available when Uvicorn starts.
33. As a project developer, I want reusable BPE behavior separated from HTTP and SSE presentation, so that the algorithm can be tested without a network boundary.
34. As a project developer, I want one shared `ChatRequest` validation model for Simple Chat and BPE, so that their common request constraint cannot drift.
35. As a project developer, I want one shared SSE wire-format and response configuration, so that streaming routes do not duplicate transport details.
36. As a project developer, I want Simple Chat to preserve its current request, event order, payloads, headers, and response behavior after the shared-code refactor, so that completed Phase 1 functionality does not regress.
37. As a project developer, I want the BPE implementation to use only the dependencies already declared for the Python Backend, so that Phase 2 does not introduce unnecessary packages.
38. As a project developer, I want the educational BPE behavior implemented directly rather than delegated to an opaque tokenizer library or hosted service, so that learners can inspect and understand the algorithm.
39. As a project developer, I want the current maximum merge count enforced, so that behavior and resource use remain compatible with the reference.
40. As a project developer, I want automated tests to bypass real animation waiting, so that the suite remains fast and deterministic.
41. As a project developer, I want parity expectations encoded as fixed Python test cases, so that backend testing does not require the historical TypeScript Reference Implementation.
42. As a project developer, I want backend tests to run through Poetry and pytest without Node, pnpm, Hono, or TypeScript, so that the Python Backend remains independently testable.
43. As a project developer, I want representative failures to identify whether the HTTP contract or reusable algorithm diverged, so that regressions can be diagnosed efficiently.
44. As an operator, I want `GET /health` and `POST /simple-chat` to remain available, so that Phase 2 does not disrupt existing backend functionality.
45. As an operator, I want temporary BPE state discarded after each request, so that no tokenizer model, cache, or cross-request state requires maintenance.
46. As an operator, I want unexpected failures to avoid exposing stack traces or environment data to clients, so that internal details are not leaked.
47. As an operator, I want the project’s configured pytest, Ruff, and mypy checks to remain the validation path, so that Phase 2 follows the existing backend quality workflow.

## Implementation Decisions

1. **Confirmed — Python Backend authority:** FastAPI and Python remain the only server-side runtime. The TypeScript Reference Implementation is behavioral evidence only and will not be retained or executed as a backend dependency.

2. **Confirmed — Focused Phase Migration:** Phase 2 will implement only the reusable BPE behavior required by the current Basic Tokenizer route. APIs intended only for Word2Vec, transformer training, custom pre-token patterns, or later phases will not be added.

3. **Confirmed — BPE module ownership:** The reusable BPE module will own pre-token counting, pair-frequency analysis, deterministic merge selection, non-overlapping token merging, training the ordered Merge Table, and applying that table to input text. It will not own HTTP validation, SSE payload construction, animation timing, or response headers.

4. **Confirmed — Public reusable BPE interface:** The reusable module will expose typed public operations equivalent to `count_words()`, `train_bpe()`, and `apply_merges()`. A merge data type will carry the selected pair, occurrence frequency, and combined token needed by the route. The non-overlapping sequence-rewrite operation may remain private because it is an internal mechanism rather than an external test seam.

5. **Confirmed — Pre-token behavior:** Pre-tokenization will separate ASCII-style word sequences, individual whitespace characters, and individual punctuation characters. BPE pair discovery and merging will not cross a Pre-token boundary.

6. **Confirmed — ASCII compatibility:** Python’s default Unicode word classification will not be used where it would diverge from the TypeScript Reference Implementation. The BPE Tokenizer will deliberately reproduce the confirmed ASCII-style classification behavior.

7. **Confirmed — Frequency weighting:** Repeated Pre-tokens will be represented with occurrence counts, and adjacent-pair frequencies will be weighted by those counts during training.

8. **Confirmed — Deterministic pair selection:** When multiple BPE Pairs have the same highest frequency, the first pair encountered in the reference-compatible traversal order will be selected. An unordered container must not determine the winner.

9. **Confirmed — Non-overlapping merge application:** Each selected pair will replace non-overlapping adjacent occurrences from left to right within each Pre-token.

10. **Confirmed — Ordered Merge Table:** Learned BPE Merges will be recorded in selection order. Final tokenization will apply the Merge Table to the original BPE Training Text in that same order.

11. **Confirmed — Maximum merge count:** Training will stop after at most 1,000 learned merges, or earlier when no adjacent pair remains.

12. **Confirmed — Shared request model:** One Pydantic `ChatRequest` model will define `message` as a string with a minimum length of one. Both Simple Chat and BPE will import and use this model. The model will not trim, normalize, reject whitespace-only input, or add a maximum length.

13. **Confirmed — Thin BPE route boundary:** The BPE route will own request handling, construction of frontend-compatible event payloads, event sequencing, production delays, and the streaming response. It will delegate reusable tokenization behavior to the BPE module.

14. **Confirmed — HTTP contract:** The endpoint remains `POST /bpe-tokenize` and accepts a JSON body with the single required `message` field. Empty strings are rejected through Pydantic validation with HTTP `422`; whitespace-only strings remain valid.

15. **Confirmed — BPE Event Stream:** A successful Tokenization Run emits exactly one `init` event, zero or more ordered `merge` events, and exactly one `result` event. The stream completes after the result event.

16. **Confirmed — SSE wire format:** Each event will be serialized as an SSE `event:` line followed by a `data:` line containing valid JSON and a terminating blank line.

17. **Confirmed — Shared SSE infrastructure:** A shared SSE transport module will own event formatting, `text/event-stream` response configuration, and standard headers including `Cache-Control: no-cache` and `X-Accel-Buffering: no`. Feature-specific event order and delays remain in each route’s stream generator.

18. **Confirmed — Initialization payload:** The `init` payload will use the exact camelCase fields `corpus`, `characters`, `charCount`, and `wordCount`. `characters` contains no more than the first 200 characters, while `charCount` reports the complete submitted length.

19. **Confirmed — Merge payload:** Every `merge` payload will use the exact fields `step`, `pair`, `frequency`, `newToken`, `vocabSize`, and `tokenCount`. Steps will be emitted in learned order with reference-compatible numbering.

20. **Confirmed — Result payload:** The `result` payload will use the exact fields `inputTokens`, `tokenCount`, `originalCharCount`, and `compressionRatio`. `tokenCount` equals the length of `inputTokens`; `originalCharCount` equals the submitted message length.

21. **Confirmed — Compression formatting:** Compression Ratio will preserve the reference-compatible one-decimal multiplier representation. The route will not invent a different numeric or JSON representation.

22. **Confirmed — Production animation behavior:** The production stream retains the reference initialization delay of 800 milliseconds. Tests will replace the referenced sleep operation with an immediate async substitute and will verify event order rather than elapsed time.

23. **Confirmed — Per-request lifecycle:** Every request trains a new Merge Table from its own BPE Training Text in application memory. Temporary training data is discarded after the stream completes. There is no persistence, caching, shared tokenizer state, multiprocessing, worker pool, or shared memory in this phase.

24. **Confirmed — Router registration:** The BPE router will be included in the main FastAPI application while preserving the existing health and Simple Chat routes.

25. **Confirmed — Behavior-preserving Simple Chat refactor:** Moving `ChatRequest` and SSE transport behavior out of the Simple Chat route must not change its request validation, `start → word × N → done` event sequence, payloads, delays, media type, or headers.

26. **Confirmed — Existing dependency set:** The implementation will use the packages already declared in the backend project. NumPy is not required for this string-and-collection BPE slice, and no new dependency is justified.

27. **Confirmed — Error behavior:** Validation errors remain standard FastAPI/Pydantic responses. No new SSE error-event contract will be invented. Unexpected failures must not intentionally serialize stack traces, secrets, or environment data into client payloads.

28. **Confirmed — No ADR:** This focused conversion does not create an ADR because the confirmed choices are expected, inexpensive to reverse, and do not represent a substantial architectural tradeoff.

29. **Assumption — Concrete Python type shapes:** The implementation may choose idiomatic immutable or read-only typed structures for BPE Pairs, Merge records, token sequences, and counted Pre-tokens, provided their behavior satisfies this specification and does not leak incompatible serialization into the route.

30. **Assumption — Test-only delay replacement:** The stream generator will reference the sleep operation through a stable module boundary that pytest can replace narrowly. The exact injection style is left to implementation as long as production behavior is unchanged.

31. **Assumption — Internal exception handling:** In the absence of a confirmed application-specific error-event contract, unexpected pre-stream failures will use FastAPI’s existing server-error handling, and implementation will avoid broad exception blocks that transform errors into an invented frontend payload.

## Testing Decisions

- **Approved test seam 1:** Exercise `POST /bpe-tokenize` through FastAPI’s `TestClient`.
- **Why this seam:** It is the highest practical existing boundary used by the frontend and verifies router registration, Pydantic validation, shared schema use, route orchestration, SSE formatting, headers, event order, and exact serialized payloads together.
- **Observable behavior covered:** Successful requests, HTTP status, media type, cache and buffering headers, empty-message validation, whitespace-only input, valid SSE framing, valid JSON data, `init → merge × N → result`, exact camelCase fields, 200-character initialization truncation, merge step numbering, final token consistency, reference-compatible compression formatting, and unchanged Simple Chat behavior.
- **Modules exercised:** FastAPI application, BPE route, shared schemas, shared SSE transport, reusable BPE module, and Simple Chat route regression path.
- **Approved test seam 2:** Exercise the public reusable BPE interface through its counting, training, and merge-application operations.
- **Why this seam:** The HTTP seam alone would make deterministic algorithm divergence difficult to isolate. The public BPE boundary provides fast, precise parity tests while remaining stable under internal refactoring.
- **Observable behavior covered:** Pre-token counting, ASCII-style classification, occurrence-weighted pair frequencies, deterministic tie resolution, left-to-right non-overlapping replacement, early termination, the 1,000-merge cap, ordered Merge Table application, whitespace-only input, punctuation-only input, single-character input, repeated Pre-tokens, representative Unicode behavior, and fixed reference-compatible token results.
- **Modules exercised:** Reusable BPE module only.
- **Prior art:** The supplied Simple Chat route already uses an async stream generator, named SSE events, JSON data, `StreamingResponse`, and cache/buffering headers. No existing backend tests were found in the supplied source snapshot.
- **Required fixtures and controls:** Fixed Python parity cases derived from the TypeScript Reference Implementation; a reusable SSE parser for test assertions; `TestClient`; pytest `monkeypatch` or an equivalent narrow replacement of the route’s sleep reference; inputs that deliberately trigger zero, one, and multiple merge events; an input longer than 200 characters; and representative ASCII, whitespace, punctuation, repeated, tie, and Unicode cases.
- **Required regression checks:** `GET /health` remains healthy; Simple Chat still accepts the shared `ChatRequest` and emits its established stream; BPE requests do not retain state between runs.
- **Required quality checks:** Run the configured complete pytest suite, Ruff check, and strict mypy check through Poetry. Report results honestly and do not claim success without executed output.
- **Do not test:** Private merge-helper identity, private container choices, local variable names, internal loop structure, exact generator implementation, or another implementation detail not visible through the approved seams.
- **Do not test:** Exact wall-clock animation duration. Verify that production has the configured delay and that test execution bypasses it, but assert stream ordering rather than elapsed milliseconds.
- **Known limitation:** Automated backend tests do not prove browser rendering or Vite proxy integration. A manual request through the Vite proxy is useful during implementation when both servers can be run, but it is not a dependency of the Python test suite.

## Out of Scope

- `train_bpe_on_text()` or another API used only by later phases.
- Custom pre-token regular-expression support.
- Word2Vec-specific tokenization.
- Transformer-specific tokenization.
- Phase 3 XOR neural-network work.
- Phase 4 Word2Vec embedding work.
- Phase 5 transformer work.
- Multiprocessing, worker pools, or shared memory.
- Persisted BPE models or tokenizer files.
- Cross-request BPE caching.
- Frontend source changes.
- Input trimming or normalization.
- A new application-level maximum message length.
- Replacing the educational BPE implementation with a tokenizer library.
- LangChain, LangGraph, or a hosted AI service.
- A Node, Hono, or TypeScript backend.
- Running Node, pnpm, Hono, or TypeScript as part of Python backend tests.
- Exact wall-clock assertions for animation delays.
- A new SSE error-event design.
- Client-disconnect cancellation redesign.
- Unrelated cleanup, redesign, or future-phase abstractions.
- Issue-tracker ticket creation, implementation code, commits, or code review.

## Notes

- **Dependency:** Python 3.12 or newer, Poetry, FastAPI, Pydantic, pytest, pytest-asyncio, HTTPX, Ruff, and mypy are already declared in the supplied backend project. No dependency addition is required.
- **Dependency:** The unchanged TypeScript/Vite frontend depends on the exact Frontend Contract described in this specification.
- **Risk:** Python regular expressions can classify Unicode word characters differently from JavaScript. **Safeguard:** enforce explicit ASCII-style pre-token behavior and include a parity case.
- **Risk:** A pair-frequency tie can select a different first merge and cause all later results to diverge. **Safeguard:** preserve encounter-order selection and assert a fixed tie case.
- **Risk:** Internal snake_case keys can leak into response JSON. **Safeguard:** build explicit camelCase payload dictionaries and assert exact key sets.
- **Risk:** Moving shared request and SSE code can regress Simple Chat. **Safeguard:** include Simple Chat contract regression coverage through the HTTP seam.
- **Risk:** Real animation sleeps can make tests slow and flaky. **Safeguard:** patch the stable route-level sleep reference and never assert exact elapsed time.
- **Risk:** Large messages can consume substantial CPU in the application process. **Safeguard:** accept this known limitation for the focused slice; do not add unconfirmed limits or offloading.
- **Risk:** Future-phase abstractions can expand scope before the Basic Tokenizer works. **Safeguard:** expose only the public BPE operations required by this route.
- **Evidence limitation:** The supplied Python snapshot shows empty BPE, schema, SSE, and BPE route modules and no existing tests. No implementation was performed and no tests were executed while producing this specification.
- **Evidence limitation:** The TypeScript Reference Implementation was used through the confirmed handoff as the source of parity requirements; the implementation prompt must use the latest complete supplied source snapshot as its source of truth.
- **Official framework evidence:** FastAPI documents including `APIRouter` instances in the main application, and its testing guidance supports direct application testing with `TestClient`.
- **Official testing evidence:** pytest documents `monkeypatch` as a temporary, automatically restored mechanism for replacing referenced attributes during a test.
- **Publication target:** Local `SPEC.md`. No issue-tracker configuration, working tracker tool, or publication permission was supplied.

## Source Material Consulted

- `GRILL_WITH_DOCS_RESULT.md`
- `CONTEXT.md`
- `TO_SPEC_PROMPT.md`
- `py_llm_pipeline_explorer_file_structure.md`
- Supplied backend project configuration and Python source snapshot
- Confirmed TypeScript Reference Implementation behavior recorded in the handoff
- Official FastAPI documentation: Bigger Applications — Multiple Files
- Official FastAPI documentation: Testing
- Official Pydantic documentation for string-length constraints
- Official pytest documentation: How to monkeypatch/mock modules and environments
