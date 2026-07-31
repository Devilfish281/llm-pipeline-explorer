---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "024"
source_work_item: 024-stream-deterministic-generation-from-an-exact-saved-model.md
source_specification: SPEC.md
source_context: CONTEXT.md
architecture_decision: 0003-load-saved-transformer-models-for-stateless-generation.md
code_source_of_truth: py_llm_pipeline_explorer_file_structure(113).md
frontend_code_reference: ts_llm_pipeline_explorer_file_structure.md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 024: Stream Deterministic Generation from an Exact Saved Model

## Initial checklist

- Confirm Ticket 024 is the only selected work item and that its Ticket 023 blocker is represented by the completed exact-name Saved Transformer Model loader in the latest Python Backend export.
- Treat `py_llm_pipeline_explorer_file_structure(113).md` as the current-code source of truth; older exports, prior plans, pasted model files, and historical TypeScript cache behavior are non-authoritative when they conflict with it.
- Use Ticket 024 for immediate scope and acceptance behavior, with `SPEC.md`, `CONTEXT.md`, and ADR 0003 supplying the durable stateless-inference, safety, deterministic-generation, shared-slot, and terminology decisions.
- Reuse the completed Ticket 023 loader, current decoder-only forward pass, stable softmax, top-p sampler, Mulberry32 implementation, shared SSE formatter/response, and process-local Transformer request slot.
- Add the smallest complete backend-only vertical slice: one five-field request model, one prompt preparation/generation boundary, one named-file `/load-transformer` stream, and focused public HTTP/SSE tests.
- Keep latest-model selection, full deadline/disconnect/cancellation hardening, worker-count presentation, frontend command parsing/display, token streaming, caching, sessions, model repair, and training-from-snapshot behavior outside Ticket 024.
- Preserve the user-reported pytest, Ruff, and strict-mypy baseline without describing it as tool-verified in this planning session.
- Finish implementation with focused load-route tests, affected Transformer regressions, the complete backend suite once, Ruff lint and format checks, strict mypy, and a scope-only diff inspection.

## Source-of-truth hierarchy

1. The user's latest explicit direction: plan Ticket 024 only, convert the approved TypeScript behavior to Python, and treat the latest supplied Python Backend export as current-code truth.
2. `024-stream-deterministic-generation-from-an-exact-saved-model.md` for the exact named-file route behavior, request contract, prompt rules, deterministic output, SSE order, safe errors, approved test seam, blocker, and exclusions.
3. `py_llm_pipeline_explorer_file_structure(113).md` for current source, tests, public symbols, dependencies, paths, typing conventions, and the completed Ticket 023 implementation.
4. `SPEC.md` for the durable Phase 6 decisions concerning the separate inference endpoint, strict request validation, shared process-local slot, model-owned tokenization, seed `42`, exact event payloads, parent-process inference, same-process offloading, and compatibility.
5. `0003-load-saved-transformer-models-for-stateless-generation.md` for the architectural separation between fresh-weight Transformer Training Runs and stateless Saved Transformer Generation Runs.
6. `CONTEXT.md` for the canonical meanings of Saved Transformer Model, Saved Transformer Generation Run, Saved Transformer Event Stream, Transformer Training Run, and Request-Scoped Worker Group.
7. The completed current public boundaries in:
   - `src/how_llms_work/routes/train_transformer.py`;
   - `src/how_llms_work/ml/transformer.py`;
   - `src/how_llms_work/ml/bpe.py`;
   - `src/how_llms_work/ml/math_utils.py`;
   - `src/how_llms_work/schemas.py`;
   - `src/how_llms_work/sse.py`;
   - their current focused tests and fixtures.
8. `llm_works_file_structure.md` only as historical TypeScript behavior evidence for ordered BPE application, decoder-only generation, stable top-p sampling, and text reconstruction when it agrees with the approved Python Phase 6 specification.
9. The supplied frontend export only as a compatibility reference. Ticket 024 explicitly prohibits frontend changes, so absence of a frontend edit is intentional rather than an implementation gap.
10. Official FastAPI and Pydantic documentation only as technical cross-checks for streaming responses, request validation, aliases, strict fields, finite numbers, and standard HTTP `422` behavior.
11. Older code exports, previous implementation plans, real `.data` artifacts, generated caches, and historical “load cached weights instead of training” behavior are non-authoritative when they conflict with the sources above.

## Work-item summary

Ticket 024 adds the first complete named-file Saved Transformer Generation Run to the Python Backend.

A valid `POST /load-transformer` request must contain exactly the declared public fields `modelFile`, `prompt`, `temperature`, `topP`, and `maxTokens`. FastAPI and Pydantic must reject malformed structured requests with HTTP `422` before the process-local Transformer slot, filesystem, loader, tokenizer, or numerical generation boundary is touched.

After one valid request reserves the same nonblocking process-local slot used by Transformer training, the route must:

1. select only the exact nonempty filename supplied by the caller through Ticket 023's completed safe loader;
2. trim only the prompt's leading and trailing whitespace;
3. tokenize the complete trimmed prompt with the loaded snapshot's own ordered Merge Table and Vocabulary;
4. reject an empty prompt, unsupported text, or more than sixteen model tokens with the exact safe messages required by the ticket;
5. emit `loaded` only after model loading, validation, canonical parameter materialization, tokenization, and prompt-length checks succeed;
6. generate one complete continuation in the backend parent process with the current decoder-only forward mathematics, latest-sixteen-token context rule, temperature, stable softmax, top-p sampling, Vocabulary decoding, and a fresh request-owned Mulberry32 stream seeded exactly with `42`;
7. emit exactly one `result`, then exactly one empty-object `done`;
8. discard all request-owned state and release the shared slot.

The successful event sequence is exactly:

```text
loaded → result → done
```

The route must not emit training events or fields, stream individual tokens, create a Request-Scoped Worker Group, create IPC or shared-memory resources, cache a model, resume training, silently choose another model, or modify the frontend.

## Readiness and blocker assessment

- **Selected ticket:** Ticket 024 only.
- **Ticket status:** `ready-for-agent`.
- **Declared blocker:** Ticket 023.
- **Blocker result:** Satisfied in the latest current-code export.
- **Evidence:** `src/how_llms_work/routes/train_transformer.py` contains the public `LoadedTransformerModelSnapshot`, `SavedTransformerModelLoadError`, and `load_named_transformer_model()` boundary. `tests/test_transformer_loading.py` covers exact-name selection, strict current-format validation, duplicate-key rejection, exact-case handling, link/junction safety, canonical request-owned `float32` state, one-read/no-cache behavior, isolation, safe failure, and artifact preservation.
- **Material unresolved decision:** None for Ticket 024. The ticket deliberately leaves latest selection to Ticket 025 and full lifecycle hardening to Ticket 026.
- **Planning result:** Ready for implementation.

## Baseline evidence

- **Status:** User-reported.
- **Commands and reported results:**
  - `poetry run pytest` — all tests passed before planning.
  - `poetry run ruff check .` — passed before planning.
  - `poetry run mypy src` — reported `Success: no issues found`.
- **Planning-session execution:** No pytest, Ruff, mypy, formatter, server, browser, or implementation command was run during this read-only planning workflow.
- **Planning rule:** `implement-prompt` must re-inspect the live repository and establish or reconfirm its own baseline before editing.

## Current code observations from the latest source

1. `src/how_llms_work/main.py` currently includes the existing `train_transformer_router` and all completed routers. There is no registered `/load-transformer` operation, but adding the new endpoint to the existing Transformer router can register it without changing `main.py`.
2. `src/how_llms_work/schemas.py` contains `TrainTransformerRequest` with the established strict finite-number and camelCase-alias style. There is no `LoadTransformerRequest`.
3. `src/how_llms_work/sse.py` already provides the shared JSON SSE framing and `StreamingResponse` headers required by Ticket 024.
4. `src/how_llms_work/ml/bpe.py` already provides the ordered `Merge` representation and `apply_merges()` operation. It preserves pre-token order and does not invent unknown tokens.
5. `src/how_llms_work/ml/math_utils.py` already provides the deterministic JavaScript-compatible `Mulberry32` stream needed for the exact seed-`42` rule.
6. `src/how_llms_work/ml/transformer.py` already provides:
   - canonical parameter layouts and views;
   - the decoder-only causal forward pass;
   - stable row softmax;
   - the current stable top-p nucleus sampler;
   - Vocabulary-based text reconstruction;
   - the latest-sixteen-token context rule;
   - `generate_transformer_text()` for training samples.
7. The current `generate_transformer_text()` is not directly suitable for saved-model prompting because it:
   - starts from the fixed training preprocessing seed IDs;
   - seeds randomness with `(42 + epoch) mod 2^32`;
   - returns an epoch-labelled `GeneratedTextSample`;
   - accepts the global `TransformerPreprocessingSnapshot`, not an arbitrary loaded model's Vocabulary, Merge Table, and prompt token IDs.
8. `src/how_llms_work/routes/train_transformer.py` already owns:
   - `_TRANSFORMER_RUN_SLOT`;
   - the exact model-directory resolver and Ticket 023 named loader;
   - same-process thread-offload helpers;
   - the existing Transformer router;
   - training stream ownership and slot release.
9. The current training-only overlap detail is `A Transformer Training Run is already active.` The approved shared Phase 6 contract requires wording that applies to both training and loading.
10. `tests/test_train_transformer_route.py` already provides strong prior art for:
    - the public `TestClient` seam;
    - exact SSE parsing with duplicate-key checks;
    - route-registration regressions;
    - strict request-validation probes;
    - controlled nonblocking slot behavior;
    - call-order assertions;
    - success/failure slot release;
    - controlled generation and worker collaborators.
11. `tests/test_transformer_completion.py` already protects the current forward/generation mathematics, stable top-p behavior, deterministic streams, cancellation checks, result reconstruction, and saved-model construction.
12. `tests/test_transformer_loading.py` already proves the exact loader and request-owned snapshot boundary. Ticket 024 should consume it rather than duplicate or weaken it.
13. No current test module exercises `POST /load-transformer`.
14. No production dependency or lockfile change is needed.
15. Ticket 024 requires no TypeScript or Vite source change.

## Acceptance criteria coverage

| Ticket 024 criterion | Current classification | Planned coverage |
|---|---|---|
| Register `POST /load-transformer` and preserve completed endpoints | Not implemented | Steps 1 and 5 |
| Five exact public request fields with snake_case internals and camelCase aliases | Not implemented | Steps 1 and 2 |
| Standard HTTP `422` before slot or model access | Established pattern, new request not implemented | Steps 1 and 2 |
| Exact named model selection through Ticket 023; no fallback | Already satisfied at reusable loader seam, route integration missing | Steps 1, 5, and 7 |
| Exact named-model load error event | Loader has the exact message; SSE mapping missing | Steps 1 and 5 |
| Outer prompt trimming with interior preservation | Not implemented | Steps 1 and 3 |
| Exact empty-prompt error | Not implemented | Steps 1, 3, and 5 |
| Tokenize with loaded model Merge Table and Vocabulary | BPE primitive exists; saved-model prompt boundary missing | Steps 3 and 7 |
| Exact unsupported-text error without dropping or replacing text | Not implemented | Steps 3 and 5 |
| Accept one through sixteen tokens; reject seventeen or more with exact message | Not implemented | Steps 3 and 5 |
| Emit `loaded` only after all preconditions | Not implemented | Steps 1 and 5 |
| Exact `loaded` key set and values | Not implemented | Steps 1 and 5 |
| Reuse current decoder-only inference mathematics and latest-sixteen context | Core mathematics already evidenced; arbitrary prompt integration missing | Steps 4 and 7 |
| Fresh request-owned Mulberry32 seeded exactly with `42` | Mulberry32 exists; saved-generation seed rule missing | Steps 4 and 7 |
| Add up to `maxTokens`, emit exact `result`, and preserve the trimmed prompt prefix | Core sampler exists; saved-result boundary missing | Steps 4, 5, and 7 |
| Emit exactly one empty-object `done` after `result` | Shared SSE utility exists; route behavior missing | Steps 1 and 5 |
| Only `loaded → result → done`; no training or token events | Not implemented | Steps 1, 5, and 7 |
| Closed safe semantic/internal error mapping with no raw internals | Loader safe error exists; route-wide closed mapping missing | Steps 1, 5, and 6 |
| Parent-process inference with no training workers, IPC, shared memory, or worker label | Loader and numerical boundaries are parent-local; route proof missing | Steps 5, 6, and 7 |
| Request-owned loaded state discarded after stream; no cache/checkpoint/training reuse | Loader isolation/no-cache is evidenced; route-local ownership proof missing | Steps 6 and 7 |

### Classification summary

- **Already satisfied and evidenced:** exact safe named-model loading; strict current-format validation; canonical request-owned parameter materialization; no loaded-model cache; existing decoder-only forward mathematics; stable softmax; stable top-p sampling; Mulberry32; shared SSE utilities.
- **Behavior present but evidence incomplete:** same-process helper offloading, process-local Transformer slot, parent-local generation mathematics, latest-sixteen context behavior, and Vocabulary reconstruction all exist but are not wired to a named saved-model request.
- **Partially implemented:** the exact load error constant and loader boundary exist; the shared slot exists for training; the existing generation function contains most numerical logic but has training-specific input and seed semantics.
- **Not implemented:** `LoadTransformerRequest`, prompt tokenization/validation against a loaded snapshot, request-seeded saved-model generation, `/load-transformer`, exact load SSE sequencing, route-level safe-error mapping, cross-route slot integration, and load-route HTTP/SSE tests.
- **Evidence limitation:** baselines are user-reported, not tool-verified in this planning session. The live repository must be re-inspected before implementation because the plan was produced from a current source export rather than a writable checkout.

## Files to inspect before editing

1. `src/how_llms_work/schemas.py`
   - `TrainTransformerRequest`;
   - established `Annotated`, `Field`, `ConfigDict`, strict-number, finite-number, alias, and extra-field conventions;
   - destination for `LoadTransformerRequest`.
2. `src/how_llms_work/ml/bpe.py`
   - `Merge`;
   - `apply_merges()`;
   - pre-token behavior and unsupported-token preservation.
3. `src/how_llms_work/ml/math_utils.py`
   - `Mulberry32`;
   - exact request-owned seed behavior.
4. `src/how_llms_work/ml/transformer.py`
   - `SavedTransformerMerge`;
   - `InitializedTransformerParameters`;
   - `calculate_transformer_forward()`;
   - `stable_row_softmax`;
   - `_sample_transformer_nucleus_token()`;
   - `generate_transformer_text()`;
   - existing generation argument validation and latest-sixteen-token constants;
   - destination for the reusable saved-model prompt and generation boundaries.
5. `src/how_llms_work/routes/train_transformer.py`
   - `_TRANSFORMER_RUN_SLOT`;
   - `LoadedTransformerModelSnapshot`;
   - `SavedTransformerModelLoadError`;
   - `load_named_transformer_model()`;
   - existing same-process helper orchestration;
   - `stream_transformer_training()`;
   - `train_transformer()`;
   - existing `router`;
   - destination for the named `/load-transformer` orchestration.
6. `src/how_llms_work/sse.py`
   - `format_sse()`;
   - `create_sse_response()`;
   - shared media type and headers; no change expected by default.
7. `src/how_llms_work/main.py`
   - current router registration and endpoint preservation;
   - no change expected when `/load-transformer` is added to the existing Transformer router.
8. `tests/test_train_transformer_route.py`
   - exact SSE parser;
   - `ControlledRunSlot`;
   - strict validation probe;
   - route registration and endpoint-preservation tests;
   - overlap and slot-release prior art.
9. `tests/test_transformer_loading.py`
   - complete valid current-format model fixture construction;
   - temporary model-directory patching;
   - exact safe loader behavior;
   - request isolation and no-cache evidence.
10. `tests/test_transformer_completion.py`
    - deterministic generation fixtures;
    - top-p and latest-context evidence;
    - cancellation and numerical validation;
    - destination for focused saved-prompt generation tests when that remains the smallest organization.
11. `tests/fixtures/transformer_completion_reference.json`
    - stable public numerical/model evidence; reuse without asserting giant raw model arrays in route tests.
12. `pyproject.toml`
    - Python, FastAPI, Pydantic, NumPy, pytest, Ruff, formatting, and strict-mypy configuration;
    - confirm no dependency change.
13. `024-stream-deterministic-generation-from-an-exact-saved-model.md`, `SPEC.md`, `CONTEXT.md`, and ADR 0003
    - exact public behavior, safe messages, scope, and terminology.
14. `llm_works_file_structure.md`
    - historical ordered BPE and decoder-generation behavior only; do not copy cached-training, worker-thread, or direct filesystem behavior rejected by the Python architecture.
15. The latest frontend export
    - confirm that Ticket 024 requires no frontend edit and that later File-command routing remains owned by the dedicated frontend tickets.

## Step 1 — Freeze the public named-load HTTP/SSE contract with focused tests

**Files and symbols:**
- `tests/test_load_transformer_route.py` — new focused route test module.
- `tests/test_train_transformer_route.py` — existing exact SSE parser, route-registration assertions, slot controls, and validation probes to reuse or minimally generalize.
- `tests/test_transformer_loading.py` — current valid/invalid model fixture patterns to mirror without coupling tests to its private helper names.

**Purpose:**

Create acceptance-relevant red-to-green evidence through Ticket 024's approved FastAPI `TestClient` seam before production implementation. The tests must describe public request validation, exact event names and payloads, safe errors, shared-slot behavior, and training preservation without fixing private coroutine decomposition.

**Actions:**

- Add `tests/test_load_transformer_route.py`.
- Reuse the existing exact SSE parser behavior, including duplicate-key rejection and exact payload-key assertions. Extract a small shared test-only helper only if importing or duplicating the current parser would otherwise create avoidable maintenance risk.
- Add a route-registration test proving:
  - `POST /load-transformer` exists;
  - `GET /health`, `POST /simple-chat`, `POST /bpe-tokenize`, `POST /neural-net`, `POST /train-embed`, and `POST /train-transformer` remain registered;
  - no existing path is renamed or removed.
- Add request-model introspection and public HTTP tests for exactly the declared fields and aliases:
  - `modelFile`;
  - `prompt`;
  - `temperature`;
  - `topP`;
  - `maxTokens`.
- Add table-driven HTTP `422` cases for:
  - missing required fields;
  - `null`, array, object, number, or Boolean `modelFile`;
  - empty-string `modelFile`;
  - wrong-type prompt values;
  - numeric strings;
  - Booleans;
  - non-finite temperature or top-p;
  - out-of-range temperature and top-p;
  - non-integer, Boolean, string, and out-of-range `maxTokens`.
- Include valid boundary cases proving ordinary integer-valued JSON numbers remain valid for temperature/top-p when the existing Pydantic number contract permits them.
- Instrument the slot and model loader so every invalid request proves:
  - no reservation attempt;
  - no model-directory or loader call;
  - no generation call.
- Add controlled successful-stream tests asserting:
  - exact `loaded → result → done` order;
  - exactly one event of each successful type;
  - exact key sets: `loaded` has `file` and `prompt`, `result` has `text`, `done` is `{}`;
  - no `init`, `epoch`, token-level, worker, architecture, loss, model, or sample field.
- Add controlled semantic-error tests asserting one `error` event with exactly one `error` key and no forbidden raw details.
- Add exact message tests for:
  - `The saved Transformer model could not be loaded.`;
  - `The prompt must not be empty.`;
  - `The prompt contains text that this saved Transformer model cannot tokenize.`;
  - `The prompt must contain no more than 16 tokens.`
- Add failure-order tests proving no `loaded` is emitted for model, empty-prompt, unsupported-text, or prompt-length failure.
- Add one controlled post-`loaded` generation failure test proving it emits only the route's stable generic generation error after `loaded`, with no raw exception, `result`, or `done`.
- Do not require a manufactured red failure for behavior already supplied by Ticket 023; loader regression assertions may pass immediately and serve as integration evidence.

**Guardrails:**

- Test through HTTP/SSE and intentionally public collaborators, not private route helper names, local tasks, local variables, exact thread identity, private tensor views, or sampler-loop implementation.
- Keep all filesystem tests under pytest-managed temporary directories.
- Do not read, overwrite, repair, rename, or delete real `.data` artifacts.
- Do not add latest-model (`modelFile: null`) cases; those belong to Ticket 025.
- Do not add deadline/disconnect/cancellation completion tests reserved for Ticket 026.
- Do not add frontend tests or browser automation.
- Keep route tests fast by controlling loading and generation for most cases.

**Expected result:**

- Ticket 024's observable HTTP/SSE contract is represented by focused tests that initially fail only because the new schema, generation boundary, and route are absent.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q
```

Expected at this stage:

- Newly added tests fail for clearly identified missing Ticket 024 behavior, while existing loader and training tests remain unchanged.

## Step 2 — Add the strict five-field Saved Transformer request model

**Files and symbols:**
- `src/how_llms_work/schemas.py` — new `LoadTransformerRequest`.
- `tests/test_load_transformer_route.py` — exact schema and HTTP `422` cases.

**Purpose:**

Establish the public structured-request boundary so malformed input fails through FastAPI/Pydantic before slot reservation, model access, tokenization, or generation.

**Actions:**

- Add `LoadTransformerRequest` with exactly these declared internal attributes:
  - `model_file` with alias `modelFile`;
  - `prompt`;
  - `temperature`;
  - `top_p` with alias `topP`;
  - `max_tokens` with alias `maxTokens`.
- Make all five fields required for Ticket 024.
- Require `model_file` to be a strict string with minimum length one. Leave path grammar, exact case, canonical filename rules, and whitespace-only filename rejection to Ticket 023's semantic loader boundary.
- Require `prompt` to be a strict string but do not use a minimum-length Pydantic rule, because empty and whitespace-only prompts must produce the ticket's streamed semantic `error`, not HTTP `422`.
- Mirror the existing Transformer strict and finite number behavior:
  - temperature `0.1..2.0`;
  - top-p `0.1..1.0`;
  - maximum tokens strict integer `3..500`.
- Preserve the current request-model extra-field policy rather than introducing an unrelated global validation change. The model itself must declare only the five approved fields.
- Import the new request model only where the load route needs it.

**Guardrails:**

- Do not make `modelFile` optional or nullable; Ticket 025 owns `modelFile: null`.
- Do not add defaults that would allow missing fields.
- Do not convert numeric strings.
- Do not reject empty/whitespace prompt at Pydantic time.
- Do not change `TrainTransformerRequest` ranges, defaults, aliases, or behavior.
- Do not alter other request models.

**Expected result:**

- Valid named requests reach the route.
- Malformed structured requests receive standard HTTP `422` before any route-owned slot or model action.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q -k "request or validation or 422"
```

Expected result:

- Request shape, aliases, valid boundaries, and invalid structured input behavior pass.

## Step 3 — Add a model-owned prompt preparation boundary

**Files and symbols:**
- `src/how_llms_work/ml/transformer.py` — planned public saved-prompt value and tokenization operation; exact naming may be refined during live inspection.
- `src/how_llms_work/ml/bpe.py` — existing `Merge` and `apply_merges()` reused without modification by default.
- `tests/test_transformer_completion.py` — focused prompt preparation tests, or a narrowly named new numerical test module if that is cleaner.

**Purpose:**

Convert the complete trimmed starting prompt into one through sixteen exact model token IDs using only the selected snapshot's ordered Merge Table and Vocabulary, while preserving the original prompt text for display and result-prefix requirements.

**Actions:**

- Add one route-independent, intentionally public prompt preparation boundary in `ml/transformer.py`.
- Accept:
  - the caller's prompt string;
  - the loaded snapshot's ordered Vocabulary;
  - the loaded snapshot's ordered Merge Table.
- Trim only leading and trailing whitespace with the approved ordinary string behavior.
- Reject an empty or whitespace-only trimmed prompt with a dedicated semantic exception or typed outcome that the route can map to exactly `The prompt must not be empty.`
- Reconstruct the ordered BPE merge operations from the request-owned saved merge records without changing their order or content.
- Apply the loaded Merge Table to the complete trimmed prompt using the current BPE implementation and the Transformer's approved leading-space/tokenization convention represented by the saved model.
- Build the token-to-ID mapping from the loaded ordered Vocabulary for the current request.
- Require every produced token to match one exact Vocabulary entry.
- If any token is unsupported:
  - do not drop it;
  - do not replace it;
  - do not normalize it;
  - do not substitute a default token;
  - return a dedicated semantic failure mapped to the exact unsupported-text message.
- Accept exactly one through sixteen resulting model token IDs.
- Return a dedicated overlength failure for seventeen or more tokens.
- Return an immutable or request-owned prepared value containing at least:
  - the exact trimmed prompt;
  - the ordered prompt token IDs.
- Add focused tests for:
  - leading/trailing trimming;
  - preserved interior spaces and characters;
  - one-token and sixteen-token success;
  - seventeen-token failure;
  - unsupported character or fragment failure;
  - Merge Table order sensitivity;
  - Vocabulary order/token-ID sensitivity;
  - duplicate prompt token occurrences;
  - no mutation or reuse of the loaded snapshot metadata.

**Guardrails:**

- Do not use the global training preprocessing snapshot to tokenize a loaded model prompt.
- Do not retrain BPE or reorder merges.
- Do not lowercase, repair, or normalize prompt content unless the exact saved-model Transformer tokenization contract explicitly requires that behavior; preserve the ticket's unsupported-text rule.
- Do not silently truncate prompts to the latest sixteen tokens. The maximum is a validation rule before `loaded`.
- Do not import the route-owned `LoadedTransformerModelSnapshot` into the ML module; pass the needed request-owned fields to avoid a route-to-ML cycle.
- Do not add an unknown-token ID or change the saved model format.

**Expected result:**

- The backend can prepare a valid prompt against one loaded snapshot and can classify the three exact prompt semantic failures before any `loaded` event.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_completion.py -q -k "saved and prompt"
```

Use the actual focused test selector chosen during implementation.

Expected result:

- Prompt trimming, model-owned tokenization, exact token IDs, unsupported-text rejection, and the sixteen-token boundary pass.

## Step 4 — Add deterministic arbitrary-prompt saved-model generation

**Files and symbols:**
- `src/how_llms_work/ml/transformer.py` — planned public saved-model generation operation.
- Existing symbols to reuse:
  - `calculate_transformer_forward()`;
  - `stable_row_softmax`;
  - `_sample_transformer_nucleus_token()`;
  - `Mulberry32`;
  - `TRANSFORMER_SEQUENCE_LENGTH`;
  - current generation argument validation where compatible.
- `tests/test_transformer_completion.py` — deterministic saved-generation evidence.

**Purpose:**

Reuse the completed educational Transformer inference mathematics while replacing the training-sample-specific seed and epoch behavior with Ticket 024's request-owned prompt and exact seed `42`.

**Actions:**

- Add one route-independent public operation for saved-model continuation generation.
- Accept:
  - the loaded request-owned `InitializedTransformerParameters`;
  - the loaded ordered Vocabulary;
  - the prepared trimmed prompt and prompt token IDs;
  - temperature;
  - top-p;
  - maximum new tokens;
  - the current request's cancellation token only if needed by the existing numerical boundary and without completing Ticket 026's route lifecycle.
- Validate the loaded parameter/vocabulary compatibility before generation.
- Start the accumulated token-ID sequence from all prepared prompt token IDs.
- Create a new `Mulberry32(42)` for every invocation.
- For each new token:
  - use only the latest sixteen accumulated IDs as forward context;
  - call the current decoder-only forward pass;
  - select the final-position logits;
  - apply the existing temperature scaling and finite checks;
  - apply the current stable softmax;
  - sample with the current stable minimum top-p nucleus logic;
  - append the sampled model token ID.
- Generate the requested number of new tokens unless an already-approved numerical stop condition exists. Do not invent an EOS rule.
- Construct the returned complete text so it begins byte-for-byte with the exact trimmed original prompt, followed by the Vocabulary strings for newly generated IDs. Do not reconstruct the user-supplied prefix from model tokens when doing so could change its preserved interior characters or spacing.
- Return one complete plain text result, not token updates and not an epoch-labelled training sample.
- Add focused tests proving:
  - fresh seed `42` on every call;
  - repeated identical model state and request values produce identical text;
  - a different prior call cannot advance a later request's stream;
  - generated text begins exactly with the trimmed prompt;
  - exactly the requested number of model tokens is appended in the no-EOS implementation;
  - context windows use the latest sixteen IDs after the prompt grows;
  - temperature and top-p flow to the current sampler;
  - existing training `generate_transformer_text()` behavior and fixtures remain unchanged;
  - invalid or non-finite numerical state never produces a successful result.

**Guardrails:**

- Do not alter the current training sample's `(42 + epoch)` seed contract.
- Do not use NumPy randomness, Python `random`, a shared generator, or module-level mutable random state.
- Do not duplicate the decoder forward pass, stable softmax, or top-p algorithm.
- Do not create workers, processes, pipes, queues, managers, shared-memory blocks, or GPU/framework paths.
- Do not add token streaming, sessions, conversation history, caching, checkpoint logic, or training mutation.
- Keep all loaded arrays request-owned and read-only by convention during inference; generation must not mutate model parameters.

**Expected result:**

- One synchronous parent-process numerical boundary can deterministically generate a complete prompt-plus-continuation result from any valid prepared saved-model prompt.

**Verification:**

```powershell
poetry run pytest tests/test_transformer_completion.py -q -k "saved and generation"
```

Use the actual focused selector chosen during implementation.

Expected result:

- Seed, context, sampling, exact-prefix, token-count, isolation, and training-regression tests pass.

## Step 5 — Implement the named `/load-transformer` SSE route

**Files and symbols:**
- `src/how_llms_work/routes/train_transformer.py`:
  - existing `router`;
  - `_TRANSFORMER_RUN_SLOT`;
  - `SavedTransformerModelLoadError`;
  - `load_named_transformer_model()`;
  - current same-process helper seam;
  - new `load_transformer()` endpoint;
  - new Saved Transformer Generation Run stream operation.
- `src/how_llms_work/schemas.py` — `LoadTransformerRequest`.
- `src/how_llms_work/sse.py` — `format_sse()` and `create_sse_response()` reused unchanged.
- `tests/test_load_transformer_route.py` — public HTTP/SSE contract.

**Purpose:**

Expose the completed exact loader, prompt preparation, and deterministic generation boundaries as one distinct Saved Transformer Event Stream without reinterpreting inference as training.

**Actions:**

- Add `POST /load-transformer` to the existing Transformer `APIRouter` so `main.py`'s current router inclusion registers the route automatically.
- Accept the validated `LoadTransformerRequest` and `Request`.
- After Pydantic validation, acquire `_TRANSFORMER_RUN_SLOT` nonblockingly.
- On overlap, return HTTP `429` using the approved shared wording:

  ```text
  Another Transformer request is already running.
  ```

- Keep model selection, file reading, model validation, parameter materialization, prompt preparation, and generation in the backend parent process.
- Run blocking loader/tokenizer/generator work through the current same-process thread-offload pattern or the smallest safe generalization of it. Do not create any multiprocessing resource.
- Keep all request-specific objects local to the endpoint/stream:
  - payload values;
  - loaded snapshot;
  - prepared prompt;
  - random stream inside generation;
  - result text;
  - semantic outcome.
- Inside the stream:
  1. call `load_named_transformer_model(payload.model_file)`;
  2. prepare and validate the prompt against the returned snapshot;
  3. emit `loaded` with exactly `{"file": exact_filename, "prompt": trimmed_prompt}`;
  4. run deterministic generation with the snapshot parameters, Vocabulary, prepared token IDs, temperature, top-p, and maximum tokens;
  5. emit `result` with exactly `{"text": complete_text}`;
  6. emit `done` with exactly `{}`;
  7. terminate.
- Map `SavedTransformerModelLoadError` to one `error` event with exactly:

  ```json
  {"error":"The saved Transformer model could not be loaded."}
  ```

- Map the three prompt outcomes to their exact one-key `error` payloads.
- Use one route-owned stable generic generation-failure constant for unexpected numerical/internal generation failure. Ticket 024 does not prescribe that generic wording, but it must come from a closed mapping rather than exception text.
- Log unexpected details only through the backend logger; never place them in SSE data.
- Prevent all later successful events after any error.
- Release the slot in the stream's final ownership path.
- Release the slot immediately if response/stream construction fails before ownership transfers to the stream.
- Update the existing training overlap detail to the same shared wording so either route accurately describes the shared process-local resource.

**Guardrails:**

- Do not call the latest-model selector or accept `modelFile: null`.
- Do not catch a named load failure and try another file.
- Do not emit `loaded` before complete model and prompt validation.
- Do not emit `init`, `epoch`, token events, architecture, loss, samples, worker count, model metadata, or arrays.
- Do not expose raw exceptions, paths, filenames from rejected candidates, tracebacks, resource IDs, or numerical coordinates.
- Do not add full five-minute deadline, complete disconnect observation, helper-draining, or all-outcome cancellation semantics reserved for Ticket 026. Preserve the current basic cancellation behavior where already supplied, but do not claim Ticket 026 is complete.
- Do not modify frontend source.
- Do not modify or persist the loaded model.

**Expected result:**

- A valid named request returns the shared SSE response with exactly `loaded → result → done`.
- A semantic or internal failure returns only the applicable safe `error` behavior.
- Training and loading use one nonblocking process-local slot.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q
```

Expected result:

- Route registration, standard `422`, shared `429`, exact messages, event sequencing, payload key sets, error privacy, and basic slot release pass.

## Step 6 — Prove shared-slot ownership, request isolation, and absence of training resources

**Files and symbols:**
- `tests/test_load_transformer_route.py` — shared-slot, isolation, and no-worker tests.
- `tests/test_train_transformer_route.py` — training overlap wording/regression when the shared behavior is best kept beside existing training tests.
- `src/how_llms_work/routes/train_transformer.py` — only the minimum ownership and shared-message changes required by failing public tests.

**Purpose:**

Ensure Saved Transformer Generation Runs remain stateless parent-process inference and cannot overlap a Transformer Training Run, leak the shared slot, retain loaded state, or accidentally enter the worker-group training path.

**Actions:**

- Add controlled overlap tests proving:
  - an active training request causes a valid load request to receive immediate `429`;
  - an active load request causes a valid training request to receive immediate `429`;
  - overlap does not queue;
  - the exact shared detail is returned by both routes.
- Add slot-release tests after:
  - exact model-load error;
  - empty prompt;
  - unsupported prompt;
  - prompt overlength;
  - successful `loaded → result → done`;
  - controlled internal generation error;
  - response preparation failure before stream ownership.
- After each outcome, send a subsequent valid Transformer request and prove it can reserve the slot.
- Add a no-worker test that makes `create_request_scoped_worker_group()` fail immediately if the load route calls it.
- Add equivalent tripwires for worker-group labels or training event builders where the current public seam permits.
- Assert load-route output never contains:
  - `Transformer worker processes`;
  - `init`;
  - `epoch`;
  - `architecture`;
  - `finalLoss`;
  - `samples`.
- Issue two identical controlled requests and prove:
  - the named loader is called separately for each request;
  - generation receives distinct request-owned snapshots/parameter blocks;
  - no module-level “current loaded model” or random stream is reused;
  - identical request/model inputs still produce identical public text.
- Where practical, mutate or replace the temporary model between requests and prove the second request reaches the loader again. Keep the stronger exact one-read/no-cache filesystem proof in `tests/test_transformer_loading.py`.
- Verify training still initializes fresh parameters and creates its Request-Scoped Worker Group; loading must never provide training initialization state.

**Guardrails:**

- Do not assert CPython garbage-collection timing, object destruction timing, private local-variable deletion, or memory zeroing.
- Prove request-local state through observable distinct loads, distinct snapshots, no cache, and no cross-request random-state advancement.
- Do not implement the full disconnect/deadline/cancellation matrix from Ticket 026.
- Do not introduce a queue, async waiting lock, machine-wide lock, cross-process lock, or global model cache.
- Do not weaken current training cleanup or persistence behavior.

**Expected result:**

- Both Transformer operations share one immediate process-local exclusion slot.
- All Ticket 024 outcomes release their basic ownership.
- Loading creates no training resources and retains no model or random state across requests.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py tests/test_train_transformer_route.py -q -k "slot or overlap or worker or isolation or error"
```

Expected result:

- Shared `429`, release, no-worker, state-isolation, safe-error, and training-regression tests pass.

## Step 7 — Add one bounded real loading-and-generation integration and run affected regressions

**Files and symbols:**
- `tests/test_load_transformer_route.py` — one bounded integration through real public loader, prompt preparation, forward pass, and generator.
- `tests/test_transformer_loading.py` — existing exact model loader regressions.
- `tests/test_transformer_completion.py` — current generation regressions.
- `tests/test_train_transformer_route.py` — current training route regressions.
- `tests/fixtures/transformer_completion_reference.json` — stable current-format fixture evidence.

**Purpose:**

Prove that the controlled route contract is connected to the real Ticket 023 loader and existing Transformer numerical boundaries without turning the route suite into a long training run.

**Actions:**

- Build one complete valid current-format one-layer Saved Transformer Model using the already-tested public model-construction or stable fixture boundary.
- Persist it only inside a pytest temporary model directory under its exact canonical filename.
- Patch the production model-directory seam to that temporary directory.
- Select a prompt known to tokenize completely with that model's own Merge Table and Vocabulary and to contain no more than sixteen tokens.
- Send one real public `POST /load-transformer` request with the minimum supported `maxTokens`.
- Allow the request to traverse:
  - exact safe named selection;
  - one real file read;
  - strict model validation;
  - canonical parameter materialization;
  - real prompt tokenization;
  - real decoder forward pass;
  - stable softmax;
  - top-p sampling;
  - fresh seed `42`;
  - exact SSE formatting.
- Assert:
  - HTTP success and shared SSE headers;
  - exact `loaded → result → done`;
  - exact selected filename and trimmed prompt;
  - result begins exactly with the trimmed prompt;
  - result contains the expected number of appended model tokens;
  - a repeated identical request produces identical text;
  - no worker resource is created.
- Run the existing loader, completion, training-route, worker, worker-group, persistence, and core Transformer regressions.
- Avoid training a new model in this integration test.

**Guardrails:**

- Do not use a real repository `.data` file as automated-test input.
- Do not assert giant raw weight arrays or private view identities at the route seam.
- Do not make maximum-token or maximum-layer generation part of the focused route suite.
- Do not duplicate Ticket 023's exhaustive path and malformed-model matrix.
- Do not modify persistence files or frontend code.

**Expected result:**

- One bounded public request proves the complete named-file backend vertical slice works through real loading and inference boundaries.
- All completed Transformer training, loading, numerical, persistence, and worker behavior remains intact.

**Verification:**

```powershell
poetry run pytest `
    tests/test_load_transformer_route.py `
    tests/test_transformer_loading.py `
    tests/test_transformer_completion.py `
    tests/test_train_transformer_route.py `
    tests/test_transformer.py `
    tests/test_transformer_math.py `
    tests/test_transformer_training.py `
    tests/test_transformer_worker.py `
    tests/test_transformer_worker_group.py `
    tests/test_train_transformer_persistence.py
```

Expected result:

- All affected tests pass.

## Step 8 — Complete backend validation and inspect final scope

**Files and symbols:**
- All changed backend and test files.
- Repository configuration and Git diff.

**Purpose:**

Confirm Ticket 024 is complete, typed, formatted, regression-safe, and isolated from sibling Phase 6 tickets.

**Actions:**

- Run the focused load-route suite first.
- Run affected Transformer regressions.
- Run the complete backend test suite once.
- Run Ruff lint.
- Run Ruff format check.
- Run strict mypy.
- Inspect whitespace and final diff.
- Confirm no production `.data` model, frontend file, dependency declaration, lockfile, cache, generated artifact, ticket, specification, glossary, or ADR changed.
- Confirm no Ticket 025, 026, 027, 028, or 029 behavior was accidentally implemented.
- Record actual command output honestly in the implementation handoff.
- Create no commit until every required check passes.

**Guardrails:**

- Do not report a command as passed unless its current implementation run completed successfully.
- Do not hide pre-existing failures; distinguish them from Ticket 024 regressions.
- Do not use broad formatting that touches unrelated files.
- Do not amend model files or tests merely to force a green result.

**Expected result:**

- The final diff contains only the smallest backend and test changes required for Ticket 024.
- All required backend quality gates pass before commit.

**Verification:**

```powershell
poetry run pytest tests/test_load_transformer_route.py -q

poetry run pytest `
    tests/test_transformer_loading.py `
    tests/test_transformer_completion.py `
    tests/test_train_transformer_route.py

poetry run pytest

poetry run ruff check .

poetry run ruff format --check .

poetry run mypy src

git diff --check
git status --short
```

Expected result:

- Focused tests pass.
- Affected regressions pass.
- The complete suite passes.
- Ruff reports no lint or format failures.
- Mypy reports no issues.
- `git diff --check` reports no whitespace errors.
- `git status --short` contains only intended Ticket 024 files.

## Focused verification plan

Run in this order:

```powershell
poetry run pytest tests/test_load_transformer_route.py -q
```

```powershell
poetry run pytest `
    tests/test_transformer_completion.py `
    tests/test_transformer_loading.py `
    tests/test_train_transformer_route.py
```

Expected result:

- Exact request, route, prompt, deterministic generation, SSE, safe error, slot, loader, and training regression behavior passes.

## Full verification plan

```powershell
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
git diff --check
```

Expected result:

- All tests pass.
- Ruff lint and format checks pass.
- Strict mypy succeeds.
- The diff has no whitespace errors.

## Manual backend acceptance checklist

Use direct backend HTTP testing because frontend `File:` command parsing is intentionally deferred to later tickets.

- [ ] Start the backend from the backend project directory with the established Poetry/Uvicorn command.
- [ ] Confirm `/docs` lists both `POST /train-transformer` and `POST /load-transformer`.
- [ ] Confirm all previously completed endpoints remain listed.
- [ ] Send one valid named request for an existing exact canonical model filename.
- [ ] Confirm the response media type is SSE and the successful event order is exactly `loaded`, `result`, `done`.
- [ ] Confirm `loaded` displays only the exact filename and trimmed prompt.
- [ ] Confirm `result.text` begins exactly with the trimmed prompt and contains one complete continuation.
- [ ] Repeat the identical request and confirm identical text.
- [ ] Send a differently capitalized or missing filename and confirm only `The saved Transformer model could not be loaded.` is exposed.
- [ ] Send an empty or whitespace-only prompt and confirm only `The prompt must not be empty.` is exposed.
- [ ] Send unsupported text and confirm the exact unsupported-text message.
- [ ] Send a prompt that tokenizes to at least seventeen tokens and confirm the exact sixteen-token-limit message.
- [ ] Send a malformed structured request and confirm HTTP `422`, not an SSE semantic error.
- [ ] While one Transformer request owns the slot, confirm a second valid training or loading request receives immediate HTTP `429`.
- [ ] After success and each basic handled error, confirm a later request can acquire the slot.
- [ ] Confirm no load event or text contains a worker-process label, training epoch, loss, architecture, sample collection, path, traceback, model array, or internal identifier.
- [ ] Confirm loading does not modify the selected saved model's bytes or metadata intentionally.
- [ ] Record only the observations actually performed.

Suggested direct PowerShell request shape:

```powershell
$body = @{
    modelFile  = "transformer-weights-e100-l1-d32-h2-ff128-ctx32.json"
    prompt     = "once upon a time"
    temperature = 0.8
    topP       = 0.9
    maxTokens  = 3
} | ConvertTo-Json

curl.exe -N `
    -X POST `
    "http://127.0.0.1:8000/load-transformer" `
    -H "Content-Type: application/json" `
    --data $body
```

Use a filename and prompt that are actually valid for the locally selected saved model. Do not change the model to make the smoke test pass.

## Expected files changed

Likely changed:

```text
src/how_llms_work/schemas.py
src/how_llms_work/ml/transformer.py
src/how_llms_work/routes/train_transformer.py
tests/test_load_transformer_route.py
tests/test_transformer_completion.py
```

Conditionally changed only when live test organization proves it is the smallest safe option:

```text
tests/test_train_transformer_route.py
tests/transformer_test_support.py
```

Expected unchanged:

```text
src/how_llms_work/main.py
src/how_llms_work/sse.py
src/how_llms_work/ml/bpe.py
src/how_llms_work/ml/math_utils.py
src/how_llms_work/ml/transformer_worker.py
src/how_llms_work/routes/__init__.py
pyproject.toml
poetry.lock
frontend/**
.data/**
SPEC.md
CONTEXT.md
0003-load-saved-transformer-models-for-stateless-generation.md
024-stream-deterministic-generation-from-an-exact-saved-model.md
```

## Files and areas not to change

```text
frontend/**
src/how_llms_work/routes/train_embed.py
src/how_llms_work/routes/neural_net.py
src/how_llms_work/ml/word2vec.py
src/how_llms_work/ml/neural_net.py
src/how_llms_work/ml/transformer_worker.py
.data/**
tests/fixtures unrelated to Transformer completion
poetry.lock
```

Do not add:

```text
a separate model registry
a latest-model cache
a model manifest or database
a new worker pool
a Queue or Manager
new shared-memory state
a GPU or ML framework dependency
a hosted-model client
a session or conversation store
a token-stream event
a frontend selector or new page
```

## Risk notes and safeguards

1. **Risk:** Pydantic coercion accepts numeric strings or Booleans.
   - **Safeguard:** Use strict fields and table-driven HTTP `422` tests that prove neither slot nor loader is touched.
2. **Risk:** Empty prompts become HTTP `422` instead of the required SSE semantic error.
   - **Safeguard:** Keep `prompt` type-strict but perform empty/whitespace validation after route entry.
3. **Risk:** `modelFile` becomes nullable early and accidentally implements Ticket 025.
   - **Safeguard:** Require one nonempty strict string in Ticket 024 and add an explicit `null` rejection test.
4. **Risk:** Prompt tokenization uses global training preprocessing rather than the selected model.
   - **Safeguard:** Pass only the loaded snapshot's copied Vocabulary and Merge Table to the public prompt boundary.
5. **Risk:** Prompt preprocessing silently lowercases, normalizes, drops, or substitutes unsupported text.
   - **Safeguard:** Preserve the complete trimmed prompt and require every produced token to resolve exactly; test unsupported text and exact prefix preservation.
6. **Risk:** The implementation silently truncates a long prompt to the latest sixteen tokens.
   - **Safeguard:** Validate one-through-sixteen before `loaded`; reserve latest-sixteen slicing only for growing generation context after validation.
7. **Risk:** Reconstructing the complete result from token strings changes the user's original prompt spacing.
   - **Safeguard:** Preserve the exact trimmed prompt as the public prefix and append only newly generated Vocabulary strings.
8. **Risk:** Saved-model generation reuses `(42 + epoch)` or a module-level random stream.
   - **Safeguard:** Create a fresh `Mulberry32(42)` per saved-generation invocation and protect training's existing seed contract with regressions.
9. **Risk:** The route duplicates forward, softmax, or top-p mathematics.
   - **Safeguard:** Reuse current public numerical operations and extract only the minimum shared generation core if direct reuse is otherwise impossible.
10. **Risk:** `loaded` is emitted before prompt tokenization or complete parameter validation.
    - **Safeguard:** Perform Ticket 023 loading/materialization and complete prompt preparation before the first successful event; assert call order.
11. **Risk:** A named load failure falls back to another model.
    - **Safeguard:** Call only `load_named_transformer_model()` with the supplied exact string; never enumerate alternatives in Ticket 024.
12. **Risk:** The load route accidentally creates a Request-Scoped Worker Group.
    - **Safeguard:** Keep inference in same-process helpers and add worker-creation tripwire tests.
13. **Risk:** Training-only fields leak into the Saved Transformer Event Stream.
    - **Safeguard:** Assert exact payload key sets and reject all extra successful events or fields.
14. **Risk:** Raw exceptions, paths, arrays, or numerical values reach the client.
    - **Safeguard:** Use dedicated semantic outcomes and one closed safe mapping; log unexpected details server-side only.
15. **Risk:** The shared slot queues or allows training and loading concurrently.
    - **Safeguard:** Reuse one nonblocking process-local lock and test both overlap directions.
16. **Risk:** Slot ownership leaks after a basic handled error or stream construction failure.
    - **Safeguard:** Separate pre-stream and stream ownership, release in exactly one final path, and issue a subsequent-request test after every Ticket 024 outcome.
17. **Risk:** Thread offloading is mistaken for multi-core inference or Ticket 026 lifecycle completion.
    - **Safeguard:** Describe it only as same-process responsiveness; defer the full deadline/disconnect/cancellation matrix and helper draining to Ticket 026.
18. **Risk:** Loaded arrays or random state survive across requests.
    - **Safeguard:** Keep all state local, retain no cache, use a new loader invocation and random stream every request, and test changed-between-request and repeated-identical behavior.
19. **Risk:** Route tests become slow by performing real model generation in every case.
    - **Safeguard:** Use controlled collaborators for contract/error tests and exactly one bounded real integration.
20. **Risk:** Tests alter trained models.
    - **Safeguard:** Use `tmp_path` for every automated model file and never write to production `.data`.
21. **Risk:** A separate load router duplicates or disconnects the shared slot.
    - **Safeguard:** Add `/load-transformer` to the existing Transformer router by default; introduce a shared resource module only if live repository evidence proves it necessary and smaller.
22. **Risk:** Ticket 024 drifts into later Phase 6 work.
    - **Safeguard:** Explicitly exclude latest selection, five-minute deadline, complete disconnect/cancellation hardening, worker-count display, frontend command routing, and frontend result display.

## Commit guidance after tests pass

Suggested outcome-oriented commit title:

```text
feat: stream generation from exact saved transformer
```

The commit body should mention:

- strict five-field named load request validation;
- exact Ticket 023 model selection with no fallback;
- model-owned prompt tokenization and sixteen-token validation;
- deterministic request-owned seed `42`;
- reused decoder-only forward and top-p mathematics;
- exact `loaded → result → done` SSE behavior;
- closed safe error mapping;
- shared nonblocking Transformer slot;
- parent-process inference with no training worker resources;
- request-local no-cache state;
- the exact focused and full commands actually run and their observed results.

Do not create a commit during `to-plan-prompt`. Do not include frontend files, production `.data` artifacts, dependencies, lockfiles, caches, or sibling-ticket behavior.

## Handoff to implement-prompt

Run `implement-prompt` in a fresh conversation using:

- `plan024.md`;
- `024-stream-deterministic-generation-from-an-exact-saved-model.md`;
- the completed Ticket 023 implementation or Ticket 023 document as blocker evidence;
- `SPEC.md`;
- `CONTEXT.md`;
- `0003-load-saved-transformer-models-for-stateless-generation.md`;
- the latest live Python Backend repository, with `py_llm_pipeline_explorer_file_structure(113).md` as the supplied current-code reference;
- the latest TypeScript reference only for approved behavior evidence;
- `llm_works_file_structure.md`.

`implement-prompt` must:

1. re-inspect the live repository and preserve user changes;
2. establish its own baseline before editing;
3. implement only Ticket 024;
4. add acceptance-relevant tests before or alongside the minimum production changes;
5. reuse Ticket 023 and the existing numerical/SSE/slot boundaries;
6. run focused verification before the complete suite;
7. report every command result honestly;
8. inspect final scope;
9. create the implementation commit only after all required checks pass.
