---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "028"
source_work_item: 028-parse-and-route-file-commands-in-the-existing-transformer-input.md
source_specification: SPEC.md
source_context: CONTEXT.md
architecture_decision: 0003-load-saved-transformer-models-for-stateless-generation.md
backend_code_reference: py_llm_pipeline_explorer_file_structure(146).md
frontend_code_source_of_truth: ts_llm_pipeline_explorer_file_structure.md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 028: Parse and Route File Commands in the Existing Transformer Input

## Initial checklist

- Confirm Ticket 028 is the only selected work item. It is ready, has no blockers, and is limited to the TypeScript/Vite frontend command and request-start boundary.
- Treat the latest live frontend repository as implementation authority. Use the latest `ts_llm_pipeline_explorer_file_structure.md` and `llm_works_file_structure.md` as planning evidence, and re-inspect the live files before editing.
- Treat `py_llm_pipeline_explorer_file_structure(146).md` as backend compatibility evidence only. Ticket 028 must not change Python schemas, routes, loading, generation, lifecycle, worker behavior, or persistence.
- Preserve the existing numeric Transformer parser exactly, including its current five fields, defaults, `Number.parseInt`/`Number.parseFloat` conversion behavior, fallback behavior, and handling of omitted or extra positions.
- Classify a case-insensitive `File:` prefix after leading whitespace is ignored and before the numeric parser can run.
- Add a pure typed submission-planning boundary that either returns one exact browser request descriptor or one local validation result with no request descriptor.
- Clear only the Transformer Learning Demo's prior messages/results at request start; leave Simple Chat, BPE, XOR, and Word2Vec history behavior unchanged.
- Use the already declared Vitest dependency, explicit imports from `vitest`, and a one-shot test command. Do not add a framework, DOM environment, or browser-automation dependency.
- Keep saved-model event rendering outside this plan. Ticket 029 owns typed `loaded`, `result`, `done`, and `error` display behavior.
- Report the user's pytest, Ruff, and mypy results as user-reported baseline evidence only. Do not claim any planning-session command was executed.

## Source-of-truth hierarchy

1. The user's current request: produce one implementation plan for Ticket 028 and preserve the reported passing baseline.
2. `028-parse-and-route-file-commands-in-the-existing-transformer-input.md` for immediate scope, acceptance criteria, approved test seam, exclusions, and handoff.
3. The live TypeScript/Vite frontend repository at implementation time.
4. The latest `ts_llm_pipeline_explorer_file_structure.md` for current frontend files, imports, endpoint literals, hook behavior, package scripts, and declared Vitest version.
5. `SPEC.md` for the durable command grammar, result-clearing scope, endpoint separation, request fields, validation bounds, and pure-function testing decisions.
6. `CONTEXT.md` for the canonical Frontend Contract and Saved Transformer Generation Run terminology.
7. ADR 0003 for the existing-input `File:` command and separate stateless inference endpoint.
8. `py_llm_pipeline_explorer_file_structure(146).md` for confirmation that the Python Backend already exposes the compatible `/train-transformer` and `/load-transformer` operations and five-field load request.
9. `llm_works_file_structure.md` only as compatible historical behavior evidence for the current numeric Transformer parser and shared SSE flow.
10. Official Vitest documentation for one-shot `vitest run`, `.test.`/`.spec.` discovery, and explicit test API imports; official JavaScript documentation for strict finite/integer checks and the leniency of `parseFloat`; official Vite documentation for `/api` proxy rewriting.

## Work-item summary and scope correction

Ticket 028 is a **frontend TypeScript ticket**, even though the larger project goal is to migrate server behavior from TypeScript to Python. The Python Backend work needed by this command already exists in the supplied backend snapshot. This plan therefore does not convert or add Python code. It connects the existing TypeScript/Vite input to the existing Python endpoints while preserving the current numeric training path.

The implementation must create one pure typed boundary that converts a submitted Transformer input into one of two outcomes:

- a valid request descriptor containing the browser endpoint and exact request body; or
- a local validation result containing helpful assistant text and no request descriptor.

For browser code, retain the current Vite proxy convention:

- numeric training command → browser request to `/api/train-transformer`, which Vite rewrites to backend `POST /train-transformer`;
- valid `File:` command → browser request to `/api/load-transformer`, which Vite rewrites to backend `POST /load-transformer`.

Do not replace the current `/api` browser endpoints with direct port-8000 URLs.

## Baseline evidence and evidence limitations

### User-reported backend baseline

The user reports these commands passed before planning:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Record that evidence as `user-reported`; do not describe it as independently verified during this planning session.

### Frontend baseline to establish during implementation

The latest frontend export declares Vitest but does not show a `test` script. Before changing code, run from `frontend/`:

```powershell
pnpm typecheck
pnpm lint
pnpm build
pnpm exec vitest run
```

Interpret “no test files found” separately from a failing test. Preserve the exact output in the implementation record.

### Planning limitations

- No repository checkout was mounted for direct edits or command execution while this plan was written.
- No frontend or backend tests, lint checks, type checks, builds, servers, or browser checks were run during planning.
- The current exports are evidence, but the live repository must be re-inspected because user changes may be newer.
- Exact private helper names may vary during implementation. Public behavior and the approved pure test seam are binding.

## Current-code observations

### `frontend/src/client/hooks/use-train-transformer-chat.tsx`

The current hook:

- uses a fixed browser endpoint, `/api/train-transformer`;
- tokenizes every input with `input.trim().split(/\s+/)`;
- converts the first five positions into `epochs`, `temperature`, `topP`, `numLayers`, and `maxTokens`;
- uses the established defaults `300`, `0.8`, `0.9`, `2`, and `40`;
- relies on `Number.parseInt(...) || default` and `Number.parseFloat(...) || default`;
- ignores positions after the fifth;
- initializes and renders training state only.

Without a classification boundary, `File:` is currently consumed by the numeric parser and falls into training defaults. The numeric body construction must be moved or wrapped without changing its observable behavior.

### `frontend/src/client/hooks/use-sse-chat.ts`

The shared hook currently:

- receives one fixed `endpoint` and one `buildBody` function;
- appends a new user message and empty assistant message to prior history;
- clears the input and sets loading before calling `readSSE`;
- passes the fixed endpoint and built body to the shared SSE reader;
- replaces the current assistant message as events arrive;
- preserves accumulated messages for every Learning Demo.

Ticket 028 needs dynamic endpoint/body selection, pre-fetch local rejection, and Transformer-only replacement of prior messages. A small optional, default-preserving request-preparation seam is preferable to duplicating the entire hook or changing every demo.

### `frontend/src/client/lib/sse.ts`

The shared SSE reader already accepts an endpoint and body for each call. It should not require a production change for Ticket 028. The command boundary should choose the endpoint before `readSSE` is invoked.

### `frontend/package.json`

The project already declares Vitest. No dependency installation or lockfile update is expected. A one-shot script such as `"test": "vitest run"` is a likely small package-script change if the live file still lacks a test command.

### Backend compatibility

The supplied backend snapshot already contains:

- `POST /train-transformer` with the existing five-field training request;
- `POST /load-transformer` with exactly `modelFile`, `prompt`, `temperature`, `topP`, and `maxTokens`;
- `modelFile` accepting a filename or `null` for latest selection;
- the same generation bounds required by Ticket 028.

No backend file belongs in this ticket's production diff.

## Accepted implementation shape

Use the following smallest complete shape unless the live repository reveals an even smaller equivalent that preserves every acceptance criterion.

### Pure Transformer command module

Add one frontend-only module, preferably:

```text
frontend/src/client/lib/transformer-command.ts
```

Expose stable pure behavior for:

- the exact existing numeric training-body construction;
- `File:` classification before numeric parsing;
- exact saved-model grammar validation;
- endpoint and body selection;
- helpful local validation output;
- Transformer request-start message replacement.

A discriminated result should make invalid states unrepresentable as network requests. Conceptually, the result is either:

- a request outcome with `mode`, `endpoint`, and `body`; or
- a validation outcome with assistant content and no endpoint/body.

Use typed training and load body shapes so a load result cannot accidentally include `epochs` or `numLayers`, and a training result cannot include `modelFile` or `prompt`.

### Minimal shared-hook extension

Extend `useSSEChat` through optional configuration with defaults that reproduce current behavior for all existing callers. The preferred concepts are:

- an optional submission-preparation callback that can choose endpoint/body or return local validation;
- an optional message-start callback or history mode that can replace, rather than append, messages.

Keep the static endpoint/body branch available so Simple Chat, BPE, XOR, and Word2Vec require no behavior changes. Prefer a TypeScript union for static versus prepared request configuration so callers cannot provide contradictory request settings.

### Transformer integration

Update only the Transformer hook to use:

- the pure command planner;
- Transformer-only replacement of prior messages at submission start;
- the existing training event state and renderer.

Do not add saved-model event display in this ticket. The temporary load stream may reach the current hook without producing the final Ticket 029 visualization; that known gap is intentionally deferred.

## Acceptance-criteria coverage map

| Acceptance area | Planned evidence |
|---|---|
| Exact numeric command preservation | Pure tests compare endpoint and exact five-field body across full, partial, malformed, zero-like, and extra-token inputs against the current parser rules. |
| Prefix classification | Pure tests cover leading spaces/tabs/newlines and several `File:` capitalization variants; near-matches remain training commands. |
| Exact three-section grammar | Pure tests cover two separators, missing sections, additional separators, and prompt-contained `|`. |
| Selector behavior | Pure tests cover a nonempty selector preserved byte-for-byte and an exactly empty selector mapped to `null`, with no extra flag or magic name. |
| Exact load body | Exact key-set assertion for `modelFile`, `prompt`, `temperature`, `topP`, and `maxTokens`. |
| Prompt normalization | Tests prove only outer whitespace is removed and interior spaces/tabs/characters remain unchanged. |
| Strict settings validation | Tests cover exact count, full-token numeric syntax, finite checks, Boolean-like text, bounds, and integer-only `maxTokens`. |
| No fetch on rejection | Invalid pure result contains no request descriptor; shared hook returns before `readSSE`. Review and focused hook-boundary tests prove the call is unreachable. |
| Helpful local result | Each invalid category returns stable useful text, including a compact usage form. |
| Transformer-only clearing | Pure message-start test proves replacement; unchanged default shared-hook behavior and demo smoke checks prove other histories still append. |
| Pure TypeScript test seam | Tests import command/state functions directly; no JSX rendering, DOM, browser, or real request. |
| Existing UI preserved | No route, component layout, input, button, or CSS changes. |
| Existing tooling only | Use declared Vitest; no new package or framework. |

## Files to inspect before editing

1. `frontend/package.json`
   - Confirm scripts, exact Vitest version, package manager assumptions, and whether a one-shot test script already exists.
2. `frontend/tsconfig.json`
   - Confirm strict mode, module resolution, included files, and whether tests can use explicit imports without adding global Vitest types.
3. `frontend/vite.config.ts`
   - Confirm `/api` proxy rewriting and retain `/api/train-transformer`/`/api/load-transformer` browser endpoints.
4. `frontend/src/client/hooks/use-train-transformer-chat.tsx`
   - Capture the exact current numeric parser, endpoint literal, training state, and imports before moving behavior.
5. `frontend/src/client/hooks/use-sse-chat.ts`
   - Capture the exact send order, message append behavior, loading transitions, error handling, and generic types.
6. `frontend/src/client/lib/sse.ts`
   - Confirm `readSSE` already accepts per-call endpoint and body and identify the clean call boundary.
7. `frontend/src/shared/types/message.ts`
   - Reuse the existing `Message` and role/content types for a pure request-start state helper.
8. `frontend/src/client/components/chat-bubble/index.tsx`
   - Inspect only to confirm how empty assistant content and local string validation content display; do not edit unless live evidence proves necessary.
9. Other demo hooks:
   - `use-simple-chat.ts`
   - `use-bpe-tokenize-chat.tsx`
   - `use-neural-net-chat.tsx`
   - `use-train-embed-chat.tsx`
   - Confirm they rely on the shared hook's current default history behavior.
10. `028-parse-and-route-file-commands-in-the-existing-transformer-input.md`, `SPEC.md`, `CONTEXT.md`, and ADR 0003
    - Reconfirm exact public wording and scope before implementation.
11. `backend/src/how_llms_work/schemas.py`, `backend/src/how_llms_work/main.py`, and the Transformer route files
    - Read-only compatibility inspection; do not modify.

## Step 1 — Establish the live frontend baseline and freeze the current numeric contract

**Files and symbols**

- `frontend/package.json`
- `frontend/tsconfig.json`
- `frontend/vite.config.ts`
- `frontend/src/client/hooks/use-train-transformer-chat.tsx`
- `frontend/src/client/hooks/use-sse-chat.ts`
- Current demo hooks and shared `Message` type

**Purpose**

Prevent a frontend-only routing ticket from silently changing existing numeric training, package tooling, endpoint proxy behavior, or history semantics.

**Actions**

- Run the frontend baseline commands and record exact output.
- Record the exact current Transformer parser expression and produce a small behavior matrix before moving it. Include:
  - all five explicit values;
  - omitted trailing values;
  - malformed tokens that fall back;
  - zero-like values that currently fall back because of `||`;
  - extra sixth and later tokens that are ignored;
  - non-`File:` text that currently follows the numeric parser.
- Confirm that browser endpoints include `/api` and Vite strips it before proxying.
- Confirm the current generic hook appends messages for all demos.
- Check `git status --short` and preserve unrelated user changes.

**Guardrails**

- Do not “improve” the numeric parser in Ticket 028.
- Do not replace current defaults with backend defaults copied from memory; copy the live behavior.
- Do not modify `vite.config.ts` unless the live repository contradicts the supplied proxy evidence.
- Do not run dependency-upgrade commands.

**Expected result**

A written implementation note identifies the exact numeric behavior and frontend baseline that later steps must preserve.

**Verification**

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm typecheck
pnpm lint
pnpm build
pnpm exec vitest run
git status --short
```

## Step 2 — Extract the unchanged numeric training parser into a pure typed boundary

**Files and symbols**

- New: `frontend/src/client/lib/transformer-command.ts`
- Existing: `frontend/src/client/hooks/use-train-transformer-chat.tsx`
- Suggested stable exports:
  - a typed training request body;
  - a pure numeric training-body builder;
  - the final Transformer submission planner.

**Purpose**

Make the existing training behavior directly testable and reusable by the new classifier without changing any numeric semantics.

**Actions**

- Move the exact five-position conversion logic into a pure function.
- Preserve the same whitespace split, `parseInt`/`parseFloat` calls, radix, defaults, `||` fallback behavior, and ignored extra positions.
- Return exactly these fields and no others:
  - `epochs`;
  - `temperature`;
  - `topP`;
  - `numLayers`;
  - `maxTokens`.
- Pair the body with the current browser endpoint `/api/train-transformer` in the final command planner.
- Keep numeric parsing as the fallback for every input not classified as `File:`.

**Guardrails**

- Do not enforce backend ranges locally for numeric training commands in this ticket.
- Do not reject malformed numeric training input that currently falls back to defaults.
- Do not trim, lowercase, normalize, or otherwise reinterpret non-`File:` input beyond the existing parser.
- Do not change the backend public route or request aliases.

**Expected result**

All existing numeric Transformer inputs produce the same browser endpoint and body they produced before extraction.

**Verification**

- Add table-driven pure tests comparing exact objects.
- Include regression cases that distinguish current `||` fallback behavior from stricter number validation.
- Run the focused Vitest file after Step 6 creates it.

## Step 3 — Add strict `File:` classification, grammar parsing, and load request construction

**Files and symbols**

- `frontend/src/client/lib/transformer-command.ts`
- Pure discriminated command/submission result types
- Closed local validation message set

**Purpose**

Recognize saved-model commands before numeric parsing and create an exact safe request descriptor only when the complete local grammar is valid.

**Actions**

### Classification

- Remove or skip only leading whitespace for prefix detection.
- Compare the first five prefix characters to `File:` case-insensitively.
- Perform this check before invoking the numeric parser.
- Do not classify `Files:`, `File =`, text containing `File:` later, or another near-match as a load command.

### Three-section grammar

- Remove the recognized prefix and split the remaining command on the literal `|` separator.
- Require exactly three resulting sections:
  1. file selector;
  2. starting prompt;
  3. generation settings.
- Reject fewer or more sections, including an extra `|` in the prompt.
- Do not add quoting, escaping, backslash rules, or alternate separators.

### File selector

- Preserve a nonempty selector exactly as entered after the prefix and before the first `|`.
- Map only the empty selector section to `modelFile: null`.
- Do not create `useLatest`, an empty filename string, a sentinel, or a magic filename.
- Do not perform browser-side path safety or filename semantic validation beyond the ticket's grammar. The backend owns that trust boundary.

### Prompt

- Trim only leading and trailing whitespace from the prompt section.
- Reject the prompt when the trimmed value is empty.
- Preserve all interior characters, spacing, tabs, punctuation, and capitalization exactly.

### Generation settings

- Trim the settings section and split on whitespace.
- Require exactly three tokens in the order temperature, top-p, and maximum tokens.
- Use strict full-token decimal conversion for saved-model settings rather than `parseFloat`/`parseInt`, because those legacy functions can accept a valid numeric prefix and ignore trailing junk.
- Reject empty, malformed, Boolean-like, NaN-like, infinity-like, and non-finite values.
- Validate:
  - temperature: finite and within `0.1..2.0` inclusive;
  - top-p: finite and within `0.1..1.0` inclusive;
  - maximum tokens: finite integer within `3..500` inclusive.
- Keep strict load parsing separate from the intentionally unchanged numeric training parser.

### Request result

- Return `/api/load-transformer` and a body containing exactly:
  - `modelFile`;
  - `prompt`;
  - `temperature`;
  - `topP`;
  - `maxTokens`.
- Return no request descriptor for invalid commands.
- Return useful stable assistant text. Include a compact usage form and a specific reason where practical without exposing backend details.

**Guardrails**

- Do not silently route a malformed `File:` command to training.
- Do not trim a nonempty selector if “verbatim” remains the live ticket wording.
- Do not use `parseFloat` or `parseInt` for strict load settings.
- Do not add backend validation logic, path rules, or filesystem assumptions to the browser.
- Do not add load event rendering.

**Expected result**

Every submission has exactly one classification. A valid command yields one exact endpoint/body; an invalid `File:` command yields local assistant feedback and no possible network request.

**Verification**

Use pure table-driven tests covering all acceptance categories, including exact key sets and exact string preservation.

## Step 4 — Add a default-preserving request-preparation seam to `useSSEChat`

**Files and symbols**

- `frontend/src/client/hooks/use-sse-chat.ts`
- Existing `UseSSEChatOptions` and `UseSSEChatReturn`
- New optional prepared-submission and message-start types/callbacks
- `frontend/src/shared/types/message.ts`

**Purpose**

Allow one demo to select an endpoint dynamically, reject locally before fetch, and replace history while every other demo keeps current behavior.

**Actions**

- Refactor the options typing into a common section plus mutually exclusive request configuration branches:
  - current static endpoint plus optional body builder; or
  - prepared-submission callback returning request or local validation.
- Retain the current static branch as the default used by existing demos.
- Add an optional message-start callback/history strategy. Its default must remain “append user and assistant messages to previous messages.”
- Call the prepared-submission function before setting loading and before invoking `readSSE`.
- For a local validation outcome:
  - create/update the current assistant result with the supplied useful text;
  - apply the configured message-start strategy;
  - keep loading false;
  - do not invoke `readSSE`;
  - return from `sendMessage`.
- For a valid request outcome:
  - apply the message-start strategy synchronously so stale content disappears immediately;
  - preserve the current input-clearing and loading behavior;
  - pass the prepared endpoint/body to `readSSE`;
  - preserve existing open, event, HTTP-error, exception, and completion handling.
- Keep UUID creation and side effects in the hook; keep command/state decisions pure.

**Guardrails**

- Existing callers must continue compiling without new configuration.
- Do not globally replace history.
- Do not move or duplicate SSE parsing.
- Do not introduce a fetch abstraction or dependency.
- Do not clear messages on `onOpen`; that is too late to satisfy immediate result clearing.
- Do not swallow existing network errors or alter their visible format as unrelated cleanup.

**Expected result**

`useSSEChat` supports Ticket 028 while preserving default request and history behavior for all other demos.

**Verification**

- Type checking proves all existing hook call sites remain valid.
- Pure tests prove the Transformer message-start function returns only the current user/assistant pair, regardless of prior messages.
- Manual demo checks confirm non-Transformer pages still append history.

## Step 5 — Integrate the command planner into the Transformer hook only

**Files and symbols**

- `frontend/src/client/hooks/use-train-transformer-chat.tsx`
- `frontend/src/client/lib/transformer-command.ts`
- Existing training event/state types and `TrainTransformerResult`

**Purpose**

Connect the existing input and Send button to the new pure boundary without changing the page, training display, or other demos.

**Actions**

- Remove the inline numeric `buildBody` implementation only after its exact behavior exists in the pure module.
- Configure `useSSEChat` with the prepared-submission callback.
- Configure Transformer request-start state to replace prior messages/results with the current user submission and current assistant result.
- Ensure the replacement happens for both valid numeric training and valid saved-model commands before the stream begins.
- For local validation, replace stale Transformer content with the submitted command and helpful assistant validation text, with no network request.
- Preserve title, tagline unless a minimal grammar hint is explicitly required by the ticket, existing input, Send button, route page, and layout.
- Preserve current training `init`, `epoch`, and `done` event handling exactly.
- Leave load event display for Ticket 029.

**Guardrails**

- Do not add a Train/Load selector, second input, new page, modal, or layout change.
- Do not modify the Transformer result component for saved-model rendering.
- Do not add load-event types merely to partially implement Ticket 029.
- Do not alter worker-label behavior from Ticket 027.
- Do not change any non-Transformer hook unless required to preserve type compatibility in the shared optional seam.

**Expected result**

The current Transformer input routes numeric and `File:` submissions to the correct browser endpoints, malformed `File:` submissions fail locally, and every new Transformer submission replaces stale Transformer output immediately.

**Verification**

- Type check the frontend.
- Inspect the browser Network panel during the manual acceptance pass.
- Confirm unchanged numeric training still begins and the load request reaches `/api/load-transformer`.

## Step 6 — Add focused Vitest coverage at the approved pure seam

**Files and symbols**

- New: `frontend/src/client/lib/transformer-command.test.ts`
- `frontend/package.json` if a one-shot test script is absent
- Public exports from `transformer-command.ts`

**Purpose**

Prove routing, request construction, validation, no-request rejection, and request-start replacement without rendering a browser component or making a real request.

**Actions**

- Import `describe`, `expect`, `it`/`test` explicitly from `vitest`; do not enable globals or add `vitest/globals` to `tsconfig.json` solely for this ticket.
- Add or confirm a one-shot script using `vitest run`.
- Organize tests by observable behavior rather than private helper names.

### Numeric regression cases

- Exact full command.
- Partial command defaults.
- Malformed-token fallbacks.
- Zero-like current fallback behavior.
- Extra positions ignored.
- Inputs that contain no leading `File:` continue to training.
- Exact endpoint `/api/train-transformer` and exact five-field body.

### Classification cases

- `File:`, `file:`, `FILE:`, and mixed capitalization.
- Leading spaces, tabs, and line breaks.
- Near-matches that must stay on the numeric path.
- Classification occurs before the numeric parser.

### Load request cases

- Named selector preserved exactly.
- Empty selector becomes `null`.
- Exact `/api/load-transformer` endpoint.
- Exact five-key body and no `useLatest`, `epochs`, or `numLayers`.
- Prompt outer trimming and interior preservation.

### Grammar rejection cases

- No separators.
- One separator.
- Extra separator.
- Empty/whitespace-only prompt.
- Missing settings.
- Too few or too many setting tokens.
- Prompt containing `|` through the resulting extra-section rejection.

### Number rejection and bounds

- Non-numeric and Boolean-like text.
- NaN/infinity forms.
- Trailing junk such as a numeric prefix followed by letters.
- Temperature just below/at/above bounds.
- Top-p just below/at/above bounds.
- `maxTokens` below/at/above bounds.
- Fractional `maxTokens` rejected.

### Request-start state

- Prior Transformer messages are discarded for a valid training command.
- Prior Transformer messages are discarded for a valid load command.
- Local validation replaces stale output with useful text.
- Invalid outcome contains no endpoint/body/request descriptor.

**Guardrails**

- Do not render JSX or use a DOM test environment.
- Do not mock browser layout, hook private variables, or component trees.
- Do not call `fetch` or a real server.
- Do not assert a particular regular-expression or helper decomposition.
- Do not add snapshots when exact object/string assertions are clearer.

**Expected result**

Focused deterministic tests cover every Ticket 028 acceptance criterion owned by the frontend command/request-start boundary.

**Verification**

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm exec vitest run src/client/lib/transformer-command.test.ts
pnpm typecheck
pnpm lint
```

## Step 7 — Run full regressions and the two-server manual acceptance pass

**Files and symbols**

- Entire frontend project
- Existing backend route tests and complete backend project
- Live FastAPI and Vite servers
- Browser Transformer and other Learning Demo pages

**Purpose**

Confirm pure behavior integrates with the real proxy/request flow and does not regress completed demos or the Python Backend.

**Actions**

### Frontend full checks

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm test
pnpm typecheck
pnpm lint
pnpm build
```

If the live package uses a different established one-shot test script, use it and record the exact command.

### Backend regression checks

Because no backend file should change, begin with focused contract tests, then run the project's complete gates once before handoff:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"

poetry run pytest tests/test_train_transformer_route.py tests/test_load_transformer_route.py
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
```

Report actual output only. Do not overwrite the user's earlier baseline with an unexecuted claim.

### Start the two-server application

Backend terminal:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"

poetry run uvicorn how_llms_work.main:app `
    --app-dir src `
    --reload `
    --host 127.0.0.1 `
    --port 8000
```

Frontend terminal:

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm dev
```

Open:

```text
http://127.0.0.1:5173/train-transformer
```

### Manual Transformer checks

- Submit a valid numeric command such as `50 1.0 0.6 1 3`.
  - Confirm the browser sends `POST /api/train-transformer`.
  - Confirm the body contains exactly the current five training fields.
  - Confirm previous Transformer output disappears immediately.
- Submit one named command with no formatting ambiguity, for example:
  - `File:transformer-weights-e100-l1-d32-h2-ff128-ctx32.json|once upon a time|0.8 0.9 3`
  - Confirm `POST /api/load-transformer` and the exact five load fields.
- Submit a latest-model command using an exactly empty selector section:
  - `File:|once upon a time|0.8 0.9 3`
  - Confirm `modelFile` is JSON `null`.
- Repeat the named command with `file:`, `FILE:`, and leading whitespace.
- Submit malformed commands for each local validation family.
  - Confirm no `/api/load-transformer` or `/api/train-transformer` request appears in the Network panel.
  - Confirm helpful text replaces stale Transformer output.
- Submit a prompt with repeated interior spaces.
  - Confirm the request preserves interior spacing and trims only the outer edges.
- Confirm no page, selector, second input, button, or layout change appeared.

### Other-demo history checks

- In Simple Chat, submit two messages and confirm both remain.
- Repeat a two-submission smoke check in BPE, XOR, and Word2Vec when practical.
- Confirm only the Transformer page replaces its previous message/result set.

**Guardrails**

- Ticket 029 display gaps are not Ticket 028 failures if the request routes correctly and the current hook does not yet render load events.
- Do not alter backend code to make the manual display look complete.
- Do not modify real saved model files during testing.
- Do not commit generated caches, build output, or `.data` changes.

**Expected result**

All automated checks pass, valid commands route correctly through Vite to FastAPI, invalid commands create no request, Transformer history clears, other demos preserve history, and the existing UI remains unchanged.

**Verification**

Record:

- exact commands and exit codes;
- focused and full test counts;
- type-check/lint/build output;
- browser request endpoints and JSON bodies;
- manual history observations;
- any Ticket 029 display limitation observed.

## Focused verification matrix

| Input category | Expected mode | Expected request | Expected local state |
|---|---|---|---|
| Valid five-number input | training | `/api/train-transformer`; exact training body | Replace prior Transformer messages, then stream training. |
| Partial or malformed non-`File:` input | training | Current fallback body | Preserve existing numeric behavior. |
| Leading-whitespace mixed-case `File:` | load | `/api/load-transformer`; exact load body | Replace prior Transformer messages before stream. |
| Named selector | load | `modelFile` exact nonempty selector | No browser-side filename repair. |
| Exactly empty selector | load | `modelFile: null` | No magic filename or extra flag. |
| Empty prompt | invalid | None | Helpful assistant validation replaces stale output. |
| Wrong pipe count | invalid | None | Usage/section feedback. |
| Wrong setting count | invalid | None | Settings-count feedback. |
| Malformed/non-finite setting | invalid | None | Numeric feedback. |
| Out-of-range setting | invalid | None | Bound-specific feedback. |
| Fractional max tokens | invalid | None | Integer feedback. |

## Full verification order

1. Focused Transformer command Vitest file.
2. Frontend type check.
3. Frontend lint.
4. Frontend full Vitest run.
5. Frontend production build.
6. Focused backend training/load route tests.
7. Complete backend pytest suite.
8. Backend Ruff lint.
9. Backend Ruff format check.
10. Backend strict mypy.
11. Two-server manual proxy and history checks.
12. `git diff --check`, `git diff --stat`, `git status --short`, and scope-only diff review.

## Expected files changed

### Likely production changes

```text
frontend/src/client/lib/transformer-command.ts               # new
frontend/src/client/hooks/use-sse-chat.ts                    # minimal optional prepared-request/history seam
frontend/src/client/hooks/use-train-transformer-chat.tsx     # Transformer-only integration
```

### Likely test/tooling changes

```text
frontend/src/client/lib/transformer-command.test.ts          # new pure Vitest coverage
frontend/package.json                                        # add one-shot test script only if absent
```

### Conditional only if live evidence requires it

```text
frontend/src/client/lib/transformer-message-state.ts          # only if separating state helpers materially improves purity
```

No dependency or lockfile change is expected. If `pnpm-lock.yaml` changes solely because of a script edit, inspect and revert unintended churn.

## Files not to change

```text
backend/src/how_llms_work/main.py
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/sse.py
backend/src/how_llms_work/ml/
backend/src/how_llms_work/routes/
backend/tests/
backend/.data/
frontend/vite.config.ts
frontend/tsconfig.json
frontend/src/client/lib/sse.ts
frontend/src/client/components/train-transformer-result/
frontend/src/client/components/chat-input/
frontend/src/client/components/app/
frontend/src/client/routes.tsx
frontend/src/client/hooks/use-simple-chat.ts
frontend/src/client/hooks/use-bpe-tokenize-chat.tsx
frontend/src/client/hooks/use-neural-net-chat.tsx
frontend/src/client/hooks/use-train-embed-chat.tsx
frontend/src/client/components/
SPEC.md
CONTEXT.md
0003-load-saved-transformer-models-for-stateless-generation.md
028-parse-and-route-file-commands-in-the-existing-transformer-input.md
```

A file may leave this list only when live repository evidence proves a direct Ticket 028 requirement. Document the reason before editing it.

## Risk notes and safeguards

1. **Risk: `File:` still reaches numeric defaults.**
   - **Safeguard:** classify after leading-whitespace handling and before calling the numeric builder; test several capitalization variants and a regression that distinguishes load from defaults.
2. **Risk: numeric training behavior changes during extraction.**
   - **Safeguard:** preserve the exact live expressions and add cases for malformed values, zeros, omitted fields, and extra fields.
3. **Risk: load settings accept trailing junk.**
   - **Safeguard:** use full-token strict conversion plus finite/integer checks; never reuse the permissive training `parseFloat`/`parseInt` path.
4. **Risk: selector normalization violates “verbatim.”**
   - **Safeguard:** preserve every character of a nonempty selector and use an exactly empty section for `null`; leave semantic filename validation to FastAPI.
5. **Risk: browser endpoint omits `/api`.**
   - **Safeguard:** preserve the current Vite proxy convention and test exact browser endpoint strings.
6. **Risk: local rejection still starts a request or spinner.**
   - **Safeguard:** prepare/validate before loading state and return before `readSSE`; invalid results contain no request descriptor.
7. **Risk: Transformer clearing becomes a global history reset.**
   - **Safeguard:** make replacement an optional per-hook strategy whose default is the current append behavior; smoke-check every other demo.
8. **Risk: stale Transformer output remains until HTTP open.**
   - **Safeguard:** replace messages synchronously at submission start, not in `onOpen` or the first event.
9. **Risk: shared-hook refactor creates impossible or ambiguous options.**
   - **Safeguard:** use mutually exclusive typed static/prepared request branches and keep existing call sites unchanged.
10. **Risk: Ticket 029 event rendering enters scope.**
    - **Safeguard:** leave training event handling and result component unchanged; record the temporary display limitation explicitly.
11. **Risk: test setup expands tooling.**
    - **Safeguard:** use explicit Vitest imports and the existing Node environment; no global configuration, DOM emulator, or browser runner.
12. **Risk: package/lockfile churn.**
    - **Safeguard:** add only a script if needed; do not change dependency versions or run upgrade commands.
13. **Risk: backend validation is duplicated incorrectly in TypeScript.**
    - **Safeguard:** validate only command grammar and obvious generation settings locally; transmit selector verbatim and let backend security/model rules remain authoritative.
14. **Risk: local error strings become brittle implementation-detail tests.**
    - **Safeguard:** stabilize a small public message set and assert useful content/categories, not private helper wording beyond approved usage text.
15. **Risk: real training makes manual checks slow.**
    - **Safeguard:** use the smallest supported numeric command for routing smoke and rely on pure tests for parser coverage.

## Commit guidance after all checks pass

Suggested outcome-oriented commit title:

```text
feat: route transformer file commands
```

Suggested commit body topics:

- pure typed numeric/load command classification and exact request construction;
- strict local `File:` grammar and generation-setting validation;
- Transformer-only result replacement with default-preserving shared history behavior;
- focused Vitest coverage using the existing toolchain;
- actual frontend and backend validation commands and observed results;
- saved-model event rendering intentionally deferred to Ticket 029.

Do not create the commit until all required checks pass. Do not include backend source changes, `.data` artifacts, generated build output, caches, dependency upgrades, lockfile churn, specifications, ADRs, or unrelated formatting.

## Handoff to `implement-prompt`

Run `implement-prompt` in a fresh conversation using:

- `plan028.md`;
- `028-parse-and-route-file-commands-in-the-existing-transformer-input.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- `0003-load-saved-transformer-models-for-stateless-generation.md`;
- the latest `ts_llm_pipeline_explorer_file_structure.md` as planning evidence;
- `py_llm_pipeline_explorer_file_structure(146).md` as backend compatibility evidence;
- `llm_works_file_structure.md` as historical behavior reference;
- the live frontend repository as implementation authority.

`implement-prompt` must inspect the live repository again, establish its own frontend baseline, preserve user changes, implement only Ticket 028, keep backend and Ticket 029 behavior out of scope, run focused verification before full checks, report actual outcomes honestly, inspect final scope, and create the implementation commit only after every required check passes.
