---
workflow: engineering-prompt-chain
document_type: implementation_plan
prompt_name: to-plan-prompt
status: ready-for-implementation
version: 1
plan_number: "029"
source_work_item: 029-display-saved-model-generation-and-verify-the-complete-phase-6-workflow.md
source_specification: SPEC.md
source_context: CONTEXT.md
architecture_decision: 0003-load-saved-transformer-models-for-stateless-generation.md
backend_code_reference: py_llm_pipeline_explorer_file_structure.md
frontend_code_source_of_truth: ts_llm_pipeline_explorer_file_structure(16).md
behavior_reference: llm_works_file_structure.md
baseline_test_status: user-reported
recommended_next_prompt: implement-prompt
---

# Plan for Issue 029: Display Saved-Model Generation and Verify the Complete Phase 6 Workflow

## Initial checklist

- Confirm Ticket 029 is the only selected work item and that its four blockers are represented by the current sources: latest-valid model selection, safe saved-model lifecycle handling, first-training-sample worker-process presentation, and `File:` command parsing/routing.
- Treat `ts_llm_pipeline_explorer_file_structure(16).md` as the current frontend source of truth for planning. Re-inspect the live repository before editing and preserve any newer user changes.
- Treat the latest `py_llm_pipeline_explorer_file_structure.md` as backend compatibility evidence. Ticket 029 should not change backend production code unless live verification exposes a direct contract defect that prevents the approved frontend integration.
- Preserve the Ticket 028 command planner and Transformer-only immediate message replacement. Do not reopen command grammar, numeric parsing, request-body construction, Vite proxy routing, or result-clearing decisions unless a failing acceptance test demonstrates a defect.
- Make the smallest safe transport adjustment needed to retain named SSE event identity for Transformer streams. Do not redesign all SSE consumers or change the existing payload-only mode used by other Learning Demos.
- Introduce separately typed training and Saved Transformer display state, exact event-shape guards, and pure event-to-state transitions before wiring JSX rendering.
- Keep the existing Train Transformer page, input, Send button, result area, and layout. Add no selector, second input, page, model-management control, token stream, or dependency.
- Finish with focused frontend tests, the configured frontend checks, focused and complete backend regressions, the established Poetry/Ruff/mypy gates, and a practical two-server FastAPI/Vite smoke pass. Report only observed outcomes.

## Source-of-truth hierarchy

1. The user's latest direction: run `to-plan-prompt` for Ticket 029 using the supplied current Python Backend and TypeScript/Vite frontend exports.
2. `029-display-saved-model-generation-and-verify-the-complete-phase-6-workflow.md` for the immediate scope, acceptance criteria, approved test seams, blockers, constraints, and completion evidence.
3. The live TypeScript/Vite frontend repository at implementation time.
4. `ts_llm_pipeline_explorer_file_structure(16).md` for current frontend modules, exact hook behavior, command-planning integration, message replacement, result rendering, shared SSE reader behavior, tests, package scripts, and Vite proxy configuration.
5. The live Python Backend repository at implementation time, with the latest `py_llm_pipeline_explorer_file_structure.md` as current compatibility evidence for `POST /train-transformer`, `POST /load-transformer`, exact SSE payloads, safe errors, worker labeling, lifecycle behavior, and regression commands.
6. `SPEC.md` for durable Phase 6 decisions about separate frontend state, named SSE-event discrimination, exact display fields, result clearing, no loading worker label, no new dependency, and the combined automated/manual test seam.
7. `CONTEXT.md` for the canonical distinction between a Transformer Training Run and a Saved Transformer Generation Run, plus the meanings of Transformer Event Stream and Saved Transformer Event Stream.
8. ADR 0003 for the separate stateless inference endpoint, `loaded → result → done` success sequence, safe `error` behavior, no inference worker processes, and minimal frontend consequence.
9. Completed Tickets 025 through 028 and their current implementations as prerequisite evidence only; do not reimplement their concerns in Ticket 029.
10. `llm_works_file_structure.md` only as historical behavior evidence when it agrees with the accepted Phase 6 specification and current code. A current direct copy was not available in this planning handoff, so it is not used to invent missing behavior.
11. Official MDN Server-Sent Events documentation, TypeScript narrowing/discriminated-union documentation, Vitest documentation, and Vite proxy documentation only as technical cross-checks. Project contracts remain authoritative.
12. Older exports, generated `dist` files, prior plans, screenshots, and assumptions are non-authoritative when they conflict with the selected ticket or current source.

## Work-item summary and scope correction

Ticket 029 is the final **frontend integration and complete workflow verification** ticket for Phase 6. The broader project goal remains migration of backend behavior from TypeScript to Python, but the Python Saved Transformer Generation Run is already implemented. This ticket should not translate additional backend code. It should make the current TypeScript/Vite frontend faithfully consume and render the Python Backend's completed event contract.

The current command boundary already distinguishes numeric training from case-insensitive `File:` commands, sends the correct browser endpoint and request body, rejects malformed load commands locally, and clears previous Transformer messages at request start. The current backend already returns successful saved-model streams in this form:

```text
event: loaded
data: {"file":"<exact selected filename>","prompt":"<trimmed prompt>"}

event: result
data: {"text":"<complete prompt-plus-continuation>"}

event: done
data: {}
```

The remaining gap is downstream of request routing:

- the shared frontend JSON SSE mode discards every `event:` line;
- the Transformer hook receives only the parsed `data:` object;
- the hook recognizes only training payload shapes;
- `loaded`, `result`, and empty load `done` payloads therefore produce no content;
- the current Transformer result component can render only training state.

The smallest complete solution is to preserve named SSE envelopes in a new opt-in JSON-envelope mode, use that mode only for the Transformer hook, reduce exact named events into a typed display-state union with training and saved-model branches, and render the saved-model branch in the existing result area.

## Baseline evidence

### Status

User-reported. No command was executed by this planning workflow.

### User-reported backend baseline

The user previously reported successful results for:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Record these as user-reported only. The implementation run must establish its own current baseline before editing.

### User-reported frontend evidence

The user previously supplied output indicating:

- the focused/full Vitest suite completed with 56 passing tests;
- `pnpm typecheck` completed successfully;
- `pnpm build` completed successfully;
- `pnpm lint` could not run because the project did not contain the required ESLint flat configuration file.

Do not silently fix or broaden into ESLint configuration work under Ticket 029. Re-run the configured command and record its actual current result separately from the feature checks.

### User-observed integration evidence

A manual request through the Vite proxy returned HTTP `200`, `text/event-stream`, and the exact successful sequence `loaded → result → done`, including a complete generated result. This proves the backend route and proxy path work and proves that event-name loss in the current frontend reader is an actual integration gap rather than a speculative concern.

### Planning rule

The implementation run must:

1. inspect `git status --short` in both projects;
2. preserve unrelated user changes;
3. run and record the current frontend and backend baselines before editing;
4. distinguish a known missing lint configuration from a feature regression;
5. never state that a check passed unless the command was actually run and succeeded.

## Current code observations from the latest source

### Ticket 028 command and request-start behavior already exists

`frontend/src/client/lib/transformer-command.ts` already provides a pure typed submission planner that:

- preserves the existing five-number training parser;
- classifies a case-insensitive `File:` prefix before numeric parsing;
- validates the three-section load grammar;
- produces `/api/train-transformer` or `/api/load-transformer` descriptors;
- maps an empty selector to `modelFile: null`;
- builds the exact five-field load request;
- returns local validation text with no request descriptor on malformed load input.

`frontend/src/client/lib/transformer-command.test.ts` already contains extensive pure tests. Ticket 029 should retain them unchanged unless a real regression is found.

`frontend/src/client/hooks/use-sse-chat.ts` already supports:

- a dynamic `prepareSubmission` callback;
- local validation before `readSSE`;
- a per-demo `startMessages` strategy;
- Transformer-only replacement of prior user/assistant messages;
- default append behavior for all other Learning Demos.

`frontend/src/client/hooks/use-train-transformer-chat.tsx` already uses `planTransformerSubmission` and `replaceTransformerMessages`, satisfying the immediate stale-result-clearing boundary for numeric, valid load, and locally invalid load submissions.

### The shared JSON SSE mode loses required event identity

`frontend/src/client/lib/sse.ts` currently has two modes:

- `json`, which ignores `event:` lines and parses each `data:` line as a standalone JSON value;
- `multiline`, which preserves an event name and raw joined data text in an envelope.

The Transformer hook uses `json`. This works for older training handling because the hook infers event meaning from payload fields. It does not safely support the Saved Transformer Event Stream:

- `loaded` data becomes only `{file, prompt}`;
- `result` data becomes only `{text}`;
- `done` data becomes `{}` and cannot identify itself as terminal;
- `error` data cannot be distinguished by event name from another one-key object without relying on loose payload inference.

Changing the existing `json` mode globally would risk every current JSON SSE consumer. The minimal safe change is a new opt-in mode that preserves the named envelope while parsing JSON data.

### The Transformer hook contains training state only

`use-train-transformer-chat.tsx` currently defines one training-oriented state containing initialization data, epoch updates, generated training samples, and final summary data. Its event union contains training payloads only. Event handling discriminates by payload properties such as:

- `vocabSize` and `totalParams` for initialization;
- `epoch` for progress;
- `architecture` for training completion.

Load events do not satisfy those checks and return `undefined`, so the assistant message stays empty and `ChatBubble` continues showing loading dots.

### The current result component renders only training output

`frontend/src/client/components/train-transformer-result/index.tsx` accepts training initialization, epochs, samples, and summary props. It has no saved-model branch, no loaded filename, no prompt section, no generated-text section, and no load error branch.

Its current sample text styling preserves line breaks, which is compatible with Ticket 027's first-sample worker-process prefix. Ticket 029 must not strip, parse, duplicate, or synthesize that prefix.

### Backend behavior needed by Ticket 029 is already present

The current backend evidence includes:

- separate `POST /train-transformer` and `POST /load-transformer` routes;
- named and latest model selection;
- strict saved-model validation and request-owned snapshots;
- exact successful load events;
- safe load/prompt/generation/deadline errors;
- the shared immediate overlap `429` contract;
- no training worker group for loading;
- first-training-sample worker count as presentation-only text;
- unchanged raw samples, `done.samples`, deterministic generation, and persisted models;
- focused route, loading, lifecycle, numerical, worker, worker-group, completion, and persistence tests.

No backend production change is expected for Ticket 029.

## Acceptance criteria coverage

### Already satisfied and evidenced

- Valid numeric and `File:` commands are classified before request dispatch.
- Named load requests send the exact parsed filename.
- Empty selectors send `modelFile: null`.
- Browser requests use the existing `/api` Vite proxy convention.
- A new Transformer submission replaces prior Transformer messages immediately.
- Other Learning Demos retain default append-style history behavior.
- The backend supports named and latest selection and emits exact successful load events.
- Saved-model generation is stateless and starts no training worker processes.
- The backend training path supplies the first-sample worker-process label without contaminating raw samples, final sample history, or persisted models.
- Backend focused and complete regression suites already exist for later execution.

### Behavior present but evidence incomplete

- Numeric training rendering should remain compatible with the worker label because sample text is displayed with preserved whitespace, but the final two-server acceptance pass must confirm the exact first-sample presentation and one-occurrence rule in the current browser.
- The Vite proxy is configured and one manual load request succeeded, but the complete browser rendering path for named/latest/error states has not yet been demonstrated.
- Transformer-only result clearing is implemented, but the final smoke pass must confirm it visually for completed generated text and load errors.

### Partially implemented

- The shared SSE reader can already preserve an event name in multiline mode, but its JSON mode discards event names and no named-JSON envelope mode exists.
- The Transformer hook already receives dynamically routed requests and training events, but it lacks saved-model event types, exact payload guards, saved-model state, and load rendering.
- The result component already owns the correct page area and styling conventions, but it has no saved-model branch.
- Existing error infrastructure handles HTTP failures and thrown reader errors, but safe SSE `error` events from `/load-transformer` are not transformed into the approved saved-model display state.

### Not implemented

- A typed Saved Transformer display state separate from training state.
- Exact named-envelope preservation for JSON Transformer events.
- Strict acceptance of `loaded`, `result`, load `done`, and safe `error` payloads.
- A pure event-to-display-state transition seam.
- Loaded filename and prompt rendering.
- Complete generated-text rendering.
- Safe load error rendering that clears all stale training/load state.
- Explicit proof that load `done` adds no display data.
- Focused frontend tests for saved-model transitions and unchanged training rendering.
- The final complete Phase 6 two-server smoke record.

### Evidence limitations

- The live repositories were not mounted as editable checkouts during planning; implementation must re-inspect live files.
- No current direct copy of `llm_works_file_structure.md` was available in this planning handoff.
- No command was run during planning.
- Browser pixels and private JSX structure are intentionally outside the approved test seam.

## Files to inspect before editing

1. `frontend/package.json`
   - Confirm current scripts, Vitest availability, and whether no dependency or script change is needed.
2. `frontend/tsconfig.json`
   - Confirm strictness, module resolution, and test inclusion.
3. `frontend/vite.config.ts`
   - Confirm the existing `/api` proxy and no required change.
4. `frontend/src/client/lib/sse.ts`
   - Inspect mode typing, chunk buffering, event-block parsing, JSON parsing, HTTP error handling, and final-buffer behavior.
5. Any existing frontend SSE tests
   - Reuse current stream/fetch controls if present rather than inventing a browser test harness.
6. `frontend/src/client/hooks/use-sse-chat.ts`
   - Confirm the mode option, `onEvent` contract, prepared submission flow, message replacement, and error behavior. Prefer no change unless the new envelope mode cannot be selected by the Transformer hook alone.
7. `frontend/src/client/hooks/use-train-transformer-chat.tsx`
   - Inspect current training types, state mutation, event discrimination, renderer creation, command planner, and message replacement.
8. `frontend/src/client/components/train-transformer-result/index.tsx`
   - Inspect the existing training renderer and the smallest location for a saved-model renderer or discriminated prop union.
9. `frontend/src/client/components/train-transformer-result/styles.module.css`
   - Reuse existing label, section, mono-text, and wrapping conventions; add only minimal saved-result styles.
10. `frontend/src/client/components/chat-bubble/index.tsx`
    - Confirm empty assistant content is the sole cause of loading dots; no change expected.
11. `frontend/src/client/lib/parse-error.ts`
    - Inspect HTTP `429` handling. Modify only if exact safe display of a string FastAPI `detail` value requires a small general parser extension.
12. `frontend/src/client/lib/transformer-command.ts` and its tests
    - Regression inspection only; Ticket 029 should consume, not redesign, this boundary.
13. Other demo hooks using `useSSEChat`
    - Confirm a new opt-in SSE mode does not change their default behavior.
14. `backend/src/how_llms_work/routes/train_transformer.py`, `backend/src/how_llms_work/sse.py`, and `backend/tests/test_load_transformer_route.py`
    - Read-only contract confirmation for exact event names, payload key sets, safe errors, overlap behavior, and no worker label on loading.
15. `backend/tests/test_train_transformer_route.py`
    - Read-only confirmation of the first-sample label and unchanged training payloads.
16. Ticket 029, `SPEC.md`, `CONTEXT.md`, and ADR 0003
    - Reconfirm public wording and scope immediately before implementation.

## Step 1 — Establish the live baseline and freeze the exact Transformer event contracts

**Files and symbols:**

- Frontend and backend `package`/project configuration.
- `frontend/src/client/lib/sse.ts`.
- `frontend/src/client/hooks/use-train-transformer-chat.tsx`.
- `backend/src/how_llms_work/routes/train_transformer.py`.
- Existing frontend and backend tests.

**Purpose:**

Prevent the final integration ticket from changing already completed command routing, training rendering, backend payloads, or unrelated demos while solving only the missing event/display boundary.

**Actions:**

- Run `git status --short` in the repository root or both project directories and record unrelated modifications.
- Run the current frontend baseline before editing:

  ```powershell
  pnpm exec vitest run
  pnpm typecheck
  pnpm lint
  pnpm build
  ```

- Record the current lint result exactly; do not add ESLint configuration under Ticket 029 unless the user separately approves it.
- Run focused backend contract tests before editing:

  ```powershell
  poetry run pytest tests/test_train_transformer_route.py tests/test_load_transformer_route.py -q
  ```

- Inspect the live backend tests/source and record the exact public event names and key sets for:
  - training `init`;
  - training `epoch`;
  - training `done`;
  - load `loaded`;
  - load `result`;
  - load `done`;
  - load `error`.
- Confirm that load `done` is an empty JSON object and therefore cannot be safely identified after event-name removal.
- Confirm the current Transformer command tests pass before changing event/display code.
- Preserve the exact current numeric training renderer output and sample strings as the regression baseline.

**Guardrails:**

- Do not edit code while establishing the baseline.
- Do not infer exact event fields from older plans when live backend tests define them.
- Do not change backend event payloads to compensate for frontend parsing.
- Do not treat the known lint-configuration absence as a Ticket 029 feature failure.

**Expected result:**

A concise implementation record establishes the exact pre-change behavior, identifies named event preservation as the only required transport gap, and confirms that command routing/backend generation are prerequisite behavior rather than Ticket 029 implementation work.

**Verification:**

Record command, exit code, test count, and relevant output. No success statement belongs in this plan.

## Step 2 — Add one opt-in named JSON-envelope mode to the shared SSE reader

**Files and symbols:**

- `frontend/src/client/lib/sse.ts`.
- New or existing focused SSE test file, preferably `frontend/src/client/lib/sse.test.ts`.
- A small exported envelope type or parser seam if needed for direct pure testing.

**Purpose:**

Preserve the backend's `event:` field for Transformer streams while parsing `data:` as JSON, without changing the existing payload-only mode used by other Learning Demos.

**Actions:**

- Add a third explicit SSE mode conceptually equivalent to `json-envelope`.
- Define one typed envelope containing:
  - the exact event name as a string;
  - the parsed JSON data value.
- Parse complete SSE event blocks rather than treating every `data:` line as an unrelated event.
- Preserve the current chunk buffer so split network chunks and CRLF/LF boundaries remain safe.
- Collect all `data:` lines belonging to one event according to existing SSE framing, join them consistently, and JSON-decode the resulting data text.
- Emit exactly one envelope per completed event block.
- Flush a final complete buffered event when the stream ends, following the current reader's established end-of-stream behavior.
- Keep the existing `json` mode behavior byte-for-byte compatible for Simple Chat/BPE/XOR/Word2Vec/training consumers that still use it.
- Keep the existing `multiline` behavior unchanged.
- Do not alter request dispatch, headers, error parsing, `onOpen`, or response status handling.
- Add focused tests proving:
  - `event: loaded` plus JSON data becomes one `{event, data}` envelope;
  - `result`, empty-object `done`, and `error` retain their exact event names;
  - multiple events in one chunk are emitted in order;
  - one event split across chunks is reconstructed once;
  - CRLF and LF separators are accepted according to current support;
  - existing payload-only JSON mode still ignores event names and returns the same parsed values;
  - existing multiline mode remains unchanged.

**Guardrails:**

- Do not replace or reinterpret the existing `json` mode globally.
- Do not make every demo handle envelopes.
- Do not create a general EventSource abstraction or new dependency.
- Do not add token-by-token behavior.
- Do not expose raw event blocks to the Transformer hook when valid JSON parsing succeeds.
- Do not overfit tests to private local-variable names or a particular loop decomposition.

**Expected result:**

The shared reader gains one backward-compatible opt-in mode that accurately represents the Python Backend's named JSON SSE contract. Existing consumers retain their current behavior.

**Verification:**

```powershell
pnpm exec vitest run src/client/lib/sse.test.ts
pnpm typecheck
```

Use the actual discovered test filename if the live repository already has an SSE test module.

## Step 3 — Create a pure typed Transformer event-to-display-state boundary

**Files and symbols:**

- New: `frontend/src/client/lib/transformer-event-state.ts`.
- New: `frontend/src/client/lib/transformer-event-state.test.ts`.
- Existing training event types from `use-train-transformer-chat.tsx`, moved or shared only as needed.

**Purpose:**

Make training and saved-model state structurally separate, validate named event payloads at one pure boundary, and make every display transition deterministic and directly testable without JSX, a DOM, or a real request.

**Actions:**

- Define a named Transformer SSE envelope type consumed by the reducer.
- Define separate typed state branches, for example conceptually:
  - an initial/empty state;
  - a training state containing only training initialization, epochs, raw samples, and optional summary;
  - a saved-model state containing only:
    - loaded exact filename and returned prompt;
    - optional complete generated text;
    - optional safe error.
- Use a discriminated union so saved-model events cannot populate training arrays/summary and training events cannot populate loaded-model fields.
- Keep `done` terminality out of the visible data model unless a private non-rendered discriminator is necessary. The public saved state must not gain epoch, loss, architecture, sample collection, worker count, or model metadata.
- Add exact, closed payload guards for each event:
  - training `init` with its current approved field set;
  - training `epoch` with its current approved field set;
  - training `done` with its current approved field set;
  - `loaded` with exactly `file` and `prompt`, both strings;
  - `result` with exactly `text`, a string;
  - load `done` with an empty object;
  - `error` with exactly `error`, a string.
- Treat an event as load `done` only when the event name is `done`, the current state is the saved-model branch, and the payload is the approved empty object.
- Treat a training `done` only when the event name and training summary shape both match.
- Preserve the returned `loaded.prompt` exactly; do not re-trim or normalize backend output.
- Preserve the complete `result.text` exactly; do not split or stream it.
- On safe `error`:
  - replace the entire current Transformer display state with a saved-model error state;
  - retain only the approved error string;
  - remove prior loaded filename, prompt, generated result, training initialization, epochs, samples, and summary from current state.
- Ignore or reject unknown event names and malformed payloads without coercing them into another event kind. Do not show raw payloads or parsing details.
- Preserve each training epoch sample string exactly, including the first worker-process label and blank line supplied by the backend.
- Keep reducer operations immutable or use an explicit testable update contract; do not rely on JSX rendering to define state correctness.

**Guardrails:**

- Do not use broad property-presence inference once named envelopes are available.
- Do not define one bag-of-optional-fields state that permits training and load data to coexist accidentally.
- Do not synthesize or parse a worker count in the frontend.
- Do not add a worker label to saved-model text.
- Do not accept extra fields as harmless if the ticket requires the approved exact safe shape.
- Do not derive expected values by calling the renderer under test.

**Expected result:**

One pure public seam converts exact Transformer SSE envelopes into one of two independent display-state families. The seam is exhaustive enough that load `done {}` is unambiguous and safe `error` replacement cannot leave stale data.

**Verification:**

Add table-driven Vitest coverage for:

- complete training `init → epoch → done` state progression;
- exact preservation of a first training sample containing `Transformer worker processes: N\n\n...`;
- `loaded` creating only saved-model filename/prompt state;
- `result` adding one complete generated text value;
- load `done {}` making no visible state change;
- `error` replacing loaded/result state;
- `error` replacing prior training state;
- latest selection displaying the actual filename from `loaded`, not `null` or the submitted selector;
- malformed/extra-key payload rejection;
- no worker-label field or synthesized text in saved-model state;
- training and load events not crossing state branches.

Run:

```powershell
pnpm exec vitest run src/client/lib/transformer-event-state.test.ts
pnpm typecheck
```

## Step 4 — Render Saved Transformer state in the existing result area

**Files and symbols:**

- `frontend/src/client/components/train-transformer-result/index.tsx`.
- `frontend/src/client/components/train-transformer-result/styles.module.css`.
- Pure display-state types from `transformer-event-state.ts`.

**Purpose:**

Show the selected filename, returned prompt, complete generated text, or safe error clearly in the current Transformer assistant area while leaving the existing training experience intact.

**Actions:**

- Preserve the current training renderer and its props/labels unless converting it to one branch of a discriminated render union produces a smaller type-safe integration.
- Add a saved-model renderer in the same component area, either as:
  - a separately exported `SavedTransformerResult`; or
  - a saved-model branch of one discriminated `TransformerResult` component.
- For a successfully loaded state, render:

  ```text
  Loaded: <exact filename returned by loaded>

  Prompt:
  <returned prompt>
  ```

- When complete result text exists, additionally render:

  ```text
  Generated text:
  <complete result.text>
  ```

- Show the filename, prompt, and generated text as distinct semantic sections using the current visual language, not a new page or panel system.
- Preserve prompt and generated-text whitespace with appropriate wrapping. Do not HTML-interpret, tokenize, truncate, or alter the strings.
- On a saved-model error, render only the safe error message. Do not render stale filename, prompt, generated text, training architecture, epochs, samples, or summary.
- Render no worker-process label or worker-count field for a saved-model state. If generated text naturally contains those words, display the backend result as ordinary text rather than adding presentation semantics.
- Keep load `done` invisible; it should not add a completion heading or empty section.
- Add only the minimal CSS module classes needed for section spacing, labels, mono/pre-wrapped text, and safe error presentation.

**Guardrails:**

- Do not add a second result component area to the page layout.
- Do not test or depend on exact CSS class names or pixels.
- Do not add a progress spinner or token stream after `result`.
- Do not alter existing training summary labels, epoch list, sample text, or architecture presentation.
- Do not parse the worker label out of the first training sample.

**Expected result:**

The existing assistant result area can render either the current training experience or the approved Saved Transformer result experience, with no mixed state and no layout redesign.

**Verification:**

- Use the pure state tests as the main automated proof.
- Add a small component-level pure assertion only if the current Hono JSX/test tooling already supports it without a DOM dependency; otherwise rely on type checking plus the manual smoke seam as approved by the ticket.
- Run:

```powershell
pnpm typecheck
pnpm build
```

## Step 5 — Integrate named envelopes and typed state in the Transformer hook only

**Files and symbols:**

- `frontend/src/client/hooks/use-train-transformer-chat.tsx`.
- `frontend/src/client/lib/sse.ts` new mode.
- `frontend/src/client/lib/transformer-event-state.ts`.
- `frontend/src/client/components/train-transformer-result/index.tsx`.
- `frontend/src/client/hooks/use-sse-chat.ts` only if the live mode typing requires a minimal generic adjustment.

**Purpose:**

Connect numeric training and saved-model streams to the new exact event boundary while preserving Ticket 028 routing and every other Learning Demo.

**Actions:**

- Configure only the Transformer hook to use the new named JSON-envelope mode.
- Change the Transformer event generic from payload-only training objects to the typed named envelope accepted by the pure reducer.
- Replace the current loose property checks in the hook with one call to the pure event/state transition boundary.
- Keep one small mutable holder only if required by `useSSEChat`'s current state callback contract; place the actual branch state inside it and replace it through the pure reducer.
- Render the reducer's current branch:
  - training branch → existing training result renderer;
  - saved-model loaded/result branch → saved-model renderer;
  - saved-model error branch → saved-model error renderer;
  - initial/unrecognized event → no display update.
- Preserve the current `planTransformerSubmission` callback unchanged.
- Preserve `replaceTransformerMessages` unchanged so all new numeric/load/local-validation submissions remove stale Transformer content immediately.
- Ensure `loaded` returns visible content immediately, replacing loading dots with filename and prompt before generation completes.
- Ensure `result` replaces/updates the same current assistant message with one complete text value.
- Ensure load `done` causes no additional visible content and no training summary.
- Ensure safe SSE `error` updates the current assistant message instead of throwing or appending a second stale message.
- Keep HTTP errors such as overlap `429` on the existing shared HTTP error path. Inspect `parse-error.ts` and add support for a safe string-valued FastAPI `detail` only if the current UI otherwise displays raw JSON and the change can preserve all existing error behavior.
- Do not modify non-Transformer hooks. Their mode and payload-only event handling must remain unchanged.

**Guardrails:**

- Do not add load state to the training state interface.
- Do not make command mode a global or server-retained selection.
- Do not reinterpret `loaded`, `result`, or empty load `done` through training field checks.
- Do not change `useSSEChat` message history defaults.
- Do not duplicate the SSE reader inside the Transformer hook.
- Do not modify the Vite proxy or call port `8000` directly from browser code.

**Expected result:**

A single existing Transformer page now supports both complete workflows:

- numeric command → named training envelopes → unchanged training renderer;
- `File:` command → named saved-model envelopes → filename/prompt/complete-result or safe-error renderer.

All other demos continue using the existing SSE modes and history behavior.

**Verification:**

```powershell
pnpm exec vitest run `
    src/client/lib/transformer-command.test.ts `
    src/client/lib/sse.test.ts `
    src/client/lib/transformer-event-state.test.ts

pnpm typecheck
pnpm build
```

Replace paths with the actual live filenames where necessary.

## Step 6 — Complete focused frontend acceptance coverage

**Files and symbols:**

- `frontend/src/client/lib/transformer-event-state.test.ts`.
- `frontend/src/client/lib/sse.test.ts`.
- Existing `frontend/src/client/lib/transformer-command.test.ts`.
- Conditional error-parser test if `parse-error.ts` changes.

**Purpose:**

Prove every Ticket 029 frontend acceptance criterion through stable pure behavior, without browser layout assertions or a new testing dependency.

**Actions:**

Build a focused acceptance matrix that covers:

### Named and latest integration inputs

- Retain Ticket 028 tests proving exact named request body.
- Retain Ticket 028 tests proving empty selector becomes `modelFile: null`.
- Add event-state tests proving the displayed filename always comes from `loaded.file`, including a latest request whose submitted selector was `null`.

### Exact successful load transitions

- `loaded` accepts only exact `{file, prompt}` data.
- Loaded state exposes only returned filename and prompt.
- `result` accepts only exact `{text}` data.
- Result state contains one complete string, not token fragments or an array.
- `done {}` makes no visible mutation and introduces no training fields.
- Event order remains `loaded → result → done` in transport tests.

### Safe failures

- Safe `error` from an empty current state displays only its message.
- Safe `error` after `loaded` clears filename and prompt.
- Safe `error` after `result` clears generated text.
- Safe `error` after training clears training state.
- Representative messages for named-load, latest-none, prompt, deadline, and generation failures pass through exactly as safe strings without adding internal details.
- HTTP `429` parser behavior is tested only if a parser change is required.

### Training preservation

- Existing training `init`, `epoch`, and `done` events still produce the same state values.
- The first epoch sample string containing the worker-process label and exactly one blank line is preserved unchanged.
- Later samples remain unchanged and no frontend prefix is added.
- Final `done.samples` values are rendered as supplied and no load fields enter training state.

### No loading worker label

- Saved-model state has no worker-count property.
- Loaded/result transition functions never prefix `Transformer worker processes`.
- The saved-model renderer receives only filename, prompt, result text, and optional error.

### State clearing and isolation

- Retain Ticket 028 `replaceTransformerMessages` tests.
- Confirm a new request begins with a fresh per-stream reducer state rather than the previous training/load branch.
- Confirm sequential saved-model streams do not reuse filename, prompt, result, or error state.

### Transport compatibility

- Existing payload-only JSON mode behavior remains tested.
- Existing multiline mode behavior remains tested.
- Other demo hooks compile unchanged.

**Guardrails:**

- Do not render a browser DOM or assert exact CSS.
- Do not mock private hook variables.
- Do not use a real backend in pure Vitest tests.
- Do not use snapshots when exact value assertions are clearer.
- Do not add browser automation.

**Expected result:**

The focused frontend test suite directly proves event identity, typed state separation, exact display values, safe replacement, unchanged training data, and absence of a synthesized loading worker label.

**Verification:**

```powershell
pnpm exec vitest run src/client/lib/transformer-command.test.ts
pnpm exec vitest run src/client/lib/sse.test.ts
pnpm exec vitest run src/client/lib/transformer-event-state.test.ts
pnpm typecheck
```

Then run the complete configured test command:

```powershell
pnpm test
```

## Step 7 — Run backend regressions and the complete two-server Phase 6 smoke pass

**Files and symbols:**

- Entire frontend project.
- Entire backend project.
- Existing FastAPI/Uvicorn and Vite development servers.
- Existing Train Transformer page and other Learning Demos.

**Purpose:**

Prove the final frontend integrates with the real Python Backend and Vite proxy while preserving every completed Phase 1 through Phase 6 behavior.

**Actions:**

### Frontend checks

From `frontend/` run and record actual output:

```powershell
pnpm test
pnpm typecheck
pnpm lint
pnpm build
```

If `pnpm lint` still fails solely because no ESLint configuration exists, record that exact pre-existing tooling limitation. Do not describe the overall frontend feature checks as fully green without distinguishing it.

### Focused backend checks

From `backend/` run:

```powershell
poetry run pytest `
    tests/test_train_transformer_route.py `
    tests/test_load_transformer_route.py `
    tests/test_transformer_loading.py `
    -q
```

### Complete backend checks

```powershell
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
```

### Start the existing two-server application

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

Open the existing Train Transformer page through Vite.

### Manual numeric training check

- Submit a small valid numeric command such as:

  ```text
  50 1.0 0.6 1 3
  ```

- Confirm the browser sends `POST /api/train-transformer` through the Vite proxy.
- Confirm prior Transformer output disappears immediately at submission.
- Confirm the existing architecture, epoch, loss, samples, and final summary remain visible.
- Confirm the first displayed training sample begins with:

  ```text
  Transformer worker processes: <1..4>
  ```

- Confirm exactly one blank line separates the label from raw generated text.
- Confirm no later training sample contains the label.

### Manual named Saved Transformer check

- Submit a valid named command using an existing current-format file, for example:

  ```text
  File:transformer-weights-e50-l1-d32-h2-ff128-ctx32.json|once upon a time|0.8 0.9 3
  ```

- Confirm the browser sends `POST /api/load-transformer` with exactly the five load fields.
- Confirm prior output disappears immediately.
- Confirm `Loaded:` shows the exact filename returned by the backend.
- Confirm `Prompt:` shows the returned trimmed prompt.
- Confirm `Generated text:` shows one complete result.
- Confirm no training architecture, epoch, loss, sample history, summary, or worker label is displayed.

### Manual latest Saved Transformer check

- Submit:

  ```text
  File:|once upon a time|0.8 0.9 3
  ```

- Confirm the request body contains JSON `modelFile: null`.
- Confirm the display uses the actual filename returned by `loaded`, not `null`, an empty string, or a guessed newest filename.
- Confirm the rest of the result display matches the named path.

### Manual safe-error checks

Use safe scenarios that do not mutate real model artifacts:

- a canonical-looking but nonexistent named filename;
- an unsupported prompt for an existing model;
- a prompt that exceeds the approved token limit;
- a direct malformed/empty prompt case where the frontend or backend owns the rejection;
- overlap by starting one longer Transformer request and submitting the other route while it is active.

For each:

- confirm only a safe message is shown;
- confirm no path, traceback, exception text, model values, token IDs, resource identifiers, or numerical state appears;
- confirm stale filename, prompt, generated result, epochs, samples, and summary are absent from the current display;
- confirm a later valid request can succeed.

The five-minute deadline remains primarily automated backend evidence; do not wait five real minutes during the manual smoke pass.

### Immediate clearing check

- After a completed load result, submit a new numeric command and observe the old filename/prompt/text disappear immediately.
- After a training result, submit a load command and observe the old training summary disappear immediately.
- After a load error, submit another command and observe the old error disappear immediately.

### Other-demo regressions

- Confirm Simple Chat, BPE, XOR, and Word2Vec still open and submit requests through their existing paths.
- Confirm their default message-history behavior remains unchanged.

### Final scope inspection

Run:

```powershell
git diff --check
git diff --stat
git status --short
```

Inspect the complete diff and reject generated `dist`, cache, `.data`, dependency, lockfile, backend, or unrelated UI changes.

**Guardrails:**

- Do not edit or delete real saved models to manufacture latest-selection failures.
- Do not wait five minutes for a deadline test.
- Do not accept a successful curl response alone as proof of browser rendering.
- Do not treat one browser observation as a substitute for pure state tests or backend regressions.
- Do not claim a known lint configuration failure passed.

**Expected result:**

The complete Phase 6 workflow is visibly functional through the existing page and Vite proxy, all focused and complete automated checks have recorded outcomes, and the final diff remains limited to the missing frontend event/display boundary.

**Verification:**

Record exact commands, exit codes, test counts, selected filenames, displayed prompt/text, worker count, safe messages, and any remaining tooling limitation.

## Focused verification plan

### Frontend

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm exec vitest run src/client/lib/transformer-command.test.ts
pnpm exec vitest run src/client/lib/sse.test.ts
pnpm exec vitest run src/client/lib/transformer-event-state.test.ts
pnpm typecheck
pnpm build
```

Expected result:

- Existing command-routing tests remain green.
- Named JSON-envelope parsing tests pass.
- Exact training/load event-to-state tests pass.
- TypeScript compilation and production build complete successfully.

Do not claim those results until observed.

### Backend

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"

poetry run pytest `
    tests/test_train_transformer_route.py `
    tests/test_load_transformer_route.py `
    tests/test_transformer_loading.py `
    -q
```

Expected result:

- Training event/worker-label, saved-model event/error/lifecycle, named/latest selection, and no-worker regressions remain green.

## Full verification plan

### Frontend

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\frontend"

pnpm test
pnpm typecheck
pnpm lint
pnpm build
```

Expected result:

- All configured tests pass.
- Type checking and build pass.
- Lint outcome is recorded honestly; the known missing-configuration result is not concealed or fixed out of scope.

### Backend

```powershell
cd "C:\Users\ME\Documents\Python\2026\Built_an_LLM_From_Scratch\projects\llm-pipeline-explorer\backend"

poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
```

Expected result:

- Complete backend tests, lint, formatting check, and strict type check succeed.

Do not claim these future outcomes in advance.

## Manual acceptance checklist

### Numeric training

- [ ] A valid numeric command sends `POST /api/train-transformer` through Vite.
- [ ] Prior Transformer content clears immediately.
- [ ] Architecture, epochs, losses, samples, and final summary render as before.
- [ ] The first public sample begins with the exact worker-process label.
- [ ] The displayed count is between `1` and `4` and is the backend-selected count.
- [ ] Exactly one blank line separates the label from raw generated text.
- [ ] No later displayed sample contains the label.

### Named Saved Transformer generation

- [ ] A named `File:` command sends `POST /api/load-transformer` with the exact five-field body.
- [ ] `Loaded:` displays the exact filename returned by `loaded`.
- [ ] `Prompt:` displays the backend-returned trimmed prompt.
- [ ] `Generated text:` displays one complete result.
- [ ] No training architecture, epoch, loss, sample collection, summary, or worker label is present.
- [ ] Load `done` adds no visible data.

### Latest Saved Transformer generation

- [ ] Empty selector sends JSON `modelFile: null`.
- [ ] The display uses the actual backend-selected filename.
- [ ] Prompt and generated text render identically to named selection.

### Safe failures and lifecycle

- [ ] Named-model failure displays only its safe message.
- [ ] No-valid-latest behavior remains covered by backend tests and, where practical, a safe controlled manual environment.
- [ ] Unsupported-prompt and overlength-prompt failures display only safe messages.
- [ ] Overlap returns/rendered `429` safely and does not queue a request.
- [ ] Deadline behavior remains covered by controlled backend tests rather than a five-minute manual wait.
- [ ] An error clears stale filename, prompt, result, epochs, samples, and summary.
- [ ] A later valid request succeeds after each practical failure.

### Clearing and compatibility

- [ ] Starting a numeric command clears prior load results/errors immediately.
- [ ] Starting a load command clears prior training results immediately.
- [ ] Starting another load command clears prior load results/errors immediately.
- [ ] Simple Chat, BPE, XOR, and Word2Vec retain their established behavior.
- [ ] No page, selector, second input, new layout, token stream, or browser automation was added.

## Expected files changed

### Likely production changes

```text
frontend/src/client/lib/sse.ts
frontend/src/client/lib/transformer-event-state.ts                  # new
frontend/src/client/hooks/use-train-transformer-chat.tsx
frontend/src/client/components/train-transformer-result/index.tsx
frontend/src/client/components/train-transformer-result/styles.module.css
```

### Likely test changes

```text
frontend/src/client/lib/sse.test.ts                                 # new or existing
frontend/src/client/lib/transformer-event-state.test.ts              # new
```

### Conditional only if live evidence requires it

```text
frontend/src/client/hooks/use-sse-chat.ts                            # only for minimal new-mode typing/selection support
frontend/src/client/lib/parse-error.ts                               # only to render safe string FastAPI detail cleanly
frontend/src/client/lib/parse-error.test.ts                          # only if parser changes
```

No backend production or test file is expected to change. Backend files should change only if focused tests reveal a genuine pre-existing public-contract defect that blocks Ticket 029; document that evidence before editing.

## Files not to change

```text
backend/src/how_llms_work/
backend/tests/
backend/.data/
backend/pyproject.toml
backend/poetry.lock
frontend/src/client/lib/transformer-command.ts
frontend/src/client/lib/transformer-command.test.ts
frontend/src/client/hooks/use-simple-chat.ts
frontend/src/client/hooks/use-bpe-tokenize-chat.tsx
frontend/src/client/hooks/use-neural-net-chat.tsx
frontend/src/client/hooks/use-train-embed-chat.tsx
frontend/src/client/components/chat-input/
frontend/src/client/routes.tsx
frontend/vite.config.ts
frontend/tsconfig.json
frontend/package.json
frontend/pnpm-lock.yaml
frontend/dist/
SPEC.md
CONTEXT.md
0003-load-saved-transformer-models-for-stateless-generation.md
029-display-saved-model-generation-and-verify-the-complete-phase-6-workflow.md
```

A file may leave this list only when live source or a failing acceptance test proves it is directly necessary. Record the reason before editing.

## Risk notes and safeguards

1. **Risk: a global SSE change breaks every demo.**
   - **Safeguard:** add an opt-in named JSON-envelope mode and retain existing `json` and `multiline` semantics with regression tests.
2. **Risk: event names are still discarded for Transformer streams.**
   - **Safeguard:** configure the Transformer hook explicitly for the new mode and test empty-object `done` identity.
3. **Risk: training and load fields coexist in one permissive state object.**
   - **Safeguard:** use a discriminated state union with separate training and saved-model branches.
4. **Risk: load `done {}` is mistaken for no-op garbage or training completion.**
   - **Safeguard:** classify by event name, current branch, and exact payload shape; add a terminal no-visible-change test.
5. **Risk: loose payload guards accept internal or unexpected fields.**
   - **Safeguard:** validate approved exact key sets and primitive types before state transitions.
6. **Risk: a safe load error leaves stale successful or training data visible.**
   - **Safeguard:** replace the entire current branch with error-only saved-model state.
7. **Risk: the frontend synthesizes a worker count for loading.**
   - **Safeguard:** saved-model state contains no worker property; render only backend filename, prompt, text, or safe error.
8. **Risk: the frontend changes the first training sample.**
   - **Safeguard:** preserve every epoch sample string byte-for-byte; do not parse or format the worker label.
9. **Risk: named/latest display uses submitted selector instead of backend selection.**
   - **Safeguard:** populate visible filename only from `loaded.file` and test latest request with a different actual filename.
10. **Risk: result is treated as token fragments.**
    - **Safeguard:** accept one exact `result` event with one complete `text` string and render it once.
11. **Risk: new request clearing regresses other demos.**
    - **Safeguard:** retain Ticket 028's Transformer-only `startMessages` strategy and leave shared defaults unchanged.
12. **Risk: HTTP `429` exposes raw JSON or internals.**
    - **Safeguard:** reuse the safe server detail; conditionally extend the existing error parser only for string `detail`, with regression tests.
13. **Risk: a malformed stream leaks a raw parser exception into the UI.**
    - **Safeguard:** keep closed event guards and existing generic transport error handling; do not render raw event payloads.
14. **Risk: tests depend on JSX hierarchy or CSS.**
    - **Safeguard:** test the pure reducer/view state, then use typecheck/build and the existing manual browser seam for rendering.
15. **Risk: generated `dist` files enter the diff.**
    - **Safeguard:** inspect `git status`, exclude build output, and review `git diff --stat` before commit.
16. **Risk: backend code changes unnecessarily.**
    - **Safeguard:** treat backend as a verified contract provider and make frontend changes first; require a focused failing backend test before any backend edit.
17. **Risk: known frontend lint configuration is expanded into unrelated tooling work.**
    - **Safeguard:** rerun and report the current lint result without adding a new configuration in this ticket.
18. **Risk: implementation claims complete Phase 6 without a real proxy/browser check.**
    - **Safeguard:** record numeric, named, latest, error, worker-label, and clearing observations through the existing two-server setup.

## Commit guidance after all checks pass

Do not create a commit during `to-plan-prompt`.

Use the repository's established outcome-oriented convention.

Suggested subject:

```text
Display saved Transformer generation results
```

The commit body should mention:

- one opt-in named JSON-envelope SSE mode with unchanged existing modes;
- exact preservation of `loaded`, `result`, load `done`, and `error` event identity;
- separately typed training and saved-model display state;
- exact loaded filename, returned prompt, and complete generated-text rendering;
- error-only replacement with no stale training/load state;
- unchanged numeric training rendering and byte-for-byte first-sample worker label;
- no loading worker label, token stream, new page, selector, dependency, backend change, or session state;
- retained Ticket 028 command routing and immediate clearing;
- focused Vitest, frontend typecheck/build/lint result, focused/full backend pytest, Ruff, mypy, and two-server observations actually executed;
- any known tooling limitation reported honestly.

## Handoff to `implement-prompt`

Run `implement-prompt` in a fresh conversation using:

- this `plan029.md`;
- `029-display-saved-model-generation-and-verify-the-complete-phase-6-workflow.md`;
- `SPEC.md`;
- `CONTEXT.md`;
- ADR 0003;
- the latest live frontend repository and `ts_llm_pipeline_explorer_file_structure(16).md`;
- the latest live Python Backend repository and `py_llm_pipeline_explorer_file_structure.md`;
- completed Ticket 028 command-planner tests and the current backend Transformer route tests.

`implement-prompt` must re-inspect the live repositories, preserve user changes, establish its own baseline, implement only Ticket 029, add a minimal opt-in named JSON-envelope mode, create pure typed event-to-display-state transitions, preserve exact training strings and state, render saved-model results in the existing area, run focused and complete verification, record the two-server smoke observations, inspect final scope, and create a commit only after every required check has an honestly reported outcome.
