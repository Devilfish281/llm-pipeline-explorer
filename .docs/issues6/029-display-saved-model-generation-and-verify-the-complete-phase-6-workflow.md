---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "029"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 029: Display Saved-Model Generation and Verify the Complete Phase 6 Workflow

## What to build

Finish the learner-visible Phase 6 workflow by connecting the approved Transformer command boundary to the completed backend load route, maintaining saved-model display state separately from training state, consuming the exact load SSE events, and rendering the loaded filename, prompt, completed generated text, or safe error in the existing Transformer result area.

The integrated experience must preserve numeric training, show the worker-process label only in the first training sample, support both named and latest `File:` commands, clear stale Transformer output at request start, and prove the complete FastAPI/Vite interaction through focused automated checks plus the existing two-server smoke seam. This ticket does not create a new page or model-management interface.

## Acceptance Criteria

- [ ] A valid numeric command still calls `POST /train-transformer`, renders the existing architecture, epoch, loss, sample, and summary experience, and differs only by the approved first-sample worker-process label.
- [ ] A valid named `File:` command calls `POST /load-transformer` with the exact parsed body and renders the exact filename returned by `loaded`.
- [ ] A valid empty-selector `File:` command sends `modelFile: null`, loads the newest strictly valid model selected by the backend, and renders that exact returned filename.
- [ ] Saved-model state is typed separately from training state and contains only loaded filename/prompt, completed result text, and an optional safe error; inference events are not coerced into training records.
- [ ] A `loaded` event is accepted only with the approved safe payload shape and updates the display to show `Loaded: <filename>` and the returned trimmed prompt.
- [ ] A `result` event is accepted only with the approved `text` payload and displays one complete generated text value rather than token-by-token output.
- [ ] The loaded result area clearly distinguishes the selected filename, `Prompt:`, and `Generated text:` using the existing component area and layout.
- [ ] A load-route `done` event is terminal and adds no epoch, loss, architecture, sample collection, worker count, or other display data.
- [ ] A safe `error` event displays only its approved message and does not preserve a stale loaded filename, prompt, result, training summary, or prior request output as current state.
- [ ] Load event handling never interprets `loaded`, `result`, or load `done` as training `init`, `epoch`, or training `done` data.
- [ ] No worker-process label is rendered for Saved Transformer Generation Runs.
- [ ] Starting a new numeric or `File:` command clears the previous Transformer result immediately, including previous load errors and completed generated text.
- [ ] The existing shared SSE transport continues to work without a reader-wide redesign unless implementation evidence proves a minimal safe change is necessary.
- [ ] Named-model failures, no-valid-latest failures, empty prompts, unsupported prompts, prompt-length failures, deadline failures, and overlap `429` responses are rendered safely without internal path, exception, model, resource, or numerical details.
- [ ] Existing Health, Simple Chat, BPE, XOR, Word2Vec, Transformer numerical, persistence, worker, worker-group, completion, and route regressions remain preserved.
- [ ] Focused frontend Vitest tests prove event-to-display transitions, typed state separation, exact loaded/result rendering, safe errors, unchanged training rendering, and absence of a loading worker label.
- [ ] Focused backend tests and the complete backend suite are run, followed by `poetry run ruff check .`, `poetry run ruff format --check .`, and `poetry run mypy src`; actual outcomes are recorded without claiming unexecuted success.
- [ ] The frontend's configured type check and focused Vitest tests are run and their actual outcomes are recorded.
- [ ] A practical two-server FastAPI/Vite smoke check records observed results for one numeric training command, one named `File:` command, one latest `File:` command, first-sample worker count, safe load errors, and immediate new-request clearing.
- [ ] The smoke check uses the existing Vite proxy and browser page; no new browser-automation dependency is required.
- [ ] The final integrated workflow remains stateless: each saved-model command selects, reads, validates, generates, displays one result, and retains no server-side model selection, conversation, or loaded-model state afterward.

## Testing Expectations

- **Approved test seam:** Pure TypeScript load-event-to-display-state functions through Vitest, FastAPI `TestClient` regressions for the completed backend contracts, and the focused existing two-server FastAPI/Vite browser smoke seam.
- **Behavior to verify:** Named and latest command integration, exact loaded/result/done/error transitions, typed state separation, approved result display, safe failure rendering, immediate clearing, unchanged training except the worker label, no loading worker label, complete regression preservation, and real Vite-proxy interaction.
- **Relevant prior art:** Current Transformer result component and hook, shared SSE reader and error parser, existing frontend message-state patterns, route-level exact SSE tests, Phase 5 deterministic and lifecycle regressions, and the project's current two-server development workflow.
- **Do not test through:** Browser layout pixels, private component hierarchy, exact CSS classes, private hook variable names, token-by-token UI behavior, private backend helpers, or exact internal thread and scheduling implementation.

## Blocked By

- [Ticket 025 — Load the Newest Strictly Valid Saved Transformer Model](025-load-the-newest-strictly-valid-saved-transformer-model.md)
- [Ticket 026 — Stop and Clean Saved-Model Generation Safely](026-stop-and-clean-saved-model-generation-safely.md)
- [Ticket 027 — Show the Actual Transformer Worker-Process Count During Training](027-show-the-actual-transformer-worker-process-count-during-training.md)
- [Ticket 028 — Parse and Route File Commands in the Existing Transformer Input](028-parse-and-route-file-commands-in-the-existing-transformer-input.md)

## Constraints and Out of Scope

- Reuse the existing Train Transformer page, input box, Send button, result component area, and layout; do not add a selector, second input, model page, or redesign.
- Do not add token-by-token saved-model display, server-side sessions, remembered chat history, a model registry, cache, database, download, deletion, rollback, or management controls.
- Do not resume, continue, fine-tune, or initialize training from a Saved Transformer Model.
- Do not change the Transformer corpus, architecture, optimizer, training algorithm, worker protocol, Logical Training Shards, shared-memory design, or numerical compatibility rules.
- Do not add GPU, CUDA, PyTorch, TensorFlow, JAX, hosted models, LangChain, LangGraph, browser automation, or another production/test dependency.
- Report only validation commands and browser observations that were actually executed during implementation.

## Source

- `SPEC.md` — complete Phase 6 frontend state, event rendering, compatibility, regression, validation, and smoke-test decisions.
- `GRILL_WITH_DOCS_RESULT.md` — confirmed learner-facing display and safety behavior.
- `CONTEXT.md` — canonical Transformer Training Run, Saved Transformer Generation Run, and event-stream terminology.
- ADR 0003 — stateless inference boundary and minimal frontend consequence.
- [Ticket 025](025-load-the-newest-strictly-valid-saved-transformer-model.md), [Ticket 026](026-stop-and-clean-saved-model-generation-safely.md), [Ticket 027](027-show-the-actual-transformer-worker-process-count-during-training.md), and [Ticket 028](028-parse-and-route-file-commands-in-the-existing-transformer-input.md).
- Latest Python Backend source export and TypeScript/Vite frontend reference supplied with the source specification.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
