---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "024"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 024: Stream Deterministic Generation from an Exact Saved Model

## What to build

Make one complete named-file Saved Transformer Generation Run available through `POST /load-transformer`. A valid structured request must select the exact model snapshot supplied by Ticket 023, tokenize one trimmed starting prompt with that model's own ordered Merge Table and Vocabulary, reuse the existing decoder-only inference mathematics in the backend parent process, and stream one deterministic complete prompt-plus-continuation result.

The route must remain distinct from Transformer training. It owns request validation handoff, shared Transformer slot reservation for the basic named workflow, safe semantic-error mapping, exact SSE sequencing, and final release for the outcomes introduced in this slice. It must not create a Request-Scoped Worker Group or reinterpret inference as training progress.

## Acceptance Criteria

- [ ] `POST /load-transformer` is registered without removing or renaming any completed backend endpoint.
- [ ] The public request contains exactly `modelFile`, `prompt`, `temperature`, `topP`, and `maxTokens`, with snake_case internal attributes and the approved camelCase aliases.
- [ ] `modelFile` accepts one strict nonempty string for this named path; malformed types, numeric strings, Booleans, non-finite values, fractional integers, missing required fields, and values outside temperature `0.1..2.0`, top-p `0.1..1.0`, or maximum tokens `3..500` fail with HTTP `422` before slot reservation or model access.
- [ ] A valid named request selects only the exact filename through Ticket 023; an unsafe, missing, differently capitalized, unreadable, malformed, damaged, or incompatible named model never falls back to another file.
- [ ] A named-model selection or validation failure emits one safe `error` event containing exactly `The saved Transformer model could not be loaded.` and no successful load events.
- [ ] The route trims only the prompt's leading and trailing whitespace and preserves all interior characters and spacing.
- [ ] Empty or whitespace-only prompts emit one `error` event with exactly `The prompt must not be empty.`
- [ ] The complete trimmed prompt is tokenized with the loaded model's ordered Merge Table and Vocabulary, and every produced token must resolve to a model token ID.
- [ ] Unsupported prompt text is never dropped, replaced, normalized, or silently defaulted and emits exactly `The prompt contains text that this saved Transformer model cannot tokenize.`
- [ ] Prompts containing one through sixteen model tokens are accepted; prompts containing seventeen or more emit exactly `The prompt must contain no more than 16 tokens.`
- [ ] `loaded` is emitted only after path safety, complete model validation, canonical parameter materialization, prompt tokenization, and prompt-length validation have all succeeded.
- [ ] The `loaded` payload contains exactly `file` and `prompt`, using the selected exact filename and trimmed prompt.
- [ ] Saved-model inference reuses the current decoder-only forward pass, causal context behavior, latest-sixteen-token context rule, stable softmax, temperature, top-p sampling, Vocabulary decoding, and generation bounds without introducing a hosted model or ML framework.
- [ ] Every Saved Transformer Generation Run uses a new request-owned Mulberry32 stream seeded exactly with `42`; identical validated model bytes and identical request values reproduce identical completed text.
- [ ] Generation adds up to `maxTokens` new tokens and emits one `result` whose payload contains exactly `text` and whose text begins with the trimmed original prompt followed by its continuation.
- [ ] After `result`, the route emits exactly one `done` event with an empty JSON object and no epoch, loss, architecture, sample collection, worker information, or model data.
- [ ] The only successful event order is `loaded → result → done`; no `init`, `epoch`, token-level event, training field, or duplicate successful event is emitted.
- [ ] A semantic or internal generation failure emits one closed-mapping safe `error` string and does not expose raw exception text, traceback, paths, model arrays, resource identifiers, or numerical state.
- [ ] Saved-model generation occurs in the backend parent process and creates no Request-Scoped Worker Group, child process, pipe, queue, manager, shared-memory block, or worker-process label.
- [ ] Loaded model state is request-owned and discarded when this named request stream ends; it is never training state, a checkpoint, or a cross-request cache.

## Testing Expectations

- **Approved test seam:** FastAPI's public HTTP/SSE seam through the in-process `TestClient`, using the existing exact SSE parser, temporary model directories, deterministic model fixtures, and controlled generation collaborators; include one bounded integration through the real public loading and generation boundaries.
- **Behavior to verify:** Exact route and request contract, standard `422`, exact named selection, prompt validation, model-owned tokenization, deterministic seed `42`, complete result text, exact payload key sets, `loaded → result → done`, safe errors, state disposal, and absence of training workers and training events.
- **Relevant prior art:** Existing Transformer route tests, exact SSE event parsing, Pydantic strict request tests, deterministic generation fixtures, public Saved Transformer Model construction, and route registration regressions.
- **Do not test through:** Private route helper names, local coroutine decomposition, exact internal sampler loops, private tensor views, exact thread identity, or model weight values beyond stable public deterministic evidence.

## Blocked By

- [Ticket 023 — Safely Load Exact Saved Transformer Model Snapshots](023-safely-load-exact-saved-transformer-model-snapshots.md)

## Constraints and Out of Scope

- Training remains at `POST /train-transformer`, initializes fresh weights, and never loads a Saved Transformer Model.
- Do not implement latest-model selection in this ticket; `modelFile: null` is completed by Ticket 025.
- Do not complete the full deadline, disconnect, cancellation, and all-outcome slot lifecycle hardening reserved for Ticket 026.
- Do not stream tokens, start training workers, cache a model, resume training, maintain conversation state, or add sessions.
- Preserve the shared SSE media type and headers, public camelCase fields, current numerical intent, and all completed endpoints.
- Do not modify the frontend or add a new dependency.

## Source

- `SPEC.md` — dedicated load request, named model semantics, prompt validation, deterministic generation, SSE payloads, and safe messages.
- `CONTEXT.md` — canonical Saved Transformer Generation Run and Saved Transformer Event Stream terminology.
- ADR 0003 — separate deterministic stateless inference endpoint.
- [Ticket 023](023-safely-load-exact-saved-transformer-model-snapshots.md) — trusted exact model snapshot boundary.
- Latest Python Backend source export and TypeScript Reference Implementation supplied with the source specification.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
