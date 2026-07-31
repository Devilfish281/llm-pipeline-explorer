---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "028"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 028: Parse and Route File Commands in the Existing Transformer Input

## What to build

Add a pure, typed frontend command boundary that classifies the existing Transformer input as either the unchanged numeric training command or the new saved-model command before any request is sent. It must recognize case-insensitive `File:` after leading whitespace, validate the exact three-section grammar, preserve the existing numeric training body, construct the exact named or latest load body, select the correct endpoint, and provide helpful local feedback for malformed commands.

The same request-start boundary must clear the previous Transformer result immediately for either a valid training command or a valid saved-model command, while preserving the message-history behavior of other Learning Demos. The current Train Transformer page, input box, Send button, and layout remain in place.

## Acceptance Criteria

- [ ] Existing valid numeric input continues to select `POST /train-transformer` and constructs the exact current five-field training request without changing defaults, field names, or number conversion behavior.
- [ ] After leading whitespace is ignored, `File:`, `file:`, `FILE:`, and other capitalization variants select `POST /load-transformer` before the numeric parser is applied.
- [ ] A saved-model command contains exactly three `|`-separated sections: file selector, starting prompt, and generation settings.
- [ ] A nonempty file selector is sent verbatim as `modelFile`; an empty selector is represented as `modelFile: null` and no `useLatest` flag or magic filename is created.
- [ ] The load body contains exactly `modelFile`, `prompt`, `temperature`, `topP`, and `maxTokens`.
- [ ] Only leading and trailing prompt whitespace is removed; interior characters and spacing are preserved exactly.
- [ ] A missing or whitespace-only prompt is rejected locally before fetch with helpful text in the existing Transformer result area.
- [ ] Missing command sections, extra command sections, missing generation settings, or an extra `|` separator are rejected locally before fetch; no escaping grammar is introduced.
- [ ] Generation settings require exactly temperature, top-p, and maximum tokens and reject missing, extra, malformed, non-finite, or Boolean-like values before fetch.
- [ ] Temperature outside `0.1..2.0`, top-p outside `0.1..1.0`, and maximum tokens outside integer range `3..500` are rejected locally before fetch.
- [ ] A locally rejected command performs no network request and leaves a useful usage or validation result in the current assistant-result area.
- [ ] A command without the `File:` prefix continues through the existing training parser rather than being silently reclassified as saved-model loading.
- [ ] Starting any valid Transformer training or load command clears the previous Transformer messages/results immediately before the new stream begins.
- [ ] Result clearing is scoped to the Transformer Learning Demo and does not change history behavior for Simple Chat, BPE, XOR, or Word2Vec.
- [ ] The command parser and endpoint/body selection are exposed as pure TypeScript behavior that can be tested without rendering a browser component or making a fetch request.
- [ ] The change reuses the current Train Transformer page, input box, Send button, and layout; no selector, second input, new page, or layout redesign is introduced.
- [ ] No new frontend dependency or test framework is added; the configured Vitest runner is used.

## Testing Expectations

- **Approved test seam:** Pure TypeScript Transformer command-classification, parsing, endpoint-selection, request-body, validation-result, and request-start state functions exercised through the configured Vitest runner.
- **Behavior to verify:** Exact numeric-command preservation; case-insensitive and leading-whitespace `File:` routing; named and `null` selector bodies; verbatim selector handling; prompt outer trimming and interior preservation; exact section and range validation; no fetch on local rejection; helpful local output; and immediate prior-result clearing.
- **Relevant prior art:** The current Transformer hook's five-field body construction, existing SSE chat hook request flow, shared frontend error parser, current message/result state pattern, and the project's declared Vitest tooling.
- **Do not test through:** Browser layout pixels, private hook variable names, exact component tree structure, real network calls, a specific regular-expression implementation, or internal parser helper decomposition.

## Blocked By

- None — can start immediately.

## Constraints and Out of Scope

- Keep the existing page, input, Send button, result area, and overall layout.
- Do not implement backend loading, model validation, generation, SSE lifecycle, or worker labeling in this ticket.
- Do not allow `|` inside the starting prompt and do not add escaping.
- Do not accept arbitrary filesystem paths beyond transmitting the verbatim selector for backend semantic validation.
- Do not alter other Learning Demos, introduce browser automation, or add a dependency.
- Do not yet complete saved-model event rendering; that cross-stack display is Ticket 029.

## Source

- `SPEC.md` — command routing, grammar, request construction, prompt normalization, local validation, result clearing, and frontend test seam.
- `GRILL_WITH_DOCS_RESULT.md` — confirmed minimal frontend change and display constraints.
- `CONTEXT.md` — canonical Frontend Contract and Saved Transformer Generation Run terminology.
- ADR 0003 — existing-input command form and separate endpoint.
- Latest TypeScript/Vite frontend reference supplied with the source specification.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
