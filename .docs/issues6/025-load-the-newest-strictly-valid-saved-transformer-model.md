---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "025"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 025: Load the Newest Strictly Valid Saved Transformer Model

## What to build

Complete the `modelFile: null` path so a learner can request the newest usable Saved Transformer Model without knowing a filename. The existing load endpoint must enumerate only approved ordinary model candidates, apply deterministic newest-first ordering, validate candidates through the exact trust boundary from Ticket 023, skip invalid candidates only for this latest-selection path, and run the same deterministic generation and SSE workflow established by Ticket 024.

Latest selection must remain safe and predictable. A damaged newest artifact must not block an older valid model, equal modification times must have a stable filename tie-break, and the route must never rewrite invalid candidates or weaken the named-file no-fallback rule.

## Acceptance Criteria

- [ ] A structurally valid request with `modelFile: null` is accepted by the same public load request contract and uses the same prompt and generation validation as a named request.
- [ ] Latest selection considers only ordinary non-link, non-junction files inside the genuine model directory whose exact names match the approved Transformer persistence grammar.
- [ ] Matching candidates are ordered by descending modification time and then descending exact filename.
- [ ] Candidates are validated in that exact order through the same strict current-format boundary used by named selection, and the first strictly valid candidate is selected.
- [ ] If the newest candidate is unsafe, unreadable, malformed, damaged, incompatible, or filename/configuration mismatched, latest selection skips it and examines the next candidate without exposing the failure details.
- [ ] If multiple valid candidates have equal modification times, the alphabetically greatest exact filename is selected deterministically.
- [ ] If no matching candidate exists or every matching candidate is invalid, the stream emits one `error` event containing exactly `No valid saved Transformer model was found.`
- [ ] A successful latest request emits the exact selected filename in `loaded`, then follows the same deterministic `loaded → result → done` contract and payload key sets as a successful named request.
- [ ] The selected file is read once for that request, generation uses the validated in-memory snapshot, and a later request repeats enumeration, read, and validation rather than using cached state.
- [ ] Invalid candidates remain byte-for-byte untouched by inspection; latest selection does not delete, repair, rewrite, rename, quarantine, or update metadata intentionally.
- [ ] Safe latest-selection errors expose no internal path, rejected filename, exception, traceback, model value, or validation detail.
- [ ] Named requests continue to validate only the named file and never inherit latest-selection fallback behavior.
- [ ] Candidate ordering and tie behavior are independent of directory enumeration order and are stable across repeated requests with unchanged metadata and contents.

## Testing Expectations

- **Approved test seam:** The specification's public model-selection boundary with temporary directories, followed by FastAPI `TestClient` HTTP/SSE tests for the `modelFile: null` route path.
- **Behavior to verify:** Candidate filtering, newest-to-oldest ordering, descending-filename ties, invalid-candidate skipping, no-valid-model message, exact selected filename in `loaded`, named-path non-fallback regression, one-read/no-cache behavior, artifact immutability, and the inherited successful SSE sequence.
- **Relevant prior art:** Temporary persistence-directory tests, controlled modification-time fixtures, exact filename tests, strict model-validation fixtures, exact SSE parsing, and deterministic generation tests from Tickets 023 and 024.
- **Do not test through:** Private sorting helper names, the filesystem's raw enumeration order, internal exception classes, private validation decomposition, or exact implementation data structures.

## Blocked By

- [Ticket 024 — Stream Deterministic Generation from an Exact Saved Model](024-stream-deterministic-generation-from-an-exact-saved-model.md)

## Constraints and Out of Scope

- `modelFile: null` means latest valid model; do not introduce a `useLatest` flag or magic filename value.
- Skip invalid candidates only for automatic latest selection; never substitute another model after a named request fails.
- Do not add a model registry, manifest, database, cache, rollback mechanism, repair operation, file-size cap, or model-management UI.
- Do not follow symbolic links or Windows junctions and do not accept arbitrary paths.
- Do not change generation mathematics, prompt rules, SSE payloads, training behavior, or frontend layout in this ticket.

## Source

- `SPEC.md` — latest selection ordering, invalid-candidate handling, tie-break, no-valid-model message, and no-cache decisions.
- `GRILL_WITH_DOCS_RESULT.md` — confirmed latest-model user path and safeguards.
- `CONTEXT.md` — Saved Transformer Model and Saved Transformer Generation Run relationships.
- ADR 0003 — newest strictly valid model selection semantics.
- [Ticket 023](023-safely-load-exact-saved-transformer-model-snapshots.md) and [Ticket 024](024-stream-deterministic-generation-from-an-exact-saved-model.md).

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
