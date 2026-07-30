---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "018"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 018: Generate deterministic text and construct Saved Transformer Models

## What to build

Provide the parent-side completion boundary that generates one deterministic Generated Text Sample for a reported epoch, evaluates final loss from final weights, and converts a completed finite Transformer state into the exact JSON-compatible Saved Transformer Model.

Generation must own an independent epoch-seeded Sample Random Stream and preserve the reference latest-context, temperature, stable-softmax, and top-p behavior. Model construction must contain only durable configuration, BPE, Vocabulary, and final parameter data—never process, shared-memory, optimizer, cache, path, or request-lifecycle state.

## Acceptance Criteria

- [ ] Each Generated Text Sample starts from the fixed first three corpus-derived generation seed IDs.
- [ ] Generation uses at most the latest sixteen token IDs as model input while preserving the model's learned positional and causal behavior.
- [ ] Temperature scaling accepts only the validated request range and produces finite logits and probabilities.
- [ ] Top-p nucleus selection uses stable descending probability order, preserves the approved tie behavior, includes the minimum prefix whose cumulative probability reaches the requested topP, and never selects outside that nucleus.
- [ ] Every reported epoch creates a fresh Mulberry32 Sample Random Stream seeded with `(42 + epoch) modulo 2^32`.
- [ ] Sampling for one epoch cannot consume Weight Initialization randomness or change any later epoch's sample stream.
- [ ] Generation emits no more than the requested maxTokens and returns text through the approved token-to-text reconstruction behavior.
- [ ] Generation checks cooperative cancellation between generated tokens and stops without returning a successful sample after cancellation.
- [ ] Selected generated texts for small fixed states and the supported reference environment match independent exact fixtures.
- [ ] Final evaluation runs only after the last Adam update, processes Transformer Training Sequences in fixed order, checks cooperative cancellation between sequences, and computes loss from final weights rather than reusing a pre-update training loss.
- [ ] Final loss is finite and its public form uses the shared six-decimal rounding helper.
- [ ] The Saved Transformer Model preserves top-level insertion order `type`, `config`, `vocab`, `merges`, and `weights`.
- [ ] The nested configuration and weight key order matches the established TypeScript Reference Implementation and canonical parameter layout.
- [ ] The complete ordered Vocabulary and complete ordered Transformer Merge Table are included.
- [ ] Every final parameter coordinate is converted to an ordinary JSON-compatible six-decimal number, with signed zero normalized to positive `0.0`.
- [ ] The complete model rejects NaN and infinity and contains no NumPy scalar, array, memoryview, process object, pipe, shared-memory name, optimizer moment, scratch buffer, activation cache, gradient, path, timestamp, request ID, or checkpoint metadata.
- [ ] Model construction returns fresh plain-Python containers and cannot mutate final weights, shared preprocessing, or data returned by another construction call.
- [ ] Independent exact fixtures verify field sets, key order, configuration, Vocabulary, merges, rounded weights, signed-zero normalization, and complete deterministic serialization-ready contents.

## Testing Expectations

- **Approved test seam:** The stable public generation, final-evaluation, and Saved Transformer Model construction boundary.
- **Behavior to verify:** Epoch-seeded Sample Random Streams, latest-context generation, temperature and stable top-p selection, token limits, cooperative cancellation, final post-update loss, exact model field/key order, six-decimal values, signed-zero normalization, finite-state checks, and isolation.
- **Relevant prior art:** The TypeScript frontend sample contract, existing public model-conversion fixture patterns, and Phase 5 exact generation/model fixtures.
- **Do not test through:** Private sampling helper names, a particular sort implementation, filesystem writes, process timing, route event formatting, or optimizer internals.

## Blocked By

- [Ticket 016 — Execute reference-compatible Transformer forward and backward passes](016-execute-reference-compatible-transformer-forward-and-backward-passes.md)

## Constraints and Out of Scope

- Use one independent Sample Random Stream per reported epoch; do not share a mutable global generator.
- Do not load, resume, cache, or skip training from a Saved Transformer Model.
- Do not write intermediate checkpoints or include transient process state in the final model.
- Do not implement atomic filesystem persistence, worker supervision, HTTP validation, SSE framing, or presentation delays in this ticket.
- Do not change frontend code or add a model-download feature.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — generation, independent sample streams, final evaluation, and Saved Transformer Model decisions.
- `CONTEXT.md` — Generated Text Sample, Sample Random Stream, and Saved Transformer Model terminology.
- ADR 0002 — epoch seed formula, top-p behavior, cooperative cancellation, final-loss order, and model structure.
- Ticket 016 — forward computation used by generation and evaluation.
- Latest TypeScript Reference Implementation — text reconstruction, frontend samples, and model structure evidence.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
