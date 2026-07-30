---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "017"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 017: Advance Transformer epochs with Ordered Gradient Reduction and Adam

## What to build

Provide the parent-owned numerical state transition for advancing freshly initialized Transformer weights through inclusive training epochs. Consume exactly four validated Logical Training Shard results, combine them only through Ordered Gradient Reduction, apply one deterministic Adam update, enforce finite state, and expose stable report boundaries for later SSE orchestration.

The boundary must be independent of worker completion order and physical worker count. It may be exercised with direct shard calculations or synthetic shard fixtures without starting operating-system processes.

## Acceptance Criteria

- [ ] One epoch transition accepts exactly one result for each Logical Training Shard `0`, `1`, `2`, and `3` and rejects missing, duplicate, out-of-range, malformed, or non-finite results.
- [ ] Shard losses are accumulated separately in Python floating-point order `0 → 1 → 2 → 3`.
- [ ] Canonical gradient blocks are combined through one reusable `float32` reduction workspace strictly in order `0 → 1 → 2 → 3`, independent of the order in which results became available.
- [ ] Gradients remain unaveraged; no division by shard count, Training Sequence count, or worker count is introduced.
- [ ] Empty-shard zero gradients and zero loss do not change the reduced result.
- [ ] The parent is the only owner that modifies weights and Adam state.
- [ ] Adam applies exactly one update per inclusive epoch using learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, and optimizer step `epoch + 1`.
- [ ] First and second moments are stored as `float32` after their completed stages, and updated parameters are stored as `float32` in canonical flat order.
- [ ] Exactly two reusable parent-local `float64` scratch arrays support Adam calculations without becoming persisted or worker-visible state.
- [ ] Training processes epochs inclusively from epoch zero through the exact requested final epoch.
- [ ] Training Sequence order remains fixed, with no shuffling, gradient clipping, weight decay, learning-rate schedule, early stopping, gradient averaging, or intermediate checkpointing.
- [ ] The report schedule uses `max(1, floor(epochs / 50))` and includes epoch zero, every divisible report boundary, and the exact requested final epoch without duplicates.
- [ ] Public report losses use the shared TypeScript-compatible six-decimal rounding helper while internal loss remains available only at its approved precision.
- [ ] Weights, reduced gradients, losses, both Adam moment arrays, and completed update candidates are checked for finiteness before the epoch can complete.
- [ ] A failed reduction or Adam update leaves the previously completed weights and optimizer state unchanged or marks the Training Run failed before later sampling or persistence.
- [ ] Independent fixtures prove that one-through-four simulated physical worker completion orders produce identical reduced gradients, losses, Adam state, weights, and report epochs.
- [ ] Selected moment and parameter coordinates match independently calculated expected values within explicit tight tolerances, while report epochs and six-decimal losses match exactly.
- [ ] Fresh Training Runs never load a Saved Transformer Model and do not share mutable weights, moments, scratch arrays, or report state.

## Testing Expectations

- **Approved test seam:** The stable parent-side epoch transition, Ordered Gradient Reduction, Adam, and report-schedule boundary exercised without HTTP or real processes.
- **Behavior to verify:** Four-shard validation, canonical reduction order, unaveraged gradients, inclusive epochs, exact Adam constants and steps, mixed-precision state, report boundaries, finite-state enforcement, failure transactionality, and worker-count independence.
- **Relevant prior art:** The completed deterministic training-run patterns from earlier phases and Phase 5 independent shard/optimizer fixtures.
- **Do not test through:** Private optimizer helper names, a particular iterator class, worker pipe timing, shared-memory names, route delays, or universal bit-for-bit equality for every unrounded floating-point intermediate.

## Blocked By

- [Ticket 016 — Execute reference-compatible Transformer forward and backward passes](016-execute-reference-compatible-transformer-forward-and-backward-passes.md)

## Constraints and Out of Scope

- Keep all weight and optimizer mutation parent-owned.
- Do not introduce optimizer options, learning-rate configuration, shuffling, batching, clipping, regularization, schedules, early stopping, or model resumption.
- Do not couple reduction order to CPU count, worker assignment, or completion order.
- Do not generate samples, perform final post-training evaluation, construct or persist a Saved Transformer Model, or expose SSE in this ticket.
- Do not start real worker processes; process protocol belongs to later tickets.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — Ordered Gradient Reduction, inclusive training, Adam, reporting, and finite-state decisions.
- `CONTEXT.md` — Transformer Training Run, Transformer Epoch Update, and Ordered Gradient Reduction terminology.
- ADR 0002 — parent-owned updates, exact Adam constants, mixed-precision buffers, and four-shard ordering.
- Ticket 016 — complete shard loss and gradient boundary.
- Latest TypeScript Reference Implementation — optimizer formulas and training intent evidence.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
