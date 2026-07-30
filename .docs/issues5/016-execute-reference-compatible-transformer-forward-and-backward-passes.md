---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "016"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 016: Execute reference-compatible Transformer forward and backward passes

## What to build

Provide the complete reusable decoder-only Transformer mathematics for one ordered Training Sequence and one Logical Training Shard. Preserve the reference architecture, formulas, causal behavior, cache contents needed for analytical backpropagation, mixed-precision contract, repeated-token gradient accumulation, and finite-state enforcement.

This ticket creates the mathematical boundary used identically by direct deterministic tests and spawned workers. It does not decide process scheduling, optimizer timing, HTTP streaming, or persistence.

## Acceptance Criteria

- [ ] Forward computation uses learned token embeddings and learned absolute positional embeddings for each ordered input token position.
- [ ] Repeated Vocabulary Tokens read the same embedding row, and backward computation accumulates every repeated-token contribution into that row rather than overwriting an earlier contribution.
- [ ] Each Transformer block uses pre-normalized attention and feed-forward sublayers with residual connections in the approved order.
- [ ] Layer Normalization operates per position across 32 features, uses population variance and epsilon `1e-5`, and preserves approved gamma and beta behavior.
- [ ] Multi-head causal self-attention uses exactly two heads, head dimension `16`, scale `0.25`, and the approved query, key, value, and output projections.
- [ ] Causal masking makes every future attention probability exactly zero and prevents any future-position gradient contribution.
- [ ] Stable row softmax is used without mutating score matrices or exposing invalid masked rows.
- [ ] The feed-forward sublayer is exactly `32 → 128 → 32` with ReLU and the approved residual connection.
- [ ] The final Layer Normalization and Vocabulary output head produce logits and finite probabilities for every sequence position.
- [ ] Average next-token cross-entropy uses the approved `1e-10` protection term and divides by the sequence length represented by the target positions.
- [ ] The analytical backward pass returns one complete gradient in canonical parameter layout order and follows the reference derivatives for output head, final normalization, residual paths, feed-forward network, attention, positional embeddings, and token embeddings.
- [ ] Materialized weights, activations, caches, logits, probabilities, attention matrices, activation gradients, and parameter gradients are `float32` and C-contiguous where represented as arrays.
- [ ] Scalar statistics and reductions use Python float or NumPy `float64`, with completed tensors materialized explicitly as `float32`.
- [ ] Forward and backward operations leave input weights, token IDs, target IDs, and shared preprocessing unchanged.
- [ ] A Logical Training Shard calculation processes its assigned Training Sequences in fixed order, returns unaveraged canonical gradients plus accumulated shard loss, and returns exact zero loss and all-zero gradients for an empty shard.
- [ ] NaN or infinity in weights, activations, probabilities, loss, or gradients prevents a successful result.
- [ ] Independent fixtures cover selected forward activations, normalization statistics, attention scores and probabilities, logits, loss, backward coordinates, repeated-token accumulation, residual paths, and an empty shard with explicit per-boundary tolerances.
- [ ] Exact assertions cover shapes, dtypes, masks, zero future probabilities and gradients, canonical gradient layout, and public discrete behavior.
- [ ] Sequential and concurrent calls use independent caches and gradients and cannot mutate another call's numerical state.

## Testing Expectations

- **Approved test seam:** The stable public Transformer mathematical boundary for one Training Sequence and one Logical Training Shard.
- **Behavior to verify:** Forward activations, Layer Normalization, causal multi-head attention, residual and feed-forward behavior, output probabilities, cross-entropy, complete analytical backpropagation, repeated-token accumulation, shard gradients, dtype/shape contracts, finiteness, and state isolation.
- **Relevant prior art:** Small independent numerical fixtures and the project's explicit exact-versus-`assert_allclose` comparison policy.
- **Do not test through:** Private cache class names, a specific vectorization expression, loop ordering not required by the approved formulas, BLAS details, process scheduling, or route payloads.

## Blocked By

- [Ticket 013 — Provide strict float32 matrix primitives](013-provide-strict-float32-matrix-primitives.md)
- [Ticket 014 — Produce an immutable Transformer Preprocessing Snapshot](014-produce-immutable-transformer-preprocessing-snapshot.md)
- [Ticket 015 — Build canonical Transformer parameter layouts and initialization](015-build-canonical-transformer-parameter-layouts-and-initialization.md)

## Constraints and Out of Scope

- Preserve the educational decoder-only architecture and analytical gradients; do not introduce automatic differentiation or a machine-learning framework.
- Do not shuffle or batch Training Sequences, average gradients, clip gradients, apply optimizer updates, or generate text in this ticket.
- Do not change the fixed architecture dimensions or causal mask semantics.
- Do not permit hidden broadcasting, mutable shared caches, or float64 materialized model state.
- Do not start processes, expose HTTP behavior, or persist models.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — approved Transformer mathematical seam, formulas, dtypes, comparison policy, and shard behavior.
- `CONTEXT.md` — Transformer Training Sequence, Logical Training Shard, and Transformer Training Compatibility terminology.
- ADR 0002 — forward/backward, mixed precision, causal mask, and finite-state contracts.
- Tickets 013–015 — matrix, preprocessing, and canonical parameter foundations.
- Latest TypeScript Reference Implementation — decoder-only formulas and analytical backward evidence.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
