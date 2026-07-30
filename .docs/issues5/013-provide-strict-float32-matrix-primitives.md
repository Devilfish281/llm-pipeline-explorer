---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "013"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 013: Provide strict float32 matrix primitives

## What to build

Provide the stateless, shape-checked NumPy matrix boundary required by the educational Transformer. The boundary must make dtype, shape, contiguity, aliasing, finiteness, masking, accumulation precision, and mutation rules explicit so later Transformer mathematics can rely on one stable numerical contract.

Pure operations return independent C-contiguous `float32` arrays and leave inputs unchanged. Only explicitly named in-place operations may mutate a separate non-overlapping destination, and those mutations must be transactional so a failed calculation never leaves a partially modified destination.

## Acceptance Criteria

- [ ] Every public matrix operation validates the exact supported ranks and shapes and rejects accidental broadcasting except for the approved row-bias operation.
- [ ] Numerical inputs and materialized outputs use NumPy `float32`; unsupported dtypes are rejected or deliberately converted only at the documented public boundary.
- [ ] Pure operations return independent C-contiguous arrays and do not mutate or alias any input array.
- [ ] Operations reject NaN and positive infinity at their approved input or output validation boundaries.
- [ ] Matrix multiplication accumulates in `float64` and materializes the completed result once as `float32`.
- [ ] Column sums accumulate in `float64` and materialize the completed result once as `float32`.
- [ ] Approved row-bias addition accepts exactly one compatible row and rejects all other broadcasting shapes.
- [ ] An explicitly in-place addition calculates the complete candidate in non-overlapping `float64` scratch state, validates the complete `float32` candidate, and commits only after every value is finite.
- [ ] An in-place failure leaves the destination byte-for-byte unchanged.
- [ ] In-place operations reject overlapping source and destination memory, including partially overlapping views.
- [ ] The stable row softmax accepts finite scores plus intentional negative-infinity causal masks, leaves the score matrix unchanged, and returns a separate finite probability matrix.
- [ ] Every valid softmax row sums to one within an explicit tight tolerance, and masked future positions are exactly zero.
- [ ] A softmax row containing no finite selectable value, NaN, or positive infinity fails before returning probabilities.
- [ ] Transpose, elementwise, scalar, concatenation, slicing, or other primitives required by the approved Transformer formulas preserve the same shape, purity, contiguity, and finiteness contract.
- [ ] Exact shape, dtype, contiguity, aliasing, and mask assertions plus tolerance-based numerical fixtures cover every public primitive.
- [ ] Tests use independently calculated small matrices and include failure cases that distinguish transactional behavior from partial mutation.

## Testing Expectations

- **Approved test seam:** The stable public matrix boundary exercised directly with small NumPy arrays.
- **Behavior to verify:** Shape validation, supported broadcasting, float32 materialization, float64 accumulation, C-contiguity, purity, aliasing rejection, transactional in-place mutation, finiteness, causal masks, and stable row softmax.
- **Relevant prior art:** Existing NumPy-based numerical tests and the specification's exact-versus-tolerance comparison policy.
- **Do not test through:** Private helper decomposition, a specific vectorization expression, temporary-array identity, BLAS implementation details, or exact wall-clock performance.

## Blocked By

- None — can start immediately.

## Constraints and Out of Scope

- Keep this boundary stateless and general only to the operations already required by Phase 5; do not create a general machine-learning framework.
- Do not add automatic differentiation, tensors with hidden gradient state, GPU support, sparse matrices, batching abstractions, or third-party numerical frameworks.
- Do not silently broaden shapes through NumPy broadcasting.
- Do not mutate input arrays from pure operations.
- Do not implement Transformer architecture, training, workers, routing, or persistence in this ticket.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — approved matrix seam, numerical precision rules, and observable matrix behavior.
- `CONTEXT.md` — Transformer Training Compatibility terminology.
- ADR 0002 — stateless matrix contract, float32 materialization, float64 reductions, transactionality, and softmax masking rules.
- Latest Python Backend source export — empty Phase 5 matrix destination and current NumPy conventions.
- Latest TypeScript Reference Implementation — matrix formulas and shape evidence.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
