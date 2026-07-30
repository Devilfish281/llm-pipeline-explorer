---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: 004
source_document: SPEC.md
recommended_next_prompt: to-plan-prompt
---

# Ticket 004: Provide reference-compatible XOR Training Runs

## What to build

Implement the reusable numerical behavior for the XOR Neural Network Demo through a stable public boundary in the neural-network module.

A Training Run must support Single-Layer Mode and Multi-Layer Mode, use independent NumPy `float32` numerical state, preserve the fixed XOR truth-table order and the confirmed learning procedure, produce reference-compatible Epoch Updates, and calculate the final ordered XOR Predictions and exact Training Verdict.

Production initialization must remain random. Tests must be able to supply a NumPy `Generator` so one verified seed can deterministically demonstrate the intended educational contrast:

```text
Single-Layer Mode → FAILED
Multi-Layer Mode  → SUCCESS
```

The public boundary must also convert the completed numerical state into the exact plain-Python object required for a mode-specific Saved Weight Snapshot. It must remain independent of FastAPI request validation, SSE framing, client-disconnect checks, presentation delays, and filesystem writes.

## Acceptance Criteria

- [ ] Single-Layer Mode represents a `2 → 1` sigmoid network with two `float32` weights and one zero-initialized bias.
- [ ] Multi-Layer Mode represents a `2 → 4 → 1` sigmoid network with `float32` state shaped as `(2, 4)`, `(4,)`, `(4,)`, and one scalar output bias, with all biases initialized to zero.
- [ ] Every production Training Run receives a fresh NumPy `Generator` and initializes each weight independently in `[-1, 1)`.
- [ ] Tests can supply a NumPy `Generator` without adding a seed field to the HTTP contract or relying on NumPy global random state.
- [ ] Training processes `[0,0] → 0`, `[0,1] → 1`, `[1,0] → 1`, and `[1,1] → 0` in that fixed order during every epoch.
- [ ] The sigmoid function, derivative from the sigmoid output, learning rate `1.0`, immediate per-example updates, and mean-squared loss over four examples match the confirmed reference procedure.
- [ ] Multi-Layer Mode computes hidden error values from the current output weights before updating those output weights, then updates the output layer before the input-to-hidden layer.
- [ ] Training advances from epoch `0` through the requested epoch value, inclusive.
- [ ] The reporting step is `max(1, floor(epochs / 50))`, and an Epoch Update is produced at every regular reporting boundary and at the final requested epoch.
- [ ] Serialized Epoch Update loss values use reference-compatible six-decimal rounding.
- [ ] Final XOR Predictions are ordered `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`, and each `actual` value is rounded to two decimal places before success is calculated.
- [ ] A Training Run succeeds only when every rounded prediction differs from its expected value by strictly less than `0.1`.
- [ ] The exact Single-Layer Mode architecture label and success/failure verdict strings are preserved.
- [ ] The exact Multi-Layer Mode architecture label and success/failure verdict strings are preserved.
- [ ] The single-layer Saved Weight Snapshot object contains exactly `type`, `w1`, `w2`, and `bias` as ordinary Python JSON-compatible values.
- [ ] The multi-layer Saved Weight Snapshot object contains exactly `type`, `w1`, `b1`, `w2`, and `b2` as ordinary Python JSON-compatible values and lists.
- [ ] Separate Training Runs do not share mutable weights, epoch state, predictions, or generator state.
- [ ] One verified deterministic seed and explicit numerical tolerances are selected from actual Python execution and recorded in tests.
- [ ] With that seed and the approved test epoch count, Single-Layer Mode returns `FAILED — loss stuck, predictions are random guesses`.
- [ ] With that seed and the approved test epoch count, Multi-Layer Mode returns `SUCCESS — network learned XOR via backpropagation`, and every rounded prediction is within `0.1` of its expected target.
- [ ] The public numerical behavior is fully typed and passes the project’s strict mypy configuration.

## Testing Expectations

- **Approved test seam:** Public numerical neural-network module.
- **Behavior to verify:** XOR example order, sigmoid and derivative values, initialization bounds, zero biases, `float32` dtypes, mode-specific shapes, fixed sample order, immediate updates, mean-squared loss, multi-layer backpropagation order, reporting boundaries, rounded outputs, exact labels and verdicts, deterministic educational outcomes, request-state isolation, and exact JSON-compatible snapshot objects.
- **Relevant prior art:** The existing public BPE module tests use fixed reference-compatible cases, exact assertions for discrete contracts, and no runtime TypeScript dependency.
- **Required comparison strategy:** Use exact assertions for discrete values, dtypes, shapes, rounded payloads, labels, verdicts, and snapshot structures. Use explicit `numpy.testing.assert_allclose()` tolerances for unrounded floating-point calculations.
- **Required reporting cases:** Cover `epochs=100`, a non-divisible value such as `101`, and the default schedule of `5000` without production presentation delays.
- **Do not test through:** Private helper names, local variable names, a particular dataclass or state-container identity, exact loop syntax, or another implementation detail invisible through the public numerical boundary.

## Blocked By

- None — can start immediately.

## User Stories Addressed

- User stories 1–12 — Learners can observe both XOR modes, progress loss, ordered predictions, and exact educational verdicts.
- User stories 29–36 — Training Runs are isolated, NumPy `float32` state is explicit, production remains random, deterministic tests are supported, and serialized contracts are exact.
- User story 51 — pytest, Ruff, and strict mypy remain the validation path.

## Constraints and Out of Scope

- Use only Python and the dependencies already declared for the backend.
- Keep the reusable behavior focused on the XOR Neural Network Demo.
- Preserve controlled floating-point compatibility rather than requiring bit-for-bit JavaScript equality.
- Do not add HTTP request fields for seed, learning rate, hidden size, activation, optimizer, persistence, dtype, or output path.
- Do not add batching, shuffling, momentum, Adam, regularization, dropout, early stopping, learning-rate schedules, or replacement initialization schemes.
- Do not introduce a general matrix framework, PyTorch, TensorFlow, JAX, scikit-learn, GPU support, LangChain, LangGraph, or hosted services.
- Do not implement Word2Vec, transformer training, frontend changes, HTTP/SSE behavior, filesystem persistence, or saved-weight loading in this ticket.

## Assumptions and Evidence Limitations

- The exact public function, state-object, iterator, dataclass, or protocol names are implementation choices provided the stable observable behavior can be exercised without private access.
- The exact deterministic seed and tolerances are intentionally calibrated during implementation from actual Python results; they are not unresolved product decisions.
- Random production initialization is allowed to produce a failed Multi-Layer Mode Training Run. The deterministic test proves one stable educational case, not universal convergence.

## Source

- [Phase 3 XOR Neural Network specification](../SPEC.md)
- [Canonical project context](../CONTEXT.md)
- Latest complete Python backend source snapshot supplied with the specification.

## Recommended Next Step

Run `to-plan-prompt` in a fresh conversation using this ticket, the source specification, the canonical context, and the latest complete backend source snapshot.
