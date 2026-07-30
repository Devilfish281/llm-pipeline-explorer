---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "009"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 009: Run deterministic reference-compatible Skip-gram training

## What to build

Provide a bounded Embedding Training Run that consumes the immutable preprocessing from Ticket 007 and reproduces the TypeScript Reference Implementation's deterministic Skip-gram training with negative sampling. Each run must own its mutable Training Pair order, Mulberry32 generator, input and output matrices, epoch state, gradients, and losses.

Preserve the exact random-consumption sequence, numerical formulas, online update order, inclusive epoch semantics, and public Embedding Epoch Update schedule required for Deterministic Embedding Compatibility.

## Acceptance Criteria

- [ ] Every Embedding Training Run creates one request-owned Mulberry32 generator seeded with `42`, with exact verified 32-bit wraparound and output sequence.
- [ ] No production path substitutes NumPy-native randomness or consumes random values outside the confirmed reference order.
- [ ] Input and output weights are separate NumPy `float64` matrices with the requested dimensions and no shared memory between the matrices or separate runs.
- [ ] Both matrices are initialized in the exact reference random-call order using `scale = 0.5 / dimensions` and `weight = (random - 0.5) × scale`.
- [ ] The negative-sampling cumulative distribution is derived from Vocabulary frequencies raised to `0.75` and matches independent fixtures.
- [ ] Each epoch shuffles the run-owned Training Pair sequence in place with Mulberry32-driven Fisher–Yates behavior matching independent fixtures.
- [ ] Epochs execute inclusively from epoch `0` through the requested final epoch.
- [ ] The learning rate follows the confirmed linear schedule from `0.025` to `0.001` at the reference epoch boundaries.
- [ ] Every Training Pair applies its positive target-context update before any negative candidate is drawn or updated.
- [ ] Exactly the requested number of negative candidates is drawn; a candidate equal to the true context is skipped without a replacement draw.
- [ ] Positive and negative updates occur immediately and dimension by dimension rather than through accumulated or matrix-wide gradients.
- [ ] Updating an output coordinate uses the saved pre-update input coordinate required by the reference operation order.
- [ ] Sigmoid clipping, positive and negative loss formulas, and the `1e-10` protection term match independent numerical fixtures.
- [ ] Epoch loss is divided by the number of positive Training Pairs, not by the combined count of positive and effective negative examples.
- [ ] Progress uses `report_step = max(1, floor(epochs / 50))`, includes epoch zero and the requested final epoch, and reports each public loss rounded to six decimals.
- [ ] A short complete deterministic Embedding Training Run matches independently calculated unrounded state within explicit tight tolerances and matches rounded public Epoch Updates exactly.
- [ ] A run that encounters a non-finite weight, gradient, score, or loss cannot produce a successful completion value.
- [ ] Sequential and concurrent runs with identical inputs reproduce the same observable result without sharing mutable pair order, PRNG state, matrices, gradients, or losses.
- [ ] Expected random sequences, shuffles, updates, matrices, and losses are fixed independent evidence rather than values calculated through the production helper under test.

## Testing Expectations

- **Approved test seam:** The stable public Word2Vec module boundary for deterministic randomness, initialization, bounded Training Run advancement, and numerical state transitions.
- **Behavior to verify:** Mulberry32, Fisher–Yates, negative sampling, `float64` matrix separation and initialization, one positive update, one negative update, skip-without-redraw behavior, inclusive epochs, learning-rate and reporting schedules, short-run compatibility, finite-state enforcement, and request isolation.
- **Relevant prior art:** The current Python Backend's bounded XOR Training Run pattern and the specification's exact-versus-tolerance comparison policy.
- **Do not test through:** Private loop variables, helper identity, exact iterator class design, a particular thread-pool object, or universal bit-for-bit equality for every unrounded transcendental intermediate.

## Blocked By

- [Ticket 007 — Produce immutable reference-compatible Word2Vec training data](007-produce-immutable-reference-compatible-word2vec-training-data.md)

## Constraints and Out of Scope

- Preserve deterministic operation order even when a vectorized or library implementation would be shorter or faster.
- Do not use Gensim, PyTorch, TensorFlow, JAX, scikit-learn, hosted embeddings, NumPy-native randomness, batching, hierarchical softmax, subsampling, gradient clipping, early stopping, or a redesigned optimizer.
- Do not implement Query Word result selection, Nearest Neighbors, Similarity Pairs, Vector Analogies, HTTP or SSE orchestration, persistence, frontend changes, Transformer work, multiprocessing, or shared memory.
- Do not add configurable seed, learning rate, corpus, sigmoid clipping, or report-count controls.

## Source

- `SPEC.md` — deterministic numerical behavior and approved public Word2Vec test seam.
- `CONTEXT.md` — canonical Skip-gram Training, Negative Sample, Embedding Training Run, and Embedding Epoch Update terminology.
- ADR 0001 — exact random-call and operation order with tolerance-aware hidden numerical checks.
- Ticket 007 and the supplied TypeScript Reference Implementation.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, Ticket 007, `SPEC.md`, `CONTEXT.md`, ADR 0001, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
