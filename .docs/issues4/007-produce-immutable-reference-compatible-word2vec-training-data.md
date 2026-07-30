---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "007"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 007: Produce immutable reference-compatible Word2Vec training data

## What to build

Provide a stable public Word2Vec preprocessing boundary that derives all reusable training data from the fixed Embedding Training Corpus. Preserve every corpus sentence and its order, train the exact 500-rule Merge Table from the lowercased joined corpus, apply the learned merges within Pre-token boundaries, build the ordered Vocabulary and token indices, and generate ordered Training Pairs for every supported context-window size.

The corpus-derived result must be safe to share across Embedding Training Runs without allowing one caller to mutate the data observed by another caller. Submitted Query Words select final display results only and must not change the Embedding Training Corpus or its derived preprocessing.

## Acceptance Criteria

- [ ] The public preprocessing result preserves the exact fixed Embedding Training Corpus and sentence order established by the TypeScript Reference Implementation.
- [ ] The BPE training text is the exact lowercased corpus joined in the confirmed reference form.
- [ ] Preprocessing learns exactly 500 ordered BPE Merges and preserves their pair, replacement token, frequency, and order through independently verified fixtures.
- [ ] Applying the Merge Table never crosses a Pre-token boundary and replays merges strictly in learned order.
- [ ] Representative corpus sentences produce the independently verified token sequences, including boundary-sensitive cases.
- [ ] Token frequencies are counted in first-encounter order, then stably sorted by descending frequency so equal-frequency Vocabulary Tokens retain first-encounter order.
- [ ] The ordered Vocabulary determines one deterministic index for every Vocabulary Token.
- [ ] Training Pairs are produced in sentence order, target-token order, and context-position order.
- [ ] Training Pair fixtures cover every supported window size from 1 through 5.
- [ ] Repeated calls return equivalent preprocessing while preventing mutation through one result from changing later results or another concurrent Embedding Training Run.
- [ ] Query Words are not incorporated into the corpus, Merge Table, Vocabulary, frequencies, indices, or Training Pairs.
- [ ] Exact expected structures are fixed independent evidence and are not calculated by invoking the production operation being tested.

## Testing Expectations

- **Approved test seam:** The stable public Word2Vec module boundary for corpus preprocessing, ordered Vocabulary construction, and Training Pair generation.
- **Behavior to verify:** Exact corpus data, BPE training text, complete Merge Table compatibility, representative tokenization, frequencies, stable ties, indices, Training Pair order for all supported windows, and shared-data mutation isolation.
- **Relevant prior art:** The existing public BPE interface and its deterministic exact-assertion tests; the specification's independent-fixture policy for reference-compatible Word2Vec behavior.
- **Do not test through:** Private helper names, internal container choices, cache identity, local loop structure, or a particular immutable representation.

## Blocked By

- None — can start immediately.

## Constraints and Out of Scope

- Keep the Python Backend authoritative; the TypeScript implementation is compatibility evidence only.
- Reuse the established BPE boundary where it matches the reference behavior rather than copying BPE logic into a second implementation.
- Do not add numerical Skip-gram updates, Embedding Result construction, HTTP or SSE behavior, filesystem persistence, frontend changes, Transformer work, or a general-purpose machine-learning framework.
- Do not add a configurable corpus, seed, learning rate, optimizer, or preprocessing cache policy.
- Do not add or change dependencies unless current implementation evidence proves one is necessary.

## Source

- `SPEC.md` — *Deliver Deterministic Phase 4 Word2Vec Embeddings Through the Python Backend*.
- `CONTEXT.md` — canonical Embedding Training Corpus, Vocabulary, Training Pair, and Deterministic Embedding Compatibility terminology.
- ADR 0001 — *Preserve Deterministic Word2Vec Compatibility*.
- Latest Python Backend source export and TypeScript Reference Implementation supplied with the specification.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, `SPEC.md`, `CONTEXT.md`, ADR 0001, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
