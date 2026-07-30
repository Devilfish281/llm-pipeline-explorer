---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "010"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 010: Construct exact Embedding Results and Saved Embedding Models

## What to build

Convert a completed deterministic Embedding Training Run into the exact frontend-facing Embedding Result and the complete persistence-ready Saved Embedding Model. Preserve submitted Query Word positions, lookup and warning behavior, public-vector precision, Nearest Neighbor ranking, Similarity Pair generation, the seven predefined Vector Analogies, stable ties, and exact public field sets.

The Embedding Result contains only selected display data. The Saved Embedding Model contains the complete trained Vocabulary model and is suitable for Ticket 008's persistence boundary.

## Acceptance Criteria

- [ ] Every public Word Embedding comes only from the corresponding input-weight row and rounds each coordinate to six decimals with TypeScript-compatible public rounding.
- [ ] Output-weight rows and unrounded internal matrices are never exposed in the Embedding Result or Saved Embedding Model.
- [ ] Query Words remain in submitted order, including duplicate positions, and are not trimmed, split, deduplicated, filtered, or removed.
- [ ] Lookup lowercases each complete submitted entry only for Vocabulary recognition and warning analysis; original text is preserved for display and warnings.
- [ ] Uppercase recognized Query Words resolve case-insensitively, while leading-space, trailing-space, whitespace-only, unknown, and multi-token entries retain their confirmed recognition behavior.
- [ ] Every unrecognized or multi-token Query Word contributes exactly one warning in the format `"<submitted word>" is not a single BPE token — it splits into [<comma-separated tokens>]`.
- [ ] Unrecognized positions are omitted from selected embeddings, neighbor groups, and pairwise similarities without failing an otherwise successful run.
- [ ] Duplicate recognized positions produce repeated selected embeddings and repeated Nearest Neighbor groups.
- [ ] Similarity Pairs cover every pair of recognized Query Word positions, including duplicate positions that resolve to the same Vocabulary Token.
- [ ] A request whose Query Words are all unrecognized produces empty selected embeddings, neighbors, and similarities plus ordered warnings, while predefined analogies are still evaluated.
- [ ] Nearest Neighbor cosine scores use public six-decimal vectors, round to two decimals before ranking, sort stably by descending rounded score, exclude the Query Word's own Vocabulary Token, and return at most five candidates.
- [ ] Similarity Pair scores use public six-decimal vectors and TypeScript-compatible two-decimal rounding.
- [ ] The seven predefined Vector Analogies are evaluated in their established order.
- [ ] Each analogy query uses raw input-weight rows for `a - b + c`, compares against six-decimal public candidate vectors, excludes all three source tokens, preserves first-Vocabulary-candidate tie behavior, and rounds the final score to two decimals.
- [ ] The Embedding Result contains exactly `embeddings`, `neighbors`, `similarities`, `analogies`, and `warnings` with no complete model, persistence metadata, or internal numerical state.
- [ ] The Saved Embedding Model contains exactly `type`, `dimensions`, `vocab`, `merges`, and `embeddings`.
- [ ] The Saved Embedding Model preserves the complete ordered Vocabulary and Merge Table and includes a six-decimal public Word Embedding for every Vocabulary Token, not only submitted Query Words.
- [ ] Neither successful public object contains NaN or infinity.
- [ ] Exact rounded outputs, ordering, warnings, tie outcomes, field sets, and Saved Embedding Model contents match independent fixtures; any unrounded score checks use explicit tight tolerances.
- [ ] Result construction does not mutate the completed Training Run, shared preprocessing, Query Word sequence, or data observed by another result-construction call.

## Testing Expectations

- **Approved test seam:** The stable public Word2Vec module boundary for result construction and Saved Embedding Model conversion.
- **Behavior to verify:** Public vectors, duplicate and unrecognized Query Word positions, exact warning text, neighbor and similarity precision, stable ranking ties, analogy mixed precision and exclusions, exact payload fields, complete saved-model conversion, finite values, and state isolation.
- **Relevant prior art:** The project's exact frontend-payload tests and the specification's independent exact fixtures for rounded public behavior.
- **Do not test through:** Private data classes, helper names, local sorting implementation, internal container identity, route serialization, filesystem operations, or frontend component rendering.

## Blocked By

- [Ticket 009 — Run deterministic reference-compatible Skip-gram training](009-run-deterministic-reference-compatible-skip-gram-training.md)

## Constraints and Out of Scope

- Preserve the confirmed mixed-precision analogy behavior; do not simplify all computations to either raw or rounded vectors.
- Query Words select display results only and must not retrain or alter the model.
- Do not implement HTTP validation, SSE framing, delays, disconnection handling, filesystem persistence, frontend changes, model loading, history, Transformer work, or a general-purpose vector database.
- Do not expose output weights, raw input matrices, paths, timestamps, request identifiers, or persistence controls.

## Source

- `SPEC.md` — Embedding Result, Query Word, ranking, analogy, warning, and Saved Embedding Model contracts.
- `CONTEXT.md` — canonical Query Word, Nearest Neighbor, Similarity Pair, Vector Analogy, Embedding Result, and Saved Embedding Model terminology.
- ADR 0001 — exact rounded public compatibility and mixed exact/tolerance numerical verification.
- Ticket 009 and the supplied TypeScript Reference Implementation.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, Ticket 009, `SPEC.md`, `CONTEXT.md`, ADR 0001, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
