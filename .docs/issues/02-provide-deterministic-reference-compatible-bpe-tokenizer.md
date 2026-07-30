---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: 02
source_document: SPEC.md
recommended_next_prompt: to-plan-prompt
---

# Ticket 02: Provide a deterministic reference-compatible BPE Tokenizer

## What to build

Implement the reusable educational BPE Tokenizer behavior required by the Basic Tokenizer learning demo through typed public operations equivalent to `count_words()`, `train_bpe()`, and `apply_merges()`.

The tokenizer must preserve Strict TypeScript Compatibility for Pre-token boundaries, ASCII-style word classification, repeated Pre-token frequency weighting, deterministic BPE Pair selection, non-overlapping left-to-right merge behavior, learned Merge Table order, and the maximum of 1,000 learned merges.

The public BPE boundary must remain independent of HTTP validation, SSE payload construction, response headers, animation timing, persistence, caching, and future Word2Vec or transformer functionality.

## Acceptance Criteria

- [ ] Pre-tokenization separates ASCII-style word sequences, individual whitespace characters, and individual punctuation characters.
- [ ] BPE Pair discovery and BPE Merges never cross a Pre-token boundary.
- [ ] Repeated Pre-tokens are counted and weight adjacent-pair frequencies by their occurrence counts.
- [ ] When multiple BPE Pairs share the highest frequency, the first pair encountered in the reference-compatible traversal order is selected.
- [ ] A selected pair replaces non-overlapping adjacent occurrences from left to right within each Pre-token.
- [ ] Learned BPE Merges are recorded in selection order in the Merge Table.
- [ ] Training stops after at most 1,000 learned merges or earlier when no adjacent BPE Pair remains.
- [ ] Applying the Merge Table to the original BPE Training Text uses the learned order and returns reference-compatible tokens.
- [ ] Single-character, whitespace-only, and punctuation-only inputs complete successfully without inventing invalid cross-boundary merges.
- [ ] Representative Unicode input follows the confirmed ASCII-style classification behavior rather than Python's default Unicode word classification.
- [ ] Separate public-interface calls do not share or retain mutable Tokenization Run state.
- [ ] The public BPE operations and Merge record use complete modern Python type annotations compatible with strict mypy checking.

## Testing Expectations

- **Approved test seam:** Exercise the public reusable BPE interface through its counting, training, and merge-application operations.
- **Behavior to verify:** Pre-token counting, ASCII-style classification, occurrence-weighted frequencies, deterministic ties, non-overlapping merging, early termination, the 1,000-merge limit, ordered Merge Table application, fixed reference-compatible results, punctuation, whitespace, repeated Pre-tokens, single-character input, and representative Unicode behavior.
- **Relevant prior art:** Fixed parity cases are to be derived from the confirmed TypeScript Reference Implementation behavior recorded in the specification; no runtime TypeScript execution is required.
- **Do not test through:** Private merge-helper identity, private container choices, internal loop structure, local variables, or other implementation details invisible through the public BPE seam.

## Blocked By

- None — can start immediately.

## User Stories Addressed

- User stories 5–14 — Minimal input, punctuation, whitespace, repeated Pre-tokens, deterministic and non-overlapping learning, boundary preservation, ordered application, early termination, and ASCII-compatible Unicode behavior.
- User story 31 — Separate requests or calls train independently.
- User story 33 — Reusable BPE behavior remains separate from HTTP and SSE presentation.
- User stories 37–39 — Existing dependencies are sufficient, the educational algorithm remains directly implemented, and the 1,000-merge limit is preserved.
- User stories 41 and 43 — Fixed Python parity cases isolate reusable algorithm divergence.
- User story 45 — Temporary BPE state is discardable and does not require maintenance.

## Constraints and Out of Scope

- Implement only reusable behavior needed by the Phase 2 Basic Tokenizer route.
- Do not implement `train_bpe_on_text()`, custom Pre-token patterns, Word2Vec-specific behavior, transformer-specific behavior, or later-phase abstractions.
- Do not use an opaque tokenizer library, hosted service, LangChain, or LangGraph.
- Do not add persistence, tokenizer files, cross-request caching, multiprocessing, worker pools, or shared memory.
- Do not add NumPy usage when string and collection operations are sufficient.
- Do not own HTTP validation, SSE payload construction, animation delays, response headers, or route registration.

## Assumptions and Evidence Limitations

- The implementation may choose idiomatic immutable or read-only Python structures for BPE Pairs, Merge records, token sequences, and counted Pre-tokens as long as behavior remains compatible.
- No existing BPE tests were visible in the supplied Python snapshot.

## Source

- [Phase 2 BPE Tokenizer specification](../../../SPEC.md)

## Recommended Next Step

Run `to-plan-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
