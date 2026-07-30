---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "008"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 008: Persist complete Saved Embedding Models safely

## What to build

Provide an independently testable persistence boundary for a complete Saved Embedding Model. Persist the latest successful model to the confirmed backend-owned `embedding-weights.json` destination using exact JSON serialization, a unique temporary file in the destination directory, and atomic replacement only after the complete temporary document is closed successfully.

A failed save must leave the previous Saved Embedding Model untouched. Concurrent successful saves must never interleave partial content; the complete model from the last successful replacement becomes the destination.

## Acceptance Criteria

- [ ] The persisted document contains exactly `type`, `dimensions`, `vocab`, `merges`, and `embeddings` at the top level.
- [ ] `type` is exactly `word2vec-skipgram`.
- [ ] `vocab` preserves every ordered Vocabulary Token, `merges` preserves the complete ordered Merge Table, and `embeddings` contains one public Word Embedding for every Vocabulary Token.
- [ ] Serialization uses two-space indentation and exactly one trailing newline.
- [ ] Serialization rejects NaN, positive infinity, and negative infinity before the destination can be replaced.
- [ ] The production destination is resolved from the backend project rather than the process's current working directory.
- [ ] The backend data directory is created when it does not exist.
- [ ] Each save owns a unique temporary file in the same directory as the final destination.
- [ ] The complete serialized document is written and the temporary writer is closed before replacement begins.
- [ ] The destination changes only after one successful atomic replacement of a complete temporary document.
- [ ] Serialization, temporary-file creation, write, close, and replacement failures preserve the previous destination byte for byte.
- [ ] A failed save removes only its own temporary file when cleanup succeeds.
- [ ] A cleanup failure does not replace or remove the previous valid destination and remains available for internal error reporting together with the original failure.
- [ ] Controlled concurrent successful saves leave one complete valid model, with the last successful finisher becoming the final document.
- [ ] A failed concurrent save cannot truncate, corrupt, or remove a model written successfully by another save.
- [ ] No persistence path reads, resumes, fine-tunes, caches, or otherwise reuses a previously saved model.
- [ ] Filesystem tests write only under pytest-managed temporary directories and never alter the real backend data destination.

## Testing Expectations

- **Approved test seam:** The route-owned Saved Embedding Model serialization and atomic replacement boundary exercised with a pytest temporary directory.
- **Behavior to verify:** Exact JSON document, backend-root path resolution, directory creation, unique same-directory temporary files, close-before-replace ordering, failure preservation, cleanup, and deterministic concurrency outcomes.
- **Relevant prior art:** The completed Neural Network Saved Weight Snapshot persistence tests and route-owned atomic-replacement pattern in the current Python Backend.
- **Do not test through:** A particular temporary-file library, private helper names, local variable names, exact operating-system call wrappers, or elapsed timing to establish concurrency order.

## Blocked By

- None — can start immediately.

## Constraints and Out of Scope

- Accept a complete persistence-ready Saved Embedding Model through a typed public boundary; use fixed model fixtures so this ticket can proceed independently of numerical training.
- Do not implement `POST /train-embed`, request validation, SSE events, presentation delays, disconnection handling, or router registration.
- Do not add model loading, history, manifests, registries, checkpoints, rollback, cross-process locking, a global save lock, a training queue, or a frontend download feature.
- Do not add a new dependency unless current project inspection proves the existing standard-library and test facilities insufficient.

## Source

- `SPEC.md` — Saved Embedding Model and approved atomic-persistence seam.
- `CONTEXT.md` — canonical Saved Embedding Model terminology.
- ADR 0001 — persistence-before-completion and deterministic public-model compatibility.
- Existing Python Backend Neural Network persistence behavior as prior art.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, `SPEC.md`, `CONTEXT.md`, ADR 0001, and the latest Python Backend source export.
