---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "021"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 021: Persist configuration-specific Saved Transformer Models safely

## What to build

Provide an independently testable persistence boundary for complete Saved Transformer Models. Retain one final model for each approved epochs-and-layer-count training configuration using the exact configuration-specific filename, complete in-memory serialization, a secure unique same-directory temporary file, file flush and `fsync`, close-before-replace ordering, and one atomic replacement.

A failed save must preserve the prior valid destination byte-for-byte and must not affect another configuration's model. Persisted models are output artifacts only and are never loaded, resumed, cached, or used to skip future training.

## Acceptance Criteria

- [ ] The production filename is exactly `transformer-weights-e{epochs}-l{numLayers}-d32-h2-ff128-ctx32.json` for the validated training configuration.
- [ ] The production destination resolves from the backend module location and is independent of the process working directory.
- [ ] The backend data directory is created when absent.
- [ ] The JSON document preserves top-level insertion order `type`, `config`, `vocab`, `merges`, and `weights`, plus the established nested configuration and weight order supplied by the complete Saved Transformer Model.
- [ ] Serialization uses UTF-8, two-space indentation, insertion-order keys, no key sorting, `allow_nan=False`, `
` line endings, and exactly one trailing newline.
- [ ] Serialization completes fully in memory and rejects NaN, positive infinity, negative infinity, unsupported objects, or malformed model structure before the destination can be replaced.
- [ ] Each save securely creates one unique temporary file in the same directory as its final destination.
- [ ] The complete serialized document is written, flushed, file-synchronized, and closed before atomic replacement begins.
- [ ] The final destination changes only through one successful `os.replace()` of a complete closed temporary file.
- [ ] Serialization, directory creation, temporary-file creation, write, flush, file synchronization, close, and replacement failures preserve the previous destination byte-for-byte.
- [ ] A failed save removes only its own temporary file when cleanup succeeds.
- [ ] A temporary-file cleanup failure preserves the prior destination and remains available for internal reporting together with the original failure.
- [ ] Directory synchronization is not required and is not presented as part of the durability guarantee.
- [ ] Distinct epochs or numLayers configurations write distinct destinations and cannot overwrite one another.
- [ ] Controlled concurrent successful saves to the same configuration leave exactly one complete valid document, with the last successful atomic replacement becoming the destination.
- [ ] A failed concurrent save cannot truncate, corrupt, remove, or replace a model written successfully by another save.
- [ ] No persistence operation reads a prior model or uses its existence to skip, resume, fine-tune, cache, or alter a new Transformer Training Run.
- [ ] No intermediate epoch checkpoint, manifest, history file, latest-model alias, rollback artifact, or cross-process lock is created.
- [ ] All filesystem tests use pytest-managed temporary directories and never alter the real backend data destination.

## Testing Expectations

- **Approved test seam:** The route-owned Saved Transformer Model serialization and atomic replacement boundary exercised under a pytest temporary directory.
- **Behavior to verify:** Exact configuration filenames, path resolution, directory creation, model ordering and formatting, in-memory finite serialization, unique same-directory temporary files, flush/fsync/close-before-replace, failure preservation, cleanup, configuration isolation, and deterministic concurrency outcomes.
- **Relevant prior art:** The completed XOR and Word2Vec atomic persistence tests, strengthened by the Phase 5 file-`fsync` and configuration-specific filename contract.
- **Do not test through:** A particular temporary-file helper, private serializer names, raw operating-system wrapper identity, exact elapsed concurrency timing, or model training mathematics.

## Blocked By

- [Ticket 018 — Generate deterministic text and construct Saved Transformer Models](018-generate-deterministic-text-and-construct-saved-transformer-models.md)

## Constraints and Out of Scope

- Persist only a complete final Saved Transformer Model after training; do not write intermediate checkpoints.
- Do not load, resume, fine-tune, cache, or skip training from saved files.
- Do not add a global latest-model file, model registry, history, manifest, rollback system, frontend download, or cross-process file lock.
- Do not persist optimizer moments, gradients, shared-memory names, process state, caches, request identifiers, or paths inside the model.
- Do not implement HTTP validation, SSE orchestration, worker supervision, or Generated Text Sample creation in this ticket.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — final-only model retention, failure preservation, and persistence-before-done requirement.
- `CONTEXT.md` — Saved Transformer Model terminology.
- ADR 0002 — exact filename, JSON ordering/formatting, fsync, atomic replacement, cleanup, and no-load policy.
- Ticket 018 — complete serialization-ready Saved Transformer Model.
- Latest Python Backend source export — established route-owned atomic persistence conventions.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
