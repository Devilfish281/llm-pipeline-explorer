---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: 005
source_document: SPEC.md
recommended_next_prompt: to-plan-prompt
---

# Ticket 005: Persist completed XOR Training Runs safely

## What to build

Add the route-owned persistence boundary that converts a completed Training Run’s JSON-compatible weight object into the mode-specific Saved Weight Snapshot required by the XOR Neural Network Demo.

A successful save must resolve the backend project’s `.data` directory independently of the shell’s current working directory, create the directory when necessary, write a complete two-space-indented JSON document with one final newline to a unique same-directory temporary file, and atomically replace the selected mode’s destination.

Persistence must preserve the previous successful snapshot when serialization, writing, replacement, or cleanup fails. Concurrent Training Runs must remain independent, and the last successful finisher for one mode must become that mode’s complete Saved Weight Snapshot.

This ticket establishes and verifies the persistence behavior through temporary directories. It does not expose `POST /neural-net`, emit SSE events, or perform client-disconnect orchestration.

## Acceptance Criteria

- [ ] A completed Single-Layer Mode Training Run is saved as `single-layer-weights.json`.
- [ ] A completed Multi-Layer Mode Training Run is saved as `multi-layer-weights.json`.
- [ ] The destination directory is resolved from the backend project root rather than the process working directory.
- [ ] The destination directory is created when it does not exist.
- [ ] The single-layer JSON document contains exactly `type`, `w1`, `w2`, and `bias`.
- [ ] The multi-layer JSON document contains exactly `type`, `w1`, `b1`, `w2`, and `b2`.
- [ ] All serialized values are ordinary JSON numbers and arrays rather than NumPy objects.
- [ ] No epochs, seed, timestamp, loss, verdict, architecture, dtype, shape metadata, request ID, manifest, or history record is added.
- [ ] JSON is written with two-space indentation and exactly one final newline.
- [ ] Every save uses a unique temporary file in the same directory as the final destination.
- [ ] The temporary file is closed before the final destination is replaced.
- [ ] The selected destination is replaced only after the complete JSON document has been written successfully.
- [ ] A failed serialization, write, or replacement leaves the previous successful destination unchanged.
- [ ] Temporary files are removed when failure occurs before replacement completes.
- [ ] Simultaneous successful saves for different modes never target the same destination.
- [ ] Controlled simultaneous successful saves for the same mode produce one complete, non-corrupt destination, and the last successful finisher wins.
- [ ] Saving a new Training Run never loads or continues from a previous snapshot.
- [ ] Tests use pytest temporary directories and never write into the repository’s real `.data` directory.
- [ ] The persistence boundary is fully typed and passes the project’s strict mypy configuration.

## Testing Expectations

- **Approved test seam:** Persistence helper or route-owned Saved Weight Snapshot boundary.
- **Behavior to verify:** Mode-specific names, project-root resolution, directory creation, exact keys, plain JSON values, two-space formatting, final newline, same-directory unique temporary files, complete atomic replacement, previous-snapshot preservation, temporary-file cleanup, different-mode isolation, and last-successful-finisher-wins concurrency.
- **Relevant prior art:** Existing backend tests patch route-owned dependencies and use deterministic fixtures; the source specification explicitly approves a temporary-directory persistence seam.
- **Required fixtures and controls:** Use `tmp_path` or an equivalent pytest temporary directory and controlled completion ordering for concurrent saves.
- **Do not test through:** A particular temporary-file API, private helper name, exact local variable, or implementation-specific locking identity when the observable atomic-save contract is satisfied.

## Blocked By

- [Ticket 004: Provide reference-compatible XOR Training Runs](004-provide-reference-compatible-xor-training-runs.md)

## User Stories Addressed

- User stories 13–15 — Learners receive inspectable mode-specific Saved Weight Snapshots with the exact architecture state.
- User story 42 — Tests isolate snapshot writes under temporary directories.
- User story 43 — Concurrent same-mode writes remain complete and non-corrupt.
- User story 47 — Failed or abandoned work must not replace the prior successful snapshot.
- User story 50 — The latest successful finisher becomes the Saved Weight Snapshot.
- User story 51 — pytest, Ruff, and strict mypy remain the validation path.

## Constraints and Out of Scope

- Keep persistence local to the Phase 3 neural-network route boundary; do not create a general model registry or checkpoint framework.
- Do not add snapshot history, versions, manifests, metadata, cross-process file locking, inference loading, or continued-training support.
- Do not expose Saved Weight Snapshots through SSE.
- Do not implement HTTP request validation, route registration, event streaming, presentation delays, client-disconnect checks, or worker-thread orchestration in this ticket.
- Do not modify the frontend, BPE Tokenizer, Simple Chat, Word2Vec, transformer, matrix, or math utility behavior.

## Assumptions and Evidence Limitations

- The persistence operation may be implemented as a typed helper owned by the neural-network route module or another narrowly justified boundary already allowed by the specification.
- The exact temporary-file primitive is an implementation choice provided the observable same-directory atomic replacement and cleanup behavior is satisfied.
- The tests prove same-process concurrency. Cross-process locking is explicitly out of scope.

## Source

- [Phase 3 XOR Neural Network specification](../SPEC.md)
- [Canonical project context](../CONTEXT.md)
- [Ticket 004](004-provide-reference-compatible-xor-training-runs.md)

## Recommended Next Step

Run `to-plan-prompt` in a fresh conversation using this ticket, Ticket 004, the source specification, the canonical context, and the latest complete backend source snapshot.
