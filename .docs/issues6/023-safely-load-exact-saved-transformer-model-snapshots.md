---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "023"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 023: Safely Load Exact Saved Transformer Model Snapshots

## What to build

Create the reusable trust boundary that turns one specifically named Saved Transformer Model into one request-owned inference snapshot. A caller must be able to request one exact approved filename, have it selected only from the genuine backend model directory, read exactly once, strictly validate the complete current Python Phase 5 format, and receive canonical `float32` parameters plus the model's ordered Vocabulary and Merge Table without mutating shared state.

This boundary must treat browser-controlled filenames and persisted JSON as untrusted. It must enforce exact capitalization even on Windows, reject path syntax and path indirection, reject ambiguous or incompatible JSON, derive all parameter lengths and ordering from the existing canonical Transformer layout, and leave every rejected artifact unchanged. It must not expose paths, exception details, model values, or numerical state through its public failure outcome.

## Acceptance Criteria

- [ ] A valid current-format file whose exact filename matches an ordinary entry inside the genuine resolved model directory is read once and produces one complete request-owned Saved Transformer Model snapshot.
- [ ] The snapshot preserves the exact validated configuration, ordered Vocabulary, ordered Merge Table, Transformer block count, and canonical parameter ordering from the file.
- [ ] Validated parameters are materialized as an independent canonical `float32` block and usable semantic views without aliasing global preprocessing, training state, another request, or the parsed JSON containers.
- [ ] The loader accepts only the approved configuration-specific Transformer filename grammar and rejects absolute paths, drive-letter forms, parent references, `/`, `\`, empty named strings, and unapproved filenames before opening a candidate.
- [ ] Named selection enumerates directory entries and requires exact string equality, so a differently capitalized filename is rejected even on a case-insensitive filesystem.
- [ ] A model directory or candidate that is a symbolic link or Windows junction is rejected, and the resolved candidate must remain inside the genuine resolved model directory.
- [ ] Only ordinary non-link files are eligible; directories, special files, and path-indirected candidates are not loaded.
- [ ] JSON objects with duplicate, missing, or unexpected keys, wrong container types, or unsupported values are rejected rather than normalized or repaired.
- [ ] Validation requires exactly the current top-level model shape, model type `decoder-transformer`, exact configuration fields, exact weight groups, exact block fields, and a block count equal to `numLayers`.
- [ ] Every expected parameter-array length and flattening position is derived from the existing canonical Transformer parameter-layout boundary rather than duplicated layout constants.
- [ ] Boolean, string, `null`, missing, NaN, infinite, or otherwise non-finite parameter entries are rejected; ordinary finite JSON numbers are accepted only at the exact required positions.
- [ ] The ordered Vocabulary length agrees with `vocabSize`, and the Vocabulary and Merge Table form a coherent model that can map token IDs and reconstruct output without guessing, dropping, or repairing data.
- [ ] Architecture fields encoded in the approved filename agree with the validated model configuration; the epoch filename segment remains artifact metadata rather than reconstructed model state.
- [ ] A selected file is parsed and validated from one in-memory read snapshot and is not reopened during that invocation.
- [ ] Repeating the public load operation rereads the disk file and observes changed contents; no loaded-model cache survives between invocations.
- [ ] No candidate is rejected solely because of a preconfigured application file-size limit; concrete read, memory, decode, parse, or validation failures produce a stable non-sensitive loading failure.
- [ ] Invalid candidates are not deleted, rewritten, repaired, renamed, or otherwise modified by the load attempt.
- [ ] Windows junction behavior is exercised when the test environment permits it; otherwise the public path-classification seam is tested and the platform limitation is recorded without claiming junction coverage ran.

## Testing Expectations

- **Approved test seam:** The specification's backend model-loading seam: focused pure selection and validation tests at the stable public reusable boundary, using temporary model directories and fixed valid and invalid Saved Transformer Model fixtures.
- **Behavior to verify:** Exact-case and safe-path selection, one-read/no-cache behavior, duplicate-key rejection, exact current-format validation, canonical length authority, finite numerical values, request-owned `float32` materialization, BPE coherence, filename/configuration agreement, artifact immutability, and sanitized failure outcomes.
- **Relevant prior art:** Existing Transformer persistence tests, canonical parameter-layout tests, temporary-directory persistence tests, duplicate-key-aware JSON/SSE parsing patterns, and deterministic public-model builders in the supplied Python Backend.
- **Do not test through:** Private helper names, a particular JSON-library implementation, local sorting or loop structure, private array-view construction, raw exception classes, or internal container identity beyond proving state isolation.

## Blocked By

- None — can start immediately.

## Constraints and Out of Scope

- Accept only complete current Python Phase 5 Saved Transformer Models; do not support, migrate, infer, or repair old TypeScript, partial, or alternate formats.
- Search only the real backend model directory resolved from application code; do not accept arbitrary filesystem paths.
- Do not follow symbolic links or Windows junctions.
- Do not cache loaded models across requests and do not add a model registry, manifest, database, deletion feature, or download feature.
- Do not implement HTTP route orchestration, SSE sequencing, prompt generation, latest-model fallback, frontend behavior, worker labeling, or training changes in this ticket.
- Use the existing Python 3.12+, NumPy, Pydantic, FastAPI, and standard-library dependency set; do not add a production dependency.

## Source

- `SPEC.md` — strict named selection, current-format model validation, one-snapshot, no-cache, numerical-layout, path-safety, and testing decisions.
- `GRILL_WITH_DOCS_RESULT.md` — confirmed safe-loading behavior and failure safeguards.
- `CONTEXT.md` — canonical Saved Transformer Model and Saved Transformer Generation Run terminology.
- ADR 0003 — stateless saved-model inference boundary.
- Latest Python Backend source export and TypeScript Reference Implementation supplied with the source specification.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
