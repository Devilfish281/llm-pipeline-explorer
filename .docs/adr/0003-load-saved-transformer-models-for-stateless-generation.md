---
status: accepted
date: 2026-07-30
---

---

# ADR 0003 — Load Saved Transformer Models for Stateless Generation

## Context

Phase 5 saves complete Transformer models under configuration-specific JSON filenames, while every Transformer Training Run begins with fresh weights. Phase 6 requires users to load one of those completed models and generate text without starting another training run or turning the saved model into a checkpoint.

## Decision

Add a separate `POST /load-transformer` endpoint for stateless Saved Transformer Generation Runs.

The existing Transformer input box will recognize case-insensitive commands in this form:

```text
File: filename | starting text | temperature top-P max-tokens
```

An empty filename after `File:` selects the newest strictly valid Saved Transformer Model. A supplied filename selects only that exact file and must never silently fall back to another model.

The loader will search only the real `.data` directory, reject symbolic links, junctions, path separators, parent-directory references, absolute paths, incorrectly capitalized filenames, malformed files, incompatible files, and files that do not match the current Python Phase 5 model format. Automatic latest-model selection will examine candidates from newest to oldest, skip invalid candidates, and use the alphabetically greatest filename when valid files have equal modification times.

Each request will read one file snapshot, strictly validate it, tokenize a non-empty prompt of no more than 16 tokens, and perform deterministic generation with seed `42`. Loading and generation will run in the backend parent process without creating Transformer worker processes. The loaded model will not be cached after the request.

`POST /train-transformer` and `POST /load-transformer` will share one nonblocking Transformer request slot. Saved-model generation will use the ordered SSE sequence `loaded → result → done`, or one sanitized `error` event on failure. Disconnects and the five-minute generation deadline will stop cooperatively between token calculations and release the shared request slot.

Transformer training remains unchanged: every training request starts with fresh weights, uses its request-scoped worker group, and never loads a Saved Transformer Model. The first public training sample will display `Transformer worker processes: <count>` as presentation text only.

## Consequences

The frontend requires a small command-parser and event-handling change, but no new input box, selector, page, or layout.

Saved-model requests are deterministic, stateless, and isolated from training. Reading and validating the model on every request favors correctness and simplicity over caching speed.

ADR 0002 remains the authority for Transformer training and process lifecycle. This ADR governs only inference from completed Saved Transformer Models.
