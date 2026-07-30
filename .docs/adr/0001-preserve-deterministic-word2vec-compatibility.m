# ADR 0001: Preserve Deterministic Word2Vec Compatibility

**Status:** Accepted

## Context

Phase 4 replaces the TypeScript `train-embed` backend with a Python FastAPI implementation while retaining the existing TypeScript/Vite frontend and its observable contract. A mathematically valid Word2Vec implementation could still produce different losses, vectors, rankings, analogies, warnings, SSE output, or saved model data if it changed the random-number generator, numerical precision, corpus preparation, update order, rounding sequence, or ranking rules.

## Decision

The Python Word2Vec implementation shall preserve deterministic compatibility with the TypeScript Reference Implementation. It shall retain the fixed corpus and BPE preprocessing order, ordered vocabulary and Training Pairs, Mulberry32 seed `42`, reference-compatible Fisher–Yates shuffling, NumPy `float64` input and output weight matrices, Skip-gram negative-sampling formulas and update order, inclusive epoch schedule, and exact public rounding and tie-breaking rules.

Public Word Embeddings shall be six-decimal vectors derived only from the trained input-weight matrix. Identical valid requests shall produce exact rounded SSE payloads, result ordering, warnings, and Saved Embedding Model contents. Unrounded internal floating-point comparisons may use explicitly verified tight tolerances rather than claiming universal bit-for-bit equality across JavaScript and Python runtimes.

The implementation shall preserve the complete `POST /train-embed` Frontend Contract and save the complete model to `backend/.data/embedding-weights.json` before emitting `done`. Persistence shall use complete same-directory temporary writes followed by atomic replacement so failed runs preserve the previous valid model.

## Consequences

The implementation will use explicit reference-compatible numerical loops instead of a third-party Word2Vec library, NumPy’s random generator, batched gradient updates, or optimizations that change operation order. This favors reproducibility, educational consistency, and frontend compatibility over maximum training performance and conventional Python numerical design.

Independent deterministic tests must protect the random sequence, preprocessing, training updates, rounded public output, SSE contract, and saved-model schema. A future change that intentionally alters these compatibility rules should supersede this ADR rather than silently modifying the implementation.
