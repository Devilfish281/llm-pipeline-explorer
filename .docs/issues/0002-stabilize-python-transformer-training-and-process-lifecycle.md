# ADR 0002 — Stabilize Python Transformer Training and Process Lifecycle

**Status:** Accepted

## Context

Phase 5 converts the educational decoder-only Transformer backend from the TypeScript Reference Implementation to the Python Backend while keeping the TypeScript/Vite frontend unchanged.

The reference implementation provides the required model architecture, corpus-derived BPE behavior, request fields, SSE payloads, training formulas, generated-text behavior, and saved-model structure. It also contains implementation details that are unstable across hosts or unsuitable for a Windows-first Python service:

- worker count and data partitioning depend on host CPU count;
- gradient-reduction order depends on worker partitioning;
- one mutable random stream is shared by initialization and generated-text sampling;
- existing model files may skip training;
- intermediate 100-epoch checkpoints are written;
- progress uses a fixed ten-epoch interval rather than a stable approximate report count;
- worker and shared-memory cleanup is not a complete Python lifecycle contract.

These choices are difficult to reverse after implementation because they determine numerical fixtures, shared-memory layout, process protocol, route failure behavior, persistence, and acceptance tests.

## Decision

### Compatibility boundary

Phase 5 uses **Transformer Training Compatibility**, not exact host-dependent TypeScript worker parity.

The Python Backend preserves:

- the exact five-field frontend request contract;
- the `POST /train-transformer` endpoint;
- `init`, `epoch`, and `done` SSE event names and payload shapes;
- the fixed story corpus and Transformer-specific BPE preprocessing;
- the decoder-only architecture and training formulas;
- TypeScript-compatible Mulberry32 and public decimal rounding;
- `float32` materialized numerical state;
- the named Saved Transformer Model structure;
- the educational behavior seen by the browser.

It intentionally stabilizes process partitioning, reporting, random-stream ownership, model loading, checkpointing, cleanup, failure handling, and persistence.

### Public request and stream

The public request contains only:

- `epochs`: strict integer, `50..2000`, default `300`;
- `temperature`: strict finite JSON number, `0.1..2.0`, default `0.8`;
- `topP`: strict finite JSON number, `0.1..1.0`, default `0.9`;
- `numLayers`: strict integer, `1..6`, default `2`;
- `maxTokens`: strict integer, `3..500`, default `40`.

Unknown fields are ignored. Numeric strings, Booleans, fractional integer fields, NaN, and infinity are rejected before streaming.

The successful event sequence is:

```text
init → approximately 50 epoch events → done
```

The first streamed event is one exact eleven-field `init` payload. It is prepared before `StreamingResponse` is returned and is yielded before shared-memory allocation or worker startup.

Progress uses:

```text
report_step = max(1, floor(epochs / 50))
```

Epoch zero, every report boundary, and the exact requested final epoch are emitted. Every `epoch` payload contains epoch, six-decimal loss, and one Generated Text Sample. The established 20-millisecond presentation delay is retained after each epoch event.

The final `done` payload contains exactly:

- `architecture`;
- six-decimal `finalLoss`;
- the complete ordered `samples` collection.

Final loss is recomputed after the last Adam update and `done` is emitted only after final-model persistence succeeds.

The existing shared `format_sse()` and `create_sse_response()` functions remain the sole SSE framing and response-header boundary.

### Fixed preprocessing and model

The fixed Transformer Training Corpus is processed into one immutable, lazily initialized, application-wide Transformer Preprocessing Snapshot.

Initialization uses a module-local `threading.Lock` with double-checked access. The snapshot is published only after complete construction, validation, and conversion of nested collections to immutable forms. Failed initialization publishes nothing and a later request may retry.

The snapshot contains the reference corpus, Transformer-specific BPE Merge Table, tokenized stories, ordered Vocabulary, token indices, complete token-ID sequence, Training Sequences, first three generation seed IDs, and four Logical Training Shard boundaries.

The model fixes:

- context length `32`;
- embedding dimension `32`;
- two attention heads;
- head dimension `16`;
- feed-forward dimension `128`;
- training sequence length `16`;
- requested layer count from one through six.

Only `numLayers` changes the architecture.

### Canonical weight layout and initialization

One stable layout builder is the sole authority for keys, block indices, float offsets, lengths, shapes, and total parameter count.

The flat order is:

1. `tokEmb`
2. `posEmb`
3. for each block: `ln1Gamma`, `ln1Beta`, `wQ`, `bQ`, `wK`, `bK`, `wV`, `bV`, `wO`, `bO`, `ln2Gamma`, `ln2Beta`, `ff1W`, `ff1B`, `ff2W`, `ff2B`
4. `lnFGamma`
5. `lnFBeta`
6. `headW`
7. `headB`

Views use NumPy `float32`, C order, and `byte_offset = float_offset × 4`.

Weight Initialization has a separate traversal matching TypeScript random-consumption order:

- blocks in ascending layer order;
- per block: `wQ`, `wK`, `wV`, `wO`, `ff1W`, `ff2W`;
- then `tokEmb`, `posEmb`, and `headW`.

Each Xavier coordinate is filled one at a time in explicit C-order using one Mulberry32 draw and:

```text
limit = sqrt(6 / (fan_in + fan_out))
value = (random × 2 - 1) × limit
```

The completed coordinate is stored immediately as `float32`. Bias and beta arrays are zero; Layer Normalization gamma arrays are one; deterministic fills consume no random values.

### Numerical contract

Materialized weights, activations, caches, logits, probabilities, attention matrices, activation gradients, parameter gradients, and optimizer buffers are `float32`.

Scalar statistics and reductions use Python `float` or NumPy `float64`. Completed tensors are explicitly stored as `float32`.

The Python implementation preserves the reference formulas and shapes but is not required to reproduce every scalar TypeScript loop or accumulation order. Carefully selected NumPy vectorization is permitted. Hidden unrounded numerical values use explicit tight tolerances; discrete behavior and rounded public or persisted values use exact fixtures.

`ml/matrix.py` contains only stateless, shape-checked NumPy primitives. Pure operations allocate independent C-contiguous `float32` outputs and leave inputs unchanged. Only helpers explicitly named `_in_place` may mutate a separate, non-overlapping destination. In-place additions use transactional `float64` calculation, `float32` candidate validation, and commit only after the complete result is finite.

Matrix products and column sums use `float64` accumulation and one completed `float32` materialization. `stable_row_softmax()` accepts finite scores plus intentional negative-infinity causal masks, rejects invalid rows, leaves scores unchanged, and returns a separate probability array.

The Transformer preserves:

- learned token and learned absolute positional embeddings;
- repeated-token gradient accumulation;
- per-position Layer Normalization across 32 features with population variance and epsilon `1e-5`;
- two-head causal scaled dot-product attention with scale `0.25`;
- exact causal masking and zero future probabilities and gradients;
- pre-normalized attention and feed-forward sublayers with residual connections;
- a `32 → 128 → 32` ReLU feed-forward network;
- final Layer Normalization and Vocabulary output head;
- stabilized row-wise softmax;
- average next-token cross-entropy using `1e-10`;
- the corresponding analytical backward pass.

Training uses unaveraged gradients, fixed sequence order, no clipping, no weight decay, no learning-rate schedule, no shuffling, and no early stopping.

Adam applies one update per inclusive epoch with:

- learning rate `0.001`;
- beta1 `0.9`;
- beta2 `0.999`;
- epsilon `1e-8`;
- optimizer step `epoch + 1`.

Adam uses the canonical flat arrays and exactly two reusable parent-local `float64` scratch arrays. First and second moments are stored as `float32` after their completed stages, and updated parameters are stored into the shared `float32` weight block. Complete weights and Adam buffers are checked for finiteness after every update and before sample generation.

All Transformer randomness uses the shared JavaScript-compatible Mulberry32 implementation owned by `ml/math_utils.py`. Word2Vec re-exports that same class. TypeScript-compatible decimal rounding also moves to `ml/math_utils.py`; existing Word2Vec and XOR wrapper names remain compatible.

### Logical shards and multiprocessing

Every training run uses exactly four fixed Logical Training Shards.

```text
shard_size = ceil(training_sequence_count / 4)
```

Shards are contiguous slices and are reduced in order `0 → 1 → 2 → 3`.

The actual worker count is calculated once:

```text
min(4, max(1, os.cpu_count() or 1))
```

Static assignment uses:

```text
worker_index = shard_id % actual_worker_count
```

Each worker processes assigned shards in ascending order. Worker count affects performance only and cannot change shard boundaries or reduction order.

Every request creates a Request-Scoped Worker Group with one through four non-daemonic processes through a local `multiprocessing.get_context("spawn")` context. Workers are not shared across requests.

Every request owns exactly five shared-memory blocks:

- one flat `float32` weight block;
- four flat `float32` gradient blocks, one per Logical Training Shard.

Adam moments and reduction workspaces remain parent-local. The parent is the sole creator and unlinking owner. Workers attach by generated name, validate minimum capacity and complete layout, close their handles in `finally`, and never unlink. Capacity may exceed the logical required bytes; all views remain limited to the exact canonical range.

Workers receive non-writeable weight views and write only to assigned gradient blocks. The parent alone modifies weights and Adam state, performs Ordered Gradient Reduction, generates samples, evaluates final loss, builds saved models, persists, and emits SSE.

Each shard gradient block is zeroed immediately before that shard is computed. Empty shards report zero loss and all-zero gradients.

The parent reduces gradients through one reusable `float32` workspace and accumulates shard losses separately in Python floating-point order. All reduced gradients and losses must be finite before Adam.

### Worker protocol

Each actual worker has one dedicated duplex `multiprocessing.Pipe`. The parent closes its local copy of the worker endpoint immediately after a successful start.

All startup and control records are top-level frozen, slotted dataclasses with protocol version `1`. Worker-originated records also identify `worker_index`. Exact types and fields are validated; unknown records or mismatches fail the run.

The protocol contains:

- `WorkerStartupConfig`
- `ReadyMessage`
- `ComputeMessage`
- `ResultMessage`
- `FailureMessage`
- `StopMessage`
- `StoppedMessage`

Control records carry only primitive values, stable enums, and immutable tuples. Numerical arrays remain in shared memory.

Workers follow:

```text
STARTING → READY → COMPUTING → READY → STOPPING → STOPPED
```

Only one compute command may be outstanding. Epoch pipelining is prohibited.

A worker sends `ready` only after all required shared-memory attachment, capacity, layout, dtype, contiguity, and ownership validation succeeds.

A worker sends exactly one `ResultMessage` after all assigned shards are completely computed and validated. That message is the parent’s commit marker for reading the corresponding gradient blocks. The parent rejects stale, duplicate, missing, malformed, non-finite, or unassigned shard results.

Worker failures are sanitized. `FailureMessage` contains only worker index, closed `StrEnum` phase and code, optional epoch, and shard IDs. It never contains exception text, traceback data, filesystem paths, shared-memory names, model state, or numerical values. Full exceptions remain in worker logs.

A controlled worker failure exits with status `1`; a successful cooperative `stop → stopped` shutdown exits with status `0`. Pipe messages and process exit codes are independent integrity signals.

### Async orchestration, deadlines, and cleanup

The parent polls worker pipes and process sentinels through `multiprocessing.connection.wait()` in `0.1`-second calls executed with `asyncio.to_thread()`. Browser disconnection is checked after every poll.

Deadlines measured with `time.monotonic()` are:

- 30 seconds for the complete worker group to become ready;
- five minutes for each complete four-shard epoch computation;
- five minutes for each individual Generated Text Sample;
- five minutes for final post-training evaluation.

Generated Text Sample creation and final evaluation run through `asyncio.to_thread()` and use a request-owned `threading.Event` for cooperative cancellation. Generation checks between tokens; final evaluation checks between Training Sequences. The route waits for a helper thread to return before releasing numerical memory it may still access.

Each sample starts from the first three corpus token IDs and otherwise preserves the reference latest-16-token, temperature-scaled, stable-softmax, stable-tie, top-p nucleus sampling algorithm. Each reported epoch uses an independent Mulberry32 stream seeded with:

```text
(42 + epoch) modulo 2³²
```

Each FastAPI process permits one active Transformer Training Run. A module-local `threading.Lock` is acquired nonblockingly before preprocessing. An overlap returns HTTP `429` before SSE or resource allocation and is not queued.

Preprocessing or `init` construction failure occurs before `StreamingResponse` and returns a sanitized HTTP `500`. Validation errors return HTTP `422`.

After streaming begins, disconnection, worker failure, timeout, non-finite state, generation failure, final-evaluation failure, serialization failure, write failure, or replacement failure terminates quietly:

- no new SSE `error` event;
- no `done`;
- no replacement of a prior model;
- no client-visible internal details.

Cleanup is non-short-circuiting and preserves the original outcome. It sets cancellation, attempts cooperative stop, waits up to two seconds, terminates surviving workers, waits two more seconds, kills any remaining worker, joins workers, records exit codes, closes pipe and process objects, releases array views, closes shared-memory handles, and attempts each unlink independently. Shared memory is released only after no worker remains alive. The run lock is released last. `asyncio.CancelledError` propagates after cleanup.

### Persistence

Every valid request trains from fresh weights. Saved models are never loaded, resumed, or used to skip training. Intermediate checkpoints are not written.

One final Saved Transformer Model is retained per training configuration at:

```text
backend/.data/transformer-weights-e{epochs}-l{numLayers}-d32-h2-ff128-ctx32.json
```

The path is resolved from the backend module location, not the process working directory.

The named JSON model preserves top-level order:

1. `type`
2. `config`
3. `vocab`
4. `merges`
5. `weights`

Nested configuration and weight keys use their established order. Six-decimal values use the shared TypeScript-compatible rounding helper and normalize signed zero to positive `0.0`.

Serialization uses two-space indentation, UTF-8, insertion-order keys, `allow_nan=False`, no key sorting, and exactly one trailing newline.

Persistence serializes completely in memory, securely creates a unique temporary file in the same `.data` directory, writes using `\n`, flushes, calls `os.fsync()`, closes the file, and uses `os.replace()` on the configuration-specific destination. A failed save removes the temporary file where possible and preserves the prior destination. Directory `fsync()` is not required.

### Module ownership

- `ml/math_utils.py`: shared Mulberry32 and TypeScript-compatible rounding.
- `ml/matrix.py`: stateless validated NumPy primitives.
- `ml/transformer.py`: immutable preprocessing, layout, initialization, forward and backward mathematics, Adam, generation, final evaluation, and saved-model construction.
- `ml/transformer_worker.py`: typed worker records, spawn entry point, shared-memory attachment and validation, shard computation, and worker-group supervision.
- `routes/train_transformer.py`: HTTP boundary, active-run reservation, SSE orchestration, disconnect and cancellation handling, presentation delay, persistence, logging, and quiet post-stream failures.
- `schemas.py`: exact Train Transformer request model.
- `main.py`: router registration only.
- `sse.py`: unchanged shared transport.

The frontend remains unchanged.

### Verification

Phase 5 uses layered pytest coverage:

- exact fixtures for preprocessing, layouts, random streams, report schedules, public rounding, samples, event fields, and deterministic serialization;
- tight tolerance-based fixtures for unrounded activations, gradients, losses, Adam state, and weights;
- real `"spawn"` processes, pipes, sentinels, and shared memory in the ordinary test suite;
- controlled route seams for expensive numerical work, clocks, delays, disconnection, and persistence;
- separately marked slow/manual maximum-configuration endurance tests.

The ordinary validation path remains:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

No command result is established by this ADR.

## Consequences

### Positive

- Mathematical partitioning and public behavior are stable across one through four workers.
- Windows-compatible process startup and explicit cleanup are first-class requirements.
- Parent-only weight mutation eliminates worker update races.
- Independent sample streams remove hidden random coupling.
- Final persistence cannot expose intentionally partial JSON.
- The frontend contract remains unchanged.
- Numerical and process boundaries can be tested independently.

### Negative

- Worker startup and shared-memory orchestration are more complex than a single-process implementation.
- One request uses at most four workers even on larger machines.
- Every reported epoch generates text, increasing runtime.
- `float64` calculation boundaries and transactional helpers use additional temporary memory.
- Only one Transformer Training Run may execute per FastAPI process.
- Strict deadlines may fail on severely overloaded machines.
- Saved model files accumulate across epoch and layer configurations.
- The stabilized Python contract intentionally differs from incidental TypeScript process behavior.
