# Grill With Docs Result: Phase 5 Transformer Behavior and Process Infrastructure

## Original idea

Convert the TypeScript decoder-only Transformer backend into the Python FastAPI backend as Phase 5, including from-scratch Transformer mathematics, multiprocessing, shared memory, SSE streaming, generated text, and final-model persistence.

The browser frontend remains TypeScript/Vite and continues communicating through the existing `POST /train-transformer` HTTP and SSE contract.

## Problem

The current Python Backend has completed the earlier Learning Demos, but its Transformer modules and route are empty and the Transformer router is not registered.

The TypeScript Reference Implementation contains the required educational model and frontend behavior, but its process partitioning, random-stream ownership, progress cadence, cache behavior, checkpointing, and cleanup are not stable enough to copy literally into a Windows-first Python multiprocessing design.

Phase 5 therefore needs to preserve the educational and frontend-facing Transformer behavior while explicitly defining deterministic, testable, and safely cleaned-up Python process infrastructure.

## Desired outcome

A valid `POST /train-transformer` request will:

1. validate the exact existing five-field frontend request;
2. reserve the process-local Transformer run slot;
3. retrieve immutable fixed-corpus preprocessing;
4. emit the exact `init` event;
5. create request-scoped shared memory and one through four spawned workers;
6. train a fixed decoder-only Transformer through four Logical Training Shards;
7. reduce gradients in stable order and apply Adam;
8. emit approximately fifty sampled `epoch` events;
9. recompute final loss from final weights;
10. atomically persist one complete configuration-specific model;
11. emit exactly one `done` event;
12. clean up every process, pipe, thread-offloaded operation, shared-memory block, and run-slot resource.

The unchanged frontend will not need to know that Hono/TypeScript worker threads were replaced by FastAPI/Python multiprocessing.

## Primary users or stakeholders

- Learners using the **Train Transformer** Learning Demo.
- The project owner maintaining the Python Backend on Windows 11.
- The unchanged TypeScript/Vite frontend consuming `POST /train-transformer`.
- Future maintainers of the educational Transformer, numerical tests, and process infrastructure.
- The Poetry, pytest, Ruff, and strict-mypy backend workflow.

## Confirmed scope

- Implement the complete Phase 5 Python backend vertical slice.
- Preserve the exact public endpoint and frontend field names.
- Preserve the fixed story corpus and Transformer-specific BPE preprocessing.
- Preserve the fixed decoder-only architecture and analytical forward and backward passes.
- Use NumPy and from-scratch educational operations rather than a machine-learning framework.
- Add deterministic Mulberry32 initialization and independent per-epoch sampling streams.
- Use exactly four fixed Logical Training Shards with Ordered Gradient Reduction.
- Use one through four request-scoped spawned processes.
- Use one shared weight block and four shared shard-gradient blocks.
- Use dedicated duplex pipes and typed worker protocol records.
- Stream `init`, sampled `epoch`, and `done` through the existing SSE helpers.
- Persist one complete final model per epoch/layer training configuration.
- Add layered numerical, process, route, persistence, failure, and regression tests.
- Update the project glossary and record ADR 0002.

## Out of scope

- Frontend TypeScript, JSX, hooks, components, styling, routes, or Vite-proxy changes.
- A TypeScript, Node, or Hono backend runtime.
- PyTorch, TensorFlow, JAX, scikit-learn, LangChain, LangGraph, hosted models, or another library that hides the educational Transformer.
- GPU or CUDA support.
- A configurable corpus, seed, learning rate, embedding dimension, head count, feed-forward dimension, context length, sequence length, optimizer, or checkpoint interval.
- Loading, caching, resuming, fine-tuning, or continuing from a Saved Transformer Model.
- Intermediate epoch checkpoints.
- A global model registry, manifest, rollback system, or download feature.
- A new SSE `error` event, heartbeat, event ID, retry field, or Transformer-specific transport.
- Cross-process file locking.
- A machine-wide training lock; the confirmed lock is per FastAPI process.
- Queuing overlapping Transformer requests.
- Maximum-configuration endurance testing in the ordinary pytest suite.
- Changes to completed Learning Demo behavior except the approved extraction of shared Mulberry32 and TypeScript-compatible rounding helpers.
- Specification writing, implementation tickets, production implementation, commits, or code review in this workflow.

## Confirmed decisions

1. **Stabilized compatibility:** Preserve model, request, SSE, numerical intent, and educational behavior, but replace incidental host-dependent TypeScript process behavior with a stable Python-owned contract.
2. **Fixed logical partitioning:** Use exactly four contiguous Logical Training Shards and always reduce `0 → 1 → 2 → 3`.
3. **Adaptive physical workers:** Use one through four actual workers based on `min(4, max(1, os.cpu_count() or 1))`; worker count cannot alter shard boundaries or reduction order.
4. **Static assignment:** Assign shards with `worker_index = shard_id % actual_worker_count`; each worker processes its shards in ascending order.
5. **Request-scoped resources:** Create fresh workers, pipes, weights, gradients, cancellation state, and shared memory for each request; share none between requests.
6. **Single active run:** Permit one active Transformer Training Run per FastAPI process and reject overlap with HTTP `429` without queuing.
7. **Fresh training:** Every valid request initializes new weights and performs full training; saved models never skip or resume a run.
8. **Final-only persistence:** Write no intermediate checkpoints; persist only the complete final model.
9. **Configuration-specific models:** Retain one final model per `epochs` and `numLayers` training configuration.
10. **Stable progress:** Use `max(1, floor(epochs / 50))`, including epoch zero and the exact final epoch.
11. **Sample every report:** Attach one Generated Text Sample to every emitted Transformer Epoch Update.
12. **Independent sample randomness:** Use one Mulberry32 stream per reported epoch with seed `(42 + epoch) modulo 2³²`.
13. **Exact request contract:** Preserve `epochs`, `temperature`, `topP`, `numLayers`, and `maxTokens`, defaults, bounds, aliases, strict typing, finite values, ignored extras, and pre-stream HTTP `422`.
14. **Fixed corpus and architecture:** Preserve the exact ordered story corpus, BPE preprocessing, Vocabulary order, context `32`, embedding dimension `32`, two heads, feed-forward dimension `128`, sequence length `16`, and configurable one-through-six layers.
15. **Reference optimizer:** Preserve unaveraged gradients, fixed sequence order, one Adam update per inclusive epoch, learning rate `0.001`, beta values `0.9` and `0.999`, epsilon `1e-8`, and step `epoch + 1`.
16. **Parent-only mutation:** Workers compute shard gradients; only the parent reduces gradients and mutates weights or Adam state.
17. **Spawn everywhere:** Use a request-local `"spawn"` multiprocessing context on every supported OS and do not change the application-wide start method.
18. **Dedicated pipes:** Use one duplex pipe per actual worker; do not use a shared queue or Manager.
19. **Typed protocol:** Use frozen, slotted, top-level dataclasses, protocol version `1`, worker identity, closed failure enums, and exact state validation.
20. **Result commit marker:** The parent reads shard buffers only after accepting one complete matching `ResultMessage`.
21. **Sanitized failures:** Worker protocol and SSE never expose exception text, tracebacks, paths, shared-memory names, or numerical state.
22. **Bounded asynchronous waits:** Poll pipes and process sentinels every `0.1` seconds through `asyncio.to_thread()` with disconnect checks.
23. **Deadlines:** Use one 30-second group-startup deadline and five-minute deadlines for each epoch computation, sample, and final evaluation.
24. **Cooperative thread cancellation:** Use a request-owned `threading.Event`; never forcefully kill a numerical helper thread.
25. **Bounded worker escalation:** Stop cooperatively, then terminate, then kill if required; forced termination always fails the run.
26. **Non-short-circuiting cleanup:** Continue cleanup after secondary errors, preserve the original outcome, and release the run lock last.
27. **Shared-memory ownership:** The parent creates and unlinks all five blocks; workers attach, validate, close, and never unlink.
28. **Canonical layout:** Use one exact flat layout builder for parameter offsets, shapes, total count, worker views, Adam traversal, and serialization.
29. **Reference initialization:** Preserve Mulberry32, Xavier formulas, exact matrix order, exact C-order draw-to-coordinate mapping, and immediate `float32` storage.
30. **Mixed precision:** Store materialized arrays as `float32`; use `float64` scalar reductions and approved calculation stages; do not require every TypeScript scalar loop.
31. **Exact model formulas:** Preserve learned token and positional embeddings, Layer Normalization, two-head causal attention, ReLU feed-forward layers, residuals, final head, cross-entropy, and analytical backward dependencies.
32. **Strict matrix boundary:** Keep `matrix.py` stateless, validated, non-coercing, non-broadcasting except explicit row bias, and explicit about pure versus `_in_place` mutation.
33. **TypeScript-compatible utilities:** Move the single Mulberry32 and decimal-rounding implementations to `math_utils.py` while preserving Word2Vec and XOR wrapper imports.
34. **Final-result order:** Recompute final loss after the final Adam update, persist successfully, then emit `done`.
35. **Quiet post-stream failure:** After SSE starts, failures end the stream without persistence, `done`, or a new SSE error event.
36. **Pre-stream failure:** Preprocessing or `init` construction failure returns a sanitized HTTP `500` before SSE or resource allocation.
37. **Deterministic persistence:** Serialize ordered UTF-8 JSON with two-space indentation, exactly one trailing newline, finite values only, same-directory temporary file, file `fsync()`, and `os.replace()`.
38. **No directory fsync:** Use the approved portable Windows-first file persistence boundary without mandatory directory synchronization.
39. **Project-root destination:** Resolve `backend/.data` from the module location, not the current working directory.
40. **Module ownership:** Keep shared utilities, matrix primitives, Transformer mathematics, process infrastructure, route orchestration, schemas, and SSE transport in their approved modules.
41. **Layered testing:** Use exact and tolerance-based fixtures, real spawned-process integration tests in ordinary pytest, controlled route seams, and separately marked maximum endurance tests.
42. **No dependency or frontend change:** Use the existing standard library, NumPy, FastAPI, Pydantic, pytest, Ruff, and mypy toolchain unless later implementation evidence proves an unavoidable dependency.

## Current behavior verified from files or tools

- The latest Python snapshot is a Python 3.12+, FastAPI, Pydantic, NumPy, SSE, Poetry, pytest, Ruff, and mypy backend intended to preserve the frontend contract.
- The current Python tree contains `math_utils.py`, `matrix.py`, `transformer.py`, `transformer_worker.py`, and `routes/train_transformer.py`.
- The latest snapshot shows `transformer.py` and `transformer_worker.py` empty; the Phase 5 numerical and process implementation is not present.
- The current project already has shared SSE helpers and completed route/persistence conventions from earlier phases.
- The latest existing glossary contains BPE, XOR, Word2Vec, compatibility, and migration terminology but does not yet contain the confirmed Transformer domain terms.
- The current `SPEC.md` is the Phase 4 Word2Vec specification and explicitly defers Transformer behavior and multiprocessing to Phase 5.
- The TypeScript reference contains a from-scratch decoder-only Transformer, multi-head causal attention, Layer Normalization, feed-forward layers, analytical backpropagation, Adam, top-p sampling, worker-thread data parallelism, shared numerical buffers, SSE streaming, and JSON model persistence.
- The TypeScript route emits `init`, `epoch`, and `done`, and the frontend consumes that contract.
- The TypeScript worker count and partitioning are host-dependent, and the reference includes cached-model skipping, intermediate checkpoints, a shared mutable random stream, and a fixed ten-epoch reporting interval.
- No Phase 5 production implementation was performed during this grill-with-docs workflow.
- No pytest, Ruff, mypy, browser, or two-server command was executed during this grill-with-docs workflow.

## Desired behavior

- `POST /train-transformer` becomes a complete registered FastAPI route without changing completed endpoints.
- Request validation occurs before run reservation and resource allocation.
- Immutable preprocessing and request-dependent `init` construction complete before SSE starts.
- The first body event is exactly one `init`.
- Four fixed shards are processed by one through four spawned workers through request-owned shared memory.
- The parent performs deterministic reduction and one Adam update per epoch.
- Every public progress event contains rounded loss and generated text.
- Final loss describes the final updated weights.
- Final model persistence succeeds before `done`.
- Disconnection and failure paths never produce a false completion or leak internal details.
- Every run releases its workers, pipes, helper thread work, shared memory, temporary files, and process-local lock.

## Domain model

### Terms created or changed

- **Transformer Training Corpus:** Fixed ordered stories used for Transformer BPE and Training Sequences.
- **Transformer Training Sequence:** Fixed-length token-ID input and next-token target sequence.
- **Transformer Preprocessing Snapshot:** Immutable application-wide derived corpus data.
- **Transformer Training Run:** One full fresh-weight training execution.
- **Generated Text Sample:** Autoregressive text produced at each public report.
- **Sample Random Stream:** Independent per-epoch Mulberry32 sampler.
- **Transformer Epoch Update:** Public epoch, six-decimal loss, and sample.
- **Transformer Event Stream:** `init → epoch × approximately 50 → done`.
- **Saved Transformer Model:** Complete final named model for one configuration.
- **Logical Training Shard:** One of four fixed contiguous sequence subsets.
- **Ordered Gradient Reduction:** Canonical shard reduction `0 → 1 → 2 → 3`.
- **Request-Scoped Worker Group:** Per-run one-through-four process group.
- **Request-Scoped Shared Memory:** Per-run shared weight block and four shard-gradient blocks.
- **Transformer Training Compatibility:** Reference model and frontend compatibility plus stable Python-owned process and lifecycle rules.

### Important relationships

- One Transformer Preprocessing Snapshot is reused by many Transformer Training Runs.
- One Transformer Training Run owns one Request-Scoped Worker Group and five Request-Scoped Shared Memory blocks.
- One Transformer Training Run always has exactly four Logical Training Shards.
- One actual worker may execute multiple Logical Training Shards.
- One Logical Training Shard always writes to its own gradient block.
- The parent owns all weight and Adam mutation.
- One Transformer Epoch Update contains exactly one Generated Text Sample.
- One successful Transformer Training Run persists one Saved Transformer Model for its training configuration.
- Saved Transformer Models do not influence later Transformer Training Runs.

### Domain artifacts

- [CONTEXT.md](CONTEXT.md)
- [ADR 0002 — Stabilize Python Transformer Training and Process Lifecycle](docs/adr/0002-stabilize-python-transformer-training-and-process-lifecycle.md)

## Architectural decisions

- [ADR 0002 — Stabilize Python Transformer Training and Process Lifecycle](docs/adr/0002-stabilize-python-transformer-training-and-process-lifecycle.md)

ADR 0002 passed the decision gate because its process partitioning, shared-memory ownership, deterministic reduction, concurrency, caching, checkpoint, timeout, failure, and persistence choices are difficult to reverse, surprising without context, and involve material tradeoffs.

## Constraints

- Python 3.12 or newer.
- Windows 11 is the primary development environment.
- FastAPI/Uvicorn is the only backend runtime.
- Poetry manages Python dependencies and the virtual environment.
- NumPy provides numerical arrays and operations.
- Pydantic provides request validation.
- Python `multiprocessing` and `multiprocessing.shared_memory` provide process infrastructure.
- pytest, Ruff, and strict mypy remain required checks.
- The frontend remains TypeScript/Vite.
- The educational algorithms remain visible and from scratch.
- Existing completed endpoints and shared SSE behavior must not regress.
- No command result may be claimed unless actually executed.
- Phase 5 is the final confirmed phase; no future backend phase is included.

## Edge cases and failure behavior

- Numeric strings, Booleans, non-finite values, fractional integer fields, and out-of-range fields return HTTP `422`.
- An overlapping valid request returns HTTP `429` immediately and is not queued.
- Preprocessing or `init` construction failure returns sanitized HTTP `500`.
- A startup failure after `init` produces no epoch and no `done`.
- Empty Logical Training Shards produce zero loss and zero gradients.
- Larger platform shared-memory allocations are accepted when minimum required capacity exists; extra bytes are unused.
- Stale, duplicate, missing, unassigned, malformed, or non-finite worker results fail the run.
- A worker that reports failure exits nonzero; a clean `stop → stopped` worker exits zero.
- A startup, epoch, sample, or final-evaluation timeout fails the run.
- A browser disconnect sets cooperative cancellation and prevents later work.
- A helper thread may finish one already-started forward pass before stopping.
- Forced worker termination prevents persistence and `done`.
- Non-finite gradients, losses, weights, or Adam state fail at the earliest confirmed boundary.
- Generation preserves intentional negative-infinity causal masks and rejects invalid softmax rows.
- Repeated token IDs accumulate each embedding-gradient contribution.
- Persistence failure preserves the prior configuration-specific model and removes temporary files where possible.
- Cleanup failures are logged independently and cannot prevent later cleanup or final run-lock release.
- Post-stream failures terminate quietly and expose no internal details.

## Testing expectations

- Exact request validation, aliases, defaults, bounds, and ignored extras.
- Exact immutable preprocessing, Vocabulary, token IDs, Transformer Training Sequences, generation seeds, and shard boundaries.
- Exact Mulberry32 streams, wraparound behavior, draw counts, and per-epoch sample seeds.
- Exact TypeScript-compatible decimal rounding, including negative half ties and signed zero.
- Exact layouts for `numLayers` one through six, with no gaps or overlaps.
- Exact initialization random-consumption order and selected coordinates.
- Matrix shape, dtype, contiguity, aliasing, purity, transactionality, and finiteness tests.
- Tolerance-based Layer Normalization, attention, feed-forward, output-head, cross-entropy, embedding-gradient, shard-gradient, Adam, and final-evaluation tests.
- Exact causal-mask probabilities and gradients.
- Exact report epochs and SSE field sets.
- Exact generated text in the supported reference environment.
- Exact deterministic JSON order, formatting, rounding, and trailing newline.
- Real `"spawn"` tests for `ready`, `compute`, `result`, `failure`, `stop`, `stopped`, sentinels, exits, timeouts, and cleanup.
- One-through-four worker tests proving worker-count-independent public results.
- Route tests for `422`, `429`, sanitized pre-stream `500`, exact event order, persistence-before-`done`, quiet failures, disconnect, cancellation, and run-lock release.
- Persistence tests using `tmp_path`.
- Completed-route regression tests for Health, Simple Chat, BPE, XOR, and Word2Vec.
- Ordinary `poetry run pytest` includes bounded real process tests.
- Maximum-size endurance is marked slow/manual.
- Final implementation validation commands:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

## Risks and safeguards

- **Risk:** Host CPU count changes numerical results.
  - **Safeguard:** Four fixed shards and Ordered Gradient Reduction.
- **Risk:** Worker completion order changes updates.
  - **Safeguard:** Parent validates all results and reduces only `0 → 1 → 2 → 3`.
- **Risk:** A worker mutates shared weights.
  - **Safeguard:** Parent-only mutation, non-writeable worker views, module boundaries, and tests.
- **Risk:** Stale gradients leak between epochs.
  - **Safeguard:** Zero each complete shard-gradient block before every computation.
- **Risk:** A partial shared-memory result is read as complete.
  - **Safeguard:** One validated `ResultMessage` is the commit marker.
- **Risk:** A process hangs indefinitely.
  - **Safeguard:** Startup and compute deadlines plus bounded stop/terminate/kill escalation.
- **Risk:** A parent-side numerical helper blocks the event loop or survives disconnect.
  - **Safeguard:** `asyncio.to_thread()`, 0.1-second polling, deadlines, and cooperative cancellation.
- **Risk:** Shared-memory blocks leak or are removed by the wrong process.
  - **Safeguard:** Parent-only creation/unlink ownership and non-short-circuiting cleanup.
- **Risk:** Concurrent requests multiply CPU and memory usage.
  - **Safeguard:** One nonblocking process-local run lock and HTTP `429`.
- **Risk:** Random consumption in one sample changes later samples.
  - **Safeguard:** Independent per-epoch Sample Random Streams.
- **Risk:** NumPy vectorization causes hidden numerical drift.
  - **Safeguard:** Explicit mixed precision, exact public fixtures, and tight hidden tolerances.
- **Risk:** A harmless layout refactor corrupts parent-worker interpretation.
  - **Safeguard:** One canonical testable layout builder.
- **Risk:** Python and JavaScript rounding differ.
  - **Safeguard:** One shared TypeScript-compatible rounding helper.
- **Risk:** Saved JSON is partial or nondeterministic.
  - **Safeguard:** Complete deterministic serialization, same-directory temporary file, file `fsync()`, and `os.replace()`.
- **Risk:** A failed run destroys a previous model.
  - **Safeguard:** Replace only after complete successful persistence.
- **Risk:** A post-stream failure leaks sensitive internal state.
  - **Safeguard:** Sanitized protocol records, internal logging, and quiet SSE termination.
- **Risk:** Mock-only tests miss Windows spawn failures.
  - **Safeguard:** Bounded real-spawn tests in ordinary pytest.
- **Risk:** Scope expands into a framework or frontend redesign.
  - **Safeguard:** Explicit module ownership and out-of-scope boundaries.

## Open questions

- None block writing the Phase 5 specification.
- Exact independently captured fixture values and per-boundary `rtol` and `atol` values must be calibrated during implementation from small verified cases. This is an implementation calibration task, not an unresolved product decision.
- Exact internal function signatures and test-file decomposition may be refined by `to-spec-prompt` without changing the confirmed behavior.
- No Phase 5 code was implemented and no validation command was run during this grill-with-docs workflow.

## Source material consulted

- `GRILL_WITH_DOCS_PROMPT.md`
- Current Phase 4 `SPEC.md`
- Existing root `CONTEXT.md`
- Latest `py_llm_pipeline_explorer_file_structure.md`
- Latest `llm_works_file_structure.md`
- Current backend project structure, source modules, dependencies, shared SSE helpers, completed route conventions, persistence conventions, and tests
- TypeScript Train Transformer request schema
- TypeScript Train Transformer route and frontend consumer
- TypeScript Transformer corpus, matrix helpers, weight layout, model, trainer, worker, generation, and persistence behavior
- Official FastAPI documentation for routers, exceptions, validation, and streaming responses
- Official Starlette documentation for requests, disconnection, and streaming responses
- Official Pydantic documentation for strict mode, aliases, extra fields, finite values, and before validators
- Official Python documentation for multiprocessing, pipes, process sentinels, spawn, shared memory, threads, cancellation, temporary files, `fsync()`, and `os.replace()`
- Official NumPy documentation for dtypes, matrix multiplication, reductions, ufuncs, memory overlap, repeated-index accumulation, finite checks, softmax-related operations, and tolerance-aware assertions
- Official pytest documentation for `tmp_path`, monkeypatching, and markers
- Glorot and Bengio, _Understanding the Difficulty of Training Deep Feedforward Neural Networks_
- Kingma and Ba, _Adam: A Method for Stochastic Optimization_
- Ba, Kiros, and Hinton, _Layer Normalization_
- Vaswani et al., _Attention Is All You Need_
- Holtzman et al., _The Curious Case of Neural Text Degeneration_

## Recommended next step

Run `to-spec-prompt` using this file, the updated `CONTEXT.md`, ADR 0002, the latest complete Python Backend source export, the current Phase 4 `SPEC.md`, and the latest TypeScript Reference Implementation.

The resulting specification must remain limited to Phase 5 Transformer behavior and process infrastructure. It must not introduce frontend changes, a TypeScript backend, model loading or resuming, intermediate checkpoints, third-party machine-learning frameworks, a new SSE event, or later phases.
