---
workflow: engineering-prompt-chain
document_type: grill_with_docs_result
prompt_name: grill-with-docs-prompt
status: confirmed
version: 1
recommended_next_prompt: to-spec-prompt
---

# Grill With Docs Result: Deliver Phase 4 Word2Vec Embeddings Through the Python Backend

## Original idea

Phase 1, Phase 2, and Phase 3 of the `llm-pipeline-explorer` migration are working through the Python Backend:

- Phase 1 provides the frontend server foundation and Python `POST /simple-chat`.
- Phase 2 provides the Python BPE tokenizer and `POST /bpe-tokenize`.
- Phase 3 provides the Python XOR neural network and `POST /neural-net`.

The proposed change is Phase 4: convert the TypeScript Reference Implementation for `train-embed` into Python Backend behavior while leaving the TypeScript/Vite frontend unchanged.

The browser will continue communicating through the existing HTTP and Server-Sent Events interface. The Python Backend must reproduce the frontend-visible behavior, deterministic educational Word2Vec results, and saved embedding-model artifact previously produced by the TypeScript reference code.

Phase 5, Transformer training, remains future work.

## Problem

The Python Backend already contains the intended destination files:

```text
backend/src/how_llms_work/ml/word2vec.py
backend/src/how_llms_work/routes/train_embed.py
```

but both are empty in the supplied current source. The FastAPI application does not register `POST /train-embed`, and the current request schemas do not define the Word2Vec request contract.

The unchanged TypeScript/Vite frontend already expects a precise contract:

- one validated `POST /train-embed` request;
- one `init` event;
- approximately fifty `epoch` events;
- one `done` event containing embeddings, nearest neighbors, similarities, analogies, and warnings.

The TypeScript Reference Implementation also establishes a deterministic training procedure and writes a complete embedding model before completion. A Python implementation that merely trains a generally valid Word2Vec model could still differ in preprocessing, random calls, weight initialization, update order, loss values, vectors, rankings, analogies, warnings, event order, or persisted JSON.

The migration therefore needs behavioral compatibility, not only conceptual equivalence.

## Desired outcome

A valid request to `POST /train-embed` starts one independent, deterministic Word2Vec Skip-gram Training Run through FastAPI.

The Python Backend will:

1. validate the exact existing request contract;
2. train from the fixed curated Embedding Training Corpus;
3. preserve the reference BPE preprocessing and Vocabulary order;
4. preserve the Mulberry32 random sequence and online negative-sampling update order;
5. stream reference-compatible initialization and loss updates;
6. construct the exact rounded Embedding Result expected by the frontend;
7. atomically save the complete Saved Embedding Model;
8. emit `done` only after persistence succeeds;
9. stop safely after disconnection or internal failure;
10. preserve all working Phase 1 through Phase 3 routes and tests.

The frontend remains unchanged and does not need to know that the backend implementation changed from Hono/TypeScript to FastAPI/Python.

## Primary users or stakeholders

- Learners using the **Train Embeddings** Learning Demo in the browser.
- The project owner maintaining the `llm-pipeline-explorer` Python Backend.
- The unchanged TypeScript/Vite frontend that consumes the `POST /train-embed` Frontend Contract.
- Future developers who may refactor or optimize the educational numerical implementation.
- Phase 5 Transformer work, which may later reuse the shared corpus and BPE-derived artifacts without being included in this phase.

## Confirmed scope

- Implement reusable Word2Vec behavior in:

  ```text
  backend/src/how_llms_work/ml/word2vec.py
  ```

- Implement FastAPI request orchestration, SSE streaming, disconnect behavior, failure handling, and persistence in:

  ```text
  backend/src/how_llms_work/routes/train_embed.py
  ```

- Add the Pydantic request model and exact frontend-facing aliases to:

  ```text
  backend/src/how_llms_work/schemas.py
  ```

- Register `POST /train-embed` without removing existing routers in:

  ```text
  backend/src/how_llms_work/main.py
  ```

- Preserve the fixed curated corpus and reference-compatible BPE preprocessing.
- Use the existing reusable BPE implementation where it matches the TypeScript Reference Implementation.
- Use NumPy `float64` arrays for the Word2Vec input and output weight matrices.
- Preserve deterministic TypeScript-compatible random behavior.
- Preserve all request fields, defaults, limits, SSE event names, payload shapes, event ordering, and presentation delays.
- Preserve duplicate Query Word entries and their order.
- Return warnings for unrecognized or multi-token Query Words.
- Construct embeddings, nearest neighbors, pairwise similarities, and predefined analogies.
- Save the complete model to:

  ```text
  backend/.data/embedding-weights.json
  ```

- Use complete same-directory temporary writes and atomic replacement.
- Add focused numerical, route, persistence, failure, disconnection, concurrency, integration, and regression tests.
- Validate the eventual implementation with pytest, Ruff, and strict mypy.

## Out of scope

- Any TypeScript/Vite frontend change.
- Phase 5 Transformer implementation.
- Node or TypeScript backend files in the current backend.
- Node worker threads, Python multiprocessing, or shared memory for Phase 4.
- Gensim, PyTorch, TensorFlow, LangChain, LangGraph, hosted embeddings, or another Word2Vec implementation.
- Batched, matrix-wide, or vectorized gradient updates that alter operation order.
- A redesigned optimizer, early stopping, gradient clipping, frequent-token subsampling, or hierarchical softmax.
- Loading the Saved Embedding Model.
- Resuming, fine-tuning, caching, or reusing a completed model.
- Model history, a model registry, version manifests, or checkpoint management.
- Application-level training queues, semaphores, global locks, timeouts, quotas, or rate limits.
- A new SSE `error` event.
- A frontend-visible model-download feature.
- A general-purpose matrix or machine-learning framework.
- Writing production code, implementation tickets, or the final specification during this workflow.

## Confirmed decisions

1. **Deterministic TypeScript compatibility:** Identical valid requests must reproduce the TypeScript Reference Implementation’s random sequence, update order, rounded public output, ordering, warnings, and Saved Embedding Model.

2. **Numerical representation:** Use separate NumPy `float64` input and output weight matrices because the TypeScript implementation uses `Float64Array`. Do not apply the `Float32Array` mapping from other phases to this feature.

3. **Fixed PRNG:** Reproduce Mulberry32 with seed `42`, including its 32-bit integer behavior and exact random-call order. Do not substitute `numpy.random.default_rng()`.

4. **Complete Frontend Contract:** Preserve the public endpoint, request fields, camelCase names, defaults, bounds, strict numerical typing, SSE names, field names, payload structures, event order, and approximately `20` milliseconds of presentation delay after each `epoch` event.

5. **Request schema:** The request contains:

   ```json
   {
     "words": ["king", "queen"],
     "epochs": 10000,
     "dimensions": 32,
     "windowSize": 2,
     "negativeSamples": 5
   }
   ```

   with these rules:

   ```text
   words             1–10 strings, each at least one character
   epochs            strict integer, 10–10,000, default 10,000
   dimensions        strict integer, 4–64, default 32
   windowSize        strict integer, 1–5, default 2
   negativeSamples   strict integer, 1–10, default 5
   ```

   Unknown extra object fields remain ignored. Numeric strings, booleans, and fractional numbers are rejected with HTTP `422`.

6. **Normalization boundary:** The backend does not trim, split, deduplicate, or remove submitted `words`. It lowercases each entry only for Vocabulary lookup and warning analysis. Warning text retains the originally submitted entry.

7. **Duplicate Query Words:** Duplicate entries are preserved as separate positional Query Words. They count toward the ten-entry maximum and may produce repeated embeddings, repeated neighbor groups, and same-word similarity pairs.

8. **Unrecognized Query Words:** A structurally valid Query Word that is not one recognized Vocabulary Token does not fail the request. It is excluded from embeddings, neighbors, and pairwise similarities and produces a warning in the final result.

9. **All-unrecognized completion:** A successful run may emit `done` with empty `embeddings`, `neighbors`, and `similarities` plus warnings. Predefined analogies remain independent of whether the submitted Query Words were recognized.

10. **Fixed Embedding Training Corpus:** Preserve every curated sentence and its order exactly. Query Words select results and never alter the training corpus.

11. **Reference BPE preprocessing:** Lowercase and join the full corpus, learn exactly `500` BPE merges, apply that ordered Merge Table to every lowercased sentence, and preserve Pre-token boundaries.

12. **Stable Vocabulary construction:** Count tokens in first-encounter order, sort by descending frequency, and preserve first-encounter order for frequency ties. The resulting ordered Vocabulary determines token indices.

13. **Reference Training Pair order:** Generate ordered `(target, context)` pairs by sentence order, token order, and context position within the requested window.

14. **Shared immutable preprocessing:** Corpus-derived sentences, Merge Table, Vocabulary, frequencies, token indices, and reusable token data may be computed once and shared only as immutable data.

15. **Request-owned mutable state:** Every Embedding Training Run owns its PRNG, mutable Training Pair order, `wIn`, `wOut`, epoch state, gradients, losses, and result state. Concurrent requests do not share mutable numerical state.

16. **Negative-sampling distribution:** Raise Vocabulary frequencies to the `0.75` power, normalize them, and preserve the reference cumulative sampling process.

17. **Weight initialization:** Initialize `wIn` and `wOut` as separate `float64` matrices using the reference random-call order and:

```text
scale = 0.5 / dimensions
weight = (random - 0.5) × scale
```

18. **Inclusive epoch semantics:** Train epochs inclusively from `0` through the requested `epochs` value.

19. **Reference shuffle:** Shuffle the mutable Training Pair sequence in place before each epoch with a Mulberry32-driven Fisher–Yates shuffle.

20. **Learning-rate schedule:** Decrease the learning rate linearly from `0.025` to `0.001` according to the reference epoch schedule.

21. **Online Skip-gram updates:** Process every Training Pair immediately. Do not accumulate a batch gradient or reorder coordinate updates.

22. **Positive-before-negative order:** Apply the positive target-context update before drawing and applying negative samples.

23. **Negative-sample draws:** Draw exactly `negativeSamples` candidates. When a candidate equals the true context, skip it without drawing a replacement.

24. **Coordinate update order:** Preserve dimension-by-dimension updates and use the saved pre-update input coordinate when updating the output weight coordinate.

25. **Reference sigmoid and loss:** Preserve the reference sigmoid clipping and loss formulas, including the `1e-10` protection term.

26. **Loss denominator:** Divide total epoch loss by the number of positive Training Pairs rather than the combined number of positive and negative examples.

27. **Progress reporting:** Use:

```text
report_step = max(1, floor(epochs / 50))
```

and include epoch `0` and the final requested epoch.

28. **Successful event flow:** A successful stream is:

```text
init → epoch × approximately 50 → done
```

The `init` payload contains:

```text
vocabSize
sentenceCount
embeddingDim
windowSize
totalPairs
```

29. **Public vector source:** A public Word Embedding is derived only from the corresponding `wIn` row and rounded to six decimals. `wOut` is never exposed.

30. **Loss and vector rounding:** Streamed loss and public embedding coordinates use TypeScript-compatible six-decimal rounding.

31. **Nearest-neighbor behavior:** Neighbor cosine scores are calculated from public six-decimal vectors, rounded to two decimals before ranking, stably sorted by descending rounded score, and limited to five results. The Query Word’s own Vocabulary Token is excluded.

32. **Pairwise similarities:** Build similarity records from all recognized Query Word positions, preserving duplicate positions. Scores use public six-decimal vectors and TypeScript-compatible two-decimal rounding.

33. **Analogy precision:** Preserve the TypeScript analogy computation exactly. The analogy query vector uses the selected raw `wIn` rows for `a - b + c`; candidate vectors come from the six-decimal public `wIn` representation. This precision detail supersedes any broader shorthand suggesting that the source analogy arithmetic itself is rounded first.

34. **Analogy set and order:** Evaluate the seven predefined analogies in their existing order. Exclude the three source tokens from candidate results, select the first Vocabulary candidate encountered when scores tie, and round the final score to two decimals.

35. **Two-level numerical testing:** Require exact equality for event order, integer structures, random sequence, rounded public payloads, rankings, warnings, and serialized model contents. Use explicit tight relative and absolute tolerances for verified unrounded `float64` intermediate values.

36. **Non-finite results:** A successful run must not serialize or emit `NaN` or infinity.

37. **Bounded worker-thread intervals:** Advance training through bounded same-process worker-thread intervals with `asyncio.to_thread()` so the async route regains control between public progress boundaries.

38. **Disconnection handling:** Check browser disconnection between intervals. After disconnection, stop advancing, do not persist, do not emit `done`, and discard request-owned state.

39. **Unexpected stream failures:** Log ordinary unexpected post-stream failures internally, terminate without `done`, and do not invent an `error` event or expose exception details, tracebacks, paths, or numerical state.

40. **Cancellation behavior:** Do not catch `BaseException` or suppress `asyncio.CancelledError` as an ordinary training failure.

41. **Saved Embedding Model:** Save a complete model containing:

```text
type
dimensions
vocab
merges
embeddings
```

The `embeddings` mapping contains a public six-decimal Word Embedding for every ordered Vocabulary Token, not only submitted Query Words.

42. **Destination:** Save the latest successful model at:

```text
backend/.data/embedding-weights.json
```

Resolve the backend root independently of the shell’s current working directory.

43. **Exact JSON document:** Serialize with two-space indentation, exactly one trailing newline, and non-finite values disallowed.

44. **Atomic persistence:** Serialize before touching the destination, write a complete document to a unique temporary file in the same directory, close it, and atomically replace the destination only after the write succeeds.

45. **Persistence-before-completion:** Emit `done` only after the Saved Embedding Model has been replaced successfully.

46. **Failure preservation:** Serialization, write, replacement, or cleanup failures must not replace the previous valid model. Temporary files are removed when cleanup succeeds.

47. **Persistence failure stream behavior:** Log the failure internally and terminate without `done` or a new SSE error event.

48. **Concurrent persistence:** Concurrent successful runs target one destination. Each file remains complete, and the last successful atomic replacement becomes the Saved Embedding Model.

49. **No saved-model loading:** The saved file is an inspectable latest-model artifact only. Phase 4 does not load, cache, resume, or use it to serve later requests.

50. **No application-level concurrency policy:** Do not add a semaphore, global training lock, queue, timeout, or rate limiter. Existing request bounds are Phase 4’s only application-level resource limits.

51. **Layered acceptance suite:** Require independent deterministic expectations for preprocessing, random behavior, numerical updates, result construction, HTTP/SSE behavior, persistence, concurrency, and regressions.

52. **Existing routes remain intact:** Registering `POST /train-embed` must preserve:

```text
GET  /health
POST /simple-chat
POST /bpe-tokenize
POST /neural-net
```

53. **No new dependency is currently required:** The supplied `pyproject.toml` already includes FastAPI, Pydantic, NumPy, pytest, Ruff, mypy, and an HTTP testing dependency. The implementation must inspect the actual project before changing dependencies.

## Current behavior verified from files or tools

- The supplied current Python Backend registers `POST /simple-chat`, `POST /bpe-tokenize`, and `POST /neural-net`, plus `GET /health`.
- `backend/src/how_llms_work/ml/bpe.py` contains the reusable deterministic educational BPE implementation.
- `backend/src/how_llms_work/ml/neural_net.py` contains the NumPy XOR training implementation.
- `backend/src/how_llms_work/routes/neural_net.py` already demonstrates bounded `asyncio.to_thread()` training intervals, disconnect checks, quiet post-stream failure handling, and atomic persistence-before-`done`.
- Existing tests cover Simple Chat, BPE, XOR numerical behavior, XOR persistence, and the XOR route.
- `backend/src/how_llms_work/ml/word2vec.py` is empty.
- `backend/src/how_llms_work/routes/train_embed.py` is empty.
- `backend/src/how_llms_work/main.py` does not currently include a Train Embed router.
- `backend/src/how_llms_work/schemas.py` does not currently contain a Train Embed request model.
- The current `.data` directory contains XOR weight files but no supplied `embedding-weights.json`.
- The TypeScript/Vite frontend sends requests to `/train-embed` and distinguishes initialization, epoch, and completion payloads by their fields.
- The TypeScript Reference Implementation uses Word2Vec Skip-gram with negative sampling, separate `Float64Array` input and output weights, Mulberry32, a fixed seed, a curated BPE-tokenized corpus, nearest-neighbor and analogy results, and model persistence.
- The TypeScript Reference Implementation saves `embedding-weights.json` before yielding its final Embedding Result.
- FastAPI supports streaming response bodies from generators and async generators.
- The original Word2Vec work establishes Skip-gram as an efficient method for learning continuous word representations, while the negative-sampling paper describes negative sampling as an efficient alternative training objective.
- No production code was written and no test, Ruff, or mypy command was executed as part of this grilling workflow.
- The user reports that Phases 1 through 3 are working; this workflow inspected the supplied code but did not independently execute the repository.

## Desired behavior

A valid request such as:

```json
{
  "words": ["king", "queen", "man", "woman"],
  "epochs": 10000,
  "dimensions": 32,
  "windowSize": 2,
  "negativeSamples": 5
}
```

returns HTTP `200` with the shared SSE headers and an ordered stream resembling:

```text
event: init
data: {
  "vocabSize": ...,
  "sentenceCount": ...,
  "embeddingDim": 32,
  "windowSize": 2,
  "totalPairs": ...
}

event: epoch
data: {"epoch": 0, "loss": ...}

event: epoch
data: {"epoch": ..., "loss": ...}

...

event: epoch
data: {"epoch": 10000, "loss": ...}

event: done
data: {
  "embeddings": [...],
  "neighbors": [...],
  "similarities": [...],
  "analogies": [...],
  "warnings": [...]
}
```

Before `done`, the backend atomically replaces:

```text
backend/.data/embedding-weights.json
```

with a complete document shaped like:

```json
{
  "type": "word2vec-skipgram",
  "dimensions": 32,
  "vocab": [],
  "merges": [],
  "embeddings": {}
}
```

Each request remains independent. A disconnected or failed request does not replace the saved model and does not emit `done`.

## Domain model

### Terms created or changed

- **Word2Vec Embeddings Demo:** The Learning Demo that trains word vectors with Skip-gram and negative sampling and displays relationships learned from a fixed corpus.
- **Embedding Training Corpus:** The fixed curated collection of sentences used for training; Query Words do not change it.
- **Word Embedding:** The six-decimal public `wIn` vector for one Vocabulary Token.
- **Skip-gram Training:** The online procedure that learns context relationships from ordered target-context Training Pairs.
- **Training Pair:** One ordered target-and-context Token relationship.
- **Negative Sample:** One sampled Vocabulary Token treated as a non-context example.
- **Embedding Training Run:** One independent deterministic Word2Vec execution for a validated request.
- **Embedding Epoch Update:** One streamed epoch-and-loss progress observation.
- **Query Word:** One submitted word-list entry; duplicate entries remain distinct positions.
- **Nearest Neighbor:** A Vocabulary Token ranked by reference-compatible rounded cosine similarity.
- **Similarity Pair:** Two recognized Query Word positions and their cosine-similarity score.
- **Vector Analogy:** One predefined `a - b + c` relationship evaluated against the learned Vocabulary.
- **Embedding Result:** The frontend-facing completion payload containing selected embeddings, neighbors, similarities, analogies, and warnings.
- **Saved Embedding Model:** The complete latest successfully persisted Vocabulary, Merge Table, dimensions, and embedding mapping.
- **Embedding Event Stream:** One `init`, approximately fifty `epoch` events, and one `done` for a successful run.
- **Deterministic Embedding Compatibility:** Exact deterministic observable compatibility after approved rounding, with tolerance-aware hidden numerical comparisons.

### Important relationships

- One valid browser request creates one Embedding Training Run.
- One Embedding Training Run uses exactly one validated hyperparameter set.
- Every Embedding Training Run uses the same fixed Embedding Training Corpus and immutable derived preprocessing.
- One Embedding Training Run owns one independent PRNG and two independent mutable weight matrices.
- One corpus sentence produces zero or more ordered Training Pairs.
- One Training Pair produces one positive update and up to the requested number of effective negative updates.
- One successful Embedding Training Run produces one Embedding Event Stream.
- One successful Embedding Event Stream begins with `init`, contains approximately fifty Epoch Updates, and ends with exactly one `done`.
- One Query Word may resolve to one Vocabulary Token or produce one warning.
- Duplicate Query Words may resolve to the same Vocabulary Token while remaining separate result positions.
- Recognized Query Words produce selected Word Embeddings and Nearest Neighbor groups.
- Every pair of recognized Query Word positions produces one Similarity Pair.
- The predefined analogy list is evaluated independently of the submitted Query Word list.
- One successful Embedding Training Run creates one complete Saved Embedding Model before `done`.
- The Embedding Result contains selected display data; the Saved Embedding Model contains the complete Vocabulary model.
- Concurrent Embedding Training Runs do not share mutable training state.
- Concurrent successful runs share only the final Saved Embedding Model destination; the last successful finisher wins.
- A disconnected or failed run produces no new Saved Embedding Model and no `done`.
- The TypeScript/Vite frontend consumes the Frontend Contract produced by the Python Backend.
- The TypeScript Reference Implementation defines compatibility expectations but has no runtime role in the Python Backend.

### Domain artifacts

- [CONTEXT.md](CONTEXT.md)
- [ADR 0001 — Preserve Deterministic Word2Vec Compatibility](docs/adr/0001-preserve-deterministic-word2vec-compatibility.md)

## Architectural decisions

- [ADR 0001 — Preserve Deterministic Word2Vec Compatibility](docs/adr/0001-preserve-deterministic-word2vec-compatibility.md)

  The Python implementation deliberately favors deterministic TypeScript-compatible behavior over a third-party Word2Vec library, NumPy-native random generation, batched gradients, or operation-reordering optimizations. The ADR also establishes exact rounded public compatibility, tolerance-aware hidden numerical comparisons, and persistence-before-`done`.

## Constraints

- Python 3.12 or newer.
- Poetry-managed project and dependencies.
- Windows 11 development environment.
- FastAPI and Uvicorn for the Python Backend.
- Pydantic for request validation.
- NumPy for numerical state.
- Shared Server-Sent Events formatting through the existing SSE helper.
- pytest and the project’s HTTP testing dependency for tests.
- Ruff and strict mypy.
- The frontend remains TypeScript/Vite.
- The Backend contains no TypeScript server implementation.
- Keep HTTP route orchestration separate from reusable Word2Vec mathematics.
- Preserve the current frontend route and public camelCase JSON fields.
- Preserve deterministic random-call and numerical operation order.
- Keep corpus-derived shared values immutable.
- Keep request-owned trainable values isolated.
- Do not expose internal model weights through the `done` event.
- Do not claim bit-for-bit equality for every hidden transcendental floating-point intermediate across runtimes.
- Do require exact equality for approved rounded public values and serialized artifacts.
- Use PowerShell and `poetry run` for eventual local validation commands.
- Do not claim test success unless the commands are actually executed successfully during implementation.

## Edge cases and failure behavior

- **Missing or invalid request fields:** FastAPI/Pydantic returns HTTP `422` before creating a stream.
- **Numeric string, boolean, or fractional hyperparameter:** Reject with HTTP `422`.
- **Boundary values:** Accept inclusive documented minima and maxima.
- **Unknown extra object field:** Ignore it without altering the public request contract.
- **Whitespace-only Query Word:** Structurally valid when non-empty, but normally unrecognized; retain it in the warning.
- **Uppercase Query Word:** Lowercase for lookup, while retaining the original submitted form for warnings.
- **Leading or trailing spaces:** Do not trim in the backend; lookup and warning analysis use the lowercased but otherwise preserved value.
- **Duplicate Query Word:** Preserve every occurrence and result position.
- **Unknown or multi-token Query Word:** Add a warning and omit that occurrence from embedding-derived result collections.
- **All Query Words unrecognized:** Complete with empty selected result collections and warnings when training, analogies, and persistence otherwise succeed.
- **Negative sample equals true context:** Skip it without replacement.
- **Equal rounded neighbor scores:** Preserve Vocabulary order.
- **Equal analogy scores:** Keep the first candidate encountered in Vocabulary order.
- **Non-finite calculation or serialization value:** The run must not complete successfully or replace the prior saved model.
- **Browser disconnect:** Stop between intervals, do not persist, and do not emit `done`.
- **Training failure after stream start:** Log internally and terminate without `done` or a new error event.
- **Result-construction failure:** Log internally, preserve the prior file, and terminate without `done`.
- **Serialization failure:** Preserve the previous model and create no completed replacement.
- **Temporary-file write failure:** Remove the temporary file when possible and preserve the previous model.
- **Atomic replacement failure:** Remove the temporary file when possible and preserve the previous model.
- **Cleanup failure combined with another persistence failure:** Preserve all failure information in internal logging while exposing none to the client.
- **Concurrent successful saves:** Leave one complete valid file; last successful replacement wins.
- **Concurrent failed and successful saves:** A failed run must not damage the successful run’s complete destination.
- **Request completes numerically but persistence fails:** It is not a successful completed Embedding Training Run and emits no `done`.
- **Cancellation:** Allow cancellation control flow to propagate after appropriate cleanup.

## Testing expectations

### Corpus and preprocessing

- Assert the exact curated corpus and sentence order.
- Assert the exact 500-rule BPE Merge Table or independently verified compatibility fixtures.
- Assert tokenization of representative sentences.
- Assert stable token-frequency and Vocabulary order.
- Assert token-index mappings.
- Assert Training Pair generation and order for each supported window size.
- Prove shared preprocessing cannot be mutated by one request.

### Randomness and numerical core

- Independently verify the Mulberry32 output sequence.
- Verify 32-bit wraparound behavior.
- Verify Fisher–Yates results for fixed small pair sequences.
- Verify the negative-sampling cumulative distribution.
- Verify `wIn`/`wOut` dimensions, dtype, separation, and initialization sequence.
- Verify the reference sigmoid boundaries.
- Verify one positive update independently.
- Verify one negative update independently.
- Verify pre-update coordinate use.
- Verify skip-without-replacement-draw behavior.
- Verify a short complete deterministic Training Run against independently calculated values.
- Verify inclusive epoch behavior.
- Verify the learning-rate schedule.
- Verify reporting boundaries, including epoch `0` and the final epoch.
- Verify loss rounding and non-finite protection.
- Use exact comparison for discrete behavior and rounded public values.
- Use explicit tight tolerances for unrounded `float64` values.

### Result construction

- Verify public vectors come only from `wIn`.
- Verify six-decimal vector rounding.
- Verify that `wOut` does not affect public vectors after training.
- Verify duplicate Query Word positions.
- Verify lowercase lookup without backend trimming.
- Verify warning wording and order.
- Verify unknown and all-unknown result behavior.
- Verify pairwise positional combinations.
- Verify two-decimal cosine rounding.
- Verify neighbor ranking uses rounded scores.
- Verify stable Vocabulary-order ties.
- Verify self-token exclusion and five-result limit.
- Verify the exact predefined analogy list and order.
- Verify raw-source-vector and rounded-candidate-vector analogy precision.
- Verify analogy source-token exclusion and tie behavior.

### HTTP and SSE contract

- Verify exact Pydantic fields, aliases, defaults, strictness, and inclusive bounds.
- Verify malformed requests fail before training.
- Verify `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.
- Verify exact successful event names and order.
- Verify exact `init`, `epoch`, and `done` fields.
- Verify the presentation sleep is requested after each `epoch`.
- Verify no additional events occur after `done`.
- Verify disconnect stops later intervals, persistence, and completion.
- Verify ordinary internal failures are logged and not exposed.
- Verify no invented SSE error event.
- Verify independent request state under concurrent or sequential calls.

### Persistence

- Verify destination resolution from the backend root rather than the shell working directory.
- Verify directory creation.
- Verify exact `type`, `dimensions`, `vocab`, `merges`, and `embeddings` keys.
- Verify full ordered Vocabulary coverage.
- Verify two-space indentation and exactly one trailing newline.
- Verify non-finite serialization rejection.
- Verify same-directory temporary files are unique and closed before replacement.
- Verify persistence occurs before `done`.
- Verify prior-model preservation after serialization, write, and replacement failures.
- Verify temporary-file cleanup.
- Verify one complete valid document after concurrent saves.
- Verify last-successful-finisher-wins behavior.

### Integration and regression

- Verify `POST /train-embed` is registered.
- Verify `/health`, `/simple-chat`, `/bpe-tokenize`, and `/neural-net` remain registered and retain their existing behavior.
- Run focused new tests.
- Run the complete pytest suite.
- Run Ruff.
- Run strict mypy.
- Perform a manual two-server browser/Vite-proxy check when practical.
- Record actual command results honestly; this grilling session provides requirements, not passing test evidence.

## Risks and safeguards

- **Risk — A valid but different Word2Vec implementation changes the lesson’s output.**
  - **Safeguard:** Protect the exact PRNG, corpus, preprocessing, update sequence, rounding, ordering, and persistence behavior through ADR 0001 and independent fixtures.

- **Risk — A future optimization consumes random values or updates weights in a different order.**
  - **Safeguard:** Test the random sequence, shuffle, short-run internal state, and exact rounded outputs separately.

- **Risk — Public vectors appear deterministic while hidden numerical behavior has drifted.**
  - **Safeguard:** Combine exact boundary assertions with tight internal `float64` tolerance checks.

- **Risk — Universal bit-for-bit hidden equality is impossible across JavaScript and Python math implementations.**
  - **Safeguard:** Require exact approved rounded results while selecting the smallest verified tolerances for unrounded calculations.

- **Risk — Training blocks the async event loop.**
  - **Safeguard:** Advance bounded training intervals through `asyncio.to_thread()` and return to the route between reports.

- **Risk — Python worker threads do not provide guaranteed multi-core speedup for Python-heavy loops.**
  - **Safeguard:** Treat thread offloading as an event-loop responsiveness and cancellation seam; defer process-based optimization until measured and separately designed.

- **Risk — Several maximum-size requests compete for CPU.**
  - **Safeguard:** Preserve the confirmed contract and bounded validation limits in Phase 4; treat production rate limiting as future operational work.

- **Risk — Shared preprocessing becomes mutable and leaks across requests.**
  - **Safeguard:** Represent shared derived artifacts with immutable structures or read-only arrays and test mutation isolation.

- **Risk — Duplicate Query Words create surprising repeated output.**
  - **Safeguard:** Preserve the reference behavior and cover it explicitly in tests rather than silently deduplicating.

- **Risk — HTTP `200` is misread as every Query Word having a vector.**
  - **Safeguard:** Preserve explicit warnings and test partially and fully unrecognized requests.

- **Risk — Directly writing the destination leaves partial JSON.**
  - **Safeguard:** Write and close a unique same-directory temporary file before atomic replacement.

- **Risk — A completed numerical run emits `done` even though saving failed.**
  - **Safeguard:** Make successful persistence a prerequisite for `done`.

- **Risk — Concurrent saves interleave or corrupt the model.**
  - **Safeguard:** Use independent temporary files and atomic last-successful-finisher replacement.

- **Risk — A post-stream exception exposes internal details.**
  - **Safeguard:** Log internally, terminate quietly, and do not create a new error payload.

- **Risk — Tests derive expected values by calling production helpers.**
  - **Safeguard:** Use independently calculated expected sequences and fixtures.

- **Risk — Full default training makes every test slow.**
  - **Safeguard:** Use small deterministic configurations for most numerical and failure tests, with a limited number of representative end-to-end cases.

- **Risk — Phase 4 expands into Transformer architecture or process infrastructure.**
  - **Safeguard:** Keep Transformer, multiprocessing, and shared memory explicitly outside this phase.

- **Risk — Documentation and current code disagree.**
  - **Safeguard:** Treat the latest complete Python export as current-code truth, the TypeScript files as behavior reference, and this confirmed result as the decision handoff for `to-spec-prompt`.

## Open questions

- None that block writing the Phase 4 specification.
- The implementation must establish independently verified deterministic fixture values from the TypeScript Reference Implementation or another independent reference calculation.
- Exact internal `rtol` and `atol` values are not product decisions. They must be selected during implementation from measured representative differences and must be tight enough to expose formula or ordering defects.
- The final specification may define typed class/function boundaries and precise test-file organization, provided it does not change the confirmed behavior.
- No implementation or validation commands were executed during this grilling workflow.

## Source material consulted

- `GRILL_WITH_DOCS_PROMPT.md`
- Current Phase 3 `SPEC.md`
- Updated root `CONTEXT.md`
- `docs/adr/0001-preserve-deterministic-word2vec-compatibility.md`
- Latest `py_llm_pipeline_explorer_file_structure.md`
- Latest `llm_works_file_structure.md`
- `backend/pyproject.toml`
- `backend/src/how_llms_work/main.py`
- `backend/src/how_llms_work/schemas.py`
- `backend/src/how_llms_work/sse.py`
- `backend/src/how_llms_work/ml/bpe.py`
- `backend/src/how_llms_work/ml/neural_net.py`
- `backend/src/how_llms_work/ml/word2vec.py`
- `backend/src/how_llms_work/routes/neural_net.py`
- `backend/src/how_llms_work/routes/train_embed.py`
- Existing Python tests for Simple Chat, BPE, XOR training, XOR persistence, and the XOR route
- TypeScript reference `src/routes/train-embed/corpus.ts`
- TypeScript reference `src/routes/train-embed/train.ts`
- TypeScript reference `src/routes/train-embed/index.ts`
- TypeScript reference `src/schemas/train-embed-request.ts`
- TypeScript frontend `src/client/hooks/use-train-embed-chat.tsx`
- TypeScript frontend Train Embed result component and shared SSE reader
- Official FastAPI documentation for `StreamingResponse`
- Official Starlette documentation for request-disconnection detection
- Official Python documentation for `asyncio.to_thread()`, cancellation, `tempfile`, and `os.replace()`
- Official NumPy documentation for `float64`, stable sorting, exact assertions, and tolerance-aware assertions
- Official Pydantic documentation for strict validation and field aliases
- Official pytest documentation for parametrization, temporary paths, and monkeypatching
- Mikolov et al., _Efficient Estimation of Word Representations in Vector Space_
- Mikolov et al., _Distributed Representations of Words and Phrases and their Compositionality_

## Recommended next step

Run `to-spec-prompt` using this file, the updated `CONTEXT.md`, ADR 0001, the latest complete Python Backend source export, and the latest TypeScript Reference Implementation as inputs.

The specification should remain limited to Phase 4 Word2Vec embeddings. It must not introduce frontend changes, Transformer implementation, multiprocessing, shared memory, model loading, caching, a job-management system, or a new SSE error-event contract.

`to-spec-prompt` is not included in the current prompt pack.
