---
workflow: engineering-prompt-chain
document_type: specification
prompt_name: to-spec-prompt
status: ready-for-agent
triage_label: ready-for-agent
version: 3
source_document: GRILL_WITH_DOCS_RESULT.md
recommended_next_prompt: to-tickets-prompt
---

# Specification: Deliver the Phase 3 XOR Neural Network Through the Python Backend

## Problem

Phase 1 and Phase 2 are working: the TypeScript/Vite frontend can use the Python/FastAPI `POST /simple-chat` and `POST /bpe-tokenize` endpoints through the established HTTP and Server-Sent Events infrastructure. The next learning demo, XOR Neural Network, still has no Python implementation because these destination modules are empty:

```text
backend/src/how_llms_work/ml/neural_net.py
backend/src/how_llms_work/routes/neural_net.py
```

The current FastAPI application also does not register `POST /neural-net`.

The unchanged frontend already depends on a precise Frontend Contract derived from the TypeScript Reference Implementation. It sends one of two model modes and an epoch count, consumes approximately fifty streamed loss updates, and expects one final payload containing an exact architecture label, four ordered XOR predictions, and exact success or failure wording. The reference backend also saves the completed weights to a mode-specific JSON file before it emits the final `done` event.

A Python implementation that changes the training order, numerical formulas, epoch numbering, rounding, event sequencing, final strings, prediction order, or Saved Weight Snapshot schema can break the educational demonstration even when the replacement network appears generally correct.

The work must remain focused on a small, inspectable XOR implementation. It must not expand into Word2Vec, transformer training, a general matrix framework, multiprocessing, configurable optimizers, frontend redesign, or loading saved models for additional training.

## Solution

Implement one focused Phase Migration that converts the TypeScript XOR neural-network behavior into the Python Backend while leaving the TypeScript/Vite frontend unchanged.

For each valid `POST /neural-net` request, the Python Backend will:

1. Validate the requested model mode and epoch count.
2. Create a fresh randomized NumPy `float32` Weight Initialization.
3. Train either the Single-Layer Mode or Multi-Layer Mode on the four XOR examples.
4. Preserve the reference sample order, sigmoid mathematics, mean-squared loss, learning rate, immediate per-example updates, and backpropagation order.
5. Report loss at the reference-compatible epoch schedule.
6. Stream each report as an `epoch` SSE event, with a 20-millisecond production presentation delay.
7. Cooperatively stop between training intervals when the browser disconnects.
8. Produce the exact ordered predictions, architecture label, and Training Verdict expected by the frontend.
9. Convert the final NumPy weights to the exact plain-JSON Saved Weight Snapshot schema.
10. Atomically replace the mode-specific snapshot under `backend/.data/`.
11. Emit exactly one `done` event only after persistence succeeds.
12. Discard all in-memory request state when the request completes, disconnects, or fails.

The reusable numerical behavior will be owned by `backend/src/how_llms_work/ml/neural_net.py`. The FastAPI route will own request orchestration, same-process worker-thread offloading, client-disconnect checks, SSE construction, presentation delays, atomic persistence, and the final completion event.

Tests will be self-contained Python tests. Production will use fresh random initialization, while tests will inject a seeded NumPy `Generator` to prove the core educational contrast deterministically:

```text
Single-Layer Mode → FAILED
Multi-Layer Mode  → SUCCESS
```

Tiny floating-point differences from the TypeScript implementation are allowed. Tests will use explicit numerical tolerances for hidden calculations and exact equality for the serialized frontend and persistence contracts.

## User Stories

1. As a learner, I want to select Single-Layer Mode, so that I can observe why a network without a hidden layer cannot reliably learn XOR.
2. As a learner, I want to select Multi-Layer Mode, so that I can observe a `2 → 4 → 1` network learn XOR through backpropagation.
3. As a learner, I want the network to train on the four XOR input and target pairs in a visible, predictable order.
4. As a learner, I want progress loss values during training, so that I can see whether the network is improving.
5. As a learner, I want progress updates to include epoch zero and the requested final epoch, so that the graph includes the beginning and end of training.
6. As a learner, I want approximately fifty progress updates rather than one event per epoch, so that browser rendering remains manageable.
7. As a learner, I want each displayed loss rounded to six decimal places, so that the graph uses the established level of precision.
8. As a learner, I want the four final predictions ordered as `[0,0]`, `[0,1]`, `[1,0]`, and `[1,1]`, so that I can compare them directly with the XOR truth table.
9. As a learner, I want each final actual prediction rounded to two decimal places, so that the result remains readable.
10. As a learner, I want a clear success or failure verdict, so that the educational outcome is immediately understandable.
11. As a learner, I want Single-Layer Mode to retain its exact explanatory failure wording when it does not learn XOR.
12. As a learner, I want Multi-Layer Mode to retain its exact success wording when backpropagation learns XOR.
13. As a learner, I want a completed run to create an inspectable JSON file containing the learned weights, so that I can see that a trained model is represented by stored parameters.
14. As a learner, I want the single-layer snapshot to contain two weights and one bias, so that its minimal architecture is visible.
15. As a learner, I want the multi-layer snapshot to expose its input-to-hidden weights, hidden biases, hidden-to-output weights, and output bias.
16. As the TypeScript/Vite frontend, I want to send `POST /neural-net` without modification.
17. As the TypeScript/Vite frontend, I want `mode` to remain either `single-layer` or `multi-layer`.
18. As the TypeScript/Vite frontend, I want an omitted `epochs` value to default to `5000`.
19. As the TypeScript/Vite frontend, I want invalid modes and invalid epoch values to return FastAPI/Pydantic HTTP `422`.
20. As the TypeScript/Vite frontend, I want a valid request to return `text/event-stream`.
21. As the TypeScript/Vite frontend, I want cache-prevention and proxy-buffering headers preserved through the existing shared SSE response helper.
22. As the TypeScript/Vite frontend, I want the successful event order to be `epoch × N → done`.
23. As the TypeScript/Vite frontend, I want every `epoch` payload to contain only `epoch` and `loss`.
24. As the TypeScript/Vite frontend, I want the final `done` payload to contain `architecture`, `predictions`, and `verdict` without the saved weights.
25. As the TypeScript/Vite frontend, I want the exact architecture strings preserved.
26. As the TypeScript/Vite frontend, I want the exact verdict strings preserved because success styling depends on the `SUCCESS` prefix.
27. As an API client, I want `epochs` to be an integer from `100` through `100000`.
28. As an API client, I want no new required fields such as seed, learning rate, hidden size, or output path.
29. As an API client, I want separate requests to use independent Weight Initializations and training state.
30. As a project developer, I want the XOR mathematics isolated from HTTP and SSE behavior, so that the training algorithm can be tested directly.
31. As a project developer, I want NumPy `float32` arrays to represent the network’s numerical state, so that numerical types and shapes are explicit.
32. As a project developer, I want production runs to initialize randomly, so that the demo retains the reference behavior.
33. As a project developer, I want tests to inject a seeded NumPy `Generator`, so that numerical expectations are repeatable.
34. As a project developer, I want tolerance-based checks for hidden floating-point calculations, so that harmless cross-language rounding differences do not cause false failures.
35. As a project developer, I want exact assertions for rounded payloads, labels, verdicts, event order, and JSON keys.
36. As a project developer, I want the deterministic test seed to prove that Single-Layer Mode fails and Multi-Layer Mode succeeds.
37. As a project developer, I want the existing shared SSE formatting and response helpers reused rather than duplicated.
38. As a project developer, I want CPU-bound training intervals offloaded from the primary async event-loop thread, so that the backend remains responsive.
39. As a project developer, I want worker-thread use to remain inside the existing FastAPI process, so that Phase 3 does not introduce multiprocessing or shared memory.
40. As a project developer, I want client-disconnect behavior testable through a narrow seam rather than a fragile real network interruption.
41. As a project developer, I want production delays patched out in tests, so that the suite remains fast.
42. As a project developer, I want snapshot writes isolated under pytest temporary directories, so that tests never overwrite real project data.
43. As a project developer, I want concurrent same-mode snapshot writes to be complete and non-corrupt.
44. As a project developer, I want the BPE and Simple Chat tests to remain green after the neural-network router is registered.
45. As an operator, I want `GET /health`, `POST /simple-chat`, and `POST /bpe-tokenize` to remain available.
46. As an operator, I want an abandoned browser request to stop before the next training interval, so that the server does not perform the entire remaining run unnecessarily.
47. As an operator, I want a disconnected or failed run to avoid replacing the last successful weight snapshot.
48. As an operator, I want a persistence failure to prevent the `done` event, so that the frontend never receives a false completion signal.
49. As an operator, I want internal exceptions, stack traces, and filesystem paths excluded from SSE payloads.
50. As an operator, I want the latest successful finisher for a model mode to become that mode’s Saved Weight Snapshot.
51. As an operator, I want the project’s Poetry, pytest, Ruff, and mypy commands to remain the required validation path.

## Implementation Decisions

1. **Confirmed — Python Backend authority:** FastAPI and Python remain the only server-side runtime. The supplied Hono/TypeScript neural-network modules are reference material used to establish behavior; they are not executed or retained as backend runtime dependencies.

2. **Confirmed — Focused Phase Migration:** Phase 3 implements only the XOR neural-network vertical slice required by `POST /neural-net`. It does not add Word2Vec, transformer, general matrix, optimizer, model-registry, or frontend work.

3. **Confirmed — Expected files changed:** The implementation is expected to change or add only:

   ```text
   backend/src/how_llms_work/ml/neural_net.py
   backend/src/how_llms_work/routes/neural_net.py
   backend/src/how_llms_work/schemas.py
   backend/src/how_llms_work/main.py
   backend/tests/test_neural_net.py
   backend/tests/test_neural_net_route.py
   ```

   A test helper may be placed in an existing shared test module only when the current repository already establishes that pattern. No separate serialization module is required unless implementation evidence demonstrates that it materially improves the focused design without widening scope.

4. **Confirmed — Modules left unchanged:** Phase 3 will not implement or redesign:

   ```text
   backend/src/how_llms_work/ml/math_utils.py
   backend/src/how_llms_work/ml/matrix.py
   backend/src/how_llms_work/ml/word2vec.py
   backend/src/how_llms_work/ml/transformer.py
   backend/src/how_llms_work/ml/transformer_worker.py
   backend/src/how_llms_work/routes/train_embed.py
   backend/src/how_llms_work/routes/train_transformer.py
   frontend/
   ```

5. **Confirmed — Request model:** Add a dedicated Pydantic `NeuralNetRequest` model in `schemas.py`.

6. **Confirmed — Request fields:** The request body remains:

   ```json
   {
     "mode": "single-layer",
     "epochs": 5000
   }
   ```

7. **Confirmed — Mode validation:** `mode` is required and accepts only the literal values `single-layer` and `multi-layer`.

8. **Confirmed — Epoch validation:** `epochs` is optional, defaults to `5000`, must be an integer, and is constrained to the inclusive range `100` through `100000`.

9. **Confirmed — No new HTTP controls:** Do not add request fields for a seed, learning rate, hidden-neuron count, activation function, optimizer, snapshot path, persistence switch, or numerical dtype.

10. **Confirmed — Validation behavior:** Invalid bodies use standard FastAPI/Pydantic HTTP `422` responses. The route will not transform validation failures into SSE events.

11. **Confirmed — XOR data:** The reusable module owns the four examples in this exact order:

    ```text
    [0, 0] → 0
    [0, 1] → 1
    [1, 0] → 1
    [1, 1] → 0
    ```

12. **Confirmed — Single-Layer Mode architecture:** The model has two inputs connected directly to one sigmoid output:

    ```text
    2 → 1
    ```

    Its numerical state is two scalar weights and one scalar bias.

13. **Confirmed — Multi-Layer Mode architecture:** The model has two inputs, four sigmoid hidden neurons, and one sigmoid output:

    ```text
    2 → 4 → 1
    ```

14. **Confirmed — Multi-layer weight shapes:**

    ```text
    w1: (2, 4)
    b1: (4,)
    w2: (4,)
    b2: scalar
    ```

15. **Confirmed — Numerical representation:** Use NumPy arrays and scalar values with `float32` numerical state. Public serialization must convert NumPy values to ordinary JSON-compatible Python numbers and lists.

16. **Confirmed — Random production initialization:** Every production Training Run receives a newly created NumPy `Generator` without a fixed seed and initializes all weights independently in `[-1, 1)`.

17. **Confirmed — Bias initialization:** All single-layer and multi-layer biases begin at zero.

18. **Confirmed — Deterministic test initialization:** Public or test-supported training entry points must accept a supplied NumPy `Generator`, or provide an equally narrow deterministic seam, so tests can reproduce one verified Weight Initialization without exposing a seed in the HTTP contract.

19. **Confirmed — Activation function:** Use the sigmoid function:

    ```text
    sigmoid(x) = 1 / (1 + exp(-x))
    ```

20. **Confirmed — Sigmoid derivative:** Calculate the derivative from the already-computed sigmoid output:

    ```text
    sigmoid_derivative(s) = s × (1 - s)
    ```

21. **Confirmed — Learning rate:** Use the reference learning rate exactly:

    ```text
    1.0
    ```

22. **Confirmed — Sample order:** Process the four XOR examples in the confirmed fixed order during every epoch. Do not shuffle.

23. **Confirmed — Immediate updates:** Use online/stochastic-style per-example updates. Each example updates the weights and biases immediately before the next example is processed.

24. **Confirmed — Loss:** Accumulate squared error for the four examples and report mean squared error:

    ```text
    loss = total_squared_error / 4
    ```

25. **Confirmed — Single-layer update:** For each example, compute output, error, sigmoid delta, then update both input weights and the bias using the current example.

26. **Confirmed — Multi-layer forward pass:** For each example:

    1. Compute the four hidden pre-activations.
    2. Apply sigmoid to obtain hidden activations.
    3. Compute the output pre-activation from the current output weights and output bias.
    4. Apply sigmoid to obtain the prediction.

27. **Confirmed — Multi-layer backpropagation order:** Compute hidden error values using the current output-layer weights before those weights are updated. Then update the output-layer weights and output bias, followed by the input-to-hidden weights and hidden biases.

28. **Confirmed — No algorithm redesign:** Do not add batching, vectorized batch-gradient updates, sample shuffling, momentum, Adam, regularization, early stopping, Xavier initialization, a framework optimizer, or a hosted machine-learning service.

29. **Confirmed — Epoch boundaries:** The training loop runs from epoch `0` through the requested epoch value, inclusive.

30. **Confirmed — Reporting step:** Calculate:

    ```text
    step = max(1, floor(epochs / 50))
    ```

31. **Confirmed — Reporting condition:** Produce an Epoch Update when:

    ```text
    epoch % step == 0
    or
    epoch == requested epochs
    ```

    This produces approximately fifty updates. For `5000` epochs, it produces `51` updates at `0, 100, 200, …, 5000`.

32. **Confirmed — Streamed loss rounding:** Round each serialized Epoch Update loss to six decimal places using reference-compatible decimal rounding behavior.

33. **Confirmed — Epoch event payload:** Every progress event is named `epoch` and contains exactly:

    ```json
    {
      "epoch": 100,
      "loss": 0.123456
    }
    ```

34. **Confirmed — Production progress delay:** After each `epoch` event, wait `20` milliseconds before continuing. Do not apply this delay after `done`.

35. **Confirmed — Test delay seam:** Automated tests will replace the route’s referenced delay operation and assert the requested delay value and call count rather than wall-clock duration.

36. **Confirmed — Same-process worker thread:** CPU calculations run in a worker thread within the FastAPI process. Phase 3 does not create a process pool, worker process, shared-memory region, or external task queue.

37. **Confirmed — Interval-level offloading:** Offload one reporting interval at a time, or use an equivalent bounded design that returns control to the async route between Epoch Updates. Do not run the entire maximum-length Training Run as one uninterruptible event-loop operation.

38. **Confirmed — Thread-pool restraint:** Use the existing Python/FastAPI/AnyIO thread-offloading facilities or another bounded standard-library seam. Do not create an unbounded executor or change the global thread-pool capacity as part of Phase 3.

39. **Confirmed — Client-disconnect detection:** The route accepts the FastAPI/Starlette request object and checks the client connection between training intervals.

40. **Confirmed — Cooperative cancellation:** When the client is disconnected:

    - finish only the interval already executing;
    - request no additional training interval;
    - write no Saved Weight Snapshot;
    - emit no `done` event;
    - emit no error event;
    - release all request-local state.

41. **Confirmed — Prediction order:** Build final predictions in this exact order:

    ```text
    [0, 0]
    [0, 1]
    [1, 0]
    [1, 1]
    ```

42. **Confirmed — Prediction schema:** Each prediction contains:

    ```json
    {
      "input": [0, 1],
      "expected": 1,
      "actual": 0.98
    }
    ```

43. **Confirmed — Prediction rounding:** Round every serialized `actual` value to two decimal places before the Training Verdict is calculated.

44. **Confirmed — Success threshold:** A Training Run succeeds only when all four rounded predictions satisfy:

    ```text
    abs(actual - expected) < 0.1
    ```

45. **Confirmed — Exact single-layer architecture label:**

    ```text
    Single-Layer Perceptron (2 → 1)
    ```

46. **Confirmed — Exact multi-layer architecture label:**

    ```text
    Multi-Layer Network (2 → 4 → 1)
    ```

47. **Confirmed — Exact single-layer verdicts:**

    ```text
    SUCCESS — network learned XOR
    FAILED — loss stuck, predictions are random guesses
    ```

48. **Confirmed — Exact multi-layer verdicts:**

    ```text
    SUCCESS — network learned XOR via backpropagation
    FAILED — network did not converge, try more epochs
    ```

49. **Confirmed — Done payload:** The final event is named `done` and contains exactly the frontend-facing fields:

    ```json
    {
      "architecture": "Multi-Layer Network (2 → 4 → 1)",
      "predictions": [],
      "verdict": "SUCCESS — network learned XOR via backpropagation"
    }
    ```

    The weights are never included in this SSE payload.

50. **Confirmed — Successful stream order:**

    ```text
    epoch × N → done
    ```

    A successful stream contains exactly one `done` event and no later events.

51. **Confirmed — Shared SSE infrastructure:** Reuse `format_sse()` and `create_sse_response()` from the existing shared `sse.py`. The response remains `text/event-stream` with the established `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers.

52. **Confirmed — Snapshot directory:** Save completed network weights under the backend project root:

    ```text
    backend/.data/
    ```

    Resolve this location independently of the shell’s current working directory.

53. **Confirmed — Snapshot filenames:**

    ```text
    backend/.data/single-layer-weights.json
    backend/.data/multi-layer-weights.json
    ```

54. **Confirmed — Directory creation:** Create `backend/.data/` when it does not exist.

55. **Confirmed — Single-layer snapshot schema:**

    ```json
    {
      "type": "single-layer",
      "w1": 0.0,
      "w2": 0.0,
      "bias": 0.0
    }
    ```

56. **Confirmed — Multi-layer snapshot schema:**

    ```json
    {
      "type": "multi-layer",
      "w1": [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
      ],
      "b1": [0.0, 0.0, 0.0, 0.0],
      "w2": [0.0, 0.0, 0.0, 0.0],
      "b2": 0.0
    }
    ```

57. **Confirmed — No snapshot metadata:** Do not add epochs, seed, timestamp, loss, verdict, architecture, dtype, explicit shape metadata, request ID, or history records.

58. **Confirmed — JSON formatting:** Serialize with two-space indentation and one final newline.

59. **Confirmed — Save timing:** Persist the completed weights after training and final-result calculation but before yielding `done`.

60. **Confirmed — Atomic replacement:** Write each complete snapshot to a unique temporary file inside `backend/.data/`, close it, and atomically replace the selected mode’s destination.

61. **Confirmed — Concurrent save rule:** Training Runs are independent. For simultaneous successful requests targeting the same mode, the last successful finisher to complete atomic replacement becomes the Saved Weight Snapshot.

62. **Confirmed — Temporary-file cleanup:** Remove a leftover temporary file when serialization, writing, replacement, cancellation, or another failure occurs before replacement completes.

63. **Confirmed — No saved-weight reuse:** A new Training Run never loads a previous snapshot and never continues from persisted weights.

64. **Confirmed — Failure after streaming begins:** When training or persistence fails after the response stream has started:

    - do not emit `done`;
    - do not invent an `error` SSE event;
    - do not expose exception text, stack traces, or paths in event data;
    - terminate the stream;
    - log the exception server-side through the project’s normal logging path;
    - preserve the last successful destination snapshot.

65. **Confirmed — Failure before streaming begins:** Validation remains HTTP `422`. A genuine failure that occurs before response streaming starts may use FastAPI’s generic server-error behavior without exposing internal details.

66. **Confirmed — Router registration:** Import and include the neural-network router in `backend/src/how_llms_work/main.py` while preserving the current health, Simple Chat, and BPE routers.

67. **Confirmed — No ADR:** No confirmed Phase 3 choice requires an Architecture Decision Record. The decisions are focused, visible in the feature contract, and reasonably reversible.

68. **Assumption — Public numerical API:** `ml/neural_net.py` may expose typed functions, state objects, iterators, dataclasses, or protocols for initialization, interval training, prediction, and JSON conversion. The exact names are an implementation decision provided tests can exercise stable public behavior without reaching into private loop variables.

69. **Assumption — Bounded interval representation:** The implementation may represent a reporting interval as a mutable request-local training state advanced by a synchronous function, or as another clear state machine. It must preserve the confirmed formulas and must return to the async route between reports.

70. **Assumption — Rounding implementation:** Python rounding must be checked against the fixed TypeScript reference cases. When Python’s built-in rounding differs at an exact halfway value, implementation must use a narrow reference-compatible decimal rounding helper for the serialized six-decimal losses and two-decimal predictions.

71. **Assumption — Logging:** Use the standard Python logging facilities already available. Do not add a logging dependency.

## Testing Decisions

- **Approved test seam 1 — Public numerical module:** Exercise the stable public XOR training behavior in `ml/neural_net.py`.
- **Why this seam:** It isolates mathematical, dtype, shape, initialization, update-order, reporting, prediction, verdict, and JSON-conversion regressions from HTTP orchestration.
- **Observable behavior covered:**
  - XOR input and target order;
  - sigmoid values and derivative values;
  - randomized initialization bounds;
  - zero biases;
  - `float32` types;
  - single-layer scalar structure;
  - multi-layer `(2,4)`, `(4,)`, `(4,)`, and scalar shapes;
  - fixed sample order;
  - immediate per-example updates;
  - mean-squared loss;
  - epoch range and report boundaries;
  - six-decimal report values;
  - two-decimal predictions;
  - success threshold;
  - architecture labels;
  - verdict strings;
  - exact JSON-compatible snapshot structures.

- **Approved test seam 2 — FastAPI route:** Exercise `POST /neural-net` through FastAPI’s `TestClient` or an equivalent in-process ASGI test client.
- **Why this seam:** It verifies router registration, Pydantic validation, thread offloading orchestration, shared SSE framing, headers, event order, delay references, persistence ordering, and the final browser contract.
- **Observable behavior covered:**
  - valid single-layer and multi-layer requests;
  - omitted `epochs` default;
  - invalid modes;
  - non-integer, below-minimum, and above-maximum epochs;
  - HTTP `200` and `text/event-stream`;
  - established cache and proxy-buffering headers;
  - every SSE `data:` value is valid JSON;
  - `epoch × N → done`;
  - exact event field sets;
  - epoch zero and final epoch;
  - six-decimal losses;
  - exact final payload;
  - weights excluded from `done`;
  - snapshot exists before `done` is emitted;
  - correct mode-specific snapshot path;
  - no real presentation waiting in tests.

- **Approved test seam 3 — Persistence helper or route-owned save boundary:** Test atomic Saved Weight Snapshot behavior through a temporary `.data` directory.
- **Why this seam:** Persistence has correctness and concurrency requirements that should be proven without writing into the repository’s real `.data` directory.
- **Observable behavior covered:**
  - directory creation;
  - two-space JSON formatting;
  - final newline;
  - exact single-layer and multi-layer keys;
  - plain JSON numbers and arrays;
  - complete atomic replacement;
  - prior snapshot preserved after a failed write;
  - unique temporary files;
  - temporary-file cleanup;
  - last successful finisher wins.

- **Required deterministic educational test:** Select and document one NumPy generator seed after implementing the reference formulas. With that seed and the approved test epoch count:
  - Single-Layer Mode must complete with the exact `FAILED — loss stuck, predictions are random guesses` verdict.
  - Multi-Layer Mode must complete with the exact `SUCCESS — network learned XOR via backpropagation` verdict.
  - Every rounded multi-layer prediction must be within `0.1` of its expected XOR target.
  - The seed is test data only and must not enter the HTTP schema or production default.

- **Required numerical comparisons:** Use exact assertions for discrete values, dtypes, shapes, rounded payloads, labels, verdicts, and JSON structures. Use `numpy.testing.assert_allclose()` or equivalent explicit tolerances for unrounded weights, activations, losses, and intermediate numerical state.

- **Required reporting cases:**
  - `epochs=100` to prove the minimum and a standard 51-event schedule;
  - `epochs=101` or another non-divisible value to prove the final epoch is emitted even when it is not on the regular step;
  - `epochs=5000` at the algorithm seam or through a controlled lightweight route seam to prove the default schedule without real delay;
  - validation rejection for `99` and `100001`.

- **Required request-default test:** A request containing only `{"mode":"single-layer"}` must behave as though `epochs` were `5000`.

- **Required delay control:** Patch only the neural-network route’s referenced sleep function. Verify a `0.02`-second request after each `epoch` event and no delay after `done`.

- **Required disconnect control:** Provide a narrow fake or injected disconnect check that becomes true after a selected report. Verify:
  - no later interval starts;
  - no snapshot is written or replaced;
  - no `done` event is emitted;
  - the current in-flight interval may complete before cancellation is observed.

- **Required training-failure test:** Inject a failure after one or more Epoch Updates. Verify no `done` event and no internal exception text in serialized event data.

- **Required save-failure test:** Inject a snapshot persistence failure after training. Verify:
  - no `done`;
  - the prior destination remains unchanged;
  - no partial destination file;
  - no leaked internal error details;
  - temporary files are cleaned up.

- **Required request-isolation test:** Two requests with separately supplied deterministic generators or controlled initial states must not share mutable weights, epoch state, predictions, or destination temporary files.

- **Required concurrent-save test:** Use controlled completion ordering for two same-mode successful saves and prove the last successful finisher’s complete JSON becomes the destination.

- **Required regression checks:**
  - `GET /health` remains HTTP `200` with `{"status":"healthy"}`;
  - `POST /simple-chat` retains its existing validation, headers, and `start → word × N → done` stream;
  - `POST /bpe-tokenize` retains its existing validation, headers, exact contract, and test results;
  - adding `NeuralNetRequest` does not alter `ChatRequest`.

- **Required quality checks:**

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

  Report command outputs honestly. Do not claim success unless each command is executed successfully.

- **Recommended focused commands:**

  ```powershell
  poetry run pytest tests/test_neural_net.py -q
  poetry run pytest tests/test_neural_net_route.py -q
  poetry run pytest tests/test_neural_net.py tests/test_neural_net_route.py -q
  poetry run pytest
  poetry run ruff check .
  poetry run mypy src
  ```

- **Do not test private implementation identity:** Do not assert private helper names, exact local variables, exact dataclass choices, a particular executor object, internal loop syntax, or a particular temporary-file API when the approved observable behavior is satisfied.

- **Do not test wall-clock timing:** Verify configured delay requests and stream ordering, not elapsed milliseconds.

- **Known limitation — Browser rendering:** Automated Python tests do not prove the visual graph rendering or Vite proxy behavior. A manual end-to-end check is recommended when both servers can be run.

- **Known limitation — Cancellation timing:** Cooperative cancellation may complete one already-started interval before stopping.

- **Known limitation — Random production convergence:** The deterministic seed proves one stable educational case. It does not guarantee that every random Multi-Layer Mode initialization converges within every allowed epoch count.

## Out of Scope

- Frontend TypeScript, JSX, hooks, components, styling, or Vite proxy changes.
- Phase 4 Word2Vec embedding work.
- Phase 5 transformer work.
- Changes to the completed BPE tokenizer.
- Changes to Simple Chat beyond regression fixes required by a proven Phase 3 breakage.
- A TypeScript, Node, or Hono backend runtime.
- Running TypeScript neural-network code during Python tests.
- Multiprocessing, process pools, worker processes, shared memory, or external job queues.
- Increasing or globally configuring the FastAPI/AnyIO thread-pool capacity.
- GPU, CUDA, PyTorch, TensorFlow, JAX, scikit-learn, or another machine-learning framework.
- LangChain, LangGraph, OpenAI APIs, or hosted training services.
- A general-purpose matrix library.
- Implementing `math_utils.py` or `matrix.py`.
- Batch-gradient training.
- Sample shuffling.
- Configurable learning rates.
- Configurable hidden-layer sizes.
- Configurable activations or optimizers.
- Momentum, Adam, regularization, dropout, early stopping, or learning-rate schedules.
- Xavier, He, or another replacement initialization scheme.
- A seed field in the browser request.
- Bit-for-bit equality with JavaScript floating-point execution.
- Loading Saved Weight Snapshots for inference or continued training.
- Snapshot history, versioning, metadata, manifests, model registries, or checkpoint recovery.
- Exposing weights in the SSE `done` payload.
- A new SSE `error` event.
- Sending stack traces or internal exception text to clients.
- Exact wall-clock delay tests.
- Forcefully terminating a Python worker thread.
- Cross-process file locking.
- Security hardening or resource limits beyond the confirmed epoch validation.
- Issue-tracker ticket creation, implementation code, commits, or code review.

## Notes

- **Current backend evidence:** The latest Python source snapshot already implements and registers Simple Chat and BPE. `backend/src/how_llms_work/ml/neural_net.py` and `backend/src/how_llms_work/routes/neural_net.py` are empty, and `main.py` does not yet include the neural-network router.
- **Evidence provenance:** The backend and TypeScript source snapshots were not reattached to this specific to-spec run. Current-code statements are inherited from the confirmed `GRILL_WITH_DOCS_RESULT.md` and the supplied current specification rather than independently re-inspected in this run. This does not block specification because the handoff is confirmed and reports no blocking questions.
- **Current dependency evidence:** The backend already declares Python 3.12+, FastAPI, Pydantic, NumPy, pytest, pytest-asyncio, Ruff, and mypy. No new dependency is required for the approved Phase 3 design.
- **Reference contract evidence:** The TypeScript route validates `single-layer` or `multi-layer`, defaults epochs to `5000`, streams `epoch` events with a 20-millisecond delay, saves mode-specific weights, removes weights from the final payload, and emits `done`.
- **Reference algorithm evidence:** The TypeScript trainer uses four fixed XOR examples, learning rate `1.0`, sigmoid activation, mean-squared loss, immediate per-example updates, approximately fifty reports, exact architecture labels, two-decimal predictions, and the confirmed verdict strings.
- **Reference persistence evidence:** The TypeScript serializer writes the exact single-layer or multi-layer weight object as two-space-indented JSON followed by a newline.
- **Framework evidence:** FastAPI/Starlette streaming responses accept generators and stream yielded chunks; Starlette requests expose a disconnect check for streaming or long-lived responses.
- **Python concurrency evidence:** `asyncio.to_thread()` or the framework’s bounded thread-offloading support can move synchronous work away from the event-loop thread. This is a responsiveness measure, not a guarantee of multi-core acceleration.
- **NumPy evidence:** `Generator` supports controlled randomized initialization, NumPy supports `float32` numerical arrays, and `numpy.testing.assert_allclose()` supports tolerance-aware numerical tests.
- **Filesystem evidence:** Python’s temporary-file APIs support unique files, and `os.replace()` supports replacing an existing destination after a complete same-filesystem temporary write.
- **Risk — Float32 divergence:** Repeated `float32` calculations may follow a slightly different learning curve from JavaScript number calculations.
  - **Safeguard:** preserve formulas and serialized rounding, then use tolerance-aware tests and deterministic outcome tests.
- **Risk — Rounding mismatch:** Python and JavaScript may differ at exact decimal halfway values.
  - **Safeguard:** verify fixed reference cases and use a narrow compatibility helper when required.
- **Risk — Multi-layer non-convergence:** Some random initializations may not converge in the requested epochs.
  - **Safeguard:** preserve random production behavior, use an exact failure verdict when needed, and use one verified deterministic seed in tests.
- **Risk — Event-loop blocking:** A `100000`-epoch run can perform substantial CPU work.
  - **Safeguard:** advance training in bounded same-process worker-thread intervals and return to the async route between reports.
- **Risk — Abandoned work:** A browser may disconnect during training.
  - **Safeguard:** check connection state between intervals and avoid persistence or `done` after disconnect.
- **Risk — Partial snapshots:** Direct writes can expose incomplete JSON during failure or concurrency.
  - **Safeguard:** use unique same-directory temporary files and atomic replacement.
- **Risk — Same-mode concurrency:** Two completed requests target one destination.
  - **Safeguard:** keep Training Runs independent and use the confirmed last-successful-finisher-wins rule.
- **Risk — Persistence failure after progress:** HTTP status cannot be replaced cleanly after SSE streaming has begun.
  - **Safeguard:** terminate without `done` or a new error event, preserve the previous snapshot, clean temporary files, and log internally.
- **Risk — Overengineering:** General numerical abstractions can delay the first working XOR demo.
  - **Safeguard:** keep all XOR-specific numerical behavior in `ml/neural_net.py` and leave future-phase modules empty.
- **Evidence limitation:** The deterministic seed and explicit tolerances have not yet been selected because the Python algorithm has not been implemented and executed. Selecting them is required during implementation and must be based on actual results.
- **Evidence limitation:** No Phase 3 implementation or Phase 3 tests were executed while writing this specification.
- **Publication target:** Replace the local root `SPEC.md` with this document.

## Source Material Consulted

### Directly supplied for this run

- `TO_SPEC_PROMPT.md`
- `GRILL_WITH_DOCS_RESULT.md`
- Current `SPEC.md`
- Updated `CONTEXT.md`

### Confirmed evidence inherited from the handoff

- Latest `py_llm_pipeline_explorer_file_structure.md`
- Latest `llm_works_file_structure.md`
- `backend/pyproject.toml`
- `backend/src/how_llms_work/main.py`
- `backend/src/how_llms_work/schemas.py`
- `backend/src/how_llms_work/sse.py`
- `backend/src/how_llms_work/ml/neural_net.py`
- `backend/src/how_llms_work/routes/neural_net.py`
- `backend/tests/test_simple_chat.py`
- `backend/tests/test_bpe.py`
- `backend/tests/test_bpe_tokenize.py`
- TypeScript reference `src/routes/neural-net/index.ts`
- TypeScript reference `src/routes/neural-net/train.ts`
- TypeScript reference `src/routes/neural-net/serialize.ts`
- TypeScript reference `src/schemas/neural-net-request.ts`
- TypeScript frontend neural-network hook and result component

### Authoritative technical references verified during this run

- Official FastAPI documentation for streaming responses
- Official Starlette documentation for request disconnect detection and test clients
- Official Python documentation for `asyncio.to_thread()`, temporary files, and `os.replace()`
- Official NumPy documentation for random `Generator`, `float32`, and tolerance-aware testing
- Official pytest documentation for temporary paths and monkeypatching

## Recommended Next Step

Run `to-tickets-prompt` using this updated `SPEC.md`, the updated `CONTEXT.md`, the latest complete Python backend snapshot, and the TypeScript Reference Implementation.

The resulting tickets must remain limited to Phase 3 and must not introduce frontend changes, Word2Vec, transformer work, multiprocessing, a general matrix framework, new optimizers, saved-weight loading, snapshot history, or a new SSE error-event contract.

`to-tickets-prompt` should decompose this specification into independently verifiable Python work items without writing implementation code.
