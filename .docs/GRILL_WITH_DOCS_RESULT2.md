---
workflow: engineering-prompt-chain
document_type: grill_with_docs_result
prompt_name: grill-with-docs-prompt
status: confirmed
version: 1
recommended_next_prompt: to-spec-prompt
---

# Grill With Docs Result: Phase 3 XOR Neural Network Python Backend

## Original idea

Phase 1—the frontend/server foundation and Python Simple Chat endpoint—is working. Phase 2—the Python BPE tokenizer endpoint—is also working.

Phase 3 will convert the behavior of the TypeScript XOR neural-network reference implementation into the Python-only FastAPI backend. The TypeScript/Vite browser application will remain unchanged and will continue communicating through HTTP and Server-Sent Events.

The conversion centers on the behavior represented by the TypeScript neural-network route, trainer, request schema, and weight serializer. The corresponding Python ownership will be:

```text
backend/src/how_llms_work/ml/neural_net.py
backend/src/how_llms_work/routes/neural_net.py
backend/src/how_llms_work/schemas.py
backend/src/how_llms_work/main.py
backend/tests/test_neural_net.py
backend/tests/test_neural_net_route.py
```

The TypeScript server code is a behavioral reference only. It is not a current backend runtime and will not be executed by the Python application or its tests.

## Problem

The existing frontend includes an XOR Neural Network Demo, but the Python Backend does not currently implement it:

```text
backend/src/how_llms_work/ml/neural_net.py
backend/src/how_llms_work/routes/neural_net.py
```

Both files are empty, and the FastAPI application does not register `POST /neural-net`.

The browser already expects a precise Frontend Contract:

- a request containing a model mode and epoch count;
- approximately fifty streamed `epoch` events;
- one final `done` event;
- exact payload fields;
- exact architecture labels;
- four predictions in XOR truth-table order;
- exact verdict wording;
- a mode-specific JSON file containing the completed weights.

The reference implementation also uses a deliberately simple learning procedure to teach why a single-layer network fails on XOR and why a hidden layer trained with backpropagation can succeed. Replacing that procedure with a modern optimizer, batch training, another library, or a general-purpose neural-network framework would undermine the educational purpose and would not be a faithful Phase Migration.

The endpoint must also avoid blocking the FastAPI event-loop thread for a maximum-length request, must stop abandoned work when the browser disconnects, and must avoid exposing partial or corrupt weight files during failures or concurrent requests.

## Desired outcome

The Python Backend exposes a working:

```text
POST /neural-net
```

For every valid request, it creates a fresh XOR network, trains it using the confirmed reference-compatible mathematics, streams ordered loss updates, calculates the final predictions and Training Verdict, atomically saves the completed weights, and emits `done`.

The existing TypeScript/Vite frontend works without modification.

The implementation remains a focused educational vertical slice:

- NumPy provides explicit `float32` numerical state.
- FastAPI and the existing SSE utilities provide the HTTP boundary.
- Same-process worker-thread intervals keep CPU work away from the primary async event-loop thread.
- Client disconnects stop the run cooperatively.
- Python-only tests prove the numerical behavior and exact frontend contract.
- Production remains randomly initialized.
- Tests use a deterministic NumPy generator.
- Tiny hidden floating-point differences from TypeScript are permitted.
- Rounded payloads, field names, event order, strings, and JSON schemas remain exact.

## Primary users or stakeholders

- Learners using the XOR Neural Network Demo.
- The TypeScript/Vite frontend consuming `POST /neural-net`.
- The project developer maintaining the Python Backend.
- Future workflow prompts that will turn this confirmed result into a specification, implementation tickets, and code.
- Operators running the FastAPI backend through Poetry and Uvicorn.

## Confirmed scope

- Implement the reusable XOR numerical behavior in `backend/src/how_llms_work/ml/neural_net.py`.
- Implement and register `POST /neural-net`.
- Add a dedicated `NeuralNetRequest` Pydantic model.
- Support `single-layer` and `multi-layer`.
- Default `epochs` to `5000`.
- Accept epochs from `100` through `100000`.
- Use the four XOR examples in fixed truth-table order.
- Preserve the reference sigmoid, derivative, learning rate, loss, sample order, immediate updates, and backpropagation order.
- Use NumPy `float32` numerical state.
- Use fresh random production Weight Initialization.
- Allow deterministic generator injection for tests.
- Report approximately fifty Epoch Updates, including epoch zero and the final requested epoch.
- Stream each progress report as an `epoch` SSE event.
- Preserve the 20-millisecond production delay after each `epoch` event.
- Produce the exact ordered predictions, architecture labels, and verdict strings.
- Calculate success from the two-decimal rounded predictions.
- Save one mode-specific JSON weight snapshot.
- Save before emitting `done`.
- Use atomic same-directory replacement.
- Use last-successful-finisher-wins behavior for concurrent same-mode runs.
- Advance CPU training through same-process worker-thread intervals.
- Check for client disconnects between intervals.
- Stop disconnected runs without saving weights or emitting `done`.
- Terminate failed streams without inventing an SSE `error` event.
- Add focused numerical, route, persistence, cancellation, concurrency, and regression tests.
- Preserve `GET /health`, `POST /simple-chat`, and `POST /bpe-tokenize`.

## Out of scope

- Frontend changes.
- Phase 4 Word2Vec embeddings.
- Phase 5 transformer work.
- A TypeScript, Hono, or Node backend runtime.
- Running TypeScript during Python tests.
- Multiprocessing.
- Process pools.
- Shared memory.
- External job queues.
- Increasing the global FastAPI or AnyIO thread-pool capacity.
- GPU or CUDA support.
- PyTorch, TensorFlow, JAX, scikit-learn, or another neural-network framework.
- LangChain, LangGraph, or hosted AI services.
- General-purpose matrix abstractions.
- Implementing `math_utils.py` or `matrix.py`.
- Batch-gradient training.
- Sample shuffling.
- Momentum, Adam, regularization, dropout, early stopping, or learning-rate schedules.
- Configurable learning rate.
- Configurable hidden-layer size.
- Configurable activation or optimizer.
- Xavier, He, or another replacement initialization strategy.
- A seed field in the HTTP request.
- Bit-for-bit TypeScript floating-point equality.
- Loading saved weights for inference or continued training.
- Training history, model registries, checkpoint history, or snapshot metadata.
- Exposing weights in the `done` event.
- A new SSE `error` event.
- Forcefully terminating a worker thread.
- Cross-process file locking.
- Unrelated cleanup or future-phase abstractions.
- Implementation code, tickets, commits, or code review during this workflow.

## Confirmed decisions

1. **Python Backend remains authoritative:** FastAPI and Python are the only current server-side runtime. The TypeScript implementation is reference evidence only.

2. **Focused Phase Migration:** Phase 3 implements the smallest complete XOR vertical slice and does not introduce future Word2Vec or transformer abstractions.

3. **HTTP endpoint:** The route remains:

   ```text
   POST /neural-net
   ```

4. **Request contract:** The request contains:

   ```json
   {
     "mode": "single-layer",
     "epochs": 5000
   }
   ```

5. **Mode validation:** `mode` is required and must be `single-layer` or `multi-layer`.

6. **Epoch validation:** `epochs` is optional, defaults to `5000`, must be an integer, and must be between `100` and `100000`, inclusive.

7. **No added request controls:** No seed, learning rate, hidden size, optimizer, activation, persistence flag, or output path will be exposed through HTTP.

8. **Validation failure behavior:** Invalid requests use standard FastAPI/Pydantic HTTP `422` responses.

9. **XOR data order:** Training always uses:

   ```text
   [0, 0] → 0
   [0, 1] → 1
   [1, 0] → 1
   [1, 1] → 0
   ```

10. **Single-Layer Mode:** The model is a two-input, one-output sigmoid network:

    ```text
    2 → 1
    ```

11. **Multi-Layer Mode:** The model is a two-input, four-hidden-neuron, one-output sigmoid network:

    ```text
    2 → 4 → 1
    ```

12. **Multi-layer numerical shapes:**

    ```text
    w1: (2, 4)
    b1: (4,)
    w2: (4,)
    b2: scalar
    ```

13. **NumPy representation:** Numerical state uses NumPy `float32` arrays and scalar values.

14. **Random production initialization:** Every production Training Run uses a fresh NumPy generator and independently initialized weights in `[-1, 1)`.

15. **Zero biases:** All biases begin at zero.

16. **Deterministic test initialization:** Tests provide a seeded NumPy generator through a narrow numerical seam. The seed is not part of the browser contract.

17. **Numerical compatibility:** The Python implementation must preserve the same mathematics and educational outcome, but hidden values do not need bit-for-bit equality with TypeScript.

18. **Tolerance-aware tests:** Unrounded weights, activations, losses, and intermediate state are compared with explicit numerical tolerances.

19. **Exact serialized tests:** Event names, field names, rounded values, labels, verdicts, prediction order, and weight JSON schemas require exact equality.

20. **Sigmoid activation:**

    ```text
    sigmoid(x) = 1 / (1 + exp(-x))
    ```

21. **Sigmoid derivative:** The derivative is calculated from the sigmoid output:

    ```text
    s × (1 - s)
    ```

22. **Learning rate:** The learning rate remains exactly `1.0`.

23. **Fixed sample order:** XOR examples are never shuffled.

24. **Immediate updates:** Weights and biases update after each training example, before processing the next example.

25. **Loss:** Report mean squared error over the four XOR examples.

26. **Backpropagation order:** Multi-Layer Mode calculates hidden error values using the current output weights before updating those output weights. It then updates the output layer followed by the input-to-hidden layer.

27. **No optimizer redesign:** No batching, Adam, momentum, regularization, early stopping, or replacement initialization scheme is introduced.

28. **Epoch range:** Training runs from epoch `0` through the requested epoch value, inclusive.

29. **Reporting interval:**

    ```text
    step = max(1, floor(epochs / 50))
    ```

30. **Reporting condition:** Emit an Epoch Update when the epoch is divisible by `step` or is the requested final epoch.

31. **Default report count:** At `5000` epochs, reports occur at:

    ```text
    0, 100, 200, …, 4900, 5000
    ```

    This produces 51 `epoch` events.

32. **Epoch payload:** Each progress event contains exactly:

    ```json
    {
      "epoch": 100,
      "loss": 0.123456
    }
    ```

33. **Loss rounding:** Streamed loss is rounded to six decimal places using behavior verified against fixed TypeScript reference cases.

34. **Progress delay:** Production waits `0.02` seconds after each `epoch` event and does not delay after `done`.

35. **Successful event order:**

    ```text
    epoch × N → done
    ```

36. **Same-process worker thread:** CPU calculations run outside the main async event-loop thread but inside the FastAPI process.

37. **Bounded interval execution:** Training returns control to the async route between progress reports rather than running the entire maximum request as one uninterruptible event-loop task.

38. **No thread-pool expansion:** Use existing bounded framework or standard-library thread offloading. Do not create an unbounded executor or change the global pool size.

39. **Cooperative disconnect cancellation:** Check the browser connection between intervals.

40. **Disconnect result:** A disconnected run requests no further interval, saves no snapshot, emits no `done`, and emits no error event.

41. **Prediction order:**

    ```text
    [0, 0]
    [0, 1]
    [1, 0]
    [1, 1]
    ```

42. **Prediction payload:** Each final prediction contains `input`, `expected`, and `actual`.

43. **Prediction rounding:** Each `actual` value is rounded to two decimal places before success is evaluated.

44. **Success threshold:** All four rounded predictions must satisfy:

    ```text
    abs(actual - expected) < 0.1
    ```

45. **Exact single-layer architecture label:**

    ```text
    Single-Layer Perceptron (2 → 1)
    ```

46. **Exact multi-layer architecture label:**

    ```text
    Multi-Layer Network (2 → 4 → 1)
    ```

47. **Exact single-layer verdicts:**

    ```text
    SUCCESS — network learned XOR
    FAILED — loss stuck, predictions are random guesses
    ```

48. **Exact multi-layer verdicts:**

    ```text
    SUCCESS — network learned XOR via backpropagation
    FAILED — network did not converge, try more epochs
    ```

49. **Done payload:** The `done` event contains exactly:

    ```text
    architecture
    predictions
    verdict
    ```

    The weights are removed before serialization to the browser.

50. **Saved snapshot directory:**

    ```text
    backend/.data/
    ```

    The path is resolved from the backend project root rather than the shell’s current working directory.

51. **Snapshot filenames:**

    ```text
    backend/.data/single-layer-weights.json
    backend/.data/multi-layer-weights.json
    ```

52. **Single-layer JSON schema:**

    ```json
    {
      "type": "single-layer",
      "w1": 0.0,
      "w2": 0.0,
      "bias": 0.0
    }
    ```

53. **Multi-layer JSON schema:**

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

54. **No snapshot metadata:** Do not add epochs, seed, timestamp, loss, architecture, verdict, dtype, or explicit shape fields.

55. **JSON formatting:** Use two-space indentation and a final newline.

56. **Save before completion:** The snapshot must be successfully replaced before `done` is emitted.

57. **Atomic persistence:** Write complete JSON to a unique temporary file inside `.data`, close it, and atomically replace the selected mode’s destination.

58. **Concurrent same-mode requests:** Training state remains independent. The last successful finisher to replace the destination becomes the Saved Weight Snapshot.

59. **No saved-weight reuse:** New runs never load or continue from a previous snapshot.

60. **Failure after streaming starts:** Do not emit `done`, do not invent an `error` event, do not expose internal details, preserve the prior snapshot, clean temporary files, and terminate the stream.

61. **Failure before streaming starts:** A genuine pre-stream internal failure may use FastAPI’s generic server-error behavior.

62. **Router registration:** Register the neural-network router without changing the established Health, Simple Chat, or BPE contracts.

63. **Deterministic educational test:** Select one verified NumPy seed during implementation that reliably demonstrates:

    ```text
    Single-Layer Mode → FAILED
    Multi-Layer Mode  → SUCCESS
    ```

64. **No ADR:** None of the confirmed decisions passes all three ADR gates. The choices are focused, visible in the contract, and reasonably inexpensive to revise.

## Current behavior verified from files or tools

- The backend uses Python 3.12 or newer, Poetry, FastAPI, Pydantic, NumPy, pytest, pytest-asyncio, Ruff, and strict mypy.
- `backend/src/how_llms_work/main.py` currently registers the Simple Chat and BPE routers and exposes `GET /health`.
- `backend/src/how_llms_work/schemas.py` currently contains the shared `ChatRequest`.
- `backend/src/how_llms_work/sse.py` currently contains shared SSE formatting, `text/event-stream` response creation, and cache/proxy-buffering headers.
- `POST /simple-chat` is implemented and tested.
- `POST /bpe-tokenize` is implemented and tested.
- `backend/src/how_llms_work/ml/neural_net.py` is empty.
- `backend/src/how_llms_work/routes/neural_net.py` is empty.
- No Phase 3 neural-network tests currently exist in the latest source snapshot.
- The TypeScript reference request accepts `single-layer` or `multi-layer`, defaults epochs to `5000`, and constrains epochs to `100` through `100000`.
- The TypeScript trainer uses random starting weights, zero biases, sigmoid activation, learning rate `1.0`, four fixed XOR examples, immediate per-example updates, and mean-squared loss.
- The TypeScript route emits approximately fifty `epoch` events and one `done` event.
- The reference progress delay is 20 milliseconds.
- The reference final predictions are ordered by the XOR truth table and rounded to two decimal places.
- The reference success test uses the rounded predictions and a strict difference below `0.1`.
- The reference backend saves weights before emitting `done`.
- The reference serializer writes one JSON file per model mode using exact mode-specific keys.
- The current Phase 3 `SPEC.md` reflects the confirmed decisions recorded by this grilling session.

## Desired behavior

A valid request such as:

```json
{
  "mode": "multi-layer",
  "epochs": 5000
}
```

returns HTTP `200` with `text/event-stream`.

The stream resembles:

```text
event: epoch
data: {"epoch":0,"loss":0.250123}

event: epoch
data: {"epoch":100,"loss":0.123456}

...

event: epoch
data: {"epoch":5000,"loss":0.000123}

event: done
data: {"architecture":"Multi-Layer Network (2 → 4 → 1)","predictions":[...],"verdict":"SUCCESS — network learned XOR via backpropagation"}
```

Before the final `done` event, the backend writes:

```text
backend/.data/multi-layer-weights.json
```

The browser never receives the weights.

Single-Layer Mode uses the same route and stream shape but the single-layer model, architecture label, verdict choices, and snapshot schema.

Separate requests never share mutable training state.

## Domain model

### Terms created or changed

- **XOR Neural Network Demo:** The Learning Demo that contrasts a single-layer neural network’s inability to learn XOR with a multi-layer network trained through backpropagation.
- **Single-Layer Mode:** The XOR mode using two inputs connected directly to one sigmoid output.
- **Multi-Layer Mode:** The XOR mode using two inputs, four sigmoid hidden neurons, and one sigmoid output.
- **Training Run:** One complete execution using one selected mode, one epoch count, and one initialized set of weights.
- **Weight Initialization:** The starting values assigned to network weights before training.
- **Epoch Update:** A streamed progress measurement containing an epoch number and six-decimal training loss.
- **XOR Prediction:** One final input pair, expected output, and two-decimal actual output.
- **Training Verdict:** The exact success or failure message calculated from all four rounded predictions.
- **Neural Network Event Stream:** Approximately fifty ordered `epoch` events followed by one `done` event for a successful run.
- **Saved Weight Snapshot:** The JSON representation of the latest successfully completed weights for one model mode; the last successful finisher replaces the prior snapshot.

### Important relationships

- One browser request creates one Training Run.
- One Training Run selects exactly one model mode.
- One Training Run owns one independent Weight Initialization.
- One Training Run processes the four XOR examples repeatedly.
- One successful Training Run produces one Neural Network Event Stream.
- One successful Neural Network Event Stream contains one or more Epoch Updates followed by exactly one `done` event.
- One `done` payload contains exactly four XOR Predictions and one Training Verdict.
- One successful Training Run replaces one mode-specific Saved Weight Snapshot before `done`.
- A disconnected or failed Training Run produces no new Saved Weight Snapshot and no `done`.
- Concurrent Training Runs do not share numerical state.
- Concurrent same-mode successful runs share only the final destination rule: last successful finisher wins.
- The TypeScript/Vite frontend consumes the Frontend Contract produced by the Python Backend.
- The TypeScript Reference Implementation informs compatibility expectations but has no runtime role.

### Domain artifacts

- [CONTEXT.md](CONTEXT.md)
- [SPEC.md](SPEC.md)

## Architectural decisions

None. No confirmed decision passed all three ADR gates:

1. hard to reverse;
2. surprising without context;
3. involving a substantial real tradeoff.

The worker-thread, atomic-save, and compatibility choices are meaningful implementation safeguards, but they remain localized and visible in the feature contract.

## Constraints

- Use Python 3.12 or newer.
- Use Poetry for environment and dependency management.
- Use Windows PowerShell for commands.
- Use FastAPI routers.
- Use Pydantic request validation.
- Use NumPy for numerical operations.
- Use Server-Sent Events through the existing shared SSE utilities.
- Use pytest and the project’s in-process ASGI testing support.
- Use Ruff and strict mypy.
- Preserve the existing TypeScript/Vite frontend.
- Preserve `GET /health`, `POST /simple-chat`, and `POST /bpe-tokenize`.
- Preserve exact serialized field names, event names, payload structures, labels, verdicts, and event order.
- Keep reusable numerical behavior separate from route orchestration.
- Resolve persistence from the backend project root.
- Do not hard-code a user-specific absolute filesystem path.
- Do not add dependencies unless the existing `pyproject.toml` is proven insufficient.
- Do not claim tests passed unless commands are actually executed successfully.
- Keep Phase 3 focused on XOR.

## Edge cases and failure behavior

- **Missing mode:** Return HTTP `422`.
- **Unknown mode:** Return HTTP `422`.
- **Omitted epochs:** Use `5000`.
- **Non-integer epochs:** Return HTTP `422`.
- **Epochs below 100:** Return HTTP `422`.
- **Epochs above 100000:** Return HTTP `422`.
- **Epoch count not divisible by the reporting interval:** Emit the regular reports and still emit the final requested epoch.
- **Single-layer convergence by chance:** Use the exact success verdict if all rounded predictions meet the threshold; do not force failure in production.
- **Multi-layer non-convergence:** Use the exact confirmed failure verdict.
- **Floating-point differences:** Allow small hidden differences while preserving rounded output behavior and educational outcome.
- **Decimal halfway rounding:** Verify Python serialization against fixed TypeScript reference cases and use a narrow compatibility helper when needed.
- **Client disconnect:** Finish at most the currently executing interval, then stop without persistence or `done`.
- **Training failure after progress:** Terminate without `done` or a new error event.
- **Persistence failure:** Preserve the prior destination and do not emit `done`.
- **Temporary-file failure:** Clean the temporary file when possible and leave the destination unchanged.
- **Concurrent same-mode completion:** The last successful atomic replacement wins.
- **Concurrent different modes:** Each mode uses its own independent destination.
- **Random multi-layer non-convergence:** Treat it as a valid failed Training Run rather than an internal error.
- **Maximum request:** Advance work in bounded thread-offloaded intervals so the async route regains control between reports.
- **Unexpected internal exception:** Log server-side without serializing stack traces, paths, environment data, or secrets to the browser.

## Testing expectations

### Numerical tests

- Confirm XOR examples and target order.
- Confirm sigmoid values.
- Confirm derivative values.
- Confirm initialization bounds.
- Confirm zero biases.
- Confirm NumPy `float32` numerical types.
- Confirm single-layer structure.
- Confirm multi-layer weight shapes.
- Confirm deterministic initialization from a supplied generator.
- Confirm fixed sample order.
- Confirm immediate updates.
- Confirm mean-squared loss.
- Confirm multi-layer backpropagation update order.
- Confirm epoch zero and final-epoch reporting.
- Confirm reporting for divisible and non-divisible epoch counts.
- Confirm six-decimal report values.
- Confirm two-decimal final predictions.
- Confirm exact prediction order.
- Confirm exact architecture labels.
- Confirm exact verdict strings.
- Confirm exact JSON-compatible weight dictionaries.

### Deterministic educational test

Select one seed from actual Python execution and document it in tests.

Using that seed:

```text
Single-Layer Mode → FAILED
Multi-Layer Mode  → SUCCESS
```

The multi-layer rounded predictions must all be within `0.1` of their expected XOR targets.

The seed is test data only.

### Route and SSE tests

- Confirm router registration.
- Confirm HTTP `200`.
- Confirm `text/event-stream`.
- Confirm cache and proxy-buffering headers.
- Confirm default epochs.
- Confirm invalid-body HTTP `422` cases.
- Confirm every SSE data value is valid JSON.
- Confirm event order `epoch × N → done`.
- Confirm exact epoch payload keys.
- Confirm exact done payload keys.
- Confirm weights are excluded from `done`.
- Confirm the 20-millisecond sleep is requested after each `epoch`.
- Confirm no sleep after `done`.
- Confirm persistence succeeds before `done`.
- Confirm request state is isolated.

### Persistence tests

- Use pytest temporary directories.
- Confirm `.data` creation.
- Confirm mode-specific filenames.
- Confirm exact JSON keys.
- Confirm plain Python numbers and lists.
- Confirm two-space indentation.
- Confirm final newline.
- Confirm atomic replacement.
- Confirm prior snapshot preservation on failure.
- Confirm temporary-file cleanup.
- Confirm last-successful-finisher-wins behavior.

### Disconnect and failure tests

- Inject a disconnect after a selected report.
- Confirm no later interval begins.
- Confirm no snapshot is written.
- Confirm no `done`.
- Inject a training failure after progress.
- Confirm no `done`.
- Inject a save failure.
- Confirm no `done`.
- Confirm no internal exception details appear in event data.

### Regression tests

- `GET /health` remains unchanged.
- Simple Chat remains unchanged.
- BPE remains unchanged.
- `ChatRequest` remains unchanged when `NeuralNetRequest` is added.

### Validation commands

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Results must be reported honestly.

## Risks and safeguards

- **Risk:** `float32` calculations may diverge slightly from JavaScript number calculations.
  - **Safeguard:** preserve formulas and output rounding; use explicit tolerances for hidden numerical values.

- **Risk:** Python and JavaScript may round exact halfway values differently.
  - **Safeguard:** compare fixed reference cases and implement a narrow compatibility rounding helper only when evidence requires it.

- **Risk:** Some random Multi-Layer Mode initializations may not converge.
  - **Safeguard:** preserve the confirmed failure verdict and use one verified deterministic seed for the required educational test.

- **Risk:** A maximum-length request can consume meaningful CPU time.
  - **Safeguard:** advance work in bounded same-process worker-thread intervals.

- **Risk:** A browser may leave during training.
  - **Safeguard:** check disconnection between intervals and skip persistence and `done`.

- **Risk:** Direct file writes can expose incomplete JSON.
  - **Safeguard:** use a complete same-directory temporary write followed by atomic replacement.

- **Risk:** Two requests can complete against the same snapshot.
  - **Safeguard:** keep training state isolated and use last-successful-finisher-wins atomic replacement.

- **Risk:** Persistence can fail after progress events have already been sent.
  - **Safeguard:** terminate without `done`, preserve the previous snapshot, and log internally.

- **Risk:** Thread offloading could be mistaken for a performance guarantee.
  - **Safeguard:** treat it as event-loop responsiveness, not guaranteed multi-core acceleration.

- **Risk:** Generalized matrix or optimizer abstractions can expand scope.
  - **Safeguard:** keep XOR-specific numerical behavior in `ml/neural_net.py` and leave later-phase modules unchanged.

- **Risk:** Tests could become flaky because of production randomness.
  - **Safeguard:** inject a dedicated seeded generator in tests and never rely on the global random state.

- **Risk:** Adding the route could regress completed endpoints.
  - **Safeguard:** retain and run Health, Simple Chat, and BPE regression tests.

## Open questions

None that block the next workflow step.

The exact deterministic seed and numerical tolerance values must be selected from actual Python execution during implementation. This is an implementation calibration task, not an unresolved product or architecture decision.

No Phase 3 implementation has been performed and no Phase 3 tests have been run as part of this grilling workflow.

## Source material consulted

- `GRILL_WITH_DOCS_PROMPT.md`
- Existing and updated `CONTEXT.md`
- Existing Phase 2 `SPEC.md`
- Updated Phase 3 `SPEC.md`
- Latest `py_llm_pipeline_explorer_file_structure.md`
- Latest `llm_works_file_structure.md`
- `backend/pyproject.toml`
- `backend/src/how_llms_work/main.py`
- `backend/src/how_llms_work/schemas.py`
- `backend/src/how_llms_work/sse.py`
- `backend/src/how_llms_work/ml/neural_net.py`
- `backend/src/how_llms_work/routes/neural_net.py`
- Existing Simple Chat and BPE tests
- TypeScript reference neural-network request schema
- TypeScript reference XOR trainer
- TypeScript reference neural-network route
- TypeScript reference weight serializer
- TypeScript frontend neural-network hook and result component
- Official FastAPI streaming-response documentation
- Official Starlette request-disconnect documentation
- Official Python `asyncio`, temporary-file, and filesystem-replacement documentation
- Official NumPy random-generator, `float32`, and testing documentation
- Official pytest temporary-path and monkeypatch documentation

## Recommended next step

Run `to-spec-prompt` using this file, the updated `CONTEXT.md`, the latest Python backend snapshot, and the TypeScript Reference Implementation.

In this conversation, the Phase 3 `SPEC.md` has already been produced from these confirmed decisions. Verify that it remains aligned with this result, then proceed to `to-tickets-prompt`.

The next tickets must remain limited to Phase 3 and must not introduce frontend changes, Word2Vec, transformer work, multiprocessing, general matrix abstractions, new optimizers, saved-weight loading, snapshot history, or a new SSE error-event contract.
