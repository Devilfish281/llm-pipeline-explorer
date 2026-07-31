---
workflow: engineering-prompt-chain
document_type: grill_with_docs_result
prompt_name: grill-with-docs-prompt
status: confirmed
version: 1
recommended_next_prompt: to-spec-prompt
---

# Grill With Docs Result: Phase 6 Saved Transformer Loading and Worker-Process Visibility

## Original idea

Phases 1 through 5 of the `llm-pipeline-explorer` migration are working through the Python FastAPI backend:

- Phase 1: frontend/server foundation and Simple Chat;
- Phase 2: BPE Tokenizer;
- Phase 3: XOR Neural Network;
- Phase 4: Word2Vec Embeddings;
- Phase 5: Transformer behavior, multiprocessing, shared memory, SSE streaming, and final-model persistence.

Phase 6 should address the important limitation that the application saves completed Transformer models but cannot load one later for text generation. Every current `POST /train-transformer` request still starts from fresh weights.

Phase 6 should also make the number of spawned Transformer worker processes visible during training by adding this display-only line to the first public training sample:

```text
Transformer worker processes: <count>
```

The initial reference to converting `train_embed` was resolved as stale wording because `POST /train-embed` is already implemented and registered in the Python Backend.

## Problem

The current Python Backend can train a Transformer, stream progress, and persist complete configuration-specific JSON models, but there is no endpoint or browser command for selecting a saved model and generating text from it without retraining.

The current Transformer frontend also parses its input only as five training numbers. A command beginning with `File:` would be converted into training defaults rather than reaching Python as a filename and prompt.

The backend already determines an actual worker-process count between one and four for each Transformer Training Run, but that value is not included in any public event displayed by the browser.

Phase 6 must add saved-model inference without turning saved models into checkpoints, resume points, caches, or shared conversational sessions. It must also preserve the working Phase 5 training behavior except for the approved worker-count presentation text and the shared request-slot wording required by the new endpoint.

## Desired outcome

The existing Transformer input box accepts either a training command or a saved-model command.

Training remains:

```text
1000 1 0.6 6 3
```

A named saved model is selected with:

```text
File: transformer-weights-e1000-l1-d32-h2-ff128-ctx32.json | once upon a time | 0.8 0.9 40
```

The most recently modified strictly valid saved model is selected with:

```text
File: | once upon a time | 0.8 0.9 40
```

The frontend detects `File:` without regard to capitalization and sends a structured request to:

```text
POST /load-transformer
```

A successful Saved Transformer Generation Run:

1. reserves the shared process-local Transformer request slot;
2. safely selects one ordinary model file inside the real backend `.data` directory;
3. reads that file once into one request-owned in-memory snapshot;
4. strictly validates the current Python Phase 5 model format;
5. tokenizes and validates the prompt with the saved model's own Vocabulary and Merge Table;
6. emits `loaded`;
7. generates a deterministic complete prompt-plus-continuation result with seed `42`;
8. emits `result`;
9. emits `done`;
10. discards all request-owned loaded-model state and releases the shared request slot.

The existing results area displays:

```text
Loaded: transformer-weights-e1000-l1-d32-h2-ff128-ctx32.json

Prompt:
once upon a time

Generated text:
once upon a time a small cat lived in a village...
```

During training only, the first public `epoch` event's displayed sample begins with:

```text
Transformer worker processes: 4
```

The number is dynamic and may be `1`, `2`, `3`, or `4`.

## Primary users or stakeholders

- Learners using the **Train Transformer** Learning Demo.
- The project owner running and maintaining the application on Windows 11.
- The TypeScript/Vite frontend that consumes the training and saved-model SSE contracts.
- Future maintainers of Transformer persistence, inference, validation, and process lifecycle behavior.
- The Poetry, pytest, Ruff, strict-mypy, and frontend type-checking workflow.
- Future specification, ticket, implementation, and review prompts that consume this confirmed handoff.

## Confirmed scope

- Add and register `POST /load-transformer`.
- Add a dedicated Pydantic request model for saved-model generation.
- Add safe saved-model selection, loading, strict validation, tokenization, and stateless generation.
- Support an exact filename or the newest strictly valid model.
- Use the saved model's own configuration, ordered Vocabulary, Merge Table, and trained parameters.
- Preserve current Transformer generation mathematics where applicable.
- Use deterministic seed `42` for every Saved Transformer Generation Run.
- Return the complete prompt followed by its generated continuation.
- Add `loaded`, `result`, `done`, and safe `error` SSE handling for the load route.
- Share the existing process-local Transformer request slot between training and loading.
- Stop saved-model generation cooperatively on disconnect or deadline.
- Run saved-model file work and generation outside the FastAPI event-loop thread while keeping inference in the backend parent process.
- Make the smallest necessary frontend changes to:
  - recognize case-insensitive `File:` commands;
  - validate and parse the three command sections;
  - call `/load-transformer`;
  - consume the new SSE event types;
  - clear the previous Transformer result when a new request starts;
  - display the loaded filename, prompt, and completed generated text.
- Keep the existing Transformer input box, Send button, page, component area, and layout.
- Prefix the first public training sample with `Transformer worker processes: <actualWorkerCount>`.
- Keep that worker label display-only.
- Update `CONTEXT.md` with the approved training-versus-loading terminology.
- Add ADR 0003 for the architecturally significant saved-model inference boundary.
- Add focused backend, frontend-parser, SSE, safety, lifecycle, and regression tests in the later implementation workflow.

## Out of scope

- Resuming, continuing, fine-tuning, or initializing training from a Saved Transformer Model.
- Treating saved models as training checkpoints or cache shortcuts.
- Changing the fixed Transformer Training Corpus or training architecture.
- A new model-training algorithm, optimizer, learning rate, worker protocol, Logical Training Shard design, or shared-memory design.
- Loading Saved Embedding Models or Saved XOR Weight Snapshots.
- Reimplementing `POST /train-embed`; Phase 4 is already complete.
- A continuing conversational session or remembered chat history.
- Server-side sessions, cookies, model-selection state, or conversation state.
- A separate model registry, manifest, database, rollback system, download feature, or deletion feature.
- Supporting old TypeScript saved-model JSON formats.
- Guessing, repairing, migrating, or filling missing values in malformed models.
- Automatically training a model when no saved model is available.
- Silently substituting another model when a named model fails.
- Caching a loaded model across requests.
- A file-size limit for Saved Transformer Models.
- Token-by-token display for saved-model generation.
- Starting Transformer training worker processes for saved-model generation.
- Displaying a worker-process label for `/load-transformer`.
- Adding a Train/Load selector, second input box, new page, or layout redesign.
- Allowing the `|` command separator inside the starting prompt.
- An arbitrary filesystem path supplied by the browser.
- Following symbolic links or Windows junctions.
- Multiple simultaneous Transformer jobs or a waiting queue.
- GPU, CUDA, PyTorch, TensorFlow, JAX, hosted models, LangChain, or LangGraph.
- Writing the Phase 6 specification, implementation tickets, production code, commits, or code review during this workflow.

## Confirmed decisions

1. **Separate endpoint:** Use `POST /load-transformer`. Training remains at `POST /train-transformer`.

2. **Existing input box:** Reuse the existing Transformer input box and Send button. Do not add a new page, selector, or visual layout.

3. **Small frontend change is allowed:** Modify only the command parser, endpoint selection, event handling, and current result rendering needed for saved-model generation.

4. **Automatic command routing:** After removing leading whitespace, treat any capitalization of `File:` as a saved-model command. Nonnumeric commands without that prefix continue through the existing training parser.

5. **Three-section command:** Use:

   ```text
   File: filename | starting text | temperature top-P max-tokens
   ```

6. **Named model selection:** Text after `File:` selects that exact filename.

7. **Latest-model selection:** An empty filename after `File:` means the newest strictly valid Saved Transformer Model.

8. **Optional filename request field:** The backend request uses `modelFile: string | null`; `null` requests the latest valid model. Do not add a second `useLatest` flag or use a magic `"latest"` filename.

9. **Load request fields:** The public structured request contains exactly the saved-model selection, prompt, and existing generation controls:

   ```json
   {
     "modelFile": null,
     "prompt": "once upon a time",
     "temperature": 0.8,
     "topP": 0.9,
     "maxTokens": 40
   }
   ```

10. **Existing generation ranges:** Preserve temperature `0.1..2.0`, top-p `0.1..1.0`, and maximum tokens `3..500`.

11. **Frontend format validation:** Reject malformed commands before sending a request and display a helpful usage message in the existing results area.

12. **Command separator rule:** Reject a starting prompt containing `|`; do not add escaping.

13. **Prompt trimming:** Remove whitespace only from the beginning and end of the prompt. Preserve interior spacing exactly.

14. **Nonempty prompt:** Reject an empty or whitespace-only prompt.

15. **Saved-model tokenization:** Tokenize with the selected model's own ordered Vocabulary and Merge Table.

16. **Unsupported prompt text:** Reject text that the selected saved model cannot tokenize completely. Do not drop, replace, or silently default unsupported text.

17. **Sixteen-token prompt maximum:** The tokenized starting prompt must contain at most 16 tokens. Reject longer prompts instead of silently discarding their beginning.

18. **Complete result text:** `result.text` includes the original prompt followed by the newly generated continuation.

19. **Deterministic generation:** Always use generation seed `42`, so identical valid requests against identical model contents reproduce the same text.

20. **Stateless requests:** Each command loads one model, generates one result, and remembers nothing afterward.

21. **Current format only:** Accept only the complete current Python Phase 5 Saved Transformer Model format.

22. **Strict model validation:** Before generation, validate:
    - exact model type;
    - exact required top-level and nested fields;
    - supported configuration values;
    - complete ordered Vocabulary and valid Merge Table;
    - every required parameter array;
    - exact array lengths derived from the configuration;
    - finite numeric values;
    - Transformer block count equal to `numLayers`;
    - filename configuration agreement where the approved filename format applies.

23. **Exact filename capitalization:** A named filename must match the stored filename exactly, including capitalization.

24. **Plain filename only:** Reject absolute paths, drive letters, `..`, `/`, `\`, and any value that is not one approved Transformer model filename.

25. **Real `.data` boundary:** Search only the real backend `.data` directory resolved from the application code, not the shell's current working directory.

26. **No links or junctions:** Reject a `.data` directory or selected file that is a symbolic link or Windows junction, and reject any resolved path outside the genuine `.data` directory.

27. **Named-file failure:** A missing, malformed, damaged, linked, incompatible, or incorrectly capitalized named file produces a safe loading error. Never substitute another model.

28. **Latest candidate ordering:** For `modelFile: null`, inspect matching candidates from newest to oldest by modification time.

29. **Latest invalid-file handling:** Skip damaged or incompatible candidates and select the first strictly valid model.

30. **Latest tie-break:** When valid candidates have the same modification time, choose the alphabetically greatest exact filename.

31. **No valid latest model:** Return the specific safe message `No valid saved Transformer model was found.`

32. **Read every request:** Read and strictly validate the selected JSON file for every request. Do not cache models between requests.

33. **One file snapshot:** Read the selected file once and generate only from that validated in-memory snapshot. Do not reopen the file during the same request.

34. **No file-size cap:** Do not reject a model solely because of file size. Attempt to read it with available memory and return a safe loading error on a specific read or parse failure.

35. **Parent-process inference:** Do not create a Request-Scoped Worker Group for loading. Perform inference in the backend parent process.

36. **Off-event-loop work:** Run blocking file parsing, validation, and generation away from the FastAPI event-loop thread.

37. **Saved-model event sequence:** A successful request emits:

```text
loaded → result → done
```

38. **Loaded event payload:** After the complete model and prompt pass all checks, emit:

```json
{
  "file": "transformer-weights-e1000-l1-d32-h2-ff128-ctx32.json",
  "prompt": "once upon a time"
}
```

39. **Result event payload:** Emit the completed text in:

```json
{
  "text": "once upon a time a small cat lived..."
}
```

40. **Done event payload:** Emit exactly one `done` event after `result`; it contains no training epoch or loss data.

41. **No fake training events:** Do not reuse `init`, `epoch`, or training `done` fields for saved-model generation.

42. **Completed-result display:** Show the loaded filename, prompt, and completed generated text. Do not stream individual generated tokens.

43. **Safe SSE errors:** Model-selection, model-validation, prompt-tokenization, prompt-length, generation, and deadline failures expose only approved safe messages and no exception text, traceback, path, raw model value, or numerical state.

44. **Generic named-model error:** Use `The saved Transformer model could not be loaded.` when a specifically named file cannot be safely loaded.

45. **Prompt tokenization error:** Use `The prompt contains text that this saved Transformer model cannot tokenize.`

46. **Prompt length error:** Use `The prompt must contain no more than 16 tokens.`

47. **Empty prompt error:** Use `The prompt must not be empty.`

48. **Five-minute generation deadline:** Use a monotonic five-minute deadline for saved-model token generation.

49. **Deadline error:** Use `Saved Transformer generation exceeded its time limit.`

50. **Cooperative deadline stopping:** Finish only the token calculation already in progress, stop before beginning another token, emit no `result` or `done`, and release the request slot.

51. **Cooperative disconnect stopping:** When the browser disconnects, finish only already-started token work, stop before another token, emit no later successful events, discard request state, and release the request slot.

52. **Loaded-after-validation boundary:** Emit `loaded` only after file safety, model validation, prompt tokenization, and the 1-to-16-token rule succeed.

53. **Shared Transformer slot:** Training and loading share one process-local nonblocking request slot. Only one Transformer job may run at a time.

54. **Immediate overlap rejection:** Do not queue. Return HTTP `429` with:

```json
{
  "detail": "Another Transformer request is already running."
}
```

55. **Clear old result:** Starting any new Transformer training or load command clears the previous result immediately.

56. **Training worker label:** Display exactly:

```text
Transformer worker processes: <actualWorkerCount>
```

57. **Dynamic worker count:** Obtain the displayed number from the existing actual worker-count boundary used for the current run; it may be 1 through 4.

58. **First sample only:** Prefix only the first public training `epoch` event's displayed sample with the worker label and a blank line.

59. **Display only:** The worker label must not:
    - become Transformer input;
    - change tokenization;
    - change generated text;
    - change deterministic fixtures;
    - enter the Saved Transformer Model;
    - become persistence metadata;
    - appear in later training samples.

60. **Training only:** Do not show `Transformer worker processes: 0` or another worker label during `/load-transformer`.

61. **Fresh training remains authoritative:** Every `POST /train-transformer` request still initializes fresh weights and performs full training.

62. **Inference-only loading:** A Saved Transformer Model may be loaded only for one Saved Transformer Generation Run, never as training state.

63. **ADR separation:** Keep ADR 0002 unchanged as the historical authority for Transformer training and process lifecycle. Add ADR 0003 for saved-model stateless generation.

64. **Glossary separation:** Update `CONTEXT.md` to distinguish Transformer Training Runs from Saved Transformer Generation Runs and their event streams.

65. **No model file repair:** Never delete, modify, repair, or rewrite an invalid candidate merely because a load request encountered it.

## Current behavior verified from files or tools

- The latest supplied Python Backend export implements and registers `POST /train-transformer`.
- `TrainTransformerRequest` currently exposes:
  - `epochs`, default `300`, strict range `50..2000`;
  - `temperature`, default `0.8`, finite range `0.1..2.0`;
  - `topP`, default `0.9`, finite range `0.1..1.0`;
  - `numLayers`, default `2`, strict range `1..6`;
  - `maxTokens`, default `40`, strict range `3..500`.
- The current training route reserves one process-local nonblocking run slot, returns a streaming response, and releases the slot through its stream lifecycle.
- Current successful training emits `init`, approximately fifty `epoch` events, and `done`.
- A current `epoch` payload contains exactly `epoch`, `loss`, and `sample`.
- The current training `done` payload contains exactly `architecture`, `finalLoss`, and `samples`.
- Current generated training samples use the latest 16 accumulated token IDs as the forward context.
- Current training sample randomness uses a new Mulberry32 stream seeded with `(42 + epoch) modulo 2³²`.
- The Request-Scoped Worker Group calculates its actual worker count from one `os.cpu_count()` observation, bounded to one through four.
- The worker count is currently internal and no existing public sample contains `Transformer worker processes`.
- Current worker groups use exactly four Logical Training Shards regardless of actual worker count.
- The current persistence boundary writes complete configuration-specific files named like:

  ```text
  transformer-weights-e1000-l1-d32-h2-ff128-ctx32.json
  ```

- The current persistence code already performs substantial Saved Transformer Model structure validation before serialization.
- The current Python application has no `POST /load-transformer` endpoint or saved-model generation request model.
- The current frontend training hook splits all input by whitespace and converts the first five positions to training numbers with defaults.
- The current frontend recognizes only training `init`, `epoch`, and `done` payloads.
- Therefore, a `File:` command cannot currently reach a saved-model loader without the approved parser and event-handling change.
- The current glossary states that Saved Transformer Models are never loaded for later requests, so the approved Phase 6 behavior requires a glossary revision.
- Phase 5 and ADR 0002 deliberately kept model loading, resume, and caching outside training scope.
- `POST /train-embed` is already implemented and registered; it is not a Phase 6 conversion target.
- Official FastAPI documentation confirms that `StreamingResponse` can emit yielded chunks without converting each chunk to JSON.
- Official Starlette documentation exposes `await request.is_disconnected()` for streamed-request disconnect observation.
- Official Python 3.12 `pathlib` documentation provides `Path.resolve()`, `Path.is_symlink()`, and Windows `Path.is_junction()` for the approved path-safety checks.
- Official architecture guidance supports recording the Phase 6 inference boundary as a new ADR rather than rewriting the history of ADR 0002.
- No Phase 6 production code was implemented during this grill-with-docs workflow.
- No backend tests, frontend tests, type checks, Ruff checks, mypy checks, browser checks, or server commands were executed during this workflow.

## Desired behavior

### Training command

- Continue sending numeric commands to `POST /train-transformer`.
- Continue validating the existing five-field request contract.
- Continue creating fresh weights and a request-scoped worker group.
- Continue emitting the existing training event sequence and payload fields.
- Add the worker-process label only to the first public epoch sample's display text.
- Preserve all later generated samples, final loss, persistence, cleanup, and deterministic model behavior.

### Saved-model command

- Route case-insensitive `File:` commands to `POST /load-transformer`.
- Parse exactly three `|`-separated sections:
  1. file selector;
  2. starting prompt;
  3. temperature, top-p, and maximum tokens.
- Use `modelFile: null` when the file selector is empty.
- Reject malformed commands locally before sending a request.
- Validate the structured request again in FastAPI/Pydantic.
- Reserve the same Transformer job slot used by training.
- Safely select and read one model snapshot.
- Strictly validate the current model format.
- Tokenize the trimmed prompt using the model's BPE artifacts.
- Reject zero tokens, unsupported text, or more than 16 tokens.
- Emit `loaded`, generate with seed `42`, emit `result`, then emit `done`.
- Display only the current request's filename, prompt, and completed result.
- Release all request-local state after success, error, disconnect, cancellation, or deadline.

### Display examples

Specific model:

```text
File: transformer-weights-e1000-l1-d32-h2-ff128-ctx32.json | once upon a time | 0.8 0.9 40
```

Latest valid model:

```text
File: | once upon a time | 0.8 0.9 40
```

Training sample:

```text
Transformer worker processes: 4

once upon a time a small cat lived...
```

Loaded result:

```text
Loaded: transformer-weights-e1000-l1-d32-h2-ff128-ctx32.json

Prompt:
once upon a time

Generated text:
once upon a time a small cat lived in a village...
```

## Domain model

### Terms created or changed

- **Transformer Training Run:** One complete decoder-only training execution using fresh weights, the fixed Transformer Training Corpus, and validated training and generation settings. It never loads or resumes from a Saved Transformer Model.

- **Saved Transformer Generation Run:** One independent inference request that loads one validated Saved Transformer Model and generates one continuation from one starting prompt. It retains no conversational state and never changes or resumes training.

- **Saved Transformer Model:** A complete JSON artifact produced by a successful Transformer Training Run. It may be loaded for inference but is never a checkpoint, resume point, cache shortcut, or initial training state.

- **Saved Transformer Event Stream:** The public load-route stream `loaded → result → done`, or safe `error` behavior when successful completion is not possible.

- **Request-Scoped Worker Group:** The one-through-four worker processes owned by one Transformer Training Run. A Saved Transformer Generation Run does not create this group.

### Important relationships

- One Transformer Training Run creates at most one complete configuration-specific Saved Transformer Model.
- One Saved Transformer Model may be selected by many independent Saved Transformer Generation Runs.
- One Saved Transformer Generation Run selects exactly one model snapshot.
- One Saved Transformer Generation Run has exactly one trimmed starting prompt.
- One successful Saved Transformer Generation Run emits exactly one `loaded`, one `result`, and one `done`.
- A named model request either uses that exact model or fails; it never chooses another.
- A latest-model request may inspect several candidates but loads exactly one newest strictly valid model.
- Transformer Training Runs and Saved Transformer Generation Runs compete for the same process-local request slot.
- Transformer worker processes belong only to training and are not created for loading.
- The worker-process label represents the number of spawned training helpers, not a count of guaranteed physical CPU cores.
- ADR 0002 continues to govern training; ADR 0003 governs saved-model inference.

### Domain artifacts

- [CONTEXT.md](CONTEXT.md)
- [ADR 0002 — Stabilize Python Transformer Training and Process Lifecycle](0002-stabilize-python-transformer-training-and-process-lifecycle.md)
- [ADR 0003 — Load Saved Transformer Models for Stateless Generation](docs/adr/0003-load-saved-transformer-models-for-stateless-generation.md)

## Architectural decisions

- [ADR 0003 — Load Saved Transformer Models for Stateless Generation](docs/adr/0003-load-saved-transformer-models-for-stateless-generation.md)

This decision passed the ADR gate because it creates a durable inference boundary separate from training, deliberately rejects several credible selection, caching, process, and session alternatives, and would be costly and confusing to reverse without preserving its rationale.

ADR 0002 remains unchanged as the historical and continuing authority for Transformer training, fresh initialization, multiprocessing, shared memory, cleanup, and persistence-before-`done`.

## Constraints

- Use Python 3.12 or newer.
- Use FastAPI, Pydantic, NumPy, the Python standard library, and existing project dependencies.
- Use Poetry for Python dependency and command execution.
- Use the current TypeScript/Vite frontend and its existing page structure.
- Preserve all completed backend endpoints:
  - `GET /health`;
  - `POST /simple-chat`;
  - `POST /bpe-tokenize`;
  - `POST /neural-net`;
  - `POST /train-embed`;
  - `POST /train-transformer`.
- Add `POST /load-transformer` without removing or renaming completed routes.
- Keep reusable Transformer/model operations separate from route orchestration.
- Preserve camelCase public aliases and Python snake_case internally.
- Preserve finite-number validation and the existing generation bounds.
- Search only the backend `.data` directory.
- Accept only current Python Phase 5 Saved Transformer Model files.
- Keep loaded state request-local and discard it after the response.
- Keep training and loading mutually exclusive within one FastAPI process.
- Keep generation deterministic for identical model bytes and request values.
- Keep the generated result complete rather than token-streamed.
- Keep the worker label presentation-only.
- Do not claim CPU-core affinity or guaranteed physical-core use.
- Do not claim test success until commands are executed successfully in implementation.
- Do not introduce dependencies or regenerate lockfiles unless later implementation evidence proves it unavoidable.
- Use Windows PowerShell commands in later implementation instructions.
- Treat the latest complete supplied Python Backend export as current-code truth and the TypeScript files as frontend/behavior references.

## Edge cases and failure behavior

- **Case-insensitive command prefix:** `File:`, `file:`, and `FILE:` all choose the load route.
- **Leading command whitespace:** Ignore leading whitespace before detecting `File:`.
- **Missing command sections:** Reject in the frontend and show the approved usage help.
- **Extra command separators:** Reject because the prompt cannot contain `|`.
- **Empty prompt:** Show `The prompt must not be empty.`
- **Prompt cannot tokenize:** Show `The prompt contains text that this saved Transformer model cannot tokenize.`
- **Prompt exceeds 16 tokens:** Show `The prompt must contain no more than 16 tokens.`
- **Invalid generation number or range:** Reject through frontend validation and backend Pydantic validation; structured backend validation remains HTTP `422`.
- **Specific file missing:** Emit the generic safe loading error.
- **Specific file malformed or incompatible:** Emit the generic safe loading error and never fall back.
- **Specific filename case differs:** Reject even on a normally case-insensitive Windows filesystem.
- **Specific filename contains path syntax:** Reject before opening it.
- **`.data` or file is a link/junction:** Reject.
- **Resolved path escapes `.data`:** Reject.
- **Latest search has no matching files:** Emit `No valid saved Transformer model was found.`
- **Newest matching file is invalid:** Skip it and inspect the next candidate.
- **Equal modification times:** Choose the alphabetically greatest valid filename.
- **All latest candidates invalid:** Emit `No valid saved Transformer model was found.`
- **Model changes after read:** Continue using the already validated in-memory snapshot.
- **Model changes before a later request:** The later request rereads and revalidates it.
- **Large model file:** Attempt the read without a size cap; on read, memory, decode, parse, or validation failure, expose only the safe loading error.
- **Overlapping training or loading:** Reject the second request immediately with HTTP `429`; do not queue.
- **Disconnect before successful events:** Stop without later events and release the slot.
- **Disconnect after `loaded`:** Stop before another token when observed and emit no `result` or `done`.
- **Generation deadline after `loaded`:** Emit the safe deadline error, no `result` or `done`, then release the slot.
- **Unexpected internal generation failure:** Emit only the applicable safe `error`, clean request-local state, and release the slot.
- **Training first epoch:** Prefix the sample with the dynamic worker-process label.
- **Later training epochs:** Do not repeat the label.
- **Saved-model generation:** Do not display a worker label.
- **New request:** Clear the previous Transformer result immediately.

## Testing expectations

### Frontend command and routing

- Prove numeric commands still create the exact existing training body.
- Prove `File:`, `file:`, and `FILE:` select `/load-transformer`.
- Prove named and empty file selectors create the exact structured load bodies.
- Prove outer prompt spaces are trimmed and interior spaces are preserved.
- Prove malformed section counts, missing prompt/settings, extra `|`, invalid numbers, and out-of-range values are rejected before fetch.
- Prove a new request clears the previous rendered result.
- Prove training events continue to render unchanged except for the approved first-sample label.
- Prove `loaded`, `result`, `done`, and `error` render the approved load behavior.

### Request validation

- Prove the exact `LoadTransformerRequest` fields, aliases, types, and bounds.
- Prove `modelFile` accepts either `null` or one nonempty string subject to semantic filename validation.
- Prove empty/whitespace prompt rejection.
- Prove numeric strings, Booleans, non-finite numbers, and values outside bounds receive HTTP `422`.
- Prove malformed structured requests fail before slot reservation or file access.

### Safe file selection

- Prove exact-case named lookup on Windows-compatible tests.
- Prove missing, differently capitalized, path-containing, absolute, parent-traversal, symlink, junction, and resolved-outside paths fail safely.
- Prove only ordinary files inside the real `.data` directory are candidates.
- Prove latest candidates are ordered by modification time and then descending filename.
- Prove latest selection skips invalid candidates.
- Prove a named invalid file never falls back.
- Prove no-valid-model uses its distinct safe message.
- Prove one request reads a selected file once.
- Prove a later request sees changed disk contents because no model cache exists.

### Strict model validation

- Prove rejection of wrong or missing top-level keys.
- Prove rejection of wrong model type.
- Prove fixed architecture fields and supported layer count.
- Prove Vocabulary and Merge Table types, ordering, references, and consistency.
- Prove exact required weight groups and block count.
- Prove exact parameter-array lengths from the canonical layout.
- Prove rejection of Booleans, strings, `NaN`, infinity, and missing numeric values.
- Prove filename configuration agrees with validated JSON configuration.
- Prove current Python Phase 5 files load while older/different structures do not.
- Prove validation produces fresh in-memory containers without exposing mutable global state.

### Prompt and deterministic generation

- Prove the saved Vocabulary and Merge Table, not current training preprocessing, tokenize the prompt.
- Prove unsupported text fails without partial-token fallback.
- Prove 1-token and 16-token prompts are accepted.
- Prove 17-token prompts are rejected.
- Prove seed `42` reproduces the exact same output for identical inputs.
- Prove changed model bytes or generation settings can change output.
- Prove `result.text` contains prompt plus continuation.
- Prove loading creates no worker group, process, pipe, or shared memory.
- Prove blocking work is off the route's event-loop thread where required.

### SSE contract and display

- Prove successful exact order `loaded → result → done`.
- Prove exact payload key sets for every event.
- Prove `loaded` occurs only after model and prompt validation.
- Prove file/prompt failures emit only safe `error` behavior and no successful completion events.
- Prove failures after `loaded` emit no `result` or `done`.
- Prove no training `init`, `epoch`, `loss`, architecture, or sample collection fields leak into load events.
- Prove filename, prompt, and complete generated text render in the existing results area.
- Prove internal paths, exceptions, tracebacks, model arrays, and resource identifiers never reach clients.

### Shared lifecycle

- Prove training blocks loading and loading blocks training through the same nonblocking slot.
- Prove the second request receives immediate HTTP `429` with the approved detail.
- Prove the slot is released after:
  - success;
  - named-file error;
  - no-valid-latest error;
  - prompt error;
  - generation error;
  - deadline;
  - disconnect;
  - cancellation.
- Prove no queue or delayed acquisition.
- Prove request-local loaded state is discarded between sequential requests.

### Deadline and disconnect

- Prove the monotonic five-minute generation deadline.
- Prove cancellation is checked between generated tokens.
- Prove an already-started token calculation is drained before state is discarded.
- Prove deadline emits the approved safe message.
- Prove disconnect emits no later success event.
- Prove the event loop remains responsive during file and generation work.

### Worker-process label

- Prove the displayed value equals the current run's actual worker count.
- Prove boundaries `None/0 → 1`, `1 → 1`, `2 → 2`, `4 → 4`, and values greater than four → `4`.
- Prove the first public training epoch sample is prefixed exactly once.
- Prove later epoch samples are unchanged.
- Prove raw Generated Text Sample objects, deterministic fixtures, `done.samples`, and Saved Transformer Models do not contain the label.
- Prove `/load-transformer` displays no worker label.

### Regression and quality

- Preserve all existing backend endpoint tests.
- Preserve Phase 5 numerical, worker, worker-group, route, persistence, completion, and deterministic fixtures.
- Add focused tests before running the complete suite.
- Run and report actual outcomes for:

  ```powershell
  poetry run pytest
  poetry run ruff check .
  poetry run ruff format --check .
  poetry run mypy src
  ```

- Run the frontend's configured type check and focused tests.
- Perform a practical two-server browser/Vite-proxy check for:
  - numeric training command;
  - named `File:` command;
  - latest `File:` command;
  - displayed worker-process count;
  - load errors;
  - new-request clearing.

No command was run during this grill-with-docs workflow.

## Risks and safeguards

- **Risk:** A filename command is still parsed as training defaults.
  - **Safeguard:** Detect case-insensitive `File:` before the existing numeric parser and cover both routes with frontend tests.

- **Risk:** User-supplied input escapes `.data`.
  - **Safeguard:** Accept only an exact plain filename, reject path syntax and links/junctions, resolve paths, and confirm the real destination remains inside the genuine model directory.

- **Risk:** Windows case-insensitive lookup loads a differently capitalized name.
  - **Safeguard:** Enumerate candidate names and require exact string equality before opening a named file.

- **Risk:** A damaged newest file makes `File:` unusable.
  - **Safeguard:** Validate candidates newest-to-oldest and skip invalid candidates only for latest selection.

- **Risk:** A named request silently loads the wrong model.
  - **Safeguard:** Never fall back from a named file.

- **Risk:** Partial validation lets a malformed array fail during generation.
  - **Safeguard:** Derive the complete canonical layout and validate every required array and finite value before `loaded`.

- **Risk:** Old TypeScript files are misinterpreted.
  - **Safeguard:** Accept only the current Python Phase 5 model contract.

- **Risk:** A huge JSON file exhausts memory or holds the Transformer slot for a long time.
  - **Safeguard:** Run file work off the event loop, catch specific failures behind a safe response, and document that no file-size cap was chosen.

- **Risk:** Reusing loaded state creates stale or cross-request behavior.
  - **Safeguard:** Read and validate every request and discard all loaded state afterward.

- **Risk:** The disk file changes while generation is running.
  - **Safeguard:** Generate from one validated in-memory snapshot.

- **Risk:** Unsupported prompt content is silently lost.
  - **Safeguard:** Require complete tokenization and fail explicitly.

- **Risk:** A long prompt changes the approved latest-16 generation behavior silently.
  - **Safeguard:** Reject starting prompts longer than 16 tokens.

- **Risk:** Saved-model output is nondeterministic and tests become flaky.
  - **Safeguard:** Use seed `42` for every independent generation request.

- **Risk:** Loading starts expensive training workers unnecessarily.
  - **Safeguard:** Keep generation in the parent process and offload blocking work only to same-process threads.

- **Risk:** CPU-bound work freezes FastAPI.
  - **Safeguard:** Use thread offloading and cooperative boundary checks; do not claim thread offloading provides multi-core acceleration.

- **Risk:** A disconnected request retains the only Transformer slot.
  - **Safeguard:** Observe disconnects between tokens, drain started work, clean request state, and release the slot in a final lifecycle path.

- **Risk:** A 500-token request runs indefinitely.
  - **Safeguard:** Use a five-minute monotonic generation deadline.

- **Risk:** The load route and training route run concurrently against substantial memory.
  - **Safeguard:** Share one nonblocking Transformer slot and reject overlap with HTTP `429`.

- **Risk:** The worker label changes deterministic text or saved artifacts.
  - **Safeguard:** Add it only when formatting the first public training epoch payload.

- **Risk:** The label is incorrectly described as physical CPU cores.
  - **Safeguard:** Use the exact term `Transformer worker processes`.

- **Risk:** New load events are forced into fake epoch/loss structures.
  - **Safeguard:** Use dedicated `loaded`, `result`, `done`, and safe `error` events.

- **Risk:** Frontend changes grow into a redesign.
  - **Safeguard:** Keep the current input, button, page, and result area; change only parsing, routing, event types, and rendering.

- **Risk:** ADR 0002 is rewritten and historical reasoning is lost.
  - **Safeguard:** Keep ADR 0002 and add ADR 0003 as a new decision record.

- **Risk:** The glossary continues saying models are never loaded.
  - **Safeguard:** Replace the affected Transformer definitions and add the approved loading terms before writing the Phase 6 specification.

- **Risk:** Implementation claims unsupported success.
  - **Safeguard:** Run every required command later and report observed results honestly.

## Open questions

- None block writing the Phase 6 specification.
- Exact internal function, class, module, and test-file decomposition may be refined by `to-spec-prompt` without changing the confirmed public behavior.
- The specification should resolve the smallest typed frontend state shape for `loaded`, `result`, `done`, and `error` while preserving the confirmed payloads and display.
- The specification should define the exact current-format validation seam by reusing the canonical parameter-layout and Saved Transformer Model types rather than duplicating numerical structure rules.
- The specification should preserve all approved safe messages and distinguish HTTP `422` request validation from streamed semantic/model errors.
- No Phase 6 implementation or validation command was performed during this grill-with-docs workflow.

## Source material consulted

- `GRILL_WITH_DOCS_PROMPT.md`
- Existing Phase 5 `GRILL_WITH_DOCS_RESULT.md`
- Current Phase 5 `SPEC.md`
- Current `CONTEXT.md`
- `0002-stabilize-python-transformer-training-and-process-lifecycle.md`
- Latest `py_llm_pipeline_explorer_file_structure.md` supplied on 2026-07-30
- Latest `llm_works_file_structure.md`
- `backend/pyproject.toml`
- `backend/src/how_llms_work/main.py`
- `backend/src/how_llms_work/schemas.py`
- `backend/src/how_llms_work/sse.py`
- `backend/src/how_llms_work/ml/bpe.py`
- `backend/src/how_llms_work/ml/math_utils.py`
- `backend/src/how_llms_work/ml/matrix.py`
- `backend/src/how_llms_work/ml/transformer.py`
- `backend/src/how_llms_work/ml/transformer_worker.py`
- `backend/src/how_llms_work/routes/train_transformer.py`
- Current Transformer numerical, completion, worker, worker-group, persistence, and route tests
- TypeScript frontend `use-train-transformer-chat.tsx`
- TypeScript `train-transformer-result` component
- Shared TypeScript SSE reader and error parser
- TypeScript Transformer request schema, route, trainer, generator, worker, and serializer
- Official FastAPI documentation for streaming responses
- Official Starlette documentation for request-disconnection detection
- Official Python documentation for:
  - `asyncio.to_thread()`;
  - monotonic time;
  - JSON parsing;
  - filesystem metadata;
  - path resolution;
  - symbolic-link and Windows-junction detection;
  - random seeds.
- Official Pydantic documentation for request models, aliases, strict and finite validation
- Official pytest documentation for temporary paths, monkeypatching, parametrization, and markers
- Microsoft guidance for maintaining append-only Architecture Decision Records

## Recommended next step

Run `to-spec-prompt` using:

- this `GRILL_WITH_DOCS_RESULT.md`;
- the approved replacement `CONTEXT.md`;
- ADR 0002;
- the approved new ADR 0003;
- the current Phase 5 `SPEC.md`;
- the latest Python Backend source export;
- the latest TypeScript Reference Implementation.

The Phase 6 specification must remain limited to:

- stateless Saved Transformer Model generation;
- safe model selection and strict current-format loading;
- the dedicated `/load-transformer` request and SSE contract;
- the minimal approved frontend parser/event-display change;
- the first-training-sample worker-process label;
- the shared Transformer request slot;
- lifecycle, deadline, disconnect, validation, security, and regression behavior.

It must not create implementation tickets, production code, commits, or code-review output.
