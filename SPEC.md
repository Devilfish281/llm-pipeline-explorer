---
workflow: engineering-prompt-chain
document_type: specification
prompt_name: to-spec-prompt
status: ready-for-agent
triage_label: ready-for-agent
version: 1
source_document: GRILL_WITH_DOCS_RESULT.md
recommended_next_prompt: to-tickets-prompt
---

# Specification: Generate Text from Saved Transformer Models Without Retraining

## Problem

Learners can train a decoder-only Transformer, watch its progress, and persist a complete configuration-specific Saved Transformer Model, but they cannot select that model later and generate text from a starting prompt. Every current Transformer request is interpreted as a training command and begins again with fresh weights, even when a suitable completed model already exists.

This makes the saved artifact difficult to use as an educational inference result, forces unnecessary retraining, and prevents the browser from demonstrating the distinction between a Transformer Training Run and a stateless Saved Transformer Generation Run. The current display also hides the number of worker processes created for a training run, even though that number is already determined by the backend.

The limitation affects learners using the Train Transformer Learning Demo, the TypeScript/Vite frontend that consumes its HTTP and SSE contracts, and maintainers responsible for model safety, deterministic generation, resource cleanup, and compatibility on Windows 11.

## Solution

The existing Transformer input box and Send button will accept either the existing five-number training command or a case-insensitive `File:` saved-model command. A saved-model command will select either one exact Saved Transformer Model filename or the newest strictly valid model, send a structured request to `POST /load-transformer`, and display the loaded filename, trimmed prompt, and one complete deterministic prompt-plus-continuation result.

Each Saved Transformer Generation Run will be independent and stateless. It will reserve the same process-local Transformer request slot used by training, safely select one ordinary file from the real backend `.data` directory, read it once, strictly validate the current Python Phase 5 model format, tokenize the prompt with that model's ordered Vocabulary and Merge Table, and generate in the backend parent process with seed `42`. Loading will never resume, initialize, skip, cache, or alter training.

The successful Saved Transformer Event Stream will be `loaded → result → done`. Validation, selection, tokenization, deadline, and generation failures will use sanitized error behavior without exposing paths, exceptions, model values, or numerical state. Blocking file work and generation will run away from the FastAPI event-loop thread, while disconnect and monotonic deadline checks stop cooperatively between generated-token calculations.

Transformer training will remain authoritative and fresh-weight. Its existing request shape, numerical behavior, worker protocol, persistence, and event payload fields will remain unchanged, except that the first public training sample will be prefixed once with `Transformer worker processes: <actualWorkerCount>` as display-only text and the shared overlap response will use the wording approved for either kind of Transformer request.

## User Stories

1. As a learner, I want to load a completed Saved Transformer Model, so that I can generate text without retraining it.
2. As a learner, I want to use the existing Transformer input box and Send button, so that saved-model generation feels like part of the current Learning Demo.
3. As a learner, I want numeric Transformer commands to keep starting fresh training, so that the existing training experience remains available.
4. As a learner, I want commands beginning with any capitalization of `File:` to select saved-model generation, so that command routing is forgiving about prefix capitalization.
5. As a learner, I want leading whitespace before `File:` ignored, so that harmless command indentation does not change routing.
6. As a learner, I want to name one exact Saved Transformer Model file, so that I control which trained model generates the continuation.
7. As a learner, I want an empty file selector after `File:` to choose the newest strictly valid model, so that I can use the latest usable training result without remembering its filename.
8. As a learner, I want a named-file request either to use that exact file or fail, so that another model is never silently substituted.
9. As a learner, I want equally recent valid models resolved deterministically by exact filename, so that latest-model selection is reproducible.
10. As a learner, I want malformed saved-model commands rejected before a network request, so that I receive immediate and helpful usage feedback.
11. As a learner, I want the command to contain exactly a file selector, a starting prompt, and generation settings, so that its meaning is unambiguous.
12. As a learner, I want an empty or whitespace-only prompt rejected, so that generation always has a meaningful starting context.
13. As a learner, I want only the prompt's outer whitespace trimmed, so that meaningful spacing inside the prompt remains unchanged.
14. As a learner, I want a prompt containing `|` rejected, so that the command separator cannot be interpreted ambiguously.
15. As a learner, I want invalid temperature, top-p, or maximum-token settings rejected locally, so that an obviously invalid command is not sent.
16. As a learner, I want the backend to validate the structured request again, so that bypassing the frontend cannot bypass the public contract.
17. As a learner, I want the selected model's own Vocabulary and Merge Table used for my prompt, so that generation matches the model that was actually trained.
18. As a learner, I want unsupported prompt text rejected explicitly, so that characters or fragments are never silently dropped or replaced.
19. As a learner, I want prompts from one through sixteen tokens accepted, so that I can use the complete approved starting-context range.
20. As a learner, I want prompts longer than sixteen tokens rejected, so that the backend never silently discards the beginning of my prompt.
21. As a learner, I want the generated result to begin with my trimmed prompt, so that the continuation is shown in context.
22. As a learner, I want `maxTokens` to control the number of newly generated tokens, so that I can choose the continuation length within the existing range.
23. As a learner, I want identical valid requests against identical model contents to reproduce the same text, so that the educational result is deterministic.
24. As a learner, I want the loaded filename shown before the result, so that I know which model produced the text.
25. As a learner, I want my prompt displayed separately from the generated text, so that input and output are easy to distinguish.
26. As a learner, I want one complete result rather than token-by-token display, so that saved-model inference remains distinct from the training animation.
27. As a learner, I want the previous Transformer result cleared when I start a new training or load command, so that stale output is not confused with the current request.
28. As a learner, I want a clear message when no valid saved Transformer model exists, so that absence is not mistaken for a server failure.
29. As a learner, I want a safe generic message when a named model cannot be loaded, so that internal filesystem details are not exposed.
30. As a learner, I want a specific message when my prompt cannot be tokenized by the selected model, so that I can revise the prompt.
31. As a learner, I want a specific message when my prompt exceeds sixteen tokens, so that I know the allowed limit.
32. As a learner, I want a specific message when my prompt is empty, so that I know what to correct.
33. As a learner, I want a specific message when saved-model generation exceeds five minutes, so that a stalled request does not appear to run forever.
34. As a learner, I want a disconnected saved-model request to stop without later successful events, so that abandoned work does not complete invisibly.
35. As a learner, I want a second Transformer request rejected immediately while another is active, so that the application does not queue expensive jobs.
36. As a learner, I want the overlap message to apply consistently to training and loading, so that both commands communicate the shared resource limit clearly.
37. As a learner, I want the first training sample to show the actual number of Transformer worker processes, so that the demo exposes its multiprocessing behavior.
38. As a learner, I want the worker-process label shown only once, so that later samples remain uncluttered.
39. As a learner, I want no worker-process label during saved-model generation, so that inference is not presented as multiprocessing training.
40. As a learner, I want the worker-process label to be presentation text only, so that it cannot change the generated sample or trained model.
41. As a frontend consumer, I want saved-model commands sent to `POST /load-transformer`, so that training and inference remain separate public operations.
42. As a frontend consumer, I want training commands to continue using `POST /train-transformer`, so that the existing Frontend Contract remains compatible.
43. As a frontend consumer, I want the load request to contain only `modelFile`, `prompt`, `temperature`, `topP`, and `maxTokens`, so that the request shape stays minimal and typed.
44. As a frontend consumer, I want an empty selector represented by `modelFile: null`, so that latest-model selection does not depend on a magic filename.
45. As a frontend consumer, I want exact `loaded`, `result`, `done`, and `error` behavior, so that rendering does not infer training fields for inference.
46. As a frontend consumer, I want `loaded` to include only the selected exact filename and trimmed prompt, so that the initial display state is complete and safe.
47. As a frontend consumer, I want `result` to include only the completed text, so that generated output is not mixed with model internals.
48. As a frontend consumer, I want `done` to contain no training epoch, loss, architecture, or sample collection, so that inference completion remains distinct from training completion.
49. As a frontend consumer, I want load errors represented as safe stream errors rather than fake `init` or `epoch` events, so that event handling remains semantically correct.
50. As a project maintainer, I want only current Python Phase 5 Saved Transformer Models accepted, so that old or unrelated formats are never guessed or migrated implicitly.
51. As a project maintainer, I want the model's exact top-level and nested fields validated, so that missing or unexpected structure is rejected before generation.
52. As a project maintainer, I want duplicate JSON object keys rejected, so that a model has one unambiguous value for every field.
53. As a project maintainer, I want the exact model type validated, so that unrelated JSON files cannot be treated as Transformers.
54. As a project maintainer, I want the supported configuration and layer count validated, so that the loaded architecture matches the current implementation.
55. As a project maintainer, I want every required parameter array and Transformer block validated, so that generation cannot begin with partial weights.
56. As a project maintainer, I want parameter-array lengths derived from the canonical layout, so that model validation and numerical interpretation use one authority.
57. As a project maintainer, I want Booleans, strings, missing values, NaN, and infinity rejected from numerical arrays, so that loaded parameters are finite ordinary numbers.
58. As a project maintainer, I want the ordered Vocabulary and Merge Table validated as a coherent BPE model, so that prompt tokenization is deterministic and complete.
59. As a project maintainer, I want filename configuration fields checked against validated model configuration, so that a misleading filename cannot select incompatible contents.
60. As a project maintainer, I want exact filename capitalization enforced even on Windows, so that named selection behaves consistently across filesystems.
61. As a project maintainer, I want absolute paths, drive letters, parent traversal, and separators rejected, so that browser input cannot escape the model directory.
62. As a project maintainer, I want the real backend `.data` directory resolved from application code, so that behavior does not depend on the shell's current directory.
63. As a project maintainer, I want symbolic links and Windows junctions rejected for both the model directory and selected model, so that path indirection cannot bypass the storage boundary.
64. As a project maintainer, I want resolved candidate paths confirmed inside the genuine model directory, so that link or traversal tricks cannot select external files.
65. As a project maintainer, I want latest-model selection to inspect newest candidates first and skip invalid candidates, so that one damaged recent artifact does not block an older valid model.
66. As a project maintainer, I want invalid candidates left untouched, so that a read request never deletes, repairs, or rewrites model files.
67. As a project maintainer, I want every request to reread and revalidate its selected model, so that no cross-request cache can become stale.
68. As a project maintainer, I want one selected file read exactly once per request, so that generation uses one stable in-memory snapshot.
69. As a project maintainer, I want no application-level model file-size cap, so that current-format artifacts are attempted with available memory rather than rejected by an invented limit.
70. As a project maintainer, I want read, decode, parse, and validation failures sanitized, so that clients never receive paths, exceptions, or raw model data.
71. As a project maintainer, I want loaded parameters materialized as request-owned canonical numerical state, so that one request cannot mutate another request or global training data.
72. As a project maintainer, I want saved-model generation to run in the backend parent process, so that inference does not create worker processes, pipes, or shared memory.
73. As a project maintainer, I want blocking file and generation work kept off the FastAPI event-loop thread, so that other lightweight requests remain responsive.
74. As a project maintainer, I want disconnect and deadline checks between generated tokens, so that cancellation occurs at a safe deterministic boundary.
75. As a project maintainer, I want an already-started token calculation allowed to finish before stopping, so that request-owned numerical state is not discarded while active work still references it.
76. As a project maintainer, I want no `result` or `done` after a deadline or observed disconnect, so that incomplete generation cannot appear successful.
77. As a project maintainer, I want request-local loaded state discarded after every outcome, so that no conversational or model-selection state survives.
78. As a project maintainer, I want the shared Transformer slot released after success, errors, cancellation, disconnect, and deadline, so that later requests are not permanently blocked.
79. As a project maintainer, I want training and loading mutually exclusive only within the current FastAPI process, so that the implementation does not claim a machine-wide lock.
80. As a project maintainer, I want existing Health, Simple Chat, BPE, XOR, Word2Vec, and Transformer training endpoints preserved, so that Phase 6 does not regress completed Learning Demos.
81. As a project maintainer, I want every training request to continue initializing fresh weights, so that a Saved Transformer Model is never a checkpoint or resume source.
82. As a project maintainer, I want the first-sample worker label derived from the run's existing actual worker-count boundary, so that display and process orchestration cannot disagree.
83. As a project maintainer, I want raw Generated Text Samples, deterministic fixtures, `done.samples`, and persisted models to exclude the worker label, so that presentation does not contaminate domain data.
84. As a project maintainer, I want ADR 0002 to remain the authority for training lifecycle and ADR 0003 to govern stateless inference, so that architectural history remains clear.
85. As a project maintainer, I want canonical glossary terms used in code, tests, and planning, so that Transformer Training Runs and Saved Transformer Generation Runs are not conflated.
86. As a project developer, I want backend route and SSE behavior tested through FastAPI's public HTTP seam, so that internal refactoring does not invalidate behavior-focused tests.
87. As a project developer, I want frontend command parsing and event-to-display transitions tested as pure TypeScript functions, so that routing and validation are deterministic without a browser.
88. As a project developer, I want one practical Vite-proxy browser check, so that automated unit tests are supplemented by evidence of real UI integration.
89. As a project developer, I want focused safety and lifecycle fixtures under temporary directories, so that tests never depend on or modify real saved models.
90. As a project developer, I want existing numerical, persistence, worker, worker-group, and route regressions preserved, so that Phase 6 cannot weaken Phase 5 guarantees.
91. As a project developer, I want actual pytest, Ruff, mypy, frontend type-check, and focused-test results reported only after execution, so that completion claims remain trustworthy.

## Implementation Decisions

1. **Confirmed — Separate inference endpoint:** Add and register `POST /load-transformer`; keep `POST /train-transformer` as the only Transformer Training Run endpoint.
2. **Confirmed — Existing interface:** Reuse the existing Train Transformer page, input box, Send button, assistant-result area, and layout. Do not add a page, selector, second input, or redesign.
3. **Confirmed — Command routing:** After removing leading whitespace, detect `File:` case-insensitively before applying the current numeric training parser. Commands without that prefix continue through the training path.
4. **Confirmed — Command grammar:** Saved-model commands use exactly `File: filename | starting text | temperature top-P max-tokens`. Exactly three pipe-separated sections are required, and `|` is not escapable inside the prompt.
5. **Confirmed — Frontend request construction:** A nonempty file selector is sent verbatim as `modelFile`; an empty selector is sent as `modelFile: null`. The remaining public fields are `prompt`, `temperature`, `topP`, and `maxTokens`.
6. **Confirmed — Generation bounds:** Preserve the existing ranges: temperature `0.1..2.0`, top-p `0.1..1.0`, and maximum tokens `3..500`.
7. **Confirmed — Local validation:** The frontend rejects malformed section counts, missing prompt/settings, extra separators, non-finite or malformed numbers, and values outside the approved ranges before fetch, then renders usage or validation text in the existing result area.
8. **Confirmed — Prompt normalization:** Trim only leading and trailing prompt whitespace. Preserve all interior characters and spacing exactly.
9. **Confirmed — Dedicated request model:** Add a Pydantic model for Saved Transformer Generation Runs with snake_case internal attributes and camelCase aliases. `modelFile` accepts `null` or a strict nonempty string; semantic filename rules remain in the loader boundary.
10. **Confirmed — Standard request failures:** Malformed structured requests fail through FastAPI/Pydantic with HTTP `422` before slot reservation, filesystem access, or generation work.
11. **Confirmed — Shared request slot:** Training and loading use one route-owned, process-local, nonblocking Transformer request slot. There is no queue and no machine-wide or cross-process lock.
12. **Confirmed — Overlap response:** Any second valid training or loading request receives HTTP `429` with exactly `{"detail":"Another Transformer request is already running."}`.
13. **Confirmed — Shared slot lifecycle:** Slot ownership begins before model-selection or training preparation and is released as the final lifecycle action after success, handled error, disconnect, cancellation, or deadline cleanup.
14. **Confirmed — Dedicated route orchestration:** The load route owns HTTP validation handoff, slot reservation, SSE sequencing, disconnect observation, safe error mapping, and final cleanup. Reusable model loading, validation, tokenization, and generation remain outside route orchestration.
15. **Confirmed — Real model directory:** Resolve the backend `.data` directory from application code rather than the process working directory. Model selection cannot accept an arbitrary path.
16. **Confirmed — Approved filename boundary:** Named selection accepts only one exact plain filename matching the current configuration-specific Transformer persistence grammar. Reject empty named strings, absolute paths, drive-letter forms, parent references, `/`, `\`, and any unapproved name.
17. **Confirmed — Exact-case lookup:** Enumerate directory entries and require exact string equality for a named filename before opening it, including on case-insensitive Windows filesystems.
18. **Confirmed — No link traversal:** Reject the `.data` directory or selected candidate when it is a symbolic link or Windows junction. Resolve paths and require the selected destination to remain inside the genuine resolved `.data` directory.
19. **Confirmed — Ordinary-file candidates:** Consider only ordinary, non-link, non-junction files matching the approved Transformer filename grammar.
20. **Confirmed — Named selection semantics:** A named request validates only that exact candidate. Missing, differently capitalized, unsafe, unreadable, malformed, incompatible, or damaged named files never fall back to another file.
21. **Confirmed — Latest selection ordering:** For `modelFile: null`, order matching candidates by descending modification time and then descending exact filename. Validate in that order and select the first strictly valid model.
22. **Confirmed — Latest invalid candidates:** Skip invalid latest candidates without modifying them. When no strictly valid candidate exists, emit `No valid saved Transformer model was found.`
23. **Confirmed — One snapshot per request:** Open and read the selected model once, parse and validate that one byte/text snapshot, and generate only from the resulting request-owned in-memory state. Do not reopen during the request.
24. **Confirmed — No cross-request cache:** Every Saved Transformer Generation Run repeats selection, read, parse, and validation. Discard all loaded-model state after the stream ends.
25. **Confirmed — No file-size cap:** Do not reject a candidate solely because of size. Attempt the read with available memory and map concrete read, decode, parse, or memory failures to safe client behavior.
26. **Confirmed — Strict JSON objects:** Reject duplicate object keys, missing keys, unexpected keys, wrong container types, and values that cannot be represented by the current Saved Transformer Model contract.
27. **Confirmed — Exact current model shape:** Require top-level `type`, `config`, `vocab`, `merges`, and `weights`; require model type `decoder-transformer`; and accept no older TypeScript or alternate model structure.
28. **Confirmed — Exact configuration shape:** Require exactly `vocabSize`, `contextLen`, `embDim`, `numHeads`, `ffDim`, and `numLayers`. Fixed architecture values must match the current Python Transformer, and `numLayers` must be within the supported range.
29. **Confirmed — Exact weights shape:** Require exactly `tokEmb`, `posEmb`, `blocks`, `lnFGamma`, `lnFBeta`, `headW`, and `headB`; require the block count to equal `numLayers`.
30. **Confirmed — Exact block shape:** Every block contains exactly `ln1Gamma`, `ln1Beta`, `wQ`, `bQ`, `wK`, `bK`, `wV`, `bV`, `wO`, `bO`, `ln2Gamma`, `ln2Beta`, `ff1W`, `ff1B`, `ff2W`, and `ff2B`.
31. **Confirmed — Canonical length authority:** Derive every expected array length and canonical flattening order from the existing Transformer parameter-layout boundary. Do not duplicate numerical layout constants in the loader.
32. **Confirmed — Strict numerical values:** Every parameter entry must be an ordinary finite JSON number. Reject Booleans, strings, `null`, missing values, NaN, and infinity.
33. **Confirmed — Request-owned parameters:** Materialize validated values into one independent canonical `float32` parameter block and semantic views suitable for the existing forward and top-p sampling mathematics. Loaded state must not alias global preprocessing, training state, or another request.
34. **Confirmed — Vocabulary validation:** Require an ordered Vocabulary whose length equals `config.vocabSize`, whose entries satisfy the current tokenizer/model rules, and whose ordering is preserved exactly for token IDs and output reconstruction.
35. **Confirmed — Merge Table validation:** Require an ordered Merge Table of exact pair/merged records that is coherent with the model Vocabulary and can be applied without guessing, repairing, or dropping data.
36. **Confirmed — Filename/configuration agreement:** Parse the current persistence filename grammar and compare every encoded architecture field represented in the validated configuration. The epoch filename segment remains artifact metadata and is not invented from model contents.
37. **Confirmed — Loaded boundary:** Emit `loaded` only after path safety, complete model validation, parameter materialization, prompt tokenization, nonempty-token validation, and the sixteen-token maximum all succeed.
38. **Confirmed — Loaded payload:** Emit `loaded` with exactly `file` and `prompt`, using the selected exact filename and the trimmed prompt.
39. **Confirmed — Prompt tokenization:** Tokenize the complete trimmed prompt with the loaded model's ordered Merge Table and Vocabulary. Require every resulting token to resolve to a model token ID.
40. **Confirmed — Unsupported text:** Never drop, replace, normalize, or default unsupported text. Emit `The prompt contains text that this saved Transformer model cannot tokenize.`
41. **Confirmed — Prompt length:** Accept one through sixteen prompt tokens. Emit `The prompt must contain no more than 16 tokens.` for seventeen or more.
42. **Confirmed — Empty prompt:** Reject empty or whitespace-only prompts with `The prompt must not be empty.`
43. **Confirmed — Parent-process inference:** Saved Transformer Generation Runs do not create a Request-Scoped Worker Group, child process, pipe, Queue, Manager, shared-memory block, or worker-process label.
44. **Confirmed — Reused mathematics:** Reuse the current decoder-only forward pass, causal context behavior, stable softmax, temperature, top-p sampling, Vocabulary decoding, and latest-sixteen-token context rule where applicable. Do not introduce a framework or hosted model.
45. **Confirmed — Independent deterministic stream:** Use a request-owned Mulberry32 random stream seeded exactly with `42` for each Saved Transformer Generation Run. Identical validated model bytes and request values must reproduce the same output.
46. **Confirmed — Complete result:** Generate exactly up to `maxTokens` new tokens and emit one `result` whose `text` is the trimmed original prompt followed by its generated continuation.
47. **Confirmed — Result payload:** Emit `result` with exactly `text`. Do not emit token-level events or training fields.
48. **Confirmed — Done behavior:** After one successful `result`, emit exactly one `done`, containing no epoch, loss, architecture, sample collection, worker information, or model data, then terminate the stream.
49. **Assumption — Empty done payload:** Represent the successful load-route `done` data as an empty JSON object so the existing JSON SSE transport retains one valid `data:` object without inventing completion fields.
50. **Confirmed — Successful order:** The only successful Saved Transformer Event Stream order is `loaded → result → done`.
51. **Confirmed — Error payload:** Semantic/model/generation failures use an `error` event containing one `error` string selected from the route's closed safe-message mapping. Never include raw exception text, traceback, path, filename beyond an already approved loaded value, model arrays, resource identifiers, or numerical state.
52. **Confirmed — Named-model safe error:** Use `The saved Transformer model could not be loaded.` for unsafe, missing, unreadable, malformed, damaged, or incompatible specifically named candidates.
53. **Confirmed — Deadline:** Use one monotonic five-minute deadline for saved-model token generation. On expiry, emit `Saved Transformer generation exceeded its time limit.`, emit no `result` or `done`, and release request state.
54. **Confirmed — Cooperative stopping:** Check cancellation, disconnect, and deadline between generated-token calculations. A token calculation already in progress may finish, but another token must not begin after stopping is observed.
55. **Confirmed — Disconnect behavior:** An observed browser disconnect emits no later successful event. Drain already-started helper work, discard request-owned loaded state, and release the shared slot.
56. **Confirmed — Off-event-loop work:** File enumeration, metadata checks, file read, JSON decode, strict model validation, parameter reconstruction, tokenization where blocking, and generation execute away from the FastAPI event-loop thread. Same-process thread offloading is for responsiveness, not multi-core inference claims.
57. **Confirmed — Responsive orchestration:** Async route orchestration remains able to observe disconnect and deadline state while blocking helper work proceeds. The exact private scheduling mechanism may vary as long as token boundaries and cleanup guarantees remain observable.
58. **Confirmed — Training preservation:** Every `POST /train-transformer` request continues to create fresh weights, the fixed Transformer preprocessing, one Request-Scoped Worker Group, and the existing `init → epoch × approximately 50 → done` contract.
59. **Confirmed — Shared overlap wording:** Replace the training-only active-run detail with `Another Transformer request is already running.` so both routes expose the same shared-slot contract.
60. **Confirmed — Worker count source:** Obtain the display value from the existing actual worker count calculated once from one `os.cpu_count()` observation and bounded to one through four.
61. **Confirmed — Worker label format:** Prefix only the first public training epoch sample with exactly `Transformer worker processes: <actualWorkerCount>`, followed by one blank line and the unchanged sample text.
62. **Confirmed — Worker label presentation boundary:** Add the label only while formatting the first public epoch payload. It must not enter raw Generated Text Sample records, `done.samples`, tokenization, random streams, persistence, model metadata, or later samples.
63. **Confirmed — Frontend state separation:** Maintain typed training state separately from typed saved-model state. The minimal saved-model display state consists of the loaded filename/prompt, completed result text, and optional safe error; `done` is terminal and adds no display data.
64. **Confirmed — Frontend event discrimination:** The Transformer frontend may discriminate load payloads by exact safe payload shapes or by named SSE envelopes, but it must not reinterpret them as training `init`, `epoch`, or `done` data. A reader-wide refactor is not required unless implementation evidence makes it the smallest safe change.
65. **Confirmed — Result clearing scope:** Starting a new Transformer training or load command clears prior Transformer messages/results immediately without changing the history behavior of other Learning Demos.
66. **Confirmed — Documentation:** Use the updated glossary definitions for Transformer Training Run, Saved Transformer Generation Run, Saved Transformer Model, Saved Transformer Event Stream, and Request-Scoped Worker Group. Keep ADR 0002 unchanged and retain ADR 0003 as the inference authority.
67. **Confirmed — Dependency policy:** Use Python 3.12+, FastAPI, Pydantic, NumPy, the standard library, and the frontend's existing TypeScript/Vite/Vitest toolchain. No new backend or frontend dependency is presently required.
68. **Confirmed — Compatibility:** Preserve all completed endpoints, shared SSE media type/headers, public camelCase names, existing training numerical fixtures, persistence files, deterministic behavior, and frontend page structure.
69. **Confirmed — No success claims without execution:** Specification and ticket generation must not claim pytest, Ruff, mypy, frontend type checking, Vitest, browser, or server checks passed. Those outcomes belong to implementation evidence.

## Testing Decisions

- **Approved test seam — Backend HTTP/SSE:** Exercise `POST /load-transformer` and affected `POST /train-transformer` behavior through FastAPI's in-process `TestClient`, using the existing exact SSE parser, temporary model directories, fixed valid and invalid Saved Transformer Model fixtures, controlled clocks, controlled disconnect observers, and controlled generation helpers.
- **Why this seam:** It is the highest practical existing boundary for request validation, status codes, headers, exact event names and payloads, event ordering, safe errors, shared-slot ownership, lifecycle cleanup, and training regressions. It allows route internals to be refactored without weakening observable guarantees.
- **Observable backend behavior covered:** Exact route registration; request aliases/types/bounds; standard `422`; immediate shared `429`; exact named/latest selection; exact-case handling; path rejection; symlink/junction and resolved-boundary protection; candidate ordering; one-read/no-cache behavior; strict current-format validation; prompt tokenization and length checks; deterministic seed `42`; complete result text; `loaded → result → done`; safe errors; no worker creation; deadline/disconnect stopping; slot release; and first-training-sample worker labeling.
- **Backend modules exercised:** Saved Transformer request schema, shared Transformer request slot, load-route orchestration, model-directory selection boundary, Saved Transformer Model parser/validator, canonical parameter layout and views, BPE application/token-ID mapping, existing Transformer forward/generation mathematics, shared SSE formatter/response, and training epoch payload formatting.
- **Backend prior art:** Current route tests already use `TestClient`, exact SSE parsing with duplicate-key detection, controllable nonblocking run slots, temporary persistence boundaries, controlled worker groups, deterministic clocks/cancellation, and bounded real public Transformer integration.
- **Backend fixtures and controls:** Include one complete valid current-format model built through the public Saved Transformer Model boundary; small canonical-layout models where public configuration permits; malformed top-level/nested variants; wrong-length and non-finite arrays; duplicate JSON keys; exact-case filename pairs; newest-invalid/older-valid candidates; equal-mtime candidates; changed-between-requests snapshots; controlled read counters; deterministic token/logit fixtures; monotonic fake clocks; and disconnect/cancellation events.
- **Backend test types:** Focused schema tests, pure selection/validation tests at public reusable seams, route-level HTTP/SSE tests, lifecycle/concurrency tests, security/path tests, deterministic generation tests, existing Phase 5 regressions, and at least one bounded integration using real public model loading and generation boundaries.
- **Backend platform handling:** Exercise Windows junction behavior when the test environment permits creating junctions. Where operating-system privileges prevent it, retain a deterministic public path-classification seam and record the skipped platform limitation rather than claiming unexecuted junction coverage.
- **Approved test seam — Frontend command/state functions:** Extract or expose pure TypeScript functions for Transformer command classification/parsing and load-event-to-display-state transitions, then test them with the configured Vitest runner.
- **Why this seam:** `File:` recognition, local grammar validation, endpoint selection, request-body construction, prior-result clearing, and load-result rendering occur outside the backend HTTP boundary. Pure functions provide fast deterministic evidence without requiring a new browser automation dependency.
- **Observable frontend behavior covered:** Numeric command preservation; `File:`, `file:`, and `FILE:` routing; named/null selector bodies; prompt outer trimming/interior preservation; exact range checks; missing/extra section rejection; extra `|` rejection; helpful local usage output; old-result clearing; typed `loaded`, `result`, `done`, and `error` transitions; exact loaded/result display; unchanged training rendering except the first-sample label; and no worker label for loading.
- **Frontend modules exercised:** Transformer chat hook, command parser, endpoint/body selector, SSE event adapter, saved-model display state, and the existing Transformer result component or its smallest approved extension.
- **Frontend prior art:** The current hook already builds the five-field training body, receives JSON SSE data, accumulates epochs and samples, and renders the Transformer result component. The project already declares Vitest, so no new test framework is required.
- **Integration smoke seam:** After automated checks, run the existing two-server FastAPI/Vite setup and manually verify one numeric training command, one named `File:` command, one latest `File:` command, worker-count display, safe load errors, and immediate result clearing.
- **Do not test:** Private helper names, local variable names, a particular JSON-library implementation, exact thread identities beyond proving off-event-loop execution, exact wall-clock elapsed time, exact internal sorting implementation, private array-view construction, token-by-token UI updates that are out of scope, or browser layout pixels.
- **Known testability limitation:** In-process HTTP tests do not prove Vite proxy configuration or browser rendering, and pure frontend tests do not prove real server streaming. The focused two-server smoke check supplies that complementary evidence without making full browser automation a new dependency.
- **Required implementation validation:** Run focused backend tests first, then the complete backend suite and `poetry run ruff check .`, `poetry run ruff format --check .`, and `poetry run mypy src`. Run the frontend's configured type check and focused Vitest tests, then record the practical two-server observations separately.

## Out of Scope

- Resuming, continuing, fine-tuning, or initializing training from a Saved Transformer Model.
- Treating Saved Transformer Models as checkpoints, cache shortcuts, or training-skip artifacts.
- Changing the fixed Transformer Training Corpus, model architecture, training algorithm, optimizer, learning rate, Logical Training Shards, worker protocol, shared-memory design, or numerical compatibility rules.
- Loading Saved Embedding Models or Saved XOR Weight Snapshots.
- Reimplementing `POST /train-embed`.
- A continuing conversational session, remembered chat history, server-side session, cookie, retained model selection, or conversation state.
- A model registry, manifest, database, rollback system, download feature, deletion feature, or model-management page.
- Supporting old TypeScript, partial, alternate, or migrated Saved Transformer Model formats.
- Guessing, repairing, rewriting, deleting, or filling missing values in malformed model files.
- Automatically training when no valid saved model exists.
- Falling back to another model after a named-model failure.
- Caching a loaded model across requests.
- An application-level Saved Transformer Model file-size limit.
- Token-by-token saved-model display.
- Creating training workers, pipes, or shared memory for saved-model generation.
- Showing a worker-process label for `POST /load-transformer`.
- Adding a Train/Load selector, second input box, new route page, or layout redesign.
- Supporting `|` inside a starting prompt or adding an escaping grammar.
- Accepting arbitrary browser-supplied filesystem paths.
- Following symbolic links or Windows junctions.
- Multiple simultaneous Transformer jobs, a waiting queue, a machine-wide lock, or cross-process coordination.
- GPU, CUDA, PyTorch, TensorFlow, JAX, scikit-learn, hosted models, LangChain, or LangGraph.
- New production dependencies unless later implementation evidence proves one unavoidable and the specification is revised.
- Implementation tickets, production code, commits, or code-review output in this specification step.

## Notes

- **Dependency:** Phase 6 depends on the completed Phase 5 canonical Transformer parameter layout, forward/generation mathematics, Saved Transformer Model builder and persistence format, request-scoped worker count, shared SSE utilities, and route-level request slot.
- **Dependency:** The frontend requires the current TypeScript/Vite application and its existing SSE reader, Transformer hook, result component, and configured Vitest/type-check tooling.
- **Architectural constraint:** ADR 0002 remains the authority for fresh-weight Transformer Training Runs, multiprocessing, shared memory, cleanup, and persistence-before-training-`done`. ADR 0003 is the authority for stateless Saved Transformer Generation Runs.
- **Risk — Unsafe file selection:** User text could attempt to escape `.data` or exploit Windows case folding, symbolic links, or junctions. Safeguard with plain-name grammar, exact directory enumeration, link/junction rejection, resolved containment checks, and exact-case equality.
- **Risk — Partial validation:** A malformed parameter array could fail only after `loaded`. Safeguard by validating exact structure, canonical lengths, finite values, BPE artifacts, filename agreement, and parameter materialization before `loaded`.
- **Risk — Latest unusable model:** A damaged newest artifact could block latest selection. Safeguard by validating newest-to-oldest and skipping invalid candidates only for `modelFile: null`.
- **Risk — Cross-request stale state:** A cached model could ignore disk changes or leak mutable state. Safeguard by reading once per request, never caching, and discarding all loaded containers after completion.
- **Risk — Event-loop blocking:** Large JSON parsing or CPU-bound generation could freeze unrelated requests. Safeguard with same-process offloading and bounded token-level cooperative observation; do not claim thread offloading supplies multi-core acceleration.
- **Risk — Slot leakage:** Disconnects, deadlines, or errors could retain the only Transformer slot. Safeguard with one final lifecycle path that drains active helper work, discards request state, and releases the slot last.
- **Risk — Presentation contamination:** The worker-process label could alter generated text, fixtures, or persistence. Safeguard by adding it only during formatting of the first public training epoch payload.
- **Risk — Frontend parser regression:** `File:` could still become training defaults. Safeguard by testing command classification before numeric parsing and preserving exact numeric body construction.
- **Evidence limitation:** The repository was inspected through the latest complete supplied source exports rather than a live Git checkout. Implementation must re-inspect the live files, preserve user changes, and treat the latest source as authoritative.
- **Evidence limitation:** No Phase 6 code was written and no pytest, Ruff, mypy, frontend type-check, Vitest, browser, server, symlink, or junction command was executed while producing this specification.
- **Authoritative technical evidence:** FastAPI's streaming response supports yielded chunks without automatic JSON conversion; Starlette exposes asynchronous request-disconnection observation; Python 3.12 provides path resolution, symbolic-link checks, and Windows junction checks; Pydantic supports strict/finite fields, aliases, and extra-field policy; Vitest supports controlled function mocks for deterministic pure-function tests.
- **Publication target:** Replace the project's local root `SPEC.md` with this document. No working issue-tracker publication configuration or permission was supplied.

## Source Material Consulted

- `TO_SPEC_PROMPT.md`
- `GRILL_WITH_DOCS_RESULT.md`
- Current Phase 5 `SPEC.md`
- Updated `CONTEXT.md`
- ADR 0002 — Stabilize Python Transformer Training and Process Lifecycle
- ADR 0003 — Load Saved Transformer Models for Stateless Generation
- Latest `py_llm_pipeline_explorer_file_structure.md`
- Latest `llm_works_file_structure.md`
- Backend project configuration and dependency declarations
- Current FastAPI application, request schemas, shared SSE transport, Transformer route, model layout, generation, worker-group, persistence, and focused tests
- Current TypeScript Transformer hook, generic SSE reader, result component, and frontend package configuration
- Official FastAPI documentation for streamed responses
- Official Starlette documentation for request disconnection and in-process test clients
- Official Python 3.12 documentation for `pathlib.Path.resolve()`, `Path.is_symlink()`, and `Path.is_junction()`
- Official Pydantic documentation for strict validation, finite values, aliases, and extra-field configuration
- Official Vitest documentation for deterministic function mocking

## Recommended Next Step

Run `to-tickets-prompt` using this specification, the confirmed Phase 6 handoff, the updated glossary, ADR 0002, ADR 0003, the latest complete Python Backend source export, and the latest TypeScript/Vite frontend reference as inputs.
