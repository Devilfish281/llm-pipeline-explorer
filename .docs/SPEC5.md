---
workflow: engineering-prompt-chain
document_type: specification
prompt_name: to-spec-prompt
status: ready-for-agent
triage_label: ready-for-agent
version: 4
source_document: GRILL_WITH_DOCS_RESULT.md
recommended_next_prompt: to-tickets-prompt
---

# Specification: Deliver Deterministic Phase 4 Word2Vec Embeddings Through the Python Backend

## Problem

Learners can already use the Simple Chat, BPE Tokenizer, and XOR Neural Network Learning Demos through the Python Backend, but the Train Embeddings Learning Demo still has no working Python implementation. The reusable Word2Vec module and its FastAPI route are empty, the application does not register `POST /train-embed`, and the request schema does not yet express the frontend’s embedding-training contract.

The unchanged TypeScript/Vite frontend expects more than a generally valid Word2Vec implementation. It expects a deterministic Embedding Training Run built from the fixed Embedding Training Corpus, reference-compatible BPE preprocessing, a precise Mulberry32 random sequence, an exact online Skip-gram update order, and a stable Embedding Event Stream:

```text
init → epoch × approximately 50 → done
```

It also expects exact rounded vectors, Nearest Neighbor rankings, Similarity Pairs, predefined Vector Analogies, warnings, and a complete Saved Embedding Model written before completion. A Python implementation that changes preprocessing, Vocabulary order, random-call order, pair shuffling, gradient-update order, rounding, tie handling, event sequencing, or persistence behavior can change the educational result even when the replacement algorithm is mathematically reasonable.

The project owner also needs Phase 4 to remain focused and inspectable. Expanding this work into Transformer training, multiprocessing, a third-party Word2Vec library, model loading, job management, or frontend redesign would delay the first working embeddings demonstration and weaken the from-scratch learning purpose.

## Solution

Implement one focused Phase Migration that delivers the Word2Vec Embeddings Demo through the Python Backend while leaving the TypeScript/Vite frontend unchanged.

For each valid `POST /train-embed` request, the Python Backend will:

1. Validate the existing request fields, aliases, defaults, strict integer types, and inclusive bounds.
2. Use the fixed Embedding Training Corpus rather than training from submitted Query Words.
3. Reproduce the reference BPE preprocessing, Merge Table, Vocabulary order, token frequencies, and ordered Training Pairs.
4. Create one request-owned Mulberry32 generator with seed `42`.
5. Initialize separate NumPy `float64` input and output weight matrices in the exact reference random-call order.
6. Train Skip-gram with negative sampling using the confirmed inclusive epoch schedule, Fisher–Yates shuffle, learning-rate schedule, positive-before-negative order, and immediate coordinate-level updates.
7. Emit one `init` event and approximately fifty rounded `epoch` events through the shared SSE transport.
8. Return control to the async route between progress boundaries and stop cooperatively after a browser disconnect.
9. Construct the exact frontend-facing Embedding Result, preserving duplicate Query Word positions, warnings, rankings, similarities, and analogy behavior.
10. Build a complete Saved Embedding Model containing every ordered Vocabulary Token and its public Word Embedding.
11. Serialize and atomically replace `backend/.data/embedding-weights.json`.
12. Emit exactly one `done` event only after persistence succeeds.
13. Terminate quietly without `done` after disconnection, cancellation, numerical failure, result-construction failure, or persistence failure.
14. Preserve the working Health, Simple Chat, BPE Tokenizer, and XOR Neural Network behavior.

Reusable corpus preprocessing, deterministic randomness, numerical training, and result construction will be owned by the Word2Vec module. The Train Embed route will own request orchestration, bounded same-process thread offloading, disconnect checks, SSE event construction, presentation delays, persistence, logging, and completion ordering. Shared request and SSE infrastructure will be reused without changing existing contracts.

The implementation will be verified through the approved three-layer test strategy: the stable public Word2Vec interface, the in-process FastAPI boundary, and the atomic persistence boundary. Exact assertions will protect discrete and rounded public behavior; explicit tight tolerances will protect unrounded `float64` calculations without claiming universal bit-for-bit equality for every transcendental intermediate across JavaScript and Python.

## User Stories

1. As a learner, I want to start the Train Embeddings Learning Demo with one or more Query Words, so that I can explore relationships learned from the fixed corpus.
2. As a learner, I want the demo to show Vocabulary size, sentence count, embedding dimensions, window size, and Training Pair count before progress begins, so that I understand the training setup.
3. As a learner, I want to see loss updates throughout training, so that I can observe whether the embedding model is learning.
4. As a learner, I want progress to include epoch zero and the requested final epoch, so that the displayed curve has clear beginning and ending observations.
5. As a learner, I want approximately fifty progress updates rather than one event per epoch, so that the browser remains usable during training.
6. As a learner, I want each streamed loss rounded to six decimal places, so that progress is readable and reference compatible.
7. As a learner, I want a submitted recognized Query Word to produce its learned Word Embedding, so that I can inspect its numerical representation.
8. As a learner, I want public vectors rounded to six decimal places, so that the displayed and saved model remain stable and readable.
9. As a learner, I want each recognized Query Word to show up to five Nearest Neighbors, so that I can see which Vocabulary Tokens occupy similar vector space.
10. As a learner, I want each neighbor score rounded to two decimal places, so that comparisons are easy to read.
11. As a learner, I want pairwise similarities for every pair of recognized Query Word positions, so that I can compare submitted words directly.
12. As a learner, I want the seven predefined Vector Analogies evaluated in their established order, so that I can observe vector arithmetic consistently.
13. As a learner, I want each analogy result to exclude its three source tokens, so that the result represents a new candidate.
14. As a learner, I want warnings when a Query Word is not one Vocabulary Token, so that I understand why it has no selected embedding result.
15. As a learner, I want warning text to preserve the originally submitted Query Word, so that I can identify the exact input that was not recognized.
16. As a learner, I want the warning to show the Query Word’s BPE split, so that the demo connects embeddings back to tokenization.
17. As a learner, I want uppercase Query Words to be looked up case-insensitively, so that capitalization alone does not hide a recognized Vocabulary Token.
18. As a learner, I want leading and trailing spaces preserved by the backend, so that the backend does not silently reinterpret the request.
19. As a learner, I want a whitespace-only non-empty Query Word to remain structurally valid and produce a warning when unrecognized, so that validation and Vocabulary recognition remain separate concepts.
20. As a learner, I want duplicate Query Words preserved in their submitted positions, so that repeated comparisons are not silently deduplicated.
21. As a learner, I want duplicate recognized Query Words to produce repeated embeddings and neighbor groups, so that positional request behavior remains visible.
22. As a learner, I want duplicate recognized positions to participate in Similarity Pairs, including same-token comparisons, so that results reflect the submitted list exactly.
23. As a learner, I want an all-unrecognized request to complete with warnings and empty selected embedding collections, so that recognition failure does not erase a successful training run.
24. As a learner, I want predefined analogies to remain available even when all submitted Query Words are unrecognized, so that analogies remain a property of the trained model.
25. As a learner, I want Query Words to select displayed results without changing the Embedding Training Corpus, so that identical hyperparameters always train the same model.
26. As the TypeScript/Vite frontend, I want to keep sending `POST /train-embed`, so that no frontend route change is required.
27. As the TypeScript/Vite frontend, I want `words` to remain a required array containing one through ten non-empty strings, so that the request contract stays compatible.
28. As the TypeScript/Vite frontend, I want `epochs` to default to `10000`, so that existing requests without an explicit value retain their behavior.
29. As the TypeScript/Vite frontend, I want `dimensions` to default to `32`, so that the displayed model shape remains unchanged.
30. As the TypeScript/Vite frontend, I want `windowSize` to default to `2`, so that context generation remains unchanged.
31. As the TypeScript/Vite frontend, I want `negativeSamples` to default to `5`, so that negative-sampling behavior remains unchanged.
32. As the TypeScript/Vite frontend, I want the camelCase fields `windowSize` and `negativeSamples` preserved, so that no request mapping change is required.
33. As the TypeScript/Vite frontend, I want strict integer validation for all numeric request fields, so that numeric strings, booleans, and fractional numbers are rejected.
34. As the TypeScript/Vite frontend, I want invalid request bodies to return FastAPI/Pydantic HTTP `422` before streaming begins, so that malformed requests use a stable HTTP failure.
35. As the TypeScript/Vite frontend, I want unknown extra request fields ignored, so that the existing permissive object behavior remains compatible.
36. As the TypeScript/Vite frontend, I want valid requests to return `text/event-stream`, so that the existing SSE reader continues to work.
37. As the TypeScript/Vite frontend, I want the existing cache-prevention and proxy-buffering headers preserved, so that progress can be delivered incrementally.
38. As the TypeScript/Vite frontend, I want the successful event order to be `init → epoch × N → done`, so that the existing event discriminator remains correct.
39. As the TypeScript/Vite frontend, I want the `init` payload to contain exactly `vocabSize`, `sentenceCount`, `embeddingDim`, `windowSize`, and `totalPairs`, so that initialization rendering remains stable.
40. As the TypeScript/Vite frontend, I want each `epoch` payload to contain exactly `epoch` and `loss`, so that progress rendering remains stable.
41. As the TypeScript/Vite frontend, I want the `done` payload to contain exactly `embeddings`, `neighbors`, `similarities`, `analogies`, and `warnings`, so that final rendering remains stable.
42. As the TypeScript/Vite frontend, I want no internal output-weight matrix or persistence data in `done`, so that the public result remains focused.
43. As the TypeScript/Vite frontend, I want approximately 20 milliseconds of presentation delay after each `epoch` event and no delay after `done`, so that progress animation remains compatible.
44. As the TypeScript/Vite frontend, I want exactly one `done` event and no events afterward, so that a completed stream has one unambiguous endpoint.
45. As a project developer, I want the fixed corpus and its sentence order preserved exactly, so that preprocessing and training fixtures remain deterministic.
46. As a project developer, I want exactly 500 BPE Merges learned from the lowercased joined corpus, so that the Word2Vec Vocabulary matches the reference.
47. As a project developer, I want BPE Merges applied within Pre-token boundaries and in learned order, so that tokenization remains reference compatible.
48. As a project developer, I want Vocabulary frequencies counted in first-encounter order and sorted stably by descending frequency, so that token indices remain deterministic.
49. As a project developer, I want Training Pairs generated by sentence order, token order, and context position, so that the initial pair sequence is deterministic.
50. As a project developer, I want shared corpus-derived preprocessing to be immutable, so that one request cannot change another request’s input data.
51. As a project developer, I want each Embedding Training Run to own its mutable Training Pair order, PRNG, weights, epochs, gradients, and losses, so that concurrent requests remain isolated.
52. As a project developer, I want Mulberry32 seed `42` and exact 32-bit behavior preserved, so that valid identical requests reproduce the reference random sequence.
53. As a project developer, I want random values consumed in the reference order, so that initialization, shuffling, and negative sampling do not drift.
54. As a project developer, I want separate NumPy `float64` input and output matrices, so that the numerical representation matches the reference `Float64Array` behavior.
55. As a project developer, I want the negative-sampling distribution based on frequency raised to `0.75`, so that negative candidates follow the established probability model.
56. As a project developer, I want each epoch’s mutable Training Pairs shuffled by Mulberry32-driven Fisher–Yates, so that pair processing matches the reference.
57. As a project developer, I want the learning rate to decline linearly from `0.025` to `0.001`, so that the training schedule remains compatible.
58. As a project developer, I want epochs processed inclusively from zero through the requested value, so that training and reporting boundaries match the frontend contract.
59. As a project developer, I want every positive Training Pair updated before its negative samples are drawn, so that the random and numerical sequences remain compatible.
60. As a project developer, I want exactly the requested number of negative candidates drawn, so that a true-context collision is skipped without a replacement draw.
61. As a project developer, I want coordinate updates applied immediately and in order, so that training remains online rather than batched.
62. As a project developer, I want output coordinates updated with the saved pre-update input coordinate, so that the gradient operation matches the reference.
63. As a project developer, I want sigmoid clipping and the `1e-10` loss safeguard preserved, so that extreme values and logarithms behave consistently.
64. As a project developer, I want epoch loss divided by the count of positive Training Pairs, so that reported loss retains its established scale.
65. As a project developer, I want public vectors derived from `wIn` only, so that `wOut` remains an internal training matrix.
66. As a project developer, I want neighbor and similarity calculations to use public six-decimal vectors, so that displayed ranking behavior matches the reference.
67. As a project developer, I want neighbor ranking based on two-decimal scores with stable Vocabulary-order ties, so that equal displayed scores have deterministic order.
68. As a project developer, I want analogy source arithmetic to use raw `wIn` rows and candidate comparisons to use public six-decimal vectors, so that the confirmed mixed-precision behavior is preserved.
69. As a project developer, I want exact comparisons for random sequences, ordering, rounded payloads, warnings, and serialized artifacts, so that public drift is caught immediately.
70. As a project developer, I want tight tolerance-based comparisons for unrounded `float64` values, so that cross-runtime floating-point differences do not hide formula errors or create false failures.
71. As a project developer, I want deterministic expected values calculated independently of production helpers, so that tests do not merely repeat implementation mistakes.
72. As a project developer, I want long CPU-bound work advanced in bounded same-process worker-thread intervals, so that the async event loop regains control between progress reports.
73. As a project developer, I want disconnect behavior exercised through a narrow test seam, so that cancellation can be verified without a fragile real network interruption.
74. As a project developer, I want production presentation waits replaced in tests, so that the suite remains fast and deterministic.
75. As a project developer, I want persistence tests isolated under temporary directories, so that tests never overwrite the real Saved Embedding Model.
76. As an operator, I want a browser disconnect to stop additional training intervals, so that abandoned requests do not run to completion unnecessarily.
77. As an operator, I want a disconnected run to avoid persistence and `done`, so that an incomplete run cannot appear successful.
78. As an operator, I want unexpected post-stream failures logged internally and omitted from SSE data, so that stack traces, paths, and numerical state are not exposed.
79. As an operator, I want cancellation control flow to propagate after cleanup, so that request cancellation is not converted into an ordinary training failure.
80. As an operator, I want non-finite numerical values to prevent successful completion and persistence, so that invalid JSON or corrupted vectors are never published.
81. As an operator, I want the complete Saved Embedding Model written before `done`, so that completion means the inspectable artifact is available.
82. As an operator, I want the saved model to include every ordered Vocabulary Token rather than only submitted Query Words, so that it represents the complete trained model.
83. As an operator, I want saved JSON to use two-space indentation and exactly one trailing newline, so that the artifact is stable and inspectable.
84. As an operator, I want the saved-model path resolved from the backend project rather than the shell’s current directory, so that startup location does not change the destination.
85. As an operator, I want each save written completely to a unique same-directory temporary file before atomic replacement, so that partial JSON is never exposed as the destination.
86. As an operator, I want serialization, write, and replacement failures to preserve the prior valid model, so that a failed run does not destroy the latest successful artifact.
87. As an operator, I want temporary files removed after failed persistence when cleanup succeeds, so that failures do not leave avoidable debris.
88. As an operator, I want concurrent successful runs to leave one complete model and use last-successful-finisher-wins behavior, so that writes cannot interleave into corrupted JSON.
89. As an operator, I want no application-level semaphore, global training lock, queue, timeout, or rate limiter added in Phase 4, so that the confirmed scope remains unchanged.
90. As an operator, I want `GET /health`, `POST /simple-chat`, `POST /bpe-tokenize`, and `POST /neural-net` to retain their existing behavior, so that Phase 4 does not regress completed Learning Demos.
91. As a project maintainer, I want the existing Poetry, pytest, Ruff, and strict mypy workflow to remain the validation path, so that Phase 4 fits the established backend toolchain.
92. As a project maintainer, I want no new runtime or development dependency unless current project inspection proves one necessary, so that the focused implementation remains small.
93. As a project maintainer, I want Transformer training and process-based optimization deferred, so that Phase 4 delivers one complete Word2Vec vertical slice first.

## Implementation Decisions

1. **Confirmed — Python Backend authority:** FastAPI and Python remain the only server-side runtime. TypeScript files define reference behavior and have no runtime role in the Python Backend.

2. **Confirmed — Focused Phase Migration:** Phase 4 implements only the Word2Vec Embeddings Demo required by `POST /train-embed`. Phase 5 Transformer behavior and process infrastructure remain separate work.

3. **Confirmed — Responsibility boundary:** The Word2Vec module owns the fixed corpus, BPE-derived preprocessing, Vocabulary, Training Pairs, deterministic randomness, numerical state, training progression, public vector construction, Nearest Neighbors, Similarity Pairs, Vector Analogies, warnings, and Saved Embedding Model conversion.

4. **Confirmed — Route responsibility:** The Train Embed route owns request orchestration, bounded `asyncio.to_thread()` calls, disconnect checks, SSE framing, presentation delays, persistence, failure logging, and the persistence-before-`done` sequence.

5. **Confirmed — Schema ownership:** The shared schema module will define a dedicated Train Embed request model while preserving the existing Chat and Neural Network request models.

6. **Confirmed — Router registration:** The Train Embed router will be included in the FastAPI application without removing or changing the Health, Simple Chat, BPE Tokenizer, or Neural Network routers.

7. **Confirmed — Request fields:** The public request contains only `words`, `epochs`, `dimensions`, `windowSize`, and `negativeSamples`.

8. **Confirmed — Query Word validation:** `words` is required and must contain one through ten strings, each with at least one character.

9. **Confirmed — Epoch validation:** `epochs` is a strict integer from `10` through `10000`, inclusive, and defaults to `10000`.

10. **Confirmed — Dimension validation:** `dimensions` is a strict integer from `4` through `64`, inclusive, and defaults to `32`.

11. **Confirmed — Window validation:** `windowSize` is a strict integer from `1` through `5`, inclusive, and defaults to `2`.

12. **Confirmed — Negative-sample validation:** `negativeSamples` is a strict integer from `1` through `10`, inclusive, and defaults to `5`.

13. **Confirmed — CamelCase aliases:** `windowSize` and `negativeSamples` remain the exact frontend-facing JSON names.

14. **Confirmed — Strict numerical typing:** Numeric strings, booleans, and fractional numbers are invalid for the four integer hyperparameters and produce HTTP `422`.

15. **Confirmed — Extra fields:** Unknown additional object fields remain ignored.

16. **Confirmed — No new HTTP controls:** Do not add request fields for corpus text, seed, learning rate, optimizer, sigmoid clipping, report count, save path, persistence toggle, or concurrency control.

17. **Confirmed — Normalization boundary:** The backend does not trim, split, deduplicate, filter, or remove submitted Query Words. It lowercases each complete entry only for Vocabulary lookup and warning analysis.

18. **Confirmed — Original warning input:** Warning text preserves the Query Word exactly as submitted.

19. **Confirmed — Duplicate positions:** Duplicate Query Words remain separate positions and count toward the ten-entry limit.

20. **Confirmed — Unknown Query Words:** An unknown or multi-token Query Word does not fail training. It is omitted from embeddings, neighbors, and similarities and contributes one warning.

21. **Confirmed — Exact warning contract:** The warning format is:

    ```text
    "<submitted word>" is not a single BPE token — it splits into [<comma-separated tokens>]
    ```

22. **Confirmed — All-unrecognized result:** A successful request may complete with empty `embeddings`, `neighbors`, and `similarities`, plus warnings and the independently evaluated analogies.

23. **Confirmed — Fixed Embedding Training Corpus:** Training always uses the curated reference corpus in its exact established sentence order. Query Words never change the corpus.

24. **Confirmed — Corpus preprocessing:** Lowercase and join the complete corpus for BPE training.

25. **Confirmed — Merge count:** Learn exactly `500` BPE Merges for the corpus-derived Merge Table.

26. **Confirmed — BPE reuse:** Reuse the existing educational BPE implementation where its behavior matches the TypeScript Reference Implementation; do not introduce an opaque tokenizer.

27. **Confirmed — Pre-token boundaries:** BPE learning and application preserve Pre-token boundaries.

28. **Confirmed — Sentence tokenization:** Apply the ordered corpus-derived Merge Table to every lowercased corpus sentence.

29. **Confirmed — Vocabulary frequency order:** Count token frequencies in first-encounter order.

30. **Confirmed — Stable Vocabulary sort:** Sort Vocabulary Tokens by descending frequency while preserving first-encounter order for equal frequencies.

31. **Confirmed — Token indices:** Ordered Vocabulary position defines each token index.

32. **Confirmed — Training Pair order:** Generate ordered target-context pairs by sentence order, token position, and context position within the requested window.

33. **Confirmed — Immutable shared preprocessing:** Corpus sentences, Merge Table, ordered Vocabulary, frequencies, token indices, and reusable tokenized corpus data may be shared only as immutable data.

34. **Confirmed — Request-owned mutable state:** Each Embedding Training Run owns its Mulberry32 state, mutable pair order, `wIn`, `wOut`, epoch position, gradients, accumulated loss, and result state.

35. **Confirmed — PRNG:** Reproduce Mulberry32 using seed `42`, JavaScript-compatible 32-bit integer behavior, and the exact reference random-call sequence.

36. **Confirmed — No NumPy random generator:** Do not substitute `numpy.random.default_rng()` or another PRNG for Mulberry32.

37. **Confirmed — Numerical dtype:** Use separate NumPy `float64` input and output weight matrices.

38. **Confirmed — Matrix dimensions:** Both `wIn` and `wOut` have one row per ordered Vocabulary Token and one column per requested embedding dimension.

39. **Confirmed — Initialization scale:** Calculate:

    ```text
    scale = 0.5 / dimensions
    ```

40. **Confirmed — Initialization formula:** Each weight is:

    ```text
    (random - 0.5) × scale
    ```

41. **Confirmed — Initialization call order:** Consume one random value for the corresponding `wIn` coordinate and then one for the corresponding `wOut` coordinate while traversing the flattened coordinates in reference order.

42. **Confirmed — Negative-sampling distribution:** Raise each Vocabulary frequency to `0.75`, normalize the values, and build the reference cumulative distribution.

43. **Confirmed — Negative selection:** Use the reference cumulative search with one Mulberry32 draw per candidate.

44. **Confirmed — Inclusive epochs:** Process epochs from `0` through the requested `epochs` value, inclusive.

45. **Confirmed — Pair shuffle:** Shuffle the mutable Training Pair sequence in place before each epoch with Mulberry32-driven Fisher–Yates.

46. **Confirmed — Learning-rate schedule:** Calculate:

    ```text
    learning_rate = 0.025 - (0.025 - 0.001) × (epoch / epochs)
    ```

47. **Confirmed — Online training:** Apply each Training Pair’s changes immediately. Do not accumulate or apply a batch gradient.

48. **Confirmed — Positive-first order:** Apply the positive target-context update before drawing any negative candidates for that pair.

49. **Confirmed — Negative draw count:** Draw exactly `negativeSamples` candidates for each positive pair.

50. **Confirmed — Context collision:** When a negative candidate equals the true context, skip its update without drawing a replacement.

51. **Confirmed — Sigmoid clipping:** Return `1` for an input greater than `6`, return `0` for an input less than `-6`, and otherwise use the logistic sigmoid.

52. **Confirmed — Positive loss:** Accumulate `-log(score + 1e-10)` for each positive sample.

53. **Confirmed — Negative loss:** Accumulate `-log(1 - negative_score + 1e-10)` for each effective negative sample.

54. **Confirmed — Coordinate update order:** Traverse embedding coordinates in order and update them immediately.

55. **Confirmed — Saved pre-update coordinate:** Use the saved pre-update `wIn` coordinate when updating the corresponding `wOut` coordinate for both positive and negative samples.

56. **Confirmed — Loss denominator:** Divide total epoch loss by the number of positive Training Pairs, not by the number of all positive and negative examples.

57. **Confirmed — Reporting step:** Calculate:

    ```text
    report_step = max(1, floor(epochs / 50))
    ```

58. **Confirmed — Reporting condition:** Emit an Embedding Epoch Update when the epoch is divisible by `report_step` or is the requested final epoch.

59. **Confirmed — Streamed loss rounding:** Round public loss to six decimal places using TypeScript-compatible decimal rounding.

60. **Confirmed — Event flow:** A successful Embedding Event Stream is:

    ```text
    init → epoch × approximately 50 → done
    ```

61. **Confirmed — Init payload:** The `init` payload contains exactly `vocabSize`, `sentenceCount`, `embeddingDim`, `windowSize`, and `totalPairs`.

62. **Confirmed — Epoch payload:** Each `epoch` payload contains exactly `epoch` and `loss`.

63. **Confirmed — Production delay:** Request an approximately `0.02`-second presentation wait after every `epoch` event and no presentation wait after `init` or `done`.

64. **Confirmed — Shared SSE transport:** Reuse the established SSE formatter and response factory, preserving `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`.

65. **Confirmed — Bounded thread offloading:** Advance one public training interval at a time, or use an equivalent bounded design, through same-process thread offloading so the async route regains control between progress boundaries.

66. **Confirmed — No process infrastructure:** Do not add multiprocessing, process pools, worker processes, shared memory, or an external task queue.

67. **Confirmed — Disconnect check:** Check client connection state between bounded training intervals.

68. **Confirmed — Disconnect result:** After disconnect, start no additional interval, persist no model, emit no `done` or `error` event, and release request-owned state.

69. **Confirmed — Cancellation semantics:** Do not catch `BaseException` or suppress `asyncio.CancelledError` as an ordinary training failure. Permit cancellation to propagate after necessary cleanup.

70. **Confirmed — Public vector source:** A Word Embedding is derived only from the selected `wIn` row.

71. **Confirmed — Public vector rounding:** Round every public vector coordinate to six decimal places using TypeScript-compatible rounding.

72. **Confirmed — Embedding result order:** Build selected embeddings in recognized Query Word positional order, including duplicates.

73. **Confirmed — Neighbor source:** Calculate Nearest Neighbors from public six-decimal vectors.

74. **Confirmed — Neighbor score rounding:** Round cosine similarity to two decimal places before ranking.

75. **Confirmed — Neighbor tie behavior:** Sort by descending rounded score and preserve ordered Vocabulary position for ties.

76. **Confirmed — Neighbor exclusions and limit:** Exclude the Query Word’s own Vocabulary index and return at most five candidates.

77. **Confirmed — Similarity positions:** Build a Similarity Pair for every pair of recognized Query Word positions, preserving duplicates.

78. **Confirmed — Similarity source and rounding:** Calculate Similarity Pair scores from public six-decimal vectors and round to two decimal places.

79. **Confirmed — Analogy definitions:** Evaluate these seven analogies in this order:

    ```text
    king - man + woman
    queen - woman + man
    prince - boy + girl
    kitten - cat + dog
    puppy - dog + cat
    he - man + woman
    his - man + woman
    ```

80. **Confirmed — Analogy source precision:** Construct each analogy query vector from raw `wIn` rows using `a - b + c`.

81. **Confirmed — Analogy candidate precision:** Compare the raw-source query vector with candidate public six-decimal `wIn` vectors.

82. **Confirmed — Analogy exclusions:** Exclude the three source token indices from candidate selection.

83. **Confirmed — Analogy tie behavior:** Keep the first candidate encountered in ordered Vocabulary traversal when scores tie.

84. **Confirmed — Analogy score rounding:** Round the selected analogy score to two decimal places.

85. **Confirmed — Non-finite safeguard:** A successful Embedding Training Run must not emit or serialize `NaN`, positive infinity, or negative infinity.

86. **Confirmed — Done payload:** The `done` payload contains exactly `embeddings`, `neighbors`, `similarities`, `analogies`, and `warnings`.

87. **Confirmed — Internal weights excluded:** Do not include `wIn`, `wOut`, the Merge Table, persistence paths, or the complete Saved Embedding Model in `done`.

88. **Confirmed — Saved model destination:** Persist the latest successful model to:

    ```text
    backend/.data/embedding-weights.json
    ```

    Resolve the backend root independently of the shell’s current working directory.

89. **Confirmed — Saved model fields:** The Saved Embedding Model contains exactly `type`, `dimensions`, `vocab`, `merges`, and `embeddings`.

90. **Confirmed — Saved type:** The saved `type` value is exactly `word2vec-skipgram`.

91. **Confirmed — Saved Vocabulary:** The `vocab` array contains every Vocabulary Token in ordered index order.

92. **Confirmed — Saved Merge Table:** The `merges` array contains every learned BPE Merge as its pair and merged token, preserving order.

93. **Confirmed — Saved embeddings:** The `embeddings` mapping contains the public six-decimal `wIn` vector for every ordered Vocabulary Token.

94. **Confirmed — JSON formatting:** Serialize with two-space indentation, exactly one final newline, and non-finite values disallowed.

95. **Confirmed — Serialize first:** Complete serialization before touching the destination.

96. **Confirmed — Atomic persistence:** Create and close a unique temporary file in the destination directory, write the complete document, and atomically replace the destination only after the write succeeds.

97. **Confirmed — Persistence before completion:** Persist successfully before yielding `done`.

98. **Confirmed — Failure preservation:** Serialization, write, replacement, and result-construction failures do not replace the prior valid Saved Embedding Model.

99. **Confirmed — Temporary cleanup:** Remove a temporary file after failure when cleanup succeeds.

100. **Confirmed — Post-stream failure behavior:** Log ordinary unexpected failures server-side, terminate without `done`, do not invent an SSE `error` event, and do not expose exception text, tracebacks, paths, weights, or numerical state.

101. **Confirmed — Concurrent persistence:** Concurrent successful requests use independent temporary files; each replacement is complete; the last successful finisher becomes the saved model.

102. **Confirmed — No model loading:** The Saved Embedding Model is an inspectable latest-model artifact only. Do not load, cache, resume, fine-tune, or reuse it for later requests.

103. **Confirmed — No application concurrency policy:** Do not add a semaphore, global training lock, queue, timeout, quota, or rate limiter.

104. **Confirmed — Existing route compatibility:** Preserve current observable behavior for Health, Simple Chat, BPE Tokenizer, and Neural Network endpoints.

105. **Confirmed — Existing dependency set:** The supplied backend already declares FastAPI, Pydantic, NumPy, pytest, pytest-asyncio, an HTTP testing dependency, Ruff, and mypy. Inspect the current project before changing dependencies; no new dependency is presently required.

106. **Confirmed — ADR 0001:** Preserve Deterministic Word2Vec Compatibility. Exact rounded public behavior, random-call and operation order, tolerance-aware hidden numerical verification, and persistence-before-`done` take priority over library substitution or optimization.

107. **Assumption — Public Word2Vec API shape:** The reusable module may expose typed dataclasses, immutable records, iterators, protocols, or state objects for preprocessing, initialization, bounded training, and result construction. Exact symbol names are an implementation choice as long as tests exercise stable public behavior without reaching into private loop variables.

108. **Assumption — Immutable preprocessing representation:** Shared corpus-derived data may be represented by tuples, frozen dataclasses, immutable mappings, read-only NumPy arrays, or an equivalent structure. The representation must prevent one request from mutating another request’s preprocessing.

109. **Assumption — Persistence helper decomposition:** The Train Embed route may reuse or parallel the proven Neural Network persistence pattern. Exact helper boundaries are an implementation choice provided the Saved Embedding Model contract, atomicity, failure preservation, and test seam remain stable.

110. **Assumption — Independent deterministic fixtures:** Exact Mulberry32 sequences, preprocessing fixtures, short-run weights, public payloads, and the smallest useful `rtol` and `atol` values will be selected during implementation from independently calculated or directly captured TypeScript Reference Implementation evidence. They must not be generated by calling the production Python helpers under test.

111. **Assumption — Test file organization:** The implementation may divide tests by preprocessing, numerical behavior, route behavior, result construction, and persistence, or use another clear organization. Test placement must not alter the approved three test seams.

## Testing Decisions

### Approved seam 1 — Public Word2Vec module

- **Approved test seam:** Exercise stable public preprocessing, deterministic randomness, bounded Training Run, and result-construction behavior through the Word2Vec module.
- **Why this seam:** It is the highest practical deterministic boundary for protecting the educational algorithm without HTTP, SSE, filesystem, or presentation concerns.
- **Modules exercised:** Word2Vec behavior plus the established BPE interface where corpus preprocessing depends on it.
- **Observable behavior covered:**
  - exact curated corpus and sentence order;
  - joined lowercase BPE training text;
  - 500-rule ordered Merge Table or independently verified compatibility fixtures that prove the complete table;
  - representative sentence tokenization and Pre-token boundaries;
  - token frequencies, ordered Vocabulary, indices, and stable frequency ties;
  - ordered Training Pairs for every supported window size;
  - immutable shared preprocessing;
  - Mulberry32 output sequence and 32-bit wraparound;
  - Fisher–Yates shuffle;
  - negative-sampling cumulative distribution and candidate selection;
  - `float64` matrix shapes, separation, and initialization call order;
  - sigmoid clipping;
  - one independently calculated positive update;
  - one independently calculated negative update;
  - pre-update input-coordinate use;
  - true-context skip without replacement draw;
  - inclusive epoch behavior and learning-rate schedule;
  - report boundaries and six-decimal loss;
  - a short complete deterministic Training Run;
  - public vectors from `wIn` only;
  - duplicate positions, lowercase lookup, warning text, and all-unrecognized behavior;
  - neighbors, similarities, tie handling, and limits;
  - exact analogy list, mixed-precision calculation, exclusions, ties, and rounding;
  - complete Saved Embedding Model conversion;
  - no non-finite successful output;
  - request-owned mutable-state isolation.

- **Required comparison policy:** Use exact equality for random outputs, integer structures, order, rounded public values, warnings, rankings, field sets, and JSON-compatible Saved Embedding Model contents. Use explicit tight `numpy.testing.assert_allclose()` tolerances or equivalent for unrounded `float64` matrices, scores, gradients, and losses.

- **Required independent evidence:** Expected values must be independently calculated, stored as fixed fixtures, or captured from the TypeScript Reference Implementation. A test must not compute its expectation by invoking the same production helper it is validating.

### Approved seam 2 — FastAPI `POST /train-embed`

- **Approved test seam:** Exercise the registered endpoint through FastAPI’s in-process `TestClient` or an equivalent in-process ASGI client.
- **Why this seam:** It verifies the complete Frontend Contract, route registration, schema behavior, shared SSE transport, bounded orchestration, disconnection, failure privacy, persistence ordering, and regressions without running a live server.
- **Modules exercised:** Application registration, shared schemas, shared SSE transport, Train Embed route orchestration, public Word2Vec Training Run boundary, and the route-owned persistence call.
- **Observable behavior covered:**
  - exact request fields and aliases;
  - defaults and inclusive bounds;
  - one-through-ten Query Word validation;
  - strict rejection of numeric strings, booleans, and fractions;
  - ignored unknown fields;
  - malformed requests fail before training;
  - HTTP `200` for valid requests;
  - `text/event-stream`, `Cache-Control: no-cache`, and `X-Accel-Buffering: no`;
  - valid JSON in every `data:` line;
  - exact `init → epoch × N → done` sequence;
  - exact field sets for every event;
  - epoch zero and final epoch;
  - approximately fifty updates;
  - six-decimal loss;
  - one `done` and no later event;
  - no complete model or internal matrices in `done`;
  - a requested `0.02`-second wait after each `epoch` and no wait after `init` or `done`;
  - bounded same-process worker-thread advancement;
  - persistence completes before `done`;
  - client disconnect prevents later intervals, persistence, and `done`;
  - ordinary training, result, serialization, and persistence failures terminate quietly and log internally;
  - no SSE `error` event and no leaked exception text, paths, stack traces, or numerical state;
  - cancellation is not swallowed as an ordinary exception;
  - independent sequential and concurrent request state;
  - existing route registration and behavior remain intact.

- **Required delay control:** Replace only the route’s referenced presentation-wait operation. Assert requested delay values and call counts rather than elapsed wall-clock time.

- **Required disconnect control:** Use a narrow fake or injected disconnect check that becomes true after a selected public progress boundary. Permit one already-started interval to finish, but prove no later interval starts.

- **Required failure controls:** Inject failures after one or more progress events and at result construction, serialization, write, and replacement boundaries. Verify no false completion and no client-visible internals.

### Approved seam 3 — Atomic persistence boundary

- **Approved test seam:** Exercise the route-owned Saved Embedding Model serialization and atomic replacement boundary using a pytest temporary directory.
- **Why this seam:** Persistence has correctness, cleanup, path-resolution, and concurrency requirements that should be proven independently without modifying the real backend artifact.
- **Modules exercised:** Saved-model conversion at the public Word2Vec boundary and route-owned filesystem operations.
- **Observable behavior covered:**
  - backend-root destination resolution independent of current working directory;
  - `.data` directory creation;
  - exact destination filename;
  - exact top-level key set and type value;
  - complete ordered Vocabulary;
  - complete ordered Merge Table;
  - complete embedding mapping for every Vocabulary Token;
  - two-space indentation and exactly one trailing newline;
  - non-finite serialization rejection;
  - serialization before temporary-file creation or destination replacement;
  - unique same-directory temporary files;
  - temporary file closed before replacement;
  - complete atomic replacement;
  - prior destination preserved after serialization, write, or replacement failure;
  - temporary cleanup after failure;
  - combined persistence and cleanup failures remain available to internal logging;
  - concurrent successful saves produce one complete destination;
  - controlled last-successful-finisher-wins behavior;
  - a failed concurrent request cannot damage a successful replacement.

### Required reporting and boundary cases

- Minimum `epochs=10`.
- Default `epochs=10000` through a controlled or lightweight seam that avoids real presentation waiting.
- A non-divisible epoch value proving the requested final epoch is emitted.
- Minimum and maximum dimensions: `4` and `64`.
- Minimum and maximum window sizes: `1` and `5`.
- Minimum and maximum negative samples: `1` and `10`.
- Query Word array lengths `1` and `10`.
- Rejection of `0` and `11` Query Words.
- Rejection of empty strings.
- Unknown extra fields remain ignored.
- Duplicate, uppercase, leading-space, trailing-space, whitespace-only, partially unrecognized, and all-unrecognized Query Word cases.

### Required regressions

- `GET /health` remains HTTP `200` with `{"status":"healthy"}`.
- `POST /simple-chat` retains its current validation, SSE headers, and `start → word × N → done` behavior.
- `POST /bpe-tokenize` retains its current validation, headers, and event contract.
- `POST /neural-net` retains its validation, bounded Training Run orchestration, persistence behavior, and `epoch × N → done` contract.
- Adding the Train Embed request model does not change the existing request models.
- Registering the Train Embed router does not remove or shadow existing routes.

### Required quality checks

Run from the backend project:

```powershell
poetry run pytest
poetry run ruff check .
poetry run mypy src
```

Report actual command output honestly. Do not claim success unless each command is executed successfully during implementation.

### Do not test

- Private helper names, local variables, dataclass choices, iterator implementation identity, or exact loop syntax.
- A particular thread-pool object when the confirmed bounded same-process behavior is satisfied.
- A particular temporary-file API when the atomic and cleanup contract is satisfied.
- Exact wall-clock delay duration.
- Browser component rendering, CSS, or Vite proxy behavior through backend unit tests.
- Universal bit-for-bit equality for every unrounded transcendental intermediate.
- TypeScript execution as part of the Python backend’s normal test suite.

### Known test limitations

- Automated Python tests do not prove the visual browser rendering or Vite development proxy. A manual two-server check is recommended after the automated suite passes.
- Cooperative disconnect handling may complete one already-started interval before cancellation is observed.
- Same-process thread offloading protects event-loop responsiveness and provides a cancellation boundary; it does not guarantee multi-core acceleration for Python-heavy loops.
- Full default training may be expensive, so most failure and numerical tests should use small deterministic configurations while retaining limited representative default-contract coverage.

## Out of Scope

- Any TypeScript/Vite frontend source, component, hook, route, styling, or proxy change.
- Phase 5 Transformer training.
- Node, Hono, or TypeScript backend runtime code.
- Python multiprocessing, process pools, worker processes, shared memory, or external task queues.
- Gensim, PyTorch, TensorFlow, JAX, scikit-learn, LangChain, LangGraph, hosted embeddings, or another implementation that hides the educational Word2Vec algorithm.
- NumPy-native or other replacement randomness instead of Mulberry32.
- Batched, matrix-wide, or operation-reordered gradient updates.
- A redesigned optimizer.
- Hierarchical softmax.
- Frequent-token subsampling.
- Gradient clipping.
- Early stopping.
- A configurable learning-rate schedule.
- A configurable seed.
- A configurable corpus.
- Loading the Saved Embedding Model.
- Resuming, fine-tuning, caching, or reusing a completed model.
- Model history, versioning, manifests, registries, checkpoints, or rollback.
- A frontend-visible model download feature.
- A general-purpose matrix or machine-learning framework.
- Application-level training queues, semaphores, global locks, timeouts, quotas, or rate limits.
- A new SSE `error` event.
- Exposing `wIn`, `wOut`, or the complete saved model in `done`.
- Exact wall-clock timing assertions.
- Forcefully terminating a Python worker thread.
- Cross-process file locking.
- Security hardening or production resource controls beyond the confirmed request bounds.
- Issue-tracker implementation tickets.
- Production implementation code.
- Commits.
- Code review.

## Notes

- **Dependency:** Python 3.12 or newer, Poetry, FastAPI, Pydantic, NumPy, pytest, pytest-asyncio, the current HTTP testing dependency, Ruff, and strict mypy are already declared in the supplied backend project. No dependency addition is currently required.
- **Dependency:** The unchanged TypeScript/Vite frontend depends on the exact Frontend Contract described in this specification.
- **Architectural constraint:** ADR 0001, Preserve Deterministic Word2Vec Compatibility, favors observable TypeScript compatibility over third-party libraries, NumPy-native randomness, batching, or operation-reordering optimizations.
- **Risk — Deterministic drift:** A harmless-looking refactor can change random consumption, shuffling, updates, rankings, or analogies.
  - **Safeguard:** Protect each deterministic layer with independent exact fixtures and short-run numerical checks.
- **Risk — Cross-runtime floating-point differences:** JavaScript and Python may differ slightly in unrounded transcendental calculations.
  - **Safeguard:** Require exact approved rounded public behavior and select explicit tight tolerances for unrounded `float64` values.
- **Risk — Mutable shared preprocessing:** A request could alter corpus-derived state used by later requests.
  - **Safeguard:** Use immutable shared structures and mutation-isolation tests.
- **Risk — Event-loop blocking:** Default or maximum training can perform substantial CPU work.
  - **Safeguard:** Advance bounded intervals through same-process thread offloading and return to the route between public updates.
- **Risk — Competing requests:** Several maximum requests can compete for CPU.
  - **Safeguard:** Preserve the confirmed bounded request contract in Phase 4 and defer rate limiting or scheduling to separate operational work.
- **Risk — Abandoned work:** A browser can disconnect during training.
  - **Safeguard:** Check connection state between intervals and do not persist or complete after disconnect.
- **Risk — Partial or corrupt model:** Direct writes can expose incomplete JSON.
  - **Safeguard:** Serialize completely, use unique same-directory temporary files, close them, and atomically replace.
- **Risk — False completion:** A numerically completed run may fail to save.
  - **Safeguard:** Make successful persistence a prerequisite for `done`.
- **Risk — Concurrent replacement:** Multiple runs target one filename.
  - **Safeguard:** Keep runs and temporary files independent and use complete atomic last-successful-finisher replacement.
- **Risk — Internal information exposure:** A post-stream exception could leak paths, tracebacks, or model state.
  - **Safeguard:** Log internally and terminate without a new client-visible error event.
- **Risk — Slow tests:** Full default training can make every test expensive.
  - **Safeguard:** Use small deterministic configurations for most tests and retain limited controlled default-contract coverage.
- **Evidence limitation:** The updated standalone `CONTEXT.md` and ADR file were not attached to this specific to-spec run. Their canonical Phase 4 terms and ADR decision are fully recorded in the confirmed `GRILL_WITH_DOCS_RESULT.md`, so this does not block the specification.
- **Evidence limitation:** The exact curated corpus, complete 500-rule Merge Table, deterministic expected matrices, and final rounded fixtures are not reproduced inline in this specification. They must be taken from the latest TypeScript Reference Implementation and independently captured during implementation.
- **Evidence limitation:** No production code was written and no pytest, Ruff, or mypy command was executed while creating this specification.
- **Current-code evidence:** The latest supplied Python snapshot registers Health, Simple Chat, BPE Tokenizer, and Neural Network behavior; contains reusable BPE and XOR numerical implementations; and contains prior-art route and persistence tests. The Word2Vec and Train Embed modules remain empty, and the Train Embed request model and router registration remain absent.
- **Publication target:** Replace the local root `SPEC.md` with this document. No working issue-tracker configuration or publication permission was supplied.

## Source Material Consulted

### Directly supplied for this run

- `TO_SPEC_PROMPT.md`
- `GRILL_WITH_DOCS_RESULT.md`
- Current `SPEC.md`
- Latest `py_llm_pipeline_explorer_file_structure.md`

### Confirmed project evidence

- Updated `CONTEXT.md` terminology recorded in the handoff
- ADR 0001 — Preserve Deterministic Word2Vec Compatibility, recorded in the handoff
- Backend project configuration
- FastAPI application registration
- Shared request schemas
- Shared SSE transport
- Reusable BPE module
- Reusable XOR Neural Network module
- Neural Network route and atomic persistence implementation
- Existing Simple Chat, BPE, Neural Network, route, and persistence tests
- Empty Word2Vec module
- Empty Train Embed route

### TypeScript Reference Implementation

- Train Embed request schema
- Train Embed route
- Word2Vec Skip-gram trainer
- Curated corpus and Vocabulary construction
- Shared BPE implementation
- Shared cosine-similarity behavior
- Train Embed frontend hook
- Train Embed result component
- Shared frontend SSE reader

### Authoritative technical references

- Official FastAPI documentation for streaming responses and in-process testing
- Official Starlette documentation for request-disconnection detection
- Official Python documentation for `asyncio.to_thread()`, cancellation, temporary files, and atomic replacement
- Official Pydantic documentation for strict validation and field aliases
- Official NumPy documentation for `float64`, exact assertions, tolerance-aware assertions, and stable ordering
- Official pytest documentation for parametrization, temporary paths, and monkeypatching
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*

## Recommended Next Step

Run `to-tickets-prompt` using this `SPEC.md`, the confirmed Phase 4 handoff, the latest Python Backend source export, the updated project context, ADR 0001, and the latest TypeScript Reference Implementation.

The resulting work items must remain limited to Phase 4 Word2Vec embeddings. They must not introduce frontend changes, Transformer implementation, multiprocessing, shared memory, third-party Word2Vec libraries, model loading, job management, application-level concurrency controls, or a new SSE error-event contract.
