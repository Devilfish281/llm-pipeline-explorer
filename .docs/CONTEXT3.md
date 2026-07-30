# LLM Pipeline Explorer Context

This glossary defines the canonical domain language for the `llm-pipeline-explorer` project.

## Product

**LLM Pipeline Explorer**:
An interactive educational application that demonstrates major stages of a language-model pipeline through runnable visual examples.

**Learning Demo**:
A user-facing interactive experience that teaches one pipeline concept by showing its inputs, intermediate steps, and result.

## BPE Tokenization

**BPE Tokenizer**:
The educational Byte Pair Encoding component that begins with character-level tokens inside pre-token boundaries and repeatedly merges the most frequent adjacent pair.
_Avoid_: Basic Tokenizer when referring to the algorithm itself

**Basic Tokenizer**:
The frontend display name for the BPE Tokenizer Learning Demo.
_Avoid_: Using this name for the reusable tokenization algorithm

**BPE Training Text**:
The user-provided text from which a Tokenization Run learns pair frequencies and merge operations.
_Avoid_: Corpus when referring specifically to one user message

**Pre-token**:
An initial word, whitespace character, or punctuation character that forms a boundary within which BPE merges may occur. BPE merges never cross from one Pre-token into another.

**Token**:
A string unit produced during tokenization, beginning as a character and potentially becoming a larger subword or whole-word unit through merges.

**BPE Pair**:
Two adjacent Tokens considered together as a candidate for merging.

**BPE Merge**:
A learned operation that replaces each non-overlapping occurrence of one BPE Pair with a single combined Token.

**Merge Table**:
The ordered sequence of BPE Merges learned during a Tokenization Run and later applied in that same order.

**Vocabulary**:
The set of distinct Tokens known by one learning model or Tokenization Run.

**Tokenization Run**:
One execution of the BPE Tokenizer using a single BPE Training Text, producing initialization data, zero or more merge steps, and a final tokenization result.

**Compression Ratio**:
The original character count divided by the final token count, displayed as a multiplier such as `2.4x`.

## XOR Neural Network

**XOR Neural Network Demo**:
The Learning Demo that contrasts a single-layer neural network’s inability to learn XOR with a multi-layer neural network trained through backpropagation.

**Single-Layer Mode**:
The XOR training mode using two inputs connected directly to one sigmoid output. It demonstrates that a model without a hidden layer cannot reliably learn XOR.
_Avoid_: Multi-layer perceptron, hidden-layer model

**Multi-Layer Mode**:
The XOR training mode using two inputs, four sigmoid hidden neurons, and one sigmoid output trained through backpropagation.
_Avoid_: Single-layer perceptron

**Training Run**:
One complete execution of XOR neural-network training using one selected model mode, one epoch count, and one initialized set of weights, producing Epoch Updates, final predictions, and a Training Verdict.
_Avoid_: Embedding Training Run

**Weight Initialization**:
The assignment of starting values to a neural network’s weights before training. Each production Training Run begins with newly randomized weights.

**Epoch Update**:
A streamed XOR progress measurement containing an epoch number and its six-decimal training loss. A Training Run emits approximately fifty Epoch Updates, including epoch zero and the final requested epoch.
_Avoid_: Embedding Epoch Update

**XOR Prediction**:
A final result containing one XOR input pair, its expected output, and the network’s actual output rounded to two decimal places. Predictions are ordered as `[0,0]`, `[0,1]`, `[1,0]`, and `[1,1]`.

**Training Verdict**:
The exact success or failure message calculated from all four XOR Predictions. A run succeeds only when each rounded actual value differs from its expected value by less than `0.1`.

**Neural Network Event Stream**:
The ordered updates produced by one Training Run: approximately fifty `epoch` events followed by exactly one `done` event.

**Saved Weight Snapshot**:
The JSON representation of the final trained weights from the latest successfully completed Training Run for one model mode. Concurrent Training Runs remain independent, and the last successful finisher for a mode replaces that mode’s previous Saved Weight Snapshot.
_Avoid_: Training history, model registry, checkpoint history

## Word2Vec Embeddings

**Word2Vec Embeddings Demo**:
The Learning Demo that trains word vectors using Skip-gram with negative sampling and displays relationships learned from the Embedding Training Corpus.
_Avoid_: Transformer training, hosted embedding service

**Embedding Training Corpus**:
The fixed curated collection of sentences used to train the Word2Vec model. Query Words select results after training and do not become additional training text.
_Avoid_: User prompt, query text

**Word Embedding**:
The six-decimal public vector exposed for one Vocabulary Token and used by all displayed neighbor, similarity, and analogy calculations. It is the token’s visible learned representation, not an arbitrary internal training vector.

**Skip-gram Training**:
The Word2Vec training method that uses a target Token to learn the Tokens appearing around it within a selected context window.

**Training Pair**:
One ordered target-and-context Token relationship generated from the Embedding Training Corpus for Skip-gram Training.

**Negative Sample**:
A sampled Vocabulary Token treated as a non-context example during one Skip-gram update.

**Embedding Training Run**:
One complete deterministic execution of Skip-gram Training using the Embedding Training Corpus, one set of Query Words, and one validated set of embedding hyperparameters.
_Avoid_: Training Run when referring to Word2Vec, Query Run

**Embedding Epoch Update**:
A streamed Word2Vec progress measurement containing the current epoch and its training loss.
_Avoid_: Epoch Update when the surrounding context does not make the Learning Demo clear

**Query Word**:
One submitted word-list entry whose learned embedding relationships are requested after training. Duplicate Query Word entries remain separate entries, and Query Words do not alter the Embedding Training Corpus.

**Nearest Neighbor**:
A Vocabulary Token ranked as close to a Query Word according to the reference-compatible rounded cosine similarity between their Word Embeddings.

**Similarity Pair**:
Two recognized Query Word entries together with the cosine-similarity score between their Word Embeddings.

**Vector Analogy**:
A predefined word relationship evaluated through arithmetic on learned Word Embeddings and reported with the nearest matching Vocabulary Token.

**Embedding Result**:
The completed frontend-facing Word2Vec output containing selected Word Embeddings, Nearest Neighbors, Similarity Pairs, Vector Analogies, and any Query Word warnings. It is distinct from the complete persisted Saved Embedding Model.

**Saved Embedding Model**:
The JSON representation of the latest successfully completed Embedding Training Run, containing the model type, dimensions, complete ordered Vocabulary, learned Merge Table, and a Word Embedding for every Vocabulary Token. It is replaced by later successful runs but is not loaded for caching, inference, or continued training.
_Avoid_: Embedding Result, training history, model registry

**Embedding Event Stream**:
The ordered updates produced by one successful Embedding Training Run: one `init` event, approximately fifty `epoch` events, and exactly one `done` event.

## Compatibility

**Frontend Contract**:
The request shape, endpoint, SSE event names, event order, payload field names, payload structures, validation behavior, and completion behavior consumed by the TypeScript frontend.

**Strict TypeScript Compatibility**:
The migration requirement that a Python feature reproduce the observable behavior of its TypeScript Reference Implementation, including validation, numerical intent, event sequencing, serialized payloads, and completion behavior.
_Avoid_: Bit-for-bit equivalence when controlled floating-point differences are permitted

**Deterministic Embedding Compatibility**:
The requirement that identical valid Embedding Training Run inputs reproduce the TypeScript Reference Implementation’s random sequence, update order, rounded public results, result ordering, and Saved Embedding Model. Unrounded internal floating-point values may use explicitly verified numerical tolerances.
_Avoid_: Statistically similar output, nondeterministic equivalent

**BPE Event Stream**:
The ordered updates produced for one Tokenization Run: one `init` event, zero or more `merge` events, and one `result` event.

## Migration

**Python Backend**:
The only server-side implementation in the current LLM Pipeline Explorer program, built with FastAPI and Python under `backend/src/how_llms_work/`.
_Avoid_: TypeScript backend, Node backend

**TypeScript Reference Implementation**:
The original Hono and TypeScript server code consulted only to determine behavior that the Python Backend must reproduce. It is not stored, executed, or retained as part of the current backend.
_Avoid_: Current TypeScript backend, retained TypeScript backend

**Phase Migration**:
A backend conversion phase that translates behavior from the TypeScript Reference Implementation into the Python Backend, exposes and registers its Python HTTP route, and adds Python tests proving compatibility with the Frontend Contract.
_Avoid_: Translation when referring only to syntactic source-file conversion
