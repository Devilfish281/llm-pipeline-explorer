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
The frontend display name for the BPE Tokenizer learning demo.
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
The set of distinct character Tokens and merged Tokens known during a Tokenization Run.

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

**Weight Initialization**:
The assignment of starting values to a neural network’s weights before training. Each production Training Run begins with newly randomized weights.

**Epoch Update**:
A streamed progress measurement containing an epoch number and its six-decimal training loss. A Training Run emits approximately fifty Epoch Updates, including epoch zero and the final requested epoch.

**XOR Prediction**:
A final result containing one XOR input pair, its expected output, and the network’s actual output rounded to two decimal places. Predictions are ordered as `[0,0]`, `[0,1]`, `[1,0]`, and `[1,1]`.

**Training Verdict**:
The exact success or failure message calculated from all four XOR Predictions. A run succeeds only when each rounded actual value differs from its expected value by less than `0.1`.

**Neural Network Event Stream**:
The ordered updates produced by one Training Run: approximately fifty `epoch` events followed by exactly one `done` event.

**Saved Weight Snapshot**:
The JSON representation of the final trained weights from the latest successfully completed Training Run for one model mode. Concurrent Training Runs remain independent, and the last successful finisher for a mode replaces that mode’s previous Saved Weight Snapshot.
_Avoid_: Training history, model registry, checkpoint history

## Compatibility

**Frontend Contract**:
The request shape, endpoint, SSE event names, event order, payload field names, payload structures, validation behavior, and completion behavior consumed by the TypeScript frontend.

**Strict TypeScript Compatibility**:
The migration requirement that a Python feature reproduce the observable behavior of its TypeScript Reference Implementation, including validation, numerical intent, event sequencing, serialized payloads, and completion behavior.
_Avoid_: Bit-for-bit equivalence when controlled floating-point differences are permitted

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
