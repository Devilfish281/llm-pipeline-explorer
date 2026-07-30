# src/backend/src/how_llms_work/ml/neural_net.py
"""Reference-compatible XOR neural-network training operations."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
from how_llms_work.ml.math_utils import round_typescript_decimal as _round_typescript_decimal

NetworkMode: TypeAlias = Literal["single-layer", "multi-layer"]
Float32Array: TypeAlias = npt.NDArray[np.float32]

HIDDEN_NEURONS: Final = 4
LEARNING_RATE: Final = np.float32(1.0)

SINGLE_LAYER_ARCHITECTURE: Final = "Single-Layer Perceptron (2 → 1)"
MULTI_LAYER_ARCHITECTURE: Final = "Multi-Layer Network (2 → 4 → 1)"

SINGLE_LAYER_SUCCESS_VERDICT: Final = "SUCCESS — network learned XOR"
SINGLE_LAYER_FAILURE_VERDICT: Final = "FAILED — loss stuck, predictions are random guesses"
MULTI_LAYER_SUCCESS_VERDICT: Final = "SUCCESS — network learned XOR via backpropagation"
MULTI_LAYER_FAILURE_VERDICT: Final = "FAILED — network did not converge, try more epochs"


@dataclass(frozen=True, slots=True)
class XORExample:
    """One input and target pair from the XOR truth table."""

    input: tuple[int, int]
    target: int


XOR_EXAMPLES: Final[tuple[XORExample, ...]] = (
    XORExample(input=(0, 0), target=0),
    XORExample(input=(0, 1), target=1),
    XORExample(input=(1, 0), target=1),
    XORExample(input=(1, 1), target=0),
)


class EpochPayload(TypedDict):
    """Serialized Epoch Update contract."""

    epoch: int
    loss: float


class PredictionPayload(TypedDict):
    """Serialized XOR Prediction contract."""

    input: list[int]
    expected: int
    actual: float


class FrontendResultPayload(TypedDict):
    """Frontend-facing final result without Saved Weight Snapshot data."""

    architecture: str
    predictions: list[PredictionPayload]
    verdict: str


class SingleLayerSnapshot(TypedDict):
    """Plain-Python Saved Weight Snapshot for Single-Layer Mode."""

    type: Literal["single-layer"]
    w1: float
    w2: float
    bias: float


class MultiLayerSnapshot(TypedDict):
    """Plain-Python Saved Weight Snapshot for Multi-Layer Mode."""

    type: Literal["multi-layer"]
    w1: list[list[float]]
    b1: list[float]
    w2: list[float]
    b2: float


SavedNetwork: TypeAlias = SingleLayerSnapshot | MultiLayerSnapshot


@dataclass(frozen=True, slots=True)
class EpochUpdate:
    """One reference-compatible training progress update."""

    epoch: int
    loss: float

    def to_payload(self) -> EpochPayload:
        """Return the exact plain-Python Epoch Update payload."""
        return {
            "epoch": self.epoch,
            "loss": self.loss,
        }


@dataclass(slots=True)
class Prediction:
    """One final rounded XOR Prediction."""

    input: list[int]
    expected: int
    actual: float

    def to_payload(self) -> PredictionPayload:
        """Return an isolated plain-Python prediction payload."""
        return {
            "input": list(self.input),
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(slots=True)
class SingleLayerState:
    """Mutable float32 numerical state for Single-Layer Mode."""

    weights: Float32Array
    bias: np.float32

    def to_snapshot(self) -> SingleLayerSnapshot:
        """Convert the numerical state to the exact snapshot schema."""
        return {
            "type": "single-layer",
            "w1": float(self.weights[0]),
            "w2": float(self.weights[1]),
            "bias": float(self.bias),
        }


@dataclass(slots=True)
class MultiLayerState:
    """Mutable float32 numerical state for Multi-Layer Mode."""

    w1: Float32Array
    b1: Float32Array
    w2: Float32Array
    b2: np.float32

    def to_snapshot(self) -> MultiLayerSnapshot:
        """Convert the numerical state to the exact snapshot schema."""
        return {
            "type": "multi-layer",
            "w1": [[float(value) for value in row] for row in self.w1],
            "b1": [float(value) for value in self.b1],
            "w2": [float(value) for value in self.w2],
            "b2": float(self.b2),
        }


NetworkState: TypeAlias = SingleLayerState | MultiLayerState


@dataclass(slots=True)
class TrainingResult:
    """Completed Training Run result and Saved Weight Snapshot data."""

    architecture: str
    predictions: list[Prediction]
    verdict: str
    weights: SavedNetwork

    def to_frontend_payload(self) -> FrontendResultPayload:
        """Return the final frontend payload without Saved Weight Snapshot data."""
        return {
            "architecture": self.architecture,
            "predictions": [prediction.to_payload() for prediction in self.predictions],
            "verdict": self.verdict,
        }


TrainingEvent: TypeAlias = EpochUpdate | TrainingResult


def sigmoid(value: float | np.float32) -> np.float32:
    """Return the sigmoid activation as a float32 scalar."""
    return np.float32(1.0 / (1.0 + math.exp(-float(value))))


def sigmoid_derivative(sigmoid_output: float | np.float32) -> np.float32:
    """Return the sigmoid derivative from an already-computed output."""
    output = np.float32(sigmoid_output)
    return np.float32(output * np.float32(np.float32(1.0) - output))


def round_like_typescript(
    value: float | np.float32,
    digits: int,
) -> float:
    """Round one public XOR value with TypeScript-compatible semantics."""
    return _round_typescript_decimal(
        float(value),
        digits,
    )


def predictions_are_successful(predictions: Sequence[Prediction]) -> bool:
    """Return whether every rounded XOR Prediction meets the strict threshold."""
    return all(abs(prediction.actual - prediction.expected) < 0.1 for prediction in predictions)


def _random_float32_weights(
    generator: np.random.Generator,
    shape: int | tuple[int, ...],
) -> Float32Array:
    weights = generator.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    upper_bound = np.nextafter(np.float32(1.0), np.float32(-1.0))
    weights[weights >= np.float32(1.0)] = upper_bound
    weights[weights < np.float32(-1.0)] = np.float32(-1.0)
    return weights


def _initialize_single_layer(generator: np.random.Generator) -> SingleLayerState:
    return SingleLayerState(
        weights=_random_float32_weights(generator, 2),
        bias=np.float32(0.0),
    )


def _initialize_multi_layer(generator: np.random.Generator) -> MultiLayerState:
    w1 = _random_float32_weights(generator, (2, HIDDEN_NEURONS))
    w2 = _random_float32_weights(generator, HIDDEN_NEURONS)
    return MultiLayerState(
        w1=w1,
        b1=np.zeros(HIDDEN_NEURONS, dtype=np.float32),
        w2=w2,
        b2=np.float32(0.0),
    )


def _train_single_layer_epoch(state: SingleLayerState) -> np.float32:
    total_loss = np.float32(0.0)

    for example in XOR_EXAMPLES:
        x1 = np.float32(example.input[0])
        x2 = np.float32(example.input[1])
        target = np.float32(example.target)

        weighted_sum = np.float32(x1 * state.weights[0] + x2 * state.weights[1] + state.bias)
        output = sigmoid(weighted_sum)
        error = np.float32(output - target)
        total_loss = np.float32(total_loss + error * error)

        delta = np.float32(error * sigmoid_derivative(output))
        state.weights[0] = np.float32(state.weights[0] - LEARNING_RATE * delta * x1)
        state.weights[1] = np.float32(state.weights[1] - LEARNING_RATE * delta * x2)
        state.bias = np.float32(state.bias - LEARNING_RATE * delta)

    return np.float32(total_loss / np.float32(len(XOR_EXAMPLES)))


def _train_multi_layer_epoch(state: MultiLayerState) -> np.float32:
    total_loss = np.float32(0.0)

    for example in XOR_EXAMPLES:
        x1 = np.float32(example.input[0])
        x2 = np.float32(example.input[1])
        target = np.float32(example.target)

        hidden = np.empty(HIDDEN_NEURONS, dtype=np.float32)
        for hidden_index in range(HIDDEN_NEURONS):
            hidden_sum = np.float32(
                x1 * state.w1[0, hidden_index]
                + x2 * state.w1[1, hidden_index]
                + state.b1[hidden_index]
            )
            hidden[hidden_index] = sigmoid(hidden_sum)

        output_sum = np.float32(state.b2)
        for hidden_index in range(HIDDEN_NEURONS):
            output_sum = np.float32(output_sum + hidden[hidden_index] * state.w2[hidden_index])
        output = sigmoid(output_sum)

        error = np.float32(output - target)
        total_loss = np.float32(total_loss + error * error)
        output_delta = np.float32(error * sigmoid_derivative(output))

        hidden_delta = np.empty(HIDDEN_NEURONS, dtype=np.float32)
        for hidden_index in range(HIDDEN_NEURONS):
            hidden_delta[hidden_index] = np.float32(
                output_delta * state.w2[hidden_index] * sigmoid_derivative(hidden[hidden_index])
            )

        for hidden_index in range(HIDDEN_NEURONS):
            state.w2[hidden_index] = np.float32(
                state.w2[hidden_index] - LEARNING_RATE * output_delta * hidden[hidden_index]
            )
        state.b2 = np.float32(state.b2 - LEARNING_RATE * output_delta)

        for hidden_index in range(HIDDEN_NEURONS):
            state.w1[0, hidden_index] = np.float32(
                state.w1[0, hidden_index] - LEARNING_RATE * hidden_delta[hidden_index] * x1
            )
            state.w1[1, hidden_index] = np.float32(
                state.w1[1, hidden_index] - LEARNING_RATE * hidden_delta[hidden_index] * x2
            )
            state.b1[hidden_index] = np.float32(
                state.b1[hidden_index] - LEARNING_RATE * hidden_delta[hidden_index]
            )

    return np.float32(total_loss / np.float32(len(XOR_EXAMPLES)))


def _predict_single_layer(state: SingleLayerState) -> list[Prediction]:
    predictions: list[Prediction] = []

    for example in XOR_EXAMPLES:
        x1 = np.float32(example.input[0])
        x2 = np.float32(example.input[1])
        weighted_sum = np.float32(x1 * state.weights[0] + x2 * state.weights[1] + state.bias)
        actual = round_like_typescript(sigmoid(weighted_sum), 2)
        predictions.append(
            Prediction(
                input=list(example.input),
                expected=example.target,
                actual=actual,
            )
        )

    return predictions


def _predict_multi_layer(state: MultiLayerState) -> list[Prediction]:
    predictions: list[Prediction] = []

    for example in XOR_EXAMPLES:
        x1 = np.float32(example.input[0])
        x2 = np.float32(example.input[1])

        hidden = np.empty(HIDDEN_NEURONS, dtype=np.float32)
        for hidden_index in range(HIDDEN_NEURONS):
            hidden_sum = np.float32(
                x1 * state.w1[0, hidden_index]
                + x2 * state.w1[1, hidden_index]
                + state.b1[hidden_index]
            )
            hidden[hidden_index] = sigmoid(hidden_sum)

        output_sum = np.float32(state.b2)
        for hidden_index in range(HIDDEN_NEURONS):
            output_sum = np.float32(output_sum + hidden[hidden_index] * state.w2[hidden_index])

        actual = round_like_typescript(sigmoid(output_sum), 2)
        predictions.append(
            Prediction(
                input=list(example.input),
                expected=example.target,
                actual=actual,
            )
        )

    return predictions


class TrainingRun(Iterator[TrainingEvent]):
    """Advance one XOR Training Run from one Epoch Update to the next."""

    def __init__(
        self,
        mode: NetworkMode,
        epochs: int,
        generator: np.random.Generator | None = None,
    ) -> None:
        if mode not in ("single-layer", "multi-layer"):
            raise ValueError(f"Unsupported network mode: {mode}")
        if epochs < 0:
            raise ValueError("epochs must be non-negative")

        self.mode = mode
        self.epochs = epochs
        self.generator = generator if generator is not None else np.random.default_rng()
        self.state: NetworkState = (
            _initialize_single_layer(self.generator)
            if mode == "single-layer"
            else _initialize_multi_layer(self.generator)
        )
        self._reporting_step = max(1, epochs // 50)
        self._next_epoch = 0
        self._result_emitted = False

    def __iter__(self) -> TrainingRun:
        """Return this Training Run iterator."""
        return self

    def __next__(self) -> TrainingEvent:
        """Advance through one reporting interval and return its next public event."""
        while self._next_epoch <= self.epochs:
            epoch = self._next_epoch
            loss = self._train_epoch()
            self._next_epoch += 1

            if epoch % self._reporting_step == 0 or epoch == self.epochs:
                return EpochUpdate(
                    epoch=epoch,
                    loss=round_like_typescript(loss, 6),
                )

        if not self._result_emitted:
            self._result_emitted = True
            return self._build_result()

        raise StopIteration

    def _train_epoch(self) -> np.float32:
        if isinstance(self.state, SingleLayerState):
            return _train_single_layer_epoch(self.state)
        return _train_multi_layer_epoch(self.state)

    def _build_result(self) -> TrainingResult:
        if isinstance(self.state, SingleLayerState):
            predictions = _predict_single_layer(self.state)
            success = predictions_are_successful(predictions)
            return TrainingResult(
                architecture=SINGLE_LAYER_ARCHITECTURE,
                predictions=predictions,
                verdict=(SINGLE_LAYER_SUCCESS_VERDICT if success else SINGLE_LAYER_FAILURE_VERDICT),
                weights=self.state.to_snapshot(),
            )

        predictions = _predict_multi_layer(self.state)
        success = predictions_are_successful(predictions)
        return TrainingResult(
            architecture=MULTI_LAYER_ARCHITECTURE,
            predictions=predictions,
            verdict=(MULTI_LAYER_SUCCESS_VERDICT if success else MULTI_LAYER_FAILURE_VERDICT),
            weights=self.state.to_snapshot(),
        )


def create_training_run(
    mode: NetworkMode,
    epochs: int,
    generator: np.random.Generator | None = None,
) -> TrainingRun:
    """Create an independent, bounded XOR Training Run iterator."""
    return TrainingRun(
        mode=mode,
        epochs=epochs,
        generator=generator,
    )
