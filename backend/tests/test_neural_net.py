# src/backend/tests/test_neural_net.py
import math

import numpy as np
import pytest
from how_llms_work.ml.neural_net import (
    MULTI_LAYER_ARCHITECTURE,
    MULTI_LAYER_FAILURE_VERDICT,
    MULTI_LAYER_SUCCESS_VERDICT,
    SINGLE_LAYER_ARCHITECTURE,
    SINGLE_LAYER_FAILURE_VERDICT,
    SINGLE_LAYER_SUCCESS_VERDICT,
    XOR_EXAMPLES,
    EpochUpdate,
    MultiLayerState,
    NetworkMode,
    Prediction,
    SingleLayerState,
    TrainingResult,
    create_training_run,
    predictions_are_successful,
    round_like_typescript,
    sigmoid,
    sigmoid_derivative,
)

DETERMINISTIC_SEED = 0
DETERMINISTIC_EPOCHS = 5_000
FLOAT32_RTOL = 1e-6
FLOAT32_ATOL = 1e-7


def _typescript_round(value: float, digits: int) -> float:
    scale = 10**digits
    return math.floor(value * scale + 0.5) / scale


def _reference_sigmoid(value: float) -> np.float32:
    return np.float32(1.0 / (1.0 + math.exp(-value)))


def _reference_sigmoid_derivative(output: np.float32) -> np.float32:
    return np.float32(output * np.float32(np.float32(1.0) - output))


def _reference_single_layer_epoch(
    weights: np.ndarray[tuple[int], np.dtype[np.float32]],
    bias: np.float32,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float32]], np.float32, np.float32]:
    expected_weights = weights.copy()
    expected_bias = np.float32(bias)
    total_loss = np.float32(0.0)

    for example in XOR_EXAMPLES:
        x1 = np.float32(example.input[0])
        x2 = np.float32(example.input[1])
        target = np.float32(example.target)
        output = _reference_sigmoid(
            float(x1 * expected_weights[0] + x2 * expected_weights[1] + expected_bias)
        )
        error = np.float32(output - target)
        total_loss = np.float32(total_loss + error * error)
        delta = np.float32(error * _reference_sigmoid_derivative(output))
        expected_weights[0] = np.float32(expected_weights[0] - delta * x1)
        expected_weights[1] = np.float32(expected_weights[1] - delta * x2)
        expected_bias = np.float32(expected_bias - delta)

    loss = np.float32(total_loss / np.float32(len(XOR_EXAMPLES)))
    return expected_weights, expected_bias, loss


def _reference_multi_layer_epoch(
    w1: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    b1: np.ndarray[tuple[int], np.dtype[np.float32]],
    w2: np.ndarray[tuple[int], np.dtype[np.float32]],
    b2: np.float32,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float32]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
    np.float32,
    np.float32,
]:
    expected_w1 = w1.copy()
    expected_b1 = b1.copy()
    expected_w2 = w2.copy()
    expected_b2 = np.float32(b2)
    total_loss = np.float32(0.0)

    for example in XOR_EXAMPLES:
        x1 = np.float32(example.input[0])
        x2 = np.float32(example.input[1])
        target = np.float32(example.target)

        hidden = np.empty(4, dtype=np.float32)
        for hidden_index in range(4):
            hidden[hidden_index] = _reference_sigmoid(
                float(
                    x1 * expected_w1[0, hidden_index]
                    + x2 * expected_w1[1, hidden_index]
                    + expected_b1[hidden_index]
                )
            )

        output_sum = np.float32(expected_b2)
        for hidden_index in range(4):
            output_sum = np.float32(output_sum + hidden[hidden_index] * expected_w2[hidden_index])
        output = _reference_sigmoid(float(output_sum))

        error = np.float32(output - target)
        total_loss = np.float32(total_loss + error * error)
        output_delta = np.float32(error * _reference_sigmoid_derivative(output))

        hidden_delta = np.empty(4, dtype=np.float32)
        for hidden_index in range(4):
            hidden_delta[hidden_index] = np.float32(
                output_delta
                * expected_w2[hidden_index]
                * _reference_sigmoid_derivative(hidden[hidden_index])
            )

        for hidden_index in range(4):
            expected_w2[hidden_index] = np.float32(
                expected_w2[hidden_index] - output_delta * hidden[hidden_index]
            )
        expected_b2 = np.float32(expected_b2 - output_delta)

        for hidden_index in range(4):
            expected_w1[0, hidden_index] = np.float32(
                expected_w1[0, hidden_index] - hidden_delta[hidden_index] * x1
            )
            expected_w1[1, hidden_index] = np.float32(
                expected_w1[1, hidden_index] - hidden_delta[hidden_index] * x2
            )
            expected_b1[hidden_index] = np.float32(
                expected_b1[hidden_index] - hidden_delta[hidden_index]
            )

    loss = np.float32(total_loss / np.float32(len(XOR_EXAMPLES)))
    return expected_w1, expected_b1, expected_w2, expected_b2, loss


def _collect_run(
    mode: NetworkMode,
    epochs: int,
    seed: int,
) -> tuple[list[EpochUpdate], TrainingResult]:
    run = create_training_run(
        mode,
        epochs,
        generator=np.random.default_rng(seed),
    )
    events = list(run)
    updates = [event for event in events if isinstance(event, EpochUpdate)]
    results = [event for event in events if isinstance(event, TrainingResult)]

    assert len(results) == 1
    assert events[-1] is results[0]
    return updates, results[0]


def _assert_plain_json_value(value: object) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain_json_value(item)
        return
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_plain_json_value(item)
        return
    raise AssertionError(f"Non-JSON-compatible value: {type(value)!r}")


def test_xor_examples_preserve_reference_order() -> None:
    assert [(example.input, example.target) for example in XOR_EXAMPLES] == [
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0),
    ]


def test_sigmoid_and_derivative_match_reference_values() -> None:
    np.testing.assert_allclose(
        [sigmoid(-10.0), sigmoid(0.0), sigmoid(10.0)],
        [0.00004539787, 0.5, 0.9999546],
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        sigmoid_derivative(np.float32(0.25)),
        np.float32(0.1875),
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (1.2345644, 6, 1.234564),
        (1.2345645, 6, 1.234565),
        (0.125, 2, 0.13),
        (-1.5, 0, -1.0),
    ],
)
def test_round_like_typescript_preserves_math_round_behavior(
    value: float,
    digits: int,
    expected: float,
) -> None:
    assert round_like_typescript(value, digits) == expected


def test_single_layer_initialization_uses_seeded_float32_state() -> None:
    expected_generator = np.random.default_rng(7)
    expected_weights = expected_generator.uniform(-1.0, 1.0, size=2).astype(np.float32)
    run = create_training_run(
        "single-layer",
        epochs=0,
        generator=np.random.default_rng(7),
    )

    assert isinstance(run.state, SingleLayerState)
    assert run.state.weights.shape == (2,)
    assert run.state.weights.dtype == np.float32
    assert isinstance(run.state.bias, np.float32)
    assert run.state.bias == np.float32(0.0)
    assert np.all(run.state.weights >= -1.0)
    assert np.all(run.state.weights < 1.0)
    np.testing.assert_allclose(
        run.state.weights,
        expected_weights,
        rtol=0.0,
        atol=0.0,
    )


def test_multi_layer_initialization_uses_seeded_float32_state() -> None:
    expected_generator = np.random.default_rng(7)
    expected_w1 = expected_generator.uniform(-1.0, 1.0, size=(2, 4)).astype(np.float32)
    expected_w2 = expected_generator.uniform(-1.0, 1.0, size=4).astype(np.float32)
    run = create_training_run(
        "multi-layer",
        epochs=0,
        generator=np.random.default_rng(7),
    )

    assert isinstance(run.state, MultiLayerState)
    assert run.state.w1.shape == (2, 4)
    assert run.state.b1.shape == (4,)
    assert run.state.w2.shape == (4,)
    assert run.state.w1.dtype == np.float32
    assert run.state.b1.dtype == np.float32
    assert run.state.w2.dtype == np.float32
    assert isinstance(run.state.b2, np.float32)
    assert np.array_equal(run.state.b1, np.zeros(4, dtype=np.float32))
    assert run.state.b2 == np.float32(0.0)
    assert np.all(run.state.w1 >= -1.0)
    assert np.all(run.state.w1 < 1.0)
    assert np.all(run.state.w2 >= -1.0)
    assert np.all(run.state.w2 < 1.0)
    np.testing.assert_allclose(run.state.w1, expected_w1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(run.state.w2, expected_w2, rtol=0.0, atol=0.0)


def test_production_runs_create_fresh_generators_and_state() -> None:
    first = create_training_run("single-layer", epochs=0)
    second = create_training_run("single-layer", epochs=0)

    assert first.generator is not second.generator
    assert isinstance(first.state, SingleLayerState)
    assert isinstance(second.state, SingleLayerState)
    assert not np.shares_memory(first.state.weights, second.state.weights)

    second_weights = second.state.weights.copy()
    first.state.weights[0] = np.float32(99.0)
    np.testing.assert_allclose(second.state.weights, second_weights, rtol=0.0, atol=0.0)


def test_single_layer_first_epoch_matches_immediate_update_reference() -> None:
    run = create_training_run(
        "single-layer",
        epochs=0,
        generator=np.random.default_rng(11),
    )
    assert isinstance(run.state, SingleLayerState)
    expected_weights, expected_bias, expected_loss = _reference_single_layer_epoch(
        run.state.weights,
        run.state.bias,
    )

    update = next(run)

    assert isinstance(update, EpochUpdate)
    assert update.to_payload() == {
        "epoch": 0,
        "loss": _typescript_round(float(expected_loss), 6),
    }
    np.testing.assert_allclose(
        run.state.weights,
        expected_weights,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        run.state.bias,
        expected_bias,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


def test_multi_layer_first_epoch_uses_pre_update_output_weights() -> None:
    run = create_training_run(
        "multi-layer",
        epochs=0,
        generator=np.random.default_rng(11),
    )
    assert isinstance(run.state, MultiLayerState)
    expected_w1, expected_b1, expected_w2, expected_b2, expected_loss = (
        _reference_multi_layer_epoch(
            run.state.w1,
            run.state.b1,
            run.state.w2,
            run.state.b2,
        )
    )

    update = next(run)

    assert isinstance(update, EpochUpdate)
    assert update.to_payload() == {
        "epoch": 0,
        "loss": _typescript_round(float(expected_loss), 6),
    }
    np.testing.assert_allclose(
        run.state.w1,
        expected_w1,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        run.state.b1,
        expected_b1,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        run.state.w2,
        expected_w2,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )
    np.testing.assert_allclose(
        run.state.b2,
        expected_b2,
        rtol=FLOAT32_RTOL,
        atol=FLOAT32_ATOL,
    )


@pytest.mark.parametrize(
    ("epochs", "expected_epochs"),
    [
        (100, list(range(0, 101, 2))),
        (101, [*range(0, 101, 2), 101]),
        (5_000, list(range(0, 5_001, 100))),
    ],
)
def test_reporting_schedule_includes_zero_and_final_epoch(
    epochs: int,
    expected_epochs: list[int],
) -> None:
    updates, _ = _collect_run("single-layer", epochs, seed=5)

    assert [update.epoch for update in updates] == expected_epochs
    assert all(set(update.to_payload()) == {"epoch", "loss"} for update in updates)
    assert all(update.loss == round_like_typescript(update.loss, 6) for update in updates)


def test_exact_architecture_and_verdict_constants_are_preserved() -> None:
    assert SINGLE_LAYER_ARCHITECTURE == "Single-Layer Perceptron (2 → 1)"
    assert MULTI_LAYER_ARCHITECTURE == "Multi-Layer Network (2 → 4 → 1)"
    assert SINGLE_LAYER_SUCCESS_VERDICT == "SUCCESS — network learned XOR"
    assert SINGLE_LAYER_FAILURE_VERDICT == ("FAILED — loss stuck, predictions are random guesses")
    assert MULTI_LAYER_SUCCESS_VERDICT == ("SUCCESS — network learned XOR via backpropagation")
    assert MULTI_LAYER_FAILURE_VERDICT == ("FAILED — network did not converge, try more epochs")


def test_success_threshold_is_strict_and_uses_rounded_predictions() -> None:
    passing = [
        Prediction(input=[0, 0], expected=0, actual=0.09),
        Prediction(input=[0, 1], expected=1, actual=0.91),
        Prediction(input=[1, 0], expected=1, actual=0.99),
        Prediction(input=[1, 1], expected=0, actual=0.01),
    ]
    boundary_failure = [
        Prediction(input=[0, 0], expected=0, actual=0.1),
        *passing[1:],
    ]

    assert predictions_are_successful(passing)
    assert not predictions_are_successful(boundary_failure)


def test_single_layer_result_and_snapshot_contract() -> None:
    _, result = _collect_run("single-layer", 100, seed=3)

    assert result.architecture == SINGLE_LAYER_ARCHITECTURE
    assert result.verdict in {
        SINGLE_LAYER_SUCCESS_VERDICT,
        SINGLE_LAYER_FAILURE_VERDICT,
    }
    assert [prediction.input for prediction in result.predictions] == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    assert [prediction.expected for prediction in result.predictions] == [0, 1, 1, 0]
    assert all(
        prediction.actual == round_like_typescript(prediction.actual, 2)
        for prediction in result.predictions
    )
    assert set(result.weights) == {"type", "w1", "w2", "bias"}
    assert result.weights["type"] == "single-layer"
    _assert_plain_json_value(result.weights)
    assert set(result.to_frontend_payload()) == {
        "architecture",
        "predictions",
        "verdict",
    }


def test_multi_layer_result_and_snapshot_contract() -> None:
    _, result = _collect_run("multi-layer", 100, seed=3)

    assert result.architecture == MULTI_LAYER_ARCHITECTURE
    assert result.verdict in {
        MULTI_LAYER_SUCCESS_VERDICT,
        MULTI_LAYER_FAILURE_VERDICT,
    }
    assert [prediction.input for prediction in result.predictions] == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    assert [prediction.expected for prediction in result.predictions] == [0, 1, 1, 0]
    assert all(
        prediction.actual == round_like_typescript(prediction.actual, 2)
        for prediction in result.predictions
    )
    assert set(result.weights) == {"type", "w1", "b1", "w2", "b2"}
    assert result.weights["type"] == "multi-layer"
    assert len(result.weights["w1"]) == 2
    assert all(len(row) == 4 for row in result.weights["w1"])
    assert len(result.weights["b1"]) == 4
    assert len(result.weights["w2"]) == 4
    _assert_plain_json_value(result.weights)


def test_deterministic_seed_proves_the_educational_contrast() -> None:
    _, single_result = _collect_run(
        "single-layer",
        DETERMINISTIC_EPOCHS,
        DETERMINISTIC_SEED,
    )
    _, multi_result = _collect_run(
        "multi-layer",
        DETERMINISTIC_EPOCHS,
        DETERMINISTIC_SEED,
    )

    assert single_result.verdict == SINGLE_LAYER_FAILURE_VERDICT
    assert multi_result.verdict == MULTI_LAYER_SUCCESS_VERDICT
    assert [prediction.actual for prediction in single_result.predictions] == [
        0.53,
        0.5,
        0.47,
        0.43,
    ]
    assert [prediction.actual for prediction in multi_result.predictions] == [
        0.01,
        0.99,
        0.98,
        0.02,
    ]
    assert predictions_are_successful(multi_result.predictions)


def test_completed_runs_do_not_share_returned_mutable_state() -> None:
    _, first = _collect_run("multi-layer", 100, seed=13)
    _, second = _collect_run("multi-layer", 100, seed=13)

    second_first_input = list(second.predictions[0].input)
    second_first_weight = second.weights["w1"][0][0]

    first.predictions[0].input[0] = 99
    first.weights["w1"][0][0] = 99.0

    assert second.predictions[0].input == second_first_input
    assert second.weights["w1"][0][0] == second_first_weight
