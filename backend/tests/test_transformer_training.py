# tests/test_transformer_training.py
from __future__ import annotations

import inspect
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from how_llms_work.ml.math_utils import Mulberry32
from how_llms_work.ml.transformer import (
    InitializedTransformerParameters,
    LogicalTrainingShard,
    LogicalTrainingShardResult,
    TransformerGradientBuffer,
    TransformerTrainingRun,
    build_logical_training_shards,
    build_transformer_parameter_layout,
    build_transformer_report_epochs,
    calculate_logical_training_shard,
    create_transformer_gradient_buffer,
    create_transformer_training_run,
    get_transformer_preprocessing,
    initialize_transformer_parameters,
)

_REFERENCE_PATH = Path(__file__).parent / "fixtures" / "transformer_training_reference.json"


def _load_reference() -> dict[str, Any]:
    with _REFERENCE_PATH.open(encoding="utf-8") as reference_file:
        return json.load(reference_file)


_REFERENCE = _load_reference()


def _selected_indices() -> np.ndarray:
    return np.asarray(
        _REFERENCE["layout"]["selectedIndices"],
        dtype=np.intp,
    )


def _new_initialized_parameters() -> InitializedTransformerParameters:
    layout = build_transformer_parameter_layout(1)
    initialized = initialize_transformer_parameters(
        layout,
        Mulberry32(42),
    )
    initialized.storage[_selected_indices()] = np.asarray(
        _REFERENCE["initialSelectedWeights"],
        dtype=np.float32,
    )
    return initialized


def _new_run(
    *,
    requested_epochs: int = 1,
    sequence_count: int | None = None,
) -> TransformerTrainingRun:
    effective_sequence_count = (
        int(_REFERENCE["layout"]["sequenceCount"]) if sequence_count is None else sequence_count
    )
    return create_transformer_training_run(
        _new_initialized_parameters(),
        sequence_count=effective_sequence_count,
        requested_epochs=requested_epochs,
    )


def _epoch_reference(epoch: int) -> dict[str, Any]:
    return _REFERENCE["epochs"][epoch]


def _build_epoch_results(
    run: TransformerTrainingRun,
    epoch_reference: dict[str, Any],
    arrival_order: tuple[int, int, int, int],
) -> tuple[LogicalTrainingShardResult, ...]:
    selected_indices = _selected_indices()
    by_shard: dict[int, LogicalTrainingShardResult] = {}

    for shard_index in range(4):
        gradient = create_transformer_gradient_buffer(
            run.parameters.layout,
        )
        gradient.storage[selected_indices] = np.asarray(
            epoch_reference["shardSelectedGradients"][shard_index],
            dtype=np.float32,
        )
        shard = run.logical_training_shards[shard_index]

        by_shard[shard_index] = LogicalTrainingShardResult(
            shard=shard,
            processed_sequence_count=shard.stop_index - shard.start_index,
            loss=float(epoch_reference["shardLosses"][shard_index]),
            gradient=gradient,
        )

    return tuple(by_shard[shard_index] for shard_index in arrival_order)


def _build_zero_results(
    run: TransformerTrainingRun,
    *,
    first_loss: float = 0.0,
) -> tuple[LogicalTrainingShardResult, ...]:
    results: list[LogicalTrainingShardResult] = []

    for shard_index, shard in enumerate(run.logical_training_shards):
        gradient = create_transformer_gradient_buffer(
            run.parameters.layout,
        )
        results.append(
            LogicalTrainingShardResult(
                shard=shard,
                processed_sequence_count=shard.stop_index - shard.start_index,
                loss=first_loss if shard_index == 0 else 0.0,
                gradient=gradient,
            )
        )

    return tuple(results)


def _persistent_snapshot(
    run: TransformerTrainingRun,
) -> tuple[
    bytes,
    bytes,
    bytes,
    int,
    int | None,
    float | None,
    tuple[object, ...],
]:
    return (
        run.weights.tobytes(),
        run.first_moments.tobytes(),
        run.second_moments.tobytes(),
        run.next_epoch,
        run.last_completed_epoch,
        run.last_completed_loss,
        run.updates,
    )


def _assert_failed_transition_preserved_state(
    run: TransformerTrainingRun,
    shard_results: object,
) -> None:
    before = _persistent_snapshot(run)

    with pytest.raises(
        (
            TypeError,
            ValueError,
            FloatingPointError,
        )
    ):
        run.advance_epoch(shard_results)  # type: ignore[arg-type]

    assert _persistent_snapshot(run) == before
    assert run.is_failed
    assert not run.is_active
    assert not run.is_complete

    with pytest.raises(RuntimeError, match="failed"):
        run.advance_epoch(shard_results)  # type: ignore[arg-type]


def test_reference_fixture_has_independent_provenance_and_layout() -> None:
    assert "Independent scalar" in _REFERENCE["provenance"]["source"]
    assert "imports no how_llms_work" in _REFERENCE["provenance"]["source"]
    assert _REFERENCE["layout"]["numLayers"] == 1
    assert _REFERENCE["layout"]["totalFloatCount"] == 39_272
    assert _REFERENCE["layout"]["sequenceCount"] == 3

    run = _new_run()

    assert run.parameters.layout.num_layers == 1
    assert run.parameters.layout.total_float_count == 39_272


@pytest.mark.parametrize(
    "requested_epochs",
    [
        0,
        1,
        49,
        50,
        51,
        99,
        100,
        101,
        5_000,
    ],
)
def test_transformer_report_schedule_matches_reference(
    requested_epochs: int,
) -> None:
    expected = tuple(int(epoch) for epoch in _REFERENCE["reportSchedules"][str(requested_epochs)])

    actual = build_transformer_report_epochs(requested_epochs)

    assert actual == expected
    assert actual[0] == 0
    assert actual[-1] == requested_epochs
    assert len(actual) == len(set(actual))


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("1", TypeError),
        (-1, ValueError),
    ],
)
def test_requested_final_epoch_is_strictly_validated(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        create_transformer_training_run(
            _new_initialized_parameters(),
            sequence_count=3,
            requested_epochs=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("3", TypeError),
        (-1, ValueError),
    ],
)
def test_sequence_count_is_strictly_validated(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        create_transformer_training_run(
            _new_initialized_parameters(),
            sequence_count=value,  # type: ignore[arg-type]
            requested_epochs=0,
        )


def test_fresh_runs_own_independent_float32_and_float64_state() -> None:
    initialized = _new_initialized_parameters()
    run_a = create_transformer_training_run(
        initialized,
        sequence_count=3,
        requested_epochs=1,
    )
    run_b = create_transformer_training_run(
        initialized,
        sequence_count=3,
        requested_epochs=1,
    )

    assert np.array_equal(run_a.weights, run_b.weights)
    assert not np.shares_memory(run_a.weights, initialized.storage)
    assert not np.shares_memory(run_b.weights, initialized.storage)
    assert not np.shares_memory(run_a.weights, run_b.weights)

    for run in (run_a, run_b):
        float32_arrays = (
            run.weights,
            run.first_moments,
            run.second_moments,
            run.reduced_gradient.storage,
        )

        for array in float32_arrays:
            assert array.dtype == np.dtype(np.float32)
            assert array.ndim == 1
            assert array.shape == (run.parameters.layout.total_float_count,)
            assert array.flags.c_contiguous
            assert array.flags.writeable
            assert np.isfinite(array).all()

        assert np.count_nonzero(run.first_moments) == 0
        assert np.count_nonzero(run.second_moments) == 0
        assert np.count_nonzero(run.reduced_gradient.storage) == 0

        scratch_metadata = run.adam_scratch_metadata

        assert len(scratch_metadata) == 2
        assert scratch_metadata[0][0] != scratch_metadata[1][0]
        assert scratch_metadata[0][1] == "float64"
        assert scratch_metadata[1][1] == "float64"
        assert scratch_metadata[0][2] == run.weights.shape
        assert scratch_metadata[1][2] == run.weights.shape

        assert (
            sum("adam_scratch" in slot_name for slot_name in TransformerTrainingRun.__slots__) == 2
        )
        assert not any("candidate" in slot_name for slot_name in TransformerTrainingRun.__slots__)

        for left_index, left in enumerate(float32_arrays):
            for right in float32_arrays[left_index + 1 :]:
                assert not np.shares_memory(left, right)

    assert run_a.adam_scratch_metadata[0][0] != run_b.adam_scratch_metadata[0][0]
    assert run_a.adam_scratch_metadata[1][0] != run_b.adam_scratch_metadata[1][0]
    assert run_a.updates == ()
    assert run_b.updates == ()


def test_training_factory_accepts_no_worker_or_persistence_configuration() -> None:
    factory_parameters = inspect.signature(create_transformer_training_run).parameters
    constructor_parameters = inspect.signature(TransformerTrainingRun).parameters
    forbidden_names = {
        "saved_model",
        "path",
        "file",
        "file_handle",
        "resume",
        "checkpoint",
        "worker_count",
        "process_group",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
    }

    assert forbidden_names.isdisjoint(factory_parameters)
    assert forbidden_names.isdisjoint(constructor_parameters)


def test_ordered_reduction_and_two_adam_steps_match_reference() -> None:
    run = _new_run(requested_epochs=1)
    selected_indices = _selected_indices()
    weight_identity = id(run.weights)
    first_moment_identity = id(run.first_moments)
    second_moment_identity = id(run.second_moments)
    reduction_identity = id(run.reduced_gradient.storage)
    scratch_metadata = run.adam_scratch_metadata

    epoch_zero_results = _build_epoch_results(
        run,
        _epoch_reference(0),
        (3, 2, 1, 0),
    )
    shard_gradient_bytes = tuple(result.gradient.storage.tobytes() for result in epoch_zero_results)

    observation_zero = run.advance_epoch(epoch_zero_results)
    epoch_zero = _epoch_reference(0)

    assert observation_zero.epoch == 0
    assert observation_zero.loss == epoch_zero["orderedLoss"]
    assert observation_zero.update is not None
    assert observation_zero.update.epoch == 0
    assert observation_zero.update.loss == epoch_zero["publicLoss"]

    np.testing.assert_array_equal(
        run.reduced_gradient.storage[selected_indices],
        np.asarray(
            epoch_zero["orderedSelectedGradient"],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        run.first_moments[selected_indices],
        np.asarray(
            epoch_zero["selectedFirstMoment"],
            dtype=np.float32,
        ),
        rtol=float(_REFERENCE["tolerances"]["momentRtol"]),
        atol=float(_REFERENCE["tolerances"]["momentAtol"]),
    )
    np.testing.assert_allclose(
        run.second_moments[selected_indices],
        np.asarray(
            epoch_zero["selectedSecondMoment"],
            dtype=np.float32,
        ),
        rtol=float(_REFERENCE["tolerances"]["momentRtol"]),
        atol=float(_REFERENCE["tolerances"]["momentAtol"]),
    )
    np.testing.assert_allclose(
        run.weights[selected_indices],
        np.asarray(
            epoch_zero["selectedWeight"],
            dtype=np.float32,
        ),
        rtol=float(_REFERENCE["tolerances"]["weightRtol"]),
        atol=float(_REFERENCE["tolerances"]["weightAtol"]),
    )

    assert (
        tuple(result.gradient.storage.tobytes() for result in epoch_zero_results)
        == shard_gradient_bytes
    )
    assert run.reduced_gradient.storage[selected_indices[1]] == np.float32(6.0)
    assert run.reduced_gradient.storage[selected_indices[1]] != np.float32(1.5)

    assert id(run.weights) == weight_identity
    assert id(run.first_moments) == first_moment_identity
    assert id(run.second_moments) == second_moment_identity
    assert id(run.reduced_gradient.storage) == reduction_identity
    assert run.adam_scratch_metadata == scratch_metadata
    assert run.next_epoch == 1
    assert run.last_completed_epoch == 0
    assert run.last_completed_loss == epoch_zero["orderedLoss"]
    assert run.is_active

    observation_one = run.advance_epoch(
        _build_epoch_results(
            run,
            _epoch_reference(1),
            (1, 3, 0, 2),
        )
    )
    epoch_one = _epoch_reference(1)

    assert observation_one.epoch == 1
    assert observation_one.loss == epoch_one["orderedLoss"]
    assert observation_one.update is not None
    assert observation_one.update.epoch == 1
    assert observation_one.update.loss == epoch_one["publicLoss"]

    np.testing.assert_array_equal(
        run.reduced_gradient.storage[selected_indices],
        np.asarray(
            epoch_one["orderedSelectedGradient"],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        run.first_moments[selected_indices],
        np.asarray(
            epoch_one["selectedFirstMoment"],
            dtype=np.float32,
        ),
        rtol=float(_REFERENCE["tolerances"]["momentRtol"]),
        atol=float(_REFERENCE["tolerances"]["momentAtol"]),
    )
    np.testing.assert_allclose(
        run.second_moments[selected_indices],
        np.asarray(
            epoch_one["selectedSecondMoment"],
            dtype=np.float32,
        ),
        rtol=float(_REFERENCE["tolerances"]["momentRtol"]),
        atol=float(_REFERENCE["tolerances"]["momentAtol"]),
    )
    np.testing.assert_allclose(
        run.weights[selected_indices],
        np.asarray(
            epoch_one["selectedWeight"],
            dtype=np.float32,
        ),
        rtol=float(_REFERENCE["tolerances"]["weightRtol"]),
        atol=float(_REFERENCE["tolerances"]["weightAtol"]),
    )

    assert id(run.weights) == weight_identity
    assert id(run.first_moments) == first_moment_identity
    assert id(run.second_moments) == second_moment_identity
    assert id(run.reduced_gradient.storage) == reduction_identity
    assert run.adam_scratch_metadata == scratch_metadata

    assert run.next_epoch == 2
    assert run.last_completed_epoch == 1
    assert run.last_completed_loss == epoch_one["orderedLoss"]
    assert run.is_complete
    assert not run.is_active
    assert not run.is_failed
    assert tuple(update.epoch for update in run.updates) == (0, 1)

    with pytest.raises(RuntimeError, match="completed"):
        run.advance_epoch(
            _build_epoch_results(
                run,
                _epoch_reference(1),
                (0, 1, 2, 3),
            )
        )

    assert run.is_complete
    assert not run.is_failed


def test_arrival_order_does_not_change_reduction_or_adam_result() -> None:
    completed_runs: list[TransformerTrainingRun] = []

    for arrival_order in _REFERENCE["arrivalOrders"].values():
        run = _new_run(requested_epochs=0)
        run.advance_epoch(
            _build_epoch_results(
                run,
                _epoch_reference(0),
                tuple(arrival_order),
            )
        )
        completed_runs.append(run)

    baseline = completed_runs[0]

    for run in completed_runs[1:]:
        np.testing.assert_array_equal(
            run.reduced_gradient.storage,
            baseline.reduced_gradient.storage,
        )
        np.testing.assert_array_equal(
            run.first_moments,
            baseline.first_moments,
        )
        np.testing.assert_array_equal(
            run.second_moments,
            baseline.second_moments,
        )
        np.testing.assert_array_equal(
            run.weights,
            baseline.weights,
        )
        assert run.last_completed_loss == baseline.last_completed_loss
        assert run.updates == baseline.updates
        assert run.last_completed_epoch == baseline.last_completed_epoch
        assert run.is_complete


def test_fresh_runs_remain_isolated_during_threaded_execution() -> None:
    arrival_orders = tuple(tuple(order) for order in _REFERENCE["arrivalOrders"].values())

    def execute(
        arrival_order: tuple[int, int, int, int],
    ) -> TransformerTrainingRun:
        run = _new_run(requested_epochs=0)
        run.advance_epoch(
            _build_epoch_results(
                run,
                _epoch_reference(0),
                arrival_order,
            )
        )
        return run

    with ThreadPoolExecutor(max_workers=4) as executor:
        runs = tuple(executor.map(execute, arrival_orders))

    baseline = runs[0]

    for run in runs[1:]:
        np.testing.assert_array_equal(run.weights, baseline.weights)
        np.testing.assert_array_equal(run.first_moments, baseline.first_moments)
        np.testing.assert_array_equal(run.second_moments, baseline.second_moments)
        assert run.updates == baseline.updates

    for left_index, left in enumerate(runs):
        for right in runs[left_index + 1 :]:
            assert not np.shares_memory(left.weights, right.weights)
            assert not np.shares_memory(
                left.first_moments,
                right.first_moments,
            )
            assert not np.shares_memory(
                left.second_moments,
                right.second_moments,
            )
            assert not np.shares_memory(
                left.reduced_gradient.storage,
                right.reduced_gradient.storage,
            )
            assert left.adam_scratch_metadata[0][0] != right.adam_scratch_metadata[0][0]
            assert left.adam_scratch_metadata[1][0] != right.adam_scratch_metadata[1][0]

    protected_weight = float(runs[1].weights[0])
    runs[0].weights[0] += np.float32(1.0)

    assert float(runs[1].weights[0]) == protected_weight


def test_direct_shard_calculation_integrates_with_epoch_transition() -> None:
    preprocessing = get_transformer_preprocessing()
    sequences = tuple(preprocessing.training_sequences[:4])
    initialized = _new_initialized_parameters()
    run = create_transformer_training_run(
        initialized,
        sequence_count=len(sequences),
        requested_epochs=0,
    )
    shards = build_logical_training_shards(len(sequences))
    results = tuple(
        calculate_logical_training_shard(
            sequences,
            shard,
            run.parameters.views,
        )
        for shard in shards
    )

    observation = run.advance_epoch(
        (
            results[2],
            results[0],
            results[3],
            results[1],
        )
    )

    assert observation.epoch == 0
    assert math.isfinite(observation.loss)
    assert observation.update is not None
    assert np.isfinite(run.weights).all()
    assert np.isfinite(run.first_moments).all()
    assert np.isfinite(run.second_moments).all()
    assert run.is_complete


def test_inclusive_epoch_progression_processes_zero_through_final() -> None:
    run = _new_run(
        requested_epochs=4,
        sequence_count=4,
    )
    completed_epochs: list[int] = []

    while run.is_active:
        observation = run.advance_epoch(_build_zero_results(run))
        completed_epochs.append(observation.epoch)

    assert completed_epochs == [0, 1, 2, 3, 4]
    assert run.next_epoch == 5
    assert run.last_completed_epoch == 4
    assert run.is_complete
    assert tuple(update.epoch for update in run.updates) == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "negative_shard",
        "high_shard",
        "wrong_metadata",
        "wrong_processed_count",
        "nonfinite_loss",
        "wrong_result_type",
        "empty_loss",
        "empty_gradient",
    ],
)
def test_malformed_shard_sets_fail_without_persistent_commit(
    case: str,
) -> None:
    run = _new_run(requested_epochs=0)
    results = list(
        _build_epoch_results(
            run,
            _epoch_reference(0),
            (0, 1, 2, 3),
        )
    )

    if case == "missing":
        malformed: object = tuple(results[:3])
    elif case == "duplicate":
        malformed = (
            results[0],
            results[1],
            results[2],
            results[0],
        )
    elif case == "negative_shard":
        results[0] = replace(
            results[0],
            shard=LogicalTrainingShard(
                shard_index=-1,
                start_index=0,
                stop_index=1,
            ),
        )
        malformed = tuple(results)
    elif case == "high_shard":
        results[0] = replace(
            results[0],
            shard=LogicalTrainingShard(
                shard_index=4,
                start_index=0,
                stop_index=1,
            ),
        )
        malformed = tuple(results)
    elif case == "wrong_metadata":
        results[0] = replace(
            results[0],
            shard=LogicalTrainingShard(
                shard_index=0,
                start_index=0,
                stop_index=2,
            ),
        )
        malformed = tuple(results)
    elif case == "wrong_processed_count":
        results[0] = replace(
            results[0],
            processed_sequence_count=2,
        )
        malformed = tuple(results)
    elif case == "nonfinite_loss":
        results[0] = replace(
            results[0],
            loss=float("inf"),
        )
        malformed = tuple(results)
    elif case == "wrong_result_type":
        malformed = (
            results[0],
            results[1],
            results[2],
            object(),
        )
    elif case == "empty_loss":
        results[3] = replace(
            results[3],
            loss=1.0,
        )
        malformed = tuple(results)
    else:
        results[3].gradient.storage[0] = np.float32(1.0)
        malformed = tuple(results)

    _assert_failed_transition_preserved_state(
        run,
        malformed,
    )


@pytest.mark.parametrize(
    "case",
    [
        "wrong_layout",
        "wrong_length",
        "wrong_dtype",
        "noncontiguous",
        "nonfinite",
        "foreign_views",
        "parent_alias",
    ],
)
def test_invalid_gradient_storage_fails_without_persistent_commit(
    case: str,
) -> None:
    run = _new_run(requested_epochs=0)
    results = list(
        _build_epoch_results(
            run,
            _epoch_reference(0),
            (0, 1, 2, 3),
        )
    )
    original_gradient = results[0].gradient
    total_float_count = run.parameters.layout.total_float_count

    if case == "wrong_layout":
        other_layout = build_transformer_parameter_layout(2)
        other_initialized = initialize_transformer_parameters(
            other_layout,
            Mulberry32(42),
        )
        bad_gradient = create_transformer_gradient_buffer(
            other_initialized.layout,
        )
    elif case == "wrong_length":
        bad_storage = np.zeros(
            total_float_count - 1,
            dtype=np.float32,
        )
        bad_gradient = TransformerGradientBuffer(
            layout=original_gradient.layout,
            storage=bad_storage,
            views=original_gradient.views,
        )
    elif case == "wrong_dtype":
        bad_storage = np.zeros(
            total_float_count,
            dtype=np.float64,
        )
        bad_gradient = TransformerGradientBuffer(
            layout=original_gradient.layout,
            storage=bad_storage,
            views=original_gradient.views,
        )
    elif case == "noncontiguous":
        backing = np.zeros(
            total_float_count * 2,
            dtype=np.float32,
        )
        bad_storage = backing[::2]
        assert not bad_storage.flags.c_contiguous

        bad_gradient = TransformerGradientBuffer(
            layout=original_gradient.layout,
            storage=bad_storage,
            views=original_gradient.views,
        )
    elif case == "nonfinite":
        original_gradient.storage[0] = np.float32(np.nan)
        bad_gradient = original_gradient
    elif case == "foreign_views":
        bad_storage = np.zeros(
            total_float_count,
            dtype=np.float32,
        )
        bad_gradient = TransformerGradientBuffer(
            layout=original_gradient.layout,
            storage=bad_storage,
            views=original_gradient.views,
        )
    else:
        bad_gradient = TransformerGradientBuffer(
            layout=run.parameters.layout,
            storage=run.weights,
            views=run.parameters.views,
        )

    results[0] = replace(
        results[0],
        gradient=bad_gradient,
    )

    _assert_failed_transition_preserved_state(
        run,
        tuple(results),
    )


def test_reduced_gradient_overflow_fails_without_persistent_commit() -> None:
    run = _new_run(
        requested_epochs=0,
        sequence_count=4,
    )
    results = list(_build_zero_results(run))
    maximum = np.finfo(np.float32).max
    results[0].gradient.storage[0] = maximum
    results[1].gradient.storage[0] = maximum

    _assert_failed_transition_preserved_state(
        run,
        tuple(results),
    )


def test_adam_candidate_overflow_preserves_last_completed_state() -> None:
    run = _new_run(requested_epochs=1)

    run.advance_epoch(
        _build_epoch_results(
            run,
            _epoch_reference(0),
            (0, 1, 2, 3),
        )
    )
    before = _persistent_snapshot(run)
    results = list(_build_zero_results(run))
    results[0].gradient.storage[0] = np.finfo(np.float32).max

    with pytest.raises(FloatingPointError):
        run.advance_epoch(tuple(results))

    assert _persistent_snapshot(run) == before
    assert run.last_completed_epoch == 0
    assert run.next_epoch == 1
    assert tuple(update.epoch for update in run.updates) == (0,)
    assert run.is_failed

    with pytest.raises(RuntimeError, match="failed"):
        run.advance_epoch(tuple(results))


@pytest.mark.parametrize(
    "array_name",
    [
        "weights",
        "first_moments",
        "second_moments",
    ],
)
def test_nonfinite_current_state_rejects_next_epoch_without_commit(
    array_name: str,
) -> None:
    run = _new_run(requested_epochs=1)

    run.advance_epoch(
        _build_epoch_results(
            run,
            _epoch_reference(0),
            (0, 1, 2, 3),
        )
    )

    target = getattr(run, array_name)
    target[0] = np.float32(np.inf)
    before = _persistent_snapshot(run)

    with pytest.raises(FloatingPointError):
        run.advance_epoch(_build_zero_results(run))

    assert _persistent_snapshot(run) == before
    assert run.last_completed_epoch == 0
    assert run.next_epoch == 1
    assert run.is_failed


def test_public_negative_zero_is_normalized_to_positive_zero() -> None:
    run = _new_run(
        requested_epochs=0,
        sequence_count=4,
    )
    observation = run.advance_epoch(
        _build_zero_results(
            run,
            first_loss=-0.0000004,
        )
    )

    assert observation.loss == -0.0000004
    assert observation.update is not None
    assert observation.update.loss == 0.0
    assert math.copysign(1.0, observation.update.loss) == 1.0
