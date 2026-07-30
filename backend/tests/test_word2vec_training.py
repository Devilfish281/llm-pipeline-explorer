# backend/tests/test_word2vec_training.py
from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
from how_llms_work.ml import word2vec as word2vec_module
from how_llms_work.ml.word2vec import (
    CompletedEmbeddingTraining,
    EmbeddingEpochUpdate,
    EmbeddingTrainingRun,
    Mulberry32,
    TrainingPair,
    Word2VecPreprocessing,
    create_embedding_training_run,
    embedding_sigmoid,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "word2vec_training_reference.json"

REFERENCE_FIXTURE = cast(
    dict[str, Any],
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
)

FLOAT64_RTOL = 0.0
FLOAT64_ATOL = 1e-15


def _build_preprocessing(
    *,
    vocabulary: tuple[str, ...] = (
        "alpha",
        "beta",
        "gamma",
    ),
    frequencies: tuple[int, ...] = (
        10,
        5,
        1,
    ),
    pairs: tuple[TrainingPair, ...] = (
        TrainingPair(
            target=0,
            context=1,
        ),
        TrainingPair(
            target=1,
            context=0,
        ),
        TrainingPair(
            target=0,
            context=2,
        ),
        TrainingPair(
            target=2,
            context=0,
        ),
    ),
) -> Word2VecPreprocessing:
    """Build small immutable preprocessing without using production derivation."""
    token_frequencies = MappingProxyType(
        dict(
            zip(
                vocabulary,
                frequencies,
                strict=True,
            )
        )
    )

    token_indices = MappingProxyType({token: index for index, token in enumerate(vocabulary)})

    return Word2VecPreprocessing(
        corpus=(" ".join(vocabulary),),
        training_text=" ".join(vocabulary),
        merge_limit=0,
        merges=(),
        tokenized_sentences=(vocabulary,),
        token_frequencies=token_frequencies,
        vocabulary=vocabulary,
        token_indices=token_indices,
        training_pairs=MappingProxyType(
            {
                1: pairs,
            }
        ),
    )


def _create_small_run() -> EmbeddingTrainingRun:
    """Create the independent fixture-sized deterministic Training Run."""
    fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["short_run"],
    )

    return EmbeddingTrainingRun(
        _build_preprocessing(),
        dimensions=cast(
            int,
            fixture["dimensions"],
        ),
        window_size=cast(
            int,
            fixture["window_size"],
        ),
        epochs=cast(
            int,
            fixture["epochs"],
        ),
        negative_samples=cast(
            int,
            fixture["negative_samples"],
        ),
    )


def _collect_short_run() -> tuple[
    list[EmbeddingEpochUpdate],
    CompletedEmbeddingTraining,
]:
    """Collect the public updates and one terminal numerical state."""
    events = list(_create_small_run())

    updates = [
        event
        for event in events
        if isinstance(
            event,
            EmbeddingEpochUpdate,
        )
    ]

    completions = [
        event
        for event in events
        if isinstance(
            event,
            CompletedEmbeddingTraining,
        )
    ]

    assert len(completions) == 1

    return updates, completions[0]


def test_mulberry32_matches_reference_sequences_and_wraparound() -> None:
    fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["mulberry32"],
    )

    for case_name, seed in (
        (
            "seed_42",
            42,
        ),
        (
            "wraparound_seed_4294967295",
            4_294_967_295,
        ),
    ):
        case = cast(
            dict[str, Any],
            fixture[case_name],
        )

        generator = Mulberry32(seed)

        expected_outputs = cast(
            list[float],
            case["outputs"],
        )

        assert [generator.random() for _ in expected_outputs] == expected_outputs

        assert generator.state == case["state_after"]
        assert generator.draw_count == len(expected_outputs)


def test_initialization_distribution_shuffle_and_sampling_match_reference() -> None:
    fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["small_state"],
    )

    initialization = cast(
        dict[str, Any],
        fixture["initialization"],
    )

    distribution = cast(
        dict[str, Any],
        fixture["negative_sampling"],
    )

    shuffle_fixture = cast(
        dict[str, Any],
        fixture["shuffle_after_initialization"],
    )

    sample_fixture = cast(
        dict[str, Any],
        fixture["samples_after_shuffle"],
    )

    run = _create_small_run()

    assert run.input_weights.dtype == np.float64
    assert run.output_weights.dtype == np.float64

    assert run.input_weights.shape == (
        3,
        2,
    )

    assert run.output_weights.shape == (
        3,
        2,
    )

    assert not np.shares_memory(
        run.input_weights,
        run.output_weights,
    )

    np.testing.assert_array_equal(
        run.input_weights,
        np.asarray(
            initialization["input_weights"],
            dtype=np.float64,
        ),
    )

    np.testing.assert_array_equal(
        run.output_weights,
        np.asarray(
            initialization["output_weights"],
            dtype=np.float64,
        ),
    )

    np.testing.assert_allclose(
        run.cumulative_distribution,
        np.asarray(
            distribution["cumulative"],
            dtype=np.float64,
        ),
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
    )

    assert run.random_generator.draw_count == initialization["draw_count_after"]

    assert run.random_generator.state == initialization["state_after"]

    run.shuffle_training_pairs()

    assert [
        [
            pair.target,
            pair.context,
        ]
        for pair in run.training_pairs
    ] == shuffle_fixture["pairs"]

    assert run.random_generator.draw_count == shuffle_fixture["draw_count_after"]

    assert run.random_generator.state == shuffle_fixture["state_after"]

    expected_indices = cast(
        list[int],
        sample_fixture["indices"],
    )

    assert [run.sample_negative() for _ in expected_indices] == expected_indices

    assert run.random_generator.draw_count == sample_fixture["draw_count_after"]

    assert run.random_generator.state == sample_fixture["state_after"]


def test_production_creation_owns_fresh_seed_42_state_without_numpy_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenRandom:
        def __getattr__(
            self,
            name: str,
        ) -> object:
            raise AssertionError(f"NumPy randomness was accessed through {name}")

    monkeypatch.setattr(
        word2vec_module.np,
        "random",
        ForbiddenRandom(),
    )

    factory_parameters = inspect.signature(create_embedding_training_run).parameters

    assert "seed" not in factory_parameters
    assert "preprocessing" not in factory_parameters

    first = create_embedding_training_run(
        dimensions=2,
        window_size=1,
        epochs=2,
        negative_samples=2,
    )

    second = create_embedding_training_run(
        dimensions=2,
        window_size=1,
        epochs=2,
        negative_samples=2,
    )

    assert first.random_generator is not second.random_generator

    assert first.random_generator.state == second.random_generator.state

    assert first.random_generator.draw_count == second.random_generator.draw_count

    assert first.training_pairs is not second.training_pairs

    assert not np.shares_memory(
        first.input_weights,
        second.input_weights,
    )

    assert not np.shares_memory(
        first.output_weights,
        second.output_weights,
    )

    np.testing.assert_array_equal(
        first.input_weights,
        second.input_weights,
    )

    np.testing.assert_array_equal(
        first.output_weights,
        second.output_weights,
    )


def test_positive_update_matches_independent_reference_transition() -> None:
    fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["positive_transition"],
    )

    run = _create_small_run()

    run.input_weights[:] = np.asarray(
        fixture["input_before"],
        dtype=np.float64,
    )

    run.output_weights[:] = np.asarray(
        fixture["output_before"],
        dtype=np.float64,
    )

    transition = run.apply_positive_update(
        target=cast(
            int,
            fixture["target"],
        ),
        context=cast(
            int,
            fixture["context"],
        ),
        learning_rate=cast(
            float,
            fixture["learning_rate"],
        ),
    )

    assert transition.score == pytest.approx(
        fixture["score"],
        abs=FLOAT64_ATOL,
    )

    assert transition.gradient == pytest.approx(
        fixture["gradient"],
        abs=FLOAT64_ATOL,
    )

    assert transition.loss == pytest.approx(
        fixture["loss"],
        abs=FLOAT64_ATOL,
    )

    np.testing.assert_allclose(
        run.input_weights,
        np.asarray(
            fixture["input_after"],
            dtype=np.float64,
        ),
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
    )

    np.testing.assert_allclose(
        run.output_weights,
        np.asarray(
            fixture["output_after"],
            dtype=np.float64,
        ),
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
    )


def test_negative_update_matches_independent_reference_transition() -> None:
    fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["negative_transition"],
    )

    run = _create_small_run()

    run.input_weights[:] = np.asarray(
        fixture["input_before"],
        dtype=np.float64,
    )

    run.output_weights[:] = np.asarray(
        fixture["output_before"],
        dtype=np.float64,
    )

    transition = run.apply_negative_update(
        target=cast(
            int,
            fixture["target"],
        ),
        negative=cast(
            int,
            fixture["negative"],
        ),
        learning_rate=cast(
            float,
            fixture["learning_rate"],
        ),
    )

    assert transition.score == pytest.approx(
        fixture["score"],
        abs=FLOAT64_ATOL,
    )

    assert transition.gradient == pytest.approx(
        fixture["gradient"],
        abs=FLOAT64_ATOL,
    )

    assert transition.loss == pytest.approx(
        fixture["loss"],
        abs=FLOAT64_ATOL,
    )

    np.testing.assert_allclose(
        run.input_weights,
        np.asarray(
            fixture["input_after"],
            dtype=np.float64,
        ),
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
    )

    np.testing.assert_allclose(
        run.output_weights,
        np.asarray(
            fixture["output_after"],
            dtype=np.float64,
        ),
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
    )


def test_sigmoid_clipping_matches_reference_boundaries() -> None:
    assert embedding_sigmoid(7.0) == 1.0
    assert embedding_sigmoid(-7.0) == 0.0
    assert embedding_sigmoid(0.0) == 0.5
    assert 0.0 < embedding_sigmoid(6.0) < 1.0
    assert 0.0 < embedding_sigmoid(-6.0) < 1.0


def test_true_context_collision_is_skipped_without_replacement_draw() -> None:
    preprocessing = _build_preprocessing(
        vocabulary=("only",),
        frequencies=(1,),
        pairs=(
            TrainingPair(
                target=0,
                context=0,
            ),
        ),
    )

    run = EmbeddingTrainingRun(
        preprocessing,
        dimensions=1,
        window_size=1,
        epochs=1,
        negative_samples=3,
    )

    events = list(run)

    assert [
        event.epoch
        for event in events
        if isinstance(
            event,
            EmbeddingEpochUpdate,
        )
    ] == [
        0,
        1,
    ]

    assert (
        sum(
            isinstance(
                event,
                CompletedEmbeddingTraining,
            )
            for event in events
        )
        == 1
    )

    # Two initialization draws plus three negative draws for each of two epochs.
    # All negative candidates collide with context zero and none are redrawn.
    assert run.random_generator.draw_count == 8


@pytest.mark.parametrize(
    (
        "epochs",
        "expected_epochs",
    ),
    [
        (
            4,
            [
                0,
                1,
                2,
                3,
                4,
            ],
        ),
        (
            100,
            list(
                range(
                    0,
                    101,
                    2,
                )
            ),
        ),
        (
            101,
            [
                *range(
                    0,
                    101,
                    2,
                ),
                101,
            ],
        ),
    ],
)
def test_inclusive_epoch_reporting_schedule_and_learning_rate(
    epochs: int,
    expected_epochs: list[int],
) -> None:
    preprocessing = _build_preprocessing(
        vocabulary=("only",),
        frequencies=(1,),
        pairs=(
            TrainingPair(
                target=0,
                context=0,
            ),
        ),
    )

    run = EmbeddingTrainingRun(
        preprocessing,
        dimensions=1,
        window_size=1,
        epochs=epochs,
        negative_samples=0,
    )

    events = list(run)

    updates = [
        event
        for event in events
        if isinstance(
            event,
            EmbeddingEpochUpdate,
        )
    ]

    assert [update.epoch for update in updates] == expected_epochs

    assert run.learning_rate_for_epoch(0) == 0.025

    assert run.learning_rate_for_epoch(epochs) == pytest.approx(0.001)

    assert (
        sum(
            isinstance(
                event,
                CompletedEmbeddingTraining,
            )
            for event in events
        )
        == 1
    )


def test_short_complete_run_matches_reference_updates_and_terminal_state() -> None:
    fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["short_run"],
    )

    updates, completion = _collect_short_run()

    expected_updates = cast(
        list[dict[str, Any]],
        fixture["updates"],
    )

    assert [update.to_payload() for update in updates] == [
        {
            "epoch": expected["epoch"],
            "loss": expected["loss"],
        }
        for expected in expected_updates
    ]

    assert completion.final_loss == pytest.approx(
        fixture["final_loss"],
        abs=FLOAT64_ATOL,
    )

    assert [
        [
            pair.target,
            pair.context,
        ]
        for pair in completion.training_pairs
    ] == fixture["final_training_pairs"]

    np.testing.assert_allclose(
        completion.input_weights,
        np.asarray(
            fixture["final_input_weights"],
            dtype=np.float64,
        ),
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
    )

    np.testing.assert_allclose(
        completion.output_weights,
        np.asarray(
            fixture["final_output_weights"],
            dtype=np.float64,
        ),
        rtol=FLOAT64_RTOL,
        atol=FLOAT64_ATOL,
    )


def test_non_finite_state_prevents_successful_completion() -> None:
    run = _create_small_run()

    run.input_weights[0, 0] = np.inf

    with pytest.raises(FloatingPointError):
        next(run)

    with pytest.raises(StopIteration):
        next(run)


def test_sequential_and_concurrent_runs_are_deterministic_and_isolated() -> None:
    first_updates, first_completion = _collect_short_run()
    second_updates, second_completion = _collect_short_run()

    assert [update.to_payload() for update in first_updates] == [
        update.to_payload() for update in second_updates
    ]

    assert first_completion.training_pairs is not second_completion.training_pairs

    assert not np.shares_memory(
        first_completion.input_weights,
        second_completion.input_weights,
    )

    assert not np.shares_memory(
        first_completion.output_weights,
        second_completion.output_weights,
    )

    np.testing.assert_array_equal(
        first_completion.input_weights,
        second_completion.input_weights,
    )

    np.testing.assert_array_equal(
        first_completion.output_weights,
        second_completion.output_weights,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_collect_short_run) for _ in range(2)]

        concurrent_results = [future.result() for future in futures]

    assert [[update.to_payload() for update in result[0]] for result in concurrent_results] == [
        [update.to_payload() for update in first_updates],
        [update.to_payload() for update in first_updates],
    ]

    for _updates, completion in concurrent_results:
        np.testing.assert_array_equal(
            completion.input_weights,
            first_completion.input_weights,
        )

        np.testing.assert_array_equal(
            completion.output_weights,
            first_completion.output_weights,
        )

        assert not np.shares_memory(
            completion.input_weights,
            first_completion.input_weights,
        )

        assert not np.shares_memory(
            completion.output_weights,
            first_completion.output_weights,
        )


def test_completed_state_mutation_does_not_change_other_runs_or_preprocessing() -> None:
    preprocessing = _build_preprocessing()

    first_run = EmbeddingTrainingRun(
        preprocessing,
        dimensions=2,
        window_size=1,
        epochs=2,
        negative_samples=2,
    )

    second_run = EmbeddingTrainingRun(
        preprocessing,
        dimensions=2,
        window_size=1,
        epochs=2,
        negative_samples=2,
    )

    first_completion = cast(
        CompletedEmbeddingTraining,
        list(first_run)[-1],
    )

    second_completion = cast(
        CompletedEmbeddingTraining,
        list(second_run)[-1],
    )

    second_weight = float(second_completion.input_weights[0, 0])

    original_pairs = preprocessing.training_pairs[1]

    first_completion.input_weights[0, 0] = 999.0
    first_completion.training_pairs.clear()

    assert float(second_completion.input_weights[0, 0]) == second_weight

    assert second_completion.training_pairs
    assert preprocessing.training_pairs[1] == original_pairs
