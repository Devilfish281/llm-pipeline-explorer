# tests/test_transformer_completion.py

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

import how_llms_work.ml.transformer as transformer_module
import numpy as np
import pytest
from how_llms_work.ml.math_utils import Mulberry32, round_typescript_decimal
from how_llms_work.ml.transformer import (
    EmptySavedTransformerPromptError,
    GeneratedTextSample,
    InitializedTransformerParameters,
    LogicalTrainingShardResult,
    PreparedSavedTransformerPrompt,
    SavedTransformerPromptTooLongError,
    TransformerPreprocessingSnapshot,
    TransformerTrainingRun,
    UnsupportedSavedTransformerPromptError,
    build_logical_training_shards,
    build_saved_transformer_model,
    build_transformer_parameter_layout,
    calculate_transformer_cross_entropy,
    calculate_transformer_forward,
    create_transformer_gradient_buffer,
    create_transformer_training_run,
    evaluate_transformer_final_loss,
    generate_saved_transformer_text,
    generate_transformer_text,
    get_transformer_preprocessing,
    initialize_transformer_parameters,
    prepare_saved_transformer_prompt,
)

_REFERENCE_PATH = Path(__file__).parent / "fixtures" / "transformer_completion_reference.json"

_REFERENCE: dict[str, Any] = json.loads(
    _REFERENCE_PATH.read_text(
        encoding="utf-8",
    )
)


class _CountingCancellationEvent:
    def __init__(
        self,
        cancel_at_call: int,
    ) -> None:
        self._cancel_at_call = cancel_at_call
        self.call_count = 0

    def is_set(self) -> bool:
        self.call_count += 1
        return self.call_count >= self._cancel_at_call


def _zero_initialized_parameters() -> InitializedTransformerParameters:
    layout = build_transformer_parameter_layout(1)

    initialized = initialize_transformer_parameters(
        layout,
        Mulberry32(42),
    )

    initialized.storage.fill(np.float32(0.0))

    return initialized


def _constant_logit_parameters() -> InitializedTransformerParameters:
    initialized = _zero_initialized_parameters()
    bias_reference = _REFERENCE["generation"]["headBias"]

    initialized.views.head_b.fill(np.float32(bias_reference["default"]))

    for token_id, value in bias_reference["selected"]:
        initialized.views.head_b[int(token_id)] = np.float32(value)

    return initialized


def _small_preprocessing(
    *,
    sequence_count: int = 1,
) -> TransformerPreprocessingSnapshot:
    preprocessing = get_transformer_preprocessing()

    sequences = preprocessing.training_sequences[:sequence_count]

    return replace(
        preprocessing,
        training_sequences=sequences,
        logical_training_shards=build_logical_training_shards(len(sequences)),
    )


def _zero_shard_results(
    run: TransformerTrainingRun,
    *,
    reported_loss: float = 0.0,
) -> tuple[LogicalTrainingShardResult, ...]:
    results: list[LogicalTrainingShardResult] = []

    for shard_index, shard in enumerate(run.logical_training_shards):
        results.append(
            LogicalTrainingShardResult(
                shard=shard,
                processed_sequence_count=(shard.stop_index - shard.start_index),
                loss=(reported_loss if shard_index == 0 else 0.0),
                gradient=create_transformer_gradient_buffer(run.parameters.layout),
            )
        )

    return tuple(results)


def _completed_zero_run(
    preprocessing: TransformerPreprocessingSnapshot,
    *,
    reported_loss: float = 0.0,
    selected_storage_values: list[float] | None = None,
    patterned_storage: bool = False,
) -> TransformerTrainingRun:
    initialized = _zero_initialized_parameters()

    if patterned_storage:
        coordinate_indices = np.arange(
            initialized.storage.size,
            dtype=np.int64,
        )

        patterned_values = ((coordinate_indices % 23) - 11) * 0.1234567

        initialized.storage[:] = np.asarray(
            patterned_values,
            dtype=np.float32,
        )

    if selected_storage_values is not None:
        initialized.storage[: len(selected_storage_values)] = np.asarray(
            selected_storage_values,
            dtype=np.float32,
        )

    run = create_transformer_training_run(
        initialized,
        sequence_count=len(preprocessing.training_sequences),
        requested_epochs=0,
    )

    run.advance_epoch(
        _zero_shard_results(
            run,
            reported_loss=reported_loss,
        )
    )

    return run


def _persistent_bytes(
    run: TransformerTrainingRun,
) -> tuple[
    bytes,
    bytes,
    bytes,
    bytes,
    tuple[object, ...],
]:
    return (
        run.weights.tobytes(),
        run.first_moments.tobytes(),
        run.second_moments.tobytes(),
        run.reduced_gradient.storage.tobytes(),
        run.updates,
    )


def _assert_plain_json_value(
    value: object,
) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            assert type(key) is str
            _assert_plain_json_value(nested)

        return

    if type(value) is list:
        for nested in value:
            _assert_plain_json_value(nested)

        return

    if type(value) is float:
        assert math.isfinite(value)

        assert (
            value != 0.0
            or math.copysign(
                1.0,
                value,
            )
            == 1.0
        )

        return

    assert type(value) in {
        str,
        int,
    }


def test_completion_fixture_has_independent_provenance() -> None:
    source = _REFERENCE["provenance"]["source"]

    assert "Independent scalar" in source
    assert "not produced by Ticket 018" in source


def test_saved_prompt_trims_outer_whitespace_and_preserves_supported_interior_text() -> None:
    vocabulary = [" a", ".", " b"]
    merges = [
        {
            "pair": [" ", "a"],
            "merged": " a",
        },
        {
            "pair": [" ", "b"],
            "merged": " b",
        },
    ]

    prepared = prepare_saved_transformer_prompt(
        " \t a. b \r\n",
        vocabulary,
        merges,
    )

    assert prepared == PreparedSavedTransformerPrompt(
        text="a. b",
        token_ids=(0, 1, 2),
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        " \t\r\n ",
    ],
)
def test_saved_prompt_rejects_empty_or_whitespace_only_text(
    prompt: str,
) -> None:
    with pytest.raises(EmptySavedTransformerPromptError):
        prepare_saved_transformer_prompt(
            prompt,
            [" a"],
            [
                {
                    "pair": [" ", "a"],
                    "merged": " a",
                }
            ],
        )


def test_saved_prompt_accepts_one_through_sixteen_duplicate_model_tokens() -> None:
    vocabulary = [" a"]
    merges = [
        {
            "pair": [" ", "a"],
            "merged": " a",
        }
    ]

    one_token = prepare_saved_transformer_prompt(
        "a",
        vocabulary,
        merges,
    )
    sixteen_tokens = prepare_saved_transformer_prompt(
        " ".join(["a"] * 16),
        vocabulary,
        merges,
    )

    assert one_token.token_ids == (0,)
    assert sixteen_tokens.token_ids == (0,) * 16


@pytest.mark.parametrize(
    "prompt",
    [
        "A",
        "a  a",
        "a_",
    ],
)
def test_saved_prompt_rejects_unsupported_text_without_normalizing_or_dropping(
    prompt: str,
) -> None:
    with pytest.raises(UnsupportedSavedTransformerPromptError):
        prepare_saved_transformer_prompt(
            prompt,
            [" a"],
            [
                {
                    "pair": [" ", "a"],
                    "merged": " a",
                }
            ],
        )


def test_saved_prompt_rejects_seventeen_model_tokens() -> None:
    with pytest.raises(SavedTransformerPromptTooLongError):
        prepare_saved_transformer_prompt(
            " ".join(["a"] * 17),
            [" a"],
            [
                {
                    "pair": [" ", "a"],
                    "merged": " a",
                }
            ],
        )


def test_saved_prompt_preserves_merge_order_and_vocabulary_id_order() -> None:
    ordered_merges = [
        {
            "pair": [" ", "a"],
            "merged": " a",
        },
        {
            "pair": [" a", "b"],
            "merged": " ab",
        },
        {
            "pair": [" ab", "c"],
            "merged": " abc",
        },
    ]
    reordered_merges = [
        {
            "pair": [" a", "b"],
            "merged": " ab",
        },
        {
            "pair": [" ", "a"],
            "merged": " a",
        },
        {
            "pair": [" ab", "c"],
            "merged": " abc",
        },
    ]
    word_merges = [
        {
            "pair": [" ", "a"],
            "merged": " a",
        },
        {
            "pair": [" ", "b"],
            "merged": " b",
        },
    ]

    ordered = prepare_saved_transformer_prompt(
        "abc",
        [" abc"],
        ordered_merges,
    )
    reversed_vocabulary = prepare_saved_transformer_prompt(
        "a b",
        [" b", " a"],
        word_merges,
    )

    assert ordered.token_ids == (0,)
    assert reversed_vocabulary.token_ids == (1, 0)

    with pytest.raises(UnsupportedSavedTransformerPromptError):
        prepare_saved_transformer_prompt(
            "abc",
            [" abc"],
            reordered_merges,
        )


def test_saved_prompt_does_not_mutate_or_reuse_loaded_model_metadata() -> None:
    vocabulary = [" a", " b"]
    merges = [
        {
            "pair": [" ", "a"],
            "merged": " a",
        },
        {
            "pair": [" ", "b"],
            "merged": " b",
        },
    ]
    original_vocabulary = list(vocabulary)
    original_merges = json.loads(json.dumps(merges))

    first = prepare_saved_transformer_prompt(
        "a b",
        vocabulary,
        merges,
    )
    second = prepare_saved_transformer_prompt(
        "a b",
        vocabulary,
        merges,
    )

    assert first == PreparedSavedTransformerPrompt(
        text="a b",
        token_ids=(0, 1),
    )
    assert second == first
    assert second is not first
    assert vocabulary == original_vocabulary
    assert merges == original_merges
    assert isinstance(first.token_ids, tuple)

    vocabulary.reverse()
    merges[0]["pair"][0] = "x"

    assert first == PreparedSavedTransformerPrompt(
        text="a b",
        token_ids=(0, 1),
    )


def test_saved_generation_uses_seed_42_and_preserves_exact_prompt_prefix() -> None:
    preprocessing = get_transformer_preprocessing()
    parameters = _constant_logit_parameters()
    prepared_prompt = PreparedSavedTransformerPrompt(
        text="once upon a",
        token_ids=preprocessing.generation_seed_ids,
    )
    generation_reference = _REFERENCE["generation"]
    before = parameters.storage.tobytes()

    result = generate_saved_transformer_text(
        parameters,
        preprocessing.vocabulary,
        prepared_prompt,
        temperature=float(generation_reference["temperature"]),
        top_p=float(generation_reference["topP"]),
        max_tokens=int(generation_reference["maxTokens"]),
        cancellation_event=Event(),
    )

    reconstructed_seed = "".join(
        preprocessing.vocabulary[token_id] for token_id in preprocessing.generation_seed_ids
    )
    training_reference = str(generation_reference["expected"]["0"])

    assert training_reference.startswith(reconstructed_seed)

    expected_continuation = training_reference[len(reconstructed_seed) :]

    assert result == f"{prepared_prompt.text}{expected_continuation}"
    assert result.encode("utf-8").startswith(prepared_prompt.text.encode("utf-8"))
    assert parameters.storage.tobytes() == before


def test_saved_generation_uses_latest_sixteen_context_and_sampling_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessing = get_transformer_preprocessing()
    parameters = _constant_logit_parameters()
    prompt_token_ids = tuple(range(16))
    prepared_prompt = PreparedSavedTransformerPrompt(
        text="exact prompt",
        token_ids=prompt_token_ids,
    )
    temperature = 0.5
    top_p = 0.6
    max_tokens = 4

    observed_contexts: list[tuple[int, ...]] = []
    observed_scaled_rows: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    observed_top_p: list[float] = []
    observed_generator_states: list[tuple[int, int]] = []
    sampled_token_ids: list[int] = []

    original_forward = transformer_module.calculate_transformer_forward
    original_softmax = transformer_module.stable_row_softmax
    original_sampler = transformer_module._sample_transformer_nucleus_token

    def recording_forward(
        input_ids: tuple[int, ...],
        parameter_views: transformer_module.TransformerParameterViews,
    ) -> transformer_module.TransformerForwardResult:
        observed_contexts.append(input_ids)

        return original_forward(
            input_ids,
            parameter_views,
        )

    def recording_softmax(
        scores: np.ndarray[Any, np.dtype[np.float32]],
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        if scores.shape == (
            1,
            parameters.layout.vocabulary_size,
        ):
            observed_scaled_rows.append(scores.copy())

        return original_softmax(scores)

    def recording_sampler(
        probabilities: np.ndarray[Any, np.dtype[np.float32]],
        *,
        top_p: float,
        generator: Mulberry32,
    ) -> int:
        observed_top_p.append(top_p)
        observed_generator_states.append(
            (
                generator.state,
                generator.draw_count,
            )
        )

        token_id = original_sampler(
            probabilities,
            top_p=top_p,
            generator=generator,
        )
        sampled_token_ids.append(token_id)

        return token_id

    monkeypatch.setattr(
        transformer_module,
        "calculate_transformer_forward",
        recording_forward,
    )
    monkeypatch.setattr(
        transformer_module,
        "stable_row_softmax",
        recording_softmax,
    )
    monkeypatch.setattr(
        transformer_module,
        "_sample_transformer_nucleus_token",
        recording_sampler,
    )

    result = generate_saved_transformer_text(
        parameters,
        preprocessing.vocabulary,
        prepared_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        cancellation_event=Event(),
    )

    assert len(observed_contexts) == max_tokens
    assert len(observed_scaled_rows) == max_tokens
    assert len(sampled_token_ids) == max_tokens
    assert observed_top_p == [top_p] * max_tokens
    assert observed_generator_states[0] == (42, 0)

    accumulated_ids = list(prompt_token_ids)

    for context_ids, sampled_token_id in zip(
        observed_contexts,
        sampled_token_ids,
        strict=True,
    ):
        assert context_ids == tuple(accumulated_ids[-16:])
        accumulated_ids.append(sampled_token_id)

    expected_scaled_logits = (
        np.asarray(
            parameters.views.head_b,
            dtype=np.float64,
        )
        / temperature
    )
    expected_scaled_row = np.asarray(
        expected_scaled_logits,
        dtype=np.float32,
    ).reshape(
        1,
        parameters.layout.vocabulary_size,
    )

    np.testing.assert_array_equal(
        observed_scaled_rows[0],
        expected_scaled_row,
    )

    expected_continuation = "".join(
        preprocessing.vocabulary[token_id] for token_id in sampled_token_ids
    )

    assert result == f"{prepared_prompt.text}{expected_continuation}"


def test_saved_generation_isolated_from_different_prior_calls() -> None:
    preprocessing = get_transformer_preprocessing()
    parameters = _constant_logit_parameters()
    prepared_prompt = PreparedSavedTransformerPrompt(
        text="once upon a",
        token_ids=preprocessing.generation_seed_ids,
    )

    first = generate_saved_transformer_text(
        parameters,
        preprocessing.vocabulary,
        prepared_prompt,
        temperature=0.8,
        top_p=0.9,
        max_tokens=5,
        cancellation_event=Event(),
    )

    generate_saved_transformer_text(
        parameters,
        preprocessing.vocabulary,
        prepared_prompt,
        temperature=1.2,
        top_p=0.4,
        max_tokens=3,
        cancellation_event=Event(),
    )

    second = generate_saved_transformer_text(
        parameters,
        preprocessing.vocabulary,
        prepared_prompt,
        temperature=0.8,
        top_p=0.9,
        max_tokens=5,
        cancellation_event=Event(),
    )

    assert second == first


@pytest.mark.parametrize(
    (
        "keyword",
        "value",
        "error_type",
    ),
    [
        ("temperature", 1, TypeError),
        ("temperature", math.inf, FloatingPointError),
        ("temperature", 0.09, ValueError),
        ("temperature", 2.01, ValueError),
        ("top_p", 1, TypeError),
        ("top_p", math.nan, FloatingPointError),
        ("top_p", 0.09, ValueError),
        ("top_p", 1.01, ValueError),
        ("max_tokens", True, TypeError),
        ("max_tokens", 2, ValueError),
        ("max_tokens", 501, ValueError),
    ],
)
def test_saved_generation_strictly_validates_public_arguments(
    keyword: str,
    value: object,
    error_type: type[Exception],
) -> None:
    preprocessing = get_transformer_preprocessing()
    arguments: dict[str, object] = {
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 3,
    }
    arguments[keyword] = value

    with pytest.raises(error_type):
        generate_saved_transformer_text(
            _constant_logit_parameters(),
            preprocessing.vocabulary,
            PreparedSavedTransformerPrompt(
                text="once upon a",
                token_ids=preprocessing.generation_seed_ids,
            ),
            temperature=arguments["temperature"],
            top_p=arguments["top_p"],
            max_tokens=arguments["max_tokens"],
            cancellation_event=Event(),
        )


def test_saved_generation_rejects_incompatible_or_non_finite_model_state() -> None:
    preprocessing = get_transformer_preprocessing()
    prepared_prompt = PreparedSavedTransformerPrompt(
        text="once upon a",
        token_ids=preprocessing.generation_seed_ids,
    )

    with pytest.raises(ValueError):
        generate_saved_transformer_text(
            _constant_logit_parameters(),
            preprocessing.vocabulary[:-1],
            prepared_prompt,
            temperature=0.8,
            top_p=0.9,
            max_tokens=3,
            cancellation_event=Event(),
        )

    non_finite_parameters = _constant_logit_parameters()
    non_finite_parameters.storage[0] = np.float32(np.nan)

    with pytest.raises(FloatingPointError):
        generate_saved_transformer_text(
            non_finite_parameters,
            preprocessing.vocabulary,
            prepared_prompt,
            temperature=0.8,
            top_p=0.9,
            max_tokens=3,
            cancellation_event=Event(),
        )

    overflowing_parameters = _zero_initialized_parameters()
    overflowing_parameters.views.head_b.fill(np.finfo(np.float32).max)

    with pytest.raises(FloatingPointError):
        generate_saved_transformer_text(
            overflowing_parameters,
            preprocessing.vocabulary,
            prepared_prompt,
            temperature=0.1,
            top_p=0.9,
            max_tokens=3,
            cancellation_event=Event(),
        )


@pytest.mark.parametrize(
    "epoch",
    [
        0,
        1,
        7,
    ],
)
def test_generation_matches_controlled_exact_reference(
    epoch: int,
) -> None:
    preprocessing = get_transformer_preprocessing()
    parameters = _constant_logit_parameters()
    generation_reference = _REFERENCE["generation"]
    before = parameters.storage.tobytes()

    sample = generate_transformer_text(
        parameters,
        preprocessing,
        epoch=epoch,
        temperature=float(generation_reference["temperature"]),
        top_p=float(generation_reference["topP"]),
        max_tokens=int(generation_reference["maxTokens"]),
        cancellation_event=Event(),
    )

    assert sample == GeneratedTextSample(
        epoch=epoch,
        text=generation_reference["expected"][str(epoch)],
    )

    assert parameters.storage.tobytes() == before


def test_generation_uses_latest_sixteen_ids_and_one_fresh_epoch_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessing = get_transformer_preprocessing()
    parameters = _constant_logit_parameters()
    observed_contexts: list[tuple[int, ...]] = []

    original_forward = transformer_module.calculate_transformer_forward

    def recording_forward(
        input_ids: tuple[int, ...],
        parameter_views: transformer_module.TransformerParameterViews,
    ) -> transformer_module.TransformerForwardResult:
        observed_contexts.append(input_ids)

        return original_forward(
            input_ids,
            parameter_views,
        )

    monkeypatch.setattr(
        transformer_module,
        "calculate_transformer_forward",
        recording_forward,
    )

    first = generate_transformer_text(
        parameters,
        preprocessing,
        epoch=3,
        temperature=0.8,
        top_p=0.9,
        max_tokens=20,
        cancellation_event=Event(),
    )

    first_contexts = tuple(observed_contexts)
    observed_contexts.clear()

    second = generate_transformer_text(
        parameters,
        preprocessing,
        epoch=3,
        temperature=0.8,
        top_p=0.9,
        max_tokens=20,
        cancellation_event=Event(),
    )

    assert first == second

    assert tuple(len(context) for context in first_contexts) == (tuple(range(3, 17)) + (16,) * 6)

    assert tuple(observed_contexts) == first_contexts
    assert first_contexts[-1] != first_contexts[-2]


@pytest.mark.parametrize(
    (
        "keyword",
        "value",
        "error_type",
    ),
    [
        ("epoch", True, TypeError),
        ("epoch", -1, ValueError),
        ("temperature", 1, TypeError),
        (
            "temperature",
            math.inf,
            FloatingPointError,
        ),
        ("temperature", 0.09, ValueError),
        ("temperature", 2.01, ValueError),
        ("top_p", 1, TypeError),
        (
            "top_p",
            math.nan,
            FloatingPointError,
        ),
        ("top_p", 0.09, ValueError),
        ("top_p", 1.01, ValueError),
        ("max_tokens", True, TypeError),
        ("max_tokens", 2, ValueError),
        ("max_tokens", 501, ValueError),
    ],
)
def test_generation_strictly_validates_public_arguments(
    keyword: str,
    value: object,
    error_type: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "epoch": 0,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 3,
    }

    arguments[keyword] = value

    with pytest.raises(error_type):
        generate_transformer_text(
            _constant_logit_parameters(),
            get_transformer_preprocessing(),
            epoch=arguments["epoch"],
            temperature=arguments["temperature"],
            top_p=arguments["top_p"],
            max_tokens=arguments["max_tokens"],
            cancellation_event=Event(),
        )


def test_generation_cancellation_before_between_and_after_tokens_prevents_success() -> None:
    preprocessing = get_transformer_preprocessing()
    parameters = _constant_logit_parameters()

    for cancel_at_call in (
        1,
        2,
        4,
    ):
        cancellation = _CountingCancellationEvent(cancel_at_call)

        with pytest.raises(
            RuntimeError,
            match="cancelled",
        ):
            generate_transformer_text(
                parameters,
                preprocessing,
                epoch=0,
                temperature=0.8,
                top_p=0.9,
                max_tokens=3,
                cancellation_event=cancellation,
            )


def test_final_loss_uses_final_parameters_and_not_last_completed_loss() -> None:
    preprocessing = _small_preprocessing(sequence_count=1)

    run = _completed_zero_run(
        preprocessing,
        reported_loss=123.0,
    )

    before = _persistent_bytes(run)

    final_loss = evaluate_transformer_final_loss(
        run,
        preprocessing,
        cancellation_event=Event(),
    )

    assert final_loss == _REFERENCE["finalLoss"]["expected"]

    assert final_loss != run.last_completed_loss
    assert _persistent_bytes(run) == before

    target_id = preprocessing.training_sequences[0].target_ids[0]

    run.parameters.views.head_b[target_id] = np.float32(2.0)

    changed_loss = evaluate_transformer_final_loss(
        run,
        preprocessing,
        cancellation_event=Event(),
    )

    assert changed_loss != final_loss
    assert run.last_completed_loss == 123.0


def test_final_loss_matches_direct_fixed_order_public_calculation() -> None:
    preprocessing = _small_preprocessing(sequence_count=2)

    run = _completed_zero_run(preprocessing)
    expected_sum = 0.0

    for sequence in preprocessing.training_sequences:
        forward = calculate_transformer_forward(
            sequence.input_ids,
            run.parameters.views,
        )

        expected_sum += calculate_transformer_cross_entropy(
            forward.probabilities,
            sequence.target_ids,
        )

    expected = round_typescript_decimal(
        expected_sum / len(preprocessing.training_sequences),
        6,
    )

    actual = evaluate_transformer_final_loss(
        run,
        preprocessing,
        cancellation_event=Event(),
    )

    assert actual == expected


def test_final_loss_rejects_incomplete_failed_empty_and_cancelled_runs() -> None:
    preprocessing = _small_preprocessing(sequence_count=2)

    incomplete = create_transformer_training_run(
        _zero_initialized_parameters(),
        sequence_count=2,
        requested_epochs=0,
    )

    with pytest.raises(
        RuntimeError,
        match="complete",
    ):
        evaluate_transformer_final_loss(
            incomplete,
            preprocessing,
            cancellation_event=Event(),
        )

    failed = create_transformer_training_run(
        _zero_initialized_parameters(),
        sequence_count=2,
        requested_epochs=0,
    )

    with pytest.raises(ValueError):
        failed.advance_epoch(())

    with pytest.raises(
        RuntimeError,
        match="failed",
    ):
        evaluate_transformer_final_loss(
            failed,
            preprocessing,
            cancellation_event=Event(),
        )

    empty_preprocessing = _small_preprocessing(sequence_count=0)

    empty_run = _completed_zero_run(empty_preprocessing)

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        evaluate_transformer_final_loss(
            empty_run,
            empty_preprocessing,
            cancellation_event=Event(),
        )

    run = _completed_zero_run(preprocessing)

    for cancel_at_call in (
        1,
        2,
        3,
    ):
        with pytest.raises(
            RuntimeError,
            match="cancelled",
        ):
            evaluate_transformer_final_loss(
                run,
                preprocessing,
                cancellation_event=(_CountingCancellationEvent(cancel_at_call)),
            )


def test_saved_model_has_exact_order_complete_lengths_and_plain_values() -> None:
    preprocessing = _small_preprocessing(sequence_count=1)

    model_reference = _REFERENCE["savedModel"]

    run = _completed_zero_run(
        preprocessing,
        selected_storage_values=(model_reference["selectedStorageValues"]),
        patterned_storage=True,
    )

    before = _persistent_bytes(run)

    model = build_saved_transformer_model(
        run,
        preprocessing,
    )

    assert list(model) == model_reference["topLevelKeys"]

    assert model["type"] == "decoder-transformer"

    assert list(model["config"]) == (model_reference["configKeys"])

    assert model["config"] == {
        "vocabSize": len(preprocessing.vocabulary),
        "contextLen": 32,
        "embDim": 32,
        "numHeads": 2,
        "ffDim": 128,
        "numLayers": 1,
    }

    assert model["vocab"] == list(preprocessing.vocabulary)

    assert model["merges"] == [
        {
            "pair": list(merge.pair),
            "merged": merge.merged,
        }
        for merge in preprocessing.merges
    ]

    assert list(model["weights"]) == (model_reference["weightKeys"])

    assert len(model["weights"]["blocks"]) == 1

    assert list(model["weights"]["blocks"][0]) == model_reference["blockKeys"]

    layout = run.parameters.layout

    assert len(model["weights"]["tokEmb"]) == layout.get_record("tokEmb").length

    assert len(model["weights"]["posEmb"]) == layout.get_record("posEmb").length

    assert len(model["weights"]["lnFGamma"]) == layout.get_record("lnFGamma").length

    assert len(model["weights"]["lnFBeta"]) == layout.get_record("lnFBeta").length

    assert len(model["weights"]["headW"]) == layout.get_record("headW").length

    assert len(model["weights"]["headB"]) == layout.get_record("headB").length

    for key in model_reference["blockKeys"]:
        assert (
            len(model["weights"]["blocks"][0][key])
            == layout.get_record(
                key,
                0,
            ).length
        )

    expected_top_level = {
        "tokEmb": run.parameters.views.tok_emb,
        "posEmb": run.parameters.views.pos_emb,
        "lnFGamma": run.parameters.views.ln_f_gamma,
        "lnFBeta": run.parameters.views.ln_f_beta,
        "headW": run.parameters.views.head_w,
        "headB": run.parameters.views.head_b,
    }

    for key, values in expected_top_level.items():
        expected_values = [
            round_typescript_decimal(
                float(value),
                6,
            )
            for value in values.reshape(
                -1,
                order="C",
            )
        ]

        assert model["weights"][key] == (expected_values)

    block = run.parameters.views.blocks[0]

    for key in model_reference["blockKeys"]:
        expected_values = [
            round_typescript_decimal(
                float(value),
                6,
            )
            for value in block.get(key).reshape(
                -1,
                order="C",
            )
        ]

        assert model["weights"]["blocks"][0][key] == expected_values

    assert model["weights"]["tokEmb"][:3] == (model_reference["expectedPublicValues"])

    _assert_plain_json_value(model)

    json.dumps(
        model,
        allow_nan=False,
    )

    assert _persistent_bytes(run) == before


def test_saved_model_is_fresh_and_excludes_transient_state() -> None:
    preprocessing = _small_preprocessing(sequence_count=1)

    run = _completed_zero_run(preprocessing)

    first = build_saved_transformer_model(
        run,
        preprocessing,
    )

    second = build_saved_transformer_model(
        run,
        preprocessing,
    )

    forbidden = {
        "optimizer",
        "gradient",
        "moments",
        "cache",
        "sharedMemory",
        "process",
        "path",
        "timestamp",
        "request",
        "checkpoint",
    }

    assert first == second
    assert first is not second
    assert first["config"] is not second["config"]
    assert first["vocab"] is not second["vocab"]
    assert first["merges"] is not second["merges"]
    assert first["weights"] is not second["weights"]

    assert first["weights"]["blocks"] is not second["weights"]["blocks"]

    assert forbidden.isdisjoint(first)
    assert forbidden.isdisjoint(first["weights"])

    first["vocab"][0] = "mutated"

    first["merges"][0]["pair"][0] = "mutated"

    first["weights"]["tokEmb"][0] = 999.0

    third = build_saved_transformer_model(
        run,
        preprocessing,
    )

    assert third == second


def test_completion_operations_reject_non_finite_parameter_state() -> None:
    preprocessing = _small_preprocessing(sequence_count=1)

    parameters = _constant_logit_parameters()

    parameters.storage[0] = np.float32(np.nan)

    with pytest.raises(FloatingPointError):
        generate_transformer_text(
            parameters,
            preprocessing,
            epoch=0,
            temperature=0.8,
            top_p=0.9,
            max_tokens=3,
            cancellation_event=Event(),
        )

    run = _completed_zero_run(preprocessing)

    run.weights[0] = np.float32(np.inf)

    with pytest.raises(FloatingPointError):
        evaluate_transformer_final_loss(
            run,
            preprocessing,
            cancellation_event=Event(),
        )

    with pytest.raises(FloatingPointError):
        build_saved_transformer_model(
            run,
            preprocessing,
        )


def test_repeated_and_concurrent_completion_operations_are_exact() -> None:
    preprocessing = _small_preprocessing(sequence_count=1)

    parameters = _constant_logit_parameters()
    run = _completed_zero_run(preprocessing)

    with ThreadPoolExecutor(max_workers=6) as executor:
        generated = tuple(
            executor.map(
                lambda _: generate_transformer_text(
                    parameters,
                    preprocessing,
                    epoch=1,
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=5,
                    cancellation_event=Event(),
                ),
                range(6),
            )
        )

        losses = tuple(
            executor.map(
                lambda _: evaluate_transformer_final_loss(
                    run,
                    preprocessing,
                    cancellation_event=Event(),
                ),
                range(6),
            )
        )

        models = tuple(
            executor.map(
                lambda _: build_saved_transformer_model(
                    run,
                    preprocessing,
                ),
                range(6),
            )
        )

    assert len(set(generated)) == 1
    assert len(set(losses)) == 1

    assert all(model == models[0] for model in models)

    assert all(model is not models[0] for model in models[1:])
