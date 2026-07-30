# tests/test_transformer_math.py
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from how_llms_work.ml.math_utils import Mulberry32
from how_llms_work.ml.transformer import (
    TRANSFORMER_ATTENTION_HEAD_COUNT,
    TRANSFORMER_ATTENTION_SCALE,
    TRANSFORMER_CONTEXT_LENGTH,
    TRANSFORMER_CROSS_ENTROPY_EPSILON,
    TRANSFORMER_EMBEDDING_DIMENSION,
    TRANSFORMER_FEED_FORWARD_DIMENSION,
    TRANSFORMER_HEAD_DIMENSION,
    TRANSFORMER_LAYER_NORMALIZATION_EPSILON,
    LogicalTrainingShard,
    TransformerForwardResult,
    TransformerParameterViews,
    TransformerTrainingSequence,
    build_logical_training_shards,
    build_transformer_parameter_layout,
    calculate_logical_training_shard,
    calculate_transformer_backward,
    calculate_transformer_cross_entropy,
    calculate_transformer_forward,
    calculate_transformer_sequence,
    create_transformer_gradient_buffer,
    initialize_transformer_parameters,
)
from numpy.testing import assert_allclose, assert_array_equal

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "transformer_forward_backward_reference.json"


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parameters() -> TransformerParameterViews:
    layout = build_transformer_parameter_layout(1)
    initialized = initialize_transformer_parameters(
        layout,
        Mulberry32(42),
    )
    return initialized.views


def _index(
    array: np.ndarray[Any, np.dtype[np.float32]],
    coordinates: list[int],
) -> float:
    return float(array[tuple(coordinates)])


def _assert_selected(
    array: np.ndarray[Any, np.dtype[np.float32]],
    selected: list[dict[str, Any]],
    *,
    rtol: float,
    atol: float,
) -> None:
    for item in selected:
        assert_allclose(
            _index(array, item["index"]),
            float(item["value"]),
            rtol=rtol,
            atol=atol,
        )


def _ordinary_forward(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> TransformerForwardResult:
    ordinary = reference["ordinary"]

    return calculate_transformer_forward(
        tuple(ordinary["input_ids"]),
        parameters,
    )


def test_fixture_records_independent_provenance(
    reference: dict[str, Any],
) -> None:
    provenance = reference["provenance"]

    assert provenance["production_module_imported"] is False
    assert provenance["production_forward_or_backward_called"] is False
    assert "Independent" in provenance["source"]


def test_transformer_math_constants_match_reference(
    reference: dict[str, Any],
) -> None:
    architecture = reference["architecture"]

    assert TRANSFORMER_CONTEXT_LENGTH == architecture["context_length"]
    assert TRANSFORMER_EMBEDDING_DIMENSION == architecture["embedding_dimension"]
    assert TRANSFORMER_ATTENTION_HEAD_COUNT == architecture["attention_head_count"]
    assert TRANSFORMER_HEAD_DIMENSION == architecture["head_dimension"]
    assert TRANSFORMER_ATTENTION_SCALE == architecture["attention_scale"]
    assert TRANSFORMER_FEED_FORWARD_DIMENSION == architecture["feed_forward_dimension"]
    assert TRANSFORMER_LAYER_NORMALIZATION_EPSILON == architecture["layer_normalization_epsilon"]
    assert TRANSFORMER_CROSS_ENTROPY_EPSILON == architecture["cross_entropy_epsilon"]


def test_forward_matches_selected_independent_values(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    tolerance = reference["tolerances"]["activation"]
    probability_tolerance = reference["tolerances"]["probability"]

    result = _ordinary_forward(
        reference,
        parameters,
    )
    block = result.blocks[0]

    _assert_selected(
        result.embedding_activation,
        ordinary["embedding"],
        **tolerance,
    )
    _assert_selected(
        block.first_normalization.means,
        ordinary["layer_norm_1"]["mean"],
        **tolerance,
    )
    _assert_selected(
        block.first_normalization.variances,
        ordinary["layer_norm_1"]["variance"],
        **tolerance,
    )
    _assert_selected(
        block.first_normalization.normalized,
        ordinary["layer_norm_1"]["normalized"],
        **tolerance,
    )
    _assert_selected(
        block.query,
        ordinary["query"],
        **tolerance,
    )
    _assert_selected(
        block.key,
        ordinary["key"],
        **tolerance,
    )
    _assert_selected(
        block.value,
        ordinary["value"],
        **tolerance,
    )
    _assert_selected(
        block.attention_heads[0].scores,
        ordinary["attention"]["head_0_scores"],
        **tolerance,
    )
    _assert_selected(
        block.attention_heads[1].scores,
        ordinary["attention"]["head_1_scores"],
        **tolerance,
    )
    _assert_selected(
        block.attention_heads[0].probabilities,
        ordinary["attention"]["head_0_probabilities"],
        **probability_tolerance,
    )
    _assert_selected(
        block.attention_heads[1].probabilities,
        ordinary["attention"]["head_1_probabilities"],
        **probability_tolerance,
    )
    _assert_selected(
        block.projected_attention,
        ordinary["attention_output"],
        **tolerance,
    )
    _assert_selected(
        block.first_residual,
        ordinary["first_residual"],
        **tolerance,
    )
    _assert_selected(
        block.second_normalization.output,
        ordinary["layer_norm_2"],
        **tolerance,
    )
    _assert_selected(
        block.feed_forward_pre_activation,
        ordinary["feed_forward_pre_activation"],
        **tolerance,
    )
    _assert_selected(
        block.feed_forward_activation,
        ordinary["feed_forward_activation"],
        **tolerance,
    )
    _assert_selected(
        block.feed_forward_output,
        ordinary["feed_forward_output"],
        **tolerance,
    )
    _assert_selected(
        block.output,
        ordinary["block_output"],
        **tolerance,
    )
    _assert_selected(
        result.final_normalization.output,
        ordinary["final_normalization"],
        **tolerance,
    )
    _assert_selected(
        result.logits,
        ordinary["logits"],
        **tolerance,
    )
    _assert_selected(
        result.probabilities,
        ordinary["probabilities"],
        **probability_tolerance,
    )


def test_forward_outputs_have_strict_float32_contract(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    result = _ordinary_forward(
        reference,
        parameters,
    )

    arrays = [
        result.embedding_activation,
        result.final_normalization.means,
        result.final_normalization.variances,
        result.final_normalization.normalized,
        result.final_normalization.output,
        result.logits,
        result.probabilities,
    ]

    for block in result.blocks:
        arrays.extend(
            (
                block.input_activation,
                block.first_normalization.output,
                block.query,
                block.key,
                block.value,
                block.concatenated_attention,
                block.projected_attention,
                block.first_residual,
                block.second_normalization.output,
                block.feed_forward_pre_activation,
                block.feed_forward_activation,
                block.feed_forward_output,
                block.output,
            )
        )

        for head in block.attention_heads:
            arrays.extend(
                (
                    head.probabilities,
                    head.weighted_values,
                )
            )

    for array in arrays:
        assert array.dtype == np.dtype(np.float32)
        assert array.flags.c_contiguous
        assert np.isfinite(array).all()


def test_causal_attention_masks_future_positions_exactly(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    result = _ordinary_forward(
        reference,
        parameters,
    )
    sequence_length = len(result.input_ids)
    future_mask = np.triu(
        np.ones(
            (
                sequence_length,
                sequence_length,
            ),
            dtype=np.bool_,
        ),
        k=1,
    )

    for block in result.blocks:
        for head in block.attention_heads:
            assert np.isneginf(head.scores[future_mask]).all()
            assert_array_equal(
                head.probabilities[future_mask],
                np.zeros(
                    int(np.count_nonzero(future_mask)),
                    dtype=np.float32,
                ),
            )
            assert_allclose(
                np.sum(
                    head.probabilities,
                    axis=1,
                    dtype=np.float64,
                ),
                np.ones(sequence_length),
                rtol=0.0,
                atol=1e-6,
            )


def test_future_token_changes_do_not_change_earlier_attention(
    parameters: TransformerParameterViews,
) -> None:
    first = calculate_transformer_forward(
        (0, 1, 2, 3),
        parameters,
    )
    second = calculate_transformer_forward(
        (0, 1, 2, 17),
        parameters,
    )

    for first_head, second_head in zip(
        first.blocks[0].attention_heads,
        second.blocks[0].attention_heads,
        strict=True,
    ):
        assert_array_equal(
            first_head.probabilities[:3, :3],
            second_head.probabilities[:3, :3],
        )
        assert_array_equal(
            first_head.weighted_values[:3],
            second_head.weighted_values[:3],
        )


def test_cross_entropy_matches_independent_average_loss(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    tolerance = reference["tolerances"]["loss"]
    result = _ordinary_forward(
        reference,
        parameters,
    )

    actual = calculate_transformer_cross_entropy(
        result.probabilities,
        tuple(ordinary["target_ids"]),
    )

    assert_allclose(
        actual,
        ordinary["loss"],
        **tolerance,
    )


def test_backward_matches_selected_independent_gradients(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    tolerance = reference["tolerances"]["gradient"]
    forward = _ordinary_forward(
        reference,
        parameters,
    )
    result = calculate_transformer_backward(
        forward,
        tuple(ordinary["target_ids"]),
        parameters,
    )
    gradients = ordinary["gradients"]
    block = result.gradient.views.blocks[0]

    assert_allclose(
        result.loss,
        ordinary["loss"],
        **reference["tolerances"]["loss"],
    )
    _assert_selected(
        result.gradient.views.head_w,
        gradients["head_w"],
        **tolerance,
    )
    _assert_selected(
        result.gradient.views.head_b,
        gradients["head_b"],
        **tolerance,
    )
    _assert_selected(
        result.gradient.views.ln_f_gamma,
        gradients["ln_f_gamma"],
        **tolerance,
    )
    _assert_selected(
        result.gradient.views.ln_f_beta,
        gradients["ln_f_beta"],
        **tolerance,
    )
    _assert_selected(
        block.w_q,
        gradients["w_q"],
        **tolerance,
    )
    _assert_selected(
        block.w_k,
        gradients["w_k"],
        **tolerance,
    )
    _assert_selected(
        block.w_v,
        gradients["w_v"],
        **tolerance,
    )
    _assert_selected(
        block.w_o,
        gradients["w_o"],
        **tolerance,
    )
    _assert_selected(
        block.ff1_w,
        gradients["ff1_w"],
        **tolerance,
    )
    _assert_selected(
        block.ff2_w,
        gradients["ff2_w"],
        **tolerance,
    )
    _assert_selected(
        result.gradient.views.tok_emb,
        gradients["tok_emb"],
        **tolerance,
    )
    _assert_selected(
        result.gradient.views.pos_emb,
        gradients["pos_emb"],
        **tolerance,
    )
    _assert_selected(
        result.input_gradient,
        gradients["input_gradient"],
        **tolerance,
    )
    _assert_selected(
        result.attention_score_gradients[0][0],
        gradients["head_0_score_gradient"],
        **tolerance,
    )
    _assert_selected(
        result.attention_score_gradients[0][1],
        gradients["head_1_score_gradient"],
        **tolerance,
    )


def test_attention_score_gradients_have_exact_future_zeros(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    forward = _ordinary_forward(
        reference,
        parameters,
    )
    backward = calculate_transformer_backward(
        forward,
        tuple(ordinary["target_ids"]),
        parameters,
    )
    sequence_length = len(forward.input_ids)
    future_mask = np.triu(
        np.ones(
            (
                sequence_length,
                sequence_length,
            ),
            dtype=np.bool_,
        ),
        k=1,
    )

    for block_gradients in backward.attention_score_gradients:
        for head_gradient in block_gradients:
            assert_array_equal(
                head_gradient[future_mask],
                np.zeros(
                    int(np.count_nonzero(future_mask)),
                    dtype=np.float32,
                ),
            )


def test_repeated_token_embedding_gradients_accumulate_each_position(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    repeated = reference["repeated_token"]
    tolerance = reference["tolerances"]["gradient"]
    forward = calculate_transformer_forward(
        tuple(repeated["input_ids"]),
        parameters,
    )
    backward = calculate_transformer_backward(
        forward,
        tuple(repeated["target_ids"]),
        parameters,
    )

    _assert_selected(
        backward.gradient.views.tok_emb,
        repeated["token_embedding_gradient"],
        **tolerance,
    )
    _assert_selected(
        backward.gradient.views.pos_emb,
        repeated["position_embedding_gradient"],
        **tolerance,
    )
    _assert_selected(
        backward.input_gradient,
        repeated["input_gradient"],
        **tolerance,
    )

    expected_token_zero = backward.input_gradient[0] + backward.input_gradient[2]

    assert_allclose(
        backward.gradient.views.tok_emb[0],
        expected_token_zero,
        **tolerance,
    )


def test_selected_head_gradient_matches_finite_difference(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    tolerance = reference["tolerances"]["finite_difference"]
    input_ids = tuple(ordinary["input_ids"])
    target_ids = tuple(ordinary["target_ids"])

    forward = calculate_transformer_forward(
        input_ids,
        parameters,
    )
    backward = calculate_transformer_backward(
        forward,
        target_ids,
        parameters,
    )

    original = float(parameters.head_w[0, 0])
    epsilon = 1e-3

    try:
        parameters.head_w[0, 0] = np.float32(original + epsilon)
        positive_loss = calculate_transformer_cross_entropy(
            calculate_transformer_forward(
                input_ids,
                parameters,
            ).probabilities,
            target_ids,
        )

        parameters.head_w[0, 0] = np.float32(original - epsilon)
        negative_loss = calculate_transformer_cross_entropy(
            calculate_transformer_forward(
                input_ids,
                parameters,
            ).probabilities,
            target_ids,
        )
    finally:
        parameters.head_w[0, 0] = np.float32(original)

    numerical_gradient = (positive_loss - negative_loss) / (2.0 * epsilon)

    assert_allclose(
        backward.gradient.views.head_w[0, 0],
        numerical_gradient,
        **tolerance,
    )


def test_nonempty_logical_shard_accumulates_in_fixed_order(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    repeated = reference["repeated_token"]

    first = TransformerTrainingSequence(
        input_ids=tuple(ordinary["input_ids"] * 4),
        target_ids=tuple(ordinary["target_ids"] * 4),
    )
    second = TransformerTrainingSequence(
        input_ids=tuple(repeated["input_ids"] * 4),
        target_ids=tuple(repeated["target_ids"] * 4),
    )

    sequences = (first, second) * 4
    shard = build_logical_training_shards(len(sequences))[0]

    first_sequence = calculate_transformer_sequence(
        sequences[0],
        parameters,
    )
    second_sequence = calculate_transformer_sequence(
        sequences[1],
        parameters,
    )

    expected_gradient = np.array(
        first_sequence.backward.gradient.storage.astype(np.float64)
        + second_sequence.backward.gradient.storage.astype(np.float64),
        dtype=np.float32,
        order="C",
        copy=True,
    )

    result = calculate_logical_training_shard(
        sequences,
        shard,
        parameters,
    )

    assert result.processed_sequence_count == 2
    assert result.loss == first_sequence.loss + second_sequence.loss
    assert_array_equal(
        result.gradient.storage,
        expected_gradient,
    )


def test_empty_logical_shard_returns_exact_zero_state(
    parameters: TransformerParameterViews,
) -> None:
    shard = LogicalTrainingShard(
        shard_index=3,
        start_index=0,
        stop_index=0,
    )

    result = calculate_logical_training_shard(
        (),
        shard,
        parameters,
    )

    assert result.loss == 0.0
    assert result.processed_sequence_count == 0
    assert_array_equal(
        result.gradient.storage,
        np.zeros_like(result.gradient.storage),
    )


def test_forward_and_backward_preserve_parameter_bytes(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    before = tuple(
        array.copy()
        for array in (
            parameters.tok_emb,
            parameters.pos_emb,
            parameters.blocks[0].w_q,
            parameters.blocks[0].ff1_w,
            parameters.head_w,
        )
    )

    forward = _ordinary_forward(
        reference,
        parameters,
    )
    calculate_transformer_backward(
        forward,
        tuple(ordinary["target_ids"]),
        parameters,
    )

    after = (
        parameters.tok_emb,
        parameters.pos_emb,
        parameters.blocks[0].w_q,
        parameters.blocks[0].ff1_w,
        parameters.head_w,
    )

    for original, current in zip(
        before,
        after,
        strict=True,
    ):
        assert_array_equal(current, original)


def test_repeated_and_threaded_calls_are_isolated(
    reference: dict[str, Any],
    parameters: TransformerParameterViews,
) -> None:
    ordinary = reference["ordinary"]
    input_ids = tuple(ordinary["input_ids"])
    target_ids = tuple(ordinary["target_ids"])

    def calculate() -> tuple[np.ndarray, np.ndarray]:
        forward = calculate_transformer_forward(
            input_ids,
            parameters,
        )
        backward = calculate_transformer_backward(
            forward,
            target_ids,
            parameters,
        )
        return (
            forward.probabilities,
            backward.gradient.storage,
        )

    first = calculate()
    second = calculate()

    with ThreadPoolExecutor(max_workers=2) as executor:
        threaded = tuple(
            executor.map(
                lambda _: calculate(),
                range(2),
            )
        )

    assert_array_equal(first[0], second[0])
    assert_array_equal(first[1], second[1])

    for probabilities, gradient in threaded:
        assert_array_equal(probabilities, first[0])
        assert_array_equal(gradient, first[1])
        assert not np.shares_memory(probabilities, first[0])
        assert not np.shares_memory(gradient, first[1])


def test_gradient_buffers_are_fresh_and_canonical(
    parameters: TransformerParameterViews,
) -> None:
    first = create_transformer_gradient_buffer(parameters.layout)
    second = create_transformer_gradient_buffer(parameters.layout)

    assert first.storage.shape == (parameters.layout.total_float_count,)
    assert first.storage.dtype == np.dtype(np.float32)
    assert first.storage.flags.c_contiguous
    assert_array_equal(
        first.storage,
        np.zeros_like(first.storage),
    )
    assert not np.shares_memory(
        first.storage,
        second.storage,
    )

    for parameter_array in (
        parameters.tok_emb,
        parameters.pos_emb,
        parameters.head_w,
    ):
        assert not np.shares_memory(
            first.storage,
            parameter_array,
        )


@pytest.mark.parametrize(
    "invalid_input",
    [
        [],
        (0,) * (TRANSFORMER_CONTEXT_LENGTH + 1),
        (True, 1),
        (-1, 1),
        (0, 392),
    ],
)
def test_forward_rejects_invalid_token_sequences(
    invalid_input: object,
    parameters: TransformerParameterViews,
) -> None:
    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        calculate_transformer_forward(
            invalid_input,  # type: ignore[arg-type]
            parameters,
        )


def test_forward_rejects_nonfinite_parameters_without_partial_result(
    parameters: TransformerParameterViews,
) -> None:
    original = np.float32(parameters.tok_emb[0, 0])

    try:
        parameters.tok_emb[0, 0] = np.float32(np.nan)

        with pytest.raises(
            FloatingPointError,
            match="non-finite",
        ):
            calculate_transformer_forward(
                (0, 1, 2, 3),
                parameters,
            )
    finally:
        parameters.tok_emb[0, 0] = original
