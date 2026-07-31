# tests/test_transformer.py
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Lock
from typing import Any

import how_llms_work.ml.transformer as transformer_module
import numpy as np
import pytest
from how_llms_work.ml.math_utils import Mulberry32
from how_llms_work.ml.transformer import (
    LOGICAL_TRAINING_SHARD_COUNT,
    TRANSFORMER_BPE_MERGE_LIMIT,
    TRANSFORMER_GENERATION_SEED_LENGTH,
    TRANSFORMER_SEQUENCE_LENGTH,
    TRANSFORMER_TRAINING_CORPUS,
    LogicalTrainingShard,
    TransformerPreprocessingSnapshot,
    build_logical_training_shards,
    get_transformer_preprocessing,
)

_EXPECTED_TRANSFORMER_PUBLIC_SYMBOLS = (
    "LOGICAL_TRAINING_SHARD_COUNT",
    "TRANSFORMER_BPE_MERGE_LIMIT",
    "TRANSFORMER_GENERATION_SEED_LENGTH",
    "TRANSFORMER_SEQUENCE_LENGTH",
    "TRANSFORMER_TRAINING_CORPUS",
    "LogicalTrainingShard",
    "TransformerPreprocessingSnapshot",
    "TransformerTrainingSequence",
    "build_logical_training_shards",
    "get_transformer_preprocessing",
    "TRANSFORMER_ATTENTION_HEAD_COUNT",
    "TRANSFORMER_ATTENTION_SCALE",
    "TRANSFORMER_CONTEXT_LENGTH",
    "TRANSFORMER_CROSS_ENTROPY_EPSILON",
    "TRANSFORMER_EMBEDDING_DIMENSION",
    "TRANSFORMER_FEED_FORWARD_DIMENSION",
    "TRANSFORMER_HEAD_DIMENSION",
    "TRANSFORMER_LAYER_NORMALIZATION_EPSILON",
    "TRANSFORMER_MAX_LAYER_COUNT",
    "TRANSFORMER_MIN_LAYER_COUNT",
    "InitializedTransformerParameters",
    "LogicalTrainingShardResult",
    "TransformerAttentionHeadCache",
    "TransformerBackwardResult",
    "TransformerBlockForwardCache",
    "TransformerBlockParameterViews",
    "TransformerForwardResult",
    "TransformerGradientBuffer",
    "TransformerLayerNormalizationCache",
    "TransformerParameterLayout",
    "TransformerParameterLayoutRecord",
    "TransformerParameterViews",
    "TransformerSequenceResult",
    "build_transformer_parameter_layout",
    "build_transformer_parameter_views",
    "calculate_logical_training_shard",
    "calculate_transformer_backward",
    "calculate_transformer_cross_entropy",
    "calculate_transformer_forward",
    "calculate_transformer_sequence",
    "create_transformer_gradient_buffer",
    "initialize_transformer_parameters",
    "transformer_parameter_count",
    "TransformerEpochObservation",
    "TransformerEpochUpdate",
    "TransformerTrainingRun",
    "build_transformer_report_epochs",
    "create_transformer_training_run",
    "GeneratedTextSample",
    "SavedTransformerPromptError",
    "EmptySavedTransformerPromptError",
    "UnsupportedSavedTransformerPromptError",
    "SavedTransformerPromptTooLongError",
    "PreparedSavedTransformerPrompt",
    "SavedTransformerBlockWeights",
    "SavedTransformerConfig",
    "SavedTransformerMerge",
    "SavedTransformerModel",
    "SavedTransformerWeights",
    "build_saved_transformer_model",
    "evaluate_transformer_final_loss",
    "generate_transformer_text",
    "generate_saved_transformer_text",
    "prepare_saved_transformer_prompt",
)


REFERENCE_PATH = Path(__file__).parent / "fixtures" / "transformer_preprocessing_reference.json"
LAYOUT_INITIALIZATION_REFERENCE_PATH = (
    Path(__file__).parent / "fixtures" / "transformer_layout_initialization_reference.json"
)


def _load_reference() -> dict[str, Any]:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def _serialize_shards(
    shards: tuple[LogicalTrainingShard, ...],
) -> list[list[int]]:
    return [
        [
            shard.shard_index,
            shard.start_index,
            shard.stop_index,
        ]
        for shard in shards
    ]


def test_transformer_constants_and_corpus_match_reference() -> None:
    reference = _load_reference()
    constants = reference["constants"]

    assert TRANSFORMER_BPE_MERGE_LIMIT == constants["bpe_merge_limit"]
    assert TRANSFORMER_SEQUENCE_LENGTH == constants["sequence_length"]
    assert TRANSFORMER_GENERATION_SEED_LENGTH == constants["generation_seed_length"]
    assert LOGICAL_TRAINING_SHARD_COUNT == constants["logical_shard_count"]

    assert len(TRANSFORMER_TRAINING_CORPUS) == constants["story_count"]
    assert TRANSFORMER_TRAINING_CORPUS == tuple(reference["corpus"])


def test_complete_transformer_merge_table_matches_reference() -> None:
    reference = _load_reference()
    snapshot = get_transformer_preprocessing()

    expected_training_text = "".join(f" {story}" for story in reference["corpus"]).lower()

    assert snapshot.bpe_training_text == expected_training_text
    assert len(snapshot.bpe_training_text) == reference["bpe_training_text_length"]

    actual_merge_table = [
        [
            merge.pair[0],
            merge.pair[1],
            merge.merged,
            merge.frequency,
        ]
        for merge in snapshot.merges
    ]

    assert actual_merge_table == reference["merge_table"]


def test_complete_tokenization_vocabulary_indices_and_ids_match_reference() -> None:
    reference = _load_reference()
    snapshot = get_transformer_preprocessing()

    expected_vocabulary = tuple(reference["vocabulary"])
    expected_token_ids = tuple(reference["token_ids"])
    expected_story_token_counts = reference["story_token_counts"]

    assert snapshot.vocabulary == expected_vocabulary

    assert dict(snapshot.token_indices) == {
        token: index for index, token in enumerate(expected_vocabulary)
    }

    assert snapshot.token_ids == expected_token_ids

    assert [len(tokens) for tokens in snapshot.tokenized_stories] == expected_story_token_counts

    offset = 0

    for actual_story_tokens, story_token_count in zip(
        snapshot.tokenized_stories,
        expected_story_token_counts,
        strict=True,
    ):
        story_ids = expected_token_ids[offset : offset + story_token_count]

        expected_story_tokens = tuple(expected_vocabulary[token_id] for token_id in story_ids)

        assert actual_story_tokens == expected_story_tokens

        offset += story_token_count

    assert offset == len(expected_token_ids)


def test_all_training_sequences_seed_ids_and_shards_match_reference() -> None:
    reference = _load_reference()
    snapshot = get_transformer_preprocessing()
    reference_token_ids = tuple(reference["token_ids"])

    assert len(snapshot.training_sequences) == reference["training_sequence_count"]

    for start, sequence in enumerate(snapshot.training_sequences):
        assert (
            sequence.input_ids == reference_token_ids[start : start + TRANSFORMER_SEQUENCE_LENGTH]
        )

        assert (
            sequence.target_ids
            == reference_token_ids[start + 1 : start + TRANSFORMER_SEQUENCE_LENGTH + 1]
        )

    first_sequence = snapshot.training_sequences[0]
    last_sequence = snapshot.training_sequences[-1]

    assert [
        list(first_sequence.input_ids),
        list(first_sequence.target_ids),
    ] == reference["first_training_sequence"]

    assert [
        list(last_sequence.input_ids),
        list(last_sequence.target_ids),
    ] == reference["last_training_sequence"]

    assert list(snapshot.generation_seed_ids) == (reference["generation_seed_ids"])

    assert (
        _serialize_shards(snapshot.logical_training_shards) == reference["logical_training_shards"]
    )


@pytest.mark.parametrize(
    "sequence_count",
    [0, 1, 2, 3, 4, 5, 7, 8, 9],
)
def test_logical_training_shards_cover_small_counts_exactly(
    sequence_count: int,
) -> None:
    reference = _load_reference()

    expected = reference["logical_shard_reference_cases"][str(sequence_count)]

    assert _serialize_shards(build_logical_training_shards(sequence_count)) == expected


def test_logical_training_shards_are_cpu_count_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence_count = 3_179

    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    one_cpu = build_logical_training_shards(sequence_count)

    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    many_cpus = build_logical_training_shards(sequence_count)

    assert one_cpu == many_cpus

    assert _serialize_shards(one_cpu) == [
        [0, 0, 795],
        [1, 795, 1_590],
        [2, 1_590, 2_385],
        [3, 2_385, 3_179],
    ]


def test_logical_training_shards_reject_invalid_counts() -> None:
    with pytest.raises(TypeError, match="integer"):
        build_logical_training_shards(True)

    with pytest.raises(TypeError, match="integer"):
        build_logical_training_shards(1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative"):
        build_logical_training_shards(-1)


def test_snapshot_is_deeply_immutable_and_has_no_mutable_aliases() -> None:
    snapshot = get_transformer_preprocessing()

    with pytest.raises(FrozenInstanceError):
        snapshot.corpus = ()  # type: ignore[misc]

    with pytest.raises(TypeError):
        snapshot.token_indices["new-token"] = 0  # type: ignore[index]

    with pytest.raises(TypeError):
        snapshot.tokenized_stories[0][0] = "changed"  # type: ignore[index]

    with pytest.raises(FrozenInstanceError):
        snapshot.merges[0].frequency = 0  # type: ignore[misc]

    copied_indices = dict(snapshot.token_indices)
    copied_indices["new-token"] = 0

    assert "new-token" not in snapshot.token_indices


def test_snapshot_is_reused_sequentially_and_has_no_request_input() -> None:
    assert tuple(inspect.signature(get_transformer_preprocessing).parameters) == ()

    first = get_transformer_preprocessing()

    unrelated_request_words = ["king", "queen"]
    unrelated_request_words.append("changed")

    second = get_transformer_preprocessing()

    assert first is second
    assert first.corpus == TRANSFORMER_TRAINING_CORPUS


def test_snapshot_is_initialized_once_under_concurrent_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builder = transformer_module._build_transformer_preprocessing_snapshot

    build_count = 0
    count_lock = Lock()

    def counted_builder() -> TransformerPreprocessingSnapshot:
        nonlocal build_count

        with count_lock:
            build_count += 1

        time.sleep(0.01)

        return original_builder()

    monkeypatch.setattr(
        transformer_module,
        "_TRANSFORMER_PREPROCESSING",
        None,
    )

    monkeypatch.setattr(
        transformer_module,
        "_build_transformer_preprocessing_snapshot",
        counted_builder,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        snapshots = tuple(
            executor.map(
                lambda _index: get_transformer_preprocessing(),
                range(16),
            )
        )

    assert build_count == 1
    assert all(snapshot is snapshots[0] for snapshot in snapshots)


def test_failed_initialization_is_not_published_and_later_call_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builder = transformer_module._build_transformer_preprocessing_snapshot

    attempts = 0

    def fail_once() -> TransformerPreprocessingSnapshot:
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            raise RuntimeError("controlled preprocessing failure")

        return original_builder()

    monkeypatch.setattr(
        transformer_module,
        "_TRANSFORMER_PREPROCESSING",
        None,
    )

    monkeypatch.setattr(
        transformer_module,
        "_build_transformer_preprocessing_snapshot",
        fail_once,
    )

    with pytest.raises(
        RuntimeError,
        match="controlled preprocessing failure",
    ):
        get_transformer_preprocessing()

    assert transformer_module._TRANSFORMER_PREPROCESSING is None

    snapshot = get_transformer_preprocessing()

    assert attempts == 2
    assert snapshot is transformer_module._TRANSFORMER_PREPROCESSING


def _load_layout_initialization_reference() -> dict[str, Any]:
    return json.loads(
        LAYOUT_INITIALIZATION_REFERENCE_PATH.read_text(
            encoding="utf-8",
        )
    )


def _serialize_parameter_layout(
    layout: transformer_module.TransformerParameterLayout,
) -> list[dict[str, object]]:
    return [
        {
            "key": record.key,
            "block_index": record.block_index,
            "float_offset": record.float_offset,
            "byte_offset": record.byte_offset,
            "length": record.length,
            "shape": list(record.shape),
            "total_float_count": record.total_float_count,
            "total_byte_count": record.total_byte_count,
        }
        for record in layout.records
    ]


def _float32_bits_le(
    value: np.float32,
) -> str:
    return (
        np.asarray(
            value,
            dtype="<f4",
        )
        .tobytes()
        .hex()
    )


def _storage_sha256_le(
    storage: np.ndarray,
) -> str:
    little_endian = storage.astype(
        "<f4",
        copy=False,
    )

    return hashlib.sha256(
        little_endian.tobytes(
            order="C",
        )
    ).hexdigest()


def test_transformer_parameter_public_contract_and_architecture() -> None:
    assert tuple(transformer_module.__all__) == _EXPECTED_TRANSFORMER_PUBLIC_SYMBOLS

    for symbol in _EXPECTED_TRANSFORMER_PUBLIC_SYMBOLS:
        assert hasattr(transformer_module, symbol)

    assert transformer_module.LOGICAL_TRAINING_SHARD_COUNT == 4
    assert transformer_module.TRANSFORMER_ATTENTION_HEAD_COUNT == 2
    assert transformer_module.TRANSFORMER_ATTENTION_SCALE == 0.25
    assert transformer_module.TRANSFORMER_CONTEXT_LENGTH == 32
    assert transformer_module.TRANSFORMER_CROSS_ENTROPY_EPSILON == 1e-10
    assert transformer_module.TRANSFORMER_EMBEDDING_DIMENSION == 32
    assert transformer_module.TRANSFORMER_FEED_FORWARD_DIMENSION == 128
    assert transformer_module.TRANSFORMER_HEAD_DIMENSION == 16
    assert transformer_module.TRANSFORMER_LAYER_NORMALIZATION_EPSILON == 1e-5
    assert transformer_module.TRANSFORMER_MIN_LAYER_COUNT == 1
    assert transformer_module.TRANSFORMER_MAX_LAYER_COUNT == 6

    assert (
        transformer_module.TRANSFORMER_ATTENTION_HEAD_COUNT
        * transformer_module.TRANSFORMER_HEAD_DIMENSION
        == transformer_module.TRANSFORMER_EMBEDDING_DIMENSION
    )

    expected_parameter_counts = (
        39_272,
        51_976,
        64_680,
        77_384,
        90_088,
        102_792,
    )

    for num_layers, expected_count in enumerate(
        expected_parameter_counts,
        start=1,
    ):
        assert transformer_module.transformer_parameter_count(num_layers) == (expected_count)


@pytest.mark.parametrize(
    "num_layers",
    [
        1,
        2,
        6,
    ],
)
def test_representative_transformer_layouts_match_reference(
    num_layers: int,
) -> None:
    reference = _load_layout_initialization_reference()
    layout = transformer_module.build_transformer_parameter_layout(
        num_layers,
    )

    assert (
        _serialize_parameter_layout(
            layout,
        )
        == reference["representative_layouts"][str(num_layers)]
    )


@pytest.mark.parametrize(
    "num_layers",
    [
        1,
        2,
        3,
        4,
        5,
        6,
    ],
)
def test_transformer_layout_invariants_for_every_depth(
    num_layers: int,
) -> None:
    reference = _load_layout_initialization_reference()
    summary = reference["layer_summaries"][str(num_layers)]
    layout = transformer_module.build_transformer_parameter_layout(
        num_layers,
    )

    assert layout.vocabulary_size == reference["architecture"]["vocabulary_size"]

    assert len(layout.records) == 16 * num_layers + 6 == summary["record_count"]

    assert layout.records[0].float_offset == 0

    identities: set[
        tuple[
            int | None,
            str,
        ]
    ] = set()

    for record_index, record in enumerate(layout.records):
        assert math.prod(record.shape) == record.length
        assert record.byte_offset == record.float_offset * 4
        assert record.total_float_count == layout.total_float_count
        assert record.total_byte_count == layout.total_byte_count

        identity = (
            record.block_index,
            record.key,
        )

        assert identity not in identities

        identities.add(identity)

        if record_index + 1 < len(layout.records):
            assert record.float_stop == layout.records[record_index + 1].float_offset

    assert layout.records[-1].float_stop == layout.total_float_count
    assert layout.total_float_count == summary["total_float_count"]
    assert layout.total_byte_count == summary["total_byte_count"]
    assert layout.total_byte_count == layout.total_float_count * 4
    assert transformer_module.transformer_parameter_count(num_layers) == layout.total_float_count

    block_indices = {
        record.block_index for record in layout.records if record.block_index is not None
    }

    assert block_indices == set(range(num_layers))

    assert transformer_module.build_transformer_parameter_layout(num_layers) == layout


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        1.0,
        "1",
        None,
    ],
)
def test_transformer_layout_rejects_non_integer_layer_counts(
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        transformer_module.build_transformer_parameter_layout(
            value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0,
        7,
    ],
)
def test_transformer_layout_rejects_unsupported_layer_counts(
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 1 and 6",
    ):
        transformer_module.build_transformer_parameter_layout(
            value,
        )


def test_transformer_parameter_layout_is_immutable() -> None:
    layout = transformer_module.build_transformer_parameter_layout(
        1,
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        layout.total_float_count = 0  # type: ignore[misc]

    with pytest.raises(
        FrozenInstanceError,
    ):
        layout.records[0].float_offset = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "extra_capacity",
    [
        0,
        17,
    ],
)
def test_transformer_parameter_views_share_exact_storage(
    extra_capacity: int,
) -> None:
    layout = transformer_module.build_transformer_parameter_layout(
        2,
    )
    storage = np.arange(
        layout.total_float_count + extra_capacity,
        dtype=np.float32,
    )
    tail_before = storage[layout.total_float_count :].copy()

    views = transformer_module.build_transformer_parameter_views(
        storage,
        layout,
    )

    for record_index, record in enumerate(layout.records):
        view = views.get(
            record.key,
            record.block_index,
        )

        assert view.dtype == np.dtype(np.float32)
        assert view.shape == record.shape
        assert view.flags.c_contiguous
        assert np.shares_memory(
            view,
            storage,
        )
        assert view.ctypes.data == storage.ctypes.data + record.byte_offset

        if record_index + 1 < len(layout.records):
            next_record = layout.records[record_index + 1]
            next_view = views.get(
                next_record.key,
                next_record.block_index,
            )

            assert not np.shares_memory(
                view,
                next_view,
            )

    wq_record = layout.get_record(
        "wQ",
        0,
    )

    views.get(
        "wQ",
        0,
    )[0, 0] = np.float32(-123.5)

    assert storage[wq_record.float_offset] == np.float32(-123.5)

    head_record = layout.get_record(
        "headW",
    )

    storage[head_record.float_offset + 7] = np.float32(456.25)

    assert views.get(
        "headW",
    ).reshape(-1)[7] == np.float32(456.25)

    views.get(
        "headB",
    )[-1] = np.float32(99.0)

    np.testing.assert_array_equal(
        storage[layout.total_float_count :],
        tail_before,
    )


def test_transformer_parameter_views_preserve_read_only_storage() -> None:
    layout = transformer_module.build_transformer_parameter_layout(
        1,
    )
    storage = np.zeros(
        layout.total_float_count,
        dtype=np.float32,
    )
    storage.setflags(
        write=False,
    )

    views = transformer_module.build_transformer_parameter_views(
        storage,
        layout,
    )

    assert not views.get("tokEmb").flags.writeable

    with pytest.raises(
        ValueError,
    ):
        views.get("tokEmb")[0, 0] = np.float32(1.0)


def test_transformer_parameter_views_reject_invalid_storage() -> None:
    layout = transformer_module.build_transformer_parameter_layout(
        1,
    )

    with pytest.raises(
        TypeError,
        match="NumPy",
    ):
        transformer_module.build_transformer_parameter_views(
            [0.0] * layout.total_float_count,
            layout,
        )

    with pytest.raises(
        TypeError,
        match="float32",
    ):
        transformer_module.build_transformer_parameter_views(
            np.zeros(
                layout.total_float_count,
                dtype=np.float64,
            ),
            layout,
        )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        transformer_module.build_transformer_parameter_views(
            np.zeros(
                (
                    1,
                    layout.total_float_count,
                ),
                dtype=np.float32,
            ),
            layout,
        )

    backing = np.zeros(
        layout.total_float_count * 2,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="C-contiguous",
    ):
        transformer_module.build_transformer_parameter_views(
            backing[::2],
            layout,
        )

    with pytest.raises(
        ValueError,
        match="smaller",
    ):
        transformer_module.build_transformer_parameter_views(
            np.zeros(
                layout.total_float_count - 1,
                dtype=np.float32,
            ),
            layout,
        )


def test_transformer_parameter_views_reject_forged_layouts() -> None:
    layout = transformer_module.build_transformer_parameter_layout(
        2,
    )
    storage = np.zeros(
        layout.total_float_count,
        dtype=np.float32,
    )
    records = layout.records

    gap_record = replace(
        records[1],
        float_offset=(records[1].float_offset + 1),
        byte_offset=(records[1].byte_offset + 4),
    )

    overlap_record = replace(
        records[1],
        float_offset=(records[1].float_offset - 1),
        byte_offset=(records[1].byte_offset - 4),
    )

    wrong_shape_record = replace(
        records[0],
        shape=(records[0].length + 1,),
    )

    wrong_block_record = replace(
        records[2],
        block_index=1,
    )

    wrong_record_total = replace(
        records[0],
        total_float_count=(layout.total_float_count + 1),
    )

    outside_record = replace(
        records[-1],
        float_offset=(records[-1].float_offset + 1),
        byte_offset=(records[-1].byte_offset + 4),
    )

    forged_layouts = (
        replace(
            layout,
            records=(
                records[0],
                gap_record,
                *records[2:],
            ),
        ),
        replace(
            layout,
            records=(
                records[0],
                overlap_record,
                *records[2:],
            ),
        ),
        replace(
            layout,
            records=(
                wrong_shape_record,
                *records[1:],
            ),
        ),
        replace(
            layout,
            records=(
                records[1],
                records[0],
                *records[2:],
            ),
        ),
        replace(
            layout,
            records=(
                records[0],
                records[1],
                wrong_block_record,
                *records[3:],
            ),
        ),
        replace(
            layout,
            records=(
                wrong_record_total,
                *records[1:],
            ),
        ),
        replace(
            layout,
            total_float_count=(layout.total_float_count + 1),
        ),
        replace(
            layout,
            records=(
                *records[:-1],
                outside_record,
            ),
        ),
    )

    for forged_layout in forged_layouts:
        with pytest.raises(
            ValueError,
            match="canonical",
        ):
            transformer_module.build_transformer_parameter_views(
                storage,
                forged_layout,
            )


@pytest.mark.parametrize(
    "num_layers",
    [
        1,
        2,
        3,
        4,
        5,
        6,
    ],
)
def test_transformer_initialization_matches_exact_reference(
    num_layers: int,
) -> None:
    reference = _load_layout_initialization_reference()["initialization"]["layers"][str(num_layers)]

    layout = transformer_module.build_transformer_parameter_layout(
        num_layers,
    )
    generator = Mulberry32(
        42,
    )

    initialized = transformer_module.initialize_transformer_parameters(
        layout,
        generator,
    )

    assert initialized.layout == layout
    assert initialized.storage.dtype == np.dtype(np.float32)
    assert initialized.storage.ndim == 1
    assert initialized.storage.flags.c_contiguous
    assert initialized.storage.size == layout.total_float_count
    assert np.isfinite(initialized.storage).all()
    assert generator.draw_count == reference["draw_count"]
    assert generator.state == reference["final_generator_state"]
    assert _storage_sha256_le(initialized.storage) == reference["sha256_le_float32"]

    for landmark in reference["landmarks"]:
        key = landmark["key"]
        block_index = landmark["block_index"]
        index = tuple(landmark["index"])

        view = initialized.views.get(
            key,
            block_index,
        )

        assert _float32_bits_le(view[index]) == landmark["value_bits_le"]

        record = layout.get_record(
            key,
            block_index,
        )

        relative_offset = int(
            np.ravel_multi_index(
                index,
                record.shape,
                order="C",
            )
        )

        assert record.float_offset + relative_offset == landmark["flat_offset"]


def test_transformer_initialization_sets_deterministic_families_exactly() -> None:
    layout = transformer_module.build_transformer_parameter_layout(
        6,
    )

    initialized = transformer_module.initialize_transformer_parameters(
        layout,
        Mulberry32(42),
    )

    for block in initialized.views.blocks:
        for gamma in (
            block.ln1_gamma,
            block.ln2_gamma,
        ):
            assert np.all(gamma.view(np.uint32) == np.uint32(0x3F800000))

        for zero_array in (
            block.ln1_beta,
            block.b_q,
            block.b_k,
            block.b_v,
            block.b_o,
            block.ln2_beta,
            block.ff1_b,
            block.ff2_b,
        ):
            assert np.all(zero_array.view(np.uint32) == np.uint32(0))

    assert np.all(initialized.views.ln_f_gamma.view(np.uint32) == np.uint32(0x3F800000))

    assert np.all(initialized.views.ln_f_beta.view(np.uint32) == np.uint32(0))

    assert np.all(initialized.views.head_b.view(np.uint32) == np.uint32(0))


def test_transformer_initialization_rejects_noncanonical_inputs() -> None:
    layout = transformer_module.build_transformer_parameter_layout(
        1,
    )

    with pytest.raises(
        TypeError,
        match="Mulberry32",
    ):
        transformer_module.initialize_transformer_parameters(
            layout,
            object(),  # type: ignore[arg-type]
        )

    forged_layout = replace(
        layout,
        total_byte_count=(layout.total_byte_count + 4),
    )

    with pytest.raises(
        ValueError,
        match="canonical",
    ):
        transformer_module.initialize_transformer_parameters(
            forged_layout,
            Mulberry32(42),
        )


def _initialize_transformer_for_isolation(
    num_layers: int,
) -> tuple[
    transformer_module.InitializedTransformerParameters,
    int,
    int,
]:
    generator = Mulberry32(
        42,
    )

    initialized = transformer_module.initialize_transformer_parameters(
        transformer_module.build_transformer_parameter_layout(
            num_layers,
        ),
        generator,
    )

    return (
        initialized,
        generator.draw_count,
        generator.state,
    )


def test_same_seed_transformer_initializations_are_equal_but_not_aliased() -> None:
    (
        first,
        first_draw_count,
        first_state,
    ) = _initialize_transformer_for_isolation(
        2,
    )

    (
        second,
        second_draw_count,
        second_state,
    ) = _initialize_transformer_for_isolation(
        2,
    )

    np.testing.assert_array_equal(
        first.storage,
        second.storage,
    )

    assert first.layout == second.layout
    assert first_draw_count == second_draw_count
    assert first_state == second_state
    assert first.storage is not second.storage
    assert not np.shares_memory(
        first.storage,
        second.storage,
    )

    first.storage[0] = np.float32(999.0)

    assert second.storage[0] != np.float32(999.0)


def test_transformer_depth_and_concurrent_initializations_are_isolated() -> None:
    (
        one_layer,
        one_draw_count,
        one_state,
    ) = _initialize_transformer_for_isolation(
        1,
    )

    (
        six_layers,
        six_draw_count,
        six_state,
    ) = _initialize_transformer_for_isolation(
        6,
    )

    assert one_layer.layout.total_float_count != six_layers.layout.total_float_count
    assert one_draw_count != six_draw_count
    assert one_state != six_state

    with ThreadPoolExecutor(
        max_workers=6,
    ) as executor:
        results = tuple(
            executor.map(
                _initialize_transformer_for_isolation,
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                ],
            )
        )

    reference = _load_layout_initialization_reference()

    for (
        num_layers,
        (
            initialized,
            draw_count,
            state,
        ),
    ) in enumerate(
        results,
        start=1,
    ):
        expected = reference["initialization"]["layers"][str(num_layers)]

        assert draw_count == expected["draw_count"]
        assert state == expected["final_generator_state"]
        assert _storage_sha256_le(initialized.storage) == expected["sha256_le_float32"]

    for left_index, left in enumerate(results):
        for right in results[left_index + 1 :]:
            assert not np.shares_memory(
                left[0].storage,
                right[0].storage,
            )


def test_transformer_initialization_has_no_saved_model_or_filesystem_input() -> None:
    assert tuple(
        inspect.signature(transformer_module.initialize_transformer_parameters).parameters
    ) == (
        "layout",
        "generator",
    )


def test_public_contract_exposes_only_approved_symbols() -> None:
    assert tuple(transformer_module.__all__) == _EXPECTED_TRANSFORMER_PUBLIC_SYMBOLS

    exported_values = {
        symbol: getattr(transformer_module, symbol) for symbol in transformer_module.__all__
    }

    assert tuple(exported_values) == _EXPECTED_TRANSFORMER_PUBLIC_SYMBOLS
    assert len(exported_values) == len(set(exported_values))
