# tests/test_transformer_loading.py
from __future__ import annotations

import json
import os
import stat
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias, cast

import how_llms_work.routes.train_transformer as train_transformer_route
import numpy as np
import pytest
from how_llms_work.ml.math_utils import Mulberry32
from how_llms_work.ml.transformer import (
    LogicalTrainingShardResult,
    SavedTransformerModel,
    TransformerParameterLayoutRecord,
    build_saved_transformer_model,
    build_transformer_parameter_layout,
    create_transformer_gradient_buffer,
    create_transformer_training_run,
    get_transformer_preprocessing,
    initialize_transformer_parameters,
)

_CANONICAL_MODEL_FILENAME = "transformer-weights-e50-l1-d32-h2-ff128-ctx32.json"
_PUBLIC_LOAD_FAILURE = "The saved Transformer model could not be loaded."
_ArtifactState: TypeAlias = tuple[bytes, int, int, int, int, int]


def _build_complete_one_layer_model() -> SavedTransformerModel:
    preprocessing = get_transformer_preprocessing()
    layout = build_transformer_parameter_layout(1)
    initialized = initialize_transformer_parameters(
        layout,
        Mulberry32(42),
    )
    run = create_transformer_training_run(
        initialized,
        sequence_count=len(preprocessing.training_sequences),
        requested_epochs=0,
    )

    shard_results = tuple(
        LogicalTrainingShardResult(
            shard=shard,
            processed_sequence_count=shard.stop_index - shard.start_index,
            loss=0.0,
            gradient=create_transformer_gradient_buffer(layout),
        )
        for shard in run.logical_training_shards
    )

    run.advance_epoch(shard_results)

    assert run.is_complete

    return build_saved_transformer_model(
        run,
        preprocessing,
    )


@pytest.fixture(scope="module")
def saved_transformer_model() -> SavedTransformerModel:
    return _build_complete_one_layer_model()


@pytest.fixture
def model_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    directory = tmp_path / "saved-transformer-models"
    directory.mkdir()

    monkeypatch.setattr(
        train_transformer_route,
        "get_transformer_model_directory",
        lambda: directory,
    )

    return directory


@pytest.fixture
def saved_model_path(
    model_directory: Path,
    saved_transformer_model: SavedTransformerModel,
) -> Path:
    path = model_directory / _CANONICAL_MODEL_FILENAME
    path.write_bytes(_serialize_model(saved_transformer_model))
    return path


def _serialize_model(model: SavedTransformerModel) -> bytes:
    document = json.dumps(
        model,
        indent=2,
        allow_nan=False,
    )

    return f"{document}\n".encode("utf-8")


def _write_json_value(
    path: Path,
    value: object,
) -> None:
    """Write one JSON value with the current Saved Transformer Model formatting."""
    path.write_text(
        f"{json.dumps(value, indent=2, allow_nan=False)}\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact_state(path: Path) -> _ArtifactState:
    metadata = path.stat()

    return (
        path.read_bytes(),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_ino,
    )


def _create_symbolic_link_or_skip(
    link_path: Path,
    target_path: Path,
    *,
    target_is_directory: bool,
) -> None:
    try:
        link_path.symlink_to(
            target_path,
            target_is_directory=target_is_directory,
        )
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Symbolic links are unavailable in this environment: {error}")


def _create_windows_junction_or_skip(
    junction_path: Path,
    target_directory: Path,
) -> None:
    """Create one real Windows junction or report the concrete platform limitation."""
    if os.name != "nt":
        pytest.skip("Real Windows junction creation is only available on Windows.")

    environment = os.environ.copy()
    environment["TRANSFORMER_TEST_JUNCTION_PATH"] = str(junction_path)
    environment["TRANSFORMER_TEST_JUNCTION_TARGET"] = str(target_directory)

    command = (
        "$ErrorActionPreference = 'Stop'; "
        "New-Item -ItemType Junction "
        "-Path $env:TRANSFORMER_TEST_JUNCTION_PATH "
        "-Target $env:TRANSFORMER_TEST_JUNCTION_TARGET | Out-Null"
    )

    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except OSError as error:
        pytest.skip(f"Windows PowerShell could not create a junction: {error}")
    except subprocess.TimeoutExpired:
        pytest.skip("Windows junction creation timed out after 30 seconds.")

    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout).strip()

        if not reason:
            reason = f"PowerShell exited with code {completed.returncode}."

        pytest.skip(f"Windows junction creation was unavailable: {reason}")

    if not junction_path.is_junction():
        pytest.skip(
            "PowerShell returned success, but pathlib.Path.is_junction() "
            "did not classify the created path as a junction."
        )


def _load_named_model(model_filename: str) -> Any:
    return train_transformer_route.load_named_transformer_model(model_filename)


def _assert_public_load_rejection(
    model_filename: str,
    protected_path: Path,
) -> None:
    before = _artifact_state(protected_path)
    before_entries = tuple(sorted(entry.name for entry in protected_path.parent.iterdir()))

    with pytest.raises(train_transformer_route.SavedTransformerModelLoadError) as captured:
        _load_named_model(model_filename)

    assert type(captured.value) is train_transformer_route.SavedTransformerModelLoadError
    assert captured.value.args == (_PUBLIC_LOAD_FAILURE,)
    assert str(captured.value) == _PUBLIC_LOAD_FAILURE
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__
    assert protected_path.exists()
    assert _artifact_state(protected_path) == before
    assert tuple(sorted(entry.name for entry in protected_path.parent.iterdir())) == before_entries


def _saved_values_for_record(
    model: SavedTransformerModel,
    record: TransformerParameterLayoutRecord,
) -> list[float]:
    weights = cast(dict[str, object], model["weights"])

    if record.block_index is None:
        return cast(list[float], weights[record.key])

    blocks = cast(
        list[dict[str, list[float]]],
        weights["blocks"],
    )

    return blocks[record.block_index][record.key]


def _build_invalid_structural_model(
    model: SavedTransformerModel,
    case: str,
) -> object:
    """Return one deliberately malformed current-format model candidate."""
    candidate = cast(dict[str, Any], deepcopy(model))
    config = cast(dict[str, Any], candidate["config"])
    vocabulary = cast(list[Any], candidate["vocab"])
    merges = cast(list[Any], candidate["merges"])
    weights = cast(dict[str, Any], candidate["weights"])
    blocks = cast(list[Any], weights["blocks"])
    first_merge = cast(dict[str, Any], merges[0])
    first_block = cast(dict[str, Any], blocks[0])

    if case == "top-level-not-object":
        return [candidate]

    if case == "top-level-reordered":
        model_type = candidate.pop("type")
        candidate["type"] = model_type
        return candidate

    if case == "top-level-missing-key":
        candidate.pop("merges")
        return candidate

    if case == "top-level-extra-key":
        candidate["epochs"] = 50
        return candidate

    if case == "wrong-model-type":
        candidate["type"] = "transformer"
        return candidate

    if case == "config-not-object":
        candidate["config"] = []
        return candidate

    if case == "config-reordered":
        vocab_size = config.pop("vocabSize")
        config["vocabSize"] = vocab_size
        return candidate

    if case == "config-missing-key":
        config.pop("numHeads")
        return candidate

    if case == "config-extra-key":
        config["epochs"] = 50
        return candidate

    if case == "config-boolean-integer":
        config["vocabSize"] = True
        return candidate

    if case == "config-fixed-architecture":
        config["contextLen"] = 31
        return candidate

    if case == "weights-not-object":
        candidate["weights"] = []
        return candidate

    if case == "weights-reordered":
        token_embeddings = weights.pop("tokEmb")
        weights["tokEmb"] = token_embeddings
        return candidate

    if case == "weights-missing-array":
        weights.pop("headB")
        return candidate

    if case == "weights-extra-array":
        weights["extra"] = []
        return candidate

    if case == "block-count":
        weights["blocks"] = []
        return candidate

    if case == "block-not-object":
        blocks[0] = []
        return candidate

    if case == "block-reordered":
        layer_norm_gamma = first_block.pop("ln1Gamma")
        first_block["ln1Gamma"] = layer_norm_gamma
        return candidate

    if case == "block-missing-array":
        first_block.pop("ff2B")
        return candidate

    if case == "block-extra-array":
        first_block["extra"] = []
        return candidate

    if case == "vocab-not-list":
        candidate["vocab"] = {}
        return candidate

    if case == "vocab-wrong-length":
        vocabulary.pop()
        config["vocabSize"] = len(vocabulary)
        return candidate

    if case == "vocab-reordered":
        vocabulary[0], vocabulary[1] = vocabulary[1], vocabulary[0]
        return candidate

    if case == "vocab-duplicate":
        vocabulary[0] = vocabulary[1]
        return candidate

    if case == "vocab-non-string":
        vocabulary[0] = 1
        return candidate

    if case == "merges-not-list":
        candidate["merges"] = {}
        return candidate

    if case == "merge-not-object":
        merges[0] = []
        return candidate

    if case == "merge-reordered":
        pair = first_merge.pop("pair")
        first_merge["pair"] = pair
        return candidate

    if case == "merge-missing-key":
        first_merge.pop("merged")
        return candidate

    if case == "merge-extra-key":
        first_merge["frequency"] = 1
        return candidate

    if case == "merge-pair-not-list":
        first_merge["pair"] = {}
        return candidate

    if case == "merge-pair-wrong-length":
        first_merge["pair"] = [first_merge["pair"][0]]
        return candidate

    if case == "merge-pair-non-string":
        first_merge["pair"][0] = 1
        return candidate

    if case == "merge-merged-non-string":
        first_merge["merged"] = 1
        return candidate

    if case == "merge-incoherent":
        first_merge["merged"] = f"{first_merge['merged']}x"
        return candidate

    if case == "merge-table-reordered":
        merges[0], merges[1] = merges[1], merges[0]
        return candidate

    if case == "top-level-parameter-not-list":
        weights["tokEmb"] = {}
        return candidate

    if case == "block-parameter-not-list":
        first_block["wQ"] = {}
        return candidate

    raise AssertionError(f"Unsupported structural test case: {case}")


def test_exact_valid_filename_selects_one_direct_ordinary_model_file(
    saved_model_path: Path,
) -> None:
    selected = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert saved_model_path.is_file()
    assert not saved_model_path.is_symlink()
    assert selected.model_filename == _CANONICAL_MODEL_FILENAME


def test_exact_current_phase_five_document_is_parsed_and_structurally_validated(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    loaded = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert saved_model_path.is_file()
    assert loaded.model_filename == _CANONICAL_MODEL_FILENAME
    assert loaded.config == saved_transformer_model["config"]
    assert loaded.config is not saved_transformer_model["config"]
    assert loaded.vocabulary == saved_transformer_model["vocab"]
    assert loaded.vocabulary is not saved_transformer_model["vocab"]
    assert loaded.merges == saved_transformer_model["merges"]
    assert loaded.merges is not saved_transformer_model["merges"]
    assert loaded.parameters.layout == build_transformer_parameter_layout(1)
    assert not hasattr(loaded, "weights")


def test_load_named_transformer_model_returns_complete_request_owned_snapshot(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    loaded = _load_named_model(_CANONICAL_MODEL_FILENAME)
    preprocessing = get_transformer_preprocessing()
    expected_layout = build_transformer_parameter_layout(1)

    assert saved_model_path.exists()

    assert loaded.model_filename == _CANONICAL_MODEL_FILENAME

    assert loaded.config == saved_transformer_model["config"]
    assert loaded.config is not saved_transformer_model["config"]

    assert loaded.vocabulary == saved_transformer_model["vocab"]
    assert loaded.vocabulary is not saved_transformer_model["vocab"]
    assert loaded.vocabulary is not preprocessing.vocabulary

    assert loaded.merges == saved_transformer_model["merges"]
    assert loaded.merges is not saved_transformer_model["merges"]
    assert loaded.merges is not preprocessing.merges

    assert loaded.parameters.layout == expected_layout
    assert loaded.parameters.layout.num_layers == 1

    assert loaded.parameters.storage.dtype == np.dtype(np.float32)
    assert loaded.parameters.storage.shape == (expected_layout.total_float_count,)
    assert loaded.parameters.storage.flags.c_contiguous
    assert loaded.parameters.storage.flags.writeable
    assert loaded.parameters.storage.flags.owndata

    assert loaded.parameters.views.layout == expected_layout
    assert len(loaded.parameters.views.blocks) == 1

    expected_storage = np.empty(
        expected_layout.total_float_count,
        dtype=np.float32,
        order="C",
    )

    for record in expected_layout.records:
        expected_values = np.asarray(
            _saved_values_for_record(
                saved_transformer_model,
                record,
            ),
            dtype=np.float32,
        ).reshape(
            record.shape,
            order="C",
        )

        loaded_view = loaded.parameters.views.get(
            record.key,
            record.block_index,
        )

        assert loaded_view.shape == record.shape
        assert loaded_view.dtype == np.dtype(np.float32)
        assert loaded_view.flags.c_contiguous
        assert loaded_view.flags.writeable

        assert np.shares_memory(
            loaded_view,
            loaded.parameters.storage,
        )

        assert loaded_view.ctypes.data == (
            loaded.parameters.storage.ctypes.data + record.byte_offset
        )

        np.testing.assert_array_equal(
            loaded_view,
            expected_values,
        )

        expected_storage[record.float_offset : record.float_stop] = expected_values.reshape(
            -1,
            order="C",
        )

    np.testing.assert_array_equal(
        loaded.parameters.storage,
        expected_storage,
    )

    assert loaded.parameters.views.tok_emb[0, 0] == np.float32(
        saved_transformer_model["weights"]["tokEmb"][0]
    )

    assert loaded.parameters.views.blocks[0].w_q[0, 0] == np.float32(
        saved_transformer_model["weights"]["blocks"][0]["wQ"][0]
    )

    assert loaded.parameters.views.head_b[-1] == np.float32(
        saved_transformer_model["weights"]["headB"][-1]
    )


def test_separate_named_loads_do_not_share_mutable_state(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    original_file_bytes = saved_model_path.read_bytes()
    preprocessing = get_transformer_preprocessing()

    first = _load_named_model(_CANONICAL_MODEL_FILENAME)
    second = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert first is not second

    assert first.config is not second.config
    assert first.vocabulary is not second.vocabulary
    assert first.merges is not second.merges

    assert first.parameters is not second.parameters
    assert first.parameters.storage is not second.parameters.storage
    assert first.parameters.views is not second.parameters.views

    np.testing.assert_array_equal(
        first.parameters.storage,
        second.parameters.storage,
    )

    for first_merge, second_merge in zip(
        first.merges,
        second.merges,
        strict=True,
    ):
        assert first_merge is not second_merge
        assert first_merge["pair"] is not second_merge["pair"]

    for record in first.parameters.layout.records:
        assert not np.shares_memory(
            first.parameters.views.get(
                record.key,
                record.block_index,
            ),
            second.parameters.views.get(
                record.key,
                record.block_index,
            ),
        )

    second_first_parameter = second.parameters.storage[0].copy()

    first.parameters.storage[0] = np.float32(first.parameters.storage[0] + 1.0)

    assert (
        first.parameters.views.tok_emb.reshape(
            -1,
            order="C",
        )[0]
        == first.parameters.storage[0]
    )

    assert second.parameters.storage[0] == second_first_parameter

    first.config["numLayers"] = 6
    first.vocabulary[0] = "mutated-token"
    first.merges[0]["pair"][0] = "mutated-merge-token"

    assert second.config == saved_transformer_model["config"]
    assert second.vocabulary == saved_transformer_model["vocab"]
    assert second.merges == saved_transformer_model["merges"]

    assert preprocessing.vocabulary[0] == saved_transformer_model["vocab"][0]

    assert preprocessing.merges[0].pair[0] == saved_transformer_model["merges"][0]["pair"][0]

    assert saved_model_path.read_bytes() == original_file_bytes

    assert json.loads(saved_model_path.read_text(encoding="utf-8")) == saved_transformer_model


@pytest.mark.parametrize(
    "record",
    build_transformer_parameter_layout(1).records,
    ids=lambda record: (
        f"block-{record.block_index}-{record.key}"
        if record.block_index is not None
        else f"top-level-{record.key}"
    ),
)
@pytest.mark.parametrize(
    "length_case",
    [
        "short",
        "long",
    ],
)
def test_named_loading_rejects_every_wrong_canonical_parameter_length(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
    record: TransformerParameterLayoutRecord,
    length_case: str,
) -> None:
    invalid_model = cast(
        SavedTransformerModel,
        deepcopy(saved_transformer_model),
    )
    values = _saved_values_for_record(
        invalid_model,
        record,
    )

    if length_case == "short":
        values.pop()
    elif length_case == "long":
        values.append(0.0)
    else:
        raise AssertionError(f"Unsupported length case: {length_case}")

    _write_json_value(saved_model_path, invalid_model)

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


@pytest.mark.parametrize(
    "record",
    [
        build_transformer_parameter_layout(1).get_record("tokEmb"),
        build_transformer_parameter_layout(1).get_record("wQ", 0),
    ],
    ids=[
        "top-level",
        "block",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="string"),
        pytest.param(None, id="null"),
        pytest.param([1.0], id="nested-list"),
        pytest.param({"value": 1.0}, id="nested-object"),
        pytest.param(1e39, id="float32-overflow"),
        pytest.param(10**1000, id="python-float-overflow"),
    ],
)
def test_named_loading_rejects_invalid_parameter_coordinates(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
    record: TransformerParameterLayoutRecord,
    invalid_value: object,
) -> None:
    invalid_model = cast(
        SavedTransformerModel,
        deepcopy(saved_transformer_model),
    )
    values = cast(
        list[object],
        _saved_values_for_record(
            invalid_model,
            record,
        ),
    )
    values[0] = invalid_value

    _write_json_value(saved_model_path, invalid_model)

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


def test_named_loading_preserves_signed_zero_during_float32_materialization(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    model_with_signed_zero = cast(
        SavedTransformerModel,
        deepcopy(saved_transformer_model),
    )
    model_with_signed_zero["weights"]["tokEmb"][0] = -0.0
    _write_json_value(saved_model_path, model_with_signed_zero)

    loaded = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert loaded.parameters.storage[0] == np.float32(0.0)
    assert np.signbit(loaded.parameters.storage[0])


def test_loaded_snapshot_does_not_alias_the_source_model_object(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    source_model = cast(
        SavedTransformerModel,
        deepcopy(saved_transformer_model),
    )
    _write_json_value(saved_model_path, source_model)

    loaded = _load_named_model(_CANONICAL_MODEL_FILENAME)
    loaded_first_parameter = loaded.parameters.storage[0].copy()

    source_model["config"]["numLayers"] = 6
    source_model["vocab"][0] = "mutated-source-token"
    source_model["merges"][0]["pair"][0] = "mutated-source-merge"
    source_model["weights"]["tokEmb"][0] = 999.0

    assert loaded.config["numLayers"] == 1
    assert loaded.vocabulary[0] == saved_transformer_model["vocab"][0]
    assert loaded.merges[0] == saved_transformer_model["merges"][0]
    assert loaded.parameters.storage[0] == loaded_first_parameter


def test_named_loading_rereads_changed_file_contents_on_the_next_call(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    first = _load_named_model(_CANONICAL_MODEL_FILENAME)
    first_coordinate = first.parameters.storage[0].copy()

    changed_model = cast(
        SavedTransformerModel,
        deepcopy(saved_transformer_model),
    )
    changed_value = float(changed_model["weights"]["tokEmb"][0]) + 0.25
    changed_model["weights"]["tokEmb"][0] = changed_value
    _write_json_value(saved_model_path, changed_model)

    second = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert first.parameters.storage[0] == first_coordinate
    assert second.parameters.storage[0] == np.float32(changed_value)
    assert second.parameters.storage[0] != first.parameters.storage[0]
    assert not np.shares_memory(
        first.parameters.storage,
        second.parameters.storage,
    )


def test_named_loading_read_once_uses_one_snapshot_and_rereads_without_cache(
    monkeypatch: pytest.MonkeyPatch,
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    original_read_bytes = Path.read_bytes
    original_value = np.float32(saved_transformer_model["weights"]["tokEmb"][0])

    changed_model = cast(
        SavedTransformerModel,
        deepcopy(saved_transformer_model),
    )
    changed_value = float(changed_model["weights"]["tokEmb"][0]) + 0.5
    changed_model["weights"]["tokEmb"][0] = changed_value
    changed_bytes = _serialize_model(changed_model)

    read_paths: list[Path] = []

    def read_one_snapshot(path: Path) -> bytes:
        read_paths.append(path)
        document_bytes = original_read_bytes(path)

        if path == saved_model_path and read_paths.count(saved_model_path) == 1:
            saved_model_path.write_bytes(changed_bytes)

        return document_bytes

    monkeypatch.setattr(
        Path,
        "read_bytes",
        read_one_snapshot,
    )

    first = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert read_paths == [saved_model_path]
    assert first.parameters.storage[0] == original_value
    assert original_read_bytes(saved_model_path) == changed_bytes

    second = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert read_paths == [
        saved_model_path,
        saved_model_path,
    ]
    assert second.parameters.storage[0] == np.float32(changed_value)
    assert second.parameters.storage[0] != first.parameters.storage[0]
    assert not np.shares_memory(
        first.parameters.storage,
        second.parameters.storage,
    )


def test_named_loading_has_no_application_file_size_limit(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    padded_document = (b" " * (2 * 1024 * 1024)) + _serialize_model(saved_transformer_model)

    saved_model_path.write_bytes(padded_document)
    before = _artifact_state(saved_model_path)

    loaded = _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert loaded.model_filename == _CANONICAL_MODEL_FILENAME
    assert loaded.parameters.storage[0] == np.float32(
        saved_transformer_model["weights"]["tokEmb"][0]
    )
    assert _artifact_state(saved_model_path) == before


def test_successful_named_loading_leaves_candidate_unchanged(
    saved_model_path: Path,
) -> None:
    before = _artifact_state(saved_model_path)
    before_entries = tuple(sorted(entry.name for entry in saved_model_path.parent.iterdir()))

    _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert saved_model_path.exists()
    assert _artifact_state(saved_model_path) == before
    assert (
        tuple(sorted(entry.name for entry in saved_model_path.parent.iterdir())) == before_entries
    )


def test_named_loading_sanitizes_memory_failure_and_leaves_candidate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    saved_model_path: Path,
) -> None:
    before = _artifact_state(saved_model_path)
    before_entries = tuple(sorted(entry.name for entry in saved_model_path.parent.iterdir()))
    original_read_bytes = Path.read_bytes

    def raise_memory_error(path: Path) -> bytes:
        if path == saved_model_path:
            raise MemoryError("sensitive allocation details")

        return original_read_bytes(path)

    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "read_bytes",
            raise_memory_error,
        )

        with pytest.raises(train_transformer_route.SavedTransformerModelLoadError) as captured:
            _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert type(captured.value) is train_transformer_route.SavedTransformerModelLoadError
    assert captured.value.args == (_PUBLIC_LOAD_FAILURE,)
    assert str(captured.value) == _PUBLIC_LOAD_FAILURE
    assert "sensitive allocation details" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__
    assert saved_model_path.exists()
    assert _artifact_state(saved_model_path) == before
    assert (
        tuple(sorted(entry.name for entry in saved_model_path.parent.iterdir())) == before_entries
    )


@pytest.mark.parametrize(
    "case",
    [
        "top-level-not-object",
        "top-level-reordered",
        "top-level-missing-key",
        "top-level-extra-key",
        "wrong-model-type",
        "config-not-object",
        "config-reordered",
        "config-missing-key",
        "config-extra-key",
        "config-boolean-integer",
        "config-fixed-architecture",
        "weights-not-object",
        "weights-reordered",
        "weights-missing-array",
        "weights-extra-array",
        "block-count",
        "block-not-object",
        "block-reordered",
        "block-missing-array",
        "block-extra-array",
        "vocab-not-list",
        "vocab-wrong-length",
        "vocab-reordered",
        "vocab-duplicate",
        "vocab-non-string",
        "merges-not-list",
        "merge-not-object",
        "merge-reordered",
        "merge-missing-key",
        "merge-extra-key",
        "merge-pair-not-list",
        "merge-pair-wrong-length",
        "merge-pair-non-string",
        "merge-merged-non-string",
        "merge-incoherent",
        "merge-table-reordered",
        "top-level-parameter-not-list",
        "block-parameter-not-list",
    ],
)
def test_named_loading_rejects_invalid_current_phase_five_structure(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
    case: str,
) -> None:
    invalid_model = _build_invalid_structural_model(
        saved_transformer_model,
        case,
    )
    _write_json_value(saved_model_path, invalid_model)

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


def test_named_loading_rejects_filename_config_layer_mismatch(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    invalid_model = cast(dict[str, Any], deepcopy(saved_transformer_model))
    config = cast(dict[str, Any], invalid_model["config"])
    weights = cast(dict[str, Any], invalid_model["weights"])
    blocks = cast(list[Any], weights["blocks"])

    config["numLayers"] = 2
    blocks.append(deepcopy(blocks[0]))
    _write_json_value(saved_model_path, invalid_model)

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


def test_named_loading_rejects_invalid_json_document(
    saved_model_path: Path,
) -> None:
    saved_model_path.write_bytes(b'{"type": "decoder-transformer"')

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


def test_named_loading_rejects_non_utf8_json_document(
    saved_model_path: Path,
) -> None:
    saved_model_path.write_bytes(b"\xff\xfe\xfa")

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


@pytest.mark.parametrize(
    "nonfinite_constant",
    [
        "NaN",
        "Infinity",
        "-Infinity",
    ],
)
def test_named_loading_rejects_nonstandard_nonfinite_json_constants(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
    nonfinite_constant: str,
) -> None:
    valid_document = _serialize_model(saved_transformer_model).decode("utf-8")
    first_weight = saved_transformer_model["weights"]["tokEmb"][0]
    serialized_weight = json.dumps(first_weight)

    invalid_document = valid_document.replace(
        serialized_weight,
        nonfinite_constant,
        1,
    )

    assert invalid_document != valid_document

    saved_model_path.write_text(
        invalid_document,
        encoding="utf-8",
        newline="\n",
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


@pytest.mark.parametrize(
    "model_filename",
    [
        "",
        " ",
        f"../{_CANONICAL_MODEL_FILENAME}",
        f"..\\{_CANONICAL_MODEL_FILENAME}",
        f"subdirectory/{_CANONICAL_MODEL_FILENAME}",
        f"subdirectory\\{_CANONICAL_MODEL_FILENAME}",
        f"C:\\models\\{_CANONICAL_MODEL_FILENAME}",
        f"/models/{_CANONICAL_MODEL_FILENAME}",
        f"\\models\\{_CANONICAL_MODEL_FILENAME}",
        f"C:{_CANONICAL_MODEL_FILENAME}",
        f"\\\\server\\share\\{_CANONICAL_MODEL_FILENAME}",
        f"//server/share/{_CANONICAL_MODEL_FILENAME}",
        _CANONICAL_MODEL_FILENAME.replace(
            ".json",
            ".txt",
        ),
        _CANONICAL_MODEL_FILENAME.replace(
            ".json",
            ".JSON",
        ),
        "transformer-weights-e50-l1-d32-h2-ff128.json",
        "transformer-weights-e50-l1-d32-h2-ff128-ctx32-extra.json",
        "transformer-weights-e50-l1-d64-h2-ff128-ctx32.json",
        "transformer-weights-e50-l1-d32-h4-ff128-ctx32.json",
        "transformer-weights-e50-l1-d32-h2-ff256-ctx32.json",
        "transformer-weights-e50-l1-d32-h2-ff128-ctx64.json",
        "transformer-weights-e49-l1-d32-h2-ff128-ctx32.json",
        "transformer-weights-e2001-l1-d32-h2-ff128-ctx32.json",
        "transformer-weights-e50-l0-d32-h2-ff128-ctx32.json",
        "transformer-weights-e50-l7-d32-h2-ff128-ctx32.json",
        "transformer-weights-e050-l1-d32-h2-ff128-ctx32.json",
        "transformer-weights-e50-l01-d32-h2-ff128-ctx32.json",
        "Transformer-weights-e50-l1-d32-h2-ff128-ctx32.json",
    ],
    ids=[
        "empty",
        "whitespace",
        "parent-forward-slash",
        "parent-backslash",
        "nested-forward-slash",
        "nested-backslash",
        "drive-letter",
        "absolute-posix",
        "rooted-windows",
        "drive-relative",
        "unc-backslash",
        "unc-forward-slash",
        "wrong-extension",
        "wrong-extension-case",
        "missing-architecture-segment",
        "extra-architecture-segment",
        "unsupported-embedding-dimension",
        "unsupported-head-count",
        "unsupported-feed-forward-dimension",
        "unsupported-context-length",
        "epoch-below-minimum",
        "epoch-above-maximum",
        "layer-below-minimum",
        "layer-above-maximum",
        "leading-zero-epoch",
        "leading-zero-layer",
        "exact-case-mismatch",
    ],
)
def test_named_loading_rejects_noncanonical_or_unsafe_filenames_without_mutation(
    saved_model_path: Path,
    model_filename: str,
) -> None:
    _assert_public_load_rejection(
        model_filename,
        saved_model_path,
    )


def test_named_loading_rejects_missing_exact_file_without_fallback(
    saved_model_path: Path,
) -> None:
    missing_filename = "transformer-weights-e100-l1-d32-h2-ff128-ctx32.json"

    _assert_public_load_rejection(
        missing_filename,
        saved_model_path,
    )


def test_named_loading_rejects_directory_masquerading_as_model(
    saved_model_path: Path,
) -> None:
    saved_model_path.unlink()
    saved_model_path.mkdir()

    before = saved_model_path.stat()
    before_entries = tuple(sorted(entry.name for entry in saved_model_path.parent.iterdir()))

    with pytest.raises(train_transformer_route.SavedTransformerModelLoadError) as captured:
        _load_named_model(_CANONICAL_MODEL_FILENAME)

    assert type(captured.value) is train_transformer_route.SavedTransformerModelLoadError
    assert captured.value.args == (_PUBLIC_LOAD_FAILURE,)
    assert str(captured.value) == _PUBLIC_LOAD_FAILURE
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__
    assert saved_model_path.is_dir()

    after = saved_model_path.stat()

    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns == before.st_ctime_ns
    assert after.st_ino == before.st_ino
    assert (
        tuple(sorted(entry.name for entry in saved_model_path.parent.iterdir())) == before_entries
    )


def test_named_loading_rejects_candidate_symlink_and_containment_escape(
    model_directory: Path,
    saved_model_path: Path,
) -> None:
    outside_model = model_directory.parent / "outside-transformer-model.json"
    outside_model.write_bytes(saved_model_path.read_bytes())
    saved_model_path.unlink()

    _create_symbolic_link_or_skip(
        saved_model_path,
        outside_model,
        target_is_directory=False,
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        outside_model,
    )

    assert saved_model_path.is_symlink()


def test_named_loading_rejects_symbolic_model_directory(
    monkeypatch: pytest.MonkeyPatch,
    model_directory: Path,
    saved_model_path: Path,
) -> None:
    linked_directory = model_directory.parent / "linked-model-directory"

    _create_symbolic_link_or_skip(
        linked_directory,
        model_directory,
        target_is_directory=True,
    )

    monkeypatch.setattr(
        train_transformer_route,
        "get_transformer_model_directory",
        lambda: linked_directory,
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )

    assert linked_directory.is_symlink()


@pytest.mark.parametrize(
    "classified_target",
    [
        "model-directory",
        "candidate",
    ],
)
def test_named_loading_rejects_controlled_junction_classification(
    monkeypatch: pytest.MonkeyPatch,
    model_directory: Path,
    saved_model_path: Path,
    classified_target: str,
) -> None:
    original_is_junction = Path.is_junction

    target_path = model_directory if classified_target == "model-directory" else saved_model_path

    def controlled_is_junction(path: Path) -> bool:
        if path == target_path:
            return True

        return original_is_junction(path)

    monkeypatch.setattr(
        Path,
        "is_junction",
        controlled_is_junction,
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


def test_named_loading_rejects_real_windows_junction_model_directory(
    monkeypatch: pytest.MonkeyPatch,
    model_directory: Path,
    saved_model_path: Path,
) -> None:
    junction_directory = model_directory.parent / "real-junction-model-directory"

    try:
        _create_windows_junction_or_skip(
            junction_directory,
            model_directory,
        )

        monkeypatch.setattr(
            train_transformer_route,
            "get_transformer_model_directory",
            lambda: junction_directory,
        )

        _assert_public_load_rejection(
            _CANONICAL_MODEL_FILENAME,
            saved_model_path,
        )

        assert junction_directory.is_junction()
    finally:
        if junction_directory.is_junction():
            junction_directory.rmdir()


def test_named_loading_rejects_missing_model_directory(
    monkeypatch: pytest.MonkeyPatch,
    model_directory: Path,
    saved_model_path: Path,
) -> None:
    missing_directory = model_directory.parent / "missing-model-directory"

    monkeypatch.setattr(
        train_transformer_route,
        "get_transformer_model_directory",
        lambda: missing_directory,
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )

    assert not missing_directory.exists()


def test_named_loading_rejects_ordinary_file_as_model_directory(
    monkeypatch: pytest.MonkeyPatch,
    model_directory: Path,
) -> None:
    directory_file = model_directory.parent / "model-directory-file"
    directory_file.write_bytes(b"not a directory\n")

    monkeypatch.setattr(
        train_transformer_route,
        "get_transformer_model_directory",
        lambda: directory_file,
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        directory_file,
    )


def test_named_loading_rejects_special_file_candidate_when_supported(
    model_directory: Path,
) -> None:
    make_fifo = getattr(os, "mkfifo", None)

    if make_fifo is None:
        pytest.skip("This platform cannot create a filesystem FIFO.")

    fifo_filename = "transformer-weights-e100-l1-d32-h2-ff128-ctx32.json"
    fifo_path = model_directory / fifo_filename

    try:
        make_fifo(fifo_path)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"This environment cannot create a filesystem FIFO: {error}")

    before = fifo_path.lstat()
    before_entries = tuple(sorted(entry.name for entry in model_directory.iterdir()))

    with pytest.raises(train_transformer_route.SavedTransformerModelLoadError) as captured:
        _load_named_model(fifo_filename)

    assert type(captured.value) is train_transformer_route.SavedTransformerModelLoadError
    assert captured.value.args == (_PUBLIC_LOAD_FAILURE,)
    assert str(captured.value) == _PUBLIC_LOAD_FAILURE
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__

    after = fifo_path.lstat()

    assert stat.S_ISFIFO(after.st_mode)
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns == before.st_ctime_ns
    assert after.st_ino == before.st_ino
    assert tuple(sorted(entry.name for entry in model_directory.iterdir())) == before_entries


def test_named_loading_rejects_duplicate_json_object_keys_without_mutation(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
) -> None:
    valid_document = _serialize_model(saved_transformer_model).decode("utf-8")

    duplicate_key_document = valid_document.replace(
        '  "type": "decoder-transformer",',
        ('  "type": "decoder-transformer",\n' '  "type": "decoder-transformer",'),
        1,
    )

    assert duplicate_key_document != valid_document
    assert duplicate_key_document.count('"type": "decoder-transformer"') == 2

    saved_model_path.write_text(
        duplicate_key_document,
        encoding="utf-8",
        newline="\n",
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )


@pytest.mark.parametrize(
    "scope",
    [
        "config",
        "weights",
        "block",
    ],
)
def test_named_loading_rejects_nested_duplicate_json_object_keys_without_mutation(
    saved_model_path: Path,
    saved_transformer_model: SavedTransformerModel,
    scope: str,
) -> None:
    valid_document = _serialize_model(saved_transformer_model).decode("utf-8")

    if scope == "config":
        vocab_size = saved_transformer_model["config"]["vocabSize"]
        target = f'    "vocabSize": {vocab_size},'
        duplicate_key_document = valid_document.replace(
            target,
            f"{target}\n{target}",
            1,
        )
    elif scope == "weights":
        duplicate_key_document = valid_document.replace(
            '    "blocks": [',
            '    "blocks": [],\n    "blocks": [',
            1,
        )
    elif scope == "block":
        duplicate_key_document = valid_document.replace(
            '        "ln1Gamma": [',
            '        "ln1Gamma": [],\n        "ln1Gamma": [',
            1,
        )
    else:
        raise AssertionError(f"Unsupported duplicate-key scope: {scope}")

    assert duplicate_key_document != valid_document

    saved_model_path.write_text(
        duplicate_key_document,
        encoding="utf-8",
        newline="\n",
    )

    _assert_public_load_rejection(
        _CANONICAL_MODEL_FILENAME,
        saved_model_path,
    )
