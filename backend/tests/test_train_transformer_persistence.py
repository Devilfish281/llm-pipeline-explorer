# backend/tests/test_train_transformer_persistence.py
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import cast

import pytest
from how_llms_work.ml.transformer import SavedTransformerBlockWeights, SavedTransformerModel
from how_llms_work.routes import train_transformer as train_transformer_route

_CONFIG_KEYS = (
    "vocabSize",
    "contextLen",
    "embDim",
    "numHeads",
    "ffDim",
    "numLayers",
)

_WEIGHT_KEYS = (
    "tokEmb",
    "posEmb",
    "blocks",
    "lnFGamma",
    "lnFBeta",
    "headW",
    "headB",
)

_BLOCK_KEYS = (
    "ln1Gamma",
    "ln1Beta",
    "wQ",
    "bQ",
    "wK",
    "bK",
    "wV",
    "bV",
    "wO",
    "bO",
    "ln2Gamma",
    "ln2Beta",
    "ff1W",
    "ff1B",
    "ff2W",
    "ff2B",
)


def _saved_transformer_block(marker: float) -> SavedTransformerBlockWeights:
    return {
        "ln1Gamma": [marker + 0.01],
        "ln1Beta": [marker + 0.02],
        "wQ": [marker + 0.03],
        "bQ": [marker + 0.04],
        "wK": [marker + 0.05],
        "bK": [marker + 0.06],
        "wV": [marker + 0.07],
        "bV": [marker + 0.08],
        "wO": [marker + 0.09],
        "bO": [marker + 0.10],
        "ln2Gamma": [marker + 0.11],
        "ln2Beta": [marker + 0.12],
        "ff1W": [marker + 0.13],
        "ff1B": [marker + 0.14],
        "ff2W": [marker + 0.15],
        "ff2B": [marker + 0.16],
    }


def _saved_transformer_model(
    *,
    num_layers: int,
    marker: float,
) -> SavedTransformerModel:
    return {
        "type": "decoder-transformer",
        "config": {
            "vocabSize": 3,
            "contextLen": 32,
            "embDim": 32,
            "numHeads": 2,
            "ffDim": 128,
            "numLayers": num_layers,
        },
        "vocab": [
            "<pad>",
            "alpha",
            "beta",
        ],
        "merges": [
            {
                "pair": ["a", "l"],
                "merged": "al",
            },
            {
                "pair": ["al", "pha"],
                "merged": "alpha",
            },
        ],
        "weights": {
            "tokEmb": [marker + 0.21, marker + 0.22],
            "posEmb": [marker + 0.23, marker + 0.24],
            "blocks": [
                _saved_transformer_block(marker + float(block_index))
                for block_index in range(num_layers)
            ],
            "lnFGamma": [marker + 0.25],
            "lnFBeta": [marker + 0.26],
            "headW": [marker + 0.27, marker + 0.28],
            "headB": [marker + 0.29],
        },
    }


ONE_LAYER_MODEL = _saved_transformer_model(
    num_layers=1,
    marker=1.0,
)

TWO_LAYER_MODEL = _saved_transformer_model(
    num_layers=2,
    marker=10.0,
)

SIX_LAYER_MODEL = _saved_transformer_model(
    num_layers=6,
    marker=20.0,
)

SECOND_TWO_LAYER_MODEL = _saved_transformer_model(
    num_layers=2,
    marker=30.0,
)


_PREVIOUS_DESTINATION_DOCUMENT = b'{"previous":true}\n'

_TWO_LAYER_DESTINATION_FILENAME = "transformer-weights-e300-l2-d32-h2-ff128-ctx32.json"


def _expected_document(model: SavedTransformerModel) -> str:
    return f"{json.dumps(model, indent=2, allow_nan=False)}\n"


def _temporary_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix == ".tmp")


def _seed_previous_destination(directory: Path) -> tuple[Path, Path]:
    destination = directory / _TWO_LAYER_DESTINATION_FILENAME
    unrelated_temporary = directory / ".unrelated.tmp"

    destination.write_bytes(_PREVIOUS_DESTINATION_DOCUMENT)
    unrelated_temporary.write_text(
        "unrelated",
        encoding="utf-8",
    )

    return destination, unrelated_temporary


def _raise_document_stage_failure(
    file_descriptor: int,
    document: str,
    *,
    failure_stage: str,
    persistence_error: OSError,
) -> None:
    with os.fdopen(
        file_descriptor,
        mode="w",
        encoding="utf-8",
        newline="\n",
    ) as writer:
        if failure_stage == "write":
            writer.write(document[: max(1, len(document) // 2)])
            raise persistence_error

        writer.write(document)

        if failure_stage == "flush":
            raise persistence_error

        writer.flush()

        if failure_stage == "fsync":
            raise persistence_error

        os.fsync(writer.fileno())

    if failure_stage == "close":
        raise persistence_error


def test_transformer_model_directory_is_derived_from_backend_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = Path(train_transformer_route.__file__).resolve().parents[3] / ".data"

    monkeypatch.chdir(tmp_path)

    assert train_transformer_route.get_transformer_model_directory() == expected
    assert train_transformer_route.get_transformer_model_directory() != Path.cwd() / ".data"


@pytest.mark.parametrize(
    (
        "model",
        "epochs",
        "expected_filename",
    ),
    [
        (
            ONE_LAYER_MODEL,
            50,
            "transformer-weights-e50-l1-d32-h2-ff128-ctx32.json",
        ),
        (
            TWO_LAYER_MODEL,
            300,
            "transformer-weights-e300-l2-d32-h2-ff128-ctx32.json",
        ),
        (
            SIX_LAYER_MODEL,
            2000,
            "transformer-weights-e2000-l6-d32-h2-ff128-ctx32.json",
        ),
    ],
    ids=[
        "minimum-epochs-one-layer",
        "reference-epochs-two-layers",
        "maximum-epochs-six-layers",
    ],
)
def test_prepare_transformer_model_persistence_uses_exact_configuration_filename(
    tmp_path: Path,
    model: SavedTransformerModel,
    epochs: int,
    expected_filename: str,
) -> None:
    model_directory = tmp_path / "missing" / ".data"

    destination, document = train_transformer_route.prepare_transformer_model_persistence(
        model,
        epochs=epochs,
        model_directory=model_directory,
    )

    assert destination == model_directory / expected_filename
    assert (
        train_transformer_route.get_transformer_model_filename(
            model,
            epochs=epochs,
        )
        == expected_filename
    )
    assert document == _expected_document(model)
    assert not model_directory.exists()


def test_serialize_saved_transformer_model_creates_exact_ordered_document(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "missing" / ".data"

    destination, prepared_document = train_transformer_route.prepare_transformer_model_persistence(
        TWO_LAYER_MODEL,
        epochs=300,
        model_directory=model_directory,
    )
    serialized_document = train_transformer_route.serialize_saved_transformer_model(TWO_LAYER_MODEL)

    expected_document = _expected_document(TWO_LAYER_MODEL)
    raw_bytes = serialized_document.encode("utf-8")
    parsed_document = json.loads(serialized_document)

    assert destination == (model_directory / "transformer-weights-e300-l2-d32-h2-ff128-ctx32.json")
    assert prepared_document == expected_document
    assert serialized_document == expected_document
    assert b"\r\n" not in raw_bytes
    assert serialized_document.endswith("\n")
    assert not serialized_document.endswith("\n\n")
    assert list(parsed_document) == [
        "type",
        "config",
        "vocab",
        "merges",
        "weights",
    ]
    assert list(parsed_document["config"]) == list(_CONFIG_KEYS)
    assert list(parsed_document["weights"]) == list(_WEIGHT_KEYS)
    assert len(parsed_document["weights"]["blocks"]) == 2
    assert all(list(block) == list(_BLOCK_KEYS) for block in parsed_document["weights"]["blocks"])
    assert parsed_document == TWO_LAYER_MODEL
    assert not model_directory.exists()


def test_save_transformer_model_creates_missing_directory_and_exact_document(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "missing" / ".data"

    destination = train_transformer_route.save_transformer_model(
        TWO_LAYER_MODEL,
        epochs=300,
        model_directory=model_directory,
    )

    expected_document = _expected_document(TWO_LAYER_MODEL)
    expected_destination = model_directory / "transformer-weights-e300-l2-d32-h2-ff128-ctx32.json"

    assert destination == expected_destination
    assert model_directory.is_dir()
    assert destination.read_bytes() == expected_document.encode("utf-8")
    assert _temporary_files(model_directory) == []


def test_save_transformer_model_flushes_fsyncs_and_closes_before_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_fsync = train_transformer_route.os.fsync
    original_replace = train_transformer_route.replace_transformer_model_file
    expected_document = _expected_document(TWO_LAYER_MODEL)
    expected_size = len(expected_document.encode("utf-8"))
    observed_events: list[str] = []
    observed_temporary_paths: list[Path] = []

    def observing_fsync(file_descriptor: int) -> None:
        temporary_paths = _temporary_files(tmp_path)

        assert len(temporary_paths) == 1

        temporary_path = temporary_paths[0]
        assert temporary_path.stat().st_size == expected_size

        observed_temporary_paths.append(temporary_path)
        observed_events.append("fsync")
        original_fsync(file_descriptor)

    def observing_replace(
        source: Path,
        destination: Path,
    ) -> None:
        assert observed_events == ["fsync"]
        assert observed_temporary_paths == [source]

        probe = source.with_suffix(".probe")
        os.replace(source, probe)
        os.replace(probe, source)

        observed_events.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(
        train_transformer_route.os,
        "fsync",
        observing_fsync,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "replace_transformer_model_file",
        observing_replace,
    )

    destination = train_transformer_route.save_transformer_model(
        TWO_LAYER_MODEL,
        epochs=300,
        model_directory=tmp_path,
    )

    assert observed_events == ["fsync", "replace"]
    assert len(observed_temporary_paths) == 1

    temporary_path = observed_temporary_paths[0]
    assert temporary_path != destination
    assert temporary_path.parent == destination.parent == tmp_path
    assert destination.read_bytes() == expected_document.encode("utf-8")
    assert _temporary_files(tmp_path) == []


def test_save_transformer_model_uses_unique_same_directory_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_replace = train_transformer_route.replace_transformer_model_file
    observed_temporary_paths: list[Path] = []

    def observing_replace(
        source: Path,
        destination: Path,
    ) -> None:
        observed_temporary_paths.append(source)
        original_replace(source, destination)

    monkeypatch.setattr(
        train_transformer_route,
        "replace_transformer_model_file",
        observing_replace,
    )

    first_destination = train_transformer_route.save_transformer_model(
        ONE_LAYER_MODEL,
        epochs=50,
        model_directory=tmp_path,
    )
    second_destination = train_transformer_route.save_transformer_model(
        ONE_LAYER_MODEL,
        epochs=50,
        model_directory=tmp_path,
    )

    assert first_destination == second_destination
    assert len(observed_temporary_paths) == 2
    assert len(set(observed_temporary_paths)) == 2
    assert all(temporary_path.parent == tmp_path for temporary_path in observed_temporary_paths)
    assert all(temporary_path.suffix == ".tmp" for temporary_path in observed_temporary_paths)
    assert all(temporary_path != first_destination for temporary_path in observed_temporary_paths)
    assert first_destination.read_bytes() == _expected_document(ONE_LAYER_MODEL).encode("utf-8")
    assert _temporary_files(tmp_path) == []


@pytest.mark.parametrize(
    (
        "invalid_epochs",
        "expected_error",
    ),
    [
        (True, TypeError),
        (50.0, TypeError),
        ("300", TypeError),
        (49, ValueError),
        (2001, ValueError),
    ],
    ids=[
        "boolean",
        "fractional-type",
        "string",
        "below-minimum",
        "above-maximum",
    ],
)
def test_prepare_transformer_model_rejects_invalid_epochs_before_filesystem_changes(
    tmp_path: Path,
    invalid_epochs: object,
    expected_error: type[Exception],
) -> None:
    model_directory = tmp_path / "missing" / ".data"

    with pytest.raises(expected_error):
        train_transformer_route.prepare_transformer_model_persistence(
            ONE_LAYER_MODEL,
            epochs=cast(int, invalid_epochs),
            model_directory=model_directory,
        )

    assert not model_directory.exists()


@pytest.mark.parametrize(
    (
        "invalid_num_layers",
        "expected_error",
    ),
    [
        (True, TypeError),
        (1.0, TypeError),
        ("2", TypeError),
        (0, ValueError),
        (7, ValueError),
    ],
    ids=[
        "boolean",
        "fractional-type",
        "string",
        "below-minimum",
        "above-maximum",
    ],
)
def test_prepare_transformer_model_rejects_invalid_layers_before_filesystem_changes(
    tmp_path: Path,
    invalid_num_layers: object,
    expected_error: type[Exception],
) -> None:
    model_directory = tmp_path / "missing" / ".data"
    invalid_model = deepcopy(ONE_LAYER_MODEL)
    invalid_model["config"]["numLayers"] = cast(
        int,
        invalid_num_layers,
    )

    with pytest.raises(expected_error):
        train_transformer_route.prepare_transformer_model_persistence(
            invalid_model,
            epochs=50,
            model_directory=model_directory,
        )

    assert not model_directory.exists()


@pytest.mark.parametrize(
    (
        "config_key",
        "invalid_value",
    ),
    [
        ("contextLen", 31),
        ("embDim", 31),
        ("numHeads", 1),
        ("ffDim", 127),
    ],
    ids=[
        "context-length",
        "embedding-dimension",
        "attention-head-count",
        "feed-forward-dimension",
    ],
)
def test_prepare_transformer_model_rejects_configuration_mismatch_before_filesystem_changes(
    tmp_path: Path,
    config_key: str,
    invalid_value: int,
) -> None:
    model_directory = tmp_path / "missing" / ".data"
    invalid_model = deepcopy(TWO_LAYER_MODEL)
    config = cast(dict[str, int], invalid_model["config"])
    config[config_key] = invalid_value

    with pytest.raises(ValueError):
        train_transformer_route.prepare_transformer_model_persistence(
            invalid_model,
            epochs=300,
            model_directory=model_directory,
        )

    assert not model_directory.exists()


@pytest.mark.parametrize(
    "non_finite_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_prepare_transformer_model_rejects_non_finite_values_before_filesystem_changes(
    tmp_path: Path,
    non_finite_value: float,
) -> None:
    model_directory = tmp_path / "missing" / ".data"
    invalid_model = deepcopy(TWO_LAYER_MODEL)
    invalid_model["weights"]["headB"][0] = non_finite_value

    with pytest.raises(ValueError):
        train_transformer_route.prepare_transformer_model_persistence(
            invalid_model,
            epochs=300,
            model_directory=model_directory,
        )

    assert not model_directory.exists()


def test_prepare_transformer_model_rejects_unsupported_objects_before_filesystem_changes(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "missing" / ".data"
    invalid_model = deepcopy(TWO_LAYER_MODEL)
    invalid_model["weights"]["headB"] = cast(
        list[float],
        [object()],
    )

    with pytest.raises(TypeError):
        train_transformer_route.prepare_transformer_model_persistence(
            invalid_model,
            epochs=300,
            model_directory=model_directory,
        )

    assert not model_directory.exists()


def test_prepare_transformer_model_rejects_malformed_structure_before_filesystem_changes(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "missing" / ".data"
    invalid_model = deepcopy(TWO_LAYER_MODEL)
    invalid_model_mapping = cast(dict[str, object], invalid_model)
    invalid_model_mapping.pop("weights")

    with pytest.raises(ValueError):
        train_transformer_route.prepare_transformer_model_persistence(
            invalid_model,
            epochs=300,
            model_directory=model_directory,
        )

    assert not model_directory.exists()


def test_prepare_transformer_model_rejects_block_count_mismatch_before_filesystem_changes(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "missing" / ".data"
    invalid_model = deepcopy(TWO_LAYER_MODEL)
    invalid_model["weights"]["blocks"].pop()

    with pytest.raises(ValueError):
        train_transformer_route.prepare_transformer_model_persistence(
            invalid_model,
            epochs=300,
            model_directory=model_directory,
        )

    assert not model_directory.exists()


@pytest.mark.parametrize(
    (
        "failure_stage",
        "expected_error",
    ),
    [
        ("configuration", ValueError),
        ("serialization", ValueError),
        ("directory", OSError),
        ("temporary", OSError),
        ("write", OSError),
        ("flush", OSError),
        ("fsync", OSError),
        ("close", OSError),
        ("replace", OSError),
    ],
)
def test_failed_save_preserves_destination_and_cleans_only_owned_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
    expected_error: type[Exception],
) -> None:
    destination, unrelated_temporary = _seed_previous_destination(tmp_path)
    model = deepcopy(TWO_LAYER_MODEL)
    persistence_error = OSError(f"{failure_stage} failed")
    owned_temporary_paths: list[Path] = []

    original_create = train_transformer_route.create_temporary_transformer_model_file

    def observing_create(
        directory: Path,
        final_destination: Path,
    ) -> tuple[int, Path]:
        file_descriptor, temporary_path = original_create(
            directory,
            final_destination,
        )
        owned_temporary_paths.append(temporary_path)

        return file_descriptor, temporary_path

    monkeypatch.setattr(
        train_transformer_route,
        "create_temporary_transformer_model_file",
        observing_create,
    )

    if failure_stage == "configuration":
        model["config"]["contextLen"] = 31

    elif failure_stage == "serialization":
        model["weights"]["headB"][0] = float("nan")

    elif failure_stage == "directory":

        def fail_mkdir(
            _: Path,
            mode: int = 0o777,
            parents: bool = False,
            exist_ok: bool = False,
        ) -> None:
            del mode, parents, exist_ok
            raise persistence_error

        monkeypatch.setattr(
            train_transformer_route.Path,
            "mkdir",
            fail_mkdir,
        )

    elif failure_stage == "temporary":

        def fail_temporary_creation(
            _: Path,
            __: Path,
        ) -> tuple[int, Path]:
            raise persistence_error

        monkeypatch.setattr(
            train_transformer_route,
            "create_temporary_transformer_model_file",
            fail_temporary_creation,
        )

    elif failure_stage == "replace":

        def fail_replace(
            _: Path,
            __: Path,
        ) -> None:
            raise persistence_error

        monkeypatch.setattr(
            train_transformer_route,
            "replace_transformer_model_file",
            fail_replace,
        )

    else:

        def fail_document(
            file_descriptor: int,
            document: str,
        ) -> None:
            _raise_document_stage_failure(
                file_descriptor,
                document,
                failure_stage=failure_stage,
                persistence_error=persistence_error,
            )

        monkeypatch.setattr(
            train_transformer_route,
            "write_transformer_model_document",
            fail_document,
        )

    with pytest.raises(expected_error) as error_info:
        train_transformer_route.save_transformer_model(
            model,
            epochs=300,
            model_directory=tmp_path,
        )

    if isinstance(error_info.value, OSError):
        assert error_info.value is persistence_error

    assert destination.read_bytes() == _PREVIOUS_DESTINATION_DOCUMENT
    assert unrelated_temporary.read_text(encoding="utf-8") == "unrelated"

    if failure_stage in {
        "write",
        "flush",
        "fsync",
        "close",
        "replace",
    }:
        assert len(owned_temporary_paths) == 1
        assert not owned_temporary_paths[0].exists()
    else:
        assert owned_temporary_paths == []

    assert _temporary_files(tmp_path) == [unrelated_temporary]


def test_cleanup_failure_preserves_destination_and_reports_both_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination, unrelated_temporary = _seed_previous_destination(tmp_path)
    persistence_error = OSError("replacement failed")
    cleanup_error = PermissionError("cleanup failed")
    owned_temporary_paths: list[Path] = []

    original_create = train_transformer_route.create_temporary_transformer_model_file

    def observing_create(
        directory: Path,
        final_destination: Path,
    ) -> tuple[int, Path]:
        file_descriptor, temporary_path = original_create(
            directory,
            final_destination,
        )
        owned_temporary_paths.append(temporary_path)

        return file_descriptor, temporary_path

    def fail_replace(
        _: Path,
        __: Path,
    ) -> None:
        raise persistence_error

    def fail_cleanup(path: Path) -> None:
        assert owned_temporary_paths == [path]
        raise cleanup_error

    monkeypatch.setattr(
        train_transformer_route,
        "create_temporary_transformer_model_file",
        observing_create,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "replace_transformer_model_file",
        fail_replace,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "remove_temporary_transformer_model",
        fail_cleanup,
    )

    with pytest.raises(ExceptionGroup) as error_info:
        train_transformer_route.save_transformer_model(
            TWO_LAYER_MODEL,
            epochs=300,
            model_directory=tmp_path,
        )

    assert error_info.value.exceptions == (
        persistence_error,
        cleanup_error,
    )
    assert destination.read_bytes() == _PREVIOUS_DESTINATION_DOCUMENT
    assert len(owned_temporary_paths) == 1
    assert owned_temporary_paths[0].exists()
    assert _temporary_files(tmp_path) == sorted(
        [
            unrelated_temporary,
            owned_temporary_paths[0],
        ]
    )


def test_concurrent_different_configuration_saves_remain_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_replace = train_transformer_route.replace_transformer_model_file
    replacement_barrier = Barrier(2)
    replacement_lock = Lock()
    observed_paths: list[tuple[Path, Path]] = []

    def synchronized_replace(source: Path, destination: Path) -> None:
        with replacement_lock:
            observed_paths.append((source, destination))

        replacement_barrier.wait(timeout=5)
        original_replace(source, destination)

    monkeypatch.setattr(
        train_transformer_route,
        "replace_transformer_model_file",
        synchronized_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        one_layer_future = executor.submit(
            train_transformer_route.save_transformer_model,
            deepcopy(ONE_LAYER_MODEL),
            epochs=50,
            model_directory=tmp_path,
        )
        two_layer_future = executor.submit(
            train_transformer_route.save_transformer_model,
            deepcopy(TWO_LAYER_MODEL),
            epochs=300,
            model_directory=tmp_path,
        )

        one_layer_destination = one_layer_future.result(timeout=5)
        two_layer_destination = two_layer_future.result(timeout=5)

    sources = [source for source, _ in observed_paths]
    destinations = [destination for _, destination in observed_paths]

    assert one_layer_destination != two_layer_destination
    assert len(set(sources)) == 2
    assert set(destinations) == {
        one_layer_destination,
        two_layer_destination,
    }
    assert all(
        source.parent == destination.parent == tmp_path for source, destination in observed_paths
    )
    assert one_layer_destination.read_text(encoding="utf-8") == _expected_document(ONE_LAYER_MODEL)
    assert two_layer_destination.read_text(encoding="utf-8") == _expected_document(TWO_LAYER_MODEL)
    assert _temporary_files(tmp_path) == []


@pytest.mark.parametrize(
    "last_replacer",
    [
        "first",
        "second",
    ],
)
def test_concurrent_same_configuration_saves_use_controlled_last_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    last_replacer: str,
) -> None:
    first_model = deepcopy(TWO_LAYER_MODEL)
    second_model = deepcopy(SECOND_TWO_LAYER_MODEL)
    original_replace = train_transformer_route.replace_transformer_model_file
    first_ready = Event()
    second_ready = Event()
    release_first = Event()
    release_second = Event()

    def controlled_replace(source: Path, destination: Path) -> None:
        document = json.loads(source.read_text(encoding="utf-8"))
        head_bias = document["weights"]["headB"][0]

        if head_bias == first_model["weights"]["headB"][0]:
            first_ready.set()
            assert release_first.wait(timeout=5)
        else:
            second_ready.set()
            assert release_second.wait(timeout=5)

        original_replace(source, destination)

    monkeypatch.setattr(
        train_transformer_route,
        "replace_transformer_model_file",
        controlled_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            train_transformer_route.save_transformer_model,
            first_model,
            epochs=300,
            model_directory=tmp_path,
        )
        second_future = executor.submit(
            train_transformer_route.save_transformer_model,
            second_model,
            epochs=300,
            model_directory=tmp_path,
        )

        assert first_ready.wait(timeout=5)
        assert second_ready.wait(timeout=5)

        if last_replacer == "first":
            release_second.set()
            second_future.result(timeout=5)

            release_first.set()
            destination = first_future.result(timeout=5)
            expected_model = first_model
        else:
            release_first.set()
            first_future.result(timeout=5)

            release_second.set()
            destination = second_future.result(timeout=5)
            expected_model = second_model

    assert destination.read_text(encoding="utf-8") == _expected_document(expected_model)
    assert _temporary_files(tmp_path) == []


def test_failed_concurrent_same_configuration_save_cannot_damage_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    successful_model = deepcopy(TWO_LAYER_MODEL)
    failing_model = deepcopy(SECOND_TWO_LAYER_MODEL)
    original_replace = train_transformer_route.replace_transformer_model_file
    success_ready = Event()
    failure_ready = Event()
    release_success = Event()
    release_failure = Event()

    def controlled_replace(source: Path, destination: Path) -> None:
        document = json.loads(source.read_text(encoding="utf-8"))
        head_bias = document["weights"]["headB"][0]

        if head_bias == successful_model["weights"]["headB"][0]:
            success_ready.set()
            assert release_success.wait(timeout=5)
            original_replace(source, destination)
            return

        failure_ready.set()
        assert release_failure.wait(timeout=5)
        raise OSError("controlled replacement failure")

    monkeypatch.setattr(
        train_transformer_route,
        "replace_transformer_model_file",
        controlled_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        success_future = executor.submit(
            train_transformer_route.save_transformer_model,
            successful_model,
            epochs=300,
            model_directory=tmp_path,
        )
        failure_future = executor.submit(
            train_transformer_route.save_transformer_model,
            failing_model,
            epochs=300,
            model_directory=tmp_path,
        )

        assert success_ready.wait(timeout=5)
        assert failure_ready.wait(timeout=5)

        release_success.set()
        destination = success_future.result(timeout=5)

        assert destination.read_text(encoding="utf-8") == _expected_document(successful_model)

        release_failure.set()

        with pytest.raises(
            OSError,
            match="controlled replacement failure",
        ):
            failure_future.result(timeout=5)

    assert destination.read_text(encoding="utf-8") == _expected_document(successful_model)
    assert _temporary_files(tmp_path) == []


def test_failed_concurrent_different_configuration_save_cannot_alter_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    successful_model = deepcopy(ONE_LAYER_MODEL)
    failing_model = deepcopy(TWO_LAYER_MODEL)
    failed_destination = tmp_path / _TWO_LAYER_DESTINATION_FILENAME
    failed_destination.write_bytes(_PREVIOUS_DESTINATION_DOCUMENT)
    original_replace = train_transformer_route.replace_transformer_model_file
    success_ready = Event()
    failure_ready = Event()
    release_success = Event()
    release_failure = Event()

    def controlled_replace(source: Path, destination: Path) -> None:
        if destination == failed_destination:
            failure_ready.set()
            assert release_failure.wait(timeout=5)
            raise OSError("controlled different-configuration failure")

        success_ready.set()
        assert release_success.wait(timeout=5)
        original_replace(source, destination)

    monkeypatch.setattr(
        train_transformer_route,
        "replace_transformer_model_file",
        controlled_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        success_future = executor.submit(
            train_transformer_route.save_transformer_model,
            successful_model,
            epochs=50,
            model_directory=tmp_path,
        )
        failure_future = executor.submit(
            train_transformer_route.save_transformer_model,
            failing_model,
            epochs=300,
            model_directory=tmp_path,
        )

        assert success_ready.wait(timeout=5)
        assert failure_ready.wait(timeout=5)

        release_success.set()
        successful_destination = success_future.result(timeout=5)

        release_failure.set()

        with pytest.raises(
            OSError,
            match="controlled different-configuration failure",
        ):
            failure_future.result(timeout=5)

    assert successful_destination.read_text(encoding="utf-8") == _expected_document(
        successful_model
    )
    assert failed_destination.read_bytes() == (_PREVIOUS_DESTINATION_DOCUMENT)
    assert _temporary_files(tmp_path) == []


@pytest.mark.parametrize(
    "existing_document",
    [
        b'{"old":"valid-json"}\n',
        b"not valid json\n",
        b"\x00\xffarbitrary bytes",
    ],
    ids=[
        "valid-old-json",
        "invalid-json",
        "arbitrary-bytes",
    ],
)
def test_save_transformer_model_no_read_no_cache_replaces_existing_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing_document: bytes,
) -> None:
    destination = tmp_path / _TWO_LAYER_DESTINATION_FILENAME
    destination.write_bytes(existing_document)

    def fail_path_read(
        *_: object,
        **__: object,
    ) -> object:
        raise AssertionError("persistence must not read an existing destination")

    def fail_json_loads(
        *_: object,
        **__: object,
    ) -> object:
        raise AssertionError("persistence must not parse or reuse an existing destination")

    with monkeypatch.context() as context:
        context.setattr(
            Path,
            "read_text",
            fail_path_read,
        )
        context.setattr(
            Path,
            "read_bytes",
            fail_path_read,
        )
        context.setattr(
            train_transformer_route.json,
            "loads",
            fail_json_loads,
        )

        returned_path = train_transformer_route.save_transformer_model(
            deepcopy(TWO_LAYER_MODEL),
            epochs=300,
            model_directory=tmp_path,
        )

    assert returned_path == destination
    assert destination.read_text(encoding="utf-8") == _expected_document(TWO_LAYER_MODEL)
    assert _temporary_files(tmp_path) == []
