# backend/tests/test_neural_net_persistence.py

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import cast

import pytest
from how_llms_work.ml.neural_net import (
    MultiLayerSnapshot,
    SavedNetwork,
    SingleLayerSnapshot,
)
from how_llms_work.routes import neural_net as neural_net_route

SINGLE_LAYER_SNAPSHOT: SingleLayerSnapshot = {
    "type": "single-layer",
    "w1": 0.125,
    "w2": -0.25,
    "bias": 0.5,
}

MULTI_LAYER_SNAPSHOT: MultiLayerSnapshot = {
    "type": "multi-layer",
    "w1": [
        [0.1, 0.2, 0.3, 0.4],
        [-0.1, -0.2, -0.3, -0.4],
    ],
    "b1": [0.01, 0.02, 0.03, 0.04],
    "w2": [0.5, 0.6, 0.7, 0.8],
    "b2": -0.05,
}


def _temporary_files(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.suffix == ".tmp"]


def _assert_plain_json_value(value: object) -> None:
    if value is None or isinstance(
        value,
        str | int | float | bool,
    ):
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


def test_snapshot_directory_is_derived_from_backend_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = Path(neural_net_route.__file__).resolve().parents[3] / ".data"

    monkeypatch.chdir(tmp_path)

    assert neural_net_route.get_snapshot_directory() == expected
    assert neural_net_route.get_snapshot_directory() != Path.cwd() / ".data"


@pytest.mark.parametrize(
    (
        "snapshot",
        "expected_filename",
        "expected_keys",
    ),
    [
        (
            SINGLE_LAYER_SNAPSHOT,
            "single-layer-weights.json",
            {
                "type",
                "w1",
                "w2",
                "bias",
            },
        ),
        (
            MULTI_LAYER_SNAPSHOT,
            "multi-layer-weights.json",
            {
                "type",
                "w1",
                "b1",
                "w2",
                "b2",
            },
        ),
    ],
    ids=[
        "single-layer",
        "multi-layer",
    ],
)
def test_save_network_creates_exact_mode_specific_document(
    tmp_path: Path,
    snapshot: SavedNetwork,
    expected_filename: str,
    expected_keys: set[str],
) -> None:
    snapshot_directory = tmp_path / "missing" / ".data"

    destination = neural_net_route.save_network(
        snapshot,
        snapshot_directory,
    )

    expected_document = f"{json.dumps(snapshot, indent=2, allow_nan=False)}\n"
    raw_document = destination.read_text(encoding="utf-8")
    parsed_document = json.loads(raw_document)

    assert destination == (snapshot_directory / expected_filename)
    assert snapshot_directory.is_dir()
    assert raw_document == expected_document
    assert raw_document.endswith("\n")
    assert not raw_document.endswith("\n\n")
    assert set(parsed_document) == expected_keys

    _assert_plain_json_value(parsed_document)

    assert _temporary_files(snapshot_directory) == []


def test_save_network_replaces_existing_snapshot_without_loading_it(
    tmp_path: Path,
) -> None:
    snapshot_directory = tmp_path / ".data"
    snapshot_directory.mkdir()

    destination = snapshot_directory / "single-layer-weights.json"
    destination.write_text(
        "not valid json\n",
        encoding="utf-8",
    )

    returned_path = neural_net_route.save_network(
        SINGLE_LAYER_SNAPSHOT,
        snapshot_directory,
    )

    parsed_document = json.loads(destination.read_text(encoding="utf-8"))

    assert returned_path == destination
    assert parsed_document == SINGLE_LAYER_SNAPSHOT
    assert _temporary_files(snapshot_directory) == []


def test_temporary_file_is_closed_before_same_directory_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_replace = neural_net_route.replace_snapshot_file
    observed_paths: list[tuple[Path, Path]] = []

    def observing_replace(
        source: Path,
        destination: Path,
    ) -> None:
        probe = source.with_suffix(".probe")

        os.replace(
            source,
            probe,
        )
        os.replace(
            probe,
            source,
        )

        observed_paths.append(
            (
                source,
                destination,
            )
        )

        original_replace(
            source,
            destination,
        )

    monkeypatch.setattr(
        neural_net_route,
        "replace_snapshot_file",
        observing_replace,
    )

    destination = neural_net_route.save_network(
        SINGLE_LAYER_SNAPSHOT,
        tmp_path,
    )

    assert len(observed_paths) == 1

    temporary_path, observed_destination = observed_paths[0]

    assert temporary_path != observed_destination
    assert temporary_path.parent == tmp_path
    assert observed_destination.parent == tmp_path
    assert observed_destination == destination
    assert destination.is_file()
    assert _temporary_files(tmp_path) == []


def test_serialization_failure_preserves_existing_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "single-layer-weights.json"
    previous_document = '{"previous": true}\n'

    destination.write_text(
        previous_document,
        encoding="utf-8",
    )

    def fail_serialization(
        weights: SavedNetwork,
    ) -> str:
        raise TypeError("serialization failed")

    monkeypatch.setattr(
        neural_net_route,
        "serialize_saved_network",
        fail_serialization,
    )

    with pytest.raises(
        TypeError,
        match="serialization failed",
    ):
        neural_net_route.save_network(
            SINGLE_LAYER_SNAPSHOT,
            tmp_path,
        )

    assert destination.read_text(encoding="utf-8") == previous_document
    assert _temporary_files(tmp_path) == []


def test_write_failure_preserves_existing_snapshot_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "single-layer-weights.json"
    previous_document = '{"previous": true}\n'

    destination.write_text(
        previous_document,
        encoding="utf-8",
    )

    def fail_write(
        path: Path,
        document: str,
    ) -> None:
        path.write_text(
            '{"partial":',
            encoding="utf-8",
            newline="\n",
        )

        raise OSError("write failed")

    monkeypatch.setattr(
        neural_net_route,
        "write_snapshot_document",
        fail_write,
    )

    with pytest.raises(
        OSError,
        match="write failed",
    ):
        neural_net_route.save_network(
            SINGLE_LAYER_SNAPSHOT,
            tmp_path,
        )

    assert destination.read_text(encoding="utf-8") == previous_document
    assert _temporary_files(tmp_path) == []


def test_replacement_failure_preserves_existing_snapshot_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "single-layer-weights.json"
    previous_document = '{"previous": true}\n'

    destination.write_text(
        previous_document,
        encoding="utf-8",
    )

    def fail_replacement(
        source: Path,
        selected_destination: Path,
    ) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr(
        neural_net_route,
        "replace_snapshot_file",
        fail_replacement,
    )

    with pytest.raises(
        OSError,
        match="replacement failed",
    ):
        neural_net_route.save_network(
            SINGLE_LAYER_SNAPSHOT,
            tmp_path,
        )

    assert destination.read_text(encoding="utf-8") == previous_document
    assert _temporary_files(tmp_path) == []


def test_cleanup_failure_preserves_existing_snapshot_and_reports_both_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "single-layer-weights.json"
    previous_document = '{"previous": true}\n'

    destination.write_text(
        previous_document,
        encoding="utf-8",
    )

    def fail_write(
        path: Path,
        document: str,
    ) -> None:
        path.write_text(
            '{"partial":',
            encoding="utf-8",
            newline="\n",
        )

        raise OSError("write failed")

    def fail_cleanup(path: Path) -> None:
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(
        neural_net_route,
        "write_snapshot_document",
        fail_write,
    )
    monkeypatch.setattr(
        neural_net_route,
        "remove_temporary_snapshot",
        fail_cleanup,
    )

    with pytest.raises(ExceptionGroup) as error_info:
        neural_net_route.save_network(
            SINGLE_LAYER_SNAPSHOT,
            tmp_path,
        )

    messages = {str(error) for error in error_info.value.exceptions}

    assert messages == {
        "write failed",
        "cleanup failed",
    }
    assert destination.read_text(encoding="utf-8") == previous_document
    assert len(_temporary_files(tmp_path)) == 1


def test_concurrent_different_mode_saves_use_distinct_same_directory_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_replace = neural_net_route.replace_snapshot_file
    replacement_barrier = Barrier(2)
    replacement_lock = Lock()
    observed_paths: list[tuple[Path, Path]] = []

    def synchronized_replace(
        source: Path,
        destination: Path,
    ) -> None:
        with replacement_lock:
            observed_paths.append(
                (
                    source,
                    destination,
                )
            )

        replacement_barrier.wait(timeout=5)

        original_replace(
            source,
            destination,
        )

    monkeypatch.setattr(
        neural_net_route,
        "replace_snapshot_file",
        synchronized_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        single_future = executor.submit(
            neural_net_route.save_network,
            SINGLE_LAYER_SNAPSHOT,
            tmp_path,
        )
        multi_future = executor.submit(
            neural_net_route.save_network,
            MULTI_LAYER_SNAPSHOT,
            tmp_path,
        )

        single_destination = single_future.result(timeout=5)
        multi_destination = multi_future.result(timeout=5)

    sources = [source for source, _ in observed_paths]
    destinations = [destination for _, destination in observed_paths]

    assert len(set(sources)) == 2
    assert set(destinations) == {
        single_destination,
        multi_destination,
    }

    assert all(
        source.parent == destination.parent == tmp_path for source, destination in observed_paths
    )

    single_document = json.loads(single_destination.read_text(encoding="utf-8"))
    multi_document = json.loads(multi_destination.read_text(encoding="utf-8"))

    assert single_document == SINGLE_LAYER_SNAPSHOT
    assert multi_document == MULTI_LAYER_SNAPSHOT
    assert _temporary_files(tmp_path) == []


@pytest.mark.parametrize(
    "last_finisher",
    [
        "first",
        "second",
    ],
)
def test_concurrent_same_mode_saves_use_last_successful_finisher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    last_finisher: str,
) -> None:
    first_snapshot = cast(
        SingleLayerSnapshot,
        {
            "type": "single-layer",
            "w1": 1.0,
            "w2": 1.1,
            "bias": 1.2,
        },
    )
    second_snapshot = cast(
        SingleLayerSnapshot,
        {
            "type": "single-layer",
            "w1": 2.0,
            "w2": 2.1,
            "bias": 2.2,
        },
    )

    original_replace = neural_net_route.replace_snapshot_file

    first_ready = Event()
    second_ready = Event()
    release_first = Event()
    release_second = Event()

    def controlled_replace(
        source: Path,
        destination: Path,
    ) -> None:
        document = json.loads(source.read_text(encoding="utf-8"))

        if document["w1"] == first_snapshot["w1"]:
            first_ready.set()

            assert release_first.wait(timeout=5)
        else:
            second_ready.set()

            assert release_second.wait(timeout=5)

        original_replace(
            source,
            destination,
        )

    monkeypatch.setattr(
        neural_net_route,
        "replace_snapshot_file",
        controlled_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            neural_net_route.save_network,
            first_snapshot,
            tmp_path,
        )
        second_future = executor.submit(
            neural_net_route.save_network,
            second_snapshot,
            tmp_path,
        )

        assert first_ready.wait(timeout=5)
        assert second_ready.wait(timeout=5)

        if last_finisher == "first":
            release_second.set()
            second_future.result(timeout=5)

            release_first.set()
            destination = first_future.result(timeout=5)
            expected_snapshot = first_snapshot
        else:
            release_first.set()
            first_future.result(timeout=5)

            release_second.set()
            destination = second_future.result(timeout=5)
            expected_snapshot = second_snapshot

    final_document = json.loads(destination.read_text(encoding="utf-8"))

    assert final_document == expected_snapshot
    assert _temporary_files(tmp_path) == []
