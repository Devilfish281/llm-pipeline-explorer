# backend/tests/test_train_embed_persistence.py
# Test for the persistence of Word2Vec Saved Embedding Models.
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import cast
from unittest.mock import Mock

import pytest
from how_llms_work.ml.word2vec import SavedEmbeddingModel
from how_llms_work.routes import train_embed as train_embed_route

SAVED_EMBEDDING_MODEL: SavedEmbeddingModel = {
    "type": "word2vec-skipgram",
    "dimensions": 3,
    "vocab": ["king", "queen", "royal"],
    "merges": [
        {"pair": ["k", "i"], "merged": "ki"},
        {"pair": ["ki", "ng"], "merged": "king"},
    ],
    "embeddings": {
        "king": [0.1, 0.2, 0.3],
        "queen": [-0.1, 0.4, 0.5],
        "royal": [0.25, -0.5, 0.75],
    },
}

SECOND_SAVED_EMBEDDING_MODEL: SavedEmbeddingModel = {
    "type": "word2vec-skipgram",
    "dimensions": 2,
    "vocab": ["cat", "dog"],
    "merges": [
        {"pair": ["c", "a"], "merged": "ca"},
        {"pair": ["ca", "t"], "merged": "cat"},
    ],
    "embeddings": {
        "cat": [0.9, -0.8],
        "dog": [-0.7, 0.6],
    },
}


def _temporary_files(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.suffix == ".tmp"]


def _expected_document(model: SavedEmbeddingModel) -> str:
    return f"{json.dumps(model, indent=2, allow_nan=False)}\n"


def test_embedding_model_directory_is_derived_from_backend_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = Path(train_embed_route.__file__).resolve().parents[3] / ".data"

    monkeypatch.chdir(tmp_path)

    assert train_embed_route.get_embedding_model_directory() == expected
    assert train_embed_route.get_embedding_model_directory() != Path.cwd() / ".data"


def test_save_embedding_model_creates_exact_complete_document(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "missing" / ".data"

    destination = train_embed_route.save_embedding_model(
        SAVED_EMBEDDING_MODEL,
        model_directory,
    )

    raw_document = destination.read_text(encoding="utf-8")
    parsed_document = json.loads(raw_document)

    assert destination == model_directory / "embedding-weights.json"
    assert model_directory.is_dir()
    assert raw_document == _expected_document(SAVED_EMBEDDING_MODEL)
    assert raw_document.endswith("\n")
    assert not raw_document.endswith("\n\n")
    assert list(parsed_document) == [
        "type",
        "dimensions",
        "vocab",
        "merges",
        "embeddings",
    ]
    assert parsed_document["type"] == "word2vec-skipgram"
    assert parsed_document["vocab"] == SAVED_EMBEDDING_MODEL["vocab"]
    assert parsed_document["merges"] == SAVED_EMBEDDING_MODEL["merges"]
    assert parsed_document["embeddings"] == SAVED_EMBEDDING_MODEL["embeddings"]
    assert list(parsed_document["embeddings"]) == parsed_document["vocab"]
    assert all(
        len(vector) == parsed_document["dimensions"]
        for vector in parsed_document["embeddings"].values()
    )
    assert _temporary_files(model_directory) == []


def test_temporary_model_is_closed_before_same_directory_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_replace = train_embed_route.replace_embedding_model_file
    observed_paths: list[tuple[Path, Path]] = []

    def observing_replace(source: Path, destination: Path) -> None:
        probe = source.with_suffix(".probe")

        os.replace(source, probe)
        os.replace(probe, source)
        observed_paths.append((source, destination))

        original_replace(source, destination)

    monkeypatch.setattr(
        train_embed_route,
        "replace_embedding_model_file",
        observing_replace,
    )

    destination = train_embed_route.save_embedding_model(
        SAVED_EMBEDDING_MODEL,
        tmp_path,
    )

    assert len(observed_paths) == 1
    temporary_path, observed_destination = observed_paths[0]
    assert temporary_path != observed_destination
    assert temporary_path.parent == tmp_path
    assert observed_destination.parent == tmp_path
    assert observed_destination == destination
    assert destination.read_text(encoding="utf-8") == _expected_document(SAVED_EMBEDDING_MODEL)
    assert _temporary_files(tmp_path) == []


@pytest.mark.parametrize(
    "non_finite_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_non_finite_serialization_fails_before_filesystem_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    non_finite_value: float,
) -> None:
    model_directory = tmp_path / ".data"
    model_directory.mkdir()
    destination = model_directory / "embedding-weights.json"
    previous_document = b'{"previous":true}\n'
    destination.write_bytes(previous_document)
    directory_creation = Mock()
    temporary_factory = Mock()
    replacement = Mock()
    invalid_model = cast(
        SavedEmbeddingModel,
        {
            **SAVED_EMBEDDING_MODEL,
            "embeddings": {
                **SAVED_EMBEDDING_MODEL["embeddings"],
                "king": [non_finite_value, 0.2, 0.3],
            },
        },
    )

    monkeypatch.setattr(
        Path,
        "mkdir",
        directory_creation,
    )
    monkeypatch.setattr(
        train_embed_route,
        "create_temporary_embedding_model_path",
        temporary_factory,
    )
    monkeypatch.setattr(
        train_embed_route,
        "replace_embedding_model_file",
        replacement,
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        train_embed_route.save_embedding_model(
            invalid_model,
            model_directory,
        )

    assert destination.read_bytes() == previous_document
    directory_creation.assert_not_called()
    temporary_factory.assert_not_called()
    replacement.assert_not_called()


def test_temporary_file_creation_failure_preserves_previous_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "embedding-weights.json"
    previous_document = b'{"previous":true}\n'
    unrelated_temporary = tmp_path / ".unrelated.tmp"
    destination.write_bytes(previous_document)
    unrelated_temporary.write_text("unrelated", encoding="utf-8")
    replacement = Mock()

    def fail_temporary_creation(_: Path, __: Path) -> Path:
        raise OSError("temporary creation failed")

    monkeypatch.setattr(
        train_embed_route,
        "create_temporary_embedding_model_path",
        fail_temporary_creation,
    )
    monkeypatch.setattr(
        train_embed_route,
        "replace_embedding_model_file",
        replacement,
    )

    with pytest.raises(OSError, match="temporary creation failed"):
        train_embed_route.save_embedding_model(
            SAVED_EMBEDDING_MODEL,
            tmp_path,
        )

    assert destination.read_bytes() == previous_document
    assert unrelated_temporary.read_text(encoding="utf-8") == "unrelated"
    replacement.assert_not_called()


@pytest.mark.parametrize(
    ("failure_message", "partial_document"),
    [
        ("write failed", '{"partial":'),
        ("close failed", _expected_document(SAVED_EMBEDDING_MODEL)),
    ],
    ids=["partial-write", "close-stage"],
)
def test_write_or_close_failure_preserves_destination_and_cleans_owned_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_message: str,
    partial_document: str,
) -> None:
    destination = tmp_path / "embedding-weights.json"
    previous_document = b'{"previous":true}\n'
    destination.write_bytes(previous_document)

    def fail_write_or_close(path: Path, _: str) -> None:
        path.write_text(
            partial_document,
            encoding="utf-8",
            newline="\n",
        )
        raise OSError(failure_message)

    monkeypatch.setattr(
        train_embed_route,
        "write_embedding_model_document",
        fail_write_or_close,
    )

    with pytest.raises(OSError, match=failure_message):
        train_embed_route.save_embedding_model(
            SAVED_EMBEDDING_MODEL,
            tmp_path,
        )

    assert destination.read_bytes() == previous_document
    assert _temporary_files(tmp_path) == []


def test_replacement_failure_preserves_destination_and_cleans_owned_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "embedding-weights.json"
    previous_document = b'{"previous":true}\n'
    destination.write_bytes(previous_document)

    def fail_replacement(_: Path, __: Path) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr(
        train_embed_route,
        "replace_embedding_model_file",
        fail_replacement,
    )

    with pytest.raises(OSError, match="replacement failed"):
        train_embed_route.save_embedding_model(
            SAVED_EMBEDDING_MODEL,
            tmp_path,
        )

    assert destination.read_bytes() == previous_document
    assert _temporary_files(tmp_path) == []


def test_cleanup_failure_preserves_destination_and_reports_both_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "embedding-weights.json"
    previous_document = b'{"previous":true}\n'
    destination.write_bytes(previous_document)

    def fail_write(path: Path, _: str) -> None:
        path.write_text(
            '{"partial":',
            encoding="utf-8",
            newline="\n",
        )
        raise OSError("write failed")

    def fail_cleanup(_: Path) -> None:
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(
        train_embed_route,
        "write_embedding_model_document",
        fail_write,
    )
    monkeypatch.setattr(
        train_embed_route,
        "remove_temporary_embedding_model",
        fail_cleanup,
    )

    with pytest.raises(ExceptionGroup) as error_info:
        train_embed_route.save_embedding_model(
            SAVED_EMBEDDING_MODEL,
            tmp_path,
        )

    assert [str(error) for error in error_info.value.exceptions] == [
        "write failed",
        "cleanup failed",
    ]
    assert destination.read_bytes() == previous_document
    assert len(_temporary_files(tmp_path)) == 1


def test_concurrent_saves_use_distinct_same_directory_temporary_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_replace = train_embed_route.replace_embedding_model_file
    replacement_barrier = Barrier(2)
    replacement_lock = Lock()
    observed_paths: list[tuple[Path, Path]] = []

    def synchronized_replace(source: Path, destination: Path) -> None:
        with replacement_lock:
            observed_paths.append((source, destination))

        replacement_barrier.wait(timeout=5)
        original_replace(source, destination)

    monkeypatch.setattr(
        train_embed_route,
        "replace_embedding_model_file",
        synchronized_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            train_embed_route.save_embedding_model,
            SAVED_EMBEDDING_MODEL,
            tmp_path,
        )
        second_future = executor.submit(
            train_embed_route.save_embedding_model,
            SECOND_SAVED_EMBEDDING_MODEL,
            tmp_path,
        )

        first_destination = first_future.result(timeout=5)
        second_destination = second_future.result(timeout=5)

    sources = [source for source, _ in observed_paths]
    destinations = [destination for _, destination in observed_paths]
    final_document = first_destination.read_text(encoding="utf-8")

    assert first_destination == second_destination
    assert len(set(sources)) == 2
    assert set(destinations) == {first_destination}
    assert all(
        source.parent == destination.parent == tmp_path for source, destination in observed_paths
    )
    assert final_document in {
        _expected_document(SAVED_EMBEDDING_MODEL),
        _expected_document(SECOND_SAVED_EMBEDDING_MODEL),
    }
    assert _temporary_files(tmp_path) == []


@pytest.mark.parametrize("last_finisher", ["first", "second"])
def test_concurrent_saves_use_last_successful_finisher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    last_finisher: str,
) -> None:
    original_replace = train_embed_route.replace_embedding_model_file
    first_ready = Event()
    second_ready = Event()
    release_first = Event()
    release_second = Event()

    def controlled_replace(source: Path, destination: Path) -> None:
        document = json.loads(source.read_text(encoding="utf-8"))

        if document["dimensions"] == SAVED_EMBEDDING_MODEL["dimensions"]:
            first_ready.set()
            assert release_first.wait(timeout=5)
        else:
            second_ready.set()
            assert release_second.wait(timeout=5)

        original_replace(source, destination)

    monkeypatch.setattr(
        train_embed_route,
        "replace_embedding_model_file",
        controlled_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            train_embed_route.save_embedding_model,
            SAVED_EMBEDDING_MODEL,
            tmp_path,
        )
        second_future = executor.submit(
            train_embed_route.save_embedding_model,
            SECOND_SAVED_EMBEDDING_MODEL,
            tmp_path,
        )

        assert first_ready.wait(timeout=5)
        assert second_ready.wait(timeout=5)

        if last_finisher == "first":
            release_second.set()
            second_future.result(timeout=5)
            release_first.set()
            destination = first_future.result(timeout=5)
            expected_model = SAVED_EMBEDDING_MODEL
        else:
            release_first.set()
            first_future.result(timeout=5)
            release_second.set()
            destination = second_future.result(timeout=5)
            expected_model = SECOND_SAVED_EMBEDDING_MODEL

    assert destination.read_text(encoding="utf-8") == _expected_document(expected_model)
    assert _temporary_files(tmp_path) == []


def test_failed_concurrent_save_cannot_damage_successful_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_replace = train_embed_route.replace_embedding_model_file
    success_ready = Event()
    failure_ready = Event()
    release_success = Event()
    release_failure = Event()

    def controlled_replace(source: Path, destination: Path) -> None:
        document = json.loads(source.read_text(encoding="utf-8"))

        if document["dimensions"] == SAVED_EMBEDDING_MODEL["dimensions"]:
            success_ready.set()
            assert release_success.wait(timeout=5)
            original_replace(source, destination)
            return

        failure_ready.set()
        assert release_failure.wait(timeout=5)
        raise OSError("controlled replacement failure")

    monkeypatch.setattr(
        train_embed_route,
        "replace_embedding_model_file",
        controlled_replace,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        success_future = executor.submit(
            train_embed_route.save_embedding_model,
            SAVED_EMBEDDING_MODEL,
            tmp_path,
        )
        failure_future = executor.submit(
            train_embed_route.save_embedding_model,
            SECOND_SAVED_EMBEDDING_MODEL,
            tmp_path,
        )

        assert success_ready.wait(timeout=5)
        assert failure_ready.wait(timeout=5)

        release_success.set()
        destination = success_future.result(timeout=5)
        assert destination.read_text(encoding="utf-8") == _expected_document(SAVED_EMBEDDING_MODEL)

        release_failure.set()
        with pytest.raises(OSError, match="controlled replacement failure"):
            failure_future.result(timeout=5)

    assert destination.read_text(encoding="utf-8") == _expected_document(SAVED_EMBEDDING_MODEL)
    assert _temporary_files(tmp_path) == []


def test_successful_save_replaces_invalid_existing_document_without_loading_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "embedding-weights.json"
    destination.write_text("not valid json\n", encoding="utf-8")

    returned_path = train_embed_route.save_embedding_model(
        SAVED_EMBEDDING_MODEL,
        tmp_path,
    )

    assert returned_path == destination
    assert destination.read_text(encoding="utf-8") == _expected_document(SAVED_EMBEDDING_MODEL)
    assert _temporary_files(tmp_path) == []
