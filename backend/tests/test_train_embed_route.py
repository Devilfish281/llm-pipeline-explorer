# backend/tests/test_train_embed_route.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast
from unittest.mock import AsyncMock, Mock, call

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from how_llms_work.main import app
from how_llms_work.ml.word2vec import (
    EmbeddingEpochUpdate,
    EmbeddingResult,
    EmbeddingTrainingEvent,
    SavedEmbeddingModel,
    Word2VecPreprocessing,
)
from how_llms_work.routes import train_embed as train_embed_route

COMPLETION_EVENT = cast(EmbeddingTrainingEvent, object())

CONTROLLED_RESULT: EmbeddingResult = {
    "embeddings": [
        {
            "word": "alpha",
            "vector": [0.1, 0.2, 0.3, 0.4],
        }
    ],
    "neighbors": [
        {
            "word": "alpha",
            "nearest": [
                {
                    "word": "beta",
                    "score": 0.75,
                }
            ],
        }
    ],
    "similarities": [],
    "analogies": [],
    "warnings": [],
}

CONTROLLED_MODEL: SavedEmbeddingModel = {
    "type": "word2vec-skipgram",
    "dimensions": 4,
    "vocab": [
        "alpha",
        "beta",
        "gamma",
    ],
    "merges": [],
    "embeddings": {
        "alpha": [0.1, 0.2, 0.3, 0.4],
        "beta": [0.2, 0.3, 0.4, 0.5],
        "gamma": [0.3, 0.4, 0.5, 0.6],
    },
}


class ControlledEmbeddingTrainingRun(Iterator[EmbeddingTrainingEvent]):
    def __init__(
        self,
        events: list[EmbeddingTrainingEvent],
        failure: BaseException | None = None,
    ) -> None:
        self._events = deque(events)
        self._failure = failure
        self.advance_thread_ids: list[int] = []
        self.advance_process_ids: list[int] = []

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> EmbeddingTrainingEvent:
        self.advance_thread_ids.append(threading.get_ident())
        self.advance_process_ids.append(os.getpid())

        if self._events:
            return self._events.popleft()

        if self._failure is not None:
            raise self._failure

        raise StopIteration

    @property
    def advance_count(self) -> int:
        return len(self.advance_thread_ids)


def build_controlled_preprocessing() -> Word2VecPreprocessing:
    """Build the route-visible immutable-data shape without numerical derivation."""
    return cast(
        Word2VecPreprocessing,
        SimpleNamespace(
            vocabulary=(
                "alpha",
                "beta",
                "gamma",
            ),
            corpus=(
                "alpha beta",
                "beta gamma",
            ),
            training_pairs={
                1: (
                    object(),
                    object(),
                ),
                2: (
                    object(),
                    object(),
                    object(),
                ),
                3: (
                    object(),
                    object(),
                    object(),
                    object(),
                ),
                4: (
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                ),
                5: (
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                ),
            },
        ),
    )


def parse_sse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    assert body.endswith("\n\n")

    event_blocks = body[:-2].split("\n\n")
    parsed_events: list[tuple[str, dict[str, object]]] = []

    for event_block in event_blocks:
        lines = event_block.splitlines()
        assert len(lines) == 2

        event_line, data_line = lines
        assert event_line.startswith("event: ")
        assert data_line.startswith("data: ")

        event_name = event_line.removeprefix("event: ")
        assert event_name

        parsed_data = json.loads(data_line.removeprefix("data: "))
        assert isinstance(parsed_data, dict)

        parsed_events.append(
            (
                event_name,
                cast(dict[str, object], parsed_data),
            )
        )

    return parsed_events


def install_controlled_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    training_run: Iterator[EmbeddingTrainingEvent],
    preprocessing: Word2VecPreprocessing | None = None,
    result_builder: Mock | None = None,
    model_builder: Mock | None = None,
    save_model: Mock | None = None,
) -> tuple[Word2VecPreprocessing, Mock, Mock, Mock, Mock, AsyncMock, AsyncMock]:
    controlled_preprocessing = preprocessing or build_controlled_preprocessing()
    training_factory = Mock(return_value=training_run)
    controlled_result_builder = result_builder or Mock(return_value=CONTROLLED_RESULT)
    controlled_model_builder = model_builder or Mock(return_value=CONTROLLED_MODEL)
    controlled_save_model = save_model or Mock(return_value=Path("embedding-weights.json"))
    disconnect_check = AsyncMock(return_value=False)
    delay = AsyncMock()

    monkeypatch.setattr(
        train_embed_route,
        "get_word2vec_preprocessing",
        Mock(return_value=controlled_preprocessing),
    )
    monkeypatch.setattr(
        train_embed_route,
        "create_embedding_training_run",
        training_factory,
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_embedding_result",
        controlled_result_builder,
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_saved_embedding_model",
        controlled_model_builder,
    )
    monkeypatch.setattr(
        train_embed_route,
        "save_embedding_model",
        controlled_save_model,
    )
    monkeypatch.setattr(
        train_embed_route,
        "request_is_disconnected",
        disconnect_check,
    )
    monkeypatch.setattr(
        train_embed_route,
        "presentation_sleep",
        delay,
    )

    return (
        controlled_preprocessing,
        training_factory,
        controlled_result_builder,
        controlled_model_builder,
        controlled_save_model,
        disconnect_check,
        delay,
    )


def test_train_embed_route_is_registered_without_removing_existing_routes() -> None:
    client = TestClient(app)

    health_response = client.get("/health")
    simple_chat_response = client.post(
        "/simple-chat",
        json={},
    )
    bpe_response = client.post(
        "/bpe-tokenize",
        json={},
    )
    neural_net_response = client.post(
        "/neural-net",
        json={},
    )
    train_embed_response = client.post(
        "/train-embed",
        json={},
    )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}

    assert simple_chat_response.status_code == 422
    assert bpe_response.status_code == 422
    assert neural_net_response.status_code == 422
    assert train_embed_response.status_code == 422


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {"words": []},
        {"words": [""]},
        {"words": ["alpha"] * 11},
        {"words": "alpha"},
        {"words": [1]},
        {"words": ["alpha"], "epochs": "10"},
        {"words": ["alpha"], "epochs": True},
        {"words": ["alpha"], "epochs": 9},
        {"words": ["alpha"], "epochs": 10_001},
        {"words": ["alpha"], "dimensions": "4"},
        {"words": ["alpha"], "dimensions": False},
        {"words": ["alpha"], "dimensions": 3},
        {"words": ["alpha"], "dimensions": 65},
        {"words": ["alpha"], "windowSize": "1"},
        {"words": ["alpha"], "windowSize": True},
        {"words": ["alpha"], "windowSize": 0},
        {"words": ["alpha"], "windowSize": 6},
        {"words": ["alpha"], "negativeSamples": "1"},
        {"words": ["alpha"], "negativeSamples": False},
        {"words": ["alpha"], "negativeSamples": 0},
        {"words": ["alpha"], "negativeSamples": 11},
        {"words": ["alpha"], "epochs": 10.0},
        {"words": ["alpha"], "dimensions": 4.5},
        {"words": ["alpha"], "windowSize": 1.5},
        {"words": ["alpha"], "negativeSamples": 1.5},
    ],
    ids=[
        "missing-words",
        "empty-words",
        "empty-query-word",
        "too-many-query-words",
        "words-not-list",
        "query-word-not-string",
        "epochs-numeric-string",
        "epochs-boolean",
        "epochs-too-small",
        "epochs-too-large",
        "dimensions-numeric-string",
        "dimensions-boolean",
        "dimensions-too-small",
        "dimensions-too-large",
        "window-size-numeric-string",
        "window-size-boolean",
        "window-size-too-small",
        "window-size-too-large",
        "negative-samples-numeric-string",
        "negative-samples-boolean",
        "negative-samples-too-small",
        "negative-samples-too-large",
        "epochs-fraction",
        "dimensions-fraction",
        "window-size-fraction",
        "negative-samples-fraction",
    ],
)
def test_train_embed_validation_rejects_before_training_creation(
    monkeypatch: pytest.MonkeyPatch,
    request_body: dict[str, object],
) -> None:
    training_factory = Mock()
    monkeypatch.setattr(
        train_embed_route,
        "create_embedding_training_run",
        training_factory,
    )
    client = TestClient(app)

    response = client.post(
        "/train-embed",
        json=request_body,
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    training_factory.assert_not_called()


def test_train_embed_defaults_preserve_query_word_positions_and_ignore_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_run = ControlledEmbeddingTrainingRun([COMPLETION_EVENT])
    (
        preprocessing,
        training_factory,
        result_builder,
        _model_builder,
        _save_model,
        _disconnect_check,
        _delay,
    ) = install_controlled_dependencies(
        monkeypatch,
        training_run=training_run,
    )
    client = TestClient(app)

    response = client.post(
        "/train-embed",
        json={
            "words": [
                "  ",
                "Alpha",
                "Alpha",
            ],
            "ignoredFrontendField": {
                "enabled": True,
            },
        },
    )

    assert response.status_code == 200
    training_factory.assert_called_once_with(
        dimensions=32,
        window_size=2,
        epochs=10_000,
        negative_samples=5,
    )
    assert result_builder.call_args.args == (
        COMPLETION_EVENT,
        preprocessing,
        (
            "  ",
            "Alpha",
            "Alpha",
        ),
    )


@pytest.mark.parametrize(
    (
        "request_body",
        "expected_arguments",
    ),
    [
        (
            {
                "words": ["alpha"],
                "epochs": 10,
                "dimensions": 4,
                "windowSize": 1,
                "negativeSamples": 1,
            },
            {
                "dimensions": 4,
                "window_size": 1,
                "epochs": 10,
                "negative_samples": 1,
            },
        ),
        (
            {
                "words": ["alpha"],
                "epochs": 10_000,
                "dimensions": 64,
                "windowSize": 5,
                "negativeSamples": 10,
            },
            {
                "dimensions": 64,
                "window_size": 5,
                "epochs": 10_000,
                "negative_samples": 10,
            },
        ),
    ],
    ids=[
        "lower-bounds",
        "upper-bounds",
    ],
)
def test_train_embed_accepts_inclusive_hyperparameter_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    request_body: dict[str, object],
    expected_arguments: dict[str, int],
) -> None:
    training_run = ControlledEmbeddingTrainingRun([COMPLETION_EVENT])
    (
        _preprocessing,
        training_factory,
        _result_builder,
        _model_builder,
        _save_model,
        _disconnect_check,
        _delay,
    ) = install_controlled_dependencies(
        monkeypatch,
        training_run=training_run,
    )
    client = TestClient(app)

    response = client.post(
        "/train-embed",
        json=request_body,
    )

    assert response.status_code == 200

    training_factory.assert_called_once_with(
        **expected_arguments,
    )


def test_train_embed_streams_init_epochs_and_persists_before_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocessing = build_controlled_preprocessing()
    training_run = ControlledEmbeddingTrainingRun(
        [
            EmbeddingEpochUpdate(
                epoch=0,
                loss=1.25,
            ),
            EmbeddingEpochUpdate(
                epoch=10,
                loss=0.5,
            ),
            COMPLETION_EVENT,
        ]
    )
    call_order: list[str] = []
    route_thread_ids: list[int] = []
    result_thread_ids: list[int] = []
    model_thread_ids: list[int] = []
    save_thread_ids: list[int] = []

    async def observe_disconnect(_: Request) -> bool:
        route_thread_ids.append(threading.get_ident())
        return False

    def build_result(
        completion: object,
        observed_preprocessing: object,
        query_words: object,
    ) -> EmbeddingResult:
        result_thread_ids.append(threading.get_ident())
        call_order.append("result")
        assert completion is COMPLETION_EVENT
        assert observed_preprocessing is preprocessing
        assert query_words == (
            "alpha",
            "alpha",
        )
        return CONTROLLED_RESULT

    def build_model(
        completion: object,
        observed_preprocessing: object,
    ) -> SavedEmbeddingModel:
        model_thread_ids.append(threading.get_ident())
        call_order.append("model")
        assert completion is COMPLETION_EVENT
        assert observed_preprocessing is preprocessing
        return CONTROLLED_MODEL

    def save_model(model: SavedEmbeddingModel) -> Path:
        save_thread_ids.append(threading.get_ident())
        call_order.append("save")
        assert model == CONTROLLED_MODEL
        return Path("embedding-weights.json")

    original_format_sse = train_embed_route.format_sse

    def observe_format_sse(
        event: str,
        data: dict[str, object],
    ) -> str:
        if event == "done":
            call_order.append("done")

        return original_format_sse(
            event,
            data,
        )

    training_factory = Mock(return_value=training_run)
    delay = AsyncMock()

    monkeypatch.setattr(
        train_embed_route,
        "get_word2vec_preprocessing",
        Mock(return_value=preprocessing),
    )
    monkeypatch.setattr(
        train_embed_route,
        "create_embedding_training_run",
        training_factory,
    )
    monkeypatch.setattr(
        train_embed_route,
        "request_is_disconnected",
        observe_disconnect,
    )
    monkeypatch.setattr(
        train_embed_route,
        "presentation_sleep",
        delay,
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_embedding_result",
        build_result,
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_saved_embedding_model",
        build_model,
    )
    monkeypatch.setattr(
        train_embed_route,
        "save_embedding_model",
        save_model,
    )
    monkeypatch.setattr(
        train_embed_route,
        "format_sse",
        observe_format_sse,
    )

    client = TestClient(app)

    response = client.post(
        "/train-embed",
        json={
            "words": [
                "alpha",
                "alpha",
            ],
            "epochs": 10,
            "dimensions": 4,
            "windowSize": 2,
            "negativeSamples": 1,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    assert parse_sse_events(response.text) == [
        (
            "init",
            {
                "vocabSize": 3,
                "sentenceCount": 2,
                "embeddingDim": 4,
                "windowSize": 2,
                "totalPairs": 3,
            },
        ),
        (
            "epoch",
            {
                "epoch": 0,
                "loss": 1.25,
            },
        ),
        (
            "epoch",
            {
                "epoch": 10,
                "loss": 0.5,
            },
        ),
        (
            "done",
            cast(dict[str, object], CONTROLLED_RESULT),
        ),
    ]

    assert call_order == [
        "result",
        "model",
        "save",
        "done",
    ]
    assert delay.await_args_list == [
        call(0.02),
        call(0.02),
    ]
    assert training_run.advance_count == 3
    assert route_thread_ids

    route_thread_id = route_thread_ids[0]

    assert all(thread_id != route_thread_id for thread_id in training_run.advance_thread_ids)
    assert all(thread_id != route_thread_id for thread_id in result_thread_ids)
    assert all(thread_id != route_thread_id for thread_id in model_thread_ids)
    assert all(thread_id != route_thread_id for thread_id in save_thread_ids)
    assert set(training_run.advance_process_ids) == {os.getpid()}

    assert "output_weights" not in response.text
    assert "input_weights" not in response.text
    assert "embedding-weights.json" not in response.text


def test_train_embed_disconnect_stops_before_later_intervals_or_terminal_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_run = ControlledEmbeddingTrainingRun(
        [
            EmbeddingEpochUpdate(
                epoch=0,
                loss=1.0,
            ),
            COMPLETION_EVENT,
        ]
    )
    (
        _preprocessing,
        _training_factory,
        result_builder,
        model_builder,
        save_model,
        disconnect_check,
        delay,
    ) = install_controlled_dependencies(
        monkeypatch,
        training_run=training_run,
    )
    disconnect_check.side_effect = [
        False,
        True,
    ]

    client = TestClient(app)

    response = client.post(
        "/train-embed",
        json={
            "words": ["alpha"],
            "epochs": 10,
            "dimensions": 4,
            "windowSize": 1,
            "negativeSamples": 1,
        },
    )

    assert response.status_code == 200
    assert parse_sse_events(response.text) == [
        (
            "init",
            {
                "vocabSize": 3,
                "sentenceCount": 2,
                "embeddingDim": 4,
                "windowSize": 1,
                "totalPairs": 2,
            },
        ),
        (
            "epoch",
            {
                "epoch": 0,
                "loss": 1.0,
            },
        ),
    ]

    assert training_run.advance_count == 1
    delay.assert_awaited_once_with(0.02)
    result_builder.assert_not_called()
    model_builder.assert_not_called()
    save_model.assert_not_called()
    assert "event: done" not in response.text
    assert "event: error" not in response.text


@pytest.mark.asyncio
async def test_stream_embedding_training_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    training_run = ControlledEmbeddingTrainingRun([COMPLETION_EVENT])
    preprocessing = build_controlled_preprocessing()
    save_model = Mock()

    monkeypatch.setattr(
        train_embed_route,
        "request_is_disconnected",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )
    monkeypatch.setattr(
        train_embed_route,
        "save_embedding_model",
        save_model,
    )

    caplog.set_level(
        logging.ERROR,
        logger=train_embed_route.__name__,
    )

    stream = train_embed_route.stream_embedding_training(
        request=cast(Request, object()),
        training_run=training_run,
        preprocessing=preprocessing,
        query_words=("alpha",),
        dimensions=4,
        window_size=1,
    )

    first_event = await anext(stream)

    assert first_event.startswith("event: init\n")

    with pytest.raises(asyncio.CancelledError):
        await anext(stream)

    assert training_run.advance_count == 0
    save_model.assert_not_called()
    assert "Embedding Training Run stream failed" not in caplog.text


@pytest.mark.parametrize(
    "failure_stage",
    [
        "training",
        "result",
        "model",
        "persistence",
    ],
)
def test_train_embed_ordinary_post_stream_failures_are_private(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
) -> None:
    failure_marker = f"{failure_stage.upper()}_FAILURE_MARKER"

    training_run = ControlledEmbeddingTrainingRun(
        [COMPLETION_EVENT] if failure_stage != "training" else [],
        failure=(RuntimeError(failure_marker) if failure_stage == "training" else None),
    )

    result_builder = Mock(return_value=CONTROLLED_RESULT)
    model_builder = Mock(return_value=CONTROLLED_MODEL)
    save_model = Mock(return_value=Path("embedding-weights.json"))

    if failure_stage == "result":
        result_builder.side_effect = RuntimeError(failure_marker)
    elif failure_stage == "model":
        model_builder.side_effect = RuntimeError(failure_marker)
    elif failure_stage == "persistence":
        save_model.side_effect = OSError(failure_marker)

    (
        _preprocessing,
        _training_factory,
        _result_builder,
        _model_builder,
        controlled_save_model,
        _disconnect_check,
        _delay,
    ) = install_controlled_dependencies(
        monkeypatch,
        training_run=training_run,
        result_builder=result_builder,
        model_builder=model_builder,
        save_model=save_model,
    )

    caplog.set_level(
        logging.ERROR,
        logger=train_embed_route.__name__,
    )

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/train-embed",
        json={
            "words": ["alpha"],
            "epochs": 10,
            "dimensions": 4,
            "windowSize": 1,
            "negativeSamples": 1,
        },
    )

    assert response.status_code == 200
    assert [event_name for event_name, _payload in parse_sse_events(response.text)] == ["init"]

    assert "event: done" not in response.text
    assert "event: error" not in response.text
    assert failure_marker not in response.text
    assert "Traceback" not in response.text
    assert "input_weights" not in response.text
    assert "output_weights" not in response.text
    assert failure_marker in caplog.text

    if failure_stage == "persistence":
        controlled_save_model.assert_called_once_with(CONTROLLED_MODEL)
    else:
        controlled_save_model.assert_not_called()


@pytest.mark.parametrize(
    "failure_stage",
    [
        "serialization",
        "write",
        "replacement",
    ],
)
def test_train_embed_real_persistence_failures_preserve_previous_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
) -> None:
    failure_marker = f"{failure_stage.upper()}_FAILURE_MARKER"
    previous_document = b'{"previous":true}\n'
    destination = tmp_path / "embedding-weights.json"
    destination.write_bytes(previous_document)

    training_run = ControlledEmbeddingTrainingRun([COMPLETION_EVENT])
    preprocessing = build_controlled_preprocessing()

    monkeypatch.setattr(
        train_embed_route,
        "get_word2vec_preprocessing",
        Mock(return_value=preprocessing),
    )
    monkeypatch.setattr(
        train_embed_route,
        "create_embedding_training_run",
        Mock(return_value=training_run),
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_embedding_result",
        Mock(return_value=CONTROLLED_RESULT),
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_saved_embedding_model",
        Mock(return_value=CONTROLLED_MODEL),
    )
    monkeypatch.setattr(
        train_embed_route,
        "get_embedding_model_directory",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        train_embed_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        train_embed_route,
        "presentation_sleep",
        AsyncMock(),
    )

    if failure_stage == "serialization":

        def fail_serialization(_: SavedEmbeddingModel) -> str:
            raise ValueError(failure_marker)

        monkeypatch.setattr(
            train_embed_route,
            "serialize_saved_embedding_model",
            fail_serialization,
        )

    elif failure_stage == "write":

        def fail_write(path: Path, _: str) -> None:
            path.write_text(
                '{"partial":',
                encoding="utf-8",
            )
            raise OSError(failure_marker)

        monkeypatch.setattr(
            train_embed_route,
            "write_embedding_model_document",
            fail_write,
        )

    else:

        def fail_replacement(_: Path, __: Path) -> None:
            raise OSError(failure_marker)

        monkeypatch.setattr(
            train_embed_route,
            "replace_embedding_model_file",
            fail_replacement,
        )

    caplog.set_level(
        logging.ERROR,
        logger=train_embed_route.__name__,
    )

    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/train-embed",
        json={
            "words": ["alpha"],
            "epochs": 10,
            "dimensions": 4,
            "windowSize": 1,
            "negativeSamples": 1,
        },
    )

    assert response.status_code == 200
    assert destination.read_bytes() == previous_document
    assert [path for path in tmp_path.iterdir() if path.suffix == ".tmp"] == []

    assert "event: done" not in response.text
    assert "event: error" not in response.text
    assert failure_marker not in response.text
    assert "Traceback" not in response.text
    assert str(tmp_path) not in response.text
    assert failure_marker in caplog.text


def test_train_embed_requests_keep_query_result_and_model_state_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_completion = cast(EmbeddingTrainingEvent, object())
    second_completion = cast(EmbeddingTrainingEvent, object())

    first_run = ControlledEmbeddingTrainingRun([first_completion])
    second_run = ControlledEmbeddingTrainingRun([second_completion])
    preprocessing = build_controlled_preprocessing()

    first_result = cast(
        EmbeddingResult,
        {
            **CONTROLLED_RESULT,
            "warnings": ["first"],
        },
    )
    second_result = cast(
        EmbeddingResult,
        {
            **CONTROLLED_RESULT,
            "warnings": ["second"],
        },
    )

    first_model = cast(
        SavedEmbeddingModel,
        {
            **CONTROLLED_MODEL,
            "embeddings": {
                **CONTROLLED_MODEL["embeddings"],
                "alpha": [1.0, 0.0, 0.0, 0.0],
            },
        },
    )
    second_model = cast(
        SavedEmbeddingModel,
        {
            **CONTROLLED_MODEL,
            "embeddings": {
                **CONTROLLED_MODEL["embeddings"],
                "alpha": [2.0, 0.0, 0.0, 0.0],
            },
        },
    )

    observed_queries: list[tuple[str, ...]] = []
    persisted_models: list[SavedEmbeddingModel] = []

    def build_result(
        completion: object,
        _: object,
        query_words: tuple[str, ...],
    ) -> EmbeddingResult:
        observed_queries.append(query_words)

        if completion is first_completion:
            return first_result

        return second_result

    def build_model(
        completion: object,
        _: object,
    ) -> SavedEmbeddingModel:
        if completion is first_completion:
            return first_model

        return second_model

    def save_model(model: SavedEmbeddingModel) -> Path:
        persisted_models.append(model)
        return Path("embedding-weights.json")

    monkeypatch.setattr(
        train_embed_route,
        "get_word2vec_preprocessing",
        Mock(return_value=preprocessing),
    )
    monkeypatch.setattr(
        train_embed_route,
        "create_embedding_training_run",
        Mock(side_effect=[first_run, second_run]),
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_embedding_result",
        build_result,
    )
    monkeypatch.setattr(
        train_embed_route,
        "build_saved_embedding_model",
        build_model,
    )
    monkeypatch.setattr(
        train_embed_route,
        "save_embedding_model",
        save_model,
    )
    monkeypatch.setattr(
        train_embed_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        train_embed_route,
        "presentation_sleep",
        AsyncMock(),
    )

    client = TestClient(app)

    first_response = client.post(
        "/train-embed",
        json={
            "words": [
                "first",
                "first",
            ],
            "epochs": 10,
            "dimensions": 4,
            "windowSize": 1,
            "negativeSamples": 1,
        },
    )

    second_response = client.post(
        "/train-embed",
        json={
            "words": ["second"],
            "epochs": 10,
            "dimensions": 4,
            "windowSize": 1,
            "negativeSamples": 1,
        },
    )

    assert parse_sse_events(first_response.text)[-1] == (
        "done",
        cast(dict[str, object], first_result),
    )
    assert parse_sse_events(second_response.text)[-1] == (
        "done",
        cast(dict[str, object], second_result),
    )

    assert observed_queries == [
        (
            "first",
            "first",
        ),
        ("second",),
    ]
    assert persisted_models == [
        first_model,
        second_model,
    ]
    assert first_run.advance_count == 1
    assert second_run.advance_count == 1


def test_train_embed_minimum_real_boundary_persists_complete_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        train_embed_route,
        "get_embedding_model_directory",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        train_embed_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        train_embed_route,
        "presentation_sleep",
        AsyncMock(),
    )

    client = TestClient(app)

    response = client.post(
        "/train-embed",
        json={
            "words": ["king"],
            "epochs": 10,
            "dimensions": 4,
            "windowSize": 1,
            "negativeSamples": 1,
        },
    )

    assert response.status_code == 200

    events = parse_sse_events(response.text)

    assert events[0][0] == "init"
    assert any(event_name == "epoch" for event_name, _payload in events)
    assert events[-1][0] == "done"
    assert set(events[-1][1]) == {
        "embeddings",
        "neighbors",
        "similarities",
        "analogies",
        "warnings",
    }

    destination = tmp_path / "embedding-weights.json"
    saved_model = json.loads(destination.read_text(encoding="utf-8"))

    assert set(saved_model) == {
        "type",
        "dimensions",
        "vocab",
        "merges",
        "embeddings",
    }
    assert saved_model["type"] == "word2vec-skipgram"
    assert saved_model["dimensions"] == 4
    assert len(saved_model["vocab"]) == len(saved_model["embeddings"])
