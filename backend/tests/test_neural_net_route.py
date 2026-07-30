# backend/tests/test_neural_net_route.py

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Self
from unittest.mock import AsyncMock, Mock, call

import pytest
from fastapi.testclient import TestClient
from how_llms_work.main import app
from how_llms_work.ml.neural_net import (
    MULTI_LAYER_ARCHITECTURE,
    MULTI_LAYER_SUCCESS_VERDICT,
    SINGLE_LAYER_ARCHITECTURE,
    SINGLE_LAYER_SUCCESS_VERDICT,
    EpochUpdate,
    MultiLayerSnapshot,
    NetworkMode,
    Prediction,
    SingleLayerSnapshot,
    TrainingEvent,
    TrainingResult,
)
from how_llms_work.routes import neural_net as neural_net_route
from how_llms_work.schemas import NeuralNetRequest


class ControlledTrainingRun(Iterator[TrainingEvent]):
    def __init__(
        self,
        events: list[TrainingEvent],
        failure: Exception | None = None,
    ) -> None:
        self._events = deque(events)
        self._failure = failure
        self.advance_thread_ids: list[int] = []
        self.advance_process_ids: list[int] = []

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> TrainingEvent:
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

        parsed_events.append((event_name, parsed_data))

    return parsed_events


def build_training_result(
    mode: NetworkMode,
    weight_marker: float,
) -> TrainingResult:
    predictions = [
        Prediction(input=[0, 0], expected=0, actual=0.02),
        Prediction(input=[0, 1], expected=1, actual=0.98),
        Prediction(input=[1, 0], expected=1, actual=0.97),
        Prediction(input=[1, 1], expected=0, actual=0.03),
    ]

    if mode == "single-layer":
        weights: SingleLayerSnapshot = {
            "type": "single-layer",
            "w1": weight_marker,
            "w2": weight_marker + 1.0,
            "bias": weight_marker + 2.0,
        }
        return TrainingResult(
            architecture=SINGLE_LAYER_ARCHITECTURE,
            predictions=predictions,
            verdict=SINGLE_LAYER_SUCCESS_VERDICT,
            weights=weights,
        )

    weights_multi: MultiLayerSnapshot = {
        "type": "multi-layer",
        "w1": [
            [weight_marker, 0.2, 0.3, 0.4],
            [-0.1, -0.2, -0.3, -0.4],
        ],
        "b1": [0.01, 0.02, 0.03, 0.04],
        "w2": [0.5, 0.6, 0.7, 0.8],
        "b2": weight_marker,
    }
    return TrainingResult(
        architecture=MULTI_LAYER_ARCHITECTURE,
        predictions=predictions,
        verdict=MULTI_LAYER_SUCCESS_VERDICT,
        weights=weights_multi,
    )


def test_neural_net_request_exposes_only_frontend_contract_fields() -> None:
    assert set(NeuralNetRequest.model_fields) == {"mode", "epochs"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"mode": "unknown"},
        {"mode": "single-layer", "epochs": "100"},
        {"mode": "single-layer", "epochs": 100.5},
        {"mode": "single-layer", "epochs": True},
        {"mode": "single-layer", "epochs": 99},
        {"mode": "single-layer", "epochs": 100_001},
    ],
    ids=[
        "missing-mode",
        "unknown-mode",
        "numeric-string-epochs",
        "fractional-epochs",
        "boolean-epochs",
        "below-minimum",
        "above-maximum",
    ],
)
def test_neural_net_rejects_invalid_requests_before_streaming(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    training_factory = Mock()
    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        training_factory,
    )
    client = TestClient(app)

    response = client.post(
        "/neural-net",
        json=payload,
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert "event:" not in response.text
    assert "data:" not in response.text
    training_factory.assert_not_called()


@pytest.mark.parametrize("epochs", [100, 100_000])
def test_neural_net_accepts_inclusive_epoch_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    epochs: int,
) -> None:
    result = build_training_result("single-layer", weight_marker=1.0)
    training_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.25),
            result,
        ]
    )
    training_factory = Mock(return_value=training_run)
    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        training_factory,
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        AsyncMock(),
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        neural_net_route,
        "save_network",
        Mock(),
    )
    client = TestClient(app)

    response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": epochs,
        },
    )

    assert response.status_code == 200
    training_factory.assert_called_once_with(
        mode="single-layer",
        epochs=epochs,
    )


def test_neural_net_defaults_epochs_to_five_thousand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_training_result("single-layer", weight_marker=2.0)
    training_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.25),
            result,
        ]
    )
    training_factory = Mock(return_value=training_run)
    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        training_factory,
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        AsyncMock(),
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        neural_net_route,
        "save_network",
        Mock(),
    )
    client = TestClient(app)

    response = client.post(
        "/neural-net",
        json={"mode": "single-layer"},
    )

    assert response.status_code == 200
    training_factory.assert_called_once_with(
        mode="single-layer",
        epochs=5_000,
    )


def test_neural_net_router_and_preserved_routes_are_registered() -> None:
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

    assert health_response.status_code == 200
    assert simple_chat_response.status_code == 422
    assert bpe_response.status_code == 422
    assert neural_net_response.status_code == 422


def test_neural_net_streams_exact_success_contract_and_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_training_result("single-layer", weight_marker=3.0)
    training_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.25),
            EpochUpdate(epoch=2, loss=0.20),
            result,
        ]
    )
    delay_mock = AsyncMock()
    disconnect_mock = AsyncMock(return_value=False)
    save_mock = Mock()
    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        Mock(return_value=training_run),
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        delay_mock,
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        disconnect_mock,
    )
    monkeypatch.setattr(
        neural_net_route,
        "save_network",
        save_mock,
    )
    client = TestClient(app)

    response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": 100,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse_events(response.text)

    assert events == [
        ("epoch", {"epoch": 0, "loss": 0.25}),
        ("epoch", {"epoch": 2, "loss": 0.2}),
        ("done", result.to_frontend_payload()),
    ]
    assert all(set(payload) == {"epoch", "loss"} for _, payload in events[:-1])
    assert set(events[-1][1]) == {
        "architecture",
        "predictions",
        "verdict",
    }
    assert [prediction["input"] for prediction in events[-1][1]["predictions"]] == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    assert "weights" not in response.text
    assert [event_name for event_name, _ in events] == ["epoch", "epoch", "done"]
    delay_mock.assert_has_awaits(
        [
            call(0.02),
            call(0.02),
        ]
    )
    assert delay_mock.await_count == 2
    assert disconnect_mock.await_count == 2
    save_mock.assert_called_once_with(result.weights)
    assert training_run.advance_count == 3


def test_neural_net_advances_each_interval_in_same_process_worker_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_training_result("single-layer", weight_marker=4.0)
    training_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.25),
            EpochUpdate(epoch=2, loss=0.20),
            result,
        ]
    )
    route_thread_ids: list[int] = []
    route_process_ids: list[int] = []

    async def observe_disconnect(_: object) -> bool:
        route_thread_ids.append(threading.get_ident())
        route_process_ids.append(os.getpid())
        return False

    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        Mock(return_value=training_run),
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        AsyncMock(),
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        observe_disconnect,
    )
    monkeypatch.setattr(
        neural_net_route,
        "save_network",
        Mock(),
    )
    client = TestClient(app)

    response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": 100,
        },
    )

    assert response.status_code == 200
    assert training_run.advance_count == 3
    assert route_thread_ids
    assert set(training_run.advance_process_ids) == {os.getpid()}
    assert set(route_process_ids) == {os.getpid()}
    assert all(
        worker_thread_id not in route_thread_ids
        for worker_thread_id in training_run.advance_thread_ids
    )


def test_neural_net_persists_snapshot_before_done(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = build_training_result("single-layer", weight_marker=5.0)
    training_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.25),
            result,
        ]
    )
    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        Mock(return_value=training_run),
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        AsyncMock(),
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        neural_net_route,
        "get_snapshot_directory",
        lambda: tmp_path,
    )
    client = TestClient(app)

    response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": 100,
        },
    )

    destination = tmp_path / "single-layer-weights.json"
    assert response.status_code == 200
    assert destination.is_file()
    assert json.loads(destination.read_text(encoding="utf-8")) == result.weights

    events = parse_sse_events(response.text)
    assert events[-1] == ("done", result.to_frontend_payload())
    assert "weights" not in events[-1][1]


def test_neural_net_disconnect_stops_before_later_intervals_or_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    previous_document = '{"previous": true}\n'
    destination = tmp_path / "single-layer-weights.json"
    destination.write_text(previous_document, encoding="utf-8")

    result = build_training_result("single-layer", weight_marker=6.0)
    training_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.25),
            EpochUpdate(epoch=2, loss=0.20),
            result,
        ]
    )
    delay_mock = AsyncMock()
    save_mock = Mock()
    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        Mock(return_value=training_run),
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        delay_mock,
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        neural_net_route,
        "save_network",
        save_mock,
    )
    client = TestClient(app)

    response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": 100,
        },
    )

    assert response.status_code == 200
    assert parse_sse_events(response.text) == [
        ("epoch", {"epoch": 0, "loss": 0.25}),
    ]
    assert training_run.advance_count == 1
    delay_mock.assert_awaited_once_with(0.02)
    save_mock.assert_not_called()
    assert destination.read_text(encoding="utf-8") == previous_document
    assert "done" not in response.text
    assert "event: error" not in response.text


def test_neural_net_training_failure_terminates_quietly_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure_marker = "TRAINING_FAILURE_MARKER"
    training_run = ControlledTrainingRun(
        [EpochUpdate(epoch=0, loss=0.25)],
        failure=RuntimeError(failure_marker),
    )
    save_mock = Mock()
    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        Mock(return_value=training_run),
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        AsyncMock(),
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        neural_net_route,
        "save_network",
        save_mock,
    )
    caplog.set_level(logging.ERROR, logger=neural_net_route.__name__)
    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": 100,
        },
    )

    assert response.status_code == 200
    assert parse_sse_events(response.text) == [
        ("epoch", {"epoch": 0, "loss": 0.25}),
    ]
    assert "done" not in response.text
    assert "event: error" not in response.text
    assert failure_marker not in response.text
    assert "Traceback" not in response.text
    save_mock.assert_not_called()
    assert failure_marker in caplog.text


def test_neural_net_persistence_failure_preserves_prior_snapshot_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure_marker = "PERSISTENCE_FAILURE_MARKER"
    previous_document = '{"previous": true}\n'
    destination = tmp_path / "single-layer-weights.json"
    destination.write_text(previous_document, encoding="utf-8")

    result = build_training_result("single-layer", weight_marker=7.0)
    training_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.25),
            result,
        ]
    )

    def fail_replacement(_: Path, __: Path) -> None:
        raise OSError(failure_marker)

    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        Mock(return_value=training_run),
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        AsyncMock(),
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        neural_net_route,
        "get_snapshot_directory",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        neural_net_route,
        "replace_snapshot_file",
        fail_replacement,
    )
    caplog.set_level(logging.ERROR, logger=neural_net_route.__name__)
    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": 100,
        },
    )

    assert response.status_code == 200
    assert parse_sse_events(response.text) == [
        ("epoch", {"epoch": 0, "loss": 0.25}),
    ]
    assert destination.read_text(encoding="utf-8") == previous_document
    assert [path for path in tmp_path.iterdir() if path.suffix == ".tmp"] == []
    assert "done" not in response.text
    assert "event: error" not in response.text
    assert failure_marker not in response.text
    assert "Traceback" not in response.text
    assert str(tmp_path) not in response.text
    assert failure_marker in caplog.text


def test_neural_net_requests_keep_training_and_completion_state_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_result = build_training_result("single-layer", weight_marker=8.0)
    second_result = build_training_result("multi-layer", weight_marker=9.0)
    first_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.31),
            first_result,
        ]
    )
    second_run = ControlledTrainingRun(
        [
            EpochUpdate(epoch=0, loss=0.41),
            second_result,
        ]
    )
    created_arguments: list[tuple[NetworkMode, int]] = []
    saved_weights: list[object] = []

    def create_controlled_run(
        mode: NetworkMode,
        epochs: int,
    ) -> ControlledTrainingRun:
        created_arguments.append((mode, epochs))
        return first_run if mode == "single-layer" else second_run

    def capture_save(weights: object) -> Path:
        saved_weights.append(weights)
        return Path("unused.json")

    monkeypatch.setattr(
        neural_net_route,
        "create_training_run",
        create_controlled_run,
    )
    monkeypatch.setattr(
        neural_net_route,
        "presentation_sleep",
        AsyncMock(),
    )
    monkeypatch.setattr(
        neural_net_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        neural_net_route,
        "save_network",
        capture_save,
    )
    client = TestClient(app)

    first_response = client.post(
        "/neural-net",
        json={
            "mode": "single-layer",
            "epochs": 100,
        },
    )
    second_response = client.post(
        "/neural-net",
        json={
            "mode": "multi-layer",
            "epochs": 200,
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert created_arguments == [
        ("single-layer", 100),
        ("multi-layer", 200),
    ]
    assert first_run is not second_run
    assert first_run.advance_count == 2
    assert second_run.advance_count == 2
    assert saved_weights == [
        first_result.weights,
        second_result.weights,
    ]
    assert parse_sse_events(first_response.text)[-1] == (
        "done",
        first_result.to_frontend_payload(),
    )
    assert parse_sse_events(second_response.text)[-1] == (
        "done",
        second_result.to_frontend_payload(),
    )
