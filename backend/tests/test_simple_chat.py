# backend/tests/test_simple_chat.py

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from how_llms_work.main import app
from how_llms_work.routes import simple_chat as simple_chat_route


def parse_sse_events(body: str) -> list[tuple[str, object]]:
    assert body.endswith("\n\n")

    event_blocks = body[:-2].split("\n\n")
    parsed_events: list[tuple[str, object]] = []

    for event_block in event_blocks:
        lines = event_block.splitlines()
        assert len(lines) == 2

        event_line, data_line = lines
        assert event_line.startswith("event: ")
        assert data_line.startswith("data: ")

        event_name = event_line.removeprefix("event: ")
        assert event_name

        data = json.loads(data_line.removeprefix("data: "))
        parsed_events.append((event_name, data))

    return parsed_events


def test_simple_chat_preserves_deterministic_sse_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(simple_chat_route.asyncio, "sleep", sleep_mock)
    client = TestClient(app)

    response = client.post("/simple-chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    expected_words = "Hello! How can I help you today?".split()
    expected_events: list[tuple[str, object]] = [
        ("start", {}),
        *(("word", {"word": word}) for word in expected_words),
        ("done", {}),
    ]
    assert parse_sse_events(response.text) == expected_events

    requested_delays = [call.args[0] for call in sleep_mock.await_args_list]
    assert requested_delays == [1.0, *([0.2] * len(expected_words))]


def test_simple_chat_rejects_empty_message() -> None:
    client = TestClient(app)

    response = client.post("/simple-chat", json={"message": ""})

    assert response.status_code == 422


def test_simple_chat_accepts_whitespace_only_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simple_chat_route.asyncio, "sleep", AsyncMock())
    client = TestClient(app)

    response = client.post("/simple-chat", json={"message": "   "})

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[0] == ("start", {})
    assert events[-1] == ("done", {})
    assert all(event_name == "word" for event_name, _ in events[1:-1])
    assert all(
        isinstance(payload, dict) and set(payload) == {"word"} and isinstance(payload["word"], str)
        for _, payload in events[1:-1]
    )


def test_simple_chat_accepts_long_message_without_application_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simple_chat_route.asyncio, "sleep", AsyncMock())
    client = TestClient(app)

    response = client.post(
        "/simple-chat",
        json={"message": "x" * 10_000},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[0] == ("start", {})
    assert events[-1] == ("done", {})


def test_health_behavior_remains_unchanged() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
