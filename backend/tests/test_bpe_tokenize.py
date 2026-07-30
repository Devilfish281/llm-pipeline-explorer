# backend/tests/test_bpe_tokenize.py

import json  # Added Code
from unittest.mock import AsyncMock, Mock  # Added Code

import pytest  # Added Code
from fastapi.testclient import TestClient  # Added Code
from how_llms_work.main import app  # Added Code
from how_llms_work.routes import bpe_tokenize as bpe_tokenize_route  # Added Code


def parse_sse_events(
    body: str,
) -> list[tuple[str, dict[str, object]]]:
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


def test_bpe_tokenize_streams_reference_compatible_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        bpe_tokenize_route.asyncio,
        "sleep",
        sleep_mock,
    )
    client = TestClient(app)

    response = client.post(
        "/bpe-tokenize",
        json={"message": "cat cat car"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse_events(response.text)

    assert [event_name for event_name, _ in events] == [
        "init",
        "merge",
        "merge",
        "merge",
        "result",
    ]

    assert set(events[0][1]) == {
        "corpus",
        "characters",
        "charCount",
        "wordCount",
    }

    for _, merge_payload in events[1:4]:
        assert set(merge_payload) == {
            "step",
            "pair",
            "frequency",
            "newToken",
            "vocabSize",
            "tokenCount",
        }

    assert set(events[-1][1]) == {
        "inputTokens",
        "tokenCount",
        "originalCharCount",
        "compressionRatio",
    }

    expected_events: list[tuple[str, dict[str, object]]] = [
        (
            "init",
            {
                "corpus": "cat cat car",
                "characters": list("cat cat car"),
                "charCount": 11,
                "wordCount": 3,
            },
        ),
        (
            "merge",
            {
                "step": 1,
                "pair": ["c", "a"],
                "frequency": 3,
                "newToken": "ca",
                "vocabSize": 6,
                "tokenCount": 8,
            },
        ),
        (
            "merge",
            {
                "step": 2,
                "pair": ["ca", "t"],
                "frequency": 2,
                "newToken": "cat",
                "vocabSize": 7,
                "tokenCount": 6,
            },
        ),
        (
            "merge",
            {
                "step": 3,
                "pair": ["ca", "r"],
                "frequency": 1,
                "newToken": "car",
                "vocabSize": 8,
                "tokenCount": 5,
            },
        ),
        (
            "result",
            {
                "inputTokens": ["cat", " ", "cat", " ", "car"],
                "tokenCount": 5,
                "originalCharCount": 11,
                "compressionRatio": "2.2x",
            },
        ),
    ]

    assert events == expected_events
    sleep_mock.assert_awaited_once_with(0.8)


def test_bpe_tokenize_truncates_init_characters_without_changing_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        bpe_tokenize_route.asyncio,
        "sleep",
        sleep_mock,
    )
    client = TestClient(app)
    message = "!" * 205

    response = client.post(
        "/bpe-tokenize",
        json={"message": message},
    )

    assert response.status_code == 200

    events = parse_sse_events(response.text)
    init_event_name, init_payload = events[0]
    result_event_name, result_payload = events[-1]

    assert init_event_name == "init"
    assert init_payload["corpus"] == message
    assert init_payload["characters"] == list(message[:200])
    assert init_payload["charCount"] == len(message)
    assert init_payload["wordCount"] == 1

    assert result_event_name == "result"
    assert result_payload["originalCharCount"] == len(message)
    assert result_payload["inputTokens"] == list(message)
    assert result_payload["tokenCount"] == len(message)
    assert result_payload["compressionRatio"] == "1.0x"

    sleep_mock.assert_awaited_once_with(0.8)


def test_bpe_tokenize_rejects_empty_message() -> None:
    client = TestClient(app)

    response = client.post(
        "/bpe-tokenize",
        json={"message": ""},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    ("message", "expected_tokens", "expected_word_count"),
    [
        ("   ", [" ", " ", " "], 1),
        ("!!!", ["!", "!", "!"], 1),
        ("x", ["x"], 1),
    ],
    ids=[
        "whitespace-only",
        "punctuation-only",
        "single-character",
    ],
)
def test_bpe_tokenize_accepts_minimal_unmodified_inputs(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    expected_tokens: list[str],
    expected_word_count: int,
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        bpe_tokenize_route.asyncio,
        "sleep",
        sleep_mock,
    )
    client = TestClient(app)

    response = client.post(
        "/bpe-tokenize",
        json={"message": message},
    )

    assert response.status_code == 200

    events = parse_sse_events(response.text)
    assert [event_name for event_name, _ in events] == [
        "init",
        "result",
    ]

    init_payload = events[0][1]
    result_payload = events[1][1]

    assert init_payload == {
        "corpus": message,
        "characters": list(message),
        "charCount": len(message),
        "wordCount": expected_word_count,
    }
    assert result_payload == {
        "inputTokens": expected_tokens,
        "tokenCount": len(expected_tokens),
        "originalCharCount": len(message),
        "compressionRatio": "1.0x",
    }

    sleep_mock.assert_awaited_once_with(0.8)


def test_bpe_tokenize_requests_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_mock = AsyncMock()
    monkeypatch.setattr(
        bpe_tokenize_route.asyncio,
        "sleep",
        sleep_mock,
    )
    client = TestClient(app)

    first_response = client.post(
        "/bpe-tokenize",
        json={"message": "cat cat car"},
    )
    second_response = client.post(
        "/bpe-tokenize",
        json={"message": "xy"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_events = parse_sse_events(first_response.text)
    second_events = parse_sse_events(second_response.text)

    assert first_events[-1] == (
        "result",
        {
            "inputTokens": ["cat", " ", "cat", " ", "car"],
            "tokenCount": 5,
            "originalCharCount": 11,
            "compressionRatio": "2.2x",
        },
    )

    assert second_events == [
        (
            "init",
            {
                "corpus": "xy",
                "characters": ["x", "y"],
                "charCount": 2,
                "wordCount": 1,
            },
        ),
        (
            "merge",
            {
                "step": 1,
                "pair": ["x", "y"],
                "frequency": 1,
                "newToken": "xy",
                "vocabSize": 3,
                "tokenCount": 1,
            },
        ),
        (
            "result",
            {
                "inputTokens": ["xy"],
                "tokenCount": 1,
                "originalCharCount": 2,
                "compressionRatio": "2.0x",
            },
        ),
    ]

    requested_delays = [call.args[0] for call in sleep_mock.await_args_list]
    assert requested_delays == [0.8, 0.8]


def test_bpe_tokenize_does_not_expose_internal_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_marker = "BPE_INTERNAL_FAILURE_MARKER"
    training_mock = Mock(side_effect=RuntimeError(failure_marker))
    monkeypatch.setattr(
        bpe_tokenize_route,
        "train_bpe",
        training_mock,
    )
    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/bpe-tokenize",
        json={"message": "cat"},
    )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Internal Server Error"
    assert failure_marker not in response.text
    assert "Traceback" not in response.text
    assert "event:" not in response.text
    assert "data:" not in response.text
    training_mock.assert_called_once()
