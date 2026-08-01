# backend/tests/test_load_transformer_route.py
from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from how_llms_work import schemas
from how_llms_work.main import app
from how_llms_work.ml.math_utils import Mulberry32
from how_llms_work.ml.transformer import (
    LogicalTrainingShardResult,
    SavedTransformerModel,
    build_saved_transformer_model,
    build_transformer_parameter_layout,
    create_transformer_gradient_buffer,
    create_transformer_training_run,
    get_transformer_preprocessing,
    initialize_transformer_parameters,
)
from how_llms_work.routes import train_transformer as train_transformer_route
from how_llms_work.sse import format_sse as shared_format_sse
from httpx import Response
from pydantic import ValidationError

MODEL_FILENAME: Final = "transformer-weights-e50-l1-d32-h2-ff128-ctx32.json"
LATEST_MODEL_FILENAME: Final = "transformer-weights-e100-l1-d32-h2-ff128-ctx32.json"


RAW_PROMPT: Final = "  ab ba  "
TRIMMED_PROMPT: Final = "ab ba"
GENERATED_TEXT: Final = f"{TRIMMED_PROMPT} generated"


LOAD_FAILURE: Final = "The saved Transformer model could not be loaded."
LATEST_LOAD_FAILURE: Final = "No valid saved Transformer model was found."
EMPTY_PROMPT_FAILURE: Final = "The prompt must not be empty."
UNSUPPORTED_PROMPT_FAILURE: Final = (
    "The prompt contains text that this saved Transformer model cannot tokenize."
)
LONG_PROMPT_FAILURE: Final = "The prompt must contain no more than 16 tokens."
OVERLAP_FAILURE: Final = "Another Transformer request is already running."
START_FAILURE: Final = "Saved Transformer generation could not start."
GENERATION_FAILURE: Final = "The saved Transformer model could not generate text."
GENERATION_TIMEOUT_FAILURE: Final = "Saved Transformer generation exceeded its time limit."
TEST_WAIT_SECONDS: Final = 5.0
TEST_ASYNC_TIMEOUT_SECONDS: Final = 1.0


PUBLIC_REQUEST_FIELDS: Final = (
    "modelFile",
    "prompt",
    "temperature",
    "topP",
    "maxTokens",
)
INTERNAL_REQUEST_FIELDS: Final = (
    "model_file",
    "prompt",
    "temperature",
    "top_p",
    "max_tokens",
)
EXPECTED_ALIASES: Final = {
    "model_file": "modelFile",
    "prompt": "prompt",
    "temperature": "temperature",
    "top_p": "topP",
    "max_tokens": "maxTokens",
}
REQUIRED_ROUTES: Final = {
    ("/health", "GET"),
    ("/simple-chat", "POST"),
    ("/bpe-tokenize", "POST"),
    ("/neural-net", "POST"),
    ("/train-embed", "POST"),
    ("/train-transformer", "POST"),
    ("/load-transformer", "POST"),
}
FORBIDDEN_SUCCESS_EVENT_NAMES: Final = {
    "init",
    "epoch",
    "token",
    "word",
}
FORBIDDEN_SUCCESS_PAYLOAD_KEYS: Final = {
    "architecture",
    "embeddingDim",
    "epoch",
    "ffDim",
    "finalLoss",
    "loss",
    "model",
    "numHeads",
    "numLayers",
    "sample",
    "samples",
    "token",
    "tokens",
    "totalParams",
    "vocabSize",
    "weights",
    "worker",
    "workerCount",
    "workers",
}
PRIVATE_MARKERS: Final = (
    r"C:\private\transformer-model.json",
    "private-loader-detail",
    "private-latest-loader-detail",
    "generation-marker-one",
    "generation-marker-two",
    "logits[7]",
    "traceback",
    "psm_secret_shared_memory",
)


@dataclass(frozen=True, slots=True)
class SSEEvent:
    name: str
    payload: dict[str, object]


class ControlledGeneratedText(str):
    @property
    def text(self) -> str:
        return str(self)


@dataclass(slots=True)
class LifecycleOrderRecorder:
    _entries: list[str] = field(default_factory=list)
    _guard: threading.Lock = field(default_factory=threading.Lock)

    def add(self, entry: str) -> None:
        with self._guard:
            self._entries.append(entry)

    def snapshot(self) -> tuple[str, ...]:
        with self._guard:
            return tuple(self._entries)


@dataclass(slots=True)
class ControlledRunSlot:
    allow_acquire: bool = True
    acquire_calls: int = 0
    release_calls: int = 0
    blocking_arguments: list[bool] = field(default_factory=list)
    order: LifecycleOrderRecorder | None = None
    _locked: bool = False

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        del timeout

        self.blocking_arguments.append(blocking)
        self.acquire_calls += 1

        if not self.allow_acquire or self._locked:
            return False

        self._locked = True
        return True

    def release(self) -> None:
        if not self._locked:
            raise RuntimeError("release unlocked ControlledRunSlot")

        self.release_calls += 1
        self._locked = False

        if self.order is not None:
            self.order.add("slot released")

    def locked(self) -> bool:
        return self._locked


class ControlledCancellationEvent(threading.Event):
    def __init__(self, order: LifecycleOrderRecorder) -> None:
        super().__init__()
        self._order = order

    def set(self) -> None:
        if not self.is_set():
            self._order.add("stop observed")

        super().set()


@dataclass(slots=True)
class RequestCancellationEventFactory:
    order: LifecycleOrderRecorder
    events: list[ControlledCancellationEvent] = field(default_factory=list)

    def __call__(self) -> ControlledCancellationEvent:
        event = ControlledCancellationEvent(self.order)
        self.events.append(event)
        return event


@dataclass(slots=True)
class ControlledMonotonicClock:
    values: list[float]
    observed_values: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("The controlled monotonic clock needs at least one value.")

        value = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        self.observed_values.append(value)
        return value


class ControlledEventLoop:
    def __init__(
        self,
        actual_loop: asyncio.AbstractEventLoop,
        clock: ControlledMonotonicClock,
    ) -> None:
        self._actual_loop = actual_loop
        self._clock = clock

    def time(self) -> float:
        return self._clock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._actual_loop, name)


class ControlledAsyncioModule:
    def __init__(
        self,
        actual_loop: asyncio.AbstractEventLoop,
        clock: ControlledMonotonicClock,
    ) -> None:
        self._loop = ControlledEventLoop(actual_loop, clock)

    def get_running_loop(self) -> ControlledEventLoop:
        return self._loop

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)


@dataclass(slots=True)
class ControlledTokenGenerator:
    order: LifecycleOrderRecorder
    started: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    calculation_count: int = 0
    cancellation_events: list[threading.Event] = field(default_factory=list)

    def __call__(
        self,
        parameters: object,
        vocabulary: object,
        prepared_prompt: object,
        **kwargs: object,
    ) -> ControlledGeneratedText:
        del parameters, vocabulary, prepared_prompt

        cancellation_event = cast(threading.Event, kwargs["cancellation_event"])
        self.cancellation_events.append(cancellation_event)

        for _ in range(2):
            if cancellation_event.is_set():
                break

            self.calculation_count += 1

            if self.calculation_count == 1:
                self.order.add("helper started")
                self.started.set()

                if not cancellation_event.wait(TEST_WAIT_SECONDS):
                    raise AssertionError("The controlled token calculation was not stopped.")

                self.order.add("helper finished")
                self.finished.set()

        return ControlledGeneratedText(GENERATED_TEXT)


@dataclass(frozen=True, slots=True)
class ControlledLoadDependencies:
    slot: ControlledRunSlot
    loader: Mock
    latest_loader: Mock
    prompt_preparation: Mock
    generator: Mock


@dataclass(frozen=True, slots=True)
class ControlledLifecycleDependencies:
    slot: ControlledRunSlot
    loader: Mock
    latest_loader: Mock
    prompt_preparation: Mock
    generator: Mock
    order: LifecycleOrderRecorder
    cancellation_event_factory: RequestCancellationEventFactory


@dataclass(slots=True)
class RequestValidationProbe:
    """Record whether request validation allowed route-owned work to begin."""

    run_slot_acquire: Mock = field(default_factory=Mock)
    model_loader: Mock = field(default_factory=Mock)
    prompt_preparation: Mock = field(default_factory=Mock)
    generator: Mock = field(default_factory=Mock)

    def assert_no_work_started(self) -> None:
        """Assert that invalid structured input stopped all later work."""
        self.run_slot_acquire.assert_not_called()
        self.model_loader.assert_not_called()
        self.prompt_preparation.assert_not_called()
        self.generator.assert_not_called()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}

    for key, value in pairs:
        if key in payload:
            raise AssertionError(f"duplicate JSON key in SSE data: {key}")

        payload[key] = value

    return payload


def _parse_sse(body: str) -> tuple[SSEEvent, ...]:
    normalized = body.replace("\r\n", "\n")

    assert normalized
    assert normalized.endswith("\n\n")

    events: list[SSEEvent] = []

    for block in normalized[:-2].split("\n\n"):
        lines = block.splitlines()

        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")

        event_name = lines[0].removeprefix("event: ")
        decoded_payload = json.loads(
            lines[1].removeprefix("data: "),
            object_pairs_hook=_reject_duplicate_json_keys,
        )

        assert event_name
        assert isinstance(decoded_payload, dict)

        event = SSEEvent(
            name=event_name,
            payload=dict(decoded_payload),
        )
        expected_block = shared_format_sse(
            event.name,
            event.payload,
        ).removesuffix("\n\n")

        assert block == expected_block

        events.append(event)

    return tuple(events)


def _make_loaded_snapshot(
    *,
    model_filename: str = MODEL_FILENAME,
    parameters: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_filename=model_filename,
        config={
            "vocabSize": 3,
            "contextLen": 32,
            "embDim": 32,
            "numHeads": 2,
            "ffDim": 128,
            "numLayers": 1,
        },
        vocabulary=[" ", "a", "b"],
        merges=[],
        parameters=object() if parameters is None else parameters,
    )


def _build_real_saved_transformer_model() -> SavedTransformerModel:
    """Build one complete current-format one-layer model without training math."""
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


def _can_decode_as_exact_vocabulary_token_count(
    text: str,
    vocabulary: list[str],
    *,
    token_count: int,
) -> bool:
    """Return whether text is exactly representable by the requested token count."""
    reachable_positions = {0}

    for _ in range(token_count):
        next_positions: set[int] = set()

        for start_index in reachable_positions:
            for token in vocabulary:
                if token and text.startswith(token, start_index):
                    next_positions.add(start_index + len(token))

        reachable_positions = next_positions

        if not reachable_positions:
            return False

    return len(text) in reachable_positions


def _valid_request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "modelFile": MODEL_FILENAME,
        "prompt": RAW_PROMPT,
        "temperature": 0.8,
        "topP": 0.9,
        "maxTokens": 3,
    }
    payload.update(overrides)
    return payload


def _invalid_request_parameters() -> list[Any]:
    parameters: list[Any] = []

    for field_name in PUBLIC_REQUEST_FIELDS:
        payload = _valid_request()
        del payload[field_name]
        parameters.append(pytest.param(payload, id=f"missing-{field_name}"))

    invalid_values_by_field: dict[str, tuple[object, ...]] = {
        "modelFile": (
            [],
            {},
            7,
            0.8,
            True,
            False,
            "",
        ),
        "prompt": (
            None,
            [],
            {},
            7,
            0.8,
            True,
            False,
        ),
        "temperature": (
            None,
            "0.8",
            True,
            False,
            [],
            {},
            0.099,
            2.001,
        ),
        "topP": (
            None,
            "0.9",
            True,
            False,
            [],
            {},
            0.099,
            1.001,
        ),
        "maxTokens": (
            None,
            "3",
            True,
            False,
            3.0,
            3.5,
            [],
            {},
            2,
            501,
        ),
    }

    for field_name, invalid_values in invalid_values_by_field.items():
        for value_index, invalid_value in enumerate(invalid_values):
            parameters.append(
                pytest.param(
                    _valid_request(**{field_name: invalid_value}),
                    id=f"{field_name}-invalid-{value_index}",
                )
            )

    return parameters


def _install_controlled_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slot_available: bool = True,
    slot: ControlledRunSlot | None = None,
    loader_side_effect: BaseException | None = None,
    latest_loader_side_effect: BaseException | None = None,
    generator_side_effect: object | None = None,
) -> ControlledLoadDependencies:
    controlled_slot = ControlledRunSlot(allow_acquire=slot_available) if slot is None else slot
    loader = Mock(
        name="load_named_transformer_model",
        return_value=_make_loaded_snapshot(),
    )
    latest_loader = Mock(
        name="load_latest_transformer_model",
        return_value=_make_loaded_snapshot(
            model_filename=LATEST_MODEL_FILENAME,
        ),
    )
    prompt_preparation = Mock(
        name="prepare_saved_transformer_prompt",
        wraps=train_transformer_route.prepare_saved_transformer_prompt,
    )
    generator = Mock(
        name="generate_saved_transformer_text",
        return_value=ControlledGeneratedText(GENERATED_TEXT),
    )

    if loader_side_effect is not None:
        loader.side_effect = loader_side_effect

    if latest_loader_side_effect is not None:
        latest_loader.side_effect = latest_loader_side_effect

    if generator_side_effect is not None:
        generator.side_effect = generator_side_effect

    monkeypatch.setattr(
        train_transformer_route,
        "_TRANSFORMER_RUN_SLOT",
        controlled_slot,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "load_named_transformer_model",
        loader,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "load_latest_transformer_model",
        latest_loader,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "prepare_saved_transformer_prompt",
        prompt_preparation,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "generate_saved_transformer_text",
        generator,
    )

    return ControlledLoadDependencies(
        slot=controlled_slot,
        loader=loader,
        latest_loader=latest_loader,
        prompt_preparation=prompt_preparation,
        generator=generator,
    )


def _install_lifecycle_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> ControlledLifecycleDependencies:
    order = LifecycleOrderRecorder()
    slot = ControlledRunSlot(order=order)
    controlled = _install_controlled_dependencies(
        monkeypatch,
        slot=slot,
    )
    cancellation_event_factory = RequestCancellationEventFactory(order)

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        cancellation_event_factory,
    )

    return ControlledLifecycleDependencies(
        slot=controlled.slot,
        loader=controlled.loader,
        latest_loader=controlled.latest_loader,
        prompt_preparation=controlled.prompt_preparation,
        generator=controlled.generator,
        order=order,
        cancellation_event_factory=cancellation_event_factory,
    )


def _post_load_transformer(
    payload: dict[str, object],
    *,
    raise_server_exceptions: bool = True,
) -> Response:
    with TestClient(
        app,
        raise_server_exceptions=raise_server_exceptions,
    ) as client:
        return client.post(
            "/load-transformer",
            content=json.dumps(payload, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )


def _post_train_transformer(
    *,
    raise_server_exceptions: bool = True,
) -> Response:
    with TestClient(
        app,
        raise_server_exceptions=raise_server_exceptions,
    ) as client:
        return client.post(
            "/train-transformer",
            json={
                "epochs": 50,
                "temperature": 0.8,
                "topP": 0.9,
                "numLayers": 1,
                "maxTokens": 3,
            },
        )


async def _collect_saved_transformer_stream(
    stream: AsyncIterator[str],
) -> tuple[SSEEvent, ...]:
    chunks: list[str] = []

    async for chunk in stream:
        chunks.append(chunk)

    return _parse_sse("".join(chunks))


def _assert_controlled_success_response(
    response: Response,
    *,
    selected_filename: str = MODEL_FILENAME,
) -> None:
    _assert_sse_response_headers(response)
    _assert_private_details_are_absent(response.text)

    assert _parse_sse(response.text) == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
        SSEEvent(
            name="result",
            payload={"text": GENERATED_TEXT},
        ),
        SSEEvent(
            name="done",
            payload={},
        ),
    )


def create_request_validation_app(
    probe: RequestValidationProbe,
) -> FastAPI:
    """Create a schema-only FastAPI seam for Step 2 request tests."""
    validation_app = FastAPI()

    @validation_app.post("/load-transformer")
    async def validate_load_transformer_request(
        request: schemas.LoadTransformerRequest,
    ) -> dict[str, object]:
        probe.run_slot_acquire()
        probe.model_loader()
        probe.prompt_preparation()
        probe.generator()

        return request.model_dump(by_alias=True)

    return validation_app


def _registered_methods() -> set[tuple[str, str]]:
    """Return the public HTTP operations exposed through FastAPI's OpenAPI contract."""
    http_methods = {
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "options",
        "head",
        "trace",
    }
    registered: set[tuple[str, str]] = set()

    for path, path_operations in app.openapi()["paths"].items():
        if not isinstance(path_operations, dict):
            continue

        for method in path_operations:
            if method in http_methods:
                registered.add(
                    (
                        path,
                        method.upper(),
                    )
                )

    return registered


def _assert_sse_response_headers(response: Response) -> None:
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def _assert_private_details_are_absent(body: str) -> None:
    lowered_body = body.lower()

    for marker in PRIVATE_MARKERS:
        assert marker.lower() not in lowered_body


def _assert_one_error_event(
    response: Response,
    *,
    expected_message: str,
) -> tuple[SSEEvent, ...]:
    _assert_sse_response_headers(response)
    _assert_private_details_are_absent(response.text)

    events = _parse_sse(response.text)

    assert events == (
        SSEEvent(
            name="error",
            payload={"error": expected_message},
        ),
    )

    return events


def test_load_transformer_route_is_registered_and_existing_routes_remain() -> None:
    assert REQUIRED_ROUTES <= _registered_methods()


def test_transformer_route_descriptions_state_process_local_slot_scope() -> None:
    paths = app.openapi()["paths"]

    training_description = paths["/train-transformer"]["post"]["description"]
    loading_description = paths["/load-transformer"]["post"]["description"]

    assert "shared process-local slot" in training_description
    assert "shared process-local slot" in loading_description


def test_load_transformer_request_model_declares_exact_fields_and_aliases() -> None:
    request_model = getattr(schemas, "LoadTransformerRequest", None)

    assert request_model is not None

    model_fields = request_model.model_fields

    assert tuple(model_fields) == INTERNAL_REQUEST_FIELDS
    assert {
        field_name: field.alias or field_name for field_name, field in model_fields.items()
    } == EXPECTED_ALIASES

    public_schema = request_model.model_json_schema(by_alias=True)

    assert tuple(public_schema["properties"]) == PUBLIC_REQUEST_FIELDS
    assert set(public_schema["required"]) == set(PUBLIC_REQUEST_FIELDS)

    model_file_schema = public_schema["properties"]["modelFile"]
    model_file_options = model_file_schema["anyOf"]

    assert len(model_file_options) == 2
    assert {
        "minLength": 1,
        "type": "string",
    } in model_file_options
    assert {
        "type": "null",
    } in model_file_options


def test_load_transformer_request_accepts_required_null_model_file() -> None:
    probe = RequestValidationProbe()

    with TestClient(create_request_validation_app(probe)) as client:
        response = client.post(
            "/load-transformer",
            json=_valid_request(modelFile=None),
        )

    assert response.status_code == 200
    assert response.json() == {
        "modelFile": None,
        "prompt": RAW_PROMPT,
        "temperature": 0.8,
        "topP": 0.9,
        "maxTokens": 3,
    }

    probe.run_slot_acquire.assert_called_once_with()
    probe.model_loader.assert_called_once_with()
    probe.prompt_preparation.assert_called_once_with()
    probe.generator.assert_called_once_with()


@pytest.mark.parametrize("payload", _invalid_request_parameters())
def test_load_transformer_invalid_requests_are_422_before_any_run_work(
    payload: dict[str, object],
) -> None:
    probe = RequestValidationProbe()

    with TestClient(create_request_validation_app(probe)) as client:
        response = client.post(
            "/load-transformer",
            json=payload,
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert "event:" not in response.text
    assert isinstance(response.json().get("detail"), list)

    probe.assert_no_work_started()


@pytest.mark.parametrize(
    ("temperature", "top_p", "max_tokens"),
    [
        pytest.param(0.1, 0.1, 3, id="minimum-boundaries"),
        pytest.param(2.0, 1.0, 500, id="maximum-boundaries"),
        pytest.param(1, 1, 3, id="integer-valued-json-numbers"),
    ],
)
def test_load_transformer_request_accepts_number_boundaries_and_ignores_extras(
    temperature: float | int,
    top_p: float | int,
    max_tokens: int,
) -> None:
    probe = RequestValidationProbe()
    payload = _valid_request(
        temperature=temperature,
        topP=top_p,
        maxTokens=max_tokens,
        ignored="value",
    )

    with TestClient(create_request_validation_app(probe)) as client:
        response = client.post(
            "/load-transformer",
            json=payload,
        )

    assert response.status_code == 200
    assert response.json() == {
        "modelFile": MODEL_FILENAME,
        "prompt": RAW_PROMPT,
        "temperature": float(temperature),
        "topP": float(top_p),
        "maxTokens": max_tokens,
    }

    probe.run_slot_acquire.assert_called_once_with()
    probe.model_loader.assert_called_once_with()
    probe.prompt_preparation.assert_called_once_with()
    probe.generator.assert_called_once_with()


@pytest.mark.parametrize(
    ("model_file", "prompt"),
    [
        pytest.param(
            "   ",
            RAW_PROMPT,
            id="whitespace-model-file",
        ),
        pytest.param(
            MODEL_FILENAME,
            "",
            id="empty-prompt",
        ),
        pytest.param(
            MODEL_FILENAME,
            "   ",
            id="whitespace-prompt",
        ),
    ],
)
def test_load_transformer_request_leaves_semantic_strings_for_route_validation(
    model_file: str,
    prompt: str,
) -> None:
    probe = RequestValidationProbe()

    with TestClient(create_request_validation_app(probe)) as client:
        response = client.post(
            "/load-transformer",
            json=_valid_request(
                modelFile=model_file,
                prompt=prompt,
            ),
        )

    assert response.status_code == 200
    assert response.json() == {
        "modelFile": model_file,
        "prompt": prompt,
        "temperature": 0.8,
        "topP": 0.9,
        "maxTokens": 3,
    }

    probe.run_slot_acquire.assert_called_once_with()
    probe.model_loader.assert_called_once_with()
    probe.prompt_preparation.assert_called_once_with()
    probe.generator.assert_called_once_with()


@pytest.mark.parametrize(
    "raw_json",
    [
        pytest.param(
            (
                f'{{"modelFile":"{MODEL_FILENAME}","prompt":"ab",'
                '"temperature":NaN,"topP":0.9,"maxTokens":3}'
            ),
            id="temperature-nan",
        ),
        pytest.param(
            (
                f'{{"modelFile":"{MODEL_FILENAME}","prompt":"ab",'
                '"temperature":Infinity,"topP":0.9,"maxTokens":3}'
            ),
            id="temperature-positive-infinity",
        ),
        pytest.param(
            (
                f'{{"modelFile":"{MODEL_FILENAME}","prompt":"ab",'
                '"temperature":-Infinity,"topP":0.9,"maxTokens":3}'
            ),
            id="temperature-negative-infinity",
        ),
        pytest.param(
            (
                f'{{"modelFile":"{MODEL_FILENAME}","prompt":"ab",'
                '"temperature":0.8,"topP":NaN,"maxTokens":3}'
            ),
            id="top-p-nan",
        ),
        pytest.param(
            (
                f'{{"modelFile":"{MODEL_FILENAME}","prompt":"ab",'
                '"temperature":0.8,"topP":Infinity,"maxTokens":3}'
            ),
            id="top-p-positive-infinity",
        ),
        pytest.param(
            (
                f'{{"modelFile":"{MODEL_FILENAME}","prompt":"ab",'
                '"temperature":0.8,"topP":-Infinity,"maxTokens":3}'
            ),
            id="top-p-negative-infinity",
        ),
    ],
)
def test_load_transformer_request_rejects_non_finite_json_numbers(
    raw_json: str,
) -> None:
    with pytest.raises(ValidationError) as error_info:
        schemas.LoadTransformerRequest.model_validate_json(raw_json)

    assert any(error["type"] == "finite_number" for error in error_info.value.errors())


def test_load_transformer_overlap_is_immediate_429_without_load_or_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _install_controlled_dependencies(
        monkeypatch,
        slot_available=False,
    )

    response = _post_load_transformer(_valid_request())

    assert response.status_code == 429
    assert response.json() == {"detail": OVERLAP_FAILURE}
    assert dependencies.slot.acquire_calls == 1
    assert dependencies.slot.blocking_arguments == [False]
    assert dependencies.slot.release_calls == 0
    dependencies.loader.assert_not_called()
    dependencies.generator.assert_not_called()


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
def test_load_transformer_success_stream_is_exact_and_training_free(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    dependencies = _install_controlled_dependencies(monkeypatch)

    response = _post_load_transformer(
        _valid_request(modelFile=model_file),
    )

    _assert_sse_response_headers(response)
    _assert_private_details_are_absent(response.text)

    events = _parse_sse(response.text)

    assert events == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
        SSEEvent(
            name="result",
            payload={"text": GENERATED_TEXT},
        ),
        SSEEvent(
            name="done",
            payload={},
        ),
    )
    assert sum(event.name == "loaded" for event in events) == 1
    assert sum(event.name == "result" for event in events) == 1
    assert sum(event.name == "done" for event in events) == 1

    for event in events:
        assert event.name not in FORBIDDEN_SUCCESS_EVENT_NAMES
        assert event.payload.keys().isdisjoint(FORBIDDEN_SUCCESS_PAYLOAD_KEYS)

    if model_file is None:
        selected_snapshot = dependencies.latest_loader.return_value
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        selected_snapshot = dependencies.loader.return_value
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    dependencies.prompt_preparation.assert_called_once_with(
        RAW_PROMPT,
        selected_snapshot.vocabulary,
        selected_snapshot.merges,
    )
    dependencies.generator.assert_called_once()

    generation_call = dependencies.generator.call_args

    assert generation_call.args[0] is selected_snapshot.parameters
    assert generation_call.args[1] is selected_snapshot.vocabulary
    assert generation_call.args[2].text == TRIMMED_PROMPT
    assert generation_call.kwargs["temperature"] == 0.8
    assert generation_call.kwargs["top_p"] == 0.9
    assert generation_call.kwargs["max_tokens"] == 3
    assert generation_call.kwargs["cancellation_event"] is not None
    assert dependencies.slot.acquire_calls == 1
    assert dependencies.slot.blocking_arguments == [False]
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False


def test_load_transformer_named_model_failure_is_one_exact_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _install_controlled_dependencies(
        monkeypatch,
        loader_side_effect=train_transformer_route.SavedTransformerModelLoadError(
            "private-loader-detail"
        ),
    )

    response = _post_load_transformer(_valid_request())

    _assert_one_error_event(
        response,
        expected_message=LOAD_FAILURE,
    )

    dependencies.loader.assert_called_once_with(MODEL_FILENAME)
    dependencies.latest_loader.assert_not_called()
    dependencies.prompt_preparation.assert_not_called()
    dependencies.generator.assert_not_called()
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False


def test_load_transformer_latest_model_failure_is_one_exact_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _install_controlled_dependencies(
        monkeypatch,
        latest_loader_side_effect=train_transformer_route.SavedTransformerModelLoadError(
            "private-latest-loader-detail"
        ),
    )

    response = _post_load_transformer(
        _valid_request(modelFile=None),
    )

    _assert_one_error_event(
        response,
        expected_message=LATEST_LOAD_FAILURE,
    )

    dependencies.latest_loader.assert_called_once_with()
    dependencies.loader.assert_not_called()
    dependencies.prompt_preparation.assert_not_called()
    dependencies.generator.assert_not_called()
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    "model_file",
    [
        pytest.param(MODEL_FILENAME, id="named"),
        pytest.param(None, id="latest"),
    ],
)
@pytest.mark.parametrize(
    ("prompt", "expected_message"),
    [
        pytest.param("   ", EMPTY_PROMPT_FAILURE, id="empty-after-trim"),
        pytest.param("ab🚫", UNSUPPORTED_PROMPT_FAILURE, id="unsupported-text"),
        pytest.param("a" * 17, LONG_PROMPT_FAILURE, id="seventeen-model-tokens"),
    ],
)
def test_load_transformer_prompt_failures_happen_before_loaded(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    prompt: str,
    expected_message: str,
) -> None:
    dependencies = _install_controlled_dependencies(monkeypatch)

    response = _post_load_transformer(
        _valid_request(
            modelFile=model_file,
            prompt=prompt,
        )
    )

    events = _assert_one_error_event(
        response,
        expected_message=expected_message,
    )

    assert all(event.name not in {"loaded", "result", "done"} for event in events)

    if model_file is None:
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    dependencies.prompt_preparation.assert_called_once()
    dependencies.generator.assert_not_called()
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
def test_load_transformer_generation_failures_are_stable_private_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    dependencies = _install_controlled_dependencies(
        monkeypatch,
        generator_side_effect=[
            RuntimeError(r"generation-marker-one at C:\private\transformer-model.json logits[7]"),
            ValueError("generation-marker-two psm_secret_shared_memory"),
        ],
    )

    first_response = _post_load_transformer(
        _valid_request(modelFile=model_file),
        raise_server_exceptions=False,
    )
    second_response = _post_load_transformer(
        _valid_request(modelFile=model_file),
        raise_server_exceptions=False,
    )

    parsed_responses: list[tuple[SSEEvent, ...]] = []

    for response in (first_response, second_response):
        _assert_sse_response_headers(response)
        _assert_private_details_are_absent(response.text)

        events = _parse_sse(response.text)
        parsed_responses.append(events)

        assert [event.name for event in events] == ["loaded", "error"]
        assert events[0] == SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        )
        assert set(events[1].payload) == {"error"}
        assert isinstance(events[1].payload["error"], str)
        assert events[1].payload["error"]
        assert events[1].payload["error"] not in {
            LOAD_FAILURE,
            LATEST_LOAD_FAILURE,
            EMPTY_PROMPT_FAILURE,
            UNSUPPORTED_PROMPT_FAILURE,
            LONG_PROMPT_FAILURE,
        }
        assert all(event.name not in {"result", "done"} for event in events)

    first_events, second_events = parsed_responses

    assert first_events[1].payload["error"] == second_events[1].payload["error"]

    if model_file is None:
        assert dependencies.latest_loader.call_count == 2
        dependencies.loader.assert_not_called()
    else:
        assert dependencies.loader.call_count == 2
        dependencies.latest_loader.assert_not_called()

    assert dependencies.prompt_preparation.call_count == 2
    assert dependencies.generator.call_count == 2
    assert dependencies.slot.acquire_calls == 2
    assert dependencies.slot.blocking_arguments == [False, False]
    assert dependencies.slot.release_calls == 2
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
def test_load_transformer_operation_timeout_is_generic_not_route_deadline(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    dependencies = _install_controlled_dependencies(
        monkeypatch,
        generator_side_effect=TimeoutError("private-operation-timeout"),
    )

    response = _post_load_transformer(
        _valid_request(modelFile=model_file),
        raise_server_exceptions=False,
    )

    _assert_sse_response_headers(response)
    assert "private-operation-timeout" not in response.text
    assert _parse_sse(response.text) == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
        SSEEvent(
            name="error",
            payload={"error": GENERATION_FAILURE},
        ),
    )
    assert GENERATION_TIMEOUT_FAILURE not in response.text

    if model_file is None:
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    dependencies.prompt_preparation.assert_called_once()
    dependencies.generator.assert_called_once()
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
@pytest.mark.parametrize(
    ("outcome", "first_payload", "expected_first_event_names"),
    [
        pytest.param(
            "load-error",
            _valid_request(),
            ("error",),
            id="load-error",
        ),
        pytest.param(
            "empty-prompt",
            _valid_request(prompt="   "),
            ("error",),
            id="empty-prompt",
        ),
        pytest.param(
            "unsupported-prompt",
            _valid_request(prompt="ab🚫"),
            ("error",),
            id="unsupported-prompt",
        ),
        pytest.param(
            "prompt-overlength",
            _valid_request(prompt="a" * 17),
            ("error",),
            id="prompt-overlength",
        ),
        pytest.param(
            "success",
            _valid_request(),
            ("loaded", "result", "done"),
            id="success",
        ),
        pytest.param(
            "generation-error",
            _valid_request(),
            ("loaded", "error"),
            id="generation-error",
        ),
    ],
)
def test_load_transformer_releases_slot_for_next_valid_request(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
    outcome: str,
    first_payload: dict[str, object],
    expected_first_event_names: tuple[str, ...],
) -> None:
    dependencies = _install_controlled_dependencies(monkeypatch)
    selected_loader = dependencies.latest_loader if model_file is None else dependencies.loader
    other_loader = dependencies.loader if model_file is None else dependencies.latest_loader

    if outcome == "load-error":
        selected_loader.side_effect = [
            train_transformer_route.SavedTransformerModelLoadError("private-loader-detail"),
            _make_loaded_snapshot(
                model_filename=selected_filename,
            ),
        ]
    elif outcome == "generation-error":
        dependencies.generator.side_effect = [
            RuntimeError("generation-marker-one"),
            ControlledGeneratedText(GENERATED_TEXT),
        ]

    selected_first_payload = dict(first_payload)
    selected_first_payload["modelFile"] = model_file

    first_response = _post_load_transformer(
        selected_first_payload,
        raise_server_exceptions=False,
    )

    assert (
        tuple(event.name for event in _parse_sse(first_response.text)) == expected_first_event_names
    )
    assert dependencies.slot.locked() is False

    second_response = _post_load_transformer(
        _valid_request(modelFile=model_file),
    )
    second_events = _parse_sse(second_response.text)

    assert second_events == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
        SSEEvent(
            name="result",
            payload={"text": GENERATED_TEXT},
        ),
        SSEEvent(
            name="done",
            payload={},
        ),
    )
    assert selected_loader.call_count == 2
    other_loader.assert_not_called()
    assert dependencies.slot.acquire_calls == 2
    assert dependencies.slot.blocking_arguments == [False, False]
    assert dependencies.slot.release_calls == 2
    assert dependencies.slot.locked() is False


def test_load_transformer_response_preparation_failure_releases_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _install_controlled_dependencies(monkeypatch)
    original_create_sse_response = train_transformer_route.create_sse_response
    response_construction_calls = 0

    def create_sse_response_with_first_failure(
        event_stream: Any,
    ) -> Any:
        nonlocal response_construction_calls

        response_construction_calls += 1

        if response_construction_calls == 1:
            raise RuntimeError("private response preparation failure")

        return original_create_sse_response(event_stream)

    monkeypatch.setattr(
        train_transformer_route,
        "create_sse_response",
        create_sse_response_with_first_failure,
    )

    first_response = _post_load_transformer(
        _valid_request(),
        raise_server_exceptions=False,
    )

    assert first_response.status_code == 500
    assert first_response.json() == {"detail": START_FAILURE}
    assert "event:" not in first_response.text
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False
    dependencies.loader.assert_not_called()
    dependencies.generator.assert_not_called()

    second_response = _post_load_transformer(_valid_request())

    assert tuple(event.name for event in _parse_sse(second_response.text)) == (
        "loaded",
        "result",
        "done",
    )
    assert response_construction_calls == 2
    assert dependencies.slot.acquire_calls == 2
    assert dependencies.slot.blocking_arguments == [False, False]
    assert dependencies.slot.release_calls == 2
    assert dependencies.slot.locked() is False
    dependencies.loader.assert_called_once_with(MODEL_FILENAME)
    dependencies.generator.assert_called_once()


def test_active_training_slot_rejects_load_without_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_slot = ControlledRunSlot(_locked=True)
    dependencies = _install_controlled_dependencies(
        monkeypatch,
        slot=shared_slot,
    )

    rejected_response = _post_load_transformer(_valid_request())

    assert rejected_response.status_code == 429
    assert rejected_response.json() == {"detail": OVERLAP_FAILURE}
    assert shared_slot.acquire_calls == 1
    assert shared_slot.blocking_arguments == [False]
    assert shared_slot.release_calls == 0
    assert shared_slot.locked() is True
    dependencies.loader.assert_not_called()
    dependencies.generator.assert_not_called()

    shared_slot.release()

    dependencies.loader.assert_not_called()
    dependencies.generator.assert_not_called()

    recovery_response = _post_load_transformer(_valid_request())

    _assert_controlled_success_response(recovery_response)
    assert shared_slot.acquire_calls == 2
    assert shared_slot.blocking_arguments == [False, False]
    assert shared_slot.release_calls == 2
    assert shared_slot.locked() is False
    dependencies.loader.assert_called_once_with(MODEL_FILENAME)
    dependencies.generator.assert_called_once()


def test_active_load_slot_rejects_training_without_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_slot = ControlledRunSlot(_locked=True)
    preprocessing_getter = Mock(
        name="get_transformer_preprocessing",
        side_effect=AssertionError("training preparation must not start"),
    )

    monkeypatch.setattr(
        train_transformer_route,
        "_TRANSFORMER_RUN_SLOT",
        shared_slot,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "get_transformer_preprocessing",
        preprocessing_getter,
    )

    rejected_response = _post_train_transformer()

    assert rejected_response.status_code == 429
    assert rejected_response.json() == {"detail": OVERLAP_FAILURE}
    assert shared_slot.acquire_calls == 1
    assert shared_slot.blocking_arguments == [False]
    assert shared_slot.release_calls == 0
    assert shared_slot.locked() is True
    preprocessing_getter.assert_not_called()

    shared_slot.release()

    preprocessing_getter.assert_not_called()
    assert shared_slot.acquire_calls == 1
    assert shared_slot.release_calls == 1
    assert shared_slot.locked() is False


@pytest.mark.parametrize(
    "model_file",
    [
        pytest.param(MODEL_FILENAME, id="named"),
        pytest.param(None, id="latest"),
    ],
)
def test_load_transformer_creates_no_training_workers_resources_or_labels(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
) -> None:
    dependencies = _install_controlled_dependencies(monkeypatch)
    training_tripwires = {
        "get_transformer_preprocessing": Mock(
            side_effect=AssertionError("load route requested training preprocessing")
        ),
        "initialize_transformer_parameters": Mock(
            side_effect=AssertionError("load route initialized training parameters")
        ),
        "create_transformer_training_run": Mock(
            side_effect=AssertionError("load route created a training run")
        ),
        "create_request_scoped_worker_group": Mock(
            side_effect=AssertionError("load route created training workers")
        ),
        "generate_transformer_text": Mock(
            side_effect=AssertionError("load route used the training sample generator")
        ),
        "save_transformer_model": Mock(
            side_effect=AssertionError("load route persisted training state")
        ),
    }

    for name, tripwire in training_tripwires.items():
        monkeypatch.setattr(
            train_transformer_route,
            name,
            tripwire,
        )

    response = _post_load_transformer(
        _valid_request(modelFile=model_file),
    )
    events = _parse_sse(response.text)

    assert tuple(event.name for event in events) == (
        "loaded",
        "result",
        "done",
    )

    if model_file is None:
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    for tripwire in training_tripwires.values():
        tripwire.assert_not_called()

    for forbidden_text in (
        "Transformer worker processes",
        "event: init",
        "event: epoch",
        '"architecture"',
        '"finalLoss"',
        '"samples"',
    ):
        assert forbidden_text not in response.text

    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
def test_identical_load_requests_preserve_request_isolation_with_distinct_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    first_snapshot = _make_loaded_snapshot(
        model_filename=selected_filename,
        parameters=object(),
    )
    second_snapshot = _make_loaded_snapshot(
        model_filename=selected_filename,
        parameters=object(),
    )
    dependencies = _install_controlled_dependencies(monkeypatch)
    selected_loader = dependencies.latest_loader if model_file is None else dependencies.loader
    other_loader = dependencies.loader if model_file is None else dependencies.latest_loader
    selected_loader.side_effect = [
        first_snapshot,
        second_snapshot,
    ]
    generation_inputs: list[
        tuple[
            object,
            object,
            object,
            object,
            ControlledGeneratedText,
        ]
    ] = []

    def generate_from_request_snapshot(
        parameters: object,
        vocabulary: object,
        prepared_prompt: object,
        **kwargs: object,
    ) -> ControlledGeneratedText:
        generated_text = ControlledGeneratedText(GENERATED_TEXT)
        generation_inputs.append(
            (
                parameters,
                vocabulary,
                prepared_prompt,
                kwargs["cancellation_event"],
                generated_text,
            )
        )

        return generated_text

    dependencies.generator.side_effect = generate_from_request_snapshot

    first_response = _post_load_transformer(
        _valid_request(modelFile=model_file),
    )
    second_response = _post_load_transformer(
        _valid_request(modelFile=model_file),
    )

    first_events = _parse_sse(first_response.text)
    second_events = _parse_sse(second_response.text)

    assert first_events == second_events
    assert first_events[0] == SSEEvent(
        name="loaded",
        payload={
            "file": selected_filename,
            "prompt": TRIMMED_PROMPT,
        },
    )
    assert first_events[1] == SSEEvent(
        name="result",
        payload={"text": GENERATED_TEXT},
    )
    assert selected_loader.call_count == 2
    other_loader.assert_not_called()
    assert dependencies.prompt_preparation.call_count == 2
    assert dependencies.generator.call_count == 2

    first_prompt_call, second_prompt_call = dependencies.prompt_preparation.call_args_list

    assert first_prompt_call.args == (
        RAW_PROMPT,
        first_snapshot.vocabulary,
        first_snapshot.merges,
    )
    assert second_prompt_call.args == (
        RAW_PROMPT,
        second_snapshot.vocabulary,
        second_snapshot.merges,
    )
    assert first_snapshot.vocabulary is not second_snapshot.vocabulary
    assert first_snapshot.merges is not second_snapshot.merges

    (
        first_parameters,
        first_vocabulary,
        first_prompt,
        first_cancellation,
        first_generated_text,
    ) = generation_inputs[0]
    (
        second_parameters,
        second_vocabulary,
        second_prompt,
        second_cancellation,
        second_generated_text,
    ) = generation_inputs[1]

    assert first_parameters is first_snapshot.parameters
    assert second_parameters is second_snapshot.parameters
    assert first_parameters is not second_parameters
    assert first_vocabulary is first_snapshot.vocabulary
    assert second_vocabulary is second_snapshot.vocabulary
    assert first_vocabulary is not second_vocabulary
    assert first_prompt is not second_prompt
    assert first_prompt.text == TRIMMED_PROMPT
    assert second_prompt.text == TRIMMED_PROMPT
    assert first_prompt.token_ids == second_prompt.token_ids
    assert first_cancellation is not second_cancellation
    assert first_generated_text == second_generated_text
    assert first_generated_text is not second_generated_text
    assert dependencies.slot.acquire_calls == 2
    assert dependencies.slot.release_calls == 2
    assert dependencies.slot.locked() is False


def test_named_and_latest_requests_keep_model_prompt_settings_and_output_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_snapshot = _make_loaded_snapshot(
        model_filename=MODEL_FILENAME,
        parameters=object(),
    )
    latest_snapshot = _make_loaded_snapshot(
        model_filename=LATEST_MODEL_FILENAME,
        parameters=object(),
    )
    latest_snapshot.vocabulary[:] = [" ", "b", "a"]

    dependencies = _install_controlled_dependencies(monkeypatch)
    dependencies.loader.return_value = named_snapshot
    dependencies.latest_loader.return_value = latest_snapshot
    generation_calls: list[
        tuple[
            object,
            object,
            object,
            float,
            float,
            int,
            object,
            ControlledGeneratedText,
        ]
    ] = []

    def generate_aligned_request(
        parameters: object,
        vocabulary: object,
        prepared_prompt: object,
        **kwargs: object,
    ) -> ControlledGeneratedText:
        temperature = cast(float, kwargs["temperature"])
        top_p = cast(float, kwargs["top_p"])
        max_tokens = cast(int, kwargs["max_tokens"])
        cancellation_event = kwargs["cancellation_event"]

        if parameters is named_snapshot.parameters:
            assert vocabulary is named_snapshot.vocabulary
            assert prepared_prompt.text == "ab"
            assert prepared_prompt.token_ids == (0, 1, 2)
            assert temperature == 0.6
            assert top_p == 0.7
            assert max_tokens == 3
            generated_text = ControlledGeneratedText("ab named-result")
        elif parameters is latest_snapshot.parameters:
            assert vocabulary is latest_snapshot.vocabulary
            assert prepared_prompt.text == "ba"
            assert prepared_prompt.token_ids == (0, 1, 2)
            assert temperature == 1.4
            assert top_p == 0.95
            assert max_tokens == 4
            generated_text = ControlledGeneratedText("ba latest-result")
        else:
            raise AssertionError("Generation received parameters from no selected request.")

        generation_calls.append(
            (
                parameters,
                vocabulary,
                prepared_prompt,
                temperature,
                top_p,
                max_tokens,
                cancellation_event,
                generated_text,
            )
        )

        return generated_text

    dependencies.generator.side_effect = generate_aligned_request

    named_response = _post_load_transformer(
        _valid_request(
            modelFile=MODEL_FILENAME,
            prompt="  ab  ",
            temperature=0.6,
            topP=0.7,
            maxTokens=3,
        )
    )
    latest_response = _post_load_transformer(
        _valid_request(
            modelFile=None,
            prompt="  ba  ",
            temperature=1.4,
            topP=0.95,
            maxTokens=4,
        )
    )

    assert _parse_sse(named_response.text) == (
        SSEEvent(
            name="loaded",
            payload={
                "file": MODEL_FILENAME,
                "prompt": "ab",
            },
        ),
        SSEEvent(
            name="result",
            payload={"text": "ab named-result"},
        ),
        SSEEvent(
            name="done",
            payload={},
        ),
    )
    assert _parse_sse(latest_response.text) == (
        SSEEvent(
            name="loaded",
            payload={
                "file": LATEST_MODEL_FILENAME,
                "prompt": "ba",
            },
        ),
        SSEEvent(
            name="result",
            payload={"text": "ba latest-result"},
        ),
        SSEEvent(
            name="done",
            payload={},
        ),
    )

    dependencies.loader.assert_called_once_with(MODEL_FILENAME)
    dependencies.latest_loader.assert_called_once_with()

    named_prompt_call, latest_prompt_call = dependencies.prompt_preparation.call_args_list

    assert named_prompt_call.args == (
        "  ab  ",
        named_snapshot.vocabulary,
        named_snapshot.merges,
    )
    assert latest_prompt_call.args == (
        "  ba  ",
        latest_snapshot.vocabulary,
        latest_snapshot.merges,
    )
    assert named_snapshot.parameters is not latest_snapshot.parameters
    assert named_snapshot.vocabulary is not latest_snapshot.vocabulary
    assert named_snapshot.merges is not latest_snapshot.merges
    assert generation_calls[0][0] is named_snapshot.parameters
    assert generation_calls[1][0] is latest_snapshot.parameters
    assert generation_calls[0][1] is named_snapshot.vocabulary
    assert generation_calls[1][1] is latest_snapshot.vocabulary
    assert generation_calls[0][2] is not generation_calls[1][2]
    assert generation_calls[0][6] is not generation_calls[1][6]
    assert generation_calls[0][7] is not generation_calls[1][7]
    assert dependencies.slot.acquire_calls == 2
    assert dependencies.slot.release_calls == 2
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    "model_file",
    [
        pytest.param(MODEL_FILENAME, id="named"),
        pytest.param(None, id="latest"),
    ],
)
def test_load_transformer_real_model_integration_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_file: str | None,
) -> None:
    model_directory = tmp_path / "models"
    model_directory.mkdir()

    model = _build_real_saved_transformer_model()
    model_path = model_directory / MODEL_FILENAME
    model_path.write_text(
        train_transformer_route.serialize_saved_transformer_model(model),
        encoding="utf-8",
        newline="\n",
    )
    original_model_bytes = model_path.read_bytes()

    original_read_bytes = Path.read_bytes
    selected_model_reads: list[Path] = []

    def recording_read_bytes(candidate: Path) -> bytes:
        if candidate == model_path:
            selected_model_reads.append(candidate)

        return original_read_bytes(candidate)

    worker_factory = Mock(
        name="create_request_scoped_worker_group",
        side_effect=AssertionError(
            "Saved Transformer generation must not create training workers."
        ),
    )
    real_generator = train_transformer_route.generate_saved_transformer_text
    generation = Mock(
        name="generate_saved_transformer_text",
        wraps=real_generator,
    )

    monkeypatch.setattr(
        train_transformer_route,
        "get_transformer_model_directory",
        lambda: model_directory,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "create_request_scoped_worker_group",
        worker_factory,
    )
    monkeypatch.setattr(
        train_transformer_route,
        "generate_saved_transformer_text",
        generation,
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        recording_read_bytes,
    )

    trimmed_prompt = "once upon a"
    payload = {
        "modelFile": model_file,
        "prompt": f"  {trimmed_prompt}  ",
        "temperature": 0.8,
        "topP": 0.9,
        "maxTokens": 3,
    }

    with TestClient(app) as client:
        first_response = client.post(
            "/load-transformer",
            json=payload,
        )
        second_response = client.post(
            "/load-transformer",
            json=payload,
        )

    _assert_sse_response_headers(first_response)
    _assert_sse_response_headers(second_response)

    first_events = _parse_sse(first_response.text)
    second_events = _parse_sse(second_response.text)

    assert tuple(event.name for event in first_events) == (
        "loaded",
        "result",
        "done",
    )
    assert first_events[0] == SSEEvent(
        name="loaded",
        payload={
            "file": MODEL_FILENAME,
            "prompt": trimmed_prompt,
        },
    )
    assert set(first_events[1].payload) == {"text"}
    assert first_events[2] == SSEEvent(
        name="done",
        payload={},
    )
    assert second_events == first_events

    first_text = first_events[1].payload["text"]
    second_text = second_events[1].payload["text"]

    assert type(first_text) is str
    assert type(second_text) is str
    assert first_text.startswith(trimmed_prompt)
    assert second_text == first_text

    assert generation.call_count == 2

    for generation_call in generation.call_args_list:
        vocabulary = generation_call.args[1]

        assert type(vocabulary) is list
        assert generation_call.kwargs["temperature"] == 0.8
        assert generation_call.kwargs["top_p"] == 0.9
        assert generation_call.kwargs["max_tokens"] == 3
        assert _can_decode_as_exact_vocabulary_token_count(
            first_text[len(trimmed_prompt) :],
            vocabulary,
            token_count=3,
        )

    assert selected_model_reads == [
        model_path,
        model_path,
    ]
    assert original_read_bytes(model_path) == original_model_bytes
    worker_factory.assert_not_called()


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
@pytest.mark.asyncio
async def test_load_transformer_generation_deadline_is_absolute_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    dependencies = _install_lifecycle_dependencies(monkeypatch)
    token_generator = ControlledTokenGenerator(dependencies.order)
    dependencies.generator.side_effect = token_generator

    disconnect_observer = AsyncMock(return_value=False)
    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        disconnect_observer,
    )

    actual_loop = asyncio.get_running_loop()
    clock = ControlledMonotonicClock([1_000.0, 1_299.9, 1_300.0])
    monkeypatch.setattr(
        train_transformer_route,
        "asyncio",
        ControlledAsyncioModule(actual_loop, clock),
    )

    assert dependencies.slot.acquire(blocking=False) is True
    stream = train_transformer_route.stream_saved_transformer_generation(
        request=cast(Any, object()),
        model_filename=model_file,
        prompt=RAW_PROMPT,
        temperature=0.8,
        top_p=0.9,
        max_tokens=3,
    )

    async def collect_stream_body() -> str:
        chunks: list[str] = []

        async for chunk in stream:
            chunks.append(chunk)

        return "".join(chunks)

    try:
        response_body = await asyncio.wait_for(
            collect_stream_body(),
            timeout=TEST_ASYNC_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        pytest.fail("Saved Transformer generation did not enforce its deadline.")

    expected_body = shared_format_sse(
        "loaded",
        {
            "file": selected_filename,
            "prompt": TRIMMED_PROMPT,
        },
    ) + shared_format_sse(
        "error",
        {"error": GENERATION_TIMEOUT_FAILURE},
    )

    assert response_body == expected_body

    events = _parse_sse(response_body)

    assert events == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
        SSEEvent(
            name="error",
            payload={"error": GENERATION_TIMEOUT_FAILURE},
        ),
    )

    lowered_body = response_body.lower()

    for private_marker in (
        r"c:\private\deadline-model.json",
        "timeouterror",
        "traceback",
        "task",
        "future",
        "thread",
        "token_id",
        "tokenids",
        "model_array",
        "sharedmemory",
        "psm_",
        "1000.0",
        "1299.9",
        "1300.0",
    ):
        assert private_marker.lower() not in lowered_body

    assert token_generator.calculation_count == 1
    assert token_generator.started.is_set()
    assert token_generator.finished.is_set()
    assert len(dependencies.cancellation_event_factory.events) == 1
    assert dependencies.cancellation_event_factory.events[0].is_set()

    observed_values = clock.observed_values
    assert observed_values[0] == 1_000.0
    assert observed_values[-1] - observed_values[0] == 300.0

    first_run_order = dependencies.order.snapshot()
    assert first_run_order.index("helper started") < first_run_order.index("stop observed")
    assert first_run_order.index("stop observed") < first_run_order.index("helper finished")
    assert first_run_order.index("helper finished") < first_run_order.index("slot released")
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False

    if model_file is None:
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    monkeypatch.setattr(train_transformer_route, "asyncio", asyncio)
    dependencies.generator.side_effect = None
    dependencies.generator.return_value = ControlledGeneratedText(GENERATED_TEXT)

    recovery_response = await asyncio.to_thread(
        _post_load_transformer,
        _valid_request(modelFile=model_file),
    )
    _assert_controlled_success_response(
        recovery_response,
        selected_filename=selected_filename,
    )
    assert dependencies.slot.release_calls == 2


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
@pytest.mark.asyncio
async def test_load_transformer_disconnect_after_generation_completion_emits_no_late_success(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    dependencies = _install_lifecycle_dependencies(monkeypatch)
    generation_finished = threading.Event()

    def complete_generation(
        parameters: object,
        vocabulary: object,
        prepared_prompt: object,
        **kwargs: object,
    ) -> ControlledGeneratedText:
        del parameters, vocabulary, prepared_prompt, kwargs
        generation_finished.set()
        return ControlledGeneratedText(GENERATED_TEXT)

    dependencies.generator.side_effect = complete_generation

    async def disconnect_after_generation_finished(
        request: object,
    ) -> bool:
        del request
        return generation_finished.is_set()

    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        AsyncMock(side_effect=disconnect_after_generation_finished),
    )

    assert dependencies.slot.acquire(blocking=False) is True
    stream = train_transformer_route.stream_saved_transformer_generation(
        request=cast(Any, object()),
        model_filename=model_file,
        prompt=RAW_PROMPT,
        temperature=0.8,
        top_p=0.9,
        max_tokens=3,
    )

    events = await asyncio.wait_for(
        _collect_saved_transformer_stream(stream),
        timeout=TEST_ASYNC_TIMEOUT_SECONDS,
    )

    assert events == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
    )
    assert generation_finished.is_set()
    dependencies.generator.assert_called_once()

    if model_file is None:
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    assert len(dependencies.cancellation_event_factory.events) == 1
    assert dependencies.cancellation_event_factory.events[0].is_set()
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
def test_load_transformer_disconnect_drains_generation_without_late_success(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    dependencies = _install_lifecycle_dependencies(monkeypatch)
    token_generator = ControlledTokenGenerator(dependencies.order)
    dependencies.generator.side_effect = token_generator

    async def disconnect_after_generation_starts(request: object) -> bool:
        del request
        return token_generator.started.is_set()

    disconnect_observer = AsyncMock(side_effect=disconnect_after_generation_starts)
    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        disconnect_observer,
    )

    response = _post_load_transformer(
        _valid_request(modelFile=model_file),
    )
    _assert_sse_response_headers(response)
    assert _parse_sse(response.text) == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
    )

    assert token_generator.calculation_count == 1
    assert token_generator.finished.is_set()
    assert len(dependencies.cancellation_event_factory.events) == 1
    assert dependencies.cancellation_event_factory.events[0].is_set()

    first_run_order = dependencies.order.snapshot()
    assert first_run_order.index("helper started") < first_run_order.index("stop observed")
    assert first_run_order.index("stop observed") < first_run_order.index("helper finished")
    assert first_run_order.index("helper finished") < first_run_order.index("slot released")
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False

    if model_file is None:
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    disconnect_observer.side_effect = None
    disconnect_observer.return_value = False
    dependencies.generator.side_effect = None
    dependencies.generator.return_value = ControlledGeneratedText(GENERATED_TEXT)

    recovery_response = _post_load_transformer(
        _valid_request(modelFile=model_file),
    )
    _assert_controlled_success_response(
        recovery_response,
        selected_filename=selected_filename,
    )
    assert dependencies.slot.release_calls == 2


@pytest.mark.parametrize(
    ("model_file", "selected_filename"),
    [
        pytest.param(MODEL_FILENAME, MODEL_FILENAME, id="named"),
        pytest.param(None, LATEST_MODEL_FILENAME, id="latest"),
    ],
)
@pytest.mark.asyncio
async def test_load_transformer_task_cancellation_drains_generation_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    model_file: str | None,
    selected_filename: str,
) -> None:
    dependencies = _install_lifecycle_dependencies(monkeypatch)
    token_generator = ControlledTokenGenerator(dependencies.order)
    dependencies.generator.side_effect = token_generator
    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        AsyncMock(return_value=False),
    )

    assert dependencies.slot.acquire(blocking=False) is True
    stream = train_transformer_route.stream_saved_transformer_generation(
        request=cast(Any, object()),
        model_filename=model_file,
        prompt=RAW_PROMPT,
        temperature=0.8,
        top_p=0.9,
        max_tokens=3,
    )

    loaded_chunk = await anext(stream)
    assert _parse_sse(loaded_chunk) == (
        SSEEvent(
            name="loaded",
            payload={
                "file": selected_filename,
                "prompt": TRIMMED_PROMPT,
            },
        ),
    )

    generation_task = asyncio.create_task(anext(stream))
    assert await asyncio.to_thread(
        token_generator.started.wait,
        TEST_WAIT_SECONDS,
    )

    generation_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await generation_task

    assert token_generator.calculation_count == 1
    assert token_generator.finished.is_set()
    assert len(dependencies.cancellation_event_factory.events) == 1
    assert dependencies.cancellation_event_factory.events[0].is_set()

    first_run_order = dependencies.order.snapshot()
    assert first_run_order.index("helper started") < first_run_order.index("stop observed")
    assert first_run_order.index("stop observed") < first_run_order.index("helper finished")
    assert first_run_order.index("helper finished") < first_run_order.index("slot released")
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False

    if model_file is None:
        dependencies.latest_loader.assert_called_once_with()
        dependencies.loader.assert_not_called()
    else:
        dependencies.loader.assert_called_once_with(MODEL_FILENAME)
        dependencies.latest_loader.assert_not_called()

    dependencies.generator.side_effect = None
    dependencies.generator.return_value = ControlledGeneratedText(GENERATED_TEXT)

    recovery_response = await asyncio.to_thread(
        _post_load_transformer,
        _valid_request(modelFile=model_file),
    )
    _assert_controlled_success_response(
        recovery_response,
        selected_filename=selected_filename,
    )
    assert dependencies.slot.release_calls == 2


def test_load_transformer_release_waits_for_interrupted_helper_to_finish_using_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _install_lifecycle_dependencies(monkeypatch)
    snapshot = _make_loaded_snapshot(parameters=object())
    dependencies.loader.return_value = snapshot

    helper_started = threading.Event()
    helper_observed_stop = threading.Event()
    allow_helper_to_finish = threading.Event()
    helper_finished = threading.Event()
    state_in_use: list[object] = []
    first_result: dict[str, Response] = {}

    def blocked_generation(
        parameters: object,
        vocabulary: object,
        prepared_prompt: object,
        **kwargs: object,
    ) -> ControlledGeneratedText:
        del vocabulary, prepared_prompt

        cancellation_event = cast(
            threading.Event,
            kwargs["cancellation_event"],
        )
        state_in_use.append(parameters)
        dependencies.order.add("helper started")
        helper_started.set()

        if not cancellation_event.wait(TEST_WAIT_SECONDS):
            raise AssertionError("The interrupted generation helper did not observe stopping.")

        helper_observed_stop.set()

        if not allow_helper_to_finish.wait(TEST_WAIT_SECONDS):
            raise AssertionError("The interrupted generation helper was not released.")

        state_in_use.clear()
        dependencies.order.add("helper finished")
        helper_finished.set()

        return ControlledGeneratedText(GENERATED_TEXT)

    dependencies.generator.side_effect = blocked_generation

    async def disconnect_after_generation_starts(
        request: object,
    ) -> bool:
        del request
        return helper_started.is_set()

    disconnect_observer = AsyncMock(side_effect=disconnect_after_generation_starts)
    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        disconnect_observer,
    )

    def run_first_request() -> None:
        first_result["response"] = _post_load_transformer(_valid_request())

    first_thread = threading.Thread(
        target=run_first_request,
        daemon=True,
    )
    first_thread.start()

    try:
        assert helper_started.wait(TEST_WAIT_SECONDS)
        assert helper_observed_stop.wait(TEST_WAIT_SECONDS)
        assert state_in_use == [snapshot.parameters]
        assert dependencies.slot.locked() is True
        assert dependencies.slot.release_calls == 0

        overlap_response = _post_load_transformer(_valid_request())

        assert overlap_response.status_code == 429
        assert overlap_response.json() == {
            "detail": OVERLAP_FAILURE,
        }
        assert first_thread.is_alive()
        assert helper_finished.is_set() is False
        assert state_in_use == [snapshot.parameters]
        assert dependencies.slot.release_calls == 0
    finally:
        allow_helper_to_finish.set()
        first_thread.join(timeout=TEST_WAIT_SECONDS)

    assert first_thread.is_alive() is False
    assert helper_finished.is_set()
    assert state_in_use == []

    first_response = first_result["response"]
    _assert_sse_response_headers(first_response)
    assert _parse_sse(first_response.text) == (
        SSEEvent(
            name="loaded",
            payload={
                "file": MODEL_FILENAME,
                "prompt": TRIMMED_PROMPT,
            },
        ),
    )

    first_run_order = dependencies.order.snapshot()
    assert first_run_order.index("helper started") < first_run_order.index("stop observed")
    assert first_run_order.index("stop observed") < first_run_order.index("helper finished")
    assert first_run_order.index("helper finished") < first_run_order.index("slot released")
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False

    disconnect_observer.side_effect = None
    disconnect_observer.return_value = False
    dependencies.generator.side_effect = None
    dependencies.generator.return_value = ControlledGeneratedText(GENERATED_TEXT)

    recovery_response = _post_load_transformer(_valid_request())
    _assert_controlled_success_response(recovery_response)
    assert dependencies.slot.release_calls == 2


def test_load_transformer_blocking_load_keeps_health_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependencies = _install_lifecycle_dependencies(monkeypatch)
    load_started = threading.Event()
    allow_load_to_finish = threading.Event()

    def blocking_loader(model_filename: str) -> object:
        assert model_filename == MODEL_FILENAME
        dependencies.order.add("helper started")
        load_started.set()

        if not allow_load_to_finish.wait(TEST_WAIT_SECONDS):
            raise AssertionError("The controlled model load was not released.")

        dependencies.order.add("helper finished")
        return _make_loaded_snapshot()

    dependencies.loader.side_effect = blocking_loader

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            load_future = executor.submit(
                client.post,
                "/load-transformer",
                content=json.dumps(_valid_request()),
                headers={"Content-Type": "application/json"},
            )

            try:
                assert load_started.wait(TEST_WAIT_SECONDS)
                health_response = client.get("/health")
            finally:
                allow_load_to_finish.set()

            load_response = load_future.result(timeout=TEST_WAIT_SECONDS)

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}
    _assert_controlled_success_response(load_response)

    order = dependencies.order.snapshot()
    assert order.index("helper started") < order.index("helper finished")
    assert order.index("helper finished") < order.index("slot released")
    assert dependencies.slot.release_calls == 1
    assert dependencies.slot.locked() is False
