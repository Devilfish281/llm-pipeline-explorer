# backend/tests/test_train_transformer_route.py
from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, cast
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from how_llms_work.main import app
from how_llms_work.ml.transformer import (
    GeneratedTextSample,
    InitializedTransformerParameters,
    LogicalTrainingShard,
    LogicalTrainingShardResult,
    SavedTransformerModel,
    TransformerEpochObservation,
    TransformerEpochUpdate,
    TransformerParameterLayout,
    TransformerPreprocessingSnapshot,
    TransformerTrainingRun,
    TransformerTrainingSequence,
)
from how_llms_work.ml.transformer_worker import (
    RequestScopedWorkerGroup,
    RequestScopedWorkerGroupCleanupFailureCode,
    RequestScopedWorkerGroupCleanupReport,
    RequestScopedWorkerGroupFailureCode,
    RequestScopedWorkerGroupState,
)
from how_llms_work.routes import train_transformer as train_transformer_route
from how_llms_work.schemas import TrainTransformerRequest
from how_llms_work.sse import format_sse as shared_format_sse
from pydantic import ValidationError

VALID_REQUEST: Final[dict[str, object]] = {
    "epochs": 50,
    "temperature": 0.8,
    "topP": 0.9,
    "numLayers": 2,
    "maxTokens": 3,
}

TOTAL_PARAMETERS: Final = 12_345
FINAL_LOSS: Final = 0.123456

EXPECTED_INIT: Final[dict[str, object]] = {
    "vocabSize": 3,
    "contextLen": 32,
    "embeddingDim": 32,
    "numHeads": 2,
    "ffDim": 128,
    "numLayers": 2,
    "totalParams": TOTAL_PARAMETERS,
    "temperature": 0.8,
    "topP": 0.9,
    "corpusSentences": 2,
    "trainingSequences": 4,
}

PRIVATE_MARKERS: Final[tuple[str, ...]] = (
    r"c:\private\transformer-model.json",
    "psm_secret_shared_memory",
    "protocol-version=999",
    "traceback-marker",
    "weights=[nan, 42.0]",
)


CONTROLLED_MODEL = cast(
    SavedTransformerModel,
    {
        "type": "decoder-transformer",
        "config": {
            "numLayers": 2,
        },
    },
)


class SentinelRouteFailure(RuntimeError):
    """Failure text that must remain private."""

    def __init__(self) -> None:
        super().__init__(" | ".join(PRIVATE_MARKERS))


@dataclass(frozen=True, slots=True)
class ParsedEvent:
    """One parsed SSE event plus framing diagnostics."""

    name: str
    payload: dict[str, object]
    duplicate_fields: tuple[str, ...]
    duplicate_payload_keys: tuple[str, ...]
    unexpected_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedStream:
    """One complete ordered SSE parse."""

    events: tuple[ParsedEvent, ...]
    trailing_data: str


def _load_json_object(
    value: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """Decode one JSON object while retaining duplicate-key evidence."""
    duplicates: list[str] = []

    def object_hook(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}

        for key, item in pairs:
            if key in result:
                duplicates.append(key)

            result[key] = item

        return result

    decoded = json.loads(
        value,
        object_pairs_hook=object_hook,
    )

    assert isinstance(decoded, dict)

    return (
        cast(dict[str, object], decoded),
        tuple(duplicates),
    )


def parse_sse(body: str) -> ParsedStream:
    """Parse SSE while retaining duplicate fields and unterminated data."""
    parts = body.split("\n\n")
    complete_blocks = parts[:-1]
    trailing_data = "" if body.endswith("\n\n") else parts[-1]
    events: list[ParsedEvent] = []

    for block in complete_blocks:
        assert block

        values: dict[str, list[str]] = {}
        duplicates: list[str] = []
        unexpected: list[str] = []

        for line in block.splitlines():
            assert ":" in line

            name, raw_value = line.split(
                ":",
                maxsplit=1,
            )
            value = raw_value[1:] if raw_value.startswith(" ") else raw_value

            if name in values:
                duplicates.append(name)

            if name not in {
                "event",
                "data",
            }:
                unexpected.append(name)

            values.setdefault(
                name,
                [],
            ).append(value)

        assert values.get("event")
        assert values.get("data")

        payload, duplicate_payload_keys = _load_json_object(values["data"][0])

        events.append(
            ParsedEvent(
                name=values["event"][0],
                payload=payload,
                duplicate_fields=tuple(duplicates),
                duplicate_payload_keys=duplicate_payload_keys,
                unexpected_fields=tuple(unexpected),
            )
        )

    return ParsedStream(
        events=tuple(events),
        trailing_data=trailing_data,
    )


def assert_exact_sse(
    parsed: ParsedStream,
) -> None:
    """Require the exact shared two-line SSE contract."""
    assert parsed.trailing_data == ""
    assert parsed.events

    for event in parsed.events:
        assert event.duplicate_fields == ()
        assert event.duplicate_payload_keys == ()
        assert event.unexpected_fields == ()


@dataclass(slots=True)
class CallOrder:
    """Thread-safe ordered route lifecycle observations."""

    values: list[str] = field(default_factory=list)
    lock: Any = field(
        default_factory=threading.Lock,
        repr=False,
    )

    def add(
        self,
        value: str,
    ) -> None:
        with self.lock:
            self.values.append(value)

    def snapshot(self) -> tuple[str, ...]:
        with self.lock:
            return tuple(self.values)


class ControlledRunSlot:
    """Nonblocking process-local run-slot substitute."""

    def __init__(
        self,
        order: CallOrder,
    ) -> None:
        self.order = order
        self.held = False
        self.guard = threading.Lock()
        self.acquire_count = 0
        self.release_count = 0

    def acquire(
        self,
        blocking: bool = True,
        timeout: float = -1.0,
    ) -> bool:
        del timeout

        assert blocking is False

        with self.guard:
            self.acquire_count += 1
            self.order.add("validate")
            self.order.add("reserve")

            if self.held:
                return False

            self.held = True
            return True

    def release(self) -> None:
        with self.guard:
            if not self.held:
                raise RuntimeError("Transformer run slot is not held.")

            self.held = False
            self.release_count += 1
            self.order.add("slot release")

    def locked(self) -> bool:
        with self.guard:
            return self.held

    def reset_for_testing(self) -> None:
        with self.guard:
            self.held = False


def _is_run_slot(
    name: str,
    value: object,
) -> bool:
    """Identify a route-owned Transformer run slot by behavior."""
    normalized = name.lower().replace(
        "_",
        "",
    )

    return (
        "transformer" in normalized
        and ("runslot" in normalized or "runlock" in normalized)
        and callable(
            getattr(
                value,
                "acquire",
                None,
            )
        )
        and callable(
            getattr(
                value,
                "release",
                None,
            )
        )
    )


def _patch_run_slot(
    monkeypatch: pytest.MonkeyPatch,
    slot: ControlledRunSlot,
) -> None:
    """Replace the run slot without requiring one private spelling."""
    patched = False

    for name, value in tuple(vars(train_transformer_route).items()):
        if _is_run_slot(
            name,
            value,
        ):
            monkeypatch.setattr(
                train_transformer_route,
                name,
                slot,
            )
            patched = True

    if not patched:
        monkeypatch.setattr(
            train_transformer_route,
            "_TRANSFORMER_RUN_SLOT",
            slot,
            raising=False,
        )


@pytest.fixture(autouse=True)
def restore_transformer_run_slot() -> Iterator[None]:
    """Prevent one failed test from reserving the next test's slot."""
    yield

    for name, value in tuple(vars(train_transformer_route).items()):
        if not _is_run_slot(
            name,
            value,
        ):
            continue

        reset = getattr(
            value,
            "reset_for_testing",
            None,
        )

        if callable(reset):
            reset()
            continue

        locked = getattr(
            value,
            "locked",
            None,
        )
        release = getattr(
            value,
            "release",
            None,
        )

        try:
            if callable(locked) and callable(release) and bool(locked()):
                release()
        except (
            RuntimeError,
            ValueError,
        ):
            pass


def controlled_preprocessing() -> TransformerPreprocessingSnapshot:
    """Build a small immutable route-visible preprocessing substitute."""
    sequences = tuple(
        TransformerTrainingSequence(
            input_ids=(
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
            ),
            target_ids=(
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                1,
            ),
        )
        for _ in range(4)
    )

    shards = tuple(
        LogicalTrainingShard(
            shard_index=index,
            start_index=index,
            stop_index=index + 1,
        )
        for index in range(4)
    )

    return cast(
        TransformerPreprocessingSnapshot,
        SimpleNamespace(
            corpus=(
                "alpha beta gamma",
                "gamma beta alpha",
            ),
            vocabulary=(
                " alpha",
                " beta",
                " gamma",
            ),
            training_sequences=sequences,
            logical_training_shards=shards,
            generation_seed_ids=(
                0,
                1,
                2,
            ),
            merges=(),
        ),
    )


class ControlledTrainingRun:
    """Inclusive report-schedule substitute without Transformer math."""

    def __init__(
        self,
        parameters: InitializedTransformerParameters,
        epochs: int,
        shards: tuple[LogicalTrainingShard, ...],
        order: CallOrder,
    ) -> None:
        self.parameters = parameters
        self.weights = parameters.storage
        self.requested_epochs = epochs
        self.logical_training_shards = shards
        self.next_epoch = 0
        self.last_completed_epoch: int | None = None
        self.last_completed_loss: float | None = None
        self.is_complete = False
        self.is_failed = False
        self.advance_epochs: list[int] = []
        self.result_counts: list[int] = []
        self.order = order
        self.report_step = max(
            1,
            epochs // 50,
        )

    @property
    def is_active(self) -> bool:
        return not self.is_complete and not self.is_failed

    def advance_epoch(
        self,
        results: Sequence[LogicalTrainingShardResult],
    ) -> TransformerEpochObservation:
        assert self.is_active
        assert len(results) == 4

        epoch = self.next_epoch
        loss = 1.0 / float(epoch + 1)

        self.order.add("Adam commit")
        self.advance_epochs.append(epoch)
        self.result_counts.append(len(results))

        update = (
            TransformerEpochUpdate(
                epoch=epoch,
                loss=round(
                    loss,
                    6,
                ),
            )
            if (epoch % self.report_step == 0 or epoch == self.requested_epochs)
            else None
        )

        self.last_completed_epoch = epoch
        self.last_completed_loss = loss
        self.next_epoch += 1

        if epoch == self.requested_epochs:
            self.is_complete = True

        return TransformerEpochObservation(
            epoch=epoch,
            loss=loss,
            update=update,
        )


class ControlledWorkerGroup:
    """One-command-at-a-time worker group with explicit cleanup."""

    def __init__(
        self,
        shards: tuple[LogicalTrainingShard, ...],
        order: CallOrder,
        failures: set[str],
        compute_gate: threading.Event,
    ) -> None:
        self.shards = shards
        self.order = order
        self.failures = failures
        self.compute_gate = compute_gate
        self.compute_started = threading.Event()
        self.compute_epochs: list[int] = []
        self.weight_ids: list[int] = []
        self.state = RequestScopedWorkerGroupState.READY
        self.cleanup_report: RequestScopedWorkerGroupCleanupReport | None = None
        self.primary_failure_code = (
            RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
            if "cleanup_primary" in failures
            else None
        )
        self.in_compute = False
        self.cleanup_attempts = 0
        self.cleanup_started = threading.Event()
        self.cleanup_gate = threading.Event()

        if "cleanup_blocked" not in failures:
            self.cleanup_gate.set()

    @property
    def successful(self) -> bool:
        return (
            self.state is RequestScopedWorkerGroupState.CLOSED
            and self.primary_failure_code is None
            and self.cleanup_report is not None
            and self.cleanup_report.successful
        )

    async def compute_epoch(
        self,
        epoch: int,
        current_weights: np.ndarray[
            Any,
            np.dtype[np.float32],
        ],
    ) -> tuple[LogicalTrainingShardResult, ...]:
        assert not self.in_compute, "Epoch commands were pipelined."
        assert self.state is RequestScopedWorkerGroupState.READY

        self.in_compute = True
        self.state = RequestScopedWorkerGroupState.COMPUTING
        self.compute_epochs.append(epoch)
        self.weight_ids.append(id(current_weights))
        self.order.add("epoch compute")
        self.compute_started.set()

        try:
            await asyncio.to_thread(self.compute_gate.wait)

            if "epoch" in self.failures:
                raise SentinelRouteFailure

            return tuple(
                cast(
                    LogicalTrainingShardResult,
                    SimpleNamespace(
                        shard=shard,
                        processed_sequence_count=1,
                        loss=0.25,
                        gradient=object(),
                    ),
                )
                for shard in self.shards
            )
        finally:
            self.in_compute = False

            if self.state is RequestScopedWorkerGroupState.COMPUTING:
                self.state = RequestScopedWorkerGroupState.READY

    async def cleanup(
        self,
    ) -> RequestScopedWorkerGroupCleanupReport:
        self.cleanup_attempts += 1

        if self.cleanup_report is not None:
            return self.cleanup_report

        self.state = RequestScopedWorkerGroupState.STOPPING
        self.cleanup_started.set()
        await asyncio.to_thread(self.cleanup_gate.wait)

        if "cleanup" in self.failures and self.cleanup_attempts == 1:
            self.order.add("worker cleanup failure")
            raise SentinelRouteFailure

        if "cleanup_kill" in self.failures:
            cooperative_shutdown_completed = False
            terminate_required = True
            kill_required = True
            process_exit_codes = (1,)
            secondary_failures = (RequestScopedWorkerGroupCleanupFailureCode.KILL,)
        elif "cleanup_terminate" in self.failures or "cleanup_unsuccessful" in self.failures:
            cooperative_shutdown_completed = False
            terminate_required = True
            kill_required = False
            process_exit_codes = (1,)
            secondary_failures = ()
        elif "cleanup_secondary" in self.failures:
            cooperative_shutdown_completed = True
            terminate_required = False
            kill_required = False
            process_exit_codes = (0,)
            secondary_failures = (RequestScopedWorkerGroupCleanupFailureCode.PIPE_CLOSE,)
        else:
            cooperative_shutdown_completed = True
            terminate_required = False
            kill_required = False
            process_exit_codes = (0,)
            secondary_failures = ()

        self.cleanup_report = RequestScopedWorkerGroupCleanupReport(
            cooperative_shutdown_completed=cooperative_shutdown_completed,
            terminate_required=terminate_required,
            kill_required=kill_required,
            process_exit_codes=process_exit_codes,
            secondary_failures=secondary_failures,
        )

        self.state = RequestScopedWorkerGroupState.CLOSED
        self.order.add("worker cleanup success" if self.successful else "worker cleanup failure")

        return self.cleanup_report


@dataclass(slots=True)
class Dependencies:
    """Installed test doubles and their observations."""

    order: CallOrder
    failures: set[str]
    preprocessing: TransformerPreprocessingSnapshot
    slot: ControlledRunSlot
    compute_gate: threading.Event
    preprocessing_getter: Mock
    initializer: Mock
    worker_factory: AsyncMock
    sample_generator: Mock
    final_evaluator: Mock
    model_builder: Mock
    save_model: Mock
    parameters: list[InitializedTransformerParameters]
    runs: list[ControlledTrainingRun]
    groups: list[ControlledWorkerGroup]


@dataclass(slots=True)
class RequestValidationProbe:
    """Record whether validation allowed later route work to begin."""

    run_slot_acquire: Mock = field(default_factory=Mock)
    preprocessing: Mock = field(default_factory=Mock)
    worker_factory: AsyncMock = field(default_factory=AsyncMock)
    persistence: Mock = field(default_factory=Mock)

    def assert_no_work_started(self) -> None:
        """Assert that request validation stopped all later route stages."""
        self.run_slot_acquire.assert_not_called()
        self.preprocessing.assert_not_called()
        self.worker_factory.assert_not_awaited()
        self.persistence.assert_not_called()


def install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    failures: set[str] | None = None,
    block_epoch: bool = False,
) -> Dependencies:
    """Install narrow doubles around expensive public boundaries."""
    order = CallOrder()
    active_failures = failures if failures is not None else set()
    preprocessing = controlled_preprocessing()
    slot = ControlledRunSlot(order)
    compute_gate = threading.Event()

    if not block_epoch:
        compute_gate.set()

    parameters: list[InitializedTransformerParameters] = []
    runs: list[ControlledTrainingRun] = []
    groups: list[ControlledWorkerGroup] = []

    def get_preprocessing() -> TransformerPreprocessingSnapshot:
        order.add("preprocess/init")

        if "preprocess" in active_failures:
            raise SentinelRouteFailure

        return preprocessing

    def build_layout(
        num_layers: int,
    ) -> TransformerParameterLayout:
        if "layout" in active_failures:
            raise SentinelRouteFailure

        return TransformerParameterLayout(
            num_layers=num_layers,
            vocabulary_size=3,
            records=(),
            total_float_count=TOTAL_PARAMETERS,
            total_byte_count=TOTAL_PARAMETERS * 4,
        )

    def initialize(
        layout: TransformerParameterLayout,
        generator: object,
    ) -> InitializedTransformerParameters:
        del generator

        order.add("initialize")

        if "initialize" in active_failures:
            raise SentinelRouteFailure

        storage = np.zeros(
            layout.total_float_count,
            dtype=np.float32,
        )

        result = cast(
            InitializedTransformerParameters,
            SimpleNamespace(
                layout=layout,
                storage=storage,
                views=object(),
            ),
        )

        parameters.append(result)

        return result

    def create_run(
        initialized: InitializedTransformerParameters,
        *,
        sequence_count: int,
        requested_epochs: int,
    ) -> TransformerTrainingRun:
        assert sequence_count == len(preprocessing.training_sequences)

        result = ControlledTrainingRun(
            initialized,
            requested_epochs,
            preprocessing.logical_training_shards,
            order,
        )

        runs.append(result)

        return cast(
            TransformerTrainingRun,
            result,
        )

    async def create_group(
        num_layers: int,
        current_weights: np.ndarray[
            Any,
            np.dtype[np.float32],
        ],
        training_sequences: tuple[
            TransformerTrainingSequence,
            ...,
        ],
        logical_shards: tuple[
            LogicalTrainingShard,
            ...,
        ],
        **kwargs: object,
    ) -> RequestScopedWorkerGroup:
        del kwargs

        order.add("worker startup")

        assert num_layers == 2
        assert training_sequences == preprocessing.training_sequences
        assert logical_shards == preprocessing.logical_training_shards
        assert current_weights is parameters[-1].storage

        if "worker_startup" in active_failures:
            raise SentinelRouteFailure

        result = ControlledWorkerGroup(
            logical_shards,
            order,
            active_failures,
            compute_gate,
        )

        groups.append(result)

        return cast(
            RequestScopedWorkerGroup,
            result,
        )

    async def disconnected(
        request: object,
    ) -> bool:
        del request
        return False

    async def delay(
        seconds: float,
    ) -> None:
        assert seconds == 0.02
        order.add("delay")

    def sample(
        parameters_value: InitializedTransformerParameters,
        preprocessing_value: TransformerPreprocessingSnapshot,
        *,
        epoch: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        cancellation_event: object,
    ) -> GeneratedTextSample:
        del cancellation_event

        order.add("sample")

        assert parameters_value is runs[-1].parameters
        assert preprocessing_value is preprocessing
        assert (
            temperature,
            top_p,
            max_tokens,
        ) == (
            0.8,
            0.9,
            3,
        )

        if "sample" in active_failures:
            raise SentinelRouteFailure

        return GeneratedTextSample(
            epoch=epoch,
            text=f"controlled sample {epoch}",
        )

    def final_evaluation(
        run: TransformerTrainingRun,
        preprocessing_value: TransformerPreprocessingSnapshot,
        *,
        cancellation_event: object,
    ) -> float:
        del cancellation_event

        order.add("final evaluation")

        assert run is cast(
            TransformerTrainingRun,
            runs[-1],
        )
        assert preprocessing_value is preprocessing

        if "final_evaluation" in active_failures:
            raise SentinelRouteFailure

        return FINAL_LOSS

    def build_model(
        run: TransformerTrainingRun,
        preprocessing_value: TransformerPreprocessingSnapshot,
    ) -> SavedTransformerModel:
        order.add("model build")

        assert run is cast(
            TransformerTrainingRun,
            runs[-1],
        )
        assert preprocessing_value is preprocessing

        if "model_build" in active_failures:
            raise SentinelRouteFailure

        return CONTROLLED_MODEL

    def persist(
        model: SavedTransformerModel,
        *,
        epochs: int,
        model_directory: Path | None = None,
    ) -> Path:
        del model_directory

        order.add("persist")

        assert model is CONTROLLED_MODEL
        assert type(epochs) is int

        if "persist" in active_failures:
            raise SentinelRouteFailure

        destination = tmp_path / (f"transformer-weights-e{epochs}-l2-d32-h2-ff128-ctx32.json")
        destination.write_text(
            "controlled\n",
            encoding="utf-8",
        )

        return destination

    def record_sse(
        event: str,
        data: dict[str, object],
    ) -> str:
        stage = {
            "init": "init yield",
            "epoch": "epoch yield",
            "done": "done",
        }.get(event)

        if stage is not None:
            order.add(stage)

        return shared_format_sse(
            event,
            data,
        )

    preprocessing_getter = Mock(side_effect=get_preprocessing)
    initializer = Mock(side_effect=initialize)
    worker_factory = AsyncMock(side_effect=create_group)
    sample_generator = Mock(side_effect=sample)
    final_evaluator = Mock(side_effect=final_evaluation)
    model_builder = Mock(side_effect=build_model)
    save_model = Mock(side_effect=persist)

    patches: dict[str, object] = {
        "get_transformer_preprocessing": (preprocessing_getter),
        "build_transformer_parameter_layout": (Mock(side_effect=build_layout)),
        "transformer_parameter_count": (Mock(return_value=TOTAL_PARAMETERS)),
        "initialize_transformer_parameters": (initializer),
        "create_transformer_training_run": (Mock(side_effect=create_run)),
        "create_request_scoped_worker_group": (worker_factory),
        "request_is_disconnected": (AsyncMock(side_effect=disconnected)),
        "presentation_sleep": (AsyncMock(side_effect=delay)),
        "generate_transformer_text": (sample_generator),
        "evaluate_transformer_final_loss": (final_evaluator),
        "build_saved_transformer_model": (model_builder),
        "format_sse": record_sse,
    }

    for name, value in patches.items():
        monkeypatch.setattr(
            train_transformer_route,
            name,
            value,
            raising=False,
        )

    monkeypatch.setattr(
        train_transformer_route,
        "save_transformer_model",
        save_model,
    )

    _patch_run_slot(
        monkeypatch,
        slot,
    )

    return Dependencies(
        order=order,
        failures=active_failures,
        preprocessing=preprocessing,
        slot=slot,
        compute_gate=compute_gate,
        preprocessing_getter=preprocessing_getter,
        initializer=initializer,
        worker_factory=worker_factory,
        sample_generator=sample_generator,
        final_evaluator=final_evaluator,
        model_builder=model_builder,
        save_model=save_model,
        parameters=parameters,
        runs=runs,
        groups=groups,
    )


_HTTP_METHODS: Final[frozenset[str]] = frozenset(
    {
        "delete",
        "get",
        "head",
        "options",
        "patch",
        "post",
        "put",
        "trace",
    }
)


def registered_methods() -> set[tuple[str, str]]:
    """Return registered HTTP path/method pairs from the public OpenAPI schema."""
    paths = app.openapi().get(
        "paths",
        {},
    )

    assert isinstance(
        paths,
        dict,
    )

    result: set[tuple[str, str]] = set()

    for path, operations in paths.items():
        assert isinstance(
            path,
            str,
        )
        assert isinstance(
            operations,
            dict,
        )

        result.update(
            (
                path,
                method.upper(),
            )
            for method in operations
            if method in _HTTP_METHODS
        )

    return result


def post_request(
    *,
    payload: dict[str, object] | None = None,
    raise_server_exceptions: bool = True,
) -> tuple[
    int,
    dict[str, str],
    str,
]:
    """Send one valid request through the public ASGI seam."""
    with TestClient(
        app,
        raise_server_exceptions=raise_server_exceptions,
    ) as client:
        response = client.post(
            "/train-transformer",
            json=(VALID_REQUEST if payload is None else payload),
        )

    return (
        response.status_code,
        dict(response.headers),
        response.text,
    )


def wait_for_group(
    dependencies: Dependencies,
) -> bool:
    """Bound the overlap test while waiting for its controlled group."""
    for _ in range(500):
        if dependencies.groups:
            return True

        threading.Event().wait(0.01)

    return False


def create_request_validation_app(
    probe: RequestValidationProbe,
) -> FastAPI:
    """Create a schema-only FastAPI seam for Step 2 request tests."""
    validation_app = FastAPI()

    @validation_app.post("/train-transformer")
    async def validate_train_transformer_request(
        request: TrainTransformerRequest,
    ) -> dict[str, object]:
        probe.run_slot_acquire()
        probe.preprocessing()
        await probe.worker_factory()
        probe.persistence()

        return request.model_dump(
            by_alias=True,
        )

    return validation_app


def test_strict_sse_parser_records_duplicates_and_trailing_data() -> None:
    parsed = parse_sse(
        ('event: init\nevent: duplicate\ndata: {"value": 1, "value": 2}\n\ntrailing')
    )

    assert len(parsed.events) == 1
    assert parsed.events[0].name == "init"
    assert parsed.events[0].duplicate_fields == ("event",)
    assert parsed.events[0].duplicate_payload_keys == ("value",)
    assert parsed.events[0].payload == {
        "value": 2,
    }
    assert parsed.trailing_data == "trailing"


def test_train_transformer_request_has_exact_public_fields() -> None:
    properties = TrainTransformerRequest.model_json_schema()["properties"]

    assert tuple(properties) == (
        "epochs",
        "temperature",
        "topP",
        "numLayers",
        "maxTokens",
    )


def test_train_transformer_request_defaults_and_aliases() -> None:
    probe = RequestValidationProbe()

    with TestClient(create_request_validation_app(probe)) as client:
        response = client.post(
            "/train-transformer",
            json={},
        )

    assert response.status_code == 200
    assert response.json() == {
        "epochs": 300,
        "temperature": 0.8,
        "topP": 0.9,
        "numLayers": 2,
        "maxTokens": 40,
    }


@pytest.mark.parametrize(
    (
        "payload",
        "expected",
    ),
    [
        (
            {
                "epochs": 50,
                "temperature": 0.1,
                "topP": 0.1,
                "numLayers": 1,
                "maxTokens": 3,
            },
            {
                "epochs": 50,
                "temperature": 0.1,
                "topP": 0.1,
                "numLayers": 1,
                "maxTokens": 3,
            },
        ),
        (
            {
                "epochs": 2_000,
                "temperature": 2.0,
                "topP": 1.0,
                "numLayers": 6,
                "maxTokens": 500,
            },
            {
                "epochs": 2_000,
                "temperature": 2.0,
                "topP": 1.0,
                "numLayers": 6,
                "maxTokens": 500,
            },
        ),
        (
            {
                "epochs": 777,
                "temperature": 1,
                "topP": 1,
                "numLayers": 4,
                "maxTokens": 123,
                "ignored": "value",
            },
            {
                "epochs": 777,
                "temperature": 1.0,
                "topP": 1.0,
                "numLayers": 4,
                "maxTokens": 123,
            },
        ),
    ],
    ids=(
        "minimums",
        "maximums",
        "middle_integer_float_and_extra",
    ),
)
def test_train_transformer_request_accepts_boundaries_and_ignores_extras(
    payload: dict[str, object],
    expected: dict[str, object],
) -> None:
    probe = RequestValidationProbe()

    with TestClient(create_request_validation_app(probe)) as client:
        response = client.post(
            "/train-transformer",
            json=payload,
        )

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {"epochs": "50"},
            id="epochs-numeric-string",
        ),
        pytest.param(
            {"temperature": "0.8"},
            id="temperature-numeric-string",
        ),
        pytest.param(
            {"topP": "0.9"},
            id="top-p-numeric-string",
        ),
        pytest.param(
            {"numLayers": "2"},
            id="layers-numeric-string",
        ),
        pytest.param(
            {"maxTokens": "40"},
            id="tokens-numeric-string",
        ),
        pytest.param(
            {"epochs": True},
            id="epochs-boolean",
        ),
        pytest.param(
            {"temperature": True},
            id="temperature-boolean",
        ),
        pytest.param(
            {"topP": False},
            id="top-p-boolean",
        ),
        pytest.param(
            {"numLayers": True},
            id="layers-boolean",
        ),
        pytest.param(
            {"maxTokens": False},
            id="tokens-boolean",
        ),
        pytest.param(
            {"epochs": 50.5},
            id="epochs-fractional",
        ),
        pytest.param(
            {"numLayers": 2.5},
            id="layers-fractional",
        ),
        pytest.param(
            {"maxTokens": 3.5},
            id="tokens-fractional",
        ),
        pytest.param(
            {"epochs": 49},
            id="epochs-below-minimum",
        ),
        pytest.param(
            {"epochs": 2_001},
            id="epochs-above-maximum",
        ),
        pytest.param(
            {"temperature": 0.099_999},
            id="temperature-below-minimum",
        ),
        pytest.param(
            {"temperature": 2.000_001},
            id="temperature-above-maximum",
        ),
        pytest.param(
            {"topP": 0.099_999},
            id="top-p-below-minimum",
        ),
        pytest.param(
            {"topP": 1.000_001},
            id="top-p-above-maximum",
        ),
        pytest.param(
            {"numLayers": 0},
            id="layers-below-minimum",
        ),
        pytest.param(
            {"numLayers": 7},
            id="layers-above-maximum",
        ),
        pytest.param(
            {"maxTokens": 2},
            id="tokens-below-minimum",
        ),
        pytest.param(
            {"maxTokens": 501},
            id="tokens-above-maximum",
        ),
        pytest.param(
            {"epochs": []},
            id="epochs-list",
        ),
        pytest.param(
            {"temperature": {}},
            id="temperature-object",
        ),
        pytest.param(
            {"topP": []},
            id="top-p-list",
        ),
        pytest.param(
            {"numLayers": {}},
            id="layers-object",
        ),
        pytest.param(
            {"maxTokens": []},
            id="tokens-list",
        ),
        pytest.param(
            [],
            id="array-body",
        ),
        pytest.param(
            "not-an-object",
            id="string-body",
        ),
        pytest.param(
            123,
            id="number-body",
        ),
        pytest.param(
            None,
            id="null-body",
        ),
    ],
)
def test_train_transformer_request_rejects_invalid_json_before_work(
    body: object,
) -> None:
    probe = RequestValidationProbe()

    with TestClient(create_request_validation_app(probe)) as client:
        response = client.post(
            "/train-transformer",
            json=body,
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert "event:" not in response.text
    assert isinstance(
        response.json().get("detail"),
        list,
    )

    probe.assert_no_work_started()


@pytest.mark.parametrize(
    "raw_json",
    [
        pytest.param(
            '{"temperature": NaN}',
            id="temperature-nan",
        ),
        pytest.param(
            '{"temperature": Infinity}',
            id="temperature-positive-infinity",
        ),
        pytest.param(
            '{"temperature": -Infinity}',
            id="temperature-negative-infinity",
        ),
        pytest.param(
            '{"topP": NaN}',
            id="top-p-nan",
        ),
        pytest.param(
            '{"topP": Infinity}',
            id="top-p-positive-infinity",
        ),
        pytest.param(
            '{"topP": -Infinity}',
            id="top-p-negative-infinity",
        ),
    ],
)
def test_train_transformer_request_rejects_non_finite_json_numbers(
    raw_json: str,
) -> None:
    with pytest.raises(ValidationError) as error_info:
        TrainTransformerRequest.model_validate_json(raw_json)

    assert any(error["type"] == "finite_number" for error in error_info.value.errors())


def test_train_transformer_route_is_registered_and_existing_routes_remain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_dependencies(
        monkeypatch,
        tmp_path,
    )

    methods = registered_methods()

    assert {
        (
            "/health",
            "GET",
        ),
        (
            "/simple-chat",
            "POST",
        ),
        (
            "/bpe-tokenize",
            "POST",
        ),
        (
            "/neural-net",
            "POST",
        ),
        (
            "/train-embed",
            "POST",
        ),
        (
            "/train-transformer",
            "POST",
        ),
    } <= methods

    assert post_request()[0] == 200


def test_train_transformer_invalid_status_is_422_before_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/train-transformer",
            json={
                **VALID_REQUEST,
                "epochs": "50",
            },
        )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert not response.headers["content-type"].startswith("text/event-stream")
    assert "event:" not in response.text
    assert dependencies.slot.acquire_count == 0
    assert dependencies.preprocessing_getter.call_count == 0
    assert dependencies.initializer.call_count == 0
    assert dependencies.worker_factory.await_count == 0
    assert dependencies.save_model.call_count == 0


def test_train_transformer_headers_and_init_precede_later_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )

    status, headers, body = post_request()

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert parsed.events[0] == ParsedEvent(
        name="init",
        payload=EXPECTED_INIT,
        duplicate_fields=(),
        duplicate_payload_keys=(),
        unexpected_fields=(),
    )
    assert tuple(event.name for event in parsed.events) == (
        "init",
        *("epoch" for _ in range(51)),
        "done",
    )

    expected_order = [
        "validate",
        "reserve",
        "preprocess/init",
        "init yield",
        "initialize",
        "worker startup",
    ]

    for _ in range(51):
        expected_order.extend(
            (
                "epoch compute",
                "Adam commit",
                "sample",
                "epoch yield",
                "delay",
            )
        )

    expected_order.extend(
        (
            "final evaluation",
            "worker cleanup success",
            "model build",
            "persist",
            "done",
            "slot release",
        )
    )

    assert dependencies.order.snapshot() == tuple(expected_order)
    assert dependencies.initializer.call_count == 1
    assert dependencies.worker_factory.await_count == 1
    assert dependencies.sample_generator.call_count == 51
    assert dependencies.final_evaluator.call_count == 1
    assert dependencies.model_builder.call_count == 1
    assert dependencies.save_model.call_count == 1
    assert dependencies.groups[0].cleanup_attempts == 2
    assert dependencies.slot.locked() is False


def test_train_transformer_final_loss_uses_post_adam_run_and_shared_cancellation_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )

    status, _headers, body = post_request()

    assert status == 200

    parsed = parse_sse(body)

    assert parsed.events[-2].name == "epoch"
    assert parsed.events[-2].payload["epoch"] == 50
    assert parsed.events[-1].name == "done"
    assert parsed.events[-1].payload["finalLoss"] == FINAL_LOSS

    final_evaluation_call = dependencies.final_evaluator.call_args

    assert final_evaluation_call is not None
    assert final_evaluation_call.args == (
        cast(
            TransformerTrainingRun,
            dependencies.runs[0],
        ),
        dependencies.preprocessing,
    )

    sample_cancellation_event = dependencies.sample_generator.call_args_list[-1].kwargs[
        "cancellation_event"
    ]
    final_cancellation_event = final_evaluation_call.kwargs["cancellation_event"]

    assert final_cancellation_event is sample_cancellation_event
    assert dependencies.runs[0].is_complete
    assert dependencies.runs[0].last_completed_epoch == 50
    assert dependencies.runs[0].last_completed_loss != FINAL_LOSS

    order = dependencies.order.snapshot()

    assert order.index("final evaluation") > max(
        index for index, stage in enumerate(order) if stage == "Adam commit"
    )
    assert order.index("final evaluation") < order.index("worker cleanup success")
    assert order.index("worker cleanup success") < order.index("model build")
    assert order.index("model build") < order.index("persist")
    assert order.index("persist") < order.index("done")
    assert dependencies.model_builder.call_count == 1
    assert dependencies.save_model.call_count == 1
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    (
        "failure_stage",
        "cooperative_shutdown_completed",
        "terminate_required",
        "kill_required",
        "secondary_failures",
        "primary_failure_code",
    ),
    [
        pytest.param(
            "cleanup_terminate",
            False,
            True,
            False,
            (),
            None,
            id="terminate-required",
        ),
        pytest.param(
            "cleanup_kill",
            False,
            True,
            True,
            (RequestScopedWorkerGroupCleanupFailureCode.KILL,),
            None,
            id="kill-required",
        ),
        pytest.param(
            "cleanup_secondary",
            True,
            False,
            False,
            (RequestScopedWorkerGroupCleanupFailureCode.PIPE_CLOSE,),
            None,
            id="secondary-failure",
        ),
        pytest.param(
            "cleanup_primary",
            True,
            False,
            False,
            (),
            RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
            id="primary-worker-failure",
        ),
    ],
)
def test_train_transformer_completion_gate_blocks_unsuccessful_worker_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
    cooperative_shutdown_completed: bool,
    terminate_required: bool,
    kill_required: bool,
    secondary_failures: tuple[
        RequestScopedWorkerGroupCleanupFailureCode,
        ...,
    ],
    primary_failure_code: RequestScopedWorkerGroupFailureCode | None,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
        failures={
            failure_stage,
        },
    )
    cancellation_event = threading.Event()
    cleanup_log = Mock()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )
    monkeypatch.setattr(
        train_transformer_route.logger,
        "error",
        cleanup_log,
    )

    status, headers, body = post_request(raise_server_exceptions=False)

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert parsed.events[-1].name == "epoch"
    assert parsed.events[-1].payload["epoch"] == 50
    assert all(event.name != "done" for event in parsed.events)

    group = dependencies.groups[0]
    cleanup_report = group.cleanup_report

    assert cleanup_report is not None
    assert cleanup_report.cooperative_shutdown_completed is cooperative_shutdown_completed
    assert cleanup_report.terminate_required is terminate_required
    assert cleanup_report.kill_required is kill_required
    assert cleanup_report.secondary_failures == secondary_failures
    assert group.primary_failure_code is primary_failure_code
    assert group.successful is False
    assert group.cleanup_attempts == 2

    assert cancellation_event.is_set()
    assert dependencies.final_evaluator.call_count == 1
    assert dependencies.model_builder.call_count == 0
    assert dependencies.save_model.call_count == 0
    assert cleanup_log.call_count >= 1
    assert dependencies.slot.locked() is False

    logged_values = " ".join(
        str(value) for call in cleanup_log.call_args_list for value in call.args
    )

    for marker in PRIVATE_MARKERS:
        assert marker.lower() not in logged_values.lower()


@pytest.mark.asyncio
async def test_train_transformer_cleanup_cancellation_finishes_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
        failures={
            "cleanup_blocked",
        },
    )
    cancellation_event = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )

    assert dependencies.slot.acquire(blocking=False)

    layout = TransformerParameterLayout(
        num_layers=2,
        vocabulary_size=3,
        records=(),
        total_float_count=TOTAL_PARAMETERS,
        total_byte_count=TOTAL_PARAMETERS * 4,
    )

    stream = train_transformer_route.stream_transformer_training(
        request=cast(
            Any,
            object(),
        ),
        init_payload=dict(EXPECTED_INIT),
        preprocessing=dependencies.preprocessing,
        layout=layout,
        epochs=50,
        temperature=0.8,
        top_p=0.9,
        num_layers=2,
        max_tokens=3,
    )

    streamed_events = [await anext(stream) for _ in range(52)]

    assert parse_sse(streamed_events[0]).events[0].name == "init"
    assert parse_sse(streamed_events[-1]).events[0].payload["epoch"] == 50

    completion_task = asyncio.create_task(anext(stream))

    assert await asyncio.to_thread(
        dependencies.groups[0].cleanup_started.wait,
        5.0,
    )

    completion_task.cancel()
    dependencies.groups[0].cleanup_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await completion_task

    group = dependencies.groups[0]

    assert cancellation_event.is_set()
    assert group.cleanup_report is not None
    assert group.cleanup_report.successful
    assert group.successful
    assert group.cleanup_attempts == 2
    assert dependencies.final_evaluator.call_count == 1
    assert dependencies.model_builder.call_count == 0
    assert dependencies.save_model.call_count == 0
    assert dependencies.slot.locked() is False

    order = dependencies.order.snapshot()

    assert order.index("final evaluation") < order.index("worker cleanup success")
    assert order.index("worker cleanup success") < order.index("slot release")


@pytest.mark.parametrize(
    (
        "request_payload",
        "epochs",
    ),
    [
        pytest.param(
            {
                **VALID_REQUEST,
                "epochs": 50,
            },
            50,
            id="minimum",
        ),
        pytest.param(
            {
                "temperature": 0.8,
                "topP": 0.9,
                "numLayers": 2,
                "maxTokens": 3,
            },
            300,
            id="default",
        ),
        pytest.param(
            {
                **VALID_REQUEST,
                "epochs": 2_000,
            },
            2_000,
            id="maximum-controlled",
        ),
        pytest.param(
            {
                **VALID_REQUEST,
                "epochs": 333,
            },
            333,
            id="non-divisible",
        ),
    ],
)
def test_train_transformer_epoch_report_schedule_is_inclusive_sequential_and_delayed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_payload: dict[str, object],
    epochs: int,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )

    status, _headers, body = post_request(
        payload=request_payload,
    )

    assert status == 200

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert parsed.events[0].name == "init"

    report_step = max(
        1,
        epochs // 50,
    )
    expected_report_epochs = tuple(
        epoch for epoch in range(epochs + 1) if (epoch % report_step == 0 or epoch == epochs)
    )

    epoch_events = parsed.events[1:-1]

    assert tuple(event.name for event in epoch_events) == tuple(
        "epoch" for _ in expected_report_epochs
    )
    assert tuple(event.payload["epoch"] for event in epoch_events) == expected_report_epochs

    for event in epoch_events:
        assert set(event.payload) == {
            "epoch",
            "loss",
            "sample",
        }

    assert expected_report_epochs[0] == 0
    assert expected_report_epochs[-1] == epochs
    assert len(set(expected_report_epochs)) == len(expected_report_epochs)

    assert dependencies.groups[0].compute_epochs == list(range(epochs + 1))
    assert dependencies.runs[0].advance_epochs == list(range(epochs + 1))
    assert dependencies.runs[0].result_counts == [4] * (epochs + 1)
    assert dependencies.sample_generator.call_count == len(expected_report_epochs)
    assert dependencies.order.snapshot().count("delay") == len(expected_report_epochs)
    assert dependencies.save_model.call_args.kwargs["epochs"] == epochs

    assert parsed.events[-1].name == "done"
    assert (
        tuple(sample["epoch"] for sample in parsed.events[-1].payload["samples"])
        == expected_report_epochs
    )
    assert dependencies.slot.locked() is False


def test_train_transformer_generation_controls_only_reach_sample_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    observed_controls: list[tuple[float, float, int]] = []
    observed_cancellation_events: list[object] = []

    def sample_with_custom_controls(
        parameters_value: InitializedTransformerParameters,
        preprocessing_value: TransformerPreprocessingSnapshot,
        *,
        epoch: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        cancellation_event: object,
    ) -> GeneratedTextSample:
        dependencies.order.add("sample")

        assert parameters_value is dependencies.runs[-1].parameters
        assert preprocessing_value is dependencies.preprocessing

        observed_controls.append(
            (
                temperature,
                top_p,
                max_tokens,
            )
        )
        observed_cancellation_events.append(cancellation_event)

        return GeneratedTextSample(
            epoch=epoch,
            text=f"controlled sample {epoch}",
        )

    dependencies.sample_generator.side_effect = sample_with_custom_controls

    status, _headers, body = post_request(
        payload={
            "epochs": 50,
            "temperature": 1.25,
            "topP": 0.75,
            "numLayers": 2,
            "maxTokens": 17,
        },
    )

    assert status == 200

    parsed = parse_sse(body)
    epoch_events = parsed.events[1:-1]

    assert len(epoch_events) == 51
    assert (
        observed_controls
        == [
            (
                1.25,
                0.75,
                17,
            )
        ]
        * 51
    )
    assert len({id(event) for event in observed_cancellation_events}) == 1

    initializer_call = dependencies.initializer.call_args

    assert initializer_call is not None
    assert len(initializer_call.args) == 2
    assert initializer_call.args[0].num_layers == 2

    worker_call = dependencies.worker_factory.await_args

    assert worker_call is not None
    assert worker_call.args[0] == 2
    assert worker_call.args[1] is dependencies.runs[0].weights
    assert worker_call.args[2] == dependencies.preprocessing.training_sequences
    assert worker_call.args[3] == dependencies.runs[0].logical_training_shards
    assert set(worker_call.kwargs) == {
        "poll_observer",
    }
    assert callable(worker_call.kwargs["poll_observer"])

    assert tuple(event.payload["loss"] for event in epoch_events) == tuple(
        round(
            1.0 / float(epoch + 1),
            6,
        )
        for epoch in range(51)
    )
    assert parsed.events[-1].name == "done"


def test_train_transformer_disconnect_before_worker_start_is_quiet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    cancellation_event = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )
    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        AsyncMock(return_value=True),
    )

    status, headers, body = post_request(raise_server_exceptions=False)

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert tuple(event.name for event in parsed.events) == ("init",)

    assert cancellation_event.is_set()
    assert dependencies.initializer.call_count == 0
    assert dependencies.worker_factory.await_count == 0
    assert dependencies.sample_generator.call_count == 0
    assert dependencies.save_model.call_count == 0
    assert dependencies.slot.locked() is False

    assert dependencies.order.snapshot() == (
        "validate",
        "reserve",
        "preprocess/init",
        "init yield",
        "slot release",
    )


def test_train_transformer_disconnect_during_sample_drains_helper_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    cancellation_event = threading.Event()
    helper_started = threading.Event()
    helper_finished = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )

    async def disconnected_when_helper_starts(
        request: object,
    ) -> bool:
        del request
        return helper_started.is_set()

    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        AsyncMock(side_effect=(disconnected_when_helper_starts)),
    )

    def sample_until_disconnected(
        parameters_value: InitializedTransformerParameters,
        preprocessing_value: TransformerPreprocessingSnapshot,
        *,
        epoch: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        cancellation_event: object,
    ) -> GeneratedTextSample:
        del (
            parameters_value,
            preprocessing_value,
            temperature,
            top_p,
            max_tokens,
        )

        dependencies.order.add("sample")
        helper_started.set()

        assert cancellation_event is not None
        assert getattr(
            cancellation_event,
            "wait",
        )(5.0)

        dependencies.order.add("helper drained")
        helper_finished.set()

        return GeneratedTextSample(
            epoch=epoch,
            text=f"controlled sample {epoch}",
        )

    dependencies.sample_generator.side_effect = sample_until_disconnected

    status, headers, body = post_request(raise_server_exceptions=False)

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert tuple(event.name for event in parsed.events) == ("init",)

    assert cancellation_event.is_set()
    assert helper_started.is_set()
    assert helper_finished.is_set()
    assert dependencies.sample_generator.call_count == 1
    assert dependencies.groups[0].cleanup_attempts == 1
    assert dependencies.save_model.call_count == 0
    assert dependencies.slot.locked() is False

    order = dependencies.order.snapshot()

    assert order.index("helper drained") < order.index("worker cleanup success")
    assert order.index("worker cleanup success") < order.index("slot release")
    assert "epoch yield" not in order
    assert "delay" not in order


def test_train_transformer_sample_timeout_drains_helper_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    cancellation_event = threading.Event()
    helper_started = threading.Event()
    helper_finished = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )
    monkeypatch.setattr(
        train_transformer_route,
        "_TRANSFORMER_HELPER_TIMEOUT_SECONDS",
        0.01,
    )

    def sample_until_timeout_cancellation(
        parameters_value: InitializedTransformerParameters,
        preprocessing_value: TransformerPreprocessingSnapshot,
        *,
        epoch: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        cancellation_event: object,
    ) -> GeneratedTextSample:
        del (
            parameters_value,
            preprocessing_value,
            temperature,
            top_p,
            max_tokens,
        )

        dependencies.order.add("sample")
        helper_started.set()

        assert getattr(
            cancellation_event,
            "wait",
        )(5.0)

        dependencies.order.add("helper drained")
        helper_finished.set()

        return GeneratedTextSample(
            epoch=epoch,
            text=f"controlled sample {epoch}",
        )

    dependencies.sample_generator.side_effect = sample_until_timeout_cancellation

    status, headers, body = post_request(raise_server_exceptions=False)

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert tuple(event.name for event in parsed.events) == ("init",)

    assert cancellation_event.is_set()
    assert helper_started.is_set()
    assert helper_finished.is_set()
    assert dependencies.sample_generator.call_count == 1
    assert dependencies.groups[0].cleanup_attempts == 1
    assert dependencies.save_model.call_count == 0
    assert dependencies.slot.locked() is False

    order = dependencies.order.snapshot()

    assert order.index("helper drained") < order.index("worker cleanup success")
    assert order.index("worker cleanup success") < order.index("slot release")
    assert "epoch yield" not in order
    assert "delay" not in order


@pytest.mark.asyncio
async def test_train_transformer_task_cancellation_drains_helper_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    cancellation_event = threading.Event()
    helper_started = threading.Event()
    helper_finished = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )

    def sample_until_task_cancellation(
        parameters_value: InitializedTransformerParameters,
        preprocessing_value: TransformerPreprocessingSnapshot,
        *,
        epoch: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        cancellation_event: object,
    ) -> GeneratedTextSample:
        del (
            parameters_value,
            preprocessing_value,
            temperature,
            top_p,
            max_tokens,
        )

        dependencies.order.add("sample")
        helper_started.set()

        assert getattr(
            cancellation_event,
            "wait",
        )(5.0)

        dependencies.order.add("helper drained")
        helper_finished.set()

        return GeneratedTextSample(
            epoch=epoch,
            text=f"controlled sample {epoch}",
        )

    dependencies.sample_generator.side_effect = sample_until_task_cancellation

    assert dependencies.slot.acquire(blocking=False)

    layout = TransformerParameterLayout(
        num_layers=2,
        vocabulary_size=3,
        records=(),
        total_float_count=TOTAL_PARAMETERS,
        total_byte_count=TOTAL_PARAMETERS * 4,
    )

    stream = train_transformer_route.stream_transformer_training(
        request=cast(
            Any,
            object(),
        ),
        init_payload=dict(EXPECTED_INIT),
        preprocessing=dependencies.preprocessing,
        layout=layout,
        epochs=50,
        temperature=0.8,
        top_p=0.9,
        num_layers=2,
        max_tokens=3,
    )

    init_event = await anext(stream)

    assert parse_sse(init_event).events[0].name == "init"

    next_event_task = asyncio.create_task(anext(stream))

    assert await asyncio.to_thread(
        helper_started.wait,
        5.0,
    )

    next_event_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await next_event_task

    assert cancellation_event.is_set()
    assert helper_finished.is_set()
    assert dependencies.groups[0].cleanup_attempts == 1
    assert dependencies.slot.locked() is False

    order = dependencies.order.snapshot()

    assert order.index("helper drained") < order.index("worker cleanup success")
    assert order.index("worker cleanup success") < order.index("slot release")


def test_train_transformer_sequential_requests_use_fresh_mutable_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = install_dependencies(
        monkeypatch,
        tmp_path,
    )

    first_status, _headers, first_body = post_request()

    assert first_status == 200

    first_events = parse_sse(first_body).events

    assert first_events[-1].name == "done"
    assert first_events[-1].payload["samples"][-1]["epoch"] == 50

    second = install_dependencies(
        monkeypatch,
        tmp_path,
    )

    second_status, _headers, second_body = post_request()

    assert second_status == 200

    second_events = parse_sse(second_body).events

    assert second_events[-1].name == "done"
    assert second_events[-1].payload["samples"][-1]["epoch"] == 50

    assert first.parameters[0] is not second.parameters[0]
    assert not np.shares_memory(
        first.parameters[0].storage,
        second.parameters[0].storage,
    )
    assert first.runs[0] is not second.runs[0]
    assert first.groups[0] is not second.groups[0]

    first_initializer_call = first.initializer.call_args
    second_initializer_call = second.initializer.call_args

    assert first_initializer_call is not None
    assert second_initializer_call is not None
    assert first_initializer_call.args[1] is not second_initializer_call.args[1]
    assert (
        getattr(
            first_initializer_call.args[1],
            "state",
        )
        == 42
    )
    assert (
        getattr(
            second_initializer_call.args[1],
            "state",
        )
        == 42
    )

    first_sample_call = first.sample_generator.call_args
    second_sample_call = second.sample_generator.call_args

    assert first_sample_call is not None
    assert second_sample_call is not None
    assert (
        first_sample_call.kwargs["cancellation_event"]
        is not second_sample_call.kwargs["cancellation_event"]
    )

    assert first.save_model.call_count == 1
    assert second.save_model.call_count == 1
    assert first.slot.locked() is False
    assert second.slot.locked() is False


def test_train_transformer_init_stream_failure_is_quiet_and_releases_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    original_format_sse = train_transformer_route.format_sse

    monkeypatch.setattr(
        train_transformer_route,
        "format_sse",
        Mock(side_effect=SentinelRouteFailure),
    )

    status, headers, body = post_request(raise_server_exceptions=False)

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert body == ""

    for marker in PRIVATE_MARKERS:
        assert marker.lower() not in body.lower()

    assert dependencies.initializer.call_count == 0
    assert dependencies.worker_factory.await_count == 0
    assert dependencies.save_model.call_count == 0
    assert dependencies.slot.locked() is False

    monkeypatch.setattr(
        train_transformer_route,
        "format_sse",
        original_format_sse,
    )

    retry_status, _headers, retry_body = post_request(raise_server_exceptions=False)

    assert retry_status == 200
    assert tuple(event.name for event in parse_sse(retry_body).events) == (
        "init",
        *("epoch" for _ in range(51)),
        "done",
    )


def test_train_transformer_headers_payloads_and_lifecycle_order_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )

    status, headers, body = post_request()

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"

    parsed = parse_sse(body)

    assert_exact_sse(parsed)

    assert parsed.events[0] == ParsedEvent(
        name="init",
        payload=EXPECTED_INIT,
        duplicate_fields=(),
        duplicate_payload_keys=(),
        unexpected_fields=(),
    )

    assert tuple(event.name for event in parsed.events) == (
        "init",
        *("epoch" for _ in range(51)),
        "done",
    )

    expected_samples: list[dict[str, object]] = []

    for epoch, event in enumerate(parsed.events[1:-1]):
        text = f"controlled sample {epoch}"

        expected_samples.append(
            {
                "epoch": epoch,
                "text": text,
            }
        )

        assert event.payload == {
            "epoch": epoch,
            "loss": round(
                1.0 / float(epoch + 1),
                6,
            ),
            "sample": text,
        }
        assert set(event.payload) == {
            "epoch",
            "loss",
            "sample",
        }

    assert parsed.events[-1].payload == {
        "architecture": ("Decoder-Only Transformer (2 layers, 32d, 2h, 128ff)"),
        "finalLoss": FINAL_LOSS,
        "samples": expected_samples,
    }
    assert set(parsed.events[-1].payload) == {
        "architecture",
        "finalLoss",
        "samples",
    }

    expected_order = [
        "validate",
        "reserve",
        "preprocess/init",
        "init yield",
        "initialize",
        "worker startup",
    ]

    for _ in range(51):
        expected_order.extend(
            (
                "epoch compute",
                "Adam commit",
                "sample",
                "epoch yield",
                "delay",
            )
        )

    expected_order.extend(
        (
            "final evaluation",
            "worker cleanup success",
            "model build",
            "persist",
            "done",
            "slot release",
        )
    )

    assert dependencies.order.snapshot() == tuple(expected_order)
    assert dependencies.runs[0].advance_epochs == list(range(51))
    assert dependencies.runs[0].result_counts == [4] * 51
    assert dependencies.groups[0].compute_epochs == list(range(51))
    assert dependencies.slot.locked() is False


def test_train_transformer_overlap_is_429_with_no_second_run_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert (
        "/train-transformer",
        "POST",
    ) in registered_methods()

    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    init_started = threading.Event()
    init_gate = threading.Event()
    first_result: dict[str, object] = {}
    original_format_sse = train_transformer_route.format_sse

    def block_first_init(
        event: str,
        data: dict[str, object],
    ) -> str:
        if event == "init":
            init_started.set()
            init_gate.wait()

        return original_format_sse(
            event,
            data,
        )

    monkeypatch.setattr(
        train_transformer_route,
        "format_sse",
        block_first_init,
    )

    def first_request() -> None:
        first_result["response"] = post_request(raise_server_exceptions=False)

    thread = threading.Thread(
        target=first_request,
        daemon=True,
    )
    thread.start()

    try:
        assert init_started.wait(timeout=5.0)
        assert dependencies.slot.locked()

        status, headers, body = post_request(raise_server_exceptions=False)

        assert status == 429
        assert not headers.get(
            "content-type",
            "",
        ).startswith("text/event-stream")
        assert "event:" not in body
        assert dependencies.preprocessing_getter.call_count == 1
        assert dependencies.initializer.call_count == 0
        assert dependencies.worker_factory.await_count == 0
        assert dependencies.save_model.call_count == 0
        assert thread.is_alive()
    finally:
        init_gate.set()
        thread.join(timeout=10.0)

    assert not thread.is_alive()

    first_status, _headers, first_body = cast(
        tuple[
            int,
            dict[str, str],
            str,
        ],
        first_result["response"],
    )

    assert first_status == 200
    assert tuple(event.name for event in parse_sse(first_body).events) == (
        "init",
        *("epoch" for _ in range(51)),
        "done",
    )
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    "stage",
    [
        "preprocess",
        "layout",
    ],
)
def test_train_transformer_pre_stream_failure_is_private_500_and_releases_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
        failures={stage},
    )

    status, headers, body = post_request(raise_server_exceptions=False)

    assert status == 500
    assert not headers.get(
        "content-type",
        "",
    ).startswith("text/event-stream")
    assert "event:" not in body

    for marker in PRIVATE_MARKERS:
        assert marker.lower() not in body.lower()

    assert dependencies.initializer.call_count == 0
    assert dependencies.worker_factory.await_count == 0
    assert dependencies.model_builder.call_count == 0
    assert dependencies.save_model.call_count == 0
    assert "init yield" not in dependencies.order.snapshot()
    assert dependencies.slot.locked() is False

    dependencies.failures.clear()

    retry_status, _headers, retry_body = post_request(raise_server_exceptions=False)

    assert retry_status == 200

    retry_events = parse_sse(retry_body).events

    assert retry_events[-2].name == "epoch"
    assert retry_events[-2].payload["epoch"] == 50
    assert retry_events[-1].name == "done"
    assert dependencies.model_builder.call_count == 1
    assert dependencies.save_model.call_count == 1
    assert dependencies.slot.locked() is False


@pytest.mark.parametrize(
    "stage",
    [
        "initialize",
        "worker_startup",
        "epoch",
        "sample",
        "final_evaluation",
        "cleanup",
        "cleanup_unsuccessful",
        "model_build",
        "persist",
    ],
)
def test_train_transformer_post_init_failure_is_quiet_private_and_releases_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
        failures={stage},
    )

    status, headers, body = post_request(raise_server_exceptions=False)

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    for marker in PRIVATE_MARKERS:
        assert marker.lower() not in body.lower()

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert parsed.events[0].name == "init"
    assert all(
        event.name
        not in {
            "done",
            "error",
        }
        for event in parsed.events
    )
    assert dependencies.slot.locked() is False

    model_builder_calls_before_retry = dependencies.model_builder.call_count
    save_model_calls_before_retry = dependencies.save_model.call_count

    if stage == "model_build":
        assert model_builder_calls_before_retry == 1
        assert save_model_calls_before_retry == 0
    elif stage == "persist":
        assert model_builder_calls_before_retry == 1
        assert save_model_calls_before_retry == 1
    else:
        assert model_builder_calls_before_retry == 0
        assert save_model_calls_before_retry == 0

    dependencies.failures.clear()

    retry_status, _headers, retry_body = post_request(raise_server_exceptions=False)

    assert retry_status == 200

    retry_events = parse_sse(retry_body).events

    assert retry_events[-2].name == "epoch"
    assert retry_events[-2].payload["epoch"] == 50
    assert retry_events[-1].name == "done"

    assert dependencies.model_builder.call_count == model_builder_calls_before_retry + 1
    assert dependencies.save_model.call_count == save_model_calls_before_retry + 1
    assert dependencies.slot.locked() is False


def test_train_transformer_disconnect_after_model_build_prevents_persistence_and_done(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    cancellation_event = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )

    async def disconnected_after_model_build(
        request: object,
    ) -> bool:
        del request
        return "model build" in dependencies.order.snapshot()

    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        AsyncMock(side_effect=disconnected_after_model_build),
    )

    status, headers, body = post_request(
        raise_server_exceptions=False,
    )

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert parsed.events[-1].name == "epoch"
    assert parsed.events[-1].payload["epoch"] == 50
    assert all(event.name != "done" for event in parsed.events)

    assert cancellation_event.is_set()
    assert dependencies.model_builder.call_count == 1
    assert dependencies.save_model.call_count == 0
    assert dependencies.slot.locked() is False

    order = dependencies.order.snapshot()

    assert order.index("worker cleanup success") < order.index("model build")
    assert order.index("model build") < order.index("slot release")


def test_train_transformer_disconnect_during_persistence_drains_before_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    cancellation_event = threading.Event()
    persistence_started = threading.Event()
    persistence_finished = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )

    async def disconnected_when_persistence_starts(
        request: object,
    ) -> bool:
        del request
        return persistence_started.is_set()

    monkeypatch.setattr(
        train_transformer_route,
        "request_is_disconnected",
        AsyncMock(side_effect=(disconnected_when_persistence_starts)),
    )

    def persist_until_disconnect(
        model: SavedTransformerModel,
        *,
        epochs: int,
        model_directory: Path | None = None,
    ) -> Path:
        del model_directory

        assert model is CONTROLLED_MODEL
        assert epochs == 50

        dependencies.order.add("persist")
        persistence_started.set()

        assert cancellation_event.wait(5.0)

        dependencies.order.add("persistence drained")
        persistence_finished.set()

        return tmp_path / "controlled-transformer.json"

    dependencies.save_model.side_effect = persist_until_disconnect

    status, headers, body = post_request(
        raise_server_exceptions=False,
    )

    assert status == 200
    assert headers["content-type"].startswith("text/event-stream")

    parsed = parse_sse(body)

    assert_exact_sse(parsed)
    assert parsed.events[-1].name == "epoch"
    assert parsed.events[-1].payload["epoch"] == 50
    assert all(event.name != "done" for event in parsed.events)

    assert cancellation_event.is_set()
    assert persistence_started.is_set()
    assert persistence_finished.is_set()
    assert dependencies.model_builder.call_count == 1
    assert dependencies.save_model.call_count == 1
    assert dependencies.slot.locked() is False

    order = dependencies.order.snapshot()

    assert order.index("persist") < order.index("persistence drained")
    assert order.index("persistence drained") < order.index("slot release")


@pytest.mark.asyncio
async def test_train_transformer_task_cancellation_during_persistence_drains_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dependencies = install_dependencies(
        monkeypatch,
        tmp_path,
    )
    cancellation_event = threading.Event()
    persistence_started = threading.Event()
    persistence_finished = threading.Event()

    monkeypatch.setattr(
        train_transformer_route,
        "Event",
        Mock(return_value=cancellation_event),
    )

    def persist_until_cancelled(
        model: SavedTransformerModel,
        *,
        epochs: int,
        model_directory: Path | None = None,
    ) -> Path:
        del model_directory

        assert model is CONTROLLED_MODEL
        assert epochs == 50

        dependencies.order.add("persist")
        persistence_started.set()

        assert cancellation_event.wait(5.0)

        dependencies.order.add("persistence drained")
        persistence_finished.set()

        return tmp_path / "controlled-transformer.json"

    dependencies.save_model.side_effect = persist_until_cancelled

    assert dependencies.slot.acquire(blocking=False)

    layout = TransformerParameterLayout(
        num_layers=2,
        vocabulary_size=3,
        records=(),
        total_float_count=TOTAL_PARAMETERS,
        total_byte_count=TOTAL_PARAMETERS * 4,
    )

    stream = train_transformer_route.stream_transformer_training(
        request=cast(
            Any,
            object(),
        ),
        init_payload=dict(EXPECTED_INIT),
        preprocessing=dependencies.preprocessing,
        layout=layout,
        epochs=50,
        temperature=0.8,
        top_p=0.9,
        num_layers=2,
        max_tokens=3,
    )

    streamed_events = [await anext(stream) for _ in range(52)]

    assert parse_sse(streamed_events[0]).events[0].name == "init"
    assert parse_sse(streamed_events[-1]).events[0].payload["epoch"] == 50

    completion_task = asyncio.create_task(anext(stream))

    assert await asyncio.to_thread(
        persistence_started.wait,
        5.0,
    )

    completion_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await completion_task

    assert cancellation_event.is_set()
    assert persistence_finished.is_set()
    assert dependencies.model_builder.call_count == 1
    assert dependencies.save_model.call_count == 1
    assert dependencies.slot.locked() is False

    order = dependencies.order.snapshot()

    assert order.index("persist") < order.index("persistence drained")
    assert order.index("persistence drained") < order.index("slot release")
