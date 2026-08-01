# src/how_llms_work/ml/transformer_worker.py
"""Spawn-safe Transformer worker protocol and shared-memory execution boundary."""

from __future__ import annotations

import asyncio
import gc
import math
import multiprocessing as mp
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection, _ConnectionBase
from multiprocessing.connection import wait as wait_for_connections
from multiprocessing.context import BaseContext, SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from threading import Event
from time import monotonic
from typing import Any, Final, NoReturn, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from how_llms_work.ml.transformer import (
    LOGICAL_TRAINING_SHARD_COUNT,
    TRANSFORMER_MAX_LAYER_COUNT,
    TRANSFORMER_MIN_LAYER_COUNT,
    TRANSFORMER_SEQUENCE_LENGTH,
    LogicalTrainingShard,
    LogicalTrainingShardResult,
    TransformerParameterLayout,
    TransformerParameterViews,
    TransformerTrainingSequence,
    build_logical_training_shards,
    build_transformer_parameter_layout,
    build_transformer_parameter_views,
    calculate_logical_training_shard,
    create_transformer_gradient_buffer,
)

__all__ = [
    "WORKER_PROTOCOL_VERSION",
    "WorkerNumericDType",
    "WorkerState",
    "RequestScopedWorkerGroupState",
    "RequestScopedWorkerGroupFailureCode",
    "RequestScopedWorkerGroupCleanupFailureCode",
    "RequestScopedWorkerGroupCleanupReport",
    "RequestScopedWorkerGroupError",
    "calculate_actual_worker_count",
    "build_worker_shard_assignments",
    "RequestScopedWorkerGroup",
    "create_request_scoped_worker_group",
    "WorkerFailurePhase",
    "WorkerFailureCode",
    "WorkerStartupConfig",
    "ReadyMessage",
    "ComputeMessage",
    "ResultMessage",
    "FailureMessage",
    "StopMessage",
    "StoppedMessage",
    "WorkerProtocolValidationError",
    "validate_worker_startup_config",
    "validate_ready_message",
    "validate_compute_message",
    "validate_result_message",
    "validate_failure_message",
    "validate_stop_message",
    "validate_stopped_message",
    "run_transformer_worker",
]

WORKER_PROTOCOL_VERSION: Final = 1


_Float32Array: TypeAlias = npt.NDArray[np.float32]
_SequencePayload: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]
_ShardPayload: TypeAlias = tuple[int, int, int]
_WorkerShardAssignments: TypeAlias = tuple[tuple[int, ...], ...]

type _WorkerConnection = _ConnectionBase[Any, Any]

_WorkerWaitable: TypeAlias = _WorkerConnection | int
_WorkerProcessTarget: TypeAlias = Callable[[Connection, int, object], None]
_WorkerConnectionWaiter: TypeAlias = Callable[
    [tuple[_WorkerWaitable, ...], float | None],
    list[_WorkerWaitable],
]
_WorkerMonotonicClock: TypeAlias = Callable[[], float]
_WorkerProcessFactory: TypeAlias = Callable[
    [
        BaseContext,
        _WorkerProcessTarget,
        _WorkerConnection,
        int,
        object,
    ],
    BaseProcess,
]

_WORKER_GROUP_STARTUP_TIMEOUT_SECONDS: Final = 30.0
_WORKER_GROUP_EPOCH_TIMEOUT_SECONDS: Final = 300.0
_WORKER_GROUP_POLL_TIMEOUT_SECONDS: Final = 0.1
_WORKER_GROUP_INTERIM_SHUTDOWN_TIMEOUT_SECONDS: Final = 2.0


class WorkerNumericDType(StrEnum):
    """Closed shared-array dtype vocabulary for protocol version 1."""

    FLOAT32 = "float32"


class WorkerState(StrEnum):
    """Closed worker lifecycle states."""

    STARTING = "starting"
    READY = "ready"
    COMPUTING = "computing"
    STOPPING = "stopping"
    STOPPED = "stopped"


class WorkerFailurePhase(StrEnum):
    """Sanitized phase in which a controlled worker failure occurred."""

    STARTUP = "startup"
    COMMAND = "command"
    COMPUTE = "compute"
    SHUTDOWN = "shutdown"


class WorkerFailureCode(StrEnum):
    """Closed sanitized failure vocabulary for protocol version 1."""

    INVALID_PROTOCOL = "invalid_protocol"
    INVALID_LAYOUT = "invalid_layout"
    SHARED_MEMORY = "shared_memory"
    INVALID_OWNERSHIP = "invalid_ownership"
    INVALID_STATE = "invalid_state"
    NUMERICAL_FAILURE = "numerical_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    CLEANUP_FAILURE = "cleanup_failure"
    INTERNAL_FAILURE = "internal_failure"


@dataclass(frozen=True, slots=True)
class WorkerStartupConfig:
    """Primitive-only startup data supplied to one spawned worker."""

    protocol_version: int
    worker_index: int
    num_layers: int
    dtype_marker: WorkerNumericDType
    canonical_float_count: int
    sequence_count: int
    weight_shared_memory_name: str
    assigned_shards: tuple[_ShardPayload, ...]
    gradient_shared_memory_names: tuple[str, ...]
    training_sequences: tuple[_SequencePayload, ...]


@dataclass(frozen=True, slots=True)
class ReadyMessage:
    """Worker startup commit marker sent after complete attachment validation."""

    protocol_version: int
    worker_index: int
    assigned_shard_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ComputeMessage:
    """One exact, non-pipelined epoch command."""

    protocol_version: int
    worker_index: int
    epoch: int
    assigned_shard_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResultMessage:
    """Commit marker for completely published assigned-shard gradients."""

    protocol_version: int
    worker_index: int
    epoch: int
    assigned_shard_ids: tuple[int, ...]
    shard_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FailureMessage:
    """Sanitized controlled-failure record with no internal exception data."""

    protocol_version: int
    worker_index: int
    phase: WorkerFailurePhase
    code: WorkerFailureCode
    epoch: int | None
    assigned_shard_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StopMessage:
    """Cooperative stop command accepted only while the worker is ready."""

    protocol_version: int
    worker_index: int


@dataclass(frozen=True, slots=True)
class StoppedMessage:
    """Clean shutdown marker sent after attached resources are closed."""

    protocol_version: int
    worker_index: int


class WorkerProtocolValidationError(ValueError):
    """Stable protocol-boundary rejection without untrusted value disclosure."""

    __slots__ = ("code",)

    def __init__(
        self,
        code: WorkerFailureCode = WorkerFailureCode.INVALID_PROTOCOL,
    ) -> None:
        super().__init__("Worker protocol validation failed.")
        self.code = code


class RequestScopedWorkerGroupState(StrEnum):
    """Closed parent-side lifecycle states for one request-owned worker group."""

    ALLOCATED = "allocated"
    STARTING = "starting"
    READY = "ready"
    COMPUTING = "computing"
    STOPPING = "stopping"
    CLOSED = "closed"
    FAILED = "failed"


class RequestScopedWorkerGroupFailureCode(StrEnum):
    """Stable public failure categories for parent-side group operations."""

    INVALID_INPUT = "invalid_input"
    INVALID_STATE = "invalid_state"
    STARTUP_FAILURE = "startup_failure"
    EPOCH_FAILURE = "epoch_failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    CLEANUP_FAILURE = "cleanup_failure"
    INTERNAL_FAILURE = "internal_failure"


class RequestScopedWorkerGroupCleanupFailureCode(StrEnum):
    """Stable secondary cleanup categories without resource or exception details."""

    CANCELLATION_SIGNAL = "cancellation_signal"
    STOP_SEND = "stop_send"
    COOPERATIVE_WAIT = "cooperative_wait"
    TERMINATE = "terminate"
    TERMINATE_WAIT = "terminate_wait"
    KILL = "kill"
    JOIN = "join"
    EXIT_CODE = "exit_code"
    PIPE_CLOSE = "pipe_close"
    PROCESS_CLOSE = "process_close"
    VIEW_RELEASE = "view_release"
    SHARED_MEMORY_CLOSE = "shared_memory_close"
    SHARED_MEMORY_UNLINK = "shared_memory_unlink"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class RequestScopedWorkerGroupCleanupReport:
    """Immutable secondary cleanup outcome for one request-owned group."""

    cooperative_shutdown_completed: bool
    terminate_required: bool
    kill_required: bool
    process_exit_codes: tuple[int | None, ...]
    secondary_failures: tuple[RequestScopedWorkerGroupCleanupFailureCode, ...]

    @property
    def successful(self) -> bool:
        """Return whether cleanup completed without forced or secondary failure."""
        return (
            self.cooperative_shutdown_completed
            and not self.terminate_required
            and not self.kill_required
            and not self.secondary_failures
            and all(exit_code == 0 for exit_code in self.process_exit_codes)
        )


class RequestScopedWorkerGroupError(RuntimeError):
    """Stable group failure boundary without internal exception or resource data."""

    __slots__ = ("code", "cleanup_report")

    def __init__(
        self,
        code: RequestScopedWorkerGroupFailureCode,
        *,
        cleanup_report: RequestScopedWorkerGroupCleanupReport | None = None,
    ) -> None:
        if type(code) is not RequestScopedWorkerGroupFailureCode:
            raise TypeError("code must be a RequestScopedWorkerGroupFailureCode.")

        if (
            cleanup_report is not None
            and type(cleanup_report) is not RequestScopedWorkerGroupCleanupReport
        ):
            raise TypeError(
                "cleanup_report must be a RequestScopedWorkerGroupCleanupReport or None."
            )

        super().__init__("Request-scoped worker group operation failed.")
        self.code = code
        self.cleanup_report = cleanup_report


_WorkerGroupPollObserver: TypeAlias = Callable[[], Awaitable[None]]


class _SharedMemoryFactory(Protocol):
    def __call__(
        self,
        name: str | None = None,
        create: bool = False,
        size: int = 0,
    ) -> SharedMemory: ...


@dataclass(frozen=True, slots=True)
class _RequestScopedSharedMemorySnapshot:
    """Immutable allocation diagnostics without handles, views, or generated names."""

    block_count: int
    array_count: int
    requested_byte_count: int
    actual_byte_capacities: tuple[int, ...]
    dtype_markers: tuple[WorkerNumericDType, ...]
    shapes: tuple[tuple[int, ...], ...]
    c_contiguous: tuple[bool, ...]
    gradient_zeroed: tuple[bool, ...]
    cancellation_requested: bool


@dataclass(frozen=True, slots=True)
class _RequestScopedWorkerStartupSnapshot:
    """Immutable startup diagnostics without process IDs or pipe handles."""

    worker_count: int
    started_worker_count: int
    ready_worker_count: int
    non_daemonic: tuple[bool, ...]
    child_endpoints_closed: tuple[bool, ...]
    alive: tuple[bool, ...]


@dataclass(slots=True)
class _ParentWorkerProcess:
    """One parent-owned process, dedicated pipe, and startup state."""

    worker_index: int
    assigned_shard_ids: tuple[int, ...]
    parent_connection: _WorkerConnection
    child_connection: _WorkerConnection | None
    process: BaseProcess
    started: bool = False
    ready: bool = False
    observed_exit_code: int | None = None
    non_daemonic: bool = True
    process_closed: bool = False


class _ParentSharedMemoryResources:
    """Parent-owned numerical blocks and exact-range views for one request."""

    __slots__ = (
        "cancellation_signal",
        "gradient_buffers",
        "gradient_shared_memories",
        "gradient_storages",
        "layout",
        "released",
        "weight_buffer",
        "weight_shared_memory",
        "weight_storage",
    )

    def __init__(
        self,
        layout: TransformerParameterLayout,
        cancellation_signal: Event,
    ) -> None:
        self.layout = layout
        self.cancellation_signal = cancellation_signal
        self.weight_shared_memory: SharedMemory | None = None
        self.gradient_shared_memories: tuple[SharedMemory, ...] = ()
        self.weight_buffer: memoryview | None = None
        self.gradient_buffers: tuple[memoryview, ...] = ()
        self.weight_storage: _Float32Array | None = None
        self.gradient_storages: tuple[_Float32Array, ...] = ()
        self.released = False

    def allocation_snapshot(self) -> _RequestScopedSharedMemorySnapshot:
        """Return immutable structural diagnostics for focused allocation tests."""
        handles = self._ordered_handles()
        storages = self._ordered_storages()

        return _RequestScopedSharedMemorySnapshot(
            block_count=len(handles),
            array_count=len(storages),
            requested_byte_count=self.layout.total_byte_count,
            actual_byte_capacities=tuple(handle.size for handle in handles),
            dtype_markers=tuple(
                WorkerNumericDType.FLOAT32
                for storage in storages
                if storage.dtype == np.dtype(np.float32)
            ),
            shapes=tuple(storage.shape for storage in storages),
            c_contiguous=tuple(storage.flags.c_contiguous for storage in storages),
            gradient_zeroed=tuple(
                bool(np.count_nonzero(storage) == 0) for storage in self.gradient_storages
            ),
            cancellation_requested=self.cancellation_signal.is_set(),
        )

    def copy_published_weights(self) -> _Float32Array:
        """Return an independent copy of the current shared weight publication."""
        storage = self.weight_storage

        if storage is None:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_STATE)

        copied = np.array(
            storage,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        return copied

    def release(self) -> tuple[RequestScopedWorkerGroupCleanupFailureCode, ...]:
        """Release views, close all handles, and attempt every parent-owned unlink."""
        if self.released:
            return ()

        self.released = True
        secondary_failures: list[RequestScopedWorkerGroupCleanupFailureCode] = []
        buffers = self._ordered_buffers()
        handles = self._ordered_handles()

        self.weight_storage = None
        self.gradient_storages = ()

        try:
            gc.collect()
        except Exception:
            _append_cleanup_failure_once(
                secondary_failures,
                RequestScopedWorkerGroupCleanupFailureCode.VIEW_RELEASE,
            )

        self.weight_buffer = None
        self.gradient_buffers = ()

        for buffer in buffers:
            try:
                _release_parent_memoryview(buffer)
            except Exception:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.VIEW_RELEASE,
                )

        for handle in handles:
            try:
                _close_parent_shared_memory(handle)
            except Exception:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_CLOSE,
                )

        for handle in handles:
            try:
                _unlink_parent_shared_memory(handle)
            except Exception:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_UNLINK,
                )

        self.weight_shared_memory = None
        self.gradient_shared_memories = ()
        return tuple(secondary_failures)

    def _ordered_handles(self) -> tuple[SharedMemory, ...]:
        weight_handle = self.weight_shared_memory

        if weight_handle is None:
            return self.gradient_shared_memories

        return (weight_handle, *self.gradient_shared_memories)

    def _ordered_buffers(self) -> tuple[memoryview, ...]:
        weight_buffer = self.weight_buffer

        if weight_buffer is None:
            return self.gradient_buffers

        return (weight_buffer, *self.gradient_buffers)

    def _ordered_storages(self) -> tuple[_Float32Array, ...]:
        weight_storage = self.weight_storage

        if weight_storage is None:
            return self.gradient_storages

        return (weight_storage, *self.gradient_storages)


class _RequestScopedWorkerGroupRuntime(Protocol):
    async def start(
        self,
        *,
        num_layers: int,
        current_weights: _Float32Array,
        training_sequences: tuple[TransformerTrainingSequence, ...],
        logical_shards: tuple[LogicalTrainingShard, ...],
        actual_worker_count: int,
        assigned_shard_ids_by_worker: _WorkerShardAssignments,
        poll_observer: _WorkerGroupPollObserver,
    ) -> None: ...

    async def compute_epoch(
        self,
        *,
        epoch: int,
        current_weights: _Float32Array,
        poll_observer: _WorkerGroupPollObserver,
    ) -> tuple[LogicalTrainingShardResult, ...]: ...

    async def cleanup(self) -> RequestScopedWorkerGroupCleanupReport: ...


class _SharedMemoryRequestScopedWorkerGroupRuntime:
    """Step 5 runtime with bounded off-event-loop worker polling."""

    __slots__ = (
        "_actual_worker_count",
        "_assigned_shard_ids_by_worker",
        "_cancellation_signal",
        "_cleanup_report",
        "_connection_waiter",
        "_context",
        "_logical_shards",
        "_monotonic_clock",
        "_poll_observer",
        "_process_factory",
        "_resources",
        "_shared_memory_factory",
        "_started",
        "_startup_timeout_seconds",
        "_training_sequences",
        "_worker_target",
        "_workers",
    )

    def __init__(
        self,
        *,
        shared_memory_factory: _SharedMemoryFactory = SharedMemory,
        _process_context: BaseContext | None = None,
        _worker_target: _WorkerProcessTarget | None = None,
        _process_factory: _WorkerProcessFactory | None = None,
        _connection_waiter: _WorkerConnectionWaiter | None = None,
        _monotonic_clock: _WorkerMonotonicClock = monotonic,
        _startup_timeout_seconds: float = _WORKER_GROUP_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        if (
            type(_startup_timeout_seconds) is not float
            or not math.isfinite(_startup_timeout_seconds)
            or _startup_timeout_seconds <= 0.0
        ):
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

        self._shared_memory_factory = shared_memory_factory
        self._context = mp.get_context("spawn") if _process_context is None else _process_context
        self._worker_target = run_transformer_worker if _worker_target is None else _worker_target
        self._process_factory = (
            _create_non_daemonic_worker_process if _process_factory is None else _process_factory
        )
        self._connection_waiter = (
            _wait_for_worker_connections if _connection_waiter is None else _connection_waiter
        )
        self._monotonic_clock = _monotonic_clock
        self._startup_timeout_seconds = _startup_timeout_seconds
        self._poll_observer: _WorkerGroupPollObserver = _noop_worker_group_poll_observer
        self._cancellation_signal = Event()
        self._resources: _ParentSharedMemoryResources | None = None
        self._training_sequences: tuple[TransformerTrainingSequence, ...] = ()
        self._logical_shards: tuple[LogicalTrainingShard, ...] = ()
        self._actual_worker_count = 0
        self._assigned_shard_ids_by_worker: _WorkerShardAssignments = ()
        self._workers: tuple[_ParentWorkerProcess, ...] = ()
        self._started = False
        self._cleanup_report: RequestScopedWorkerGroupCleanupReport | None = None

    async def start(
        self,
        *,
        num_layers: int,
        current_weights: _Float32Array,
        training_sequences: tuple[TransformerTrainingSequence, ...],
        logical_shards: tuple[LogicalTrainingShard, ...],
        actual_worker_count: int,
        assigned_shard_ids_by_worker: _WorkerShardAssignments,
        poll_observer: _WorkerGroupPollObserver,
    ) -> None:
        if (
            self._started
            or self._resources is not None
            or self._workers
            or self._cleanup_report is not None
        ):
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_STATE)

        self._poll_observer = poll_observer

        try:
            layout = build_transformer_parameter_layout(num_layers)
        except (TypeError, ValueError) as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
            ) from exc

        if actual_worker_count != len(
            assigned_shard_ids_by_worker
        ) or assigned_shard_ids_by_worker != build_worker_shard_assignments(actual_worker_count):
            _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

        resources = _ParentSharedMemoryResources(
            layout,
            self._cancellation_signal,
        )
        self._resources = resources
        self._training_sequences = training_sequences
        self._logical_shards = logical_shards
        self._actual_worker_count = actual_worker_count
        self._assigned_shard_ids_by_worker = assigned_shard_ids_by_worker

        _allocate_parent_shared_array(
            resources,
            is_weight=True,
            shared_memory_factory=self._shared_memory_factory,
        )

        for _ in range(LOGICAL_TRAINING_SHARD_COUNT):
            _allocate_parent_shared_array(
                resources,
                is_weight=False,
                shared_memory_factory=self._shared_memory_factory,
            )

        _initialize_parent_gradient_storages(resources)
        _publish_parent_weights(
            resources,
            current_weights,
        )

        startup_deadline = self._monotonic_clock() + self._startup_timeout_seconds

        for worker_index, assigned_shard_ids in enumerate(assigned_shard_ids_by_worker):
            startup_config = _build_parent_worker_startup_config(
                worker_index=worker_index,
                assigned_shard_ids=assigned_shard_ids,
                num_layers=num_layers,
                training_sequences=training_sequences,
                logical_shards=logical_shards,
                resources=resources,
            )
            self._spawn_worker(
                startup_config=startup_config,
                assigned_shard_ids=assigned_shard_ids,
            )

        await self._wait_for_complete_startup(startup_deadline)
        self._started = True

    async def compute_epoch(
        self,
        *,
        epoch: int,
        current_weights: _Float32Array,
        poll_observer: _WorkerGroupPollObserver,
    ) -> tuple[LogicalTrainingShardResult, ...]:
        resources = self._resources

        if not self._started or resources is None:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        if self._cancellation_signal.is_set():
            _raise_group_error(RequestScopedWorkerGroupFailureCode.CANCELLED)

        for worker in self._workers:
            if (
                not worker.started
                or not worker.ready
                or worker.process_closed
                or not worker.process.is_alive()
            ):
                worker.observed_exit_code = worker.process.exitcode
                _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        try:
            _publish_parent_weights(
                resources,
                current_weights,
            )
        except RequestScopedWorkerGroupError as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
            ) from exc

        epoch_deadline = self._monotonic_clock() + _WORKER_GROUP_EPOCH_TIMEOUT_SECONDS

        for worker in self._workers:
            try:
                _send_worker_message(
                    worker.parent_connection,
                    ComputeMessage(
                        protocol_version=WORKER_PROTOCOL_VERSION,
                        worker_index=worker.worker_index,
                        epoch=epoch,
                        assigned_shard_ids=worker.assigned_shard_ids,
                    ),
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise RequestScopedWorkerGroupError(
                    RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
                ) from exc

        commits_by_worker: dict[int, ResultMessage] = {}

        while len(commits_by_worker) != len(self._workers):
            waitables: tuple[_WorkerWaitable, ...] = (
                *(worker.parent_connection for worker in self._workers),
                *(worker.process.sentinel for worker in self._workers),
            )
            ready_objects = await self._poll_waitables(
                waitables,
                deadline=epoch_deadline,
                poll_observer=poll_observer,
                wait_failure_code=(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE),
                observe_cancellation=True,
            )

            if not ready_objects:
                continue

            for worker in self._workers:
                if worker.process.sentinel not in ready_objects:
                    continue

                try:
                    worker.process.join(0.0)
                except (
                    AssertionError,
                    RuntimeError,
                    ValueError,
                ):
                    pass

                worker.observed_exit_code = worker.process.exitcode
                _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

            for worker in self._workers:
                if not any(
                    ready_object is worker.parent_connection for ready_object in ready_objects
                ):
                    continue

                try:
                    raw_message = worker.parent_connection.recv()
                except (EOFError, OSError) as exc:
                    raise RequestScopedWorkerGroupError(
                        RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
                    ) from exc

                # Receive the offending duplicate before failing so it cannot
                # remain queued ahead of the later StoppedMessage.
                if worker.worker_index in commits_by_worker:
                    _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

                if type(raw_message) is FailureMessage:
                    try:
                        failure = validate_failure_message(
                            raw_message,
                            expected_worker_index=worker.worker_index,
                            expected_shard_ids=(worker.assigned_shard_ids),
                        )
                    except WorkerProtocolValidationError as exc:
                        raise RequestScopedWorkerGroupError(
                            RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
                        ) from exc

                    if failure.phase is not WorkerFailurePhase.COMPUTE or failure.epoch != epoch:
                        _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

                    _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

                try:
                    commit = validate_result_message(
                        raw_message,
                        expected_worker_index=worker.worker_index,
                        expected_epoch=epoch,
                        expected_shard_ids=(worker.assigned_shard_ids),
                    )
                except WorkerProtocolValidationError as exc:
                    raise RequestScopedWorkerGroupError(
                        RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
                    ) from exc

                commits_by_worker[worker.worker_index] = commit

        for worker in self._workers:
            if not worker.process.is_alive():
                worker.observed_exit_code = worker.process.exitcode
                _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        return _materialize_committed_epoch_results(
            resources=resources,
            logical_shards=self._logical_shards,
            workers=self._workers,
            commits_by_worker=commits_by_worker,
        )

    async def cleanup(self) -> RequestScopedWorkerGroupCleanupReport:
        if self._cleanup_report is not None:
            return self._cleanup_report

        secondary_failures: list[RequestScopedWorkerGroupCleanupFailureCode] = []
        terminate_required = False
        kill_required = False
        cancellation_was_requested = self._cancellation_signal.is_set()

        try:
            _set_worker_group_cancellation(self._cancellation_signal)
        except Exception:
            _append_cleanup_failure_once(
                secondary_failures,
                RequestScopedWorkerGroupCleanupFailureCode.CANCELLATION_SIGNAL,
            )

        cleanup_poll_observer = (
            _noop_worker_group_poll_observer if cancellation_was_requested else self._poll_observer
        )
        cooperative_required_worker_indexes = {
            worker.worker_index for worker in self._workers if worker.started and worker.ready
        }
        stop_sent_worker_indexes: set[int] = set()
        stopped_observed_worker_indexes: set[int] = set()
        stopped_worker_indexes: set[int] = set()
        cooperative_protocol_failed_worker_indexes: set[int] = set()

        live_workers = _collect_live_parent_workers(
            self._workers,
            secondary_failures,
        )
        live_worker_indexes = {worker.worker_index for worker in live_workers}

        for worker in self._workers:
            if worker.worker_index not in cooperative_required_worker_indexes:
                continue

            if worker.worker_index not in live_worker_indexes:
                cooperative_protocol_failed_worker_indexes.add(worker.worker_index)
                continue

            try:
                _send_worker_stop_message(
                    worker.parent_connection,
                    worker.worker_index,
                )
                stop_sent_worker_indexes.add(worker.worker_index)
            except Exception:
                cooperative_protocol_failed_worker_indexes.add(worker.worker_index)
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.STOP_SEND,
                )

        cooperative_deadline = (
            self._monotonic_clock() + _WORKER_GROUP_INTERIM_SHUTDOWN_TIMEOUT_SECONDS
        )
        cooperative_shutdown_completed = await self._wait_for_cooperative_shutdown_until(
            deadline=cooperative_deadline,
            poll_observer=cleanup_poll_observer,
            cooperative_required_worker_indexes=(cooperative_required_worker_indexes),
            stop_sent_worker_indexes=stop_sent_worker_indexes,
            stopped_observed_worker_indexes=(stopped_observed_worker_indexes),
            stopped_worker_indexes=stopped_worker_indexes,
            cooperative_protocol_failed_worker_indexes=(cooperative_protocol_failed_worker_indexes),
            secondary_failures=secondary_failures,
        )

        survivors = _collect_live_parent_workers(
            self._workers,
            secondary_failures,
        )

        if survivors:
            cooperative_shutdown_completed = False
            terminate_required = True

            for worker in survivors:
                try:
                    _terminate_worker_process(worker.process)
                except Exception:
                    _append_cleanup_failure_once(
                        secondary_failures,
                        RequestScopedWorkerGroupCleanupFailureCode.TERMINATE,
                    )

            terminate_deadline = (
                self._monotonic_clock() + _WORKER_GROUP_INTERIM_SHUTDOWN_TIMEOUT_SECONDS
            )
            await self._wait_for_started_workers_until(
                deadline=terminate_deadline,
                poll_observer=cleanup_poll_observer,
                wait_failure_code=(RequestScopedWorkerGroupCleanupFailureCode.TERMINATE_WAIT),
                secondary_failures=secondary_failures,
            )

        survivors = _collect_live_parent_workers(
            self._workers,
            secondary_failures,
        )

        if survivors:
            kill_required = True

            for worker in survivors:
                try:
                    _kill_worker_process(worker.process)
                except Exception:
                    _append_cleanup_failure_once(
                        secondary_failures,
                        RequestScopedWorkerGroupCleanupFailureCode.KILL,
                    )

            kill_deadline = self._monotonic_clock() + _WORKER_GROUP_INTERIM_SHUTDOWN_TIMEOUT_SECONDS
            await self._wait_for_started_workers_until(
                deadline=kill_deadline,
                poll_observer=cleanup_poll_observer,
                wait_failure_code=(RequestScopedWorkerGroupCleanupFailureCode.JOIN),
                secondary_failures=secondary_failures,
            )

        final_live_workers = _collect_live_parent_workers(
            self._workers,
            secondary_failures,
        )
        final_live_worker_indexes = {worker.worker_index for worker in final_live_workers}

        for worker in self._workers:
            if not worker.started:
                continue

            if worker.worker_index in final_live_worker_indexes:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.JOIN,
                )
                continue

            try:
                _join_worker_process(worker.process, 0.0)
            except Exception:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.JOIN,
                )

        for worker in self._workers:
            if not worker.started:
                continue

            try:
                worker.observed_exit_code = worker.process.exitcode
            except Exception:
                worker.observed_exit_code = None

            if worker.observed_exit_code != 0:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.EXIT_CODE,
                )

        process_exit_codes = tuple(
            worker.observed_exit_code for worker in self._workers if worker.started
        )

        if any(exit_code != 0 for exit_code in process_exit_codes):
            cooperative_shutdown_completed = False

        for worker in self._workers:
            for connection in (
                worker.parent_connection,
                worker.child_connection,
            ):
                if connection is None:
                    continue

                try:
                    _close_parent_worker_connection(connection)
                except Exception:
                    _append_cleanup_failure_once(
                        secondary_failures,
                        RequestScopedWorkerGroupCleanupFailureCode.PIPE_CLOSE,
                    )

            worker.child_connection = None

        for worker in self._workers:
            if worker.process_closed:
                continue

            if worker.worker_index in final_live_worker_indexes:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.PROCESS_CLOSE,
                )
                continue

            try:
                _close_parent_worker_process(worker.process)
                worker.process_closed = True
            except Exception:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.PROCESS_CLOSE,
                )

        if final_live_workers:
            _append_cleanup_failure_once(
                secondary_failures,
                RequestScopedWorkerGroupCleanupFailureCode.INTERNAL,
            )
        elif self._resources is not None:
            try:
                release_failures = self._resources.release()
            except Exception:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.INTERNAL,
                )
            else:
                for failure in release_failures:
                    _append_cleanup_failure_once(
                        secondary_failures,
                        failure,
                    )

        self._started = False
        self._training_sequences = ()
        self._logical_shards = ()
        self._actual_worker_count = 0
        self._assigned_shard_ids_by_worker = ()

        report = RequestScopedWorkerGroupCleanupReport(
            cooperative_shutdown_completed=(
                cooperative_shutdown_completed and not terminate_required and not kill_required
            ),
            terminate_required=terminate_required,
            kill_required=kill_required,
            process_exit_codes=process_exit_codes,
            secondary_failures=tuple(secondary_failures),
        )
        self._cleanup_report = report
        return report

    def allocation_snapshot(self) -> _RequestScopedSharedMemorySnapshot:
        """Return immutable allocation diagnostics without live resource exposure."""
        resources = self._resources

        if resources is None:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_STATE)

        return resources.allocation_snapshot()

    def startup_snapshot(self) -> _RequestScopedWorkerStartupSnapshot:
        """Return immutable startup diagnostics without process identifiers."""
        return _RequestScopedWorkerStartupSnapshot(
            worker_count=len(self._workers),
            started_worker_count=sum(worker.started for worker in self._workers),
            ready_worker_count=sum(worker.ready for worker in self._workers),
            non_daemonic=tuple(worker.non_daemonic for worker in self._workers),
            child_endpoints_closed=tuple(
                worker.child_connection is None for worker in self._workers
            ),
            alive=tuple(
                False if worker.process_closed else worker.started and worker.process.is_alive()
                for worker in self._workers
            ),
        )

    def copy_published_weights(self) -> _Float32Array:
        """Return an independent copy for focused weight-publication tests."""
        resources = self._resources

        if resources is None:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_STATE)

        return resources.copy_published_weights()

    def request_cancellation(self) -> None:
        """Set the request-owned cancellation signal without exposing it."""
        self._cancellation_signal.set()

    def _spawn_worker(
        self,
        *,
        startup_config: WorkerStartupConfig,
        assigned_shard_ids: tuple[int, ...],
    ) -> None:
        try:
            parent_connection, child_connection = self._context.Pipe(
                duplex=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
            ) from exc

        try:
            process = self._process_factory(
                self._context,
                self._worker_target,
                child_connection,
                startup_config.worker_index,
                startup_config,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _close_worker_connection_after_start_failure(
                parent_connection,
            )
            _close_worker_connection_after_start_failure(
                child_connection,
            )
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
            ) from exc

        worker = _ParentWorkerProcess(
            worker_index=startup_config.worker_index,
            assigned_shard_ids=assigned_shard_ids,
            parent_connection=parent_connection,
            child_connection=child_connection,
            process=process,
            non_daemonic=not bool(process.daemon),
        )
        self._workers = (*self._workers, worker)

        if not worker.non_daemonic:
            _close_worker_connection_after_start_failure(
                parent_connection,
            )
            _close_worker_connection_after_start_failure(
                child_connection,
            )
            worker.child_connection = None
            _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

        try:
            process.start()
            worker.started = True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _close_worker_connection_after_start_failure(
                parent_connection,
            )
            _close_worker_connection_after_start_failure(
                child_connection,
            )
            worker.child_connection = None
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
            ) from exc

        try:
            child_connection.close()
        except OSError as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
            ) from exc

        worker.child_connection = None

    async def _wait_for_complete_startup(
        self,
        startup_deadline: float,
    ) -> None:
        while not all(worker.ready for worker in self._workers):
            waitables = self._startup_waitables()

            if not waitables:
                _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

            ready_objects = await self._poll_waitables(
                waitables,
                deadline=startup_deadline,
                poll_observer=self._poll_observer,
                wait_failure_code=(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE),
                observe_cancellation=True,
            )

            if not ready_objects:
                continue

            for worker in self._workers:
                if not worker.started:
                    continue

                if worker.process.sentinel in ready_objects:
                    try:
                        worker.process.join(0.0)
                    except (AssertionError, RuntimeError, ValueError):
                        pass

                    worker.observed_exit_code = worker.process.exitcode

                    if not worker.ready:
                        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

            for worker in self._workers:
                if not worker.started:
                    continue

                if not any(
                    ready_object is worker.parent_connection for ready_object in ready_objects
                ):
                    continue

                if worker.ready:
                    _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

                try:
                    raw_message = worker.parent_connection.recv()
                except (EOFError, OSError) as exc:
                    raise RequestScopedWorkerGroupError(
                        RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
                    ) from exc

                try:
                    validate_ready_message(
                        raw_message,
                        expected_worker_index=worker.worker_index,
                        expected_shard_ids=worker.assigned_shard_ids,
                    )
                except WorkerProtocolValidationError as exc:
                    raise RequestScopedWorkerGroupError(
                        RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
                    ) from exc

                worker.ready = True

        for worker in self._workers:
            if not worker.started or not worker.process.is_alive():
                worker.observed_exit_code = worker.process.exitcode
                _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

    async def _wait_for_cooperative_shutdown_until(
        self,
        *,
        deadline: float,
        poll_observer: _WorkerGroupPollObserver,
        cooperative_required_worker_indexes: set[int],
        stop_sent_worker_indexes: set[int],
        stopped_observed_worker_indexes: set[int],
        stopped_worker_indexes: set[int],
        cooperative_protocol_failed_worker_indexes: set[int],
        secondary_failures: list[RequestScopedWorkerGroupCleanupFailureCode],
    ) -> bool:
        while True:
            live_workers = _collect_live_parent_workers(
                self._workers,
                secondary_failures,
            )
            pending_stopped_workers = tuple(
                worker
                for worker in self._workers
                if worker.worker_index in stop_sent_worker_indexes
                and worker.worker_index not in stopped_observed_worker_indexes
            )

            if not live_workers and not pending_stopped_workers:
                break

            waitables: tuple[_WorkerWaitable, ...] = (
                *(worker.parent_connection for worker in pending_stopped_workers),
                *(worker.process.sentinel for worker in live_workers),
            )

            if not waitables:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.COOPERATIVE_WAIT,
                )
                return False

            try:
                ready_objects = await self._poll_waitables(
                    waitables,
                    deadline=deadline,
                    poll_observer=poll_observer,
                    wait_failure_code=(RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE),
                    observe_cancellation=False,
                )
            except RequestScopedWorkerGroupError:
                _append_cleanup_failure_once(
                    secondary_failures,
                    RequestScopedWorkerGroupCleanupFailureCode.COOPERATIVE_WAIT,
                )
                return False

            for worker in pending_stopped_workers:
                if not any(
                    ready_object is worker.parent_connection for ready_object in ready_objects
                ):
                    continue

                stopped_observed_worker_indexes.add(worker.worker_index)

                try:
                    raw_message = worker.parent_connection.recv()
                    validate_stopped_message(
                        raw_message,
                        expected_worker_index=worker.worker_index,
                    )
                except (
                    EOFError,
                    OSError,
                    WorkerProtocolValidationError,
                ):
                    cooperative_protocol_failed_worker_indexes.add(worker.worker_index)
                    _append_cleanup_failure_once(
                        secondary_failures,
                        RequestScopedWorkerGroupCleanupFailureCode.COOPERATIVE_WAIT,
                    )
                else:
                    stopped_worker_indexes.add(worker.worker_index)

            for worker in live_workers:
                if worker.process.sentinel not in ready_objects:
                    continue

                try:
                    _join_worker_process(worker.process, 0.0)
                except Exception:
                    _append_cleanup_failure_once(
                        secondary_failures,
                        RequestScopedWorkerGroupCleanupFailureCode.COOPERATIVE_WAIT,
                    )

                try:
                    worker.observed_exit_code = worker.process.exitcode
                except Exception:
                    worker.observed_exit_code = None

        if cooperative_protocol_failed_worker_indexes:
            return False

        if not cooperative_required_worker_indexes.issubset(stopped_worker_indexes):
            _append_cleanup_failure_once(
                secondary_failures,
                RequestScopedWorkerGroupCleanupFailureCode.COOPERATIVE_WAIT,
            )
            return False

        return not _collect_live_parent_workers(
            self._workers,
            secondary_failures,
        )

    async def _poll_waitables(
        self,
        waitables: tuple[_WorkerWaitable, ...],
        *,
        deadline: float,
        poll_observer: _WorkerGroupPollObserver,
        wait_failure_code: RequestScopedWorkerGroupFailureCode,
        observe_cancellation: bool,
    ) -> list[_WorkerWaitable]:
        remaining = deadline - self._monotonic_clock()

        if remaining <= 0.0:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.TIMEOUT)

        poll_timeout = min(
            _WORKER_GROUP_POLL_TIMEOUT_SECONDS,
            remaining,
        )
        wait_task = asyncio.create_task(
            asyncio.to_thread(
                self._connection_waiter,
                waitables,
                poll_timeout,
            )
        )

        try:
            ready_objects = await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            self._cancellation_signal.set()

            # The delegated wait is bounded to at most 0.1 seconds. Let that
            # thread finish before cleanup can close its waitable resources.
            if not wait_task.done():
                try:
                    await wait_task
                except (OSError, ValueError):
                    pass

            raise
        except (OSError, ValueError) as exc:
            raise RequestScopedWorkerGroupError(
                wait_failure_code,
            ) from exc

        try:
            await poll_observer()
        except asyncio.CancelledError:
            self._cancellation_signal.set()
            raise

        if observe_cancellation and self._cancellation_signal.is_set():
            _raise_group_error(RequestScopedWorkerGroupFailureCode.CANCELLED)

        if self._monotonic_clock() >= deadline:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.TIMEOUT)

        return ready_objects

    async def _wait_for_started_workers_until(
        self,
        *,
        deadline: float,
        poll_observer: _WorkerGroupPollObserver,
        wait_failure_code: RequestScopedWorkerGroupCleanupFailureCode,
        secondary_failures: list[RequestScopedWorkerGroupCleanupFailureCode],
    ) -> bool:
        while True:
            live_workers = _collect_live_parent_workers(
                self._workers,
                secondary_failures,
            )

            if not live_workers:
                return True

            waitables: tuple[_WorkerWaitable, ...] = tuple(
                worker.process.sentinel for worker in live_workers
            )

            try:
                ready_objects = await self._poll_waitables(
                    waitables,
                    deadline=deadline,
                    poll_observer=poll_observer,
                    wait_failure_code=(RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE),
                    observe_cancellation=False,
                )
            except RequestScopedWorkerGroupError:
                _append_cleanup_failure_once(
                    secondary_failures,
                    wait_failure_code,
                )
                return False

            for worker in live_workers:
                if worker.process.sentinel not in ready_objects:
                    continue

                try:
                    _join_worker_process(worker.process, 0.0)
                except Exception:
                    _append_cleanup_failure_once(
                        secondary_failures,
                        wait_failure_code,
                    )

                try:
                    worker.observed_exit_code = worker.process.exitcode
                except Exception:
                    worker.observed_exit_code = None

    def _startup_waitables(self) -> tuple[_WorkerWaitable, ...]:
        connections: tuple[_WorkerWaitable, ...] = tuple(
            worker.parent_connection for worker in self._workers if worker.started
        )
        sentinels: tuple[_WorkerWaitable, ...] = tuple(
            worker.process.sentinel for worker in self._workers if worker.started
        )
        return (*connections, *sentinels)


async def _noop_worker_group_poll_observer() -> None:
    return None


def _raise_group_error(
    code: RequestScopedWorkerGroupFailureCode,
) -> NoReturn:
    raise RequestScopedWorkerGroupError(code)


def _append_cleanup_failure_once(
    failures: list[RequestScopedWorkerGroupCleanupFailureCode],
    code: RequestScopedWorkerGroupCleanupFailureCode,
) -> None:
    if code not in failures:
        failures.append(code)


def _create_non_daemonic_worker_process(
    context: BaseContext,
    target: _WorkerProcessTarget,
    child_connection: _WorkerConnection,
    worker_index: int,
    startup_config: object,
) -> BaseProcess:
    spawn_context = cast(
        SpawnContext,
        context,
    )

    return spawn_context.Process(
        target=target,
        args=(
            child_connection,
            worker_index,
            startup_config,
        ),
        daemon=False,
    )


def _wait_for_worker_connections(
    waitables: tuple[_WorkerWaitable, ...],
    timeout: float | None,
) -> list[_WorkerWaitable]:
    ready_objects = wait_for_connections(
        waitables,
        timeout=timeout,
    )

    return cast(
        list[_WorkerWaitable],
        ready_objects,
    )


def _send_worker_message(
    connection: _WorkerConnection,
    message: object,
) -> None:
    connection.send(message)


def _set_worker_group_cancellation(signal: Event) -> None:
    signal.set()


def _send_worker_stop_message(
    connection: _WorkerConnection,
    worker_index: int,
) -> None:
    _send_worker_message(
        connection,
        StopMessage(
            protocol_version=WORKER_PROTOCOL_VERSION,
            worker_index=worker_index,
        ),
    )


def _terminate_worker_process(process: BaseProcess) -> None:
    process.terminate()


def _kill_worker_process(process: BaseProcess) -> None:
    process.kill()


def _join_worker_process(
    process: BaseProcess,
    timeout: float | None,
) -> None:
    process.join(timeout)


def _close_parent_worker_connection(
    connection: _WorkerConnection,
) -> None:
    connection.close()


def _close_parent_worker_process(process: BaseProcess) -> None:
    process.close()


def _release_parent_memoryview(buffer: memoryview) -> None:
    buffer.release()


def _close_parent_shared_memory(handle: SharedMemory) -> None:
    handle.close()


def _unlink_parent_shared_memory(handle: SharedMemory) -> None:
    handle.unlink()


def _collect_live_parent_workers(
    workers: tuple[_ParentWorkerProcess, ...],
    secondary_failures: list[RequestScopedWorkerGroupCleanupFailureCode],
) -> tuple[_ParentWorkerProcess, ...]:
    live_workers: list[_ParentWorkerProcess] = []

    for worker in workers:
        if not worker.started or worker.process_closed:
            continue

        try:
            is_alive = worker.process.is_alive()
        except Exception:
            _append_cleanup_failure_once(
                secondary_failures,
                RequestScopedWorkerGroupCleanupFailureCode.INTERNAL,
            )
            is_alive = True

        if is_alive:
            live_workers.append(worker)

    return tuple(live_workers)


def _close_worker_connection_after_start_failure(
    connection: _WorkerConnection,
) -> None:
    try:
        connection.close()
    except OSError:
        return


def _validate_parent_shared_array(
    storage: object,
    *,
    layout: TransformerParameterLayout,
) -> _Float32Array:
    if (
        type(storage) is not np.ndarray
        or storage.dtype != np.dtype(np.float32)
        or storage.shape != (layout.total_float_count,)
        or not storage.flags.c_contiguous
        or not storage.flags.writeable
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

    return storage


def _allocate_parent_shared_array(
    resources: _ParentSharedMemoryResources,
    *,
    is_weight: bool,
    shared_memory_factory: _SharedMemoryFactory,
) -> _Float32Array:
    try:
        handle = shared_memory_factory(
            create=True,
            size=resources.layout.total_byte_count,
        )
    except (OSError, ValueError) as exc:
        raise RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        ) from exc

    if is_weight:
        if resources.weight_shared_memory is not None:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)
        resources.weight_shared_memory = handle
    else:
        if len(resources.gradient_shared_memories) >= LOGICAL_TRAINING_SHARD_COUNT:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)
        resources.gradient_shared_memories = (
            *resources.gradient_shared_memories,
            handle,
        )

    if handle.size < resources.layout.total_byte_count:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

    shared_buffer = handle.buf

    if shared_buffer is None:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

    try:
        exact_buffer = shared_buffer[: resources.layout.total_byte_count]
    except (BufferError, ValueError) as exc:
        raise RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        ) from exc

    if is_weight:
        resources.weight_buffer = exact_buffer
    else:
        resources.gradient_buffers = (
            *resources.gradient_buffers,
            exact_buffer,
        )

    try:
        storage = np.ndarray(
            (resources.layout.total_float_count,),
            dtype=np.float32,
            buffer=exact_buffer,
            order="C",
        )
    except (BufferError, TypeError, ValueError) as exc:
        raise RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        ) from exc

    validated = _validate_parent_shared_array(
        storage,
        layout=resources.layout,
    )

    if is_weight:
        resources.weight_storage = validated
    else:
        resources.gradient_storages = (
            *resources.gradient_storages,
            validated,
        )

    return validated


def _initialize_parent_gradient_storages(
    resources: _ParentSharedMemoryResources,
) -> None:
    if len(resources.gradient_storages) != LOGICAL_TRAINING_SHARD_COUNT:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

    for storage in resources.gradient_storages:
        storage.fill(np.float32(0.0))

        if np.count_nonzero(storage) != 0:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)


def _publish_parent_weights(
    resources: _ParentSharedMemoryResources,
    current_weights: _Float32Array,
) -> None:
    destination = resources.weight_storage

    if destination is None:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

    if (
        current_weights.dtype != np.dtype(np.float32)
        or current_weights.shape != (resources.layout.total_float_count,)
        or not current_weights.flags.c_contiguous
        or not np.isfinite(current_weights).all()
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    try:
        np.copyto(
            destination,
            current_weights,
            casting="no",
        )
    except (TypeError, ValueError) as exc:
        raise RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        ) from exc

    if not np.isfinite(destination).all():
        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)


def _materialize_committed_epoch_results(
    *,
    resources: _ParentSharedMemoryResources,
    logical_shards: tuple[LogicalTrainingShard, ...],
    workers: tuple[_ParentWorkerProcess, ...],
    commits_by_worker: dict[int, ResultMessage],
) -> tuple[LogicalTrainingShardResult, ...]:
    if (
        len(logical_shards) != LOGICAL_TRAINING_SHARD_COUNT
        or len(resources.gradient_storages) != LOGICAL_TRAINING_SHARD_COUNT
        or len(commits_by_worker) != len(workers)
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

    losses_by_shard: list[float | None] = [None for _ in range(LOGICAL_TRAINING_SHARD_COUNT)]

    for worker in workers:
        commit = commits_by_worker.get(worker.worker_index)

        if commit is None:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        for shard_id, committed_shard_loss in zip(
            worker.assigned_shard_ids,
            commit.shard_losses,
            strict=True,
        ):
            if (
                shard_id < 0
                or shard_id >= LOGICAL_TRAINING_SHARD_COUNT
                or losses_by_shard[shard_id] is not None
                or not math.isfinite(committed_shard_loss)
            ):
                _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

            losses_by_shard[shard_id] = float(committed_shard_loss)

    results: list[LogicalTrainingShardResult] = []

    for shard_id in range(LOGICAL_TRAINING_SHARD_COUNT):
        shard = logical_shards[shard_id]
        committed_loss = losses_by_shard[shard_id]
        source = resources.gradient_storages[shard_id]

        if (
            shard.shard_index != shard_id
            or committed_loss is None
            or source.dtype != np.dtype(np.float32)
            or source.shape != (resources.layout.total_float_count,)
            or not source.flags.c_contiguous
            or not np.isfinite(source).all()
        ):
            _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        copied_gradient = create_transformer_gradient_buffer(resources.layout)

        try:
            np.copyto(
                copied_gradient.storage,
                source,
                casting="no",
            )
        except (TypeError, ValueError) as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
            ) from exc

        if not np.isfinite(copied_gradient.storage).all():
            _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        results.append(
            LogicalTrainingShardResult(
                shard=shard,
                processed_sequence_count=(shard.stop_index - shard.start_index),
                loss=committed_loss,
                gradient=copied_gradient,
            )
        )

    return tuple(results)


def _build_parent_worker_startup_config(
    *,
    worker_index: int,
    assigned_shard_ids: tuple[int, ...],
    num_layers: int,
    training_sequences: tuple[TransformerTrainingSequence, ...],
    logical_shards: tuple[LogicalTrainingShard, ...],
    resources: _ParentSharedMemoryResources,
) -> WorkerStartupConfig:
    weight_shared_memory = resources.weight_shared_memory
    gradient_shared_memories = resources.gradient_shared_memories

    if (
        weight_shared_memory is None
        or len(gradient_shared_memories) != LOGICAL_TRAINING_SHARD_COUNT
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE)

    shard_by_id = {shard.shard_index: shard for shard in logical_shards}

    try:
        assigned_shards = tuple(
            (
                shard_by_id[shard_id].shard_index,
                shard_by_id[shard_id].start_index,
                shard_by_id[shard_id].stop_index,
            )
            for shard_id in assigned_shard_ids
        )
        gradient_shared_memory_names = tuple(
            gradient_shared_memories[shard_id].name for shard_id in assigned_shard_ids
        )
    except (AttributeError, IndexError, KeyError) as exc:
        raise RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        ) from exc

    config = WorkerStartupConfig(
        protocol_version=WORKER_PROTOCOL_VERSION,
        worker_index=worker_index,
        num_layers=num_layers,
        dtype_marker=WorkerNumericDType.FLOAT32,
        canonical_float_count=resources.layout.total_float_count,
        sequence_count=len(training_sequences),
        weight_shared_memory_name=weight_shared_memory.name,
        assigned_shards=assigned_shards,
        gradient_shared_memory_names=gradient_shared_memory_names,
        training_sequences=tuple(
            (
                sequence.input_ids,
                sequence.target_ids,
            )
            for sequence in training_sequences
        ),
    )

    try:
        return validate_worker_startup_config(
            config,
            expected_worker_index=worker_index,
        )
    except WorkerProtocolValidationError as exc:
        raise RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        ) from exc


def calculate_actual_worker_count(reported_cpu_count: int | None) -> int:
    """Bound one reported host CPU count to the supported worker range."""
    if reported_cpu_count is not None and type(reported_cpu_count) is not int:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    return min(
        LOGICAL_TRAINING_SHARD_COUNT,
        max(1, reported_cpu_count or 1),
    )


def build_worker_shard_assignments(
    actual_worker_count: int,
) -> _WorkerShardAssignments:
    """Assign all four logical shard IDs statically by modulo worker index."""
    if (
        type(actual_worker_count) is not int
        or actual_worker_count < 1
        or actual_worker_count > LOGICAL_TRAINING_SHARD_COUNT
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    return tuple(
        tuple(
            shard_id
            for shard_id in range(LOGICAL_TRAINING_SHARD_COUNT)
            if shard_id % actual_worker_count == worker_index
        )
        for worker_index in range(actual_worker_count)
    )


def _validate_group_epoch(epoch: object) -> int:
    if type(epoch) is not int or epoch < 0:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    return epoch


def _canonical_group_weight_snapshot(
    current_weights: object,
    *,
    layout: TransformerParameterLayout,
) -> _Float32Array:
    if (
        type(current_weights) is not np.ndarray
        or current_weights.dtype != np.dtype(np.float32)
        or current_weights.shape != (layout.total_float_count,)
        or not current_weights.flags.c_contiguous
        or not np.isfinite(current_weights).all()
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    snapshot = np.array(
        current_weights,
        dtype=np.float32,
        order="C",
        copy=True,
    )

    if (
        snapshot.dtype != np.dtype(np.float32)
        or snapshot.shape != (layout.total_float_count,)
        or not snapshot.flags.c_contiguous
        or not np.isfinite(snapshot).all()
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INTERNAL_FAILURE)

    return snapshot


def _validate_group_training_sequences(
    training_sequences: object,
    *,
    vocabulary_size: int,
) -> tuple[TransformerTrainingSequence, ...]:
    if type(training_sequences) is not tuple:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    validated: list[TransformerTrainingSequence] = []

    for sequence in training_sequences:
        if type(sequence) is not TransformerTrainingSequence:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

        if (
            type(sequence.input_ids) is not tuple
            or type(sequence.target_ids) is not tuple
            or len(sequence.input_ids) != TRANSFORMER_SEQUENCE_LENGTH
            or len(sequence.target_ids) != TRANSFORMER_SEQUENCE_LENGTH
        ):
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

        for token_id in (*sequence.input_ids, *sequence.target_ids):
            if type(token_id) is not int or token_id < 0 or token_id >= vocabulary_size:
                _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

        validated.append(sequence)

    return tuple(validated)


def _validate_group_logical_shards(
    logical_shards: object,
    *,
    sequence_count: int,
) -> tuple[LogicalTrainingShard, ...]:
    if type(logical_shards) is not tuple:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    if any(type(shard) is not LogicalTrainingShard for shard in logical_shards):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    expected = build_logical_training_shards(sequence_count)

    if logical_shards != expected:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    return cast(tuple[LogicalTrainingShard, ...], logical_shards)


def _validate_group_poll_observer(
    poll_observer: object,
) -> _WorkerGroupPollObserver:
    if poll_observer is None:
        return _noop_worker_group_poll_observer

    if not callable(poll_observer):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    return cast(_WorkerGroupPollObserver, poll_observer)


def _validate_group_runtime(
    runtime: object,
) -> _RequestScopedWorkerGroupRuntime:
    if not all(
        callable(getattr(runtime, method_name, None))
        for method_name in ("start", "compute_epoch", "cleanup")
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

    return cast(_RequestScopedWorkerGroupRuntime, runtime)


def _validate_group_cleanup_report(
    report: object,
) -> RequestScopedWorkerGroupCleanupReport:
    if type(report) is not RequestScopedWorkerGroupCleanupReport:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE)

    if (
        type(report.cooperative_shutdown_completed) is not bool
        or type(report.terminate_required) is not bool
        or type(report.kill_required) is not bool
        or type(report.process_exit_codes) is not tuple
        or type(report.secondary_failures) is not tuple
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE)

    if any(
        exit_code is not None and type(exit_code) is not int
        for exit_code in report.process_exit_codes
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE)

    if any(
        type(failure) is not RequestScopedWorkerGroupCleanupFailureCode
        for failure in report.secondary_failures
    ):
        _raise_group_error(RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE)

    return report


def _copy_group_epoch_results(
    results: object,
    *,
    expected_shards: tuple[LogicalTrainingShard, ...],
    layout: TransformerParameterLayout,
) -> tuple[LogicalTrainingShardResult, ...]:
    if type(results) is not tuple or len(results) != LOGICAL_TRAINING_SHARD_COUNT:
        _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

    copied_results: list[LogicalTrainingShardResult] = []

    for result, expected_shard in zip(
        results,
        expected_shards,
        strict=True,
    ):
        if type(result) is not LogicalTrainingShardResult:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        validated_result = result

        try:
            source = _validate_shard_result(
                validated_result,
                expected_shard=expected_shard,
                layout=layout,
            )
        except _WorkerExecutionError as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
            ) from exc

        copied_gradient = create_transformer_gradient_buffer(layout)

        try:
            np.copyto(
                copied_gradient.storage,
                source,
                casting="no",
            )
        except (TypeError, ValueError) as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
            ) from exc

        if not np.isfinite(copied_gradient.storage).all():
            _raise_group_error(RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE)

        copied_results.append(
            LogicalTrainingShardResult(
                shard=expected_shard,
                processed_sequence_count=validated_result.processed_sequence_count,
                loss=float(validated_result.loss),
                gradient=copied_gradient,
            )
        )

    return tuple(copied_results)


class RequestScopedWorkerGroup:
    """Parent-owned lifecycle boundary for one Transformer worker group."""

    __slots__ = (
        "_actual_worker_count",
        "_assigned_shard_ids_by_worker",
        "_cleanup_report",
        "_initial_weights",
        "_layout",
        "_logical_shards",
        "_poll_observer",
        "_primary_failure_code",
        "_runtime",
        "_startup_completed",
        "_state",
        "_training_sequences",
    )

    def __init__(
        self,
        num_layers: int,
        current_weights: _Float32Array,
        training_sequences: tuple[TransformerTrainingSequence, ...],
        logical_shards: tuple[LogicalTrainingShard, ...],
        *,
        poll_observer: _WorkerGroupPollObserver | None = None,
        _runtime: _RequestScopedWorkerGroupRuntime | None = None,
    ) -> None:
        if (
            type(num_layers) is not int
            or num_layers < TRANSFORMER_MIN_LAYER_COUNT
            or num_layers > TRANSFORMER_MAX_LAYER_COUNT
        ):
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_INPUT)

        try:
            layout = build_transformer_parameter_layout(num_layers)
        except (TypeError, ValueError) as exc:
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.INVALID_INPUT,
            ) from exc

        validated_sequences = _validate_group_training_sequences(
            training_sequences,
            vocabulary_size=layout.vocabulary_size,
        )
        validated_shards = _validate_group_logical_shards(
            logical_shards,
            sequence_count=len(validated_sequences),
        )

        actual_worker_count = calculate_actual_worker_count(os.cpu_count())
        assigned_shard_ids_by_worker = build_worker_shard_assignments(
            actual_worker_count,
        )

        self._layout = layout
        self._initial_weights: _Float32Array | None = _canonical_group_weight_snapshot(
            current_weights,
            layout=layout,
        )
        self._training_sequences = validated_sequences
        self._logical_shards = validated_shards
        self._actual_worker_count = actual_worker_count
        self._assigned_shard_ids_by_worker = assigned_shard_ids_by_worker
        self._poll_observer = _validate_group_poll_observer(poll_observer)
        self._runtime = _validate_group_runtime(
            _SharedMemoryRequestScopedWorkerGroupRuntime() if _runtime is None else _runtime
        )
        self._startup_completed = False
        self._state = RequestScopedWorkerGroupState.ALLOCATED
        self._cleanup_report: RequestScopedWorkerGroupCleanupReport | None = None
        self._primary_failure_code: RequestScopedWorkerGroupFailureCode | None = None

    @property
    def state(self) -> RequestScopedWorkerGroupState:
        """Return the current stable parent-side lifecycle state."""
        return self._state

    @property  #  Added Code
    def actual_worker_count(self) -> int:  #  Added Code
        """Return the worker-process count selected for this training run."""
        return self._actual_worker_count  #  Added Code

    @property
    def cleanup_report(self) -> RequestScopedWorkerGroupCleanupReport | None:
        """Return the published immutable cleanup report, when available."""
        return self._cleanup_report

    @property
    def primary_failure_code(
        self,
    ) -> RequestScopedWorkerGroupFailureCode | None:
        """Return the first run failure without replacing it with cleanup failures."""
        return self._primary_failure_code

    @property
    def successful(self) -> bool:
        """Return whether the group closed without run, forced, or cleanup failure."""
        return (
            self._startup_completed
            and self._state is RequestScopedWorkerGroupState.CLOSED
            and self._primary_failure_code is None
            and self._cleanup_report is not None
            and self._cleanup_report.successful
        )

    def _require_state(
        self,
        required: RequestScopedWorkerGroupState,
    ) -> None:
        if self._state is not required:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_STATE)

    def _record_primary_failure(
        self,
        code: RequestScopedWorkerGroupFailureCode,
    ) -> None:
        if self._primary_failure_code is None:
            self._primary_failure_code = code

        if self._state not in (
            RequestScopedWorkerGroupState.STOPPING,
            RequestScopedWorkerGroupState.CLOSED,
        ):
            self._state = RequestScopedWorkerGroupState.FAILED

    async def _start(self) -> None:
        self._require_state(RequestScopedWorkerGroupState.ALLOCATED)
        self._state = RequestScopedWorkerGroupState.STARTING
        initial_weights = self._initial_weights

        if initial_weights is None:
            self._record_primary_failure(
                RequestScopedWorkerGroupFailureCode.INTERNAL_FAILURE,
            )
            report = await self.cleanup()
            raise RequestScopedWorkerGroupError(
                RequestScopedWorkerGroupFailureCode.INTERNAL_FAILURE,
                cleanup_report=report,
            )

        try:
            await self._runtime.start(
                num_layers=self._layout.num_layers,
                current_weights=initial_weights,
                training_sequences=self._training_sequences,
                logical_shards=self._logical_shards,
                actual_worker_count=self._actual_worker_count,
                assigned_shard_ids_by_worker=(self._assigned_shard_ids_by_worker),
                poll_observer=self._poll_observer,
            )
        except asyncio.CancelledError:
            self._record_primary_failure(
                RequestScopedWorkerGroupFailureCode.CANCELLED,
            )
            self._initial_weights = None
            await self.cleanup()
            raise
        except RequestScopedWorkerGroupError as exc:
            self._record_primary_failure(exc.code)
            self._initial_weights = None
            report = await self.cleanup()
            raise RequestScopedWorkerGroupError(
                exc.code,
                cleanup_report=report,
            ) from exc
        except Exception as exc:
            code = RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE
            self._record_primary_failure(code)
            self._initial_weights = None
            report = await self.cleanup()
            raise RequestScopedWorkerGroupError(
                code,
                cleanup_report=report,
            ) from exc

        self._initial_weights = None
        self._startup_completed = True
        self._state = RequestScopedWorkerGroupState.READY

    async def compute_epoch(
        self,
        epoch: int,
        current_weights: _Float32Array,
    ) -> tuple[LogicalTrainingShardResult, ...]:
        """Compute one non-overlapping epoch and return independent shard results."""
        self._require_state(RequestScopedWorkerGroupState.READY)
        validated_epoch = _validate_group_epoch(epoch)
        weight_snapshot = _canonical_group_weight_snapshot(
            current_weights,
            layout=self._layout,
        )
        self._state = RequestScopedWorkerGroupState.COMPUTING

        try:
            raw_results = await self._runtime.compute_epoch(
                epoch=validated_epoch,
                current_weights=weight_snapshot,
                poll_observer=self._poll_observer,
            )
            copied_results = _copy_group_epoch_results(
                raw_results,
                expected_shards=self._logical_shards,
                layout=self._layout,
            )
        except asyncio.CancelledError:
            self._record_primary_failure(
                RequestScopedWorkerGroupFailureCode.CANCELLED,
            )
            await self.cleanup()
            raise
        except RequestScopedWorkerGroupError as exc:
            self._record_primary_failure(exc.code)
            raise
        except Exception as exc:
            code = RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
            self._record_primary_failure(code)
            raise RequestScopedWorkerGroupError(code) from exc

        if self._state is not RequestScopedWorkerGroupState.COMPUTING:
            code = RequestScopedWorkerGroupFailureCode.CANCELLED
            self._record_primary_failure(code)
            raise RequestScopedWorkerGroupError(code)

        self._state = RequestScopedWorkerGroupState.READY
        return copied_results

    async def cleanup(
        self,
    ) -> RequestScopedWorkerGroupCleanupReport:
        """Idempotently clean the group and publish secondary diagnostics."""
        if self._cleanup_report is not None:
            return self._cleanup_report

        if self._state is RequestScopedWorkerGroupState.STOPPING:
            _raise_group_error(RequestScopedWorkerGroupFailureCode.INVALID_STATE)

        self._state = RequestScopedWorkerGroupState.STOPPING
        cleanup_task = asyncio.create_task(self._runtime.cleanup())
        cancellation: asyncio.CancelledError | None = None

        try:
            raw_report = await asyncio.shield(cleanup_task)
            report = _validate_group_cleanup_report(raw_report)
        except asyncio.CancelledError as exc:
            cancellation = exc
            self._record_primary_failure(
                RequestScopedWorkerGroupFailureCode.CANCELLED,
            )

            try:
                raw_report = await cleanup_task
                report = _validate_group_cleanup_report(raw_report)
            except Exception:
                report = RequestScopedWorkerGroupCleanupReport(
                    cooperative_shutdown_completed=False,
                    terminate_required=False,
                    kill_required=False,
                    process_exit_codes=(),
                    secondary_failures=(RequestScopedWorkerGroupCleanupFailureCode.INTERNAL,),
                )
        except Exception:
            report = RequestScopedWorkerGroupCleanupReport(
                cooperative_shutdown_completed=False,
                terminate_required=False,
                kill_required=False,
                process_exit_codes=(),
                secondary_failures=(RequestScopedWorkerGroupCleanupFailureCode.INTERNAL,),
            )

        self._cleanup_report = report

        if not report.successful and self._primary_failure_code is None:
            self._primary_failure_code = RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE

        self._initial_weights = None
        self._state = RequestScopedWorkerGroupState.CLOSED

        if cancellation is not None:
            raise cancellation

        return report


async def create_request_scoped_worker_group(
    num_layers: int,
    current_weights: _Float32Array,
    training_sequences: tuple[TransformerTrainingSequence, ...],
    logical_shards: tuple[LogicalTrainingShard, ...],
    *,
    poll_observer: _WorkerGroupPollObserver | None = None,
    _runtime: _RequestScopedWorkerGroupRuntime | None = None,
) -> RequestScopedWorkerGroup:
    """Create and completely start one independent request-owned worker group."""
    group = RequestScopedWorkerGroup(
        num_layers,
        current_weights,
        training_sequences,
        logical_shards,
        poll_observer=poll_observer,
        _runtime=_runtime,
    )
    await group._start()
    return group


class _WorkerExecutionError(RuntimeError):
    """Internal sanitized failure classification used by the worker target."""

    __slots__ = ("phase", "code", "epoch")

    def __init__(
        self,
        phase: WorkerFailurePhase,
        code: WorkerFailureCode,
        *,
        epoch: int | None = None,
    ) -> None:
        super().__init__("Transformer worker execution failed.")
        self.phase = phase
        self.code = code
        self.epoch = epoch


def _raise_protocol_error(
    code: WorkerFailureCode = WorkerFailureCode.INVALID_PROTOCOL,
) -> NoReturn:
    raise WorkerProtocolValidationError(code)


def _require_exact_int(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _raise_protocol_error()
    return value


def _require_exact_tuple(value: object) -> tuple[object, ...]:
    if type(value) is not tuple:
        _raise_protocol_error()
    return value


def _validate_protocol_version(protocol_version: object) -> None:
    if type(protocol_version) is not int or protocol_version != WORKER_PROTOCOL_VERSION:
        _raise_protocol_error()


def _validate_worker_identity(worker_index: object, expected_worker_index: int) -> int:
    validated_expected = _require_exact_int(expected_worker_index)
    validated_worker = _require_exact_int(worker_index)

    if validated_worker != validated_expected:
        _raise_protocol_error()

    return validated_worker


def _validate_shard_ids(
    shard_ids: object,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    values = _require_exact_tuple(shard_ids)
    validated: list[int] = []

    for value in values:
        shard_id = _require_exact_int(value)

        if shard_id >= LOGICAL_TRAINING_SHARD_COUNT:
            _raise_protocol_error()

        validated.append(shard_id)

    result = tuple(validated)

    if not allow_empty and not result:
        _raise_protocol_error()

    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        _raise_protocol_error()

    return result


def _validate_expected_shard_ids(expected_shard_ids: object) -> tuple[int, ...]:
    return _validate_shard_ids(
        expected_shard_ids,
        allow_empty=True,
    )


def _validate_epoch(epoch: object) -> int:
    return _require_exact_int(epoch)


def _validate_protocol_float(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        _raise_protocol_error()

    return value


def _validate_shared_memory_name(value: object) -> str:
    if type(value) is not str or not value:
        _raise_protocol_error()

    return value


def _validate_training_sequences(
    payload: object,
    *,
    expected_sequence_count: int,
    vocabulary_size: int,
) -> tuple[_SequencePayload, ...]:
    sequence_values = _require_exact_tuple(payload)

    if len(sequence_values) != expected_sequence_count:
        _raise_protocol_error()

    validated_sequences: list[_SequencePayload] = []

    for sequence_value in sequence_values:
        pair = _require_exact_tuple(sequence_value)

        if len(pair) != 2:
            _raise_protocol_error()

        input_values = _require_exact_tuple(pair[0])
        target_values = _require_exact_tuple(pair[1])

        if (
            len(input_values) != TRANSFORMER_SEQUENCE_LENGTH
            or len(target_values) != TRANSFORMER_SEQUENCE_LENGTH
        ):
            _raise_protocol_error()

        validated_input: list[int] = []
        validated_target: list[int] = []

        for token_id in input_values:
            validated_id = _require_exact_int(token_id)

            if validated_id >= vocabulary_size:
                _raise_protocol_error()

            validated_input.append(validated_id)

        for token_id in target_values:
            validated_id = _require_exact_int(token_id)

            if validated_id >= vocabulary_size:
                _raise_protocol_error()

            validated_target.append(validated_id)

        validated_sequences.append(
            (
                tuple(validated_input),
                tuple(validated_target),
            )
        )

    return tuple(validated_sequences)


def _validate_assigned_shards(
    payload: object,
    *,
    sequence_count: int,
) -> tuple[_ShardPayload, ...]:
    shard_values = _require_exact_tuple(payload)

    if not shard_values:
        _raise_protocol_error()

    canonical_shards = build_logical_training_shards(sequence_count)
    validated_shards: list[_ShardPayload] = []

    for shard_value in shard_values:
        descriptor = _require_exact_tuple(shard_value)

        if len(descriptor) != 3:
            _raise_protocol_error()

        shard_id = _require_exact_int(descriptor[0])
        start_index = _require_exact_int(descriptor[1])
        stop_index = _require_exact_int(descriptor[2])

        if shard_id >= LOGICAL_TRAINING_SHARD_COUNT or stop_index < start_index:
            _raise_protocol_error()

        canonical = canonical_shards[shard_id]
        candidate = (shard_id, start_index, stop_index)
        expected = (
            canonical.shard_index,
            canonical.start_index,
            canonical.stop_index,
        )

        if candidate != expected:
            _raise_protocol_error()

        validated_shards.append(candidate)

    result = tuple(validated_shards)
    shard_ids = tuple(shard[0] for shard in result)

    if shard_ids != tuple(sorted(shard_ids)) or len(set(shard_ids)) != len(shard_ids):
        _raise_protocol_error()

    return result


def validate_worker_startup_config(
    config: object,
    *,
    expected_worker_index: int,
) -> WorkerStartupConfig:
    """Validate one exact startup record at the worker trust boundary."""
    if type(config) is not WorkerStartupConfig:
        _raise_protocol_error()

    validated = config
    _validate_protocol_version(validated.protocol_version)
    _validate_worker_identity(validated.worker_index, expected_worker_index)

    num_layers = _require_exact_int(validated.num_layers)

    if num_layers < TRANSFORMER_MIN_LAYER_COUNT or num_layers > TRANSFORMER_MAX_LAYER_COUNT:
        _raise_protocol_error(WorkerFailureCode.INVALID_LAYOUT)

    if type(validated.dtype_marker) is not WorkerNumericDType:
        _raise_protocol_error(WorkerFailureCode.INVALID_LAYOUT)

    if validated.dtype_marker is not WorkerNumericDType.FLOAT32:
        _raise_protocol_error(WorkerFailureCode.INVALID_LAYOUT)

    sequence_count = _require_exact_int(validated.sequence_count)
    canonical_float_count = _require_exact_int(
        validated.canonical_float_count,
        minimum=1,
    )

    try:
        layout = build_transformer_parameter_layout(num_layers)
    except (TypeError, ValueError):
        _raise_protocol_error(WorkerFailureCode.INVALID_LAYOUT)

    if canonical_float_count != layout.total_float_count:
        _raise_protocol_error(WorkerFailureCode.INVALID_LAYOUT)

    weight_name = _validate_shared_memory_name(validated.weight_shared_memory_name)

    try:
        sequences = _validate_training_sequences(
            validated.training_sequences,
            expected_sequence_count=sequence_count,
            vocabulary_size=layout.vocabulary_size,
        )
    except WorkerProtocolValidationError as exc:
        raise WorkerProtocolValidationError(WorkerFailureCode.INVALID_LAYOUT) from exc

    try:
        assigned_shards = _validate_assigned_shards(
            validated.assigned_shards,
            sequence_count=sequence_count,
        )
    except WorkerProtocolValidationError as exc:
        raise WorkerProtocolValidationError(WorkerFailureCode.INVALID_OWNERSHIP) from exc

    gradient_names_raw = _require_exact_tuple(validated.gradient_shared_memory_names)
    gradient_names = tuple(_validate_shared_memory_name(name) for name in gradient_names_raw)

    if len(gradient_names) != len(assigned_shards):
        _raise_protocol_error(WorkerFailureCode.INVALID_OWNERSHIP)

    if len(set(gradient_names)) != len(gradient_names):
        _raise_protocol_error(WorkerFailureCode.INVALID_OWNERSHIP)

    if weight_name in gradient_names:
        _raise_protocol_error(WorkerFailureCode.INVALID_OWNERSHIP)

    if (
        validated.num_layers != num_layers
        or validated.canonical_float_count != canonical_float_count
        or validated.sequence_count != sequence_count
        or validated.weight_shared_memory_name != weight_name
        or validated.assigned_shards != assigned_shards
        or validated.gradient_shared_memory_names != gradient_names
        or validated.training_sequences != sequences
    ):
        _raise_protocol_error()

    return validated


def validate_ready_message(
    message: object,
    *,
    expected_worker_index: int,
    expected_shard_ids: tuple[int, ...],
) -> ReadyMessage:
    """Validate one exact ready record before parent startup commit."""
    if type(message) is not ReadyMessage:
        _raise_protocol_error()

    validated = message
    _validate_protocol_version(validated.protocol_version)
    _validate_worker_identity(
        validated.worker_index,
        expected_worker_index,
    )
    actual_shard_ids = _validate_shard_ids(
        validated.assigned_shard_ids,
        allow_empty=False,
    )
    expected = _validate_expected_shard_ids(expected_shard_ids)

    if actual_shard_ids != expected:
        _raise_protocol_error()

    return validated


def validate_compute_message(
    message: object,
    *,
    expected_worker_index: int,
    expected_shard_ids: tuple[int, ...],
) -> ComputeMessage:
    """Validate one exact compute command at the worker trust boundary."""
    if type(message) is not ComputeMessage:
        _raise_protocol_error()

    validated = message
    _validate_protocol_version(validated.protocol_version)
    _validate_worker_identity(
        validated.worker_index,
        expected_worker_index,
    )
    _validate_epoch(validated.epoch)
    actual_shard_ids = _validate_shard_ids(
        validated.assigned_shard_ids,
        allow_empty=False,
    )
    expected = _validate_expected_shard_ids(expected_shard_ids)

    if actual_shard_ids != expected:
        _raise_protocol_error()

    return validated


def validate_result_message(
    message: object,
    *,
    expected_worker_index: int,
    expected_epoch: int,
    expected_shard_ids: tuple[int, ...],
) -> ResultMessage:
    """Validate one complete result commit marker before reading shared buffers."""
    if type(message) is not ResultMessage:
        _raise_protocol_error()

    validated = message
    _validate_protocol_version(validated.protocol_version)
    _validate_worker_identity(
        validated.worker_index,
        expected_worker_index,
    )
    expected_validated_epoch = _validate_epoch(expected_epoch)
    actual_epoch = _validate_epoch(validated.epoch)

    if actual_epoch != expected_validated_epoch:
        _raise_protocol_error()

    actual_shard_ids = _validate_shard_ids(
        validated.assigned_shard_ids,
        allow_empty=False,
    )
    expected = _validate_expected_shard_ids(expected_shard_ids)

    if actual_shard_ids != expected:
        _raise_protocol_error()

    losses_raw = _require_exact_tuple(validated.shard_losses)
    losses = tuple(_validate_protocol_float(loss) for loss in losses_raw)

    if len(losses) != len(actual_shard_ids) or validated.shard_losses != losses:
        _raise_protocol_error()

    return validated


def validate_failure_message(
    message: object,
    *,
    expected_worker_index: int,
    expected_shard_ids: tuple[int, ...],
) -> FailureMessage:
    """Validate one sanitized worker failure record."""
    if type(message) is not FailureMessage:
        _raise_protocol_error()

    validated = message
    _validate_protocol_version(validated.protocol_version)
    _validate_worker_identity(
        validated.worker_index,
        expected_worker_index,
    )

    if type(validated.phase) is not WorkerFailurePhase:
        _raise_protocol_error()

    if type(validated.code) is not WorkerFailureCode:
        _raise_protocol_error()

    if validated.epoch is not None:
        _validate_epoch(validated.epoch)

    actual_shard_ids = _validate_shard_ids(
        validated.assigned_shard_ids,
        allow_empty=True,
    )
    expected = _validate_expected_shard_ids(expected_shard_ids)

    if actual_shard_ids != expected:
        _raise_protocol_error()

    return validated


def validate_stop_message(
    message: object,
    *,
    expected_worker_index: int,
) -> StopMessage:
    """Validate one cooperative stop command."""
    if type(message) is not StopMessage:
        _raise_protocol_error()

    validated = message
    _validate_protocol_version(validated.protocol_version)
    _validate_worker_identity(
        validated.worker_index,
        expected_worker_index,
    )
    return validated


def validate_stopped_message(
    message: object,
    *,
    expected_worker_index: int,
) -> StoppedMessage:
    """Validate one clean worker shutdown marker."""
    if type(message) is not StoppedMessage:
        _raise_protocol_error()

    validated = message
    _validate_protocol_version(validated.protocol_version)
    _validate_worker_identity(
        validated.worker_index,
        expected_worker_index,
    )
    return validated


def _iter_parameter_arrays(
    views: TransformerParameterViews,
) -> tuple[_Float32Array, ...]:
    arrays: list[_Float32Array] = [
        views.tok_emb,
        views.pos_emb,
    ]

    for block in views.blocks:
        arrays.extend(
            [
                block.ln1_gamma,
                block.ln1_beta,
                block.w_q,
                block.b_q,
                block.w_k,
                block.b_k,
                block.w_v,
                block.b_v,
                block.w_o,
                block.b_o,
                block.ln2_gamma,
                block.ln2_beta,
                block.ff1_w,
                block.ff1_b,
                block.ff2_w,
                block.ff2_b,
            ]
        )

    arrays.extend(
        [
            views.ln_f_gamma,
            views.ln_f_beta,
            views.head_w,
            views.head_b,
        ]
    )

    return tuple(arrays)


def _validate_flat_shared_array(
    array: _Float32Array,
    *,
    layout: TransformerParameterLayout,
    require_writable: bool,
) -> None:
    if (
        type(array) is not np.ndarray
        or array.dtype != np.dtype(np.float32)
        or array.shape != (layout.total_float_count,)
        or not array.flags.c_contiguous
        or array.flags.writeable is not require_writable
    ):
        raise _WorkerExecutionError(
            WorkerFailurePhase.STARTUP,
            WorkerFailureCode.INVALID_LAYOUT,
        )


def _validate_parameter_view_writability(
    views: TransformerParameterViews,
    *,
    require_writable: bool,
) -> None:
    for array in _iter_parameter_arrays(views):
        if array.flags.writeable is not require_writable:
            raise _WorkerExecutionError(
                WorkerFailurePhase.STARTUP,
                WorkerFailureCode.INVALID_LAYOUT,
            )


def _probe_weight_read_only_enforcement(
    storage: _Float32Array,
    views: TransformerParameterViews,
) -> None:
    first_view = _iter_parameter_arrays(views)[0]
    before = storage.tobytes(order="C")
    original = np.float32(first_view.reshape(-1, order="C")[0])

    try:
        first_view.reshape(-1, order="C")[0] = original
    except ValueError:
        pass
    else:
        raise _WorkerExecutionError(
            WorkerFailurePhase.STARTUP,
            WorkerFailureCode.INVALID_OWNERSHIP,
        )

    if storage.tobytes(order="C") != before:
        raise _WorkerExecutionError(
            WorkerFailurePhase.STARTUP,
            WorkerFailureCode.INVALID_OWNERSHIP,
        )


def _reconstruct_sequences(
    config: WorkerStartupConfig,
) -> tuple[TransformerTrainingSequence, ...]:
    return tuple(
        TransformerTrainingSequence(
            input_ids=input_ids,
            target_ids=target_ids,
        )
        for input_ids, target_ids in config.training_sequences
    )


def _reconstruct_shards(
    config: WorkerStartupConfig,
) -> tuple[LogicalTrainingShard, ...]:
    return tuple(
        LogicalTrainingShard(
            shard_index=shard_id,
            start_index=start_index,
            stop_index=stop_index,
        )
        for shard_id, start_index, stop_index in config.assigned_shards
    )


@dataclass(slots=True)
class _AttachedWorkerResources:
    layout: TransformerParameterLayout
    sequences: tuple[TransformerTrainingSequence, ...]
    shards: tuple[LogicalTrainingShard, ...]
    weight_shared_memory: SharedMemory | None
    gradient_shared_memories: tuple[SharedMemory, ...]
    weight_buffer: memoryview | None
    gradient_buffers: tuple[memoryview, ...]
    weight_storage: _Float32Array | None
    gradient_storages: dict[int, _Float32Array]
    weight_views: TransformerParameterViews | None
    gradient_views: dict[int, TransformerParameterViews]

    def close(self) -> None:
        """Drop exported views before closing every worker-owned handle."""
        self.weight_views = None
        self.gradient_views.clear()
        self.weight_storage = None
        self.gradient_storages.clear()
        gc.collect()

        buffers = self.gradient_buffers
        weight_buffer = self.weight_buffer
        self.gradient_buffers = ()
        self.weight_buffer = None

        had_error = False

        for buffer in buffers:
            try:
                buffer.release()
            except (BufferError, ValueError):
                had_error = True

        if weight_buffer is not None:
            try:
                weight_buffer.release()
            except (BufferError, ValueError):
                had_error = True

        gradient_handles = self.gradient_shared_memories
        weight_handle = self.weight_shared_memory
        self.gradient_shared_memories = ()
        self.weight_shared_memory = None

        for handle in gradient_handles:
            try:
                handle.close()
            except (BufferError, OSError):
                had_error = True

        if weight_handle is not None:
            try:
                weight_handle.close()
            except (BufferError, OSError):
                had_error = True

        if had_error:
            raise _WorkerExecutionError(
                WorkerFailurePhase.SHUTDOWN,
                WorkerFailureCode.CLEANUP_FAILURE,
            )


def _new_empty_attached_resources(
    layout: TransformerParameterLayout,
    sequences: tuple[TransformerTrainingSequence, ...],
    shards: tuple[LogicalTrainingShard, ...],
) -> _AttachedWorkerResources:
    return _AttachedWorkerResources(
        layout=layout,
        sequences=sequences,
        shards=shards,
        weight_shared_memory=None,
        gradient_shared_memories=(),
        weight_buffer=None,
        gradient_buffers=(),
        weight_storage=None,
        gradient_storages={},
        weight_views=None,
        gradient_views={},
    )


def _attach_shared_array(
    name: str,
    *,
    layout: TransformerParameterLayout,
) -> tuple[SharedMemory, memoryview, _Float32Array]:
    try:
        handle = SharedMemory(
            name=name,
            create=False,
        )
    except (FileNotFoundError, OSError) as exc:
        raise _WorkerExecutionError(
            WorkerFailurePhase.STARTUP,
            WorkerFailureCode.SHARED_MEMORY,
        ) from exc

    if handle.size < layout.total_byte_count:
        try:
            handle.close()
        except (BufferError, OSError):
            pass

        raise _WorkerExecutionError(
            WorkerFailurePhase.STARTUP,
            WorkerFailureCode.SHARED_MEMORY,
        )

    shared_buffer = handle.buf

    if shared_buffer is None:
        try:
            handle.close()
        except (BufferError, OSError):
            pass

        raise _WorkerExecutionError(
            WorkerFailurePhase.STARTUP,
            WorkerFailureCode.SHARED_MEMORY,
        )

    exact_buffer = shared_buffer[: layout.total_byte_count]

    try:
        array = np.ndarray(
            (layout.total_float_count,),
            dtype=np.float32,
            buffer=exact_buffer,
            order="C",
        )
    except Exception as exc:
        try:
            exact_buffer.release()
        except (BufferError, ValueError):
            pass

        try:
            handle.close()
        except (BufferError, OSError):
            pass

        raise _WorkerExecutionError(
            WorkerFailurePhase.STARTUP,
            WorkerFailureCode.SHARED_MEMORY,
        ) from exc

    return handle, exact_buffer, cast(_Float32Array, array)


def _attach_worker_resources(
    config: WorkerStartupConfig,
) -> _AttachedWorkerResources:
    layout = build_transformer_parameter_layout(config.num_layers)
    sequences = _reconstruct_sequences(config)
    shards = _reconstruct_shards(config)
    resources = _new_empty_attached_resources(
        layout,
        sequences,
        shards,
    )
    weight_buffer: memoryview | None = None
    weight_storage: _Float32Array | None = None
    weight_views: TransformerParameterViews | None = None
    gradient_buffer: memoryview | None = None
    gradient_storage: _Float32Array | None = None
    gradient_views: TransformerParameterViews | None = None

    try:
        weight_handle, weight_buffer, weight_storage = _attach_shared_array(
            config.weight_shared_memory_name,
            layout=layout,
        )
        resources.weight_shared_memory = weight_handle
        resources.weight_buffer = weight_buffer
        resources.weight_storage = weight_storage

        weight_storage.setflags(write=False)
        _validate_flat_shared_array(
            weight_storage,
            layout=layout,
            require_writable=False,
        )

        if not np.isfinite(weight_storage).all():
            raise _WorkerExecutionError(
                WorkerFailurePhase.STARTUP,
                WorkerFailureCode.NUMERICAL_FAILURE,
            )

        weight_views = build_transformer_parameter_views(
            weight_storage,
            layout,
        )
        resources.weight_views = weight_views
        _validate_parameter_view_writability(
            weight_views,
            require_writable=False,
        )
        _probe_weight_read_only_enforcement(
            weight_storage,
            weight_views,
        )

        gradient_handles: list[SharedMemory] = []
        gradient_buffers: list[memoryview] = []

        for shard, name in zip(
            shards,
            config.gradient_shared_memory_names,
            strict=True,
        ):
            (
                gradient_handle,
                gradient_buffer,
                gradient_storage,
            ) = _attach_shared_array(
                name,
                layout=layout,
            )
            gradient_handles.append(gradient_handle)
            gradient_buffers.append(gradient_buffer)
            resources.gradient_shared_memories = tuple(gradient_handles)
            resources.gradient_buffers = tuple(gradient_buffers)
            resources.gradient_storages[shard.shard_index] = gradient_storage

            _validate_flat_shared_array(
                gradient_storage,
                layout=layout,
                require_writable=True,
            )
            gradient_views = build_transformer_parameter_views(
                gradient_storage,
                layout,
            )
            resources.gradient_views[shard.shard_index] = gradient_views
            _validate_parameter_view_writability(
                gradient_views,
                require_writable=True,
            )

        if tuple(resources.gradient_storages) != tuple(shard.shard_index for shard in shards):
            raise _WorkerExecutionError(
                WorkerFailurePhase.STARTUP,
                WorkerFailureCode.INVALID_OWNERSHIP,
            )

        return resources
    except Exception:
        weight_buffer = None
        weight_storage = None
        weight_views = None
        gradient_buffer = None
        gradient_storage = None
        gradient_views = None
        gc.collect()

        try:
            resources.close()
        except _WorkerExecutionError:
            pass

        raise


def _validate_shard_result(
    result: LogicalTrainingShardResult,
    *,
    expected_shard: LogicalTrainingShard,
    layout: TransformerParameterLayout,
) -> _Float32Array:
    if type(result) is not LogicalTrainingShardResult:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.INVALID_LAYOUT,
        )

    if result.shard != expected_shard:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.INVALID_OWNERSHIP,
        )

    expected_sequence_count = expected_shard.stop_index - expected_shard.start_index

    if result.processed_sequence_count != expected_sequence_count:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.INVALID_LAYOUT,
        )

    if type(result.loss) is not float or not math.isfinite(result.loss):
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.NUMERICAL_FAILURE,
        )

    if result.gradient.layout != layout:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.INVALID_LAYOUT,
        )

    storage = result.gradient.storage

    if (
        type(storage) is not np.ndarray
        or storage.dtype != np.dtype(np.float32)
        or storage.shape != (layout.total_float_count,)
        or not storage.flags.c_contiguous
        or not np.isfinite(storage).all()
    ):
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.NUMERICAL_FAILURE,
        )

    return storage


def _raise_if_command_is_pending(
    connection: Connection,
    *,
    epoch: int,
) -> None:
    try:
        pending = connection.poll()
    except (OSError, EOFError) as exc:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.COMMUNICATION_FAILURE,
            epoch=epoch,
        ) from exc

    if pending:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMMAND,
            WorkerFailureCode.INVALID_STATE,
            epoch=epoch,
        )


def _compute_assigned_shards(
    connection: Connection,
    command: ComputeMessage,
    resources: _AttachedWorkerResources,
) -> ResultMessage:
    weight_views = resources.weight_views

    if weight_views is None:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.INVALID_STATE,
            epoch=command.epoch,
        )

    losses: list[float] = []

    _raise_if_command_is_pending(
        connection,
        epoch=command.epoch,
    )

    for shard in resources.shards:
        destination = resources.gradient_storages.get(shard.shard_index)

        if destination is None:
            raise _WorkerExecutionError(
                WorkerFailurePhase.COMPUTE,
                WorkerFailureCode.INVALID_OWNERSHIP,
                epoch=command.epoch,
            )

        destination.fill(np.float32(0.0))

        try:
            shard_result = calculate_logical_training_shard(
                resources.sequences,
                shard,
                weight_views,
            )
        except (
            TypeError,
            ValueError,
            RuntimeError,
            FloatingPointError,
        ) as exc:
            raise _WorkerExecutionError(
                WorkerFailurePhase.COMPUTE,
                WorkerFailureCode.NUMERICAL_FAILURE,
                epoch=command.epoch,
            ) from exc

        source = _validate_shard_result(
            shard_result,
            expected_shard=shard,
            layout=resources.layout,
        )

        try:
            np.copyto(
                destination,
                source,
                casting="no",
            )
        except (TypeError, ValueError) as exc:
            raise _WorkerExecutionError(
                WorkerFailurePhase.COMPUTE,
                WorkerFailureCode.INVALID_LAYOUT,
                epoch=command.epoch,
            ) from exc

        if not np.isfinite(destination).all():
            raise _WorkerExecutionError(
                WorkerFailurePhase.COMPUTE,
                WorkerFailureCode.NUMERICAL_FAILURE,
                epoch=command.epoch,
            )

        losses.append(float(shard_result.loss))

        _raise_if_command_is_pending(
            connection,
            epoch=command.epoch,
        )

    for shard in resources.shards:
        published = resources.gradient_storages[shard.shard_index]

        if not np.isfinite(published).all():
            raise _WorkerExecutionError(
                WorkerFailurePhase.COMPUTE,
                WorkerFailureCode.NUMERICAL_FAILURE,
                epoch=command.epoch,
            )

    _raise_if_command_is_pending(
        connection,
        epoch=command.epoch,
    )

    return ResultMessage(
        protocol_version=WORKER_PROTOCOL_VERSION,
        worker_index=command.worker_index,
        epoch=command.epoch,
        assigned_shard_ids=command.assigned_shard_ids,
        shard_losses=tuple(losses),
    )


def _send_message(
    connection: Connection,
    message: ReadyMessage | ResultMessage | StoppedMessage,
    *,
    phase: WorkerFailurePhase,
    epoch: int | None = None,
) -> None:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError) as exc:
        raise _WorkerExecutionError(
            phase,
            WorkerFailureCode.COMMUNICATION_FAILURE,
            epoch=epoch,
        ) from exc


def _receive_command(connection: Connection) -> object:
    try:
        return cast(object, connection.recv())
    except (EOFError, OSError) as exc:
        raise _WorkerExecutionError(
            WorkerFailurePhase.COMMAND,
            WorkerFailureCode.COMMUNICATION_FAILURE,
        ) from exc


def _send_failure_once(
    connection: Connection,
    failure: FailureMessage,
) -> None:
    try:
        connection.send(failure)
    except (BrokenPipeError, EOFError, OSError):
        return


def _failure_for_unexpected_exception(
    state: WorkerState,
    *,
    epoch: int | None,
) -> _WorkerExecutionError:
    if state is WorkerState.STARTING:
        phase = WorkerFailurePhase.STARTUP
    elif state is WorkerState.COMPUTING:
        phase = WorkerFailurePhase.COMPUTE
    elif state is WorkerState.STOPPING:
        phase = WorkerFailurePhase.SHUTDOWN
    else:
        phase = WorkerFailurePhase.COMMAND

    return _WorkerExecutionError(
        phase,
        WorkerFailureCode.INTERNAL_FAILURE,
        epoch=epoch,
    )


def run_transformer_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    """Run one importable spawned worker over one dedicated duplex pipe."""
    worker_index = _require_exact_int(expected_worker_index)
    state = WorkerState.STARTING
    resources: _AttachedWorkerResources | None = None
    assigned_shard_ids: tuple[int, ...] = ()
    current_epoch: int | None = None
    failure: _WorkerExecutionError | None = None
    clean_stop_requested = False

    try:
        try:
            config = validate_worker_startup_config(
                startup_config,
                expected_worker_index=worker_index,
            )
        except WorkerProtocolValidationError as exc:
            raise _WorkerExecutionError(
                WorkerFailurePhase.STARTUP,
                exc.code,
            ) from exc

        assigned_shard_ids = tuple(shard[0] for shard in config.assigned_shards)
        resources = _attach_worker_resources(config)

        _send_message(
            connection,
            ReadyMessage(
                protocol_version=WORKER_PROTOCOL_VERSION,
                worker_index=worker_index,
                assigned_shard_ids=assigned_shard_ids,
            ),
            phase=WorkerFailurePhase.STARTUP,
        )
        state = WorkerState.READY

        while True:
            raw_command = _receive_command(connection)

            if type(raw_command) is StopMessage:
                try:
                    validate_stop_message(
                        raw_command,
                        expected_worker_index=worker_index,
                    )
                except WorkerProtocolValidationError as exc:
                    raise _WorkerExecutionError(
                        WorkerFailurePhase.COMMAND,
                        WorkerFailureCode.INVALID_PROTOCOL,
                    ) from exc

                if state is not WorkerState.READY:
                    raise _WorkerExecutionError(
                        WorkerFailurePhase.COMMAND,
                        WorkerFailureCode.INVALID_STATE,
                    )

                state = WorkerState.STOPPING
                clean_stop_requested = True
                break

            if type(raw_command) is not ComputeMessage:
                raise _WorkerExecutionError(
                    WorkerFailurePhase.COMMAND,
                    WorkerFailureCode.INVALID_PROTOCOL,
                )

            if state is not WorkerState.READY:
                raise _WorkerExecutionError(
                    WorkerFailurePhase.COMMAND,
                    WorkerFailureCode.INVALID_STATE,
                )

            try:
                command = validate_compute_message(
                    raw_command,
                    expected_worker_index=worker_index,
                    expected_shard_ids=assigned_shard_ids,
                )
            except WorkerProtocolValidationError as exc:
                raise _WorkerExecutionError(
                    WorkerFailurePhase.COMMAND,
                    WorkerFailureCode.INVALID_PROTOCOL,
                ) from exc

            state = WorkerState.COMPUTING
            current_epoch = command.epoch

            result = _compute_assigned_shards(
                connection,
                command,
                resources,
            )

            _send_message(
                connection,
                result,
                phase=WorkerFailurePhase.COMPUTE,
                epoch=command.epoch,
            )

            current_epoch = None
            state = WorkerState.READY
    except _WorkerExecutionError as exc:
        failure = exc
    except Exception:
        failure = _failure_for_unexpected_exception(
            state,
            epoch=current_epoch,
        )

    if resources is not None:
        try:
            resources.close()
        except _WorkerExecutionError as cleanup_error:
            if failure is None:
                failure = cleanup_error

    if failure is None and clean_stop_requested:
        try:
            _send_message(
                connection,
                StoppedMessage(
                    protocol_version=WORKER_PROTOCOL_VERSION,
                    worker_index=worker_index,
                ),
                phase=WorkerFailurePhase.SHUTDOWN,
            )
            state = WorkerState.STOPPED
        except _WorkerExecutionError as exc:
            failure = exc

    if failure is not None:
        _send_failure_once(
            connection,
            FailureMessage(
                protocol_version=WORKER_PROTOCOL_VERSION,
                worker_index=worker_index,
                phase=failure.phase,
                code=failure.code,
                epoch=failure.epoch,
                assigned_shard_ids=assigned_shard_ids,
            ),
        )

    try:
        connection.close()
    except OSError:
        if failure is None:
            failure = _WorkerExecutionError(
                WorkerFailurePhase.SHUTDOWN,
                WorkerFailureCode.CLEANUP_FAILURE,
            )

    if failure is not None:
        raise SystemExit(1)

    if state is not WorkerState.STOPPED:
        raise SystemExit(1)
