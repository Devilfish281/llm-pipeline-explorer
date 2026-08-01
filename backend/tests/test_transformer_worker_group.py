# tests/test_transformer_worker_group.py

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import FrozenInstanceError, fields
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from threading import get_ident
from time import sleep
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from how_llms_work.ml import transformer_worker as transformer_worker_module
from how_llms_work.ml.transformer import (
    LogicalTrainingShard,
    LogicalTrainingShardResult,
    TransformerTrainingSequence,
    build_logical_training_shards,
    build_transformer_parameter_layout,
    build_transformer_parameter_views,
    calculate_logical_training_shard,
    create_transformer_gradient_buffer,
)
from how_llms_work.ml.transformer_worker import (
    RequestScopedWorkerGroup,
    RequestScopedWorkerGroupCleanupFailureCode,
    RequestScopedWorkerGroupCleanupReport,
    RequestScopedWorkerGroupError,
    RequestScopedWorkerGroupFailureCode,
    RequestScopedWorkerGroupState,
    build_worker_shard_assignments,
    calculate_actual_worker_count,
    create_request_scoped_worker_group,
)

_Float32Array = npt.NDArray[np.float32]
_PollObserver = Callable[[], Awaitable[None]]


_EXPECTED_PUBLIC_SYMBOLS = (
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
)


def _clean_report() -> RequestScopedWorkerGroupCleanupReport:
    return RequestScopedWorkerGroupCleanupReport(
        cooperative_shutdown_completed=True,
        terminate_required=False,
        kill_required=False,
        process_exit_codes=(),
        secondary_failures=(),
    )


def _group_inputs() -> tuple[
    int,
    _Float32Array,
    tuple[TransformerTrainingSequence, ...],
    tuple[LogicalTrainingShard, ...],
]:
    num_layers = 1
    layout = build_transformer_parameter_layout(num_layers)
    weights = np.zeros(
        layout.total_float_count,
        dtype=np.float32,
        order="C",
    )
    sequence = TransformerTrainingSequence(
        input_ids=tuple(range(16)),
        target_ids=tuple(range(1, 17)),
    )
    sequences = (sequence,)
    shards = build_logical_training_shards(len(sequences))
    return num_layers, weights, sequences, shards


def _direct_shard_results(
    num_layers: int,
    weights: _Float32Array,
    sequences: tuple[TransformerTrainingSequence, ...],
    shards: tuple[LogicalTrainingShard, ...],
) -> tuple[LogicalTrainingShardResult, ...]:
    layout = build_transformer_parameter_layout(num_layers)
    parameters = build_transformer_parameter_views(
        weights,
        layout,
    )
    return tuple(
        calculate_logical_training_shard(
            sequences,
            shard,
            parameters,
        )
        for shard in shards
    )


def _assert_shard_results_match(
    actual_results: tuple[LogicalTrainingShardResult, ...],
    expected_results: tuple[LogicalTrainingShardResult, ...],
) -> None:
    assert tuple(result.shard.shard_index for result in actual_results) == (0, 1, 2, 3)
    assert len(actual_results) == len(expected_results)

    for actual, expected in zip(
        actual_results,
        expected_results,
        strict=True,
    ):
        assert actual.shard == expected.shard
        assert actual.processed_sequence_count == expected.processed_sequence_count
        assert actual.loss == pytest.approx(
            expected.loss,
            rel=1e-6,
            abs=1e-6,
        )
        np.testing.assert_allclose(
            actual.gradient.storage,
            expected.gradient.storage,
            rtol=1e-6,
            atol=1e-6,
        )


def _shard_results(
    shards: tuple[LogicalTrainingShard, ...],
) -> tuple[LogicalTrainingShardResult, ...]:
    layout = build_transformer_parameter_layout(1)
    results: list[LogicalTrainingShardResult] = []

    for shard in shards:
        gradient = create_transformer_gradient_buffer(layout)
        gradient.storage.fill(np.float32(shard.shard_index + 1))
        results.append(
            LogicalTrainingShardResult(
                shard=shard,
                processed_sequence_count=shard.stop_index - shard.start_index,
                loss=float(shard.shard_index),
                gradient=gradient,
            )
        )

    return tuple(results)


async def _observe_poll() -> None:
    return None


class _SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> float:
        if self._index >= len(self._values):
            return self._values[-1]

        value = self._values[self._index]
        self._index += 1
        return value


class _DelegatingWaitSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[float | None, int]] = []

    def __call__(
        self,
        waitables: tuple[
            transformer_worker_module._WorkerWaitable,
            ...,
        ],
        timeout: float | None,
    ) -> list[transformer_worker_module._WorkerWaitable]:
        self.calls.append((timeout, get_ident()))
        return transformer_worker_module._wait_for_worker_connections(
            waitables,
            timeout,
        )


class _CompletionOrderWaiter:
    def __init__(
        self,
        completion_order: tuple[int, ...],
    ) -> None:
        self._completion_order = completion_order
        self._ready_connection_indexes: set[int] = set()
        self.returned_connection_orders: list[tuple[int, ...]] = []

    def __call__(
        self,
        waitables: tuple[
            transformer_worker_module._WorkerWaitable,
            ...,
        ],
        timeout: float | None,
    ) -> list[transformer_worker_module._WorkerWaitable]:
        ready_objects = transformer_worker_module._wait_for_worker_connections(
            waitables,
            timeout,
        )
        connections = tuple(waitable for waitable in waitables if not isinstance(waitable, int))

        if len(connections) != len(self._completion_order):
            return ready_objects

        ready_sentinels = [
            ready_object for ready_object in ready_objects if isinstance(ready_object, int)
        ]

        if ready_sentinels:
            return ready_sentinels

        for ready_object in ready_objects:
            for connection_index, connection in enumerate(connections):
                if ready_object is connection:
                    self._ready_connection_indexes.add(connection_index)
                    break

        if len(self._ready_connection_indexes) != len(connections):
            return []

        self._ready_connection_indexes.clear()
        self.returned_connection_orders.append(self._completion_order)
        return [connections[connection_index] for connection_index in self._completion_order]


def _assigned_shard_ids_from_startup_config(
    startup_config: object,
    expected_worker_index: int,
) -> tuple[int, ...]:
    config = transformer_worker_module.validate_worker_startup_config(
        startup_config,
        expected_worker_index=expected_worker_index,
    )
    return tuple(shard[0] for shard in config.assigned_shards)


def _controlled_malformed_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    del expected_worker_index, startup_config

    try:
        connection.send(("ready",))
    finally:
        connection.close()


def _controlled_failure_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.FailureMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                phase=transformer_worker_module.WorkerFailurePhase.STARTUP,
                code=transformer_worker_module.WorkerFailureCode.INTERNAL_FAILURE,
                epoch=None,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
    finally:
        connection.close()


def _controlled_wrong_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION + 1),
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
    finally:
        connection.close()


def _controlled_exit_before_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    del expected_worker_index, startup_config
    connection.close()


def _controlled_missing_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    del expected_worker_index, startup_config

    try:
        sleep(0.05)
    finally:
        connection.close()


def _controlled_late_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        sleep(0.05)
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_slow_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        sleep(0.25)
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        raw_message = connection.recv()
        transformer_worker_module.validate_stop_message(
            raw_message,
            expected_worker_index=expected_worker_index,
        )
        connection.send(
            transformer_worker_module.StoppedMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
            )
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_slow_shutdown_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        raw_message = connection.recv()
        transformer_worker_module.validate_stop_message(
            raw_message,
            expected_worker_index=expected_worker_index,
        )
        sleep(0.4)
        connection.send(
            transformer_worker_module.StoppedMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
            )
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_never_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    del expected_worker_index, startup_config

    try:
        sleep(0.4)
    finally:
        connection.close()


def _controlled_duplicate_ready_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )
    ready = transformer_worker_module.ReadyMessage(
        protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
        worker_index=expected_worker_index,
        assigned_shard_ids=assigned_shard_ids,
    )

    try:
        if expected_worker_index == 0:
            connection.send(ready)
            connection.send(ready)

        sleep(0.2)
    finally:
        connection.close()


def _complete_controlled_worker_shutdown(
    connection: Connection,
    expected_worker_index: int,
) -> None:
    raw_message = connection.recv()
    transformer_worker_module.validate_stop_message(
        raw_message,
        expected_worker_index=expected_worker_index,
    )
    connection.send(
        transformer_worker_module.StoppedMessage(
            protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
            worker_index=expected_worker_index,
        )
    )


def _controlled_wrong_epoch_result_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        command = transformer_worker_module.validate_compute_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
            expected_shard_ids=assigned_shard_ids,
        )
        connection.send(
            transformer_worker_module.ResultMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                epoch=command.epoch + 1,
                assigned_shard_ids=assigned_shard_ids,
                shard_losses=(0.0,) * len(assigned_shard_ids),
            )
        )
        _complete_controlled_worker_shutdown(
            connection,
            expected_worker_index,
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_failure_epoch_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        command = transformer_worker_module.validate_compute_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
            expected_shard_ids=assigned_shard_ids,
        )
        connection.send(
            transformer_worker_module.FailureMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                phase=transformer_worker_module.WorkerFailurePhase.COMPUTE,
                code=transformer_worker_module.WorkerFailureCode.INTERNAL_FAILURE,
                epoch=command.epoch,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        _complete_controlled_worker_shutdown(
            connection,
            expected_worker_index,
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_malformed_epoch_result_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        transformer_worker_module.validate_compute_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
            expected_shard_ids=assigned_shard_ids,
        )
        connection.send(("result",))
        _complete_controlled_worker_shutdown(
            connection,
            expected_worker_index,
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_duplicate_epoch_result_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        command = transformer_worker_module.validate_compute_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
            expected_shard_ids=assigned_shard_ids,
        )

        if expected_worker_index == 0:
            result = transformer_worker_module.ResultMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                epoch=command.epoch,
                assigned_shard_ids=assigned_shard_ids,
                shard_losses=(0.0,) * len(assigned_shard_ids),
            )
            connection.send(result)
            connection.send(result)

        _complete_controlled_worker_shutdown(
            connection,
            expected_worker_index,
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_exit_during_epoch_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        transformer_worker_module.validate_compute_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
            expected_shard_ids=assigned_shard_ids,
        )
    finally:
        connection.close()


def _controlled_never_result_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        transformer_worker_module.validate_compute_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
            expected_shard_ids=assigned_shard_ids,
        )
        _complete_controlled_worker_shutdown(
            connection,
            expected_worker_index,
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _controlled_partial_commit_with_nonfinite_gradient_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    config = transformer_worker_module.validate_worker_startup_config(
        startup_config,
        expected_worker_index=expected_worker_index,
    )
    assigned_shard_ids = tuple(shard[0] for shard in config.assigned_shards)
    gradient_shared_memory: SharedMemory | None = None
    gradient_buffer: memoryview | None = None
    gradient_storage: _Float32Array | None = None

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=transformer_worker_module.WORKER_PROTOCOL_VERSION,
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        command = transformer_worker_module.validate_compute_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
            expected_shard_ids=assigned_shard_ids,
        )

        if expected_worker_index == 0:
            gradient_shared_memory = SharedMemory(
                name=config.gradient_shared_memory_names[0],
                create=False,
            )
            canonical_byte_count = config.canonical_float_count * np.dtype(np.float32).itemsize
            gradient_buffer = gradient_shared_memory.buf[:canonical_byte_count]
            gradient_storage = np.ndarray(
                (config.canonical_float_count,),
                dtype=np.float32,
                buffer=gradient_buffer,
                order="C",
            )
            gradient_storage[0] = np.float32(np.nan)
            connection.send(
                transformer_worker_module.ResultMessage(
                    protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                    worker_index=expected_worker_index,
                    epoch=command.epoch,
                    assigned_shard_ids=assigned_shard_ids,
                    shard_losses=(0.0,) * len(assigned_shard_ids),
                )
            )

        _complete_controlled_worker_shutdown(
            connection,
            expected_worker_index,
        )
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        gradient_storage = None

        if gradient_buffer is not None:
            gradient_buffer.release()

        if gradient_shared_memory is not None:
            gradient_shared_memory.close()

        connection.close()


def _controlled_exit_without_stopped_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        transformer_worker_module.validate_stop_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
        )
    finally:
        connection.close()


def _controlled_ignore_stop_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        transformer_worker_module.validate_stop_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
        )
        sleep(5.0)
    finally:
        connection.close()


def _controlled_stopped_nonzero_exit_worker(
    connection: Connection,
    expected_worker_index: int,
    startup_config: object,
) -> None:
    assigned_shard_ids = _assigned_shard_ids_from_startup_config(
        startup_config,
        expected_worker_index,
    )

    try:
        connection.send(
            transformer_worker_module.ReadyMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
                assigned_shard_ids=assigned_shard_ids,
            )
        )
        transformer_worker_module.validate_stop_message(
            connection.recv(),
            expected_worker_index=expected_worker_index,
        )
        connection.send(
            transformer_worker_module.StoppedMessage(
                protocol_version=(transformer_worker_module.WORKER_PROTOCOL_VERSION),
                worker_index=expected_worker_index,
            )
        )
    finally:
        connection.close()

    raise SystemExit(3)


class _StartFailingProcess:
    daemon = False

    def __init__(self) -> None:
        self.close_calls = 0

    def start(self) -> None:
        raise OSError("controlled process-start failure")

    def is_alive(self) -> bool:
        return False

    def close(self) -> None:
        self.close_calls += 1


class _TrackedSharedMemory:
    def __init__(self, handle: SharedMemory) -> None:
        self._handle = handle
        self.close_calls = 0
        self.unlink_calls = 0

    @property
    def name(self) -> str:
        return self._handle.name

    @property
    def size(self) -> int:
        return self._handle.size

    @property
    def buf(self) -> memoryview:
        return self._handle.buf

    def close(self) -> None:
        self.close_calls += 1
        self._handle.close()

    def unlink(self) -> None:
        self.unlink_calls += 1
        self._handle.unlink()


@pytest.mark.parametrize("actual_worker_count", (1, 2, 3, 4))
@pytest.mark.asyncio
async def test_real_spawn_group_requires_one_ready_record_from_every_worker(
    monkeypatch: pytest.MonkeyPatch,
    actual_worker_count: int,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: actual_worker_count,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime()

    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    snapshot = runtime.startup_snapshot()

    assert group.state is RequestScopedWorkerGroupState.READY
    assert snapshot.worker_count == actual_worker_count
    assert snapshot.started_worker_count == actual_worker_count
    assert snapshot.ready_worker_count == actual_worker_count
    assert snapshot.non_daemonic == (True,) * actual_worker_count
    assert snapshot.child_endpoints_closed == (True,) * actual_worker_count
    assert snapshot.alive == (True,) * actual_worker_count

    report = await group.cleanup()

    assert report.successful
    assert report.process_exit_codes == (0,) * actual_worker_count
    assert runtime.startup_snapshot().alive == (False,) * actual_worker_count


@pytest.mark.asyncio
async def test_process_start_failure_closes_partial_worker_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    failing_process = _StartFailingProcess()

    def process_factory(
        context: BaseContext,
        target: transformer_worker_module._WorkerProcessTarget,
        child_connection: Connection,
        worker_index: int,
        startup_config: object,
    ) -> BaseProcess:
        del context, target, child_connection, worker_index, startup_config
        return cast(BaseProcess, failing_process)

    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _process_factory=cast(
            transformer_worker_module._WorkerProcessFactory,
            process_factory,
        ),
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=_observe_poll,
            _runtime=runtime,
        )

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE
    assert raised.value.cleanup_report is not None
    assert failing_process.close_calls == 1
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().child_endpoints_closed == (True,)
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.asyncio
async def test_startup_rejects_duplicate_ready_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 2,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_duplicate_ready_worker,
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=_observe_poll,
            _runtime=runtime,
        )

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE
    assert raised.value.cleanup_report is not None
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False, False)


@pytest.mark.parametrize(
    "worker_target",
    (
        _controlled_malformed_ready_worker,
        _controlled_failure_ready_worker,
        _controlled_wrong_ready_worker,
        _controlled_exit_before_ready_worker,
    ),
)
@pytest.mark.asyncio
async def test_startup_rejects_malformed_failed_wrong_or_exited_worker(
    monkeypatch: pytest.MonkeyPatch,
    worker_target: Callable[[Connection, int, object], None],
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=cast(
            transformer_worker_module._WorkerProcessTarget,
            worker_target,
        ),
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=_observe_poll,
            _runtime=runtime,
        )

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE
    assert raised.value.cleanup_report is not None
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.parametrize(
    "worker_target",
    (
        _controlled_missing_ready_worker,
        _controlled_late_ready_worker,
    ),
)
@pytest.mark.asyncio
async def test_startup_uses_one_absolute_deadline_for_missing_or_late_ready(
    monkeypatch: pytest.MonkeyPatch,
    worker_target: Callable[[Connection, int, object], None],
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=cast(
            transformer_worker_module._WorkerProcessTarget,
            worker_target,
        ),
        _monotonic_clock=_SequenceClock((100.0, 131.0)),
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=_observe_poll,
            _runtime=runtime,
        )

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.TIMEOUT
    assert raised.value.cleanup_report is not None
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.asyncio
async def test_startup_waits_off_event_loop_and_observes_after_every_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    event_loop_thread_id = get_ident()
    wait_spy = _DelegatingWaitSpy()
    observer_thread_ids: list[int] = []
    heartbeat_count = 0
    heartbeat_running = True

    async def observe_poll() -> None:
        observer_thread_ids.append(get_ident())

    async def heartbeat() -> None:
        nonlocal heartbeat_count

        while heartbeat_running:
            heartbeat_count += 1
            await asyncio.sleep(0)

    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_slow_ready_worker,
        _connection_waiter=wait_spy,
    )
    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        group = await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=observe_poll,
            _runtime=runtime,
        )
    finally:
        heartbeat_running = False
        await heartbeat_task

    startup_wait_calls = tuple(wait_spy.calls)

    assert len(startup_wait_calls) >= 2
    assert all(
        timeout is not None
        and 0.0 < timeout <= transformer_worker_module._WORKER_GROUP_POLL_TIMEOUT_SECONDS
        for timeout, _ in startup_wait_calls
    )
    assert all(thread_id != event_loop_thread_id for _, thread_id in startup_wait_calls)
    assert observer_thread_ids == ([event_loop_thread_id] * len(startup_wait_calls))
    assert heartbeat_count > 0

    report = await group.cleanup()

    assert report.successful


@pytest.mark.asyncio
async def test_startup_timeout_uses_repeated_bounded_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    wait_spy = _DelegatingWaitSpy()
    observer_calls = 0

    async def observe_poll() -> None:
        nonlocal observer_calls
        observer_calls += 1

    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_never_ready_worker,
        _connection_waiter=wait_spy,
        _startup_timeout_seconds=0.21,
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=observe_poll,
            _runtime=runtime,
        )

    observed_wait_calls = tuple(wait_spy.calls)

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.TIMEOUT
    assert raised.value.cleanup_report is not None
    assert len(observed_wait_calls) >= 2
    assert observer_calls == len(observed_wait_calls)
    assert all(
        timeout is not None
        and 0.0 < timeout <= transformer_worker_module._WORKER_GROUP_POLL_TIMEOUT_SECONDS
        for timeout, _ in observed_wait_calls
    )
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.asyncio
async def test_poll_observer_task_cancellation_propagates_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )

    async def cancel_after_poll() -> None:
        raise asyncio.CancelledError

    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_slow_ready_worker,
    )

    with pytest.raises(asyncio.CancelledError):
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=cancel_after_poll,
            _runtime=runtime,
        )

    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.asyncio
async def test_task_cancellation_during_cleanup_finishes_cleanup_then_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_slow_shutdown_worker,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )
    cleanup_task = asyncio.create_task(group.cleanup())
    await asyncio.sleep(0.001)
    cleanup_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    assert group.state is RequestScopedWorkerGroupState.CLOSED
    assert group.primary_failure_code is RequestScopedWorkerGroupFailureCode.CANCELLED
    assert group.cleanup_report is not None
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.asyncio
async def test_pre_requested_runtime_cancellation_is_primary_and_cleans_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_slow_ready_worker,
    )
    runtime.request_cancellation()

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=_observe_poll,
            _runtime=runtime,
        )

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.CANCELLED
    assert raised.value.cleanup_report is not None
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.parametrize("actual_worker_count", (1, 2, 3, 4))
@pytest.mark.asyncio
async def test_real_spawn_group_computes_complete_epoch_in_canonical_shard_order(
    monkeypatch: pytest.MonkeyPatch,
    actual_worker_count: int,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    expected_results = _direct_shard_results(
        num_layers,
        weights,
        sequences,
        shards,
    )
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: actual_worker_count,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
    )

    try:
        actual_results = await group.compute_epoch(
            0,
            weights,
        )

        assert tuple(result.shard.shard_index for result in actual_results) == (0, 1, 2, 3)

        for actual, expected in zip(
            actual_results,
            expected_results,
            strict=True,
        ):
            assert actual.shard == expected.shard
            assert actual.processed_sequence_count == expected.processed_sequence_count
            assert actual.loss == pytest.approx(
                expected.loss,
                rel=1e-6,
                abs=1e-6,
            )
            np.testing.assert_allclose(
                actual.gradient.storage,
                expected.gradient.storage,
                rtol=1e-6,
                atol=1e-6,
            )
    finally:
        report = await group.cleanup()

    assert report.successful
    assert report.process_exit_codes == (0,) * actual_worker_count


class _RecordingSharedMemoryFactory:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.call_count = 0
        self.created: list[_TrackedSharedMemory] = []

    def __call__(
        self,
        name: str | None = None,
        create: bool = False,
        size: int = 0,
    ) -> SharedMemory:
        self.call_count += 1

        if self.fail_on_call == self.call_count:
            raise OSError("controlled shared-memory allocation failure")

        tracked = _TrackedSharedMemory(
            SharedMemory(
                name=name,
                create=create,
                size=size,
            )
        )
        self.created.append(tracked)
        return cast(SharedMemory, tracked)


def _assert_tracked_shared_memory_released(
    blocks: tuple[_TrackedSharedMemory, ...],
) -> None:
    assert all(block.close_calls == 1 for block in blocks)
    assert all(block.unlink_calls == 1 for block in blocks)


class _StubRuntime:
    def __init__(
        self,
        *,
        results: tuple[LogicalTrainingShardResult, ...] | None = None,
        cleanup_report: RequestScopedWorkerGroupCleanupReport | None = None,
        start_failure: RequestScopedWorkerGroupError | None = None,
        compute_failure: RequestScopedWorkerGroupError | None = None,
    ) -> None:
        self.results = results
        self.cleanup_report = cleanup_report or _clean_report()
        self.start_failure = start_failure
        self.compute_failure = compute_failure
        self.start_calls = 0
        self.compute_calls = 0
        self.cleanup_calls = 0
        self.actual_worker_count: int | None = None
        self.assigned_shard_ids_by_worker: tuple[tuple[int, ...], ...] | None = None
        self.block_compute = False
        self.compute_started = asyncio.Event()
        self.compute_release = asyncio.Event()
        self.start_weights: _Float32Array | None = None
        self.compute_weights: _Float32Array | None = None

    async def start(
        self,
        *,
        num_layers: int,
        current_weights: _Float32Array,
        training_sequences: tuple[TransformerTrainingSequence, ...],
        logical_shards: tuple[LogicalTrainingShard, ...],
        actual_worker_count: int,
        assigned_shard_ids_by_worker: tuple[tuple[int, ...], ...],
        poll_observer: _PollObserver,
    ) -> None:
        del num_layers, training_sequences, logical_shards
        self.start_calls += 1
        self.start_weights = current_weights
        self.actual_worker_count = actual_worker_count
        self.assigned_shard_ids_by_worker = assigned_shard_ids_by_worker
        await poll_observer()

        if self.start_failure is not None:
            raise self.start_failure

    async def compute_epoch(
        self,
        *,
        epoch: int,
        current_weights: _Float32Array,
        poll_observer: _PollObserver,
    ) -> tuple[LogicalTrainingShardResult, ...]:
        del epoch
        self.compute_calls += 1
        self.compute_weights = current_weights
        self.compute_started.set()
        await poll_observer()

        if self.block_compute:
            await self.compute_release.wait()

        if self.compute_failure is not None:
            raise self.compute_failure

        if self.results is None:
            raise AssertionError("Stub runtime requires results for a successful epoch.")

        return self.results

    async def cleanup(self) -> RequestScopedWorkerGroupCleanupReport:
        self.cleanup_calls += 1
        return self.cleanup_report


@pytest.mark.parametrize(
    ("reported_cpu_count", "expected_worker_count"),
    (
        (None, 1),
        (0, 1),
        (1, 1),
        (2, 2),
        (4, 4),
        (5, 4),
        (64, 4),
    ),
)
def test_actual_worker_count_is_bounded_from_reported_cpu_count(
    reported_cpu_count: int | None,
    expected_worker_count: int,
) -> None:
    assert calculate_actual_worker_count(reported_cpu_count) == expected_worker_count


@pytest.mark.parametrize(
    ("actual_worker_count", "expected_assignments"),
    (
        (1, ((0, 1, 2, 3),)),
        (2, ((0, 2), (1, 3))),
        (3, ((0, 3), (1,), (2,))),
        (4, ((0,), (1,), (2,), (3,))),
    ),
)
def test_worker_shard_assignments_are_static_complete_and_ordered(
    actual_worker_count: int,
    expected_assignments: tuple[tuple[int, ...], ...],
) -> None:
    assignments = build_worker_shard_assignments(actual_worker_count)

    assert assignments == expected_assignments
    assert tuple(
        sorted(shard_id for worker_shards in assignments for shard_id in worker_shards)
    ) == (0, 1, 2, 3)
    assert all(worker_shards == tuple(sorted(worker_shards)) for worker_shards in assignments)


async def test_group_reads_cpu_count_once_and_retains_static_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    runtime = _StubRuntime(results=_shard_results(shards))
    cpu_count_calls = 0

    worker_count_property = inspect.getattr_static(  #  Added Code
        RequestScopedWorkerGroup,  #  Added Code
        "actual_worker_count",  #  Added Code
    )
    assert isinstance(worker_count_property, property)  #  Added Code
    assert worker_count_property.fset is None  #  Added Code

    def fake_cpu_count() -> int:
        nonlocal cpu_count_calls
        cpu_count_calls += 1
        return 3

    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        fake_cpu_count,
    )

    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    assert group.actual_worker_count == 3  #  Added Code
    assert cpu_count_calls == 1
    assert runtime.actual_worker_count == 3
    assert runtime.assigned_shard_ids_by_worker == ((0, 3), (1,), (2,))

    await group.compute_epoch(0, weights)

    assert group.actual_worker_count == 3  #  Added Code
    assert cpu_count_calls == 1  #  Added Code

    await group.cleanup()

    assert group.actual_worker_count == 3  #  Added Code
    assert cpu_count_calls == 1


@pytest.mark.asyncio
async def test_group_allocates_five_exact_range_parent_owned_shared_arrays() -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    layout = build_transformer_parameter_layout(num_layers)
    weights[0] = np.float32(1.25)
    weights[-1] = np.float32(-2.5)
    expected_weights = np.array(
        weights,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    factory = _RecordingSharedMemoryFactory()
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        shared_memory_factory=factory,
    )

    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    snapshot = runtime.allocation_snapshot()

    assert factory.call_count == 5
    assert len(factory.created) == 5
    assert len({id(block) for block in factory.created}) == 5
    assert snapshot.block_count == 5
    assert snapshot.array_count == 5
    assert snapshot.requested_byte_count == layout.total_byte_count
    assert all(capacity >= layout.total_byte_count for capacity in snapshot.actual_byte_capacities)
    assert snapshot.dtype_markers == (transformer_worker_module.WorkerNumericDType.FLOAT32,) * 5
    assert snapshot.shapes == ((layout.total_float_count,),) * 5
    assert snapshot.c_contiguous == (True,) * 5
    assert snapshot.gradient_zeroed == (True,) * 4
    assert not snapshot.cancellation_requested
    assert np.array_equal(
        runtime.copy_published_weights(),
        expected_weights,
    )

    weights.fill(np.float32(99.0))
    assert np.array_equal(
        runtime.copy_published_weights(),
        expected_weights,
    )

    report = await group.cleanup()
    released_snapshot = runtime.allocation_snapshot()

    assert report.successful
    assert released_snapshot.block_count == 0
    assert released_snapshot.array_count == 0
    assert released_snapshot.cancellation_requested
    _assert_tracked_shared_memory_released(tuple(factory.created))


@pytest.mark.parametrize("fail_on_call", (1, 2, 3, 4, 5))
@pytest.mark.asyncio
async def test_partial_shared_memory_allocation_is_cleaned_transactionally(
    fail_on_call: int,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    factory = _RecordingSharedMemoryFactory(
        fail_on_call=fail_on_call,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        shared_memory_factory=factory,
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=_observe_poll,
            _runtime=runtime,
        )

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE
    assert raised.value.cleanup_report is not None
    assert raised.value.cleanup_report.successful
    assert factory.call_count == fail_on_call
    assert len(factory.created) == fail_on_call - 1
    _assert_tracked_shared_memory_released(tuple(factory.created))


@pytest.mark.asyncio
async def test_concurrent_groups_own_independent_shared_memory_and_cancellation() -> None:
    first_num_layers, first_weights, first_sequences, first_shards = _group_inputs()
    second_num_layers, second_weights, second_sequences, second_shards = _group_inputs()
    second_weights[0] = np.float32(0.125)

    first_expected_results = _direct_shard_results(
        first_num_layers,
        first_weights,
        first_sequences,
        first_shards,
    )
    second_expected_results = _direct_shard_results(
        second_num_layers,
        second_weights,
        second_sequences,
        second_shards,
    )

    first_factory = _RecordingSharedMemoryFactory()
    second_factory = _RecordingSharedMemoryFactory()
    first_runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        shared_memory_factory=first_factory,
    )
    second_runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        shared_memory_factory=second_factory,
    )

    first_group, second_group = await asyncio.gather(
        create_request_scoped_worker_group(
            first_num_layers,
            first_weights,
            first_sequences,
            first_shards,
            poll_observer=_observe_poll,
            _runtime=first_runtime,
        ),
        create_request_scoped_worker_group(
            second_num_layers,
            second_weights,
            second_sequences,
            second_shards,
            poll_observer=_observe_poll,
            _runtime=second_runtime,
        ),
    )

    first_report: RequestScopedWorkerGroupCleanupReport | None = None

    try:
        assert {id(block) for block in first_factory.created}.isdisjoint(
            id(block) for block in second_factory.created
        )
        assert np.array_equal(
            first_runtime.copy_published_weights(),
            first_weights,
        )
        assert np.array_equal(
            second_runtime.copy_published_weights(),
            second_weights,
        )
        assert not first_runtime.allocation_snapshot().cancellation_requested
        assert not second_runtime.allocation_snapshot().cancellation_requested

        first_results, second_results = await asyncio.gather(
            first_group.compute_epoch(
                0,
                first_weights,
            ),
            second_group.compute_epoch(
                0,
                second_weights,
            ),
        )

        _assert_shard_results_match(
            first_results,
            first_expected_results,
        )
        _assert_shard_results_match(
            second_results,
            second_expected_results,
        )

        for first_result, second_result in zip(
            first_results,
            second_results,
            strict=True,
        ):
            assert not np.shares_memory(
                first_result.gradient.storage,
                second_result.gradient.storage,
            )

        first_report = await first_group.cleanup()

        assert first_report.successful
        assert first_runtime.allocation_snapshot().cancellation_requested
        assert not second_runtime.allocation_snapshot().cancellation_requested

        second_followup_results = await second_group.compute_epoch(
            1,
            second_weights,
        )
        _assert_shard_results_match(
            second_followup_results,
            second_expected_results,
        )
    finally:
        if first_report is None:
            first_report = await first_group.cleanup()

        second_report = await second_group.cleanup()

    assert first_report is not None
    assert first_report.successful
    assert second_report.successful

    _assert_tracked_shared_memory_released(tuple(first_factory.created))
    _assert_tracked_shared_memory_released(tuple(second_factory.created))


@pytest.mark.asyncio
async def test_sequential_groups_use_fresh_resources_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    expected_results = _direct_shard_results(
        num_layers,
        weights,
        sequences,
        shards,
    )
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )

    first_factory = _RecordingSharedMemoryFactory()
    first_runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        shared_memory_factory=first_factory,
    )
    first_group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=first_runtime,
    )

    first_results: tuple[LogicalTrainingShardResult, ...] | None = None

    try:
        first_results = await first_group.compute_epoch(
            0,
            weights,
        )
    finally:
        first_report = await first_group.cleanup()

    assert first_results is not None
    assert first_report.successful
    assert first_group.state is RequestScopedWorkerGroupState.CLOSED
    _assert_shard_results_match(
        first_results,
        expected_results,
    )
    _assert_tracked_shared_memory_released(tuple(first_factory.created))

    second_factory = _RecordingSharedMemoryFactory()
    second_runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        shared_memory_factory=second_factory,
    )
    second_group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=second_runtime,
    )

    second_results: tuple[LogicalTrainingShardResult, ...] | None = None

    try:
        assert second_group is not first_group
        assert second_group.state is RequestScopedWorkerGroupState.READY
        assert {id(block) for block in first_factory.created}.isdisjoint(
            id(block) for block in second_factory.created
        )
        assert first_runtime.allocation_snapshot().cancellation_requested
        assert not second_runtime.allocation_snapshot().cancellation_requested

        second_results = await second_group.compute_epoch(
            0,
            weights,
        )
    finally:
        second_report = await second_group.cleanup()

    assert second_results is not None
    assert second_report.successful
    assert second_group.state is RequestScopedWorkerGroupState.CLOSED
    _assert_shard_results_match(
        second_results,
        expected_results,
    )

    for first_result, second_result in zip(
        first_results,
        second_results,
        strict=True,
    ):
        assert first_result is not second_result
        assert not np.shares_memory(
            first_result.gradient.storage,
            second_result.gradient.storage,
        )

    _assert_tracked_shared_memory_released(tuple(second_factory.created))


def test_worker_group_public_contract_is_exact_and_immutable() -> None:
    assert tuple(transformer_worker_module.__all__) == _EXPECTED_PUBLIC_SYMBOLS

    for symbol in _EXPECTED_PUBLIC_SYMBOLS:
        assert hasattr(transformer_worker_module, symbol)

    assert tuple(RequestScopedWorkerGroupState) == (
        RequestScopedWorkerGroupState.ALLOCATED,
        RequestScopedWorkerGroupState.STARTING,
        RequestScopedWorkerGroupState.READY,
        RequestScopedWorkerGroupState.COMPUTING,
        RequestScopedWorkerGroupState.STOPPING,
        RequestScopedWorkerGroupState.CLOSED,
        RequestScopedWorkerGroupState.FAILED,
    )
    assert tuple(RequestScopedWorkerGroupFailureCode) == (
        RequestScopedWorkerGroupFailureCode.INVALID_INPUT,
        RequestScopedWorkerGroupFailureCode.INVALID_STATE,
        RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
        RequestScopedWorkerGroupFailureCode.CANCELLED,
        RequestScopedWorkerGroupFailureCode.TIMEOUT,
        RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE,
        RequestScopedWorkerGroupFailureCode.INTERNAL_FAILURE,
    )
    assert tuple(RequestScopedWorkerGroupCleanupFailureCode) == (
        RequestScopedWorkerGroupCleanupFailureCode.CANCELLATION_SIGNAL,
        RequestScopedWorkerGroupCleanupFailureCode.STOP_SEND,
        RequestScopedWorkerGroupCleanupFailureCode.COOPERATIVE_WAIT,
        RequestScopedWorkerGroupCleanupFailureCode.TERMINATE,
        RequestScopedWorkerGroupCleanupFailureCode.TERMINATE_WAIT,
        RequestScopedWorkerGroupCleanupFailureCode.KILL,
        RequestScopedWorkerGroupCleanupFailureCode.JOIN,
        RequestScopedWorkerGroupCleanupFailureCode.EXIT_CODE,
        RequestScopedWorkerGroupCleanupFailureCode.PIPE_CLOSE,
        RequestScopedWorkerGroupCleanupFailureCode.PROCESS_CLOSE,
        RequestScopedWorkerGroupCleanupFailureCode.VIEW_RELEASE,
        RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_CLOSE,
        RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_UNLINK,
        RequestScopedWorkerGroupCleanupFailureCode.INTERNAL,
    )

    report = _clean_report()

    assert dataclasses.is_dataclass(RequestScopedWorkerGroupCleanupReport)
    assert tuple(field.name for field in fields(RequestScopedWorkerGroupCleanupReport)) == (
        "cooperative_shutdown_completed",
        "terminate_required",
        "kill_required",
        "process_exit_codes",
        "secondary_failures",
    )
    assert not hasattr(report, "__dict__")
    assert report.successful

    with pytest.raises(FrozenInstanceError):
        report.terminate_required = True  # type: ignore[misc]

    error = RequestScopedWorkerGroupError(
        RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
    )
    assert str(error) == "Request-scoped worker group operation failed."
    assert error.code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    assert error.cleanup_report is None

    assert tuple(inspect.signature(create_request_scoped_worker_group).parameters) == (
        "num_layers",
        "current_weights",
        "training_sequences",
        "logical_shards",
        "poll_observer",
        "_runtime",
    )
    assert tuple(inspect.signature(RequestScopedWorkerGroup.compute_epoch).parameters) == (
        "self",
        "epoch",
        "current_weights",
    )
    assert tuple(inspect.signature(RequestScopedWorkerGroup.cleanup).parameters) == ("self",)


@pytest.mark.asyncio
async def test_group_rejects_compute_before_ready_and_after_cleanup() -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    runtime = _StubRuntime(results=_shard_results(shards))
    group = RequestScopedWorkerGroup(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    assert group.state is RequestScopedWorkerGroupState.ALLOCATED

    with pytest.raises(RequestScopedWorkerGroupError) as before_ready:
        await group.compute_epoch(0, weights)

    assert before_ready.value.code is RequestScopedWorkerGroupFailureCode.INVALID_STATE
    assert group.state is RequestScopedWorkerGroupState.ALLOCATED

    first_report = await group.cleanup()
    second_report = await group.cleanup()

    assert first_report is second_report
    assert runtime.cleanup_calls == 1
    assert group.state is RequestScopedWorkerGroupState.CLOSED
    assert not group.successful

    with pytest.raises(RequestScopedWorkerGroupError) as after_cleanup:
        await group.compute_epoch(0, weights)

    assert after_cleanup.value.code is RequestScopedWorkerGroupFailureCode.INVALID_STATE


@pytest.mark.asyncio
async def test_group_rejects_overlapping_epochs_and_returns_independent_results() -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    runtime_results = _shard_results(shards)
    runtime = _StubRuntime(results=runtime_results)
    runtime.block_compute = True
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    assert group.state is RequestScopedWorkerGroupState.READY
    assert runtime.start_weights is not None
    assert not np.shares_memory(runtime.start_weights, weights)

    compute_task = asyncio.create_task(group.compute_epoch(0, weights))
    await runtime.compute_started.wait()

    assert group.state is RequestScopedWorkerGroupState.COMPUTING

    with pytest.raises(RequestScopedWorkerGroupError) as overlapping:
        await group.compute_epoch(1, weights)

    assert overlapping.value.code is RequestScopedWorkerGroupFailureCode.INVALID_STATE

    runtime.compute_release.set()
    copied_results = await compute_task

    assert group.state is RequestScopedWorkerGroupState.READY
    assert runtime.compute_weights is not None
    assert not np.shares_memory(runtime.compute_weights, weights)
    assert len(copied_results) == 4

    for copied, original, expected_shard in zip(
        copied_results,
        runtime_results,
        shards,
        strict=True,
    ):
        assert copied.shard == expected_shard
        assert copied.processed_sequence_count == original.processed_sequence_count
        assert copied.loss == original.loss
        assert np.array_equal(copied.gradient.storage, original.gradient.storage)
        assert not np.shares_memory(
            copied.gradient.storage,
            original.gradient.storage,
        )

    original_first_value = np.float32(runtime_results[0].gradient.storage[0])
    copied_results[0].gradient.storage[0] = np.float32(999.0)
    assert runtime_results[0].gradient.storage[0] == original_first_value

    await group.cleanup()
    assert group.successful


@pytest.mark.asyncio
async def test_startup_failure_remains_primary_when_cleanup_also_fails() -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    cleanup_report = RequestScopedWorkerGroupCleanupReport(
        cooperative_shutdown_completed=False,
        terminate_required=True,
        kill_required=False,
        process_exit_codes=(1,),
        secondary_failures=(RequestScopedWorkerGroupCleanupFailureCode.PIPE_CLOSE,),
    )
    runtime = _StubRuntime(
        results=_shard_results(shards),
        cleanup_report=cleanup_report,
        start_failure=RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE,
        ),
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await create_request_scoped_worker_group(
            num_layers,
            weights,
            sequences,
            shards,
            poll_observer=_observe_poll,
            _runtime=runtime,
        )

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.STARTUP_FAILURE
    assert raised.value.cleanup_report is cleanup_report
    assert runtime.start_calls == 1
    assert runtime.cleanup_calls == 1


@pytest.mark.asyncio
async def test_epoch_failure_is_preserved_across_cleanup_failure() -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    cleanup_report = RequestScopedWorkerGroupCleanupReport(
        cooperative_shutdown_completed=False,
        terminate_required=False,
        kill_required=True,
        process_exit_codes=(None,),
        secondary_failures=(RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_UNLINK,),
    )
    runtime = _StubRuntime(
        results=_shard_results(shards),
        cleanup_report=cleanup_report,
        compute_failure=RequestScopedWorkerGroupError(
            RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE,
        ),
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await group.compute_epoch(0, weights)

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    assert group.state is RequestScopedWorkerGroupState.FAILED
    assert group.primary_failure_code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE

    report = await group.cleanup()

    assert report is cleanup_report
    assert group.state is RequestScopedWorkerGroupState.CLOSED
    assert group.primary_failure_code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    assert not group.successful


@pytest.mark.asyncio
async def test_cleanup_only_failure_prevents_successful_group_outcome() -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    cleanup_report = RequestScopedWorkerGroupCleanupReport(
        cooperative_shutdown_completed=True,
        terminate_required=True,
        kill_required=False,
        process_exit_codes=(0,),
        secondary_failures=(),
    )
    runtime = _StubRuntime(
        results=_shard_results(shards),
        cleanup_report=cleanup_report,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    report = await group.cleanup()

    assert report is cleanup_report
    assert group.primary_failure_code is RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE
    assert group.state is RequestScopedWorkerGroupState.CLOSED
    assert not group.successful


@pytest.mark.parametrize(
    "completion_order",
    (
        (3, 2, 1, 0),
        (2, 0, 3, 1),
    ),
)
@pytest.mark.asyncio
async def test_epoch_results_ignore_controlled_worker_completion_order(
    monkeypatch: pytest.MonkeyPatch,
    completion_order: tuple[int, ...],
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    expected_results = _direct_shard_results(
        num_layers,
        weights,
        sequences,
        shards,
    )
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 4,
    )
    completion_waiter = _CompletionOrderWaiter(completion_order)
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _connection_waiter=completion_waiter,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    try:
        actual_results = await group.compute_epoch(
            2,
            weights,
        )

        assert completion_waiter.returned_connection_orders[-1] == completion_order

        for actual, expected in zip(
            actual_results,
            expected_results,
            strict=True,
        ):
            assert actual.shard == expected.shard
            assert actual.loss == pytest.approx(
                expected.loss,
                rel=1e-6,
                abs=1e-6,
            )
            np.testing.assert_allclose(
                actual.gradient.storage,
                expected.gradient.storage,
                rtol=1e-6,
                atol=1e-6,
            )
    finally:
        report = await group.cleanup()

    assert report.successful
    assert report.process_exit_codes == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "worker_target",
    (
        _controlled_wrong_epoch_result_worker,
        _controlled_failure_epoch_worker,
        _controlled_malformed_epoch_result_worker,
        _controlled_exit_during_epoch_worker,
    ),
)
@pytest.mark.asyncio
async def test_epoch_rejects_invalid_failure_malformed_or_exited_worker(
    monkeypatch: pytest.MonkeyPatch,
    worker_target: Callable[[Connection, int, object], None],
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=cast(
            transformer_worker_module._WorkerProcessTarget,
            worker_target,
        ),
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    try:
        with pytest.raises(RequestScopedWorkerGroupError) as raised:
            await group.compute_epoch(
                7,
                weights,
            )

        assert raised.value.code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    finally:
        await group.cleanup()

    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.asyncio
async def test_epoch_send_failure_is_reported_without_a_second_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    original_sender = transformer_worker_module._send_worker_message
    compute_send_calls = 0

    def fail_compute_send(
        connection: transformer_worker_module._WorkerConnection,
        message: object,
    ) -> None:
        nonlocal compute_send_calls

        if type(message) is transformer_worker_module.ComputeMessage:
            compute_send_calls += 1
            raise BrokenPipeError("controlled compute send failure")

        original_sender(
            connection,
            message,
        )

    monkeypatch.setattr(
        transformer_worker_module,
        "_send_worker_message",
        fail_compute_send,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
    )

    try:
        with pytest.raises(RequestScopedWorkerGroupError) as raised:
            await group.compute_epoch(
                6,
                weights,
            )

        assert raised.value.code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
        assert compute_send_calls == 1
    finally:
        report = await group.cleanup()

    assert report.successful


@pytest.mark.asyncio
async def test_epoch_rejects_duplicate_worker_result_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 2,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_duplicate_epoch_result_worker,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    try:
        with pytest.raises(RequestScopedWorkerGroupError) as raised:
            await group.compute_epoch(
                8,
                weights,
            )

        assert raised.value.code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    finally:
        report = await group.cleanup()

    assert report.successful
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False, False)


@pytest.mark.asyncio
async def test_epoch_timeout_uses_one_absolute_deadline_and_bounded_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_WORKER_GROUP_EPOCH_TIMEOUT_SECONDS",
        0.21,
    )
    wait_spy = _DelegatingWaitSpy()
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_never_result_worker,
        _connection_waiter=wait_spy,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )
    startup_wait_call_count = len(wait_spy.calls)

    try:
        with pytest.raises(RequestScopedWorkerGroupError) as raised:
            await group.compute_epoch(
                3,
                weights,
            )

        epoch_wait_calls = tuple(wait_spy.calls[startup_wait_call_count:])

        assert raised.value.code is RequestScopedWorkerGroupFailureCode.TIMEOUT
        assert len(epoch_wait_calls) >= 2
        assert all(
            timeout is not None
            and 0.0 < timeout <= transformer_worker_module._WORKER_GROUP_POLL_TIMEOUT_SECONDS
            for timeout, _ in epoch_wait_calls
        )
    finally:
        report = await group.cleanup()

    assert report.successful
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.asyncio
async def test_epoch_does_not_read_shared_gradients_before_every_worker_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 2,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_WORKER_GROUP_EPOCH_TIMEOUT_SECONDS",
        0.21,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=(_controlled_partial_commit_with_nonfinite_gradient_worker),
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    try:
        with pytest.raises(RequestScopedWorkerGroupError) as raised:
            await group.compute_epoch(
                4,
                weights,
            )

        assert raised.value.code is RequestScopedWorkerGroupFailureCode.TIMEOUT
    finally:
        report = await group.cleanup()

    assert report.successful
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False, False)


@pytest.mark.asyncio
async def test_epoch_rejects_nonfinite_shared_gradient_after_complete_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=(_controlled_partial_commit_with_nonfinite_gradient_worker),
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    try:
        with pytest.raises(RequestScopedWorkerGroupError) as raised:
            await group.compute_epoch(
                5,
                weights,
            )

        assert raised.value.code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    finally:
        report = await group.cleanup()

    assert report.successful
    assert runtime.allocation_snapshot().block_count == 0
    assert runtime.startup_snapshot().alive == (False,)


@pytest.mark.parametrize(
    ("worker_target", "expected_failure"),
    (
        (
            _controlled_exit_without_stopped_worker,
            RequestScopedWorkerGroupCleanupFailureCode.COOPERATIVE_WAIT,
        ),
        (
            _controlled_stopped_nonzero_exit_worker,
            RequestScopedWorkerGroupCleanupFailureCode.EXIT_CODE,
        ),
    ),
)
@pytest.mark.asyncio
async def test_cooperative_shutdown_requires_stopped_record_and_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
    worker_target: Callable[[Connection, int, object], None],
    expected_failure: RequestScopedWorkerGroupCleanupFailureCode,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=cast(
            transformer_worker_module._WorkerProcessTarget,
            worker_target,
        ),
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    report = await group.cleanup()

    assert not report.successful
    assert not report.cooperative_shutdown_completed
    assert not report.terminate_required
    assert not report.kill_required
    assert expected_failure in report.secondary_failures
    assert runtime.allocation_snapshot().block_count == 0


@pytest.mark.asyncio
async def test_cleanup_terminates_worker_that_ignores_cooperative_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_WORKER_GROUP_INTERIM_SHUTDOWN_TIMEOUT_SECONDS",
        0.05,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_ignore_stop_worker,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    report = await group.cleanup()

    assert not report.successful
    assert not report.cooperative_shutdown_completed
    assert report.terminate_required
    assert not report.kill_required
    assert runtime.allocation_snapshot().block_count == 0


@pytest.mark.asyncio
async def test_cleanup_kills_survivor_after_terminate_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_WORKER_GROUP_INTERIM_SHUTDOWN_TIMEOUT_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_terminate_worker_process",
        lambda process: None,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_ignore_stop_worker,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    report = await group.cleanup()

    assert not report.successful
    assert report.terminate_required
    assert report.kill_required
    assert runtime.allocation_snapshot().block_count == 0


# Non-short-circuiting order and idempotence
@pytest.mark.asyncio
async def test_cleanup_attempts_later_resource_stages_after_earlier_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime()
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )
    events: list[str] = []

    original_pipe_close = transformer_worker_module._close_parent_worker_connection
    original_process_close = transformer_worker_module._close_parent_worker_process
    original_view_release = transformer_worker_module._release_parent_memoryview
    original_shared_close = transformer_worker_module._close_parent_shared_memory
    original_shared_unlink = transformer_worker_module._unlink_parent_shared_memory

    def fail_pipe_close_after_delegate(
        connection: transformer_worker_module._WorkerConnection,
    ) -> None:
        events.append("pipe_close")
        original_pipe_close(connection)
        raise OSError("controlled pipe-close diagnostic")

    def fail_process_close_after_delegate(process: BaseProcess) -> None:
        events.append("process_close")
        original_process_close(process)
        raise OSError("controlled process-close diagnostic")

    def fail_view_release_after_delegate(buffer: memoryview) -> None:
        events.append("view_release")
        original_view_release(buffer)
        raise BufferError("controlled view-release diagnostic")

    def fail_shared_close_after_delegate(handle: SharedMemory) -> None:
        events.append("shared_close")
        original_shared_close(handle)
        raise OSError("controlled shared-close diagnostic")

    def fail_shared_unlink_after_delegate(handle: SharedMemory) -> None:
        events.append("shared_unlink")
        original_shared_unlink(handle)
        raise OSError("controlled shared-unlink diagnostic")

    monkeypatch.setattr(
        transformer_worker_module,
        "_close_parent_worker_connection",
        fail_pipe_close_after_delegate,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_close_parent_worker_process",
        fail_process_close_after_delegate,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_release_parent_memoryview",
        fail_view_release_after_delegate,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_close_parent_shared_memory",
        fail_shared_close_after_delegate,
    )
    monkeypatch.setattr(
        transformer_worker_module,
        "_unlink_parent_shared_memory",
        fail_shared_unlink_after_delegate,
    )

    first_report = await group.cleanup()
    first_events = tuple(events)
    second_report = await group.cleanup()

    assert first_report is second_report
    assert tuple(events) == first_events

    assert first_events.index("pipe_close") < first_events.index("process_close")
    assert first_events.index("process_close") < first_events.index("view_release")
    assert first_events.index("view_release") < first_events.index("shared_close")
    assert first_events.index("shared_close") < first_events.index("shared_unlink")

    assert first_events.count("view_release") == 5
    assert first_events.count("shared_close") == 5
    assert first_events.count("shared_unlink") == 5

    assert {
        RequestScopedWorkerGroupCleanupFailureCode.PIPE_CLOSE,
        RequestScopedWorkerGroupCleanupFailureCode.PROCESS_CLOSE,
        RequestScopedWorkerGroupCleanupFailureCode.VIEW_RELEASE,
        RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_CLOSE,
        RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_UNLINK,
    }.issubset(first_report.secondary_failures)

    assert not first_report.successful


# Preserve the original epoch failure
@pytest.mark.asyncio
async def test_primary_epoch_failure_survives_secondary_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    runtime = transformer_worker_module._SharedMemoryRequestScopedWorkerGroupRuntime(
        _worker_target=_controlled_failure_epoch_worker,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
        _runtime=runtime,
    )

    with pytest.raises(RequestScopedWorkerGroupError) as raised:
        await group.compute_epoch(
            0,
            weights,
        )

    original_pipe_close = transformer_worker_module._close_parent_worker_connection

    def fail_pipe_close_after_delegate(
        connection: transformer_worker_module._WorkerConnection,
    ) -> None:
        original_pipe_close(connection)
        raise OSError("controlled pipe-close diagnostic")

    monkeypatch.setattr(
        transformer_worker_module,
        "_close_parent_worker_connection",
        fail_pipe_close_after_delegate,
    )

    report = await group.cleanup()

    assert raised.value.code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    assert group.primary_failure_code is RequestScopedWorkerGroupFailureCode.EPOCH_FAILURE
    assert RequestScopedWorkerGroupCleanupFailureCode.PIPE_CLOSE in report.secondary_failures


# Cleanup-only failure becomes the primary failure


@pytest.mark.asyncio
async def test_cleanup_only_resource_failure_prevents_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_layers, weights, sequences, shards = _group_inputs()
    monkeypatch.setattr(
        transformer_worker_module.os,
        "cpu_count",
        lambda: 1,
    )
    group = await create_request_scoped_worker_group(
        num_layers,
        weights,
        sequences,
        shards,
        poll_observer=_observe_poll,
    )
    original_unlink = transformer_worker_module._unlink_parent_shared_memory

    def fail_unlink_after_delegate(handle: SharedMemory) -> None:
        original_unlink(handle)
        raise OSError("controlled unlink diagnostic")

    monkeypatch.setattr(
        transformer_worker_module,
        "_unlink_parent_shared_memory",
        fail_unlink_after_delegate,
    )

    report = await group.cleanup()

    assert not report.successful
    assert (
        RequestScopedWorkerGroupCleanupFailureCode.SHARED_MEMORY_UNLINK in report.secondary_failures
    )
    assert group.primary_failure_code is RequestScopedWorkerGroupFailureCode.CLEANUP_FAILURE
