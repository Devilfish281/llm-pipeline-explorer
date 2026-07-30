# tests/test_transformer_worker.py
from __future__ import annotations

import dataclasses
import math
import multiprocessing as mp
import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass, fields
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from multiprocessing.process import BaseProcess
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from how_llms_work.ml.math_utils import Mulberry32
from how_llms_work.ml.transformer import (
    LogicalTrainingShard,
    TransformerParameterLayout,
    TransformerTrainingSequence,
    build_logical_training_shards,
    build_transformer_parameter_layout,
    calculate_logical_training_shard,
    initialize_transformer_parameters,
)
from how_llms_work.ml.transformer_worker import (
    WORKER_PROTOCOL_VERSION,
    ComputeMessage,
    FailureMessage,
    ReadyMessage,
    ResultMessage,
    StopMessage,
    StoppedMessage,
    WorkerFailureCode,
    WorkerFailurePhase,
    WorkerNumericDType,
    WorkerProtocolValidationError,
    WorkerStartupConfig,
    WorkerState,
    run_transformer_worker,
    validate_compute_message,
    validate_failure_message,
    validate_ready_message,
    validate_result_message,
    validate_stop_message,
    validate_stopped_message,
    validate_worker_startup_config,
)

_PROTOCOL_RECORD_TYPES = (
    WorkerStartupConfig,
    ReadyMessage,
    ComputeMessage,
    ResultMessage,
    FailureMessage,
    StopMessage,
    StoppedMessage,
)

_EXPECTED_FIELD_NAMES = {
    WorkerStartupConfig: (
        "protocol_version",
        "worker_index",
        "num_layers",
        "dtype_marker",
        "canonical_float_count",
        "sequence_count",
        "weight_shared_memory_name",
        "assigned_shards",
        "gradient_shared_memory_names",
        "training_sequences",
    ),
    ReadyMessage: (
        "protocol_version",
        "worker_index",
        "assigned_shard_ids",
    ),
    ComputeMessage: (
        "protocol_version",
        "worker_index",
        "epoch",
        "assigned_shard_ids",
    ),
    ResultMessage: (
        "protocol_version",
        "worker_index",
        "epoch",
        "assigned_shard_ids",
        "shard_losses",
    ),
    FailureMessage: (
        "protocol_version",
        "worker_index",
        "phase",
        "code",
        "epoch",
        "assigned_shard_ids",
    ),
    StopMessage: (
        "protocol_version",
        "worker_index",
    ),
    StoppedMessage: (
        "protocol_version",
        "worker_index",
    ),
}


@dataclass(slots=True)
class _ParentSharedState:
    layout: TransformerParameterLayout
    sequence: TransformerTrainingSequence
    shard: LogicalTrainingShard
    weight_shared_memory: SharedMemory
    assigned_gradient_shared_memory: SharedMemory
    unassigned_gradient_shared_memory: SharedMemory
    weight_storage: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float32],
    ]
    assigned_gradient_storage: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float32],
    ]
    unassigned_gradient_storage: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float32],
    ]
    startup_config: WorkerStartupConfig

    def cleanup(self) -> None:
        self.weight_storage = np.empty(0, dtype=np.float32)
        self.assigned_gradient_storage = np.empty(
            0,
            dtype=np.float32,
        )
        self.unassigned_gradient_storage = np.empty(
            0,
            dtype=np.float32,
        )

        errors: list[BaseException] = []

        for shared_memory in (
            self.weight_shared_memory,
            self.assigned_gradient_shared_memory,
            self.unassigned_gradient_shared_memory,
        ):
            try:
                shared_memory.close()
            except BaseException as exc:
                errors.append(exc)

            try:
                shared_memory.unlink()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                errors.append(exc)

        if errors:
            raise ExceptionGroup(
                "Parent shared-memory cleanup failed.",
                errors,
            )


@dataclass(slots=True)
class _SpawnedWorker:
    parent_connection: Connection
    process: BaseProcess

    def receive(self, timeout: float = 30.0) -> object:
        assert self.parent_connection.poll(timeout), (
            f"Worker did not publish a record within {timeout} seconds; "
            f"alive={self.process.is_alive()} "
            f"exitcode={self.process.exitcode}."
        )
        return self.parent_connection.recv()

    def join(
        self,
        expected_exit_code: int,
        timeout: float = 30.0,
    ) -> None:
        self.process.join(timeout)

        assert not self.process.is_alive(), "Worker did not exit within the bounded join timeout."
        assert self.process.exitcode == expected_exit_code

    def close(self) -> None:
        try:
            self.parent_connection.close()
        finally:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(10.0)

            self.process.close()


@dataclass(frozen=True, slots=True)
class _ReadyLookalike:
    protocol_version: int
    worker_index: int
    assigned_shard_ids: tuple[int, ...]


def _training_sequence(
    seed: int = 0,
) -> TransformerTrainingSequence:
    input_ids = tuple((seed + index) % 32 for index in range(16))
    target_ids = tuple((seed + index + 1) % 32 for index in range(16))

    return TransformerTrainingSequence(
        input_ids=input_ids,
        target_ids=target_ids,
    )


def _sequence_payload(
    sequence: TransformerTrainingSequence,
) -> tuple[
    tuple[
        tuple[int, ...],
        tuple[int, ...],
    ],
    ...,
]:
    return (
        (
            sequence.input_ids,
            sequence.target_ids,
        ),
    )


def _shard_payload(
    shard: LogicalTrainingShard,
) -> tuple[tuple[int, int, int], ...]:
    return (
        (
            shard.shard_index,
            shard.start_index,
            shard.stop_index,
        ),
    )


def _valid_protocol_records() -> tuple[object, ...]:
    layout = build_transformer_parameter_layout(1)
    sequence = _training_sequence()
    shard = build_logical_training_shards(1)[0]

    startup = WorkerStartupConfig(
        protocol_version=WORKER_PROTOCOL_VERSION,
        worker_index=0,
        num_layers=1,
        dtype_marker=WorkerNumericDType.FLOAT32,
        canonical_float_count=layout.total_float_count,
        sequence_count=1,
        weight_shared_memory_name="weight-name",
        assigned_shards=_shard_payload(shard),
        gradient_shared_memory_names=("gradient-name",),
        training_sequences=_sequence_payload(sequence),
    )

    return (
        startup,
        ReadyMessage(
            WORKER_PROTOCOL_VERSION,
            0,
            (0,),
        ),
        ComputeMessage(
            WORKER_PROTOCOL_VERSION,
            0,
            3,
            (0,),
        ),
        ResultMessage(
            WORKER_PROTOCOL_VERSION,
            0,
            3,
            (0,),
            (1.25,),
        ),
        FailureMessage(
            WORKER_PROTOCOL_VERSION,
            0,
            WorkerFailurePhase.COMPUTE,
            WorkerFailureCode.NUMERICAL_FAILURE,
            3,
            (0,),
        ),
        StopMessage(
            WORKER_PROTOCOL_VERSION,
            0,
        ),
        StoppedMessage(
            WORKER_PROTOCOL_VERSION,
            0,
        ),
    )


def _assert_protocol_value_is_private(value: object) -> None:
    if value is None or type(value) in {int, float, str}:
        return

    if isinstance(
        value,
        (
            WorkerNumericDType,
            WorkerFailurePhase,
            WorkerFailureCode,
        ),
    ):
        return

    if type(value) is tuple:
        for item in cast(tuple[object, ...], value):
            _assert_protocol_value_is_private(item)

        return

    assert not isinstance(
        value,
        (
            list,
            dict,
            set,
            bytearray,
            memoryview,
            np.generic,
            np.ndarray,
            BaseException,
            Path,
        ),
    )

    pytest.fail("Protocol value uses an unsupported runtime type: " f"{type(value)!r}")


def _shared_array(
    shared_memory: SharedMemory,
    float_count: int,
) -> np.ndarray[
    tuple[int, ...],
    np.dtype[np.float32],
]:
    return np.ndarray(
        (float_count,),
        dtype=np.float32,
        buffer=shared_memory.buf,
        order="C",
    )


@contextmanager
def _parent_shared_state(
    *,
    assigned_shard_id: int = 0,
    extra_bytes: int = 0,
    weight_size_override: int | None = None,
    gradient_size_override: int | None = None,
) -> Iterator[_ParentSharedState]:
    layout = build_transformer_parameter_layout(1)
    initialized = initialize_transformer_parameters(
        layout,
        Mulberry32(42),
    )
    sequence = _training_sequence()
    shard = build_logical_training_shards(1)[assigned_shard_id]

    weight_size = (
        weight_size_override
        if weight_size_override is not None
        else layout.total_byte_count + extra_bytes
    )
    gradient_size = (
        gradient_size_override
        if gradient_size_override is not None
        else layout.total_byte_count + extra_bytes
    )

    weight_shared_memory = SharedMemory(
        create=True,
        size=weight_size,
    )
    assigned_gradient_shared_memory = SharedMemory(
        create=True,
        size=gradient_size,
    )
    unassigned_gradient_shared_memory = SharedMemory(
        create=True,
        size=layout.total_byte_count + extra_bytes,
    )

    if weight_size >= layout.total_byte_count:
        weight_storage = _shared_array(
            weight_shared_memory,
            layout.total_float_count,
        )
        weight_storage[...] = initialized.storage
    else:
        weight_storage = np.empty(
            0,
            dtype=np.float32,
        )

    if gradient_size >= layout.total_byte_count:
        assigned_gradient_storage = _shared_array(
            assigned_gradient_shared_memory,
            layout.total_float_count,
        )
        assigned_gradient_storage.fill(np.float32(17.0))
    else:
        assigned_gradient_storage = np.empty(
            0,
            dtype=np.float32,
        )

    unassigned_gradient_storage = _shared_array(
        unassigned_gradient_shared_memory,
        layout.total_float_count,
    )
    unassigned_gradient_storage.fill(np.float32(-23.0))

    if weight_size > layout.total_byte_count:
        weight_shared_memory.buf[layout.total_byte_count :] = bytes([0xA5]) * (
            weight_size - layout.total_byte_count
        )

    if gradient_size > layout.total_byte_count:
        assigned_gradient_shared_memory.buf[layout.total_byte_count :] = bytes([0x5A]) * (
            gradient_size - layout.total_byte_count
        )

    if unassigned_gradient_shared_memory.size > layout.total_byte_count:
        unassigned_gradient_shared_memory.buf[layout.total_byte_count :] = bytes([0x3C]) * (
            unassigned_gradient_shared_memory.size - layout.total_byte_count
        )

    startup_config = WorkerStartupConfig(
        protocol_version=WORKER_PROTOCOL_VERSION,
        worker_index=0,
        num_layers=1,
        dtype_marker=WorkerNumericDType.FLOAT32,
        canonical_float_count=layout.total_float_count,
        sequence_count=1,
        weight_shared_memory_name=(weight_shared_memory.name),
        assigned_shards=_shard_payload(shard),
        gradient_shared_memory_names=(assigned_gradient_shared_memory.name,),
        training_sequences=_sequence_payload(sequence),
    )

    state = _ParentSharedState(
        layout=layout,
        sequence=sequence,
        shard=shard,
        weight_shared_memory=weight_shared_memory,
        assigned_gradient_shared_memory=(assigned_gradient_shared_memory),
        unassigned_gradient_shared_memory=(unassigned_gradient_shared_memory),
        weight_storage=weight_storage,
        assigned_gradient_storage=(assigned_gradient_storage),
        unassigned_gradient_storage=(unassigned_gradient_storage),
        startup_config=startup_config,
    )

    try:
        yield state
    finally:
        state.cleanup()


def _spawn_worker(
    context: SpawnContext,
    startup_config: WorkerStartupConfig,
) -> _SpawnedWorker:
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=run_transformer_worker,
        args=(
            child_connection,
            0,
            startup_config,
        ),
        daemon=False,
    )
    process.start()
    child_connection.close()

    return _SpawnedWorker(
        parent_connection,
        process,
    )


def _assert_shared_memory_remains_parent_owned(
    name: str,
) -> None:
    reopened = SharedMemory(
        name=name,
        create=False,
    )
    reopened.close()


def test_protocol_records_have_exact_frozen_slotted_structure() -> None:
    records = _valid_protocol_records()

    assert WORKER_PROTOCOL_VERSION == 1
    assert tuple(WorkerNumericDType) == (WorkerNumericDType.FLOAT32,)
    assert tuple(WorkerState) == (
        WorkerState.STARTING,
        WorkerState.READY,
        WorkerState.COMPUTING,
        WorkerState.STOPPING,
        WorkerState.STOPPED,
    )
    assert tuple(WorkerFailurePhase) == (
        WorkerFailurePhase.STARTUP,
        WorkerFailurePhase.COMMAND,
        WorkerFailurePhase.COMPUTE,
        WorkerFailurePhase.SHUTDOWN,
    )
    assert tuple(WorkerFailureCode) == (
        WorkerFailureCode.INVALID_PROTOCOL,
        WorkerFailureCode.INVALID_LAYOUT,
        WorkerFailureCode.SHARED_MEMORY,
        WorkerFailureCode.INVALID_OWNERSHIP,
        WorkerFailureCode.INVALID_STATE,
        WorkerFailureCode.NUMERICAL_FAILURE,
        WorkerFailureCode.COMMUNICATION_FAILURE,
        WorkerFailureCode.CLEANUP_FAILURE,
        WorkerFailureCode.INTERNAL_FAILURE,
    )

    for record_type, record in zip(
        _PROTOCOL_RECORD_TYPES,
        records,
        strict=True,
    ):
        assert dataclasses.is_dataclass(record_type)
        assert (
            tuple(field.name for field in fields(record_type)) == _EXPECTED_FIELD_NAMES[record_type]
        )
        assert not hasattr(record, "__dict__")
        assert pickle.loads(pickle.dumps(record)) == record

        with pytest.raises(FrozenInstanceError):
            setattr(
                record,
                fields(record_type)[0].name,
                -1,
            )


def test_protocol_records_contain_only_approved_immutable_values() -> None:
    for record in _valid_protocol_records():
        for field in fields(record):
            _assert_protocol_value_is_private(getattr(record, field.name))


def test_worker_side_validators_accept_exact_records() -> None:
    (
        startup,
        ready,
        compute,
        result,
        failure,
        stop,
        stopped,
    ) = _valid_protocol_records()

    assert (
        validate_worker_startup_config(
            startup,
            expected_worker_index=0,
        )
        is startup
    )

    assert (
        validate_ready_message(
            ready,
            expected_worker_index=0,
            expected_shard_ids=(0,),
        )
        is ready
    )

    assert (
        validate_compute_message(
            compute,
            expected_worker_index=0,
            expected_shard_ids=(0,),
        )
        is compute
    )

    assert (
        validate_result_message(
            result,
            expected_worker_index=0,
            expected_epoch=3,
            expected_shard_ids=(0,),
        )
        is result
    )

    assert (
        validate_failure_message(
            failure,
            expected_worker_index=0,
            expected_shard_ids=(0,),
        )
        is failure
    )

    assert (
        validate_stop_message(
            stop,
            expected_worker_index=0,
        )
        is stop
    )

    assert (
        validate_stopped_message(
            stopped,
            expected_worker_index=0,
        )
        is stopped
    )


@pytest.mark.parametrize(
    "message, validator",
    [
        (
            _ReadyLookalike(
                WORKER_PROTOCOL_VERSION,
                0,
                (0,),
            ),
            lambda value: validate_ready_message(
                value,
                expected_worker_index=0,
                expected_shard_ids=(0,),
            ),
        ),
        (
            {
                "protocol_version": 1,
                "worker_index": 0,
                "assigned_shard_ids": (0,),
            },
            lambda value: validate_ready_message(
                value,
                expected_worker_index=0,
                expected_shard_ids=(0,),
            ),
        ),
        (
            ReadyMessage(
                2,
                0,
                (0,),
            ),
            lambda value: validate_ready_message(
                value,
                expected_worker_index=0,
                expected_shard_ids=(0,),
            ),
        ),
        (
            ReadyMessage(
                1,
                True,
                (0,),
            ),
            lambda value: validate_ready_message(
                value,
                expected_worker_index=0,
                expected_shard_ids=(0,),
            ),
        ),
        (
            ComputeMessage(
                1,
                0,
                True,
                (0,),
            ),
            lambda value: validate_compute_message(
                value,
                expected_worker_index=0,
                expected_shard_ids=(0,),
            ),
        ),
        (
            ComputeMessage(
                1,
                0,
                0,
                (1, 0),
            ),
            lambda value: validate_compute_message(
                value,
                expected_worker_index=0,
                expected_shard_ids=(0, 1),
            ),
        ),
        (
            ResultMessage(
                1,
                0,
                4,
                (0,),
                (math.nan,),
            ),
            lambda value: validate_result_message(
                value,
                expected_worker_index=0,
                expected_epoch=4,
                expected_shard_ids=(0,),
            ),
        ),
        (
            ResultMessage(
                1,
                0,
                4,
                (0,),
                (1.0, 2.0),
            ),
            lambda value: validate_result_message(
                value,
                expected_worker_index=0,
                expected_epoch=4,
                expected_shard_ids=(0,),
            ),
        ),
        (
            ResultMessage(
                1,
                0,
                3,
                (0,),
                (1.0,),
            ),
            lambda value: validate_result_message(
                value,
                expected_worker_index=0,
                expected_epoch=4,
                expected_shard_ids=(0,),
            ),
        ),
    ],
)
def test_protocol_validators_reject_wrong_class_version_types_and_commit_identity(
    message: object,
    validator: Callable[[object], object],
) -> None:
    with pytest.raises(
        WorkerProtocolValidationError,
        match=("^Worker protocol validation failed" "\\.$"),
    ):
        validator(message)


def test_startup_validation_rejects_wrong_layout_dtype_sequences_and_ownership() -> None:
    startup = cast(
        WorkerStartupConfig,
        _valid_protocol_records()[0],
    )
    malformed = (
        dataclasses.replace(
            startup,
            protocol_version=2,
        ),
        dataclasses.replace(
            startup,
            worker_index=True,
        ),
        dataclasses.replace(
            startup,
            num_layers=0,
        ),
        dataclasses.replace(
            startup,
            dtype_marker="float32",
        ),
        dataclasses.replace(
            startup,
            canonical_float_count=(startup.canonical_float_count + 1),
        ),
        dataclasses.replace(
            startup,
            sequence_count=2,
        ),
        dataclasses.replace(
            startup,
            training_sequences=(
                (
                    (0,),
                    (1,),
                ),
            ),
        ),
        dataclasses.replace(
            startup,
            assigned_shards=((0, 0, 0),),
        ),
        dataclasses.replace(
            startup,
            assigned_shards=(
                (1, 1, 1),
                (0, 0, 1),
            ),
        ),
        dataclasses.replace(
            startup,
            gradient_shared_memory_names=(),
        ),
        dataclasses.replace(
            startup,
            gradient_shared_memory_names=(startup.weight_shared_memory_name,),
        ),
    )

    for candidate in malformed:
        with pytest.raises(WorkerProtocolValidationError):
            validate_worker_startup_config(
                candidate,
                expected_worker_index=0,
            )


def test_real_spawn_publishes_one_complete_owned_shard_and_stops_cleanly() -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state(extra_bytes=64) as state:
        weight_before = state.weight_storage.tobytes(order="C")
        unassigned_before = state.unassigned_gradient_storage.tobytes(order="C")
        weight_tail_before = bytes(state.weight_shared_memory.buf[state.layout.total_byte_count :])
        gradient_tail_before = bytes(
            state.assigned_gradient_shared_memory.buf[state.layout.total_byte_count :]
        )

        direct_parameters = initialize_transformer_parameters(
            state.layout,
            Mulberry32(42),
        )
        direct = calculate_logical_training_shard(
            (state.sequence,),
            state.shard,
            direct_parameters.views,
        )
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            ready = validate_ready_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(state.shard.shard_index,),
            )
            assert ready.assigned_shard_ids == (state.shard.shard_index,)

            worker.parent_connection.send(
                ComputeMessage(
                    WORKER_PROTOCOL_VERSION,
                    0,
                    7,
                    ready.assigned_shard_ids,
                )
            )
            result = validate_result_message(
                worker.receive(60.0),
                expected_worker_index=0,
                expected_epoch=7,
                expected_shard_ids=(ready.assigned_shard_ids),
            )

            assert result.shard_losses == (direct.loss,)
            np.testing.assert_array_equal(
                state.assigned_gradient_storage,
                direct.gradient.storage,
            )
            assert state.weight_storage.tobytes(order="C") == weight_before
            assert state.unassigned_gradient_storage.tobytes(order="C") == unassigned_before
            assert (
                bytes(state.weight_shared_memory.buf[state.layout.total_byte_count :])
                == weight_tail_before
            )
            assert (
                bytes(state.assigned_gradient_shared_memory.buf[state.layout.total_byte_count :])
                == gradient_tail_before
            )

            worker.parent_connection.send(
                StopMessage(
                    WORKER_PROTOCOL_VERSION,
                    0,
                )
            )
            validate_stopped_message(
                worker.receive(),
                expected_worker_index=0,
            )
            worker.join(0)

            _assert_shared_memory_remains_parent_owned(state.weight_shared_memory.name)
            _assert_shared_memory_remains_parent_owned(state.assigned_gradient_shared_memory.name)
        finally:
            worker.close()


def test_real_spawn_preserves_exact_empty_shard_behavior() -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state(assigned_shard_id=1) as state:
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            validate_ready_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(1,),
            )
            worker.parent_connection.send(
                ComputeMessage(
                    WORKER_PROTOCOL_VERSION,
                    0,
                    0,
                    (1,),
                )
            )
            result = validate_result_message(
                worker.receive(60.0),
                expected_worker_index=0,
                expected_epoch=0,
                expected_shard_ids=(1,),
            )

            assert result.shard_losses == (0.0,)
            np.testing.assert_array_equal(
                state.assigned_gradient_storage,
                np.zeros(
                    state.layout.total_float_count,
                    dtype=np.float32,
                ),
            )

            worker.parent_connection.send(
                StopMessage(
                    WORKER_PROTOCOL_VERSION,
                    0,
                )
            )
            validate_stopped_message(
                worker.receive(),
                expected_worker_index=0,
            )
            worker.join(0)
        finally:
            worker.close()


@pytest.mark.parametrize(
    "capacity_target",
    [
        "weight",
        "gradient",
    ],
)
def test_real_spawn_rejects_undersized_shared_memory_without_unlinking(
    capacity_target: str,
) -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )
    layout = build_transformer_parameter_layout(1)
    undersized = max(
        1,
        layout.total_byte_count // 2,
    )
    overrides = {
        "weight_size_override": (undersized if capacity_target == "weight" else None),
        "gradient_size_override": (undersized if capacity_target == "gradient" else None),
    }

    with _parent_shared_state(**overrides) as state:
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            failure = validate_failure_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            assert failure.phase is WorkerFailurePhase.STARTUP
            assert failure.code is WorkerFailureCode.SHARED_MEMORY
            assert failure.epoch is None
            worker.join(1)

            _assert_shared_memory_remains_parent_owned(state.weight_shared_memory.name)
            _assert_shared_memory_remains_parent_owned(state.assigned_gradient_shared_memory.name)
        finally:
            worker.close()


@pytest.mark.parametrize(
    "case_name",
    [
        "unsupported-layer",
        "wrong-dtype",
        "duplicate-shard",
        "missing-gradient-name",
        "weight-name-reused-as-gradient",
    ],
)
def test_real_spawn_rejects_invalid_startup_protocol_variants(
    case_name: str,
) -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state() as state:
        shard_zero = build_logical_training_shards(1)[0]
        config = state.startup_config
        expected_code = WorkerFailureCode.INVALID_PROTOCOL

        if case_name == "unsupported-layer":
            config = dataclasses.replace(
                config,
                num_layers=0,
            )
            expected_code = WorkerFailureCode.INVALID_LAYOUT
        elif case_name == "wrong-dtype":
            config = dataclasses.replace(
                config,
                dtype_marker="float32",
            )
            expected_code = WorkerFailureCode.INVALID_LAYOUT
        elif case_name == "duplicate-shard":
            expected_code = WorkerFailureCode.INVALID_OWNERSHIP
            shard_descriptor = (
                shard_zero.shard_index,
                shard_zero.start_index,
                shard_zero.stop_index,
            )
            config = dataclasses.replace(
                config,
                assigned_shards=(
                    shard_descriptor,
                    shard_descriptor,
                ),
                gradient_shared_memory_names=(
                    state.assigned_gradient_shared_memory.name,
                    state.unassigned_gradient_shared_memory.name,
                ),
            )
        elif case_name == "missing-gradient-name":
            config = dataclasses.replace(
                config,
                gradient_shared_memory_names=(),
            )
            expected_code = WorkerFailureCode.INVALID_OWNERSHIP
        elif case_name == "weight-name-reused-as-gradient":
            config = dataclasses.replace(
                config,
                gradient_shared_memory_names=(state.weight_shared_memory.name,),
            )
            expected_code = WorkerFailureCode.INVALID_OWNERSHIP
        else:
            pytest.fail(f"Unhandled startup case: {case_name}")

        worker = _spawn_worker(
            context,
            config,
        )

        try:
            failure = validate_failure_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(),
            )
            assert failure.phase is WorkerFailurePhase.STARTUP
            assert failure.code is expected_code
            assert failure.epoch is None
            assert state.weight_shared_memory.name not in repr(failure)
            assert state.assigned_gradient_shared_memory.name not in repr(failure)
            worker.join(1)

            _assert_shared_memory_remains_parent_owned(state.weight_shared_memory.name)
            _assert_shared_memory_remains_parent_owned(state.assigned_gradient_shared_memory.name)
        finally:
            worker.close()


@pytest.mark.parametrize(
    "malformed_command",
    [
        {
            "protocol_version": 1,
            "worker_index": 0,
        },
        ReadyMessage(
            1,
            0,
            (0,),
        ),
        ComputeMessage(
            2,
            0,
            0,
            (0,),
        ),
        ComputeMessage(
            1,
            1,
            0,
            (0,),
        ),
    ],
)
def test_real_spawn_rejects_malformed_or_illegal_parent_commands(
    malformed_command: object,
) -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state() as state:
        gradient_before = state.assigned_gradient_storage.tobytes(order="C")
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            validate_ready_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            worker.parent_connection.send(malformed_command)
            failure = validate_failure_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            assert failure.phase is WorkerFailurePhase.COMMAND
            assert failure.code is WorkerFailureCode.INVALID_PROTOCOL
            assert failure.epoch is None
            assert state.assigned_gradient_storage.tobytes(order="C") == gradient_before
            worker.join(1)
        finally:
            worker.close()


def test_real_spawn_rejects_second_outstanding_compute_without_commit_marker() -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state() as state:
        weight_before = state.weight_storage.tobytes(order="C")
        unassigned_before = state.unassigned_gradient_storage.tobytes(order="C")
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            validate_ready_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            worker.parent_connection.send(
                ComputeMessage(
                    1,
                    0,
                    0,
                    (0,),
                )
            )
            worker.parent_connection.send(
                ComputeMessage(
                    1,
                    0,
                    1,
                    (0,),
                )
            )

            message = worker.receive(60.0)
            assert type(message) is FailureMessage
            failure = validate_failure_message(
                message,
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            assert failure.phase is WorkerFailurePhase.COMMAND
            assert failure.code is WorkerFailureCode.INVALID_STATE
            assert failure.epoch == 0
            assert state.weight_storage.tobytes(order="C") == weight_before
            assert state.unassigned_gradient_storage.tobytes(order="C") == unassigned_before
            worker.join(1)
        finally:
            worker.close()


def test_real_spawn_rejects_nonfinite_weights_after_ready_and_sends_no_result() -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state() as state:
        unassigned_before = state.unassigned_gradient_storage.tobytes(order="C")
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            validate_ready_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            state.weight_storage[0] = np.float32(np.nan)
            worker.parent_connection.send(
                ComputeMessage(
                    1,
                    0,
                    5,
                    (0,),
                )
            )
            failure = validate_failure_message(
                worker.receive(60.0),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            assert failure.phase is WorkerFailurePhase.COMPUTE
            assert failure.code is WorkerFailureCode.NUMERICAL_FAILURE
            assert failure.epoch == 5

            np.testing.assert_array_equal(
                state.assigned_gradient_storage,
                np.zeros(
                    state.layout.total_float_count,
                    dtype=np.float32,
                ),
            )
            assert state.unassigned_gradient_storage.tobytes(order="C") == unassigned_before
            worker.join(1)
        finally:
            worker.close()


def test_real_spawn_accepts_stop_before_compute() -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state() as state:
        gradient_before = state.assigned_gradient_storage.tobytes(order="C")
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            validate_ready_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )
            worker.parent_connection.send(
                StopMessage(
                    1,
                    0,
                )
            )
            validate_stopped_message(
                worker.receive(),
                expected_worker_index=0,
            )
            worker.join(0)

            assert state.assigned_gradient_storage.tobytes(order="C") == gradient_before
        finally:
            worker.close()


def test_real_spawn_accepts_two_serial_compute_commands() -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    with _parent_shared_state() as state:
        worker = _spawn_worker(
            context,
            state.startup_config,
        )

        try:
            validate_ready_message(
                worker.receive(),
                expected_worker_index=0,
                expected_shard_ids=(0,),
            )

            results: list[ResultMessage] = []

            for epoch in (0, 1):
                worker.parent_connection.send(
                    ComputeMessage(
                        1,
                        0,
                        epoch,
                        (0,),
                    )
                )
                results.append(
                    validate_result_message(
                        worker.receive(60.0),
                        expected_worker_index=0,
                        expected_epoch=epoch,
                        expected_shard_ids=(0,),
                    )
                )

            assert results[0].shard_losses == results[1].shard_losses

            worker.parent_connection.send(
                StopMessage(
                    1,
                    0,
                )
            )
            validate_stopped_message(
                worker.receive(),
                expected_worker_index=0,
            )
            worker.join(0)
        finally:
            worker.close()


def test_repeated_real_spawn_workers_do_not_reuse_process_state() -> None:
    context = cast(
        SpawnContext,
        mp.get_context("spawn"),
    )

    for epoch in (11, 12):
        with _parent_shared_state() as state:
            worker = _spawn_worker(
                context,
                state.startup_config,
            )

            try:
                validate_ready_message(
                    worker.receive(),
                    expected_worker_index=0,
                    expected_shard_ids=(0,),
                )
                worker.parent_connection.send(
                    ComputeMessage(
                        1,
                        0,
                        epoch,
                        (0,),
                    )
                )
                result = validate_result_message(
                    worker.receive(60.0),
                    expected_worker_index=0,
                    expected_epoch=epoch,
                    expected_shard_ids=(0,),
                )
                assert result.epoch == epoch
                assert not worker.parent_connection.poll()

                worker.parent_connection.send(
                    StopMessage(
                        1,
                        0,
                    )
                )
                validate_stopped_message(
                    worker.receive(),
                    expected_worker_index=0,
                )
                worker.join(0)
            finally:
                worker.close()
