# backend/src/how_llms_work/routes/train_transformer.py

"""FastAPI streaming and deterministic persistence for Transformer Training Runs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from asyncio import sleep as presentation_sleep
from collections.abc import AsyncIterator, Callable
from functools import partial
from pathlib import Path
from threading import Event, Lock
from typing import Final, TypeVar, cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from how_llms_work.ml.math_utils import Mulberry32
from how_llms_work.ml.transformer import (
    TRANSFORMER_ATTENTION_HEAD_COUNT,
    TRANSFORMER_CONTEXT_LENGTH,
    TRANSFORMER_EMBEDDING_DIMENSION,
    TRANSFORMER_FEED_FORWARD_DIMENSION,
    TRANSFORMER_MAX_LAYER_COUNT,
    TRANSFORMER_MIN_LAYER_COUNT,
    SavedTransformerModel,
    TransformerParameterLayout,
    TransformerPreprocessingSnapshot,
    build_saved_transformer_model,
    build_transformer_parameter_layout,
    create_transformer_training_run,
    evaluate_transformer_final_loss,
    generate_transformer_text,
    get_transformer_preprocessing,
    initialize_transformer_parameters,
    transformer_parameter_count,
)
from how_llms_work.ml.transformer_worker import (
    RequestScopedWorkerGroup,
    RequestScopedWorkerGroupCleanupReport,
    create_request_scoped_worker_group,
)
from how_llms_work.schemas import TrainTransformerRequest
from how_llms_work.sse import create_sse_response, format_sse

logger = logging.getLogger(__name__)
router = APIRouter()
_TRANSFORMER_RUN_SLOT = Lock()


_TRANSFORMER_MIN_EPOCH_COUNT: Final = 50
_TRANSFORMER_MAX_EPOCH_COUNT: Final = 2_000
PRESENTATION_DELAY_SECONDS: Final = 0.02
_TRANSFORMER_HELPER_TIMEOUT_SECONDS: Final = 300.0
_TRANSFORMER_HELPER_POLL_SECONDS: Final = 0.1

_HelperResultT = TypeVar("_HelperResultT")


_MODEL_KEYS: Final = (
    "type",
    "config",
    "vocab",
    "merges",
    "weights",
)

_CONFIG_KEYS: Final = (
    "vocabSize",
    "contextLen",
    "embDim",
    "numHeads",
    "ffDim",
    "numLayers",
)

_WEIGHT_KEYS: Final = (
    "tokEmb",
    "posEmb",
    "blocks",
    "lnFGamma",
    "lnFBeta",
    "headW",
    "headB",
)

_BLOCK_KEYS: Final = (
    "ln1Gamma",
    "ln1Beta",
    "wQ",
    "bQ",
    "wK",
    "bK",
    "wV",
    "bV",
    "wO",
    "bO",
    "ln2Gamma",
    "ln2Beta",
    "ff1W",
    "ff1B",
    "ff2W",
    "ff2B",
)

_MERGE_KEYS: Final = (
    "pair",
    "merged",
)

_TOP_LEVEL_WEIGHT_ARRAY_KEYS: Final = (
    "tokEmb",
    "posEmb",
    "lnFGamma",
    "lnFBeta",
    "headW",
    "headB",
)


def _require_exact_dictionary(
    value: object,
    *,
    name: str,
    expected_keys: tuple[str, ...],
) -> dict[str, object]:
    """Return one dictionary after validating its insertion-ordered keys."""
    if type(value) is not dict:
        raise TypeError(f"{name} must be a dictionary.")

    mapping = cast(dict[str, object], value)

    if tuple(mapping) != expected_keys:
        raise ValueError(
            f"{name} must contain exactly these keys in order: {', '.join(expected_keys)}."
        )

    return mapping


def _require_list(
    value: object,
    *,
    name: str,
) -> list[object]:
    """Return one value after requiring an ordinary list container."""
    if type(value) is not list:
        raise TypeError(f"{name} must be a list.")

    return cast(list[object], value)


def _validate_bounded_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Return one strict integer after validating its inclusive bounds."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")

    integer_value = value

    if integer_value < minimum or integer_value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")

    return integer_value


def _validate_fixed_integer(
    value: object,
    *,
    name: str,
    expected: int,
) -> None:
    """Require one strict integer to equal a fixed architecture value."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")

    if value != expected:
        raise ValueError(f"{name} must equal {expected}.")


def _validate_saved_transformer_model_structure(
    model: SavedTransformerModel,
) -> int:
    """Validate persistence-relevant Saved Transformer Model structure."""
    model_mapping = _require_exact_dictionary(
        model,
        name="Saved Transformer Model",
        expected_keys=_MODEL_KEYS,
    )

    model_type = model_mapping["type"]

    if type(model_type) is not str:
        raise TypeError("Saved Transformer Model type must be a string.")

    if model_type != "decoder-transformer":
        raise ValueError("Saved Transformer Model type must be 'decoder-transformer'.")

    config = _require_exact_dictionary(
        model_mapping["config"],
        name="Saved Transformer Model config",
        expected_keys=_CONFIG_KEYS,
    )

    vocab_size = _validate_bounded_integer(
        config["vocabSize"],
        name="Saved Transformer Model vocabSize",
        minimum=1,
        maximum=2**31 - 1,
    )

    _validate_fixed_integer(
        config["contextLen"],
        name="Saved Transformer Model contextLen",
        expected=TRANSFORMER_CONTEXT_LENGTH,
    )
    _validate_fixed_integer(
        config["embDim"],
        name="Saved Transformer Model embDim",
        expected=TRANSFORMER_EMBEDDING_DIMENSION,
    )
    _validate_fixed_integer(
        config["numHeads"],
        name="Saved Transformer Model numHeads",
        expected=TRANSFORMER_ATTENTION_HEAD_COUNT,
    )
    _validate_fixed_integer(
        config["ffDim"],
        name="Saved Transformer Model ffDim",
        expected=TRANSFORMER_FEED_FORWARD_DIMENSION,
    )

    num_layers = _validate_bounded_integer(
        config["numLayers"],
        name="Saved Transformer Model numLayers",
        minimum=TRANSFORMER_MIN_LAYER_COUNT,
        maximum=TRANSFORMER_MAX_LAYER_COUNT,
    )

    vocabulary = _require_list(
        model_mapping["vocab"],
        name="Saved Transformer Model vocab",
    )

    if len(vocabulary) != vocab_size:
        raise ValueError("Saved Transformer Model vocab length must equal config vocabSize.")

    if any(type(token) is not str for token in vocabulary):
        raise TypeError("Every Saved Transformer Model vocab entry must be a string.")

    merges = _require_list(
        model_mapping["merges"],
        name="Saved Transformer Model merges",
    )

    for merge_index, merge_value in enumerate(merges):
        merge = _require_exact_dictionary(
            merge_value,
            name=f"Saved Transformer Model merge {merge_index}",
            expected_keys=_MERGE_KEYS,
        )

        pair = _require_list(
            merge["pair"],
            name=f"Saved Transformer Model merge {merge_index} pair",
        )

        if len(pair) != 2:
            raise ValueError(
                f"Saved Transformer Model merge {merge_index} pair "
                "must contain exactly two strings."
            )

        if any(type(token) is not str for token in pair):
            raise TypeError(
                f"Saved Transformer Model merge {merge_index} pair must contain only strings."
            )

        if type(merge["merged"]) is not str:
            raise TypeError(
                f"Saved Transformer Model merge {merge_index} merged value must be a string."
            )

    weights = _require_exact_dictionary(
        model_mapping["weights"],
        name="Saved Transformer Model weights",
        expected_keys=_WEIGHT_KEYS,
    )

    for weight_name in _TOP_LEVEL_WEIGHT_ARRAY_KEYS:
        _require_list(
            weights[weight_name],
            name=f"Saved Transformer Model weights {weight_name}",
        )

    blocks = _require_list(
        weights["blocks"],
        name="Saved Transformer Model weights blocks",
    )

    if len(blocks) != num_layers:
        raise ValueError("Saved Transformer Model block count must equal config numLayers.")

    for block_index, block_value in enumerate(blocks):
        block = _require_exact_dictionary(
            block_value,
            name=f"Saved Transformer Model block {block_index}",
            expected_keys=_BLOCK_KEYS,
        )

        for weight_name in _BLOCK_KEYS:
            _require_list(
                block[weight_name],
                name=(f"Saved Transformer Model block {block_index} weight {weight_name}"),
            )

    return num_layers


def _validate_transformer_epochs(epochs: int) -> int:
    """Validate the strict inclusive Transformer epoch range."""
    return _validate_bounded_integer(
        epochs,
        name="epochs",
        minimum=_TRANSFORMER_MIN_EPOCH_COUNT,
        maximum=_TRANSFORMER_MAX_EPOCH_COUNT,
    )


def _build_transformer_model_filename(
    *,
    epochs: int,
    num_layers: int,
) -> str:
    """Build the exact configuration-specific model filename."""
    return (
        f"transformer-weights-e{epochs}-l{num_layers}"
        f"-d{TRANSFORMER_EMBEDDING_DIMENSION}"
        f"-h{TRANSFORMER_ATTENTION_HEAD_COUNT}"
        f"-ff{TRANSFORMER_FEED_FORWARD_DIMENSION}"
        f"-ctx{TRANSFORMER_CONTEXT_LENGTH}.json"
    )


def get_transformer_model_directory() -> Path:
    """Return the backend project's Saved Transformer Model directory."""
    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / ".data"


def get_transformer_model_filename(
    model: SavedTransformerModel,
    *,
    epochs: int,
) -> str:
    """Return the exact filename for one validated configuration."""
    num_layers = _validate_saved_transformer_model_structure(model)
    validated_epochs = _validate_transformer_epochs(epochs)

    return _build_transformer_model_filename(
        epochs=validated_epochs,
        num_layers=num_layers,
    )


def serialize_saved_transformer_model(
    model: SavedTransformerModel,
) -> str:
    """Serialize one complete ordered Saved Transformer Model in memory."""
    _validate_saved_transformer_model_structure(model)

    return f"{json.dumps(model, indent=2, allow_nan=False)}\n"


def prepare_transformer_model_persistence(
    model: SavedTransformerModel,
    *,
    epochs: int,
    model_directory: Path | None = None,
) -> tuple[Path, str]:
    """Prepare the destination and complete document without filesystem I/O."""
    num_layers = _validate_saved_transformer_model_structure(model)
    validated_epochs = _validate_transformer_epochs(epochs)

    document = f"{json.dumps(model, indent=2, allow_nan=False)}\n"

    directory = get_transformer_model_directory() if model_directory is None else model_directory

    destination = directory / _build_transformer_model_filename(
        epochs=validated_epochs,
        num_layers=num_layers,
    )

    return destination, document


def create_temporary_transformer_model_file(
    directory: Path,
    destination: Path,
) -> tuple[int, Path]:
    """Create one secure unique same-directory temporary model file."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=directory,
    )

    return file_descriptor, Path(temporary_name)


def write_transformer_model_document(
    file_descriptor: int,
    document: str,
) -> None:
    """Write, flush, synchronize, and close one complete model document."""
    with os.fdopen(
        file_descriptor,
        mode="w",
        encoding="utf-8",
        newline="\n",
    ) as writer:
        writer.write(document)
        writer.flush()
        os.fsync(writer.fileno())


def replace_transformer_model_file(
    source: Path,
    destination: Path,
) -> None:
    """Atomically replace one configuration-specific model destination."""
    os.replace(source, destination)


def remove_temporary_transformer_model(path: Path) -> None:
    """Remove one owned temporary model file when it still exists."""
    path.unlink(missing_ok=True)


def save_transformer_model(
    model: SavedTransformerModel,
    *,
    epochs: int,
    model_directory: Path | None = None,
) -> Path:
    """Persist one complete Saved Transformer Model through atomic replacement."""
    destination, document = prepare_transformer_model_persistence(
        model,
        epochs=epochs,
        model_directory=model_directory,
    )

    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_path = create_temporary_transformer_model_file(
        directory,
        destination,
    )

    try:
        write_transformer_model_document(
            file_descriptor,
            document,
        )
        replace_transformer_model_file(
            temporary_path,
            destination,
        )
    except Exception as persistence_error:
        try:
            remove_temporary_transformer_model(temporary_path)
        except Exception as cleanup_error:
            raise ExceptionGroup(
                "Saved Transformer Model persistence and temporary-file cleanup failed",
                [
                    persistence_error,
                    cleanup_error,
                ],
            ) from None

        raise

    return destination


class _TransformerClientDisconnected(RuntimeError):
    """Private control-flow signal for a disconnected Transformer stream."""

    __slots__ = ()


def _build_transformer_init_payload(
    *,
    preprocessing: TransformerPreprocessingSnapshot,
    layout: TransformerParameterLayout,
    num_layers: int,
    total_parameters: int,
    temperature: float,
    top_p: float,
) -> dict[str, object]:
    """Build the exact pre-stream Transformer init payload."""
    vocabulary_size = len(preprocessing.vocabulary)

    if layout.num_layers != num_layers:
        raise ValueError("Transformer layout layer count does not match the request.")

    if layout.vocabulary_size != vocabulary_size:
        raise ValueError("Transformer layout Vocabulary size does not match preprocessing.")

    if layout.total_float_count != total_parameters:
        raise ValueError("Transformer parameter count does not match the canonical layout.")

    return {
        "vocabSize": vocabulary_size,
        "contextLen": TRANSFORMER_CONTEXT_LENGTH,
        "embeddingDim": TRANSFORMER_EMBEDDING_DIMENSION,
        "numHeads": TRANSFORMER_ATTENTION_HEAD_COUNT,
        "ffDim": TRANSFORMER_FEED_FORWARD_DIMENSION,
        "numLayers": num_layers,
        "totalParams": total_parameters,
        "temperature": temperature,
        "topP": top_p,
        "corpusSentences": len(preprocessing.corpus),
        "trainingSequences": len(preprocessing.training_sequences),
    }


async def request_is_disconnected(
    request: Request,
) -> bool:
    """Return whether the current streaming client has disconnected."""
    return await request.is_disconnected()


async def _raise_if_transformer_client_disconnected(
    request: Request,
    cancellation_event: Event,
) -> None:
    """Set cooperative cancellation and stop when the client disconnects."""
    if await request_is_disconnected(request):
        cancellation_event.set()
        raise _TransformerClientDisconnected


async def _run_bounded_transformer_helper(
    operation: Callable[[], _HelperResultT],
    *,
    request: Request,
    cancellation_event: Event,
) -> _HelperResultT:
    """Run, observe, bound, and fully drain one cooperative thread helper."""
    helper_task = asyncio.create_task(asyncio.to_thread(operation))
    event_loop = asyncio.get_running_loop()
    deadline = event_loop.time() + _TRANSFORMER_HELPER_TIMEOUT_SECONDS

    async def drain_helper_task() -> None:
        """Wait until the helper no longer accesses request-owned state."""
        try:
            await asyncio.shield(helper_task)
        except Exception:
            logger.exception("Transformer helper failed while being drained")

    try:
        while True:
            if helper_task.done():
                return helper_task.result()

            remaining_seconds = deadline - event_loop.time()

            if remaining_seconds <= 0.0:
                raise TimeoutError

            completed_tasks, _pending_tasks = await asyncio.wait(
                {helper_task},
                timeout=min(
                    _TRANSFORMER_HELPER_POLL_SECONDS,
                    remaining_seconds,
                ),
            )

            if helper_task in completed_tasks:
                return helper_task.result()

            await _raise_if_transformer_client_disconnected(
                request,
                cancellation_event,
            )
    except _TransformerClientDisconnected:
        cancellation_event.set()
        await drain_helper_task()
        raise
    except TimeoutError:
        cancellation_event.set()
        await drain_helper_task()
        raise
    except asyncio.CancelledError:
        cancellation_event.set()
        await drain_helper_task()
        raise


def _log_transformer_worker_cleanup_diagnostics(
    worker_group: RequestScopedWorkerGroup,
    cleanup_report: RequestScopedWorkerGroupCleanupReport | None,
) -> None:
    """Log only sanitized worker-group completion and cleanup categories."""
    primary_failure_code = worker_group.primary_failure_code

    if primary_failure_code is not None:
        logger.error(
            "Transformer worker group closed with primary failure code: %s",
            primary_failure_code.value,
        )

    if cleanup_report is None:
        logger.error("Transformer worker cleanup did not publish a cleanup report")
        return

    if cleanup_report.successful and primary_failure_code is None:
        return

    logger.error(
        "Transformer worker cleanup was not fully successful: "
        "cooperative_shutdown_completed=%s terminate_required=%s "
        "kill_required=%s nonzero_or_unknown_exit=%s "
        "secondary_failure_codes=%s",
        cleanup_report.cooperative_shutdown_completed,
        cleanup_report.terminate_required,
        cleanup_report.kill_required,
        any(exit_code != 0 for exit_code in cleanup_report.process_exit_codes),
        tuple(failure.value for failure in cleanup_report.secondary_failures),
    )


async def _cleanup_transformer_worker_group(
    worker_group: RequestScopedWorkerGroup,
    cancellation_event: Event,
) -> RequestScopedWorkerGroupCleanupReport | None:
    """Finish one idempotent worker cleanup attempt without losing cancellation."""
    cleanup_task = asyncio.create_task(worker_group.cleanup())

    try:
        return await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        cancellation_event.set()

        try:
            await cleanup_task
        except Exception:
            logger.exception("Transformer worker cleanup failed during cancellation")

        raise
    except Exception:
        logger.exception("Transformer worker cleanup failed")
        return None


async def _run_unbounded_transformer_helper(
    operation: Callable[[], _HelperResultT],
    *,
    request: Request,
    cancellation_event: Event,
) -> _HelperResultT:
    """Run, observe, and fully drain one unbounded thread helper."""
    helper_task = asyncio.create_task(asyncio.to_thread(operation))

    async def drain_helper_task() -> None:
        """Wait until the helper no longer owns finalization state."""
        try:
            await asyncio.shield(helper_task)
        except Exception:
            logger.exception("Transformer finalization helper failed while being drained")

    try:
        while True:
            if helper_task.done():
                return helper_task.result()

            completed_tasks, _pending_tasks = await asyncio.wait(
                {helper_task},
                timeout=_TRANSFORMER_HELPER_POLL_SECONDS,
            )

            if helper_task in completed_tasks:
                return helper_task.result()

            await _raise_if_transformer_client_disconnected(
                request,
                cancellation_event,
            )
    except _TransformerClientDisconnected:
        cancellation_event.set()
        await drain_helper_task()
        raise
    except asyncio.CancelledError:
        cancellation_event.set()
        await drain_helper_task()
        raise


async def stream_transformer_training(
    *,
    request: Request,
    init_payload: dict[str, object],
    preprocessing: TransformerPreprocessingSnapshot,
    layout: TransformerParameterLayout,
    epochs: int,
    temperature: float,
    top_p: float,
    num_layers: int,
    max_tokens: int,
) -> AsyncIterator[str]:
    """Stream one fresh Transformer Training Run through durable completion."""
    cancellation_event = Event()
    worker_group: RequestScopedWorkerGroup | None = None
    cleanup_diagnostics_logged = False

    try:
        yield format_sse(
            "init",
            init_payload,
        )

        samples: list[dict[str, object]] = []

        await _raise_if_transformer_client_disconnected(
            request,
            cancellation_event,
        )

        initialized_parameters = initialize_transformer_parameters(
            layout,
            Mulberry32(42),
        )
        training_run = create_transformer_training_run(
            initialized_parameters,
            sequence_count=len(preprocessing.training_sequences),
            requested_epochs=epochs,
        )

        await _raise_if_transformer_client_disconnected(
            request,
            cancellation_event,
        )

        async def observe_worker_poll() -> None:
            await _raise_if_transformer_client_disconnected(
                request,
                cancellation_event,
            )

        worker_group = await create_request_scoped_worker_group(
            num_layers,
            training_run.weights,
            preprocessing.training_sequences,
            training_run.logical_training_shards,
            poll_observer=observe_worker_poll,
        )

        while training_run.is_active:
            await _raise_if_transformer_client_disconnected(
                request,
                cancellation_event,
            )

            epoch = training_run.next_epoch

            shard_results = await worker_group.compute_epoch(
                epoch,
                training_run.weights,
            )

            observation = training_run.advance_epoch(shard_results)
            update = observation.update

            if update is None:
                continue

            await _raise_if_transformer_client_disconnected(
                request,
                cancellation_event,
            )

            generated_sample = await _run_bounded_transformer_helper(
                partial(
                    generate_transformer_text,
                    training_run.parameters,
                    preprocessing,
                    epoch=update.epoch,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    cancellation_event=cancellation_event,
                ),
                request=request,
                cancellation_event=cancellation_event,
            )

            await _raise_if_transformer_client_disconnected(
                request,
                cancellation_event,
            )

            if generated_sample.epoch != update.epoch:
                raise ValueError("Generated Text Sample epoch does not match its report epoch.")

            samples.append(
                {
                    "epoch": update.epoch,
                    "text": generated_sample.text,
                }
            )

            yield format_sse(
                "epoch",
                {
                    "epoch": update.epoch,
                    "loss": update.loss,
                    "sample": generated_sample.text,
                },
            )

            await _raise_if_transformer_client_disconnected(
                request,
                cancellation_event,
            )

            await presentation_sleep(PRESENTATION_DELAY_SECONDS)

        await _raise_if_transformer_client_disconnected(
            request,
            cancellation_event,
        )

        final_loss = await _run_bounded_transformer_helper(
            partial(
                evaluate_transformer_final_loss,
                training_run,
                preprocessing,
                cancellation_event=cancellation_event,
            ),
            request=request,
            cancellation_event=cancellation_event,
        )

        if type(final_loss) is not float:
            raise TypeError("Final Transformer loss must be a float.")

        await _raise_if_transformer_client_disconnected(
            request,
            cancellation_event,
        )

        cleanup_report = await _cleanup_transformer_worker_group(
            worker_group,
            cancellation_event,
        )
        _log_transformer_worker_cleanup_diagnostics(
            worker_group,
            cleanup_report,
        )
        cleanup_diagnostics_logged = True

        if cleanup_report is None or not worker_group.successful:
            cancellation_event.set()
            return

        await _raise_if_transformer_client_disconnected(
            request,
            cancellation_event,
        )

        saved_model = await _run_unbounded_transformer_helper(
            partial(
                build_saved_transformer_model,
                training_run,
                preprocessing,
            ),
            request=request,
            cancellation_event=cancellation_event,
        )

        await _raise_if_transformer_client_disconnected(
            request,
            cancellation_event,
        )

        model_num_layers = saved_model["config"]["numLayers"]

        if type(model_num_layers) is not int:
            raise TypeError("Saved Transformer Model numLayers must be an integer.")

        if model_num_layers != num_layers:
            raise ValueError("Saved Transformer Model layer count does not match the request.")

        await _run_unbounded_transformer_helper(
            partial(
                save_transformer_model,
                saved_model,
                epochs=epochs,
            ),
            request=request,
            cancellation_event=cancellation_event,
        )

        await _raise_if_transformer_client_disconnected(
            request,
            cancellation_event,
        )

        done_samples = [dict(sample) for sample in samples]

        yield format_sse(
            "done",
            {
                "architecture": (
                    "Decoder-Only Transformer " f"({model_num_layers} layers, 32d, 2h, 128ff)"
                ),
                "finalLoss": final_loss,
                "samples": done_samples,
            },
        )
        return
    except _TransformerClientDisconnected:
        cancellation_event.set()
        return
    except asyncio.CancelledError:
        cancellation_event.set()
        raise
    except Exception:
        cancellation_event.set()
        logger.exception("Transformer Training Run stream failed")
        return
    finally:
        try:
            if worker_group is not None:
                final_cleanup_report = await _cleanup_transformer_worker_group(
                    worker_group,
                    cancellation_event,
                )

                if not cleanup_diagnostics_logged:
                    _log_transformer_worker_cleanup_diagnostics(
                        worker_group,
                        final_cleanup_report,
                    )
        finally:
            _TRANSFORMER_RUN_SLOT.release()


@router.post("/train-transformer")
async def train_transformer(
    payload: TrainTransformerRequest,
    request: Request,
) -> StreamingResponse:
    """Validate, reserve, prepare, and start one Transformer Event Stream."""
    if not _TRANSFORMER_RUN_SLOT.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="A Transformer Training Run is already active.",
        )

    try:
        epochs = payload.epochs
        temperature = payload.temperature
        top_p = payload.top_p
        num_layers = payload.num_layers
        max_tokens = payload.max_tokens

        preprocessing = get_transformer_preprocessing()
        layout = build_transformer_parameter_layout(num_layers)
        total_parameters = transformer_parameter_count(num_layers)

        init_payload = _build_transformer_init_payload(
            preprocessing=preprocessing,
            layout=layout,
            num_layers=num_layers,
            total_parameters=total_parameters,
            temperature=temperature,
            top_p=top_p,
        )

        response = create_sse_response(
            stream_transformer_training(
                request=request,
                init_payload=init_payload,
                preprocessing=preprocessing,
                layout=layout,
                epochs=epochs,
                temperature=temperature,
                top_p=top_p,
                num_layers=num_layers,
                max_tokens=max_tokens,
            )
        )
    except Exception:
        logger.exception("Transformer Training Run preparation failed")
        _TRANSFORMER_RUN_SLOT.release()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transformer training could not start.",
        ) from None

    return response
