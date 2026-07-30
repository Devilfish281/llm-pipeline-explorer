# src/backend/src/how_llms_work/routes/neural_net.py

"""FastAPI streaming and persistence for XOR neural-network Training Runs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from asyncio import sleep as presentation_sleep
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Final, cast

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from how_llms_work.ml.neural_net import (
    EpochUpdate,
    SavedNetwork,
    TrainingEvent,
    create_training_run,
)
from how_llms_work.schemas import NeuralNetRequest
from how_llms_work.sse import create_sse_response, format_sse

logger = logging.getLogger(__name__)
router = APIRouter()

SINGLE_LAYER_SNAPSHOT_FILENAME: Final = "single-layer-weights.json"
MULTI_LAYER_SNAPSHOT_FILENAME: Final = "multi-layer-weights.json"
PRESENTATION_DELAY_SECONDS: Final = 0.02


def get_snapshot_directory() -> Path:
    """Return the backend project's mode-specific snapshot directory."""
    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / ".data"


def get_snapshot_filename(weights: SavedNetwork) -> str:
    """Return the destination filename selected by the snapshot mode."""
    if weights["type"] == "single-layer":
        return SINGLE_LAYER_SNAPSHOT_FILENAME

    return MULTI_LAYER_SNAPSHOT_FILENAME


def serialize_saved_network(weights: SavedNetwork) -> str:
    """Serialize one exact Saved Weight Snapshot JSON document."""
    return f"{json.dumps(weights, indent=2, allow_nan=False)}\n"


def create_temporary_snapshot_path(
    directory: Path,
    destination: Path,
) -> Path:
    """Create and close one unique same-directory temporary snapshot file."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(file_descriptor)

    return Path(temporary_name)


def write_snapshot_document(path: Path, document: str) -> None:
    """Write and close one complete UTF-8 snapshot document."""
    path.write_text(
        document,
        encoding="utf-8",
        newline="\n",
    )


def replace_snapshot_file(source: Path, destination: Path) -> None:
    """Atomically replace the destination with a completed temporary file."""
    os.replace(source, destination)


def remove_temporary_snapshot(path: Path) -> None:
    """Remove one temporary snapshot when it still exists."""
    path.unlink(missing_ok=True)


def save_network(
    weights: SavedNetwork,
    snapshot_directory: Path | None = None,
) -> Path:
    """Persist one completed Training Run as its mode-specific snapshot."""
    document = serialize_saved_network(weights)
    directory = snapshot_directory or get_snapshot_directory()
    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / get_snapshot_filename(weights)
    temporary_path = create_temporary_snapshot_path(
        directory,
        destination,
    )

    try:
        write_snapshot_document(
            temporary_path,
            document,
        )
        replace_snapshot_file(
            temporary_path,
            destination,
        )
    except Exception as persistence_error:
        try:
            remove_temporary_snapshot(temporary_path)
        except Exception as cleanup_error:
            raise ExceptionGroup(
                "Snapshot persistence and temporary-file cleanup failed",
                [
                    persistence_error,
                    cleanup_error,
                ],
            ) from None

        raise

    return destination


def advance_training_run(training_run: Iterator[TrainingEvent]) -> TrainingEvent:
    """Advance one bounded Training Run reporting interval."""
    return next(training_run)


async def request_is_disconnected(request: Request) -> bool:
    """Return whether the current streaming client has disconnected."""
    return await request.is_disconnected()


async def stream_neural_network(
    request: Request,
    training_run: Iterator[TrainingEvent],
) -> AsyncIterator[str]:
    """Stream one independent XOR Training Run through shared SSE framing."""
    while True:
        try:
            event = await asyncio.to_thread(
                advance_training_run,
                training_run,
            )

            if isinstance(event, EpochUpdate):
                yield format_sse(
                    "epoch",
                    cast(dict[str, object], event.to_payload()),
                )
                await presentation_sleep(PRESENTATION_DELAY_SECONDS)

                if await request_is_disconnected(request):
                    return

                continue

            await asyncio.to_thread(
                save_network,
                event.weights,
            )
            yield format_sse(
                "done",
                cast(dict[str, object], event.to_frontend_payload()),
            )
            return
        except Exception:
            logger.exception("Neural-network Training Run stream failed")
            return


@router.post("/neural-net")
async def neural_net(
    payload: NeuralNetRequest,
    request: Request,
) -> StreamingResponse:
    """Validate and start one independent Neural Network Event Stream."""
    training_run = create_training_run(
        mode=payload.mode,
        epochs=payload.epochs,
    )
    return create_sse_response(
        stream_neural_network(
            request=request,
            training_run=training_run,
        )
    )
