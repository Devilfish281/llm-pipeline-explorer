# backend/src/how_llms_work/routes/train_embed.py
"""FastAPI streaming and safe persistence for Word2Vec Embedding Training Runs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from asyncio import sleep as presentation_sleep
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from time import sleep as blocking_sleep
from typing import Final, cast

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from how_llms_work.ml.word2vec import (
    EmbeddingEpochUpdate,
    EmbeddingTrainingEvent,
    SavedEmbeddingModel,
    Word2VecPreprocessing,
    build_embedding_result,
    build_saved_embedding_model,
    create_embedding_training_run,
    get_word2vec_preprocessing,
)
from how_llms_work.schemas import TrainEmbedRequest
from how_llms_work.sse import create_sse_response, format_sse

logger = logging.getLogger(__name__)
router = APIRouter()

EMBEDDING_MODEL_FILENAME: Final = "embedding-weights.json"
PRESENTATION_DELAY_SECONDS: Final = 0.02
_WINDOWS_REPLACE_MAX_ATTEMPTS: Final = 20
_WINDOWS_REPLACE_RETRY_DELAY_SECONDS: Final = 0.01


def get_embedding_model_directory() -> Path:
    """Return the backend project's Saved Embedding Model directory."""
    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / ".data"


def serialize_saved_embedding_model(model: SavedEmbeddingModel) -> str:
    """Serialize one exact Saved Embedding Model JSON document."""
    return f"{json.dumps(model, indent=2, allow_nan=False)}\n"


def create_temporary_embedding_model_path(
    directory: Path,
    destination: Path,
) -> Path:
    """Create and close one unique same-directory temporary model file."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(file_descriptor)

    return Path(temporary_name)


def write_embedding_model_document(path: Path, document: str) -> None:
    """Write and close one complete UTF-8 Saved Embedding Model document."""
    path.write_text(
        document,
        encoding="utf-8",
        newline="\n",
    )


def replace_embedding_model_file(
    source: Path,
    destination: Path,
) -> None:
    """Atomically replace the destination, retrying transient Windows access races."""
    for attempt in range(_WINDOWS_REPLACE_MAX_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == _WINDOWS_REPLACE_MAX_ATTEMPTS - 1:
                raise

            blocking_sleep(_WINDOWS_REPLACE_RETRY_DELAY_SECONDS)


def remove_temporary_embedding_model(path: Path) -> None:
    """Remove one owned temporary model file when it still exists."""
    path.unlink(missing_ok=True)


def save_embedding_model(
    model: SavedEmbeddingModel,
    model_directory: Path | None = None,
) -> Path:
    """Persist one complete Saved Embedding Model through atomic replacement."""
    document = serialize_saved_embedding_model(model)
    directory = model_directory or get_embedding_model_directory()
    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / EMBEDDING_MODEL_FILENAME
    temporary_path = create_temporary_embedding_model_path(
        directory,
        destination,
    )

    try:
        write_embedding_model_document(
            temporary_path,
            document,
        )
        replace_embedding_model_file(
            temporary_path,
            destination,
        )
    except Exception as persistence_error:
        try:
            remove_temporary_embedding_model(temporary_path)
        except Exception as cleanup_error:
            raise ExceptionGroup(
                "Saved Embedding Model persistence and temporary-file cleanup failed",
                [
                    persistence_error,
                    cleanup_error,
                ],
            ) from None

        raise

    return destination


def advance_embedding_training_run(
    training_run: Iterator[EmbeddingTrainingEvent],
) -> EmbeddingTrainingEvent:
    """Advance one bounded Embedding Training Run reporting interval."""
    return next(training_run)


async def request_is_disconnected(request: Request) -> bool:
    """Return whether the current streaming client has disconnected."""
    return await request.is_disconnected()


async def stream_embedding_training(
    *,
    request: Request,
    training_run: Iterator[EmbeddingTrainingEvent],
    preprocessing: Word2VecPreprocessing,
    query_words: tuple[str, ...],
    dimensions: int,
    window_size: int,
) -> AsyncIterator[str]:
    """Stream one independent Embedding Training Run through shared SSE framing."""
    yield format_sse(
        "init",
        {
            "vocabSize": len(preprocessing.vocabulary),
            "sentenceCount": len(preprocessing.corpus),
            "embeddingDim": dimensions,
            "windowSize": window_size,
            "totalPairs": len(preprocessing.training_pairs[window_size]),
        },
    )

    try:
        while True:
            if await request_is_disconnected(request):
                return

            event = await asyncio.to_thread(
                advance_embedding_training_run,
                training_run,
            )

            if isinstance(event, EmbeddingEpochUpdate):
                yield format_sse(
                    "epoch",
                    cast(dict[str, object], event.to_payload()),
                )
                await presentation_sleep(PRESENTATION_DELAY_SECONDS)
                continue

            if await request_is_disconnected(request):
                return

            embedding_result = await asyncio.to_thread(
                build_embedding_result,
                event,
                preprocessing,
                query_words,
            )

            if await request_is_disconnected(request):
                return

            saved_model = await asyncio.to_thread(
                build_saved_embedding_model,
                event,
                preprocessing,
            )

            if await request_is_disconnected(request):
                return

            await asyncio.to_thread(
                save_embedding_model,
                saved_model,
            )

            if await request_is_disconnected(request):
                return

            yield format_sse(
                "done",
                cast(dict[str, object], embedding_result),
            )
            return
    except Exception:
        logger.exception("Embedding Training Run stream failed")
        return


@router.post("/train-embed")
async def train_embed(
    payload: TrainEmbedRequest,
    request: Request,
) -> StreamingResponse:
    """Validate and start one independent Embedding Training Event Stream."""
    query_words = tuple(payload.words)
    epochs = payload.epochs
    dimensions = payload.dimensions
    window_size = payload.window_size
    negative_samples = payload.negative_samples

    preprocessing = get_word2vec_preprocessing()

    training_run = create_embedding_training_run(
        dimensions=dimensions,
        window_size=window_size,
        epochs=epochs,
        negative_samples=negative_samples,
    )

    return create_sse_response(
        stream_embedding_training(
            request=request,
            training_run=training_run,
            preprocessing=preprocessing,
            query_words=query_words,
            dimensions=dimensions,
            window_size=window_size,
        )
    )
