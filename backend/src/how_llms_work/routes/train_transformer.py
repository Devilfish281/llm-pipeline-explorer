# backend/src/how_llms_work/routes/train_transformer.py

"""FastAPI streaming and deterministic persistence for Transformer Training Runs."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import tempfile
from asyncio import sleep as presentation_sleep
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event, Lock
from typing import Final, NoReturn, TypeVar, cast

import numpy as np
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
    EmptySavedTransformerPromptError,
    InitializedTransformerParameters,
    SavedTransformerConfig,
    SavedTransformerMerge,
    SavedTransformerModel,
    SavedTransformerPromptTooLongError,
    SavedTransformerWeights,
    TransformerParameterLayout,
    TransformerParameterLayoutRecord,
    TransformerPreprocessingSnapshot,
    UnsupportedSavedTransformerPromptError,
    build_saved_transformer_model,
    build_transformer_parameter_layout,
    build_transformer_parameter_views,
    create_transformer_training_run,
    evaluate_transformer_final_loss,
    generate_saved_transformer_text,
    generate_transformer_text,
    get_transformer_preprocessing,
    initialize_transformer_parameters,
    prepare_saved_transformer_prompt,
    transformer_parameter_count,
)
from how_llms_work.ml.transformer_worker import (
    RequestScopedWorkerGroup,
    RequestScopedWorkerGroupCleanupReport,
    create_request_scoped_worker_group,
)
from how_llms_work.schemas import LoadTransformerRequest, TrainTransformerRequest
from how_llms_work.sse import create_sse_response, format_sse

logger = logging.getLogger(__name__)
router = APIRouter()
_TRANSFORMER_RUN_SLOT = Lock()


_TRANSFORMER_MIN_EPOCH_COUNT: Final = 50
_TRANSFORMER_MAX_EPOCH_COUNT: Final = 2_000
PRESENTATION_DELAY_SECONDS: Final = 0.02
_TRANSFORMER_HELPER_TIMEOUT_SECONDS: Final = 300.0
_TRANSFORMER_HELPER_POLL_SECONDS: Final = 0.1

_SAVED_TRANSFORMER_MODEL_LOAD_FAILURE: Final = "The saved Transformer model could not be loaded."
_NO_VALID_SAVED_TRANSFORMER_MODEL_FAILURE: Final = "No valid saved Transformer model was found."
_EMPTY_SAVED_TRANSFORMER_PROMPT_FAILURE: Final = "The prompt must not be empty."
_UNSUPPORTED_SAVED_TRANSFORMER_PROMPT_FAILURE: Final = (
    "The prompt contains text that this saved Transformer model cannot tokenize."
)
_SAVED_TRANSFORMER_PROMPT_TOO_LONG_FAILURE: Final = (
    "The prompt must contain no more than 16 tokens."
)
_SAVED_TRANSFORMER_GENERATION_FAILURE: Final = (
    "The saved Transformer model could not generate text."
)
_SAVED_TRANSFORMER_START_FAILURE: Final = "Saved Transformer generation could not start."
_TRANSFORMER_REQUEST_OVERLAP_DETAIL: Final = "Another Transformer request is already running."

_TRANSFORMER_MODEL_FILENAME_PATTERN: Final = re.compile(
    r"transformer-weights-e([0-9]+)-l([0-9]+)-d32-h2-ff128-ctx32\.json"
)


_HelperResultT = TypeVar("_HelperResultT")


class SavedTransformerModelLoadError(RuntimeError):
    """Stable public failure for unsafe or unusable Saved Transformer Models."""


@dataclass(frozen=True, slots=True)
class _SelectedTransformerModelFile:
    """One exact ordinary model file selected from the genuine model directory."""

    model_filename: str
    path: Path


@dataclass(frozen=True, slots=True)
class _LatestTransformerModelCandidate:
    """One safely classified direct model candidate with deterministic ordering metadata."""

    model_filename: str
    path: Path
    modification_time_ns: int


@dataclass(frozen=True, slots=True)
class _ValidatedTransformerModelDocument:
    """One strictly validated current-format Saved Transformer Model document."""

    model_filename: str
    config: SavedTransformerConfig
    vocabulary: list[str]
    merges: list[SavedTransformerMerge]
    weights: SavedTransformerWeights
    layout: TransformerParameterLayout


@dataclass(frozen=True, slots=True, eq=False)
class LoadedTransformerModelSnapshot:
    """One complete request-owned Transformer inference snapshot."""

    model_filename: str
    config: SavedTransformerConfig
    vocabulary: list[str]
    merges: list[SavedTransformerMerge]
    parameters: InitializedTransformerParameters


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


def _parse_transformer_model_filename(model_filename: object) -> tuple[int, int]:
    """Validate one untrusted exact Phase 5 Transformer model filename."""
    if type(model_filename) is not str:
        raise TypeError("Saved Transformer Model filename must be a string.")

    if not model_filename or model_filename != model_filename.strip():
        raise ValueError("Saved Transformer Model filename must be an exact plain name.")

    windows_path = PureWindowsPath(model_filename)
    posix_path = PurePosixPath(model_filename)

    if (
        "/" in model_filename
        or "\\" in model_filename
        or windows_path.drive
        or windows_path.root
        or posix_path.root
        or ".." in windows_path.parts
        or ".." in posix_path.parts
    ):
        raise ValueError("Saved Transformer Model filename must be an exact plain name.")

    match = _TRANSFORMER_MODEL_FILENAME_PATTERN.fullmatch(model_filename)

    if match is None:
        raise ValueError("Saved Transformer Model filename is not canonical.")

    epochs = _validate_transformer_epochs(int(match.group(1)))
    num_layers = _validate_bounded_integer(
        int(match.group(2)),
        name="numLayers",
        minimum=TRANSFORMER_MIN_LAYER_COUNT,
        maximum=TRANSFORMER_MAX_LAYER_COUNT,
    )

    canonical_filename = _build_transformer_model_filename(
        epochs=epochs,
        num_layers=num_layers,
    )

    if model_filename != canonical_filename:
        raise ValueError("Saved Transformer Model filename is not canonical.")

    return epochs, num_layers


def _path_is_transformer_model_indirection(path: Path) -> bool:
    """Return whether one model path is a symbolic link or Windows junction."""
    return path.is_symlink() or path.is_junction()


def get_transformer_model_directory() -> Path:
    """Return the backend project's Saved Transformer Model directory."""
    backend_root = Path(__file__).resolve().parents[3]
    return backend_root / ".data"


def _resolve_transformer_model_directory() -> tuple[Path, Path]:
    """Return the genuine model directory and its resolved containment root."""
    model_directory = get_transformer_model_directory()

    if _path_is_transformer_model_indirection(model_directory):
        raise ValueError("Saved Transformer Model directory cannot be path-indirected.")

    if not model_directory.is_dir():
        raise ValueError("Saved Transformer Model directory must be an ordinary directory.")

    resolved_directory = model_directory.resolve(strict=True)

    if not resolved_directory.is_dir():
        raise ValueError("Saved Transformer Model directory must resolve to a directory.")

    return model_directory, resolved_directory


def _select_named_transformer_model_file(
    model_filename: object,
) -> _SelectedTransformerModelFile:
    """Select one exact ordinary direct-entry model file without opening it."""
    _parse_transformer_model_filename(model_filename)
    requested_filename = cast(str, model_filename)
    model_directory, resolved_directory = _resolve_transformer_model_directory()

    selected_path: Path | None = None

    for entry in model_directory.iterdir():
        if entry.name == requested_filename:
            selected_path = entry
            break

    if selected_path is None:
        raise FileNotFoundError("Exact Saved Transformer Model entry was not found.")

    if _path_is_transformer_model_indirection(selected_path):
        raise ValueError("Saved Transformer Model candidate cannot be path-indirected.")

    if not selected_path.is_file():
        raise ValueError("Saved Transformer Model candidate must be an ordinary file.")

    resolved_candidate = selected_path.resolve(strict=True)

    if resolved_candidate.parent != resolved_directory:
        raise ValueError("Saved Transformer Model candidate escaped its model directory.")

    return _SelectedTransformerModelFile(
        model_filename=requested_filename,
        path=selected_path,
    )


def _select_latest_transformer_model_candidates() -> tuple[_LatestTransformerModelCandidate, ...]:
    """Return safely classified direct candidates in deterministic newest-first order."""
    model_directory, resolved_directory = _resolve_transformer_model_directory()
    candidates: list[_LatestTransformerModelCandidate] = []

    for entry in model_directory.iterdir():
        try:
            _parse_transformer_model_filename(entry.name)

            if _path_is_transformer_model_indirection(entry):
                continue

            if not entry.is_file():
                continue

            resolved_candidate = entry.resolve(strict=True)

            if resolved_candidate.parent != resolved_directory:
                continue

            modification_time_ns = entry.stat().st_mtime_ns
        except (OSError, RuntimeError, TypeError, ValueError):
            continue

        candidates.append(
            _LatestTransformerModelCandidate(
                model_filename=entry.name,
                path=entry,
                modification_time_ns=modification_time_ns,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.modification_time_ns,
            candidate.model_filename,
        ),
        reverse=True,
    )

    return tuple(candidates)


def _build_unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one insertion-ordered JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError("Saved Transformer Model JSON objects cannot contain duplicate keys.")

        result[key] = value

    return result


def _reject_nonfinite_json_constant(value: str) -> NoReturn:
    """Reject non-standard NaN and infinity constants during JSON parsing."""
    raise ValueError(f"Saved Transformer Model JSON constant {value!r} is unsupported.")


def _parse_saved_transformer_model_document(
    document_bytes: bytes,
) -> object:
    """Decode and parse one captured UTF-8 model document in memory."""
    document_text = document_bytes.decode("utf-8")

    parsed_document: object = json.loads(
        document_text,
        object_pairs_hook=_build_unique_json_object,
        parse_constant=_reject_nonfinite_json_constant,
    )

    return parsed_document


def _validate_loaded_transformer_model_document(
    parsed_document: object,
    *,
    model_filename: str,
) -> _ValidatedTransformerModelDocument:
    """Strictly validate one parsed current Phase 5 model document."""
    model_mapping = _require_exact_dictionary(
        parsed_document,
        name="Loaded Saved Transformer Model",
        expected_keys=_MODEL_KEYS,
    )
    model = cast(SavedTransformerModel, model_mapping)
    num_layers = _validate_saved_transformer_model_structure(model)

    _, filename_num_layers = _parse_transformer_model_filename(model_filename)

    if filename_num_layers != num_layers:
        raise ValueError(
            "Saved Transformer Model filename layer count must equal config numLayers."
        )

    config = cast(SavedTransformerConfig, model_mapping["config"])
    layout = build_transformer_parameter_layout(num_layers)

    if config["vocabSize"] != layout.vocabulary_size:
        raise ValueError(
            "Saved Transformer Model vocabSize must equal the canonical Vocabulary size."
        )

    preprocessing = get_transformer_preprocessing()
    vocabulary = cast(list[str], model_mapping["vocab"])

    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Saved Transformer Model Vocabulary entries must be unique.")

    if tuple(vocabulary) != preprocessing.vocabulary:
        raise ValueError(
            "Saved Transformer Model Vocabulary must match the current Phase 5 ordering."
        )

    merges = cast(list[SavedTransformerMerge], model_mapping["merges"])

    if len(merges) != len(preprocessing.merges):
        raise ValueError(
            "Saved Transformer Model Merge Table must match the current Phase 5 length."
        )

    for merge_index, (saved_merge, expected_merge) in enumerate(
        zip(
            merges,
            preprocessing.merges,
            strict=True,
        )
    ):
        pair = saved_merge["pair"]
        merged = saved_merge["merged"]

        if merged != pair[0] + pair[1]:
            raise ValueError(
                f"Saved Transformer Model merge {merge_index} must coherently concatenate its pair."
            )

        if tuple(pair) != expected_merge.pair or merged != expected_merge.merged:
            raise ValueError(
                "Saved Transformer Model Merge Table must match the current Phase 5 ordering."
            )

    weights = cast(SavedTransformerWeights, model_mapping["weights"])

    return _ValidatedTransformerModelDocument(
        model_filename=model_filename,
        config=config,
        vocabulary=vocabulary,
        merges=merges,
        weights=weights,
        layout=layout,
    )


def _saved_parameter_values_for_layout_record(
    weights: SavedTransformerWeights,
    record: TransformerParameterLayoutRecord,
) -> list[object]:
    """Return one already-validated parameter list by canonical layout record."""
    weight_mapping = cast(dict[str, object], weights)

    if record.block_index is None:
        return cast(list[object], weight_mapping[record.key])

    blocks = cast(
        list[dict[str, object]],
        weight_mapping["blocks"],
    )

    return cast(
        list[object],
        blocks[record.block_index][record.key],
    )


def _materialize_float32_coordinate(
    value: object,
    *,
    name: str,
) -> np.float32:
    """Convert one strict ordinary JSON number into one finite float32 value."""
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an ordinary JSON number.")

    try:
        numeric_value = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{name} is outside the supported numerical range.") from error

    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")

    try:
        with np.errstate(over="raise", invalid="raise"):
            materialized = np.float32(numeric_value)
    except FloatingPointError as error:
        raise ValueError(f"{name} cannot be represented as float32.") from error

    if not np.isfinite(materialized):
        raise ValueError(f"{name} cannot be represented as finite float32.")

    return materialized


def _materialize_transformer_parameters(
    document: _ValidatedTransformerModelDocument,
) -> InitializedTransformerParameters:
    """Materialize exact saved coordinates into one canonical owned storage block."""
    layout = document.layout
    storage = np.empty(
        layout.total_float_count,
        dtype=np.float32,
        order="C",
    )

    for record in layout.records:
        saved_values = _saved_parameter_values_for_layout_record(
            document.weights,
            record,
        )

        if len(saved_values) != record.length:
            raise ValueError(
                "Saved Transformer Model parameter length "
                "does not match its canonical layout record."
            )

        target = storage[record.float_offset : record.float_stop]

        for coordinate_index, saved_value in enumerate(saved_values):
            target[coordinate_index] = _materialize_float32_coordinate(
                saved_value,
                name=(f"Saved Transformer Model parameter {record.key}[{coordinate_index}]"),
            )

    if (
        storage.dtype != np.dtype(np.float32)
        or storage.shape != (layout.total_float_count,)
        or not storage.flags.c_contiguous
        or not storage.flags.writeable
        or not storage.flags.owndata
        or not bool(np.all(np.isfinite(storage)))
    ):
        raise RuntimeError("Saved Transformer Model parameters were not materialized canonically.")

    views = build_transformer_parameter_views(
        storage,
        layout,
    )

    return InitializedTransformerParameters(
        layout=layout,
        storage=storage,
        views=views,
    )


def _build_loaded_transformer_model_snapshot(
    document: _ValidatedTransformerModelDocument,
) -> LoadedTransformerModelSnapshot:
    """Copy validated metadata and attach freshly materialized parameters."""
    config = SavedTransformerConfig(
        vocabSize=document.config["vocabSize"],
        contextLen=document.config["contextLen"],
        embDim=document.config["embDim"],
        numHeads=document.config["numHeads"],
        ffDim=document.config["ffDim"],
        numLayers=document.config["numLayers"],
    )

    vocabulary = list(document.vocabulary)

    merges = [
        SavedTransformerMerge(
            pair=list(merge["pair"]),
            merged=merge["merged"],
        )
        for merge in document.merges
    ]

    parameters = _materialize_transformer_parameters(document)

    return LoadedTransformerModelSnapshot(
        model_filename=document.model_filename,
        config=config,
        vocabulary=vocabulary,
        merges=merges,
        parameters=parameters,
    )


def _load_selected_transformer_model(
    selected_model: _SelectedTransformerModelFile,
) -> LoadedTransformerModelSnapshot:
    """Read one selected file once and build its request-owned snapshot."""
    document_bytes = selected_model.path.read_bytes()
    parsed_document = _parse_saved_transformer_model_document(document_bytes)
    validated_document = _validate_loaded_transformer_model_document(
        parsed_document,
        model_filename=selected_model.model_filename,
    )

    return _build_loaded_transformer_model_snapshot(validated_document)


def load_named_transformer_model(
    model_filename: str,
) -> LoadedTransformerModelSnapshot:
    """Load one exact named model into a complete request-owned snapshot."""
    try:
        selected_model = _select_named_transformer_model_file(model_filename)
        return _load_selected_transformer_model(selected_model)
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError):
        raise SavedTransformerModelLoadError(_SAVED_TRANSFORMER_MODEL_LOAD_FAILURE) from None


def load_latest_transformer_model() -> LoadedTransformerModelSnapshot:
    """Load the newest strictly valid model into a request-owned snapshot."""
    try:
        candidates = _select_latest_transformer_model_candidates()
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError):
        raise SavedTransformerModelLoadError(_SAVED_TRANSFORMER_MODEL_LOAD_FAILURE) from None

    for candidate in candidates:
        selected_model = _SelectedTransformerModelFile(
            model_filename=candidate.model_filename,
            path=candidate.path,
        )

        try:
            return _load_selected_transformer_model(selected_model)
        except (MemoryError, OSError, RuntimeError, TypeError, ValueError):
            continue

    raise SavedTransformerModelLoadError(_SAVED_TRANSFORMER_MODEL_LOAD_FAILURE) from None


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
                    f"Decoder-Only Transformer ({model_num_layers} layers, 32d, 2h, 128ff)"
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


async def stream_saved_transformer_generation(
    *,
    request: Request,
    model_filename: str | None,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> AsyncIterator[str]:
    """Stream one Saved Transformer Generation Run."""
    cancellation_event = Event()

    load_operation: Callable[[], LoadedTransformerModelSnapshot]

    if model_filename is None:
        load_operation = load_latest_transformer_model
        load_failure_message = _NO_VALID_SAVED_TRANSFORMER_MODEL_FAILURE
    else:
        load_operation = partial(
            load_named_transformer_model,
            model_filename,
        )
        load_failure_message = _SAVED_TRANSFORMER_MODEL_LOAD_FAILURE

    try:
        try:
            loaded_snapshot = await _run_unbounded_transformer_helper(
                load_operation,
                request=request,
                cancellation_event=cancellation_event,
            )
        except SavedTransformerModelLoadError:
            yield format_sse(
                "error",
                {"error": load_failure_message},
            )
            return

        try:
            prepared_prompt = await _run_unbounded_transformer_helper(
                partial(
                    prepare_saved_transformer_prompt,
                    prompt,
                    loaded_snapshot.vocabulary,
                    loaded_snapshot.merges,
                ),
                request=request,
                cancellation_event=cancellation_event,
            )
        except EmptySavedTransformerPromptError:
            yield format_sse(
                "error",
                {"error": _EMPTY_SAVED_TRANSFORMER_PROMPT_FAILURE},
            )
            return
        except UnsupportedSavedTransformerPromptError:
            yield format_sse(
                "error",
                {"error": _UNSUPPORTED_SAVED_TRANSFORMER_PROMPT_FAILURE},
            )
            return
        except SavedTransformerPromptTooLongError:
            yield format_sse(
                "error",
                {"error": _SAVED_TRANSFORMER_PROMPT_TOO_LONG_FAILURE},
            )
            return

        yield format_sse(
            "loaded",
            {
                "file": loaded_snapshot.model_filename,
                "prompt": prepared_prompt.text,
            },
        )

        try:
            complete_text = await _run_unbounded_transformer_helper(
                partial(
                    generate_saved_transformer_text,
                    loaded_snapshot.parameters,
                    loaded_snapshot.vocabulary,
                    prepared_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    cancellation_event=cancellation_event,
                ),
                request=request,
                cancellation_event=cancellation_event,
            )
        except _TransformerClientDisconnected:
            raise
        except Exception:
            cancellation_event.set()
            logger.exception("Saved Transformer text generation failed")

            yield format_sse(
                "error",
                {"error": _SAVED_TRANSFORMER_GENERATION_FAILURE},
            )
            return

        yield format_sse(
            "result",
            {"text": complete_text},
        )
        yield format_sse(
            "done",
            {},
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
        logger.exception("Saved Transformer Generation Run stream failed")

        yield format_sse(
            "error",
            {"error": _SAVED_TRANSFORMER_GENERATION_FAILURE},
        )
        return
    finally:
        cancellation_event.set()
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
            detail=_TRANSFORMER_REQUEST_OVERLAP_DETAIL,
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


@router.post("/load-transformer")
async def load_transformer(
    payload: LoadTransformerRequest,
    request: Request,
) -> StreamingResponse:
    """Validate, reserve, and start one Saved Transformer Event Stream."""
    if not _TRANSFORMER_RUN_SLOT.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=_TRANSFORMER_REQUEST_OVERLAP_DETAIL,
        )

    try:
        response = create_sse_response(
            stream_saved_transformer_generation(
                request=request,
                model_filename=payload.model_file,
                prompt=payload.prompt,
                temperature=payload.temperature,
                top_p=payload.top_p,
                max_tokens=payload.max_tokens,
            )
        )
    except Exception:
        logger.exception("Saved Transformer Generation Run preparation failed")
        _TRANSFORMER_RUN_SLOT.release()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_SAVED_TRANSFORMER_START_FAILURE,
        ) from None

    return response
