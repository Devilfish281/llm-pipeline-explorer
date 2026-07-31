# src/how_llms_work/ml/matrix.py
"""Strict stateless float32 matrix primitives for Transformer numerical work."""

from __future__ import annotations

import math
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

Float32Matrix: TypeAlias = NDArray[np.float32]
_Float64Matrix: TypeAlias = NDArray[np.float64]

_SOFTMAX_ROW_SUM_ATOL = 1e-6

__all__ = [
    "Float32Matrix",
    "add_in_place",
    "add_row_bias",
    "concatenate_rows",
    "elementwise_add",
    "elementwise_multiply",
    "elementwise_subtract",
    "matmul",
    "matmul_transposed_left",
    "matmul_transposed_right",
    "scalar_multiply",
    "slice_rows",
    "stable_row_softmax",
    "sum_columns",
    "transpose",
]


def _validate_matrix(
    value: object,
    *,
    name: str,
    allow_negative_infinity: bool = False,
    require_writable: bool = False,
) -> Float32Matrix:
    """Validate one public rank-two, C-contiguous float32 matrix argument."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a NumPy ndarray.")

    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32.")

    if value.ndim != 2:
        raise ValueError(f"{name} must have rank two.")

    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} must have no empty dimensions.")

    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")

    if require_writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable.")

    if allow_negative_infinity:
        if np.isnan(value).any() or np.isposinf(value).any():
            raise FloatingPointError(
                f"{name} may contain only finite values and negative infinity."
            )
    elif not np.isfinite(value).all():
        raise FloatingPointError(f"{name} must contain only finite values.")

    return cast(Float32Matrix, value)


def _validate_same_shape(
    left: Float32Matrix,
    right: Float32Matrix,
    *,
    operation_name: str,
) -> None:
    """Require exact shape equality for a non-broadcasting operation."""
    if left.shape != right.shape:
        raise ValueError(
            f"{operation_name} requires equal shapes; received {left.shape} and {right.shape}."
        )


def _validate_scalar(
    value: float | np.float32,
    *,
    name: str,
) -> float:
    """Validate a finite Python or NumPy float scalar."""
    if type(value) not in (float, np.float32):
        raise TypeError(f"{name} must be a float or numpy.float32 scalar.")

    scalar = float(value)

    if not math.isfinite(scalar):
        raise FloatingPointError(f"{name} must be finite.")

    return scalar


def _validate_row_bound(
    value: int,
    *,
    name: str,
) -> int:
    """Validate one explicit non-boolean Python integer row bound."""
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer.")

    return value


def _materialize_float32(
    values: _Float64Matrix,
    *,
    operation_name: str,
) -> Float32Matrix:
    """Create and validate one independent C-contiguous float32 result."""
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError(f"{operation_name} produced an invalid matrix shape.")

    if not np.isfinite(values).all():
        raise FloatingPointError(f"{operation_name} produced a non-finite float64 result.")

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        result = np.array(
            values,
            dtype=np.float32,
            order="C",
            copy=True,
        )

    if not result.flags.c_contiguous:
        raise FloatingPointError(f"{operation_name} did not produce a C-contiguous result.")

    if not np.isfinite(result).all():
        raise FloatingPointError(
            f"{operation_name} overflowed or became non-finite when materialized."
        )

    return result


def matmul(
    left: Float32Matrix,
    right: Float32Matrix,
) -> Float32Matrix:
    """Return left @ right with float64 calculation and float32 output."""
    left_matrix = _validate_matrix(
        left,
        name="left",
    )
    right_matrix = _validate_matrix(
        right,
        name="right",
    )

    if left_matrix.shape[1] != right_matrix.shape[0]:
        raise ValueError(
            "matmul requires left columns to equal right rows; "
            f"received {left_matrix.shape} and {right_matrix.shape}."
        )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        values = left_matrix.astype(
            np.float64,
            copy=True,
        ) @ right_matrix.astype(
            np.float64,
            copy=True,
        )

    return _materialize_float32(
        values,
        operation_name="matmul",
    )


def matmul_transposed_right(
    left: Float32Matrix,
    right: Float32Matrix,
) -> Float32Matrix:
    """Return left @ right.T without exposing a transpose view."""
    left_matrix = _validate_matrix(
        left,
        name="left",
    )
    right_matrix = _validate_matrix(
        right,
        name="right",
    )

    if left_matrix.shape[1] != right_matrix.shape[1]:
        raise ValueError(
            "matmul_transposed_right requires equal column counts; "
            f"received {left_matrix.shape} and {right_matrix.shape}."
        )

    left_values = left_matrix.astype(
        np.float64,
        copy=True,
    )
    right_values = right_matrix.astype(
        np.float64,
        copy=True,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        values = left_values @ right_values.T

    return _materialize_float32(
        values,
        operation_name="matmul_transposed_right",
    )


def matmul_transposed_left(
    left: Float32Matrix,
    right: Float32Matrix,
) -> Float32Matrix:
    """Return left.T @ right without exposing a transpose view."""
    left_matrix = _validate_matrix(
        left,
        name="left",
    )
    right_matrix = _validate_matrix(
        right,
        name="right",
    )

    if left_matrix.shape[0] != right_matrix.shape[0]:
        raise ValueError(
            "matmul_transposed_left requires equal row counts; "
            f"received {left_matrix.shape} and {right_matrix.shape}."
        )

    left_values = left_matrix.astype(
        np.float64,
        copy=True,
    )
    right_values = right_matrix.astype(
        np.float64,
        copy=True,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        values = left_values.T @ right_values

    return _materialize_float32(
        values,
        operation_name="matmul_transposed_left",
    )


def sum_columns(
    matrix: Float32Matrix,
) -> Float32Matrix:
    """Return one row containing each column sum using float64 accumulation."""
    validated = _validate_matrix(
        matrix,
        name="matrix",
    )

    values = np.sum(
        validated,
        axis=0,
        dtype=np.float64,
        keepdims=True,
    )

    return _materialize_float32(
        values,
        operation_name="sum_columns",
    )


def elementwise_add(
    left: Float32Matrix,
    right: Float32Matrix,
) -> Float32Matrix:
    """Return the exact-shape elementwise sum of two matrices."""
    left_matrix = _validate_matrix(
        left,
        name="left",
    )
    right_matrix = _validate_matrix(
        right,
        name="right",
    )

    _validate_same_shape(
        left_matrix,
        right_matrix,
        operation_name="elementwise_add",
    )

    values = left_matrix.astype(
        np.float64,
        copy=True,
    ) + right_matrix.astype(
        np.float64,
        copy=True,
    )

    return _materialize_float32(
        values,
        operation_name="elementwise_add",
    )


def elementwise_subtract(
    left: Float32Matrix,
    right: Float32Matrix,
) -> Float32Matrix:
    """Return the exact-shape elementwise difference of two matrices."""
    left_matrix = _validate_matrix(
        left,
        name="left",
    )
    right_matrix = _validate_matrix(
        right,
        name="right",
    )

    _validate_same_shape(
        left_matrix,
        right_matrix,
        operation_name="elementwise_subtract",
    )

    values = left_matrix.astype(
        np.float64,
        copy=True,
    ) - right_matrix.astype(
        np.float64,
        copy=True,
    )

    return _materialize_float32(
        values,
        operation_name="elementwise_subtract",
    )


def elementwise_multiply(
    left: Float32Matrix,
    right: Float32Matrix,
) -> Float32Matrix:
    """Return the exact-shape elementwise product of two matrices."""
    left_matrix = _validate_matrix(
        left,
        name="left",
    )
    right_matrix = _validate_matrix(
        right,
        name="right",
    )

    _validate_same_shape(
        left_matrix,
        right_matrix,
        operation_name="elementwise_multiply",
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        values = left_matrix.astype(
            np.float64,
            copy=True,
        ) * right_matrix.astype(
            np.float64,
            copy=True,
        )

    return _materialize_float32(
        values,
        operation_name="elementwise_multiply",
    )


def add_row_bias(
    matrix: Float32Matrix,
    bias: Float32Matrix,
) -> Float32Matrix:
    """Add one exact (1, columns) bias row to every matrix row."""
    validated = _validate_matrix(
        matrix,
        name="matrix",
    )
    validated_bias = _validate_matrix(
        bias,
        name="bias",
    )

    expected_shape = (1, validated.shape[1])

    if validated_bias.shape != expected_shape:
        raise ValueError(
            f"bias must have shape exactly {expected_shape}; received {validated_bias.shape}."
        )

    values = validated.astype(
        np.float64,
        copy=True,
    ) + validated_bias.astype(
        np.float64,
        copy=True,
    )

    return _materialize_float32(
        values,
        operation_name="add_row_bias",
    )


def scalar_multiply(
    matrix: Float32Matrix,
    scalar: float | np.float32,
) -> Float32Matrix:
    """Multiply every matrix entry by one finite floating-point scalar."""
    validated = _validate_matrix(
        matrix,
        name="matrix",
    )
    validated_scalar = _validate_scalar(
        scalar,
        name="scalar",
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        values = (
            validated.astype(
                np.float64,
                copy=True,
            )
            * validated_scalar
        )

    return _materialize_float32(
        values,
        operation_name="scalar_multiply",
    )


def transpose(
    matrix: Float32Matrix,
) -> Float32Matrix:
    """Return a copied C-contiguous matrix transpose."""
    validated = _validate_matrix(
        matrix,
        name="matrix",
    )

    values = validated.astype(
        np.float64,
        copy=True,
    ).T.copy(order="C")

    return _materialize_float32(
        values,
        operation_name="transpose",
    )


def concatenate_rows(
    top: Float32Matrix,
    bottom: Float32Matrix,
) -> Float32Matrix:
    """Concatenate two matrices along rows after exact column validation."""
    top_matrix = _validate_matrix(
        top,
        name="top",
    )
    bottom_matrix = _validate_matrix(
        bottom,
        name="bottom",
    )

    if top_matrix.shape[1] != bottom_matrix.shape[1]:
        raise ValueError(
            "concatenate_rows requires equal column counts; "
            f"received {top_matrix.shape} and {bottom_matrix.shape}."
        )

    values = np.concatenate(
        (
            top_matrix.astype(
                np.float64,
                copy=True,
            ),
            bottom_matrix.astype(
                np.float64,
                copy=True,
            ),
        ),
        axis=0,
    )

    return _materialize_float32(
        values,
        operation_name="concatenate_rows",
    )


def slice_rows(
    matrix: Float32Matrix,
    start: int,
    stop: int,
) -> Float32Matrix:
    """Return an independent half-open row slice with explicit bounds."""
    validated = _validate_matrix(
        matrix,
        name="matrix",
    )
    validated_start = _validate_row_bound(
        start,
        name="start",
    )
    validated_stop = _validate_row_bound(
        stop,
        name="stop",
    )

    if validated_start < 0 or validated_stop < 0:
        raise ValueError("slice_rows does not accept negative bounds.")

    if validated_start >= validated_stop:
        raise ValueError("slice_rows requires start to be less than stop.")

    if validated_stop > validated.shape[0]:
        raise ValueError(
            f"slice_rows stop {validated_stop} exceeds row count {validated.shape[0]}."
        )

    values = validated[validated_start:validated_stop].astype(
        np.float64,
        copy=True,
        order="C",
    )

    return _materialize_float32(
        values,
        operation_name="slice_rows",
    )


def add_in_place(
    destination: Float32Matrix,
    source: Float32Matrix,
) -> None:
    """Transactionally add a separate non-overlapping source."""
    destination_matrix = _validate_matrix(
        destination,
        name="destination",
        require_writable=True,
    )
    source_matrix = _validate_matrix(
        source,
        name="source",
    )

    _validate_same_shape(
        destination_matrix,
        source_matrix,
        operation_name="add_in_place",
    )

    if np.shares_memory(
        destination_matrix,
        source_matrix,
    ):
        raise ValueError("destination and source must not share memory.")

    destination_values = destination_matrix.astype(
        np.float64,
        copy=True,
    )
    source_values = source_matrix.astype(
        np.float64,
        copy=True,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        candidate_values = destination_values + source_values

    candidate = _materialize_float32(
        candidate_values,
        operation_name="add_in_place",
    )

    destination_matrix[...] = candidate


def stable_row_softmax(
    scores: Float32Matrix,
) -> Float32Matrix:
    """Return stable row probabilities while preserving exact masks."""
    validated = _validate_matrix(
        scores,
        name="scores",
        allow_negative_infinity=True,
    )

    selectable = np.isfinite(validated)

    if not np.all(
        np.any(
            selectable,
            axis=1,
        )
    ):
        raise ValueError("every softmax row must contain at least one finite score.")

    score_values = validated.astype(
        np.float64,
        copy=True,
    )

    finite_scores = np.where(
        selectable,
        score_values,
        -np.inf,
    )

    row_maxima = np.max(
        finite_scores,
        axis=1,
        keepdims=True,
    )

    shifted = np.zeros_like(
        score_values,
        dtype=np.float64,
    )

    np.subtract(
        score_values,
        row_maxima,
        out=shifted,
        where=selectable,
    )

    exponentials = np.zeros_like(
        score_values,
        dtype=np.float64,
    )

    with np.errstate(
        over="ignore",
        invalid="ignore",
        under="ignore",
    ):
        np.exp(
            shifted,
            out=exponentials,
            where=selectable,
        )

    denominators = np.sum(
        exponentials,
        axis=1,
        dtype=np.float64,
        keepdims=True,
    )

    if not np.isfinite(denominators).all() or np.any(denominators <= 0.0):
        raise FloatingPointError("softmax produced an invalid denominator.")

    probabilities = np.zeros_like(
        score_values,
        dtype=np.float64,
    )

    np.divide(
        exponentials,
        denominators,
        out=probabilities,
        where=selectable,
    )

    probabilities[~selectable] = 0.0

    result = _materialize_float32(
        probabilities,
        operation_name="stable_row_softmax",
    )

    if np.any(result < np.float32(0.0)):
        raise FloatingPointError("softmax produced a negative probability.")

    row_sums = np.sum(
        result,
        axis=1,
        dtype=np.float64,
    )

    if not np.allclose(
        row_sums,
        1.0,
        rtol=0.0,
        atol=_SOFTMAX_ROW_SUM_ATOL,
    ):
        raise FloatingPointError("softmax rows do not sum to one.")

    return result
