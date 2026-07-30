# test/matrix.py
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from how_llms_work.ml import matrix as matrix_module
from how_llms_work.ml.matrix import (
    Float32Matrix,
    add_in_place,
    add_row_bias,
    concatenate_rows,
    elementwise_add,
    elementwise_multiply,
    elementwise_subtract,
    matmul,
    matmul_transposed_left,
    matmul_transposed_right,
    scalar_multiply,
    slice_rows,
    stable_row_softmax,
    sum_columns,
    transpose,
)
from numpy.testing import assert_allclose, assert_array_equal

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "matrix_reference.json"

EXPECTED_PUBLIC_SYMBOLS = {
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
}


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _decode_number(value: object) -> float:
    if value == "-Infinity":
        return -np.inf
    if value == "Infinity":
        return np.inf
    if value == "NaN":
        return np.nan

    return float(value)


def _matrix(
    values: list[list[object]],
) -> Float32Matrix:
    return np.array(
        [[_decode_number(value) for value in row] for row in values],
        dtype=np.float32,
        order="C",
    )


def _assert_pure_result(
    result: Float32Matrix,
    expected: Float32Matrix,
    inputs: tuple[Float32Matrix, ...],
    before: tuple[Float32Matrix, ...],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    assert result.shape == expected.shape
    assert result.dtype == np.dtype(np.float32)
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()

    assert_allclose(
        result,
        expected,
        rtol=rtol,
        atol=atol,
    )

    for source, original in zip(
        inputs,
        before,
        strict=True,
    ):
        assert_array_equal(source, original)
        assert not np.shares_memory(result, source)


def test_public_contract_exposes_only_approved_symbols() -> None:
    assert set(matrix_module.__all__) == EXPECTED_PUBLIC_SYMBOLS
    assert "rand" not in matrix_module.__all__
    assert "reset_rand" not in matrix_module.__all__
    assert "Tensor" not in matrix_module.__all__
    assert "Matrix" not in matrix_module.__all__


def test_fixture_records_independent_provenance(
    reference: dict[str, Any],
) -> None:
    provenance = reference["provenance"]

    assert provenance["production_module_imported"] is False
    assert "does not import how_llms_work.ml.matrix" in provenance["source"]


@pytest.mark.parametrize(
    ("operation_name", "operation"),
    [
        (
            "matmul",
            lambda ref: matmul(
                _matrix(ref["matmul"]["left"]),
                _matrix(ref["matmul"]["right"]),
            ),
        ),
        (
            "matmul_transposed_right",
            lambda ref: matmul_transposed_right(
                _matrix(ref["matmul_transposed_right"]["left"]),
                _matrix(ref["matmul_transposed_right"]["right"]),
            ),
        ),
        (
            "matmul_transposed_left",
            lambda ref: matmul_transposed_left(
                _matrix(ref["matmul_transposed_left"]["left"]),
                _matrix(ref["matmul_transposed_left"]["right"]),
            ),
        ),
        (
            "sum_columns",
            lambda ref: sum_columns(_matrix(ref["sum_columns"]["matrix"])),
        ),
        (
            "elementwise_add",
            lambda ref: elementwise_add(
                _matrix(ref["elementwise"]["left"]),
                _matrix(ref["elementwise"]["right"]),
            ),
        ),
        (
            "elementwise_subtract",
            lambda ref: elementwise_subtract(
                _matrix(ref["elementwise"]["left"]),
                _matrix(ref["elementwise"]["right"]),
            ),
        ),
        (
            "elementwise_multiply",
            lambda ref: elementwise_multiply(
                _matrix(ref["elementwise"]["left"]),
                _matrix(ref["elementwise"]["right"]),
            ),
        ),
        (
            "add_row_bias",
            lambda ref: add_row_bias(
                _matrix(ref["row_bias"]["matrix"]),
                _matrix(ref["row_bias"]["bias"]),
            ),
        ),
        (
            "scalar_multiply",
            lambda ref: scalar_multiply(
                _matrix(ref["scalar_multiply"]["matrix"]),
                ref["scalar_multiply"]["scalar"],
            ),
        ),
        (
            "transpose",
            lambda ref: transpose(_matrix(ref["transpose"]["matrix"])),
        ),
        (
            "concatenate_rows",
            lambda ref: concatenate_rows(
                _matrix(ref["concatenate_rows"]["top"]),
                _matrix(ref["concatenate_rows"]["bottom"]),
            ),
        ),
        (
            "slice_rows",
            lambda ref: slice_rows(
                _matrix(ref["slice_rows"]["matrix"]),
                ref["slice_rows"]["start"],
                ref["slice_rows"]["stop"],
            ),
        ),
        (
            "stable_row_softmax",
            lambda ref: stable_row_softmax(_matrix(ref["softmax"]["ordinary"]["scores"])),
        ),
    ],
)
def test_every_pure_public_operation_returns_float32_c_contiguous_independent_result(
    reference: dict[str, Any],
    operation_name: str,
    operation: Callable[
        [dict[str, Any]],
        Float32Matrix,
    ],
) -> None:
    result = operation(reference)

    assert result.dtype == np.dtype(np.float32), operation_name
    assert result.flags.c_contiguous, operation_name
    assert np.isfinite(result).all(), operation_name

    result.fill(np.float32(123.0))

    later_result = operation(reference)
    assert not np.all(later_result == np.float32(123.0)), operation_name


def test_matmul_uses_float64_accumulation(
    reference: dict[str, Any],
) -> None:
    case = reference["matmul"]
    left = _matrix(case["left"])
    right = _matrix(case["right"])
    before = (
        left.copy(),
        right.copy(),
    )

    result = matmul(left, right)
    expected = _matrix(case["expected"])

    _assert_pure_result(
        result,
        expected,
        (left, right),
        before,
    )

    float32_default = left @ right

    assert float32_default[0, 0] == np.float32(0.0)
    assert result[0, 0] == np.float32(1.0)


def test_transposed_matmul_variants_match_independent_fixtures(
    reference: dict[str, Any],
) -> None:
    right_case = reference["matmul_transposed_right"]
    right_left = _matrix(right_case["left"])
    right_right = _matrix(right_case["right"])

    right_result = matmul_transposed_right(
        right_left,
        right_right,
    )

    _assert_pure_result(
        right_result,
        _matrix(right_case["expected"]),
        (right_left, right_right),
        (
            right_left.copy(),
            right_right.copy(),
        ),
    )

    left_case = reference["matmul_transposed_left"]
    left_left = _matrix(left_case["left"])
    left_right = _matrix(left_case["right"])

    left_result = matmul_transposed_left(
        left_left,
        left_right,
    )

    _assert_pure_result(
        left_result,
        _matrix(left_case["expected"]),
        (left_left, left_right),
        (
            left_left.copy(),
            left_right.copy(),
        ),
    )


def test_sum_columns_uses_float64_accumulation_and_one_row_shape(
    reference: dict[str, Any],
) -> None:
    case = reference["sum_columns"]
    matrix = _matrix(case["matrix"])

    result = sum_columns(matrix)
    expected = _matrix(case["expected"])

    _assert_pure_result(
        result,
        expected,
        (matrix,),
        (matrix.copy(),),
    )

    assert result.shape == (1, matrix.shape[1])

    float32_sum = np.sum(
        matrix,
        axis=0,
        dtype=np.float32,
    )

    assert float32_sum[0] == np.float32(0.0)
    assert result[0, 0] == np.float32(1.0)


@pytest.mark.parametrize(
    ("function", "expected_key"),
    [
        (elementwise_add, "add"),
        (elementwise_subtract, "subtract"),
        (elementwise_multiply, "multiply"),
    ],
)
def test_elementwise_operations_require_equal_shapes_and_match_fixture(
    reference: dict[str, Any],
    function: Callable[
        [Float32Matrix, Float32Matrix],
        Float32Matrix,
    ],
    expected_key: str,
) -> None:
    case = reference["elementwise"]
    left = _matrix(case["left"])
    right = _matrix(case["right"])

    result = function(left, right)

    _assert_pure_result(
        result,
        _matrix(case[expected_key]),
        (left, right),
        (
            left.copy(),
            right.copy(),
        ),
    )

    invalid_right = np.ones(
        (1, left.shape[1]),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="equal shapes"):
        function(left, invalid_right)


def test_row_bias_accepts_only_exact_one_row_shape(
    reference: dict[str, Any],
) -> None:
    case = reference["row_bias"]
    matrix = _matrix(case["matrix"])
    bias = _matrix(case["bias"])

    result = add_row_bias(matrix, bias)

    _assert_pure_result(
        result,
        _matrix(case["expected"]),
        (matrix, bias),
        (
            matrix.copy(),
            bias.copy(),
        ),
    )

    invalid_biases = (
        np.array(
            [0.25, -1.0, 2.0],
            dtype=np.float32,
        ),
        np.ones(
            (matrix.shape[0], 1),
            dtype=np.float32,
        ),
        np.ones(
            matrix.shape,
            dtype=np.float32,
        ),
    )

    for invalid_bias in invalid_biases:
        with pytest.raises(ValueError):
            add_row_bias(matrix, invalid_bias)


def test_scalar_transpose_concatenate_and_slice_match_fixtures(
    reference: dict[str, Any],
) -> None:
    scalar_case = reference["scalar_multiply"]
    scalar_input = _matrix(scalar_case["matrix"])

    scalar_result = scalar_multiply(
        scalar_input,
        scalar_case["scalar"],
    )

    _assert_pure_result(
        scalar_result,
        _matrix(scalar_case["expected"]),
        (scalar_input,),
        (scalar_input.copy(),),
    )

    transpose_case = reference["transpose"]
    transpose_input = _matrix(transpose_case["matrix"])

    transpose_result = transpose(transpose_input)

    _assert_pure_result(
        transpose_result,
        _matrix(transpose_case["expected"]),
        (transpose_input,),
        (transpose_input.copy(),),
    )

    concatenate_case = reference["concatenate_rows"]
    top = _matrix(concatenate_case["top"])
    bottom = _matrix(concatenate_case["bottom"])

    concatenate_result = concatenate_rows(
        top,
        bottom,
    )

    _assert_pure_result(
        concatenate_result,
        _matrix(concatenate_case["expected"]),
        (top, bottom),
        (
            top.copy(),
            bottom.copy(),
        ),
    )

    slice_case = reference["slice_rows"]
    slice_input = _matrix(slice_case["matrix"])

    slice_result = slice_rows(
        slice_input,
        slice_case["start"],
        slice_case["stop"],
    )

    _assert_pure_result(
        slice_result,
        _matrix(slice_case["expected"]),
        (slice_input,),
        (slice_input.copy(),),
    )


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        True,
        np.float64(1.0),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_scalar_multiply_rejects_unsupported_or_non_finite_scalars(
    scalar: object,
) -> None:
    matrix = np.ones(
        (2, 2),
        dtype=np.float32,
    )

    with pytest.raises((TypeError, FloatingPointError)):
        scalar_multiply(matrix, scalar)


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (-1, 1),
        (0, 0),
        (2, 1),
        (0, 4),
        (True, 2),
        (0, 2.0),
    ],
)
def test_slice_rows_rejects_python_index_shortcuts(
    start: object,
    stop: object,
) -> None:
    matrix = np.arange(
        9,
        dtype=np.float32,
    ).reshape(3, 3)

    with pytest.raises((TypeError, ValueError)):
        slice_rows(matrix, start, stop)


def test_concatenate_rows_rejects_column_mismatch() -> None:
    top = np.ones(
        (2, 3),
        dtype=np.float32,
    )
    bottom = np.ones(
        (1, 2),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="equal column counts",
    ):
        concatenate_rows(top, bottom)


@pytest.mark.parametrize(
    "bad_value",
    [
        [[1.0, 2.0], [3.0, 4.0]],
        np.ones((2, 2), dtype=np.float64),
        np.ones(4, dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.array(
            [[1.0, np.nan], [2.0, 3.0]],
            dtype=np.float32,
        ),
        np.array(
            [[1.0, np.inf], [2.0, 3.0]],
            dtype=np.float32,
        ),
        np.array(
            [[1.0, -np.inf], [2.0, 3.0]],
            dtype=np.float32,
        ),
    ],
)
def test_ordinary_operations_reject_invalid_public_inputs(
    bad_value: object,
) -> None:
    valid = np.ones(
        (2, 2),
        dtype=np.float32,
    )

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        elementwise_add(bad_value, valid)


def test_ordinary_operations_reject_non_contiguous_inputs() -> None:
    base = np.arange(
        12,
        dtype=np.float32,
    ).reshape(3, 4)
    non_contiguous = base[:, ::2]

    assert not non_contiguous.flags.c_contiguous

    with pytest.raises(
        ValueError,
        match="C-contiguous",
    ):
        elementwise_add(
            non_contiguous,
            non_contiguous,
        )


@pytest.mark.parametrize(
    ("function", "left_shape", "right_shape"),
    [
        (matmul, (2, 3), (2, 4)),
        (
            matmul_transposed_right,
            (2, 3),
            (4, 2),
        ),
        (
            matmul_transposed_left,
            (2, 3),
            (4, 5),
        ),
    ],
)
def test_multiplication_variants_reject_incompatible_shapes(
    function: Callable[
        [Float32Matrix, Float32Matrix],
        Float32Matrix,
    ],
    left_shape: tuple[int, int],
    right_shape: tuple[int, int],
) -> None:
    left = np.ones(
        left_shape,
        dtype=np.float32,
    )
    right = np.ones(
        right_shape,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        function(left, right)


def test_pure_operations_accept_read_only_sources() -> None:
    left = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    )
    right = np.array(
        [[5.0, 6.0], [7.0, 8.0]],
        dtype=np.float32,
    )

    left.setflags(write=False)
    right.setflags(write=False)

    result = elementwise_add(left, right)

    assert_array_equal(
        result,
        np.array(
            [[6.0, 8.0], [10.0, 12.0]],
            dtype=np.float32,
        ),
    )
    assert result.flags.writeable


def test_add_in_place_commits_only_destination(
    reference: dict[str, Any],
) -> None:
    case = reference["add_in_place"]
    destination = _matrix(case["destination"])
    source = _matrix(case["source"])
    source_before = source.copy()

    result = add_in_place(destination, source)

    assert result is None
    assert_array_equal(
        destination,
        _matrix(case["expected"]),
    )
    assert_array_equal(source, source_before)


def test_add_in_place_rejects_same_and_partially_overlapping_memory() -> None:
    same = np.arange(
        6,
        dtype=np.float32,
    ).reshape(2, 3)
    same_before = same.tobytes()

    with pytest.raises(
        ValueError,
        match="must not share memory",
    ):
        add_in_place(same, same)

    assert same.tobytes() == same_before

    base = np.arange(
        7,
        dtype=np.float32,
    )
    destination = base[:6].reshape(2, 3)
    source = base[1:7].reshape(2, 3)

    assert destination.flags.c_contiguous
    assert source.flags.c_contiguous
    assert np.shares_memory(destination, source)

    destination_before = destination.tobytes()

    with pytest.raises(
        ValueError,
        match="must not share memory",
    ):
        add_in_place(destination, source)

    assert destination.tobytes() == destination_before


def test_add_in_place_overflow_is_transactional() -> None:
    maximum = np.finfo(np.float32).max

    destination = np.full(
        (2, 2),
        maximum,
        dtype=np.float32,
    )
    source = np.full(
        (2, 2),
        maximum,
        dtype=np.float32,
    )

    destination_before = destination.tobytes()

    with pytest.raises(
        FloatingPointError,
        match="materialized",
    ):
        add_in_place(destination, source)

    assert destination.tobytes() == destination_before


@pytest.mark.parametrize(
    "source",
    [
        np.ones((1, 2), dtype=np.float32),
        np.array(
            [[np.nan, 1.0], [2.0, 3.0]],
            dtype=np.float32,
        ),
        np.ones((2, 2), dtype=np.float64),
    ],
)
def test_add_in_place_validation_failures_preserve_destination(
    source: np.ndarray[Any, Any],
) -> None:
    destination = np.ones(
        (2, 2),
        dtype=np.float32,
    )
    destination_before = destination.tobytes()

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        add_in_place(destination, source)

    assert destination.tobytes() == destination_before


def test_add_in_place_rejects_read_only_destination() -> None:
    destination = np.ones(
        (2, 2),
        dtype=np.float32,
    )
    source = np.ones(
        (2, 2),
        dtype=np.float32,
    )

    destination.setflags(write=False)

    with pytest.raises(
        ValueError,
        match="writable",
    ):
        add_in_place(destination, source)


@pytest.mark.parametrize(
    "case_name",
    [
        "ordinary",
        "large",
        "causal",
        "single_selectable",
    ],
)
def test_stable_row_softmax_matches_independent_fixtures(
    reference: dict[str, Any],
    case_name: str,
) -> None:
    case = reference["softmax"][case_name]
    scores = _matrix(case["scores"])
    scores_before = scores.copy()

    result = stable_row_softmax(scores)
    expected = _matrix(case["expected"])
    tolerances = reference["tolerances"]

    _assert_pure_result(
        result,
        expected,
        (scores,),
        (scores_before,),
        rtol=tolerances["softmax_rtol"],
        atol=tolerances["softmax_atol"],
    )

    assert_allclose(
        np.sum(
            result,
            axis=1,
            dtype=np.float64,
        ),
        np.ones(
            result.shape[0],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-6,
    )


def test_causal_softmax_mask_positions_are_exact_zero(
    reference: dict[str, Any],
) -> None:
    scores = _matrix(reference["softmax"]["causal"]["scores"])

    result = stable_row_softmax(scores)
    mask = np.isneginf(scores)

    assert np.all(result[mask] == np.float32(0.0))
    assert result[0, 0] == np.float32(1.0)


@pytest.mark.parametrize(
    "case_name",
    [
        "all_masked",
        "nan",
        "positive_infinity",
    ],
)
def test_stable_row_softmax_rejects_invalid_rows_without_mutation(
    reference: dict[str, Any],
    case_name: str,
) -> None:
    scores = _matrix(reference["softmax"]["invalid"][case_name])
    before = scores.tobytes()

    with pytest.raises((ValueError, FloatingPointError)):
        stable_row_softmax(scores)

    assert scores.tobytes() == before


def test_stable_row_softmax_rejects_wrong_dtype_rank_and_layout() -> None:
    with pytest.raises(TypeError):
        stable_row_softmax(
            np.ones(
                (2, 2),
                dtype=np.float64,
            )
        )

    with pytest.raises(ValueError):
        stable_row_softmax(
            np.ones(
                2,
                dtype=np.float32,
            )
        )

    base = np.arange(
        12,
        dtype=np.float32,
    ).reshape(3, 4)
    non_contiguous = base[:, ::2]

    with pytest.raises(
        ValueError,
        match="C-contiguous",
    ):
        stable_row_softmax(non_contiguous)


def test_softmax_accepts_read_only_scores_and_leaves_them_unchanged() -> None:
    scores = np.array(
        [[1.0, 2.0, -np.inf]],
        dtype=np.float32,
    )
    before = scores.copy()

    scores.setflags(write=False)

    result = stable_row_softmax(scores)

    assert_array_equal(scores, before)
    assert result.flags.writeable
    assert result[0, 2] == np.float32(0.0)


def test_repeated_and_threaded_calls_have_no_shared_mutable_state(
    reference: dict[str, Any],
) -> None:
    case = reference["matmul"]
    expected = _matrix(case["expected"])

    def run_once(_: int) -> Float32Matrix:
        return matmul(
            _matrix(case["left"]),
            _matrix(case["right"]),
        )

    sequential = [run_once(index) for index in range(4)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        threaded = list(
            executor.map(
                run_once,
                range(8),
            )
        )

    for result in sequential + threaded:
        assert_array_equal(result, expected)

    for first, second in zip(
        threaded,
        threaded[1:],
        strict=False,
    ):
        assert not np.shares_memory(first, second)


def test_module_has_no_mutable_global_array_or_random_state() -> None:
    module_values = vars(matrix_module)

    assert not any(
        isinstance(value, np.ndarray)
        for name, value in module_values.items()
        if not name.startswith("__")
    )

    assert not any(
        token in name.lower()
        for name in module_values
        for token in (
            "rand",
            "generator",
            "cache",
            "scratch",
        )
        if not name.startswith("__")
    )
