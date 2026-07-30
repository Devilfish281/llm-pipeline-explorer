# tests/test_math_utils.py
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from how_llms_work.ml import math_utils, neural_net, word2vec
from how_llms_work.ml.math_utils import (
    Mulberry32,
    normalize_public_number,
    round_typescript_decimal,
    round_typescript_decimal_raw,
)

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "math_utils_reference.json"

with REFERENCE_PATH.open(encoding="utf-8") as reference_file:
    REFERENCE: dict[str, Any] = json.load(reference_file)

MULBERRY32_CASES: tuple[dict[str, Any], ...] = tuple(REFERENCE["mulberry32_cases"])
ROUNDING_CASES: tuple[dict[str, Any], ...] = tuple(REFERENCE["rounding_cases"])


def consume_random_case(
    case: dict[str, Any],
) -> tuple[list[float], int, int]:
    """Consume one independent fixture-defined Mulberry32 stream."""
    generator = Mulberry32(case["seed"])
    outputs = [generator.random() for _ in range(case["draw_count"])]

    return (
        outputs,
        generator.state,
        generator.draw_count,
    )


def test_reference_fixture_was_captured_without_production_imports() -> None:
    provenance = REFERENCE["provenance"]

    assert provenance["production_imports_used"] is False


@pytest.mark.parametrize(
    "case",
    MULBERRY32_CASES,
    ids=lambda case: str(case["name"]),
)
def test_mulberry32_matches_independent_reference(
    case: dict[str, Any],
) -> None:
    generator = Mulberry32(case["seed"])

    assert generator.state == case["normalized_initial_state"]
    assert generator.draw_count == 0

    outputs = [generator.random() for _ in range(case["draw_count"])]

    assert outputs == case["outputs"]
    assert generator.state == case["final_state"]
    assert generator.draw_count == case["draw_count"]
    assert all(0.0 <= output < 1.0 for output in outputs)


def test_same_seed_generators_advance_independently_in_lockstep() -> None:
    expected = next(case for case in MULBERRY32_CASES if case["name"] == "seed_42")

    first = Mulberry32(expected["seed"])
    second = Mulberry32(expected["seed"])

    first_value = first.random()

    assert first_value == expected["outputs"][0]
    assert first.draw_count == 1
    assert second.state == expected["normalized_initial_state"]
    assert second.draw_count == 0

    second_value = second.random()

    assert second_value == first_value
    assert second.state == first.state
    assert second.draw_count == first.draw_count

    for expected_value in expected["outputs"][1:]:
        assert first.random() == expected_value
        assert second.random() == expected_value
        assert first.state == second.state
        assert first.draw_count == second.draw_count


def test_interleaved_generators_match_uninterrupted_controls() -> None:
    first_case = next(case for case in MULBERRY32_CASES if case["name"] == "seed_42")
    second_case = next(case for case in MULBERRY32_CASES if case["name"] == "seed_zero")

    first = Mulberry32(first_case["seed"])
    second = Mulberry32(second_case["seed"])

    first_outputs: list[float] = []
    second_outputs: list[float] = []

    for index in range(first_case["draw_count"]):
        if index % 2 == 0:
            second_outputs.append(second.random())
            first_outputs.append(first.random())
        else:
            first_outputs.append(first.random())
            second_outputs.append(second.random())

    assert first_outputs == first_case["outputs"]
    assert second_outputs == second_case["outputs"]
    assert first.state == first_case["final_state"]
    assert second.state == second_case["final_state"]
    assert first.draw_count == first_case["draw_count"]
    assert second.draw_count == second_case["draw_count"]


def test_weight_initialization_and_sample_streams_do_not_interfere() -> None:
    epoch = 17

    initialization_control = Mulberry32(42)
    sample_control = Mulberry32((42 + epoch) & 0xFFFFFFFF)

    expected_initialization = [initialization_control.random() for _ in range(8)]
    expected_samples = [sample_control.random() for _ in range(8)]

    initialization_stream = Mulberry32(42)
    sample_stream = Mulberry32((42 + epoch) & 0xFFFFFFFF)

    initialization_outputs = [initialization_stream.random() for _ in range(3)]
    sample_outputs = [sample_stream.random() for _ in range(5)]

    initialization_outputs.extend(initialization_stream.random() for _ in range(5))
    sample_outputs.extend(sample_stream.random() for _ in range(3))

    assert initialization_outputs == expected_initialization
    assert sample_outputs == expected_samples
    assert initialization_stream.state == initialization_control.state
    assert sample_stream.state == sample_control.state
    assert initialization_stream.draw_count == initialization_control.draw_count
    assert sample_stream.draw_count == sample_control.draw_count


def test_sequential_random_streams_match_fixed_reference() -> None:
    actual = [consume_random_case(case) for case in MULBERRY32_CASES]
    expected = [
        (
            case["outputs"],
            case["final_state"],
            case["draw_count"],
        )
        for case in MULBERRY32_CASES
    ]

    assert actual == expected


def test_concurrent_random_streams_match_fixed_reference() -> None:
    with ThreadPoolExecutor(max_workers=len(MULBERRY32_CASES)) as executor:
        actual = list(
            executor.map(
                consume_random_case,
                MULBERRY32_CASES,
            )
        )

    expected = [
        (
            case["outputs"],
            case["final_state"],
            case["draw_count"],
        )
        for case in MULBERRY32_CASES
    ]

    assert actual == expected


@pytest.mark.parametrize(
    "case",
    ROUNDING_CASES,
    ids=lambda case: str(case["name"]),
)
def test_typescript_decimal_rounding_matches_independent_reference(
    case: dict[str, Any],
) -> None:
    raw_result = round_typescript_decimal_raw(
        case["value"],
        case["digits"],
    )
    public_result = round_typescript_decimal(
        case["value"],
        case["digits"],
    )

    assert raw_result == case["raw_expected"]
    assert public_result == case["public_expected"]

    raw_zero_sign = case["raw_zero_sign"]

    if raw_zero_sign is not None:
        assert (
            math.copysign(
                1.0,
                raw_result,
            )
            == raw_zero_sign
        )

    if public_result == 0.0:
        assert (
            math.copysign(
                1.0,
                public_result,
            )
            == 1.0
        )


@pytest.mark.parametrize(
    "operation",
    (
        round_typescript_decimal_raw,
        round_typescript_decimal,
    ),
)
@pytest.mark.parametrize(
    (
        "value",
        "digits",
        "error_type",
        "message",
    ),
    (
        (
            1.25,
            -1,
            ValueError,
            "digits must be non-negative",
        ),
        (
            math.nan,
            6,
            FloatingPointError,
            "value is not finite",
        ),
        (
            math.inf,
            6,
            FloatingPointError,
            "value is not finite",
        ),
        (
            -math.inf,
            6,
            FloatingPointError,
            "value is not finite",
        ),
        (
            1.0,
            10_000,
            FloatingPointError,
            "scaled value is not finite",
        ),
    ),
)
def test_decimal_rounding_rejects_invalid_inputs(
    operation: Any,
    value: float,
    digits: int,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(
        error_type,
        match=message,
    ):
        operation(
            value,
            digits,
        )


def test_public_number_normalization_is_finite_and_positive_zero_only() -> None:
    for value in (
        math.nan,
        math.inf,
        -math.inf,
    ):
        with pytest.raises(
            FloatingPointError,
            match="value is not finite",
        ):
            normalize_public_number(value)

    assert normalize_public_number(-0.0) == 0.0
    assert (
        math.copysign(
            1.0,
            normalize_public_number(-0.0),
        )
        == 1.0
    )
    assert (
        math.copysign(
            1.0,
            normalize_public_number(0.0),
        )
        == 1.0
    )


def test_word2vec_reexports_shared_deterministic_utilities() -> None:
    assert word2vec.Mulberry32 is math_utils.Mulberry32
    assert word2vec.round_typescript_decimal is math_utils.round_typescript_decimal


def test_xor_rounding_wrapper_preserves_behavior_and_rejects_non_finite_values() -> None:
    finite_cases = (
        (
            np.float32(1.2345674),
            6,
            1.234567,
        ),
        (
            np.float32(-1.5),
            0,
            -1.0,
        ),
        (
            np.float32(-0.0000004),
            6,
            0.0,
        ),
    )

    for value, digits, expected in finite_cases:
        result = neural_net.round_like_typescript(
            value,
            digits,
        )

        assert result == expected

        if result == 0.0:
            assert (
                math.copysign(
                    1.0,
                    result,
                )
                == 1.0
            )

    for value in (
        math.nan,
        math.inf,
        -math.inf,
    ):
        with pytest.raises(
            FloatingPointError,
            match="value is not finite",
        ):
            neural_net.round_like_typescript(
                value,
                6,
            )
