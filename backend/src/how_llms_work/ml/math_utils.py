# backend/src/how_llms_work/ml/math_utils.py
"""Shared deterministic numerical compatibility utilities."""

from __future__ import annotations

import math
from typing import Final

_UINT32_MASK: Final[int] = 0xFFFFFFFF
_MULBERRY32_INCREMENT: Final[int] = 0x6D2B79F5
_MULBERRY32_DIVISOR: Final[float] = 4_294_967_296.0

__all__ = [
    "Mulberry32",
    "normalize_public_number",
    "round_typescript_decimal",
    "round_typescript_decimal_raw",
]


class Mulberry32:
    """One independent JavaScript-compatible Mulberry32 random stream."""

    __slots__ = (
        "_draw_count",
        "_state",
    )

    def __init__(
        self,
        seed: int,
    ) -> None:
        self._state = seed & _UINT32_MASK
        self._draw_count = 0

    @property
    def state(self) -> int:
        """Return the current unsigned 32-bit generator state."""
        return self._state

    @property
    def draw_count(self) -> int:
        """Return the number of values consumed from this generator."""
        return self._draw_count

    def random(self) -> float:
        """Return the next deterministic value in the interval [0, 1)."""
        self._state = (self._state + _MULBERRY32_INCREMENT) & _UINT32_MASK

        value = ((self._state ^ (self._state >> 15)) * (1 | self._state)) & _UINT32_MASK

        value = ((value + ((value ^ (value >> 7)) * (61 | value))) & _UINT32_MASK) ^ value

        result = ((value ^ (value >> 14)) & _UINT32_MASK) / _MULBERRY32_DIVISOR

        self._draw_count += 1

        return result


def round_typescript_decimal_raw(
    value: float,
    digits: int,
) -> float:
    """Round like Math.round(value * scale) / scale, preserving signed zero."""
    if digits < 0:
        raise ValueError("digits must be non-negative")

    if not math.isfinite(value):
        raise FloatingPointError("value is not finite")

    try:
        scale = 10.0**digits
    except OverflowError as error:
        raise FloatingPointError("scaled value is not finite") from error

    if not math.isfinite(scale):
        raise FloatingPointError("scaled value is not finite")

    scaled_value = value * scale

    if not math.isfinite(scaled_value):
        raise FloatingPointError("scaled value is not finite")

    lower_integer = math.floor(scaled_value)
    fractional_part = scaled_value - lower_integer

    if fractional_part < 0.5:
        rounded_integer = lower_integer
    else:
        rounded_integer = lower_integer + 1

    if (
        rounded_integer == 0
        and math.copysign(
            1.0,
            scaled_value,
        )
        < 0.0
    ):
        rounded = -0.0
    else:
        rounded = rounded_integer / scale

    if not math.isfinite(rounded):
        raise FloatingPointError("rounded value is not finite")

    return rounded


def normalize_public_number(
    value: float,
) -> float:
    """Reject non-finite values and normalize either zero sign to positive."""
    if not math.isfinite(value):
        raise FloatingPointError("value is not finite")

    return 0.0 if value == 0.0 else value


def round_typescript_decimal(
    value: float,
    digits: int,
) -> float:
    """Round with TypeScript semantics and normalize zero for public output."""
    return normalize_public_number(
        round_typescript_decimal_raw(
            value,
            digits,
        )
    )
