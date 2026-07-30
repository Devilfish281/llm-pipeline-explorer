# backend/src/how_llms_work/ml/bpe.py

"""Reference-compatible educational Byte Pair Encoding operations."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

MAX_MERGES: Final = 1_000

PRE_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+|\s|[^A-Za-z0-9_\s]")


@dataclass(frozen=True, slots=True)
class Merge:
    """One learned BPE pair replacement."""

    pair: tuple[str, str]
    merged: str
    frequency: int


def count_words(text: str) -> dict[str, int]:
    """Count reference-compatible Pre-tokens in first-encounter order."""
    counts: dict[str, int] = {}

    for pre_token in PRE_TOKEN_PATTERN.findall(text):
        counts[pre_token] = counts.get(pre_token, 0) + 1

    return counts


def _merge_tokens(
    tokens: Sequence[str],
    pair: tuple[str, str],
    merged: str,
) -> list[str]:
    """Replace non-overlapping pair occurrences from left to right."""
    result: list[str] = []
    index = 0

    while index < len(tokens):
        if index < len(tokens) - 1 and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            result.append(merged)
            index += 2
        else:
            result.append(tokens[index])
            index += 1

    return result


def train_bpe(
    word_frequencies: Mapping[str, int],
    max_merges: int = MAX_MERGES,
) -> tuple[Merge, ...]:
    """Learn an ordered, deterministic BPE Merge Table."""
    word_splits: dict[str, list[str]] = {word: list(word) for word in word_frequencies}
    merges: list[Merge] = []

    merge_limit = min(
        max(max_merges, 0),
        MAX_MERGES,
    )

    for _ in range(merge_limit):
        pair_frequencies: dict[tuple[str, str], int] = {}

        for word, tokens in word_splits.items():
            weight = word_frequencies[word]

            for index in range(len(tokens) - 1):
                pair = (
                    tokens[index],
                    tokens[index + 1],
                )
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + weight

        if not pair_frequencies:
            break

        pair_items = iter(pair_frequencies.items())
        best_pair, best_frequency = next(pair_items)

        for pair, frequency in pair_items:
            if frequency > best_frequency:
                best_pair = pair
                best_frequency = frequency

        merged = best_pair[0] + best_pair[1]

        for word, tokens in word_splits.items():
            word_splits[word] = _merge_tokens(
                tokens,
                best_pair,
                merged,
            )

        merges.append(
            Merge(
                pair=best_pair,
                merged=merged,
                frequency=best_frequency,
            )
        )

    return tuple(merges)


def apply_merges(
    text: str,
    merges: Sequence[Merge],
) -> list[str]:
    """Apply a learned Merge Table in order within each Pre-token."""
    result: list[str] = []

    for pre_token in PRE_TOKEN_PATTERN.findall(text):
        tokens: list[str] = list(pre_token)

        for merge in merges:
            tokens = _merge_tokens(
                tokens,
                merge.pair,
                merge.merged,
            )

        result.extend(tokens)

    return result
