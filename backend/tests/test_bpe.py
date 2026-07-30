# backend/tests/test_bpe.py

import pytest
from how_llms_work.ml.bpe import MAX_MERGES, Merge, apply_merges, count_words, train_bpe


def test_count_words_preserves_reference_pretoken_order_and_frequency() -> None:
    counts = count_words("the cat and the cat")

    assert list(counts.items()) == [
        ("the", 2),
        (" ", 4),
        ("cat", 2),
        ("and", 1),
    ]


def test_count_words_uses_ascii_words_and_unicode_whitespace() -> None:
    counts = count_words("café\u00a0café")

    assert list(counts.items()) == [
        ("caf", 2),
        ("é", 2),
        ("\u00a0", 1),
    ]


def test_train_bpe_weights_repeated_pretokens_and_records_merge_order() -> None:
    merges = train_bpe(count_words("cat cat car"))

    assert merges == (
        Merge(pair=("c", "a"), merged="ca", frequency=3),
        Merge(pair=("ca", "t"), merged="cat", frequency=2),
        Merge(pair=("ca", "r"), merged="car", frequency=1),
    )
    assert apply_merges("cat cat car", merges) == [
        "cat",
        " ",
        "cat",
        " ",
        "car",
    ]


def test_train_bpe_resolves_frequency_ties_by_first_encounter() -> None:
    merges = train_bpe(count_words("ab ac"))

    assert merges[0] == Merge(
        pair=("a", "b"),
        merged="ab",
        frequency=1,
    )


def test_train_bpe_counts_overlapping_candidates_but_merges_non_overlappingly() -> None:
    merges = train_bpe(count_words("aaaa"))

    assert merges == (
        Merge(pair=("a", "a"), merged="aa", frequency=3),
        Merge(pair=("aa", "aa"), merged="aaaa", frequency=1),
    )
    assert apply_merges("aaaa", merges) == ["aaaa"]


def test_apply_merges_replays_rules_in_order_without_crossing_boundaries() -> None:
    ordered_merges = (
        Merge(pair=("c", "a"), merged="ca", frequency=1),
        Merge(pair=("ca", "t"), merged="cat", frequency=1),
        Merge(pair=("s", "a"), merged="sa", frequency=1),
        Merge(pair=("sa", "t"), merged="sat", frequency=1),
        Merge(pair=("cat", " "), merged="cat ", frequency=1),
    )
    reversed_dependency = (
        Merge(pair=("ca", "t"), merged="cat", frequency=1),
        Merge(pair=("c", "a"), merged="ca", frequency=1),
    )

    assert apply_merges("cat sat", ordered_merges) == ["cat", " ", "sat"]
    assert apply_merges("cat", reversed_dependency) == ["ca", "t"]


@pytest.mark.parametrize(
    ("text", "expected_tokens"),
    [
        ("x", ["x"]),
        (" \t\n", [" ", "\t", "\n"]),
        ("!?!", ["!", "?", "!"]),
    ],
)
def test_minimal_inputs_finish_without_invalid_merges(
    text: str,
    expected_tokens: list[str],
) -> None:
    merges = train_bpe(count_words(text))

    assert merges == ()
    assert apply_merges(text, merges) == expected_tokens


def test_train_bpe_honors_requested_bounds_and_the_production_ceiling() -> None:
    limited_merges = train_bpe(count_words("abcdef"), max_merges=2)

    assert limited_merges == (
        Merge(pair=("a", "b"), merged="ab", frequency=1),
        Merge(pair=("ab", "c"), merged="abc", frequency=1),
    )
    assert MAX_MERGES == 1_000

    # Build one ASCII word with more than 1,000 unique adjacent pairs.
    # The resulting BPE run can continue beyond 1,000 steps unless capped.
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456"
    adjacency = {character: list(reversed(alphabet)) for character in alphabet}
    stack = [alphabet[0]]
    circuit: list[str] = []

    while stack:
        current = stack[-1]

        if adjacency[current]:
            stack.append(adjacency[current].pop())
        else:
            circuit.append(stack.pop())

    long_word = "".join(reversed(circuit))
    capped_merges = train_bpe(
        count_words(long_word),
        max_merges=MAX_MERGES + 50,
    )

    assert len(capped_merges) == MAX_MERGES


def test_public_calls_do_not_share_mutable_state() -> None:
    first_counts = count_words("cat")
    second_counts = count_words("cat")
    first_counts["cat"] = 99

    merges = train_bpe(second_counts)
    first_tokens = apply_merges("cat", merges)
    first_tokens.append("changed")

    assert second_counts == {"cat": 1}
    assert train_bpe(count_words("cat")) == merges
    assert apply_merges("cat", merges) == ["cat"]
