# backend/tests/test_word2vec.py
# Test cases for the Word2Vec implementation
from __future__ import annotations

import hashlib
import inspect
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from how_llms_work.ml.word2vec import (
    BPE_MERGE_LIMIT,
    REFERENCE_MERGE_COUNT,
    SUPPORTED_WINDOW_SIZES,
    TrainingPair,
    get_word2vec_preprocessing,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "word2vec_preprocessing_reference.json"

REFERENCE_FIXTURE = cast(
    dict[str, Any],
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
)


def _canonical_json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def _training_pair_sha256(
    pairs: tuple[TrainingPair, ...],
) -> str:
    packed = b"".join(
        struct.pack(
            ">HH",
            pair.target,
            pair.context,
        )
        for pair in pairs
    )

    return hashlib.sha256(packed).hexdigest()


def test_preprocessing_matches_reference_corpus_training_text_and_merges() -> None:
    preprocessing = get_word2vec_preprocessing()
    corpus_fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["corpus"],
    )
    merge_fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["merges"],
    )
    merge_records = [
        [
            merge.pair[0],
            merge.pair[1],
            merge.merged,
            merge.frequency,
        ]
        for merge in preprocessing.merges
    ]

    assert len(preprocessing.corpus) == corpus_fixture["count"] == 107
    assert preprocessing.corpus[0] == corpus_fixture["first"]
    assert preprocessing.corpus[-1] == corpus_fixture["last"]
    assert _canonical_json_sha256(list(preprocessing.corpus)) == corpus_fixture["sha256"]

    assert preprocessing.training_text == " ".join(preprocessing.corpus).lower()

    assert (
        hashlib.sha256(preprocessing.training_text.encode("utf-8")).hexdigest()
        == REFERENCE_FIXTURE["training_text_sha256"]
    )

    assert preprocessing.merge_limit == BPE_MERGE_LIMIT == REFERENCE_FIXTURE["merge_limit"]
    assert len(preprocessing.merges) == REFERENCE_MERGE_COUNT == merge_fixture["count"]
    assert _canonical_json_sha256(merge_records) == merge_fixture["sha256"]

    assert {
        "pair": list(preprocessing.merges[0].pair),
        "merged": preprocessing.merges[0].merged,
        "frequency": preprocessing.merges[0].frequency,
    } == merge_fixture["first"]

    assert {
        "pair": list(preprocessing.merges[-1].pair),
        "merged": preprocessing.merges[-1].merged,
        "frequency": preprocessing.merges[-1].frequency,
    } == merge_fixture["last"]


def test_preprocessing_matches_reference_tokenized_sentences() -> None:
    preprocessing = get_word2vec_preprocessing()
    tokenized_fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["tokenized_sentences"],
    )
    tokenized_records = [list(tokens) for tokens in preprocessing.tokenized_sentences]

    assert len(preprocessing.tokenized_sentences) == tokenized_fixture["count"] == 107
    assert _canonical_json_sha256(tokenized_records) == tokenized_fixture["sha256"]

    representative_fixtures = cast(
        list[dict[str, Any]],
        tokenized_fixture["representative"],
    )

    for fixture in representative_fixtures:
        sentence_index = cast(
            int,
            fixture["index"],
        )
        assert preprocessing.corpus[sentence_index] == fixture["sentence"]
        assert preprocessing.tokenized_sentences[sentence_index] == tuple(fixture["tokens"])

    for sentence, tokens in zip(
        preprocessing.corpus,
        preprocessing.tokenized_sentences,
        strict=True,
    ):
        assert "".join(tokens) == sentence.lower()


def test_preprocessing_matches_reference_frequencies_vocabulary_and_indices() -> None:
    preprocessing = get_word2vec_preprocessing()
    frequency_fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["frequencies"],
    )
    vocabulary_fixture = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["vocabulary"],
    )

    frequency_records = [
        [token, frequency] for token, frequency in preprocessing.token_frequencies.items()
    ]
    index_records = [[token, index] for token, index in preprocessing.token_indices.items()]

    assert len(preprocessing.token_frequencies) == frequency_fixture["count"] == 192
    assert _canonical_json_sha256(frequency_records) == frequency_fixture["sha256"]
    assert len(preprocessing.vocabulary) == vocabulary_fixture["count"] == 192
    assert _canonical_json_sha256(list(preprocessing.vocabulary)) == vocabulary_fixture["sha256"]
    assert list(preprocessing.vocabulary[:20]) == vocabulary_fixture["first_20"]
    assert _canonical_json_sha256(index_records) == REFERENCE_FIXTURE["indices_sha256"]

    assert list(preprocessing.token_indices.items()) == [
        (token, index) for index, token in enumerate(preprocessing.vocabulary)
    ]

    tie_groups = cast(
        list[dict[str, Any]],
        REFERENCE_FIXTURE["stable_tie_groups"],
    )

    for group in tie_groups:
        frequency = cast(
            int,
            group["frequency"],
        )
        tokens = cast(
            list[str],
            group["tokens"],
        )

        assert [preprocessing.token_frequencies[token] for token in tokens] == [frequency] * len(
            tokens
        )

        assert [preprocessing.token_indices[token] for token in tokens] == sorted(
            preprocessing.token_indices[token] for token in tokens
        )


@pytest.mark.parametrize(
    "window_size",
    SUPPORTED_WINDOW_SIZES,
)
def test_preprocessing_matches_complete_ordered_training_pair_fixture(
    window_size: int,
) -> None:
    preprocessing = get_word2vec_preprocessing()
    all_pair_fixtures = cast(
        dict[str, Any],
        REFERENCE_FIXTURE["training_pairs"],
    )
    pair_fixture = cast(
        dict[str, Any],
        all_pair_fixtures[str(window_size)],
    )
    pairs = preprocessing.training_pairs[window_size]
    pair_records = [(pair.target, pair.context) for pair in pairs]

    assert len(pairs) == pair_fixture["count"]
    assert _training_pair_sha256(pairs) == pair_fixture["sha256_big_endian_uint16_pairs"]
    assert [list(pair) for pair in pair_records[:12]] == pair_fixture["first_12"]
    assert [list(pair) for pair in pair_records[-12:]] == pair_fixture["last_12"]


def test_preprocessing_exposes_exact_supported_window_sizes() -> None:
    preprocessing = get_word2vec_preprocessing()

    assert tuple(preprocessing.training_pairs) == SUPPORTED_WINDOW_SIZES
    assert [
        len(preprocessing.training_pairs[window_size]) for window_size in SUPPORTED_WINDOW_SIZES
    ] == [2092, 3970, 5634, 7084, 8320]


def test_preprocessing_rejects_mutation_at_every_public_level() -> None:
    preprocessing = get_word2vec_preprocessing()

    with pytest.raises(FrozenInstanceError):
        cast(Any, preprocessing).training_text = "changed"

    with pytest.raises(TypeError):
        cast(Any, preprocessing.corpus)[0] = "changed"

    with pytest.raises(TypeError):
        cast(Any, preprocessing.merges)[0] = preprocessing.merges[-1]

    with pytest.raises(FrozenInstanceError):
        cast(Any, preprocessing.merges[0]).frequency = -1

    with pytest.raises(TypeError):
        cast(Any, preprocessing.tokenized_sentences)[0] = ("changed",)

    with pytest.raises(TypeError):
        cast(Any, preprocessing.tokenized_sentences[0])[0] = "changed"

    with pytest.raises(TypeError):
        cast(Any, preprocessing.token_frequencies)["changed"] = 1

    with pytest.raises(TypeError):
        cast(Any, preprocessing.vocabulary)[0] = "changed"

    with pytest.raises(TypeError):
        cast(Any, preprocessing.token_indices)["changed"] = 0

    with pytest.raises(TypeError):
        cast(Any, preprocessing.training_pairs)[1] = ()

    with pytest.raises(TypeError):
        cast(Any, preprocessing.training_pairs[1])[0] = TrainingPair(
            target=0,
            context=0,
        )

    with pytest.raises(FrozenInstanceError):
        cast(Any, preprocessing.training_pairs[1][0]).target = -1

    assert get_word2vec_preprocessing() == preprocessing


def test_preprocessing_is_equivalent_across_repeated_and_concurrent_calls() -> None:
    expected = get_word2vec_preprocessing()

    assert get_word2vec_preprocessing() == expected

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(
            executor.map(
                lambda _call: get_word2vec_preprocessing(),
                range(16),
            )
        )

    assert all(result == expected for result in results)


def test_preprocessing_boundary_has_no_query_word_input() -> None:
    expected = get_word2vec_preprocessing()
    query_words = ["cat"]

    assert inspect.signature(get_word2vec_preprocessing).parameters == {}
    assert not hasattr(
        expected,
        "query_words",
    )

    query_words.append("dog")

    assert get_word2vec_preprocessing() == expected

    with pytest.raises(TypeError):
        cast(
            Any,
            get_word2vec_preprocessing,
        )(query_words)
