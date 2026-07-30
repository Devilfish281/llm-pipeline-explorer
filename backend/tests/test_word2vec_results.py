# backend/tests/test_word2vec_results.py
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest
from how_llms_work.ml.bpe import Merge
from how_llms_work.ml.word2vec import (
    CompletedEmbeddingTraining,
    EmbeddingResult,
    SavedEmbeddingModel,
    TrainingPair,
    Word2VecPreprocessing,
    build_embedding_result,
    build_saved_embedding_model,
    round_typescript_decimal,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "word2vec_results_reference.json"
REFERENCE_FIXTURE = cast(
    dict[str, Any],
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
)

REFERENCE_VOCABULARY = (
    " ",
    "king",
    "man",
    "woman",
    "queen",
    "prince",
    "boy",
    "girl",
    "kitten",
    "cat",
    "dog",
    "puppy",
    "he",
    "his",
    "royal",
    "princess",
    "she",
    "her",
    "wolf",
)

REFERENCE_INPUT_WEIGHTS = np.asarray(
    [
        [0.50000049, 0.4999995],
        [2.0000004, 0.0000004],
        [1.0000004, 0.0000004],
        [0.0000004, 1.0000004],
        [0.0000004, 2.0000004],
        [2.0000004, 0.4000004],
        [1.0000004, 0.2000004],
        [0.2000004, 1.0000004],
        [0.8000004, 0.2000004],
        [1.00000049, -0.00000049],
        [0.0000004, 1.0000004],
        [0.2000004, 0.8000004],
        [1.1000004, 0.0000004],
        [1.2000004, 0.0000004],
        [1.0000004, 1.0000004],
        [0.4000004, 2.0000004],
        [0.1000004, 1.1000004],
        [0.2000004, 1.2000004],
        [-1.0000004, -1.0000004],
    ],
    dtype=np.float64,
)

REFERENCE_MERGES = (
    Merge(pair=("c", "a"), merged="ca", frequency=10),
    Merge(pair=("ca", "t"), merged="cat", frequency=8),
    Merge(pair=("d", "o"), merged="do", frequency=7),
    Merge(pair=("do", "g"), merged="dog", frequency=6),
)


def _build_preprocessing(
    vocabulary: tuple[str, ...] = REFERENCE_VOCABULARY,
    merges: tuple[Merge, ...] = REFERENCE_MERGES,
) -> Word2VecPreprocessing:
    token_indices = MappingProxyType({token: index for index, token in enumerate(vocabulary)})
    token_frequencies = MappingProxyType({token: 1 for token in vocabulary})

    return Word2VecPreprocessing(
        corpus=("synthetic result fixture",),
        training_text="synthetic result fixture",
        merge_limit=len(merges),
        merges=merges,
        tokenized_sentences=(vocabulary,),
        token_frequencies=token_frequencies,
        vocabulary=vocabulary,
        token_indices=token_indices,
        training_pairs=MappingProxyType(
            {
                1: (TrainingPair(target=0, context=min(1, len(vocabulary) - 1)),),
            }
        ),
    )


def _build_completion(
    vocabulary: tuple[str, ...] = REFERENCE_VOCABULARY,
    input_weights: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
) -> CompletedEmbeddingTraining:
    selected_input_weights = (
        REFERENCE_INPUT_WEIGHTS.copy() if input_weights is None else input_weights.copy()
    )
    output_weights = (
        np.arange(selected_input_weights.size, dtype=np.float64).reshape(
            selected_input_weights.shape
        )
        + 900.0
    )

    return CompletedEmbeddingTraining(
        dimensions=selected_input_weights.shape[1],
        window_size=1,
        epochs=1,
        negative_samples=0,
        vocabulary=vocabulary,
        training_pairs=[
            TrainingPair(
                target=0,
                context=min(1, len(vocabulary) - 1),
            )
        ],
        input_weights=selected_input_weights,
        output_weights=output_weights,
        final_loss=0.5,
    )


def _assert_plain_json_value(value: object) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return

    if isinstance(value, list):
        for item in value:
            _assert_plain_json_value(item)
        return

    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)

        for item in value.values():
            _assert_plain_json_value(item)
        return

    raise AssertionError(f"Non-JSON-compatible value: {type(value)!r}")


def _contains_number(value: object, expected: float) -> bool:
    if isinstance(value, float | int):
        return float(value) == expected

    if isinstance(value, list):
        return any(_contains_number(item, expected) for item in value)

    if isinstance(value, dict):
        return any(_contains_number(item, expected) for item in value.values())

    return False


def test_typescript_decimal_rounding_handles_signed_and_half_boundaries() -> None:
    assert round_typescript_decimal(1.2345644, 6) == 1.234564
    assert round_typescript_decimal(1.2345645, 6) == 1.234565
    assert round_typescript_decimal(-1.2345645, 6) == -1.234564
    assert round_typescript_decimal(-1.5, 0) == -1.0
    assert round_typescript_decimal(-0.00000049, 6) == 0.0

    with pytest.raises(ValueError, match="digits must be non-negative"):
        round_typescript_decimal(1.0, -1)

    with pytest.raises(FloatingPointError, match="value is not finite"):
        round_typescript_decimal(float("nan"), 6)


def test_saved_embedding_model_matches_exact_complete_fixture() -> None:
    preprocessing = _build_preprocessing()
    completion = _build_completion()

    model = build_saved_embedding_model(
        completion,
        preprocessing,
    )

    assert model == REFERENCE_FIXTURE["saved_embedding_model"]
    assert list(model) == [
        "type",
        "dimensions",
        "vocab",
        "merges",
        "embeddings",
    ]
    assert list(model["embeddings"]) == list(preprocessing.vocabulary)
    assert all(set(merge) == {"pair", "merged"} for merge in model["merges"])
    assert all(len(vector) == completion.dimensions for vector in model["embeddings"].values())
    _assert_plain_json_value(model)

    for sentinel in completion.output_weights.flat:
        assert not _contains_number(model, float(sentinel))


def test_embedding_result_matches_exact_query_position_fixture() -> None:
    preprocessing = _build_preprocessing()
    completion = _build_completion()
    query_words = cast(list[str], REFERENCE_FIXTURE["query_words"])

    result = build_embedding_result(
        completion,
        preprocessing,
        query_words,
    )

    assert result == REFERENCE_FIXTURE["embedding_result"]
    assert list(result) == [
        "embeddings",
        "neighbors",
        "similarities",
        "analogies",
        "warnings",
    ]
    assert all(set(item) == {"word", "vector"} for item in result["embeddings"])
    assert all(set(item) == {"word", "nearest"} for item in result["neighbors"])
    assert all(
        set(candidate) == {"word", "score"}
        for group in result["neighbors"]
        for candidate in group["nearest"]
    )
    assert all(set(item) == {"a", "b", "score"} for item in result["similarities"])
    assert all(set(item) == {"query", "result", "score"} for item in result["analogies"])
    _assert_plain_json_value(result)

    for sentinel in completion.output_weights.flat:
        assert not _contains_number(result, float(sentinel))


def test_query_word_lookup_preserves_original_text_duplicates_and_warnings() -> None:
    result = build_embedding_result(
        _build_completion(),
        _build_preprocessing(),
        cast(list[str], REFERENCE_FIXTURE["query_words"]),
    )

    assert [item["word"] for item in result["embeddings"]] == [
        "KING",
        "cat",
        "cat",
        " ",
    ]
    assert [group["word"] for group in result["neighbors"]] == [
        "KING",
        "cat",
        "cat",
        " ",
    ]
    assert result["warnings"] == REFERENCE_FIXTURE["embedding_result"]["warnings"]
    assert len(result["warnings"]) == 5


def test_neighbors_round_before_stable_ranking_and_limit_to_five() -> None:
    result = build_embedding_result(
        _build_completion(),
        _build_preprocessing(),
        ["KING", " "],
    )

    assert result["neighbors"][0] == REFERENCE_FIXTURE["embedding_result"]["neighbors"][0]
    assert result["neighbors"][1] == REFERENCE_FIXTURE["embedding_result"]["neighbors"][3]
    assert all(len(group["nearest"]) == 5 for group in result["neighbors"])
    assert "king" not in {candidate["word"] for candidate in result["neighbors"][0]["nearest"]}


def test_similarity_pairs_use_recognized_positions_including_duplicates() -> None:
    result = build_embedding_result(
        _build_completion(),
        _build_preprocessing(),
        ["KING", "cat", "cat", "unknown", " "],
    )

    assert result["similarities"] == REFERENCE_FIXTURE["embedding_result"]["similarities"]
    assert result["similarities"][3] == {
        "a": "cat",
        "b": "cat",
        "score": 1.0,
    }


def test_all_unrecognized_words_keep_analogies_and_ordered_warnings() -> None:
    result = build_embedding_result(
        _build_completion(),
        _build_preprocessing(),
        ["unknown", "cat dog"],
    )

    assert result["embeddings"] == []
    assert result["neighbors"] == []
    assert result["similarities"] == []
    assert result["analogies"] == REFERENCE_FIXTURE["embedding_result"]["analogies"]
    assert result["warnings"] == REFERENCE_FIXTURE["embedding_result"]["warnings"][:2]


def test_vector_analogies_preserve_order_exclusions_and_first_candidate_ties() -> None:
    result = build_embedding_result(
        _build_completion(),
        _build_preprocessing(),
        [],
    )

    assert result["analogies"] == REFERENCE_FIXTURE["embedding_result"]["analogies"]
    assert [analogy["query"] for analogy in result["analogies"]] == [
        "king - man + woman",
        "queen - woman + man",
        "prince - boy + girl",
        "kitten - cat + dog",
        "puppy - dog + cat",
        "he - man + woman",
        "his - man + woman",
    ]

    for analogy in result["analogies"]:
        sources = set(analogy["query"].replace(" - ", " + ").split(" + "))
        assert analogy["result"] not in sources

    assert result["analogies"][0]["result"] == " "


def test_analogy_query_uses_raw_sources_and_public_candidate_vectors() -> None:
    vocabulary = (
        "king",
        "man",
        "woman",
        "candidate-first",
        "candidate-second",
    )
    preprocessing = _build_preprocessing(
        vocabulary=vocabulary,
        merges=(),
    )
    input_weights = np.asarray(
        [
            [2.0000004, 0.0000004],
            [1.0, 0.0],
            [0.0, 0.0000004],
            [1.0, -0.000001],
            [1.0, 0.000001],
        ],
        dtype=np.float64,
    )
    completion = _build_completion(
        vocabulary=vocabulary,
        input_weights=input_weights,
    )

    result = build_embedding_result(
        completion,
        preprocessing,
        [],
    )

    assert result["analogies"] == [
        {
            "query": "king - man + woman",
            "result": "candidate-second",
            "score": 1.0,
        }
    ]


@pytest.mark.parametrize(
    "non_finite_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=[
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_non_finite_input_coordinate_prevents_both_public_objects(
    non_finite_value: float,
) -> None:
    preprocessing = _build_preprocessing()
    completion = _build_completion()
    completion.input_weights[0, 0] = non_finite_value

    with pytest.raises(FloatingPointError):
        build_embedding_result(
            completion,
            preprocessing,
            ["king"],
        )

    with pytest.raises(FloatingPointError):
        build_saved_embedding_model(
            completion,
            preprocessing,
        )


def test_zero_public_vector_prevents_non_finite_cosine_result() -> None:
    preprocessing = _build_preprocessing()
    completion = _build_completion()
    completion.input_weights[
        preprocessing.token_indices["king"],
        :,
    ] = 0.0

    with pytest.raises(
        FloatingPointError,
        match="cosine vector magnitude must be positive",
    ):
        build_embedding_result(
            completion,
            preprocessing,
            ["king"],
        )


def test_malformed_completion_or_preprocessing_alignment_is_rejected() -> None:
    preprocessing = _build_preprocessing()

    wrong_shape = _build_completion(
        input_weights=REFERENCE_INPUT_WEIGHTS[:, :1],
    )
    wrong_shape.dimensions = 2

    with pytest.raises(ValueError, match="input weight shape"):
        build_saved_embedding_model(
            wrong_shape,
            preprocessing,
        )

    reversed_vocabulary = tuple(reversed(REFERENCE_VOCABULARY))
    misaligned_completion = _build_completion(
        vocabulary=reversed_vocabulary,
    )

    with pytest.raises(ValueError, match="must match preprocessing Vocabulary"):
        build_embedding_result(
            misaligned_completion,
            preprocessing,
            [],
        )


def test_result_and_model_construction_are_repeatable_mutation_isolated_and_concurrent() -> None:
    preprocessing = _build_preprocessing()
    completion = _build_completion()
    query_words = cast(list[str], REFERENCE_FIXTURE["query_words"])
    query_snapshot = list(query_words)
    input_snapshot = completion.input_weights.copy()
    output_snapshot = completion.output_weights.copy()

    first_result = build_embedding_result(
        completion,
        preprocessing,
        query_words,
    )
    first_model = build_saved_embedding_model(
        completion,
        preprocessing,
    )

    first_result["embeddings"][0]["vector"][0] = 999.0
    first_result["neighbors"][0]["nearest"].clear()
    first_result["warnings"].append("changed")
    first_model["vocab"][0] = "changed"
    first_model["merges"][0]["pair"][0] = "changed"
    first_model["embeddings"]["king"][0] = 999.0

    second_result = build_embedding_result(
        completion,
        preprocessing,
        query_words,
    )
    second_model = build_saved_embedding_model(
        completion,
        preprocessing,
    )

    assert second_result == REFERENCE_FIXTURE["embedding_result"]
    assert second_model == REFERENCE_FIXTURE["saved_embedding_model"]
    assert query_words == query_snapshot
    np.testing.assert_array_equal(
        completion.input_weights,
        input_snapshot,
    )
    np.testing.assert_array_equal(
        completion.output_weights,
        output_snapshot,
    )

    def construct() -> tuple[EmbeddingResult, SavedEmbeddingModel]:
        return (
            build_embedding_result(
                completion,
                preprocessing,
                query_words,
            ),
            build_saved_embedding_model(
                completion,
                preprocessing,
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        concurrent_outputs = list(
            executor.map(
                lambda _index: construct(),
                range(8),
            )
        )

    assert all(
        result == REFERENCE_FIXTURE["embedding_result"]
        and model == REFERENCE_FIXTURE["saved_embedding_model"]
        for result, model in concurrent_outputs
    )
    np.testing.assert_array_equal(
        completion.input_weights,
        input_snapshot,
    )
    np.testing.assert_array_equal(
        completion.output_weights,
        output_snapshot,
    )
