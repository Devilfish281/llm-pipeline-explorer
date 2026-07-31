# src/how_llms_work/ml/word2vec.py
"""Reference-compatible immutable Word2Vec preprocessing and training."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
from how_llms_work.ml.bpe import Merge, apply_merges, count_words, train_bpe
from how_llms_work.ml.math_utils import Mulberry32, round_typescript_decimal

BPE_MERGE_LIMIT: Final = 500
REFERENCE_MERGE_COUNT: Final = 423
SUPPORTED_WINDOW_SIZES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

Float64Array: TypeAlias = npt.NDArray[np.float64]

WORD2VEC_RANDOM_SEED: Final = 42

NEGATIVE_SAMPLING_POWER: Final = 0.75
LEARNING_RATE_START: Final = 0.025
LEARNING_RATE_END: Final = 0.001
LOSS_EPSILON: Final = 1e-10
SIGMOID_CLIP: Final = 6.0


PUBLIC_VECTOR_DIGITS: Final = 6
PUBLIC_SCORE_DIGITS: Final = 2
NEAREST_NEIGHBOR_LIMIT: Final = 5

ANALOGY_DEFINITIONS: Final[tuple[tuple[str, str, str], ...]] = (
    ("king", "man", "woman"),
    ("queen", "woman", "man"),
    ("prince", "boy", "girl"),
    ("kitten", "cat", "dog"),
    ("puppy", "dog", "cat"),
    ("he", "man", "woman"),
    ("his", "man", "woman"),
)


EMBEDDING_TRAINING_CORPUS: Final[tuple[str, ...]] = (
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat is a small pet",
    "a dog is a loyal pet",
    "the cat chased the mouse",
    "the dog chased the cat",
    "the kitten is a baby cat",
    "the puppy is a baby dog",
    "the cat and the dog are pets",
    "a kitten is small and cute",
    "a puppy is small and playful",
    "the cat sleeps on the bed",
    "the dog sleeps on the floor",
    "she loves her pet cat",
    "he loves his pet dog",
    "the cat drinks milk",
    "the dog eats meat",
    "cats and dogs are popular pets",
    "the happy cat purred loudly",
    "the happy dog wagged its tail",
    "the lion is a wild animal",
    "the tiger is a wild animal",
    "the elephant is a big animal",
    "the mouse is a tiny animal",
    "lions and tigers are big cats",
    "the bear lives in the forest",
    "the wolf lives in the forest",
    "the eagle flies in the sky",
    "the fish swims in the water",
    "birds fly and fish swim",
    "i ate pizza for dinner",
    "she ate pasta for lunch",
    "he ate sushi for dinner",
    "pizza and pasta are italian food",
    "sushi is japanese food",
    "bread and cheese make a sandwich",
    "fruit and vegetables are healthy food",
    "cake and cookies are sweet food",
    "rice is a common food",
    "i love pizza and pasta",
    "she cooked dinner for the family",
    "he made lunch at home",
    "the chef cooked a delicious meal",
    "coffee and tea are hot drinks",
    "juice and water are cold drinks",
    "the king is a man who rules",
    "the queen is a woman who rules",
    "the king sits on the throne",
    "the queen sits on the throne",
    "the prince is the son of the king",
    "the princess is the daughter of the queen",
    "the king and queen rule the kingdom",
    "a prince is a young man of royal blood",
    "a princess is a young woman of royal blood",
    "the king wore a golden crown",
    "the queen wore a silver crown",
    "the prince will become king",
    "the princess will become queen",
    "the man became king of the land",
    "the woman became queen of the land",
    "the king is a powerful man",
    "the queen is a powerful woman",
    "the man was crowned king",
    "the woman was crowned queen",
    "the prince is a boy of noble birth",
    "the princess is a girl of noble birth",
    "the prince is a royal boy",
    "the princess is a royal girl",
    "the young prince played in the castle",
    "the young princess played in the castle",
    "the doctor works at the hospital",
    "the nurse works at the hospital",
    "the teacher works at the school",
    "the student learns at the school",
    "the chef works in the kitchen",
    "the doctor heals the sick",
    "the teacher helps the student learn",
    "the nurse helps the doctor",
    "the scientist works in the lab",
    "the engineer builds machines",
    "the elephant is very big",
    "the mouse is very small",
    "the cat is small and quick",
    "the dog is big and strong",
    "the lion is big and fierce",
    "the kitten is tiny and cute",
    "the bear is large and powerful",
    "the boy and the girl play together",
    "the man and the woman walk together",
    "a boy is a young man",
    "a girl is a young woman",
    "the boy will grow into a man",
    "the girl will grow into a woman",
    "the boy runs fast",
    "the girl runs fast",
    "the man is tall and strong",
    "the woman is tall and smart",
    "he is a kind man",
    "she is a kind woman",
    "the boy helped his father",
    "the girl helped her mother",
    "the sun shines in the sky",
    "the moon glows at night",
    "stars shine in the dark sky",
    "the river flows to the sea",
    "rain falls from the sky",
    "the tree grows in the forest",
)

BPE_TRAINING_TEXT: Final = " ".join(EMBEDDING_TRAINING_CORPUS).lower()


@dataclass(frozen=True, slots=True)
class TrainingPair:
    """One ordered target-and-context Vocabulary index relationship."""

    target: int
    context: int


class SavedEmbeddingMerge(TypedDict):
    """One JSON-compatible Merge record in a Saved Embedding Model."""

    pair: list[str]
    merged: str


class SavedEmbeddingModel(TypedDict):
    """Complete JSON-compatible Word2Vec model accepted by persistence."""

    type: Literal["word2vec-skipgram"]
    dimensions: int
    vocab: list[str]
    merges: list[SavedEmbeddingMerge]
    embeddings: dict[str, list[float]]


class WordEmbedding(TypedDict):
    """One frontend-facing Query Word and its public input-weight vector."""

    word: str
    vector: list[float]


class NearestNeighborCandidate(TypedDict):
    """One Vocabulary candidate and its rounded cosine score."""

    word: str
    score: float


class NearestNeighborGroup(TypedDict):
    """The stable top candidates for one recognized Query Word position."""

    word: str
    nearest: list[NearestNeighborCandidate]


class SimilarityPair(TypedDict):
    """One positional pair of recognized Query Words."""

    a: str
    b: str
    score: float


class VectorAnalogy(TypedDict):
    """One predefined vector relationship and its selected Vocabulary Token."""

    query: str
    result: str
    score: float


class EmbeddingResult(TypedDict):
    """Exact frontend-facing completion payload for Word2Vec training."""

    embeddings: list[WordEmbedding]
    neighbors: list[NearestNeighborGroup]
    similarities: list[SimilarityPair]
    analogies: list[VectorAnalogy]
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class _RecognizedQueryWord:
    """One recognized Query Word position and its Vocabulary index."""

    submitted: str
    index: int


@dataclass(frozen=True, slots=True)
class Word2VecPreprocessing:
    """Immutable corpus-derived data shared by Embedding Training Runs."""

    corpus: tuple[str, ...]
    training_text: str
    merge_limit: int
    merges: tuple[Merge, ...]
    tokenized_sentences: tuple[tuple[str, ...], ...]
    token_frequencies: Mapping[str, int]
    vocabulary: tuple[str, ...]
    token_indices: Mapping[str, int]
    training_pairs: Mapping[int, tuple[TrainingPair, ...]]


def _count_token_frequencies(
    tokenized_sentences: Sequence[Sequence[str]],
) -> dict[str, int]:
    """Count Tokens in first-encounter order."""
    frequencies: dict[str, int] = {}

    for sentence in tokenized_sentences:
        for token in sentence:
            frequencies[token] = frequencies.get(token, 0) + 1

    return frequencies


def _build_training_pairs(
    tokenized_sentences: Sequence[Sequence[str]],
    token_indices: Mapping[str, int],
    window_size: int,
) -> tuple[TrainingPair, ...]:
    """Build ordered Training Pairs without crossing sentence boundaries."""
    pairs: list[TrainingPair] = []

    for sentence in tokenized_sentences:
        indices = tuple(token_indices[token] for token in sentence)

        for target_position, target_index in enumerate(indices):
            context_start = max(
                0,
                target_position - window_size,
            )
            context_stop = min(
                len(indices) - 1,
                target_position + window_size,
            )

            for context_position in range(
                context_start,
                context_stop + 1,
            ):
                if context_position == target_position:
                    continue

                pairs.append(
                    TrainingPair(
                        target=target_index,
                        context=indices[context_position],
                    )
                )

    return tuple(pairs)


def _build_preprocessing() -> Word2VecPreprocessing:
    """Construct deterministic corpus preprocessing for Word2Vec training."""
    merges = train_bpe(
        count_words(BPE_TRAINING_TEXT),
        max_merges=BPE_MERGE_LIMIT,
    )

    if len(merges) != REFERENCE_MERGE_COUNT:
        raise RuntimeError(
            "Reference corpus must produce "
            f"{REFERENCE_MERGE_COUNT} BPE merges; "
            f"received {len(merges)}"
        )

    tokenized_sentences = tuple(
        tuple(
            apply_merges(
                sentence.lower(),
                merges,
            )
        )
        for sentence in EMBEDDING_TRAINING_CORPUS
    )

    mutable_frequencies = _count_token_frequencies(
        tokenized_sentences,
    )

    vocabulary = tuple(
        token
        for token, _frequency in sorted(
            mutable_frequencies.items(),
            key=lambda item: -item[1],
        )
    )

    mutable_indices = {token: index for index, token in enumerate(vocabulary)}

    mutable_training_pairs = {
        window_size: _build_training_pairs(
            tokenized_sentences,
            mutable_indices,
            window_size,
        )
        for window_size in SUPPORTED_WINDOW_SIZES
    }

    return Word2VecPreprocessing(
        corpus=EMBEDDING_TRAINING_CORPUS,
        training_text=BPE_TRAINING_TEXT,
        merge_limit=BPE_MERGE_LIMIT,
        merges=merges,
        tokenized_sentences=tokenized_sentences,
        token_frequencies=MappingProxyType(
            mutable_frequencies,
        ),
        vocabulary=vocabulary,
        token_indices=MappingProxyType(
            mutable_indices,
        ),
        training_pairs=MappingProxyType(
            mutable_training_pairs,
        ),
    )


_WORD2VEC_PREPROCESSING: Final = _build_preprocessing()


def get_word2vec_preprocessing() -> Word2VecPreprocessing:
    """Return the immutable reference-compatible Word2Vec preprocessing data."""
    return _WORD2VEC_PREPROCESSING


@dataclass(
    frozen=True,
    slots=True,
)
class EmbeddingEpochUpdate:
    """One public six-decimal Embedding Epoch Update."""

    epoch: int
    loss: float

    def to_payload(
        self,
    ) -> dict[str, int | float]:
        """Return the frontend-compatible epoch payload."""
        return {
            "epoch": self.epoch,
            "loss": self.loss,
        }


@dataclass(
    frozen=True,
    slots=True,
)
class EmbeddingTrainingTransition:
    """One observable positive or negative numerical transition."""

    score: float
    gradient: float
    loss: float


@dataclass(slots=True)
class CompletedEmbeddingTraining:
    """Finite numerical state after the final Embedding Epoch Update."""

    dimensions: int
    window_size: int
    epochs: int
    negative_samples: int
    vocabulary: tuple[str, ...]
    training_pairs: list[TrainingPair]
    input_weights: Float64Array
    output_weights: Float64Array
    final_loss: float


EmbeddingTrainingEvent: TypeAlias = EmbeddingEpochUpdate | CompletedEmbeddingTraining


def embedding_sigmoid(
    value: float,
) -> float:
    """Apply the clipped sigmoid used by reference Skip-gram Training."""
    if value > SIGMOID_CLIP:
        return 1.0

    if value < -SIGMOID_CLIP:
        return 0.0

    return 1.0 / (1.0 + math.exp(-value))


def round_embedding_loss(
    value: float,
    digits: int = PUBLIC_VECTOR_DIGITS,
) -> float:
    """Round public Embedding loss with TypeScript-compatible semantics."""
    return round_typescript_decimal(
        value,
        digits,
    )


class EmbeddingTrainingRun(Iterator[EmbeddingTrainingEvent]):
    """Independent bounded deterministic Skip-gram Training Run."""

    def __init__(
        self,
        preprocessing: Word2VecPreprocessing,
        *,
        dimensions: int,
        window_size: int,
        epochs: int,
        negative_samples: int,
    ) -> None:
        self._validate_configuration(
            preprocessing=preprocessing,
            dimensions=dimensions,
            window_size=window_size,
            epochs=epochs,
            negative_samples=negative_samples,
        )

        self.preprocessing = preprocessing
        self.dimensions = dimensions
        self.window_size = window_size
        self.epochs = epochs
        self.negative_samples = negative_samples

        self.random_generator = Mulberry32(WORD2VEC_RANDOM_SEED)

        self.training_pairs = list(preprocessing.training_pairs[window_size])

        self.cumulative_distribution = self._build_cumulative_distribution()

        (
            self.input_weights,
            self.output_weights,
        ) = self._initialize_weights()

        self.report_step = max(
            1,
            epochs // 50,
        )

        self._next_epoch = 0
        self._last_loss: float | None = None
        self._completion_emitted = False
        self._failed = False

    def __iter__(
        self,
    ) -> EmbeddingTrainingRun:
        """Return this Embedding Training Run iterator."""
        return self

    def __next__(
        self,
    ) -> EmbeddingTrainingEvent:
        """Advance to the next report boundary or terminal state."""
        if self._failed:
            raise StopIteration

        while self._next_epoch <= self.epochs:
            epoch = self._next_epoch

            try:
                loss = self._train_epoch(epoch)
            except ArithmeticError:
                self._failed = True
                raise

            self._last_loss = loss
            self._next_epoch += 1

            if epoch % self.report_step == 0 or epoch == self.epochs:
                return EmbeddingEpochUpdate(
                    epoch=epoch,
                    loss=round_embedding_loss(loss),
                )

        if not self._completion_emitted:
            try:
                completion = self._build_completion()
            except ArithmeticError:
                self._failed = True
                raise

            self._completion_emitted = True

            return completion

        raise StopIteration

    def learning_rate_for_epoch(
        self,
        epoch: int,
    ) -> float:
        """Return the linear learning rate for one inclusive epoch."""
        if epoch < 0 or epoch > self.epochs:
            raise ValueError(f"epoch must be between 0 and {self.epochs}; received {epoch}")

        return LEARNING_RATE_START - (LEARNING_RATE_START - LEARNING_RATE_END) * (
            epoch / self.epochs
        )

    def shuffle_training_pairs(
        self,
    ) -> None:
        """Shuffle the run-owned Training Pairs using Fisher-Yates."""
        for index in range(
            len(self.training_pairs) - 1,
            0,
            -1,
        ):
            swap_index = math.floor(self.random_generator.random() * (index + 1))

            (
                self.training_pairs[index],
                self.training_pairs[swap_index],
            ) = (
                self.training_pairs[swap_index],
                self.training_pairs[index],
            )

    def sample_negative(
        self,
    ) -> int:
        """Draw one index from the cumulative unigram distribution."""
        draw = self.random_generator.random()

        lower = 0
        upper = len(self.cumulative_distribution) - 1

        while lower < upper:
            middle = (lower + upper) >> 1

            if self.cumulative_distribution[middle] < draw:
                lower = middle + 1
            else:
                upper = middle

        return lower

    def apply_positive_update(
        self,
        target: int,
        context: int,
        learning_rate: float,
    ) -> EmbeddingTrainingTransition:
        """Apply one immediate positive target-context update."""
        dot_product = self._dot_product(
            target_index=target,
            output_index=context,
        )

        score = embedding_sigmoid(dot_product)

        gradient = learning_rate * (1.0 - score)

        self._require_finite(
            score,
            "positive score",
        )

        self._require_finite(
            gradient,
            "positive gradient",
        )

        for dimension in range(self.dimensions):
            input_value = float(
                self.input_weights[
                    target,
                    dimension,
                ]
            )

            output_value = float(
                self.output_weights[
                    context,
                    dimension,
                ]
            )

            updated_input = input_value + gradient * output_value

            # The output coordinate uses input_value captured before
            # the input coordinate is changed.
            updated_output = output_value + gradient * input_value

            self._require_finite(
                updated_input,
                "positive input weight",
            )

            self._require_finite(
                updated_output,
                "positive output weight",
            )

            self.input_weights[
                target,
                dimension,
            ] = updated_input

            self.output_weights[
                context,
                dimension,
            ] = updated_output

        loss = -math.log(score + LOSS_EPSILON)

        self._require_finite(
            loss,
            "positive loss",
        )

        return EmbeddingTrainingTransition(
            score=score,
            gradient=gradient,
            loss=loss,
        )

    def apply_negative_update(
        self,
        target: int,
        negative: int,
        learning_rate: float,
    ) -> EmbeddingTrainingTransition:
        """Apply one immediate negative target-candidate update."""
        dot_product = self._dot_product(
            target_index=target,
            output_index=negative,
        )

        score = embedding_sigmoid(dot_product)

        gradient = learning_rate * score

        self._require_finite(
            score,
            "negative score",
        )

        self._require_finite(
            gradient,
            "negative gradient",
        )

        for dimension in range(self.dimensions):
            input_value = float(
                self.input_weights[
                    target,
                    dimension,
                ]
            )

            output_value = float(
                self.output_weights[
                    negative,
                    dimension,
                ]
            )

            updated_input = input_value - gradient * output_value

            # The output coordinate uses input_value captured before
            # the input coordinate is changed.
            updated_output = output_value - gradient * input_value

            self._require_finite(
                updated_input,
                "negative input weight",
            )

            self._require_finite(
                updated_output,
                "negative output weight",
            )

            self.input_weights[
                target,
                dimension,
            ] = updated_input

            self.output_weights[
                negative,
                dimension,
            ] = updated_output

        loss = -math.log(1.0 - score + LOSS_EPSILON)

        self._require_finite(
            loss,
            "negative loss",
        )

        return EmbeddingTrainingTransition(
            score=score,
            gradient=gradient,
            loss=loss,
        )

    @staticmethod
    def _validate_configuration(
        *,
        preprocessing: Word2VecPreprocessing,
        dimensions: int,
        window_size: int,
        epochs: int,
        negative_samples: int,
    ) -> None:
        if (
            not isinstance(
                dimensions,
                int,
            )
            or isinstance(
                dimensions,
                bool,
            )
            or dimensions <= 0
        ):
            raise ValueError("dimensions must be a positive integer")

        if (
            not isinstance(
                window_size,
                int,
            )
            or isinstance(
                window_size,
                bool,
            )
            or window_size not in preprocessing.training_pairs
        ):
            raise ValueError(f"window_size {window_size} is not available in preprocessing")

        if (
            not isinstance(
                epochs,
                int,
            )
            or isinstance(
                epochs,
                bool,
            )
            or epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer")

        if (
            not isinstance(
                negative_samples,
                int,
            )
            or isinstance(
                negative_samples,
                bool,
            )
            or negative_samples < 0
        ):
            raise ValueError("negative_samples must be a non-negative integer")

        if not preprocessing.vocabulary:
            raise ValueError("preprocessing vocabulary must not be empty")

        if not preprocessing.training_pairs[window_size]:
            raise ValueError("selected Training Pair sequence must not be empty")

        vocabulary_size = len(preprocessing.vocabulary)

        for token in preprocessing.vocabulary:
            frequency = preprocessing.token_frequencies.get(token)

            if frequency is None or frequency <= 0:
                raise ValueError("every Vocabulary Token must have a positive frequency")

        for pair in preprocessing.training_pairs[window_size]:
            if not (0 <= pair.target < vocabulary_size):
                raise ValueError("Training Pair target index is out of range")

            if not (0 <= pair.context < vocabulary_size):
                raise ValueError("Training Pair context index is out of range")

    def _build_cumulative_distribution(
        self,
    ) -> Float64Array:
        vocabulary_size = len(self.preprocessing.vocabulary)

        powers = np.empty(
            vocabulary_size,
            dtype=np.float64,
        )

        power_sum = 0.0

        for index, token in enumerate(self.preprocessing.vocabulary):
            frequency = self.preprocessing.token_frequencies[token]

            power = float(frequency) ** NEGATIVE_SAMPLING_POWER

            self._require_finite(
                power,
                "negative-sampling power",
            )

            powers[index] = power
            power_sum += power

            self._require_finite(
                power_sum,
                "negative-sampling sum",
            )

        if power_sum <= 0.0:
            raise ValueError("negative-sampling sum must be positive")

        cumulative = np.empty(
            vocabulary_size,
            dtype=np.float64,
        )

        running_total = 0.0

        for index in range(vocabulary_size):
            running_total += float(powers[index]) / power_sum

            self._require_finite(
                running_total,
                "negative-sampling cumulative probability",
            )

            cumulative[index] = running_total

        return cumulative

    def _initialize_weights(
        self,
    ) -> tuple[
        Float64Array,
        Float64Array,
    ]:
        vocabulary_size = len(self.preprocessing.vocabulary)

        scale = 0.5 / self.dimensions

        input_weights = np.empty(
            (
                vocabulary_size,
                self.dimensions,
            ),
            dtype=np.float64,
        )

        output_weights = np.empty(
            (
                vocabulary_size,
                self.dimensions,
            ),
            dtype=np.float64,
        )

        for token_index in range(vocabulary_size):
            for dimension in range(self.dimensions):
                # Preserve the reference interleaved random-call order:
                # one input draw, then one output draw, per coordinate.
                input_value = (self.random_generator.random() - 0.5) * scale

                output_value = (self.random_generator.random() - 0.5) * scale

                self._require_finite(
                    input_value,
                    "initial input weight",
                )

                self._require_finite(
                    output_value,
                    "initial output weight",
                )

                input_weights[
                    token_index,
                    dimension,
                ] = input_value

                output_weights[
                    token_index,
                    dimension,
                ] = output_value

        return (
            input_weights,
            output_weights,
        )

    def _dot_product(
        self,
        *,
        target_index: int,
        output_index: int,
    ) -> float:
        dot_product = 0.0

        for dimension in range(self.dimensions):
            dot_product += float(
                self.input_weights[
                    target_index,
                    dimension,
                ]
            ) * float(
                self.output_weights[
                    output_index,
                    dimension,
                ]
            )

            self._require_finite(
                dot_product,
                "dot product",
            )

        return dot_product

    def _train_epoch(
        self,
        epoch: int,
    ) -> float:
        learning_rate = self.learning_rate_for_epoch(epoch)

        self._require_finite(
            learning_rate,
            "learning rate",
        )

        self.shuffle_training_pairs()

        total_loss = 0.0

        for pair in self.training_pairs:
            # The positive update must finish before the first
            # negative random candidate is drawn.
            positive = self.apply_positive_update(
                target=pair.target,
                context=pair.context,
                learning_rate=learning_rate,
            )

            total_loss += positive.loss

            self._require_finite(
                total_loss,
                "epoch loss",
            )

            for _ in range(self.negative_samples):
                negative_index = self.sample_negative()

                # Consume the draw but do not replace a collision.
                if negative_index == pair.context:
                    continue

                negative = self.apply_negative_update(
                    target=pair.target,
                    negative=negative_index,
                    learning_rate=learning_rate,
                )

                total_loss += negative.loss

                self._require_finite(
                    total_loss,
                    "epoch loss",
                )

        # Match the reference denominator: positive Training Pair count.
        loss = total_loss / len(self.training_pairs)

        self._require_finite(
            loss,
            "normalized epoch loss",
        )

        return loss

    def _build_completion(
        self,
    ) -> CompletedEmbeddingTraining:
        if self._last_loss is None:
            raise RuntimeError("Embedding Training Run has not trained any epochs")

        if not np.isfinite(self.input_weights).all():
            raise FloatingPointError("input weights contain a non-finite value")

        if not np.isfinite(self.output_weights).all():
            raise FloatingPointError("output weights contain a non-finite value")

        self._require_finite(
            self._last_loss,
            "final loss",
        )

        return CompletedEmbeddingTraining(
            dimensions=self.dimensions,
            window_size=self.window_size,
            epochs=self.epochs,
            negative_samples=self.negative_samples,
            vocabulary=(self.preprocessing.vocabulary),
            training_pairs=list(self.training_pairs),
            input_weights=(self.input_weights.copy()),
            output_weights=(self.output_weights.copy()),
            final_loss=self._last_loss,
        )

    @staticmethod
    def _require_finite(
        value: float,
        name: str,
    ) -> None:
        if not math.isfinite(value):
            raise FloatingPointError(f"{name} is not finite")


def create_embedding_training_run(
    *,
    dimensions: int,
    window_size: int,
    epochs: int,
    negative_samples: int,
) -> EmbeddingTrainingRun:
    """Create one canonical deterministic Embedding Training Run."""
    return EmbeddingTrainingRun(
        preprocessing=(get_word2vec_preprocessing()),
        dimensions=dimensions,
        window_size=window_size,
        epochs=epochs,
        negative_samples=negative_samples,
    )


def _require_finite_result_value(
    value: float,
    name: str,
) -> None:
    """Reject a non-finite value before it reaches a public Word2Vec object."""
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is not finite")


def _validate_result_inputs(
    completion: CompletedEmbeddingTraining,
    preprocessing: Word2VecPreprocessing,
) -> None:
    """Validate the structural assumptions required for public conversion."""
    if (
        not isinstance(completion.dimensions, int)
        or isinstance(completion.dimensions, bool)
        or completion.dimensions <= 0
    ):
        raise ValueError("completion dimensions must be a positive integer")

    if completion.vocabulary != preprocessing.vocabulary:
        raise ValueError("completion Vocabulary must match preprocessing Vocabulary in exact order")

    vocabulary_size = len(preprocessing.vocabulary)

    if vocabulary_size == 0:
        raise ValueError("Vocabulary must not be empty")

    if len(set(preprocessing.vocabulary)) != vocabulary_size:
        raise ValueError("Vocabulary Tokens must be unique")

    expected_indices = {token: index for index, token in enumerate(preprocessing.vocabulary)}

    if dict(preprocessing.token_indices) != expected_indices:
        raise ValueError("preprocessing token indices do not match Vocabulary order")

    expected_shape = (
        vocabulary_size,
        completion.dimensions,
    )

    if completion.input_weights.shape != expected_shape:
        raise ValueError(
            "input weight shape must equal "
            f"{expected_shape}; received {completion.input_weights.shape}"
        )


def _build_public_vectors(
    completion: CompletedEmbeddingTraining,
    preprocessing: Word2VecPreprocessing,
) -> list[list[float]]:
    """Build fresh six-decimal public vectors from input weights only."""
    _validate_result_inputs(
        completion,
        preprocessing,
    )

    public_vectors: list[list[float]] = []

    for token_index, token in enumerate(preprocessing.vocabulary):
        public_vector: list[float] = []

        for dimension in range(completion.dimensions):
            coordinate = float(
                completion.input_weights[
                    token_index,
                    dimension,
                ]
            )

            _require_finite_result_value(
                coordinate,
                f'input coordinate for "{token}"',
            )

            public_coordinate = round_typescript_decimal(
                coordinate,
                PUBLIC_VECTOR_DIGITS,
            )

            _require_finite_result_value(
                public_coordinate,
                f'public coordinate for "{token}"',
            )

            public_vector.append(public_coordinate)

        public_vectors.append(public_vector)

    return public_vectors


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate one finite cosine score through ordered scalar accumulation."""
    if len(left) != len(right):
        raise ValueError("cosine vectors must have the same dimensions")

    if not left:
        raise ValueError("cosine vectors must not be empty")

    dot_product = 0.0
    left_magnitude_squared = 0.0
    right_magnitude_squared = 0.0

    for left_coordinate, right_coordinate in zip(
        left,
        right,
        strict=True,
    ):
        _require_finite_result_value(
            left_coordinate,
            "left cosine coordinate",
        )
        _require_finite_result_value(
            right_coordinate,
            "right cosine coordinate",
        )

        dot_product += left_coordinate * right_coordinate
        left_magnitude_squared += left_coordinate * left_coordinate
        right_magnitude_squared += right_coordinate * right_coordinate

        _require_finite_result_value(
            dot_product,
            "cosine dot product",
        )
        _require_finite_result_value(
            left_magnitude_squared,
            "left cosine magnitude",
        )
        _require_finite_result_value(
            right_magnitude_squared,
            "right cosine magnitude",
        )

    if left_magnitude_squared <= 0.0 or right_magnitude_squared <= 0.0:
        raise FloatingPointError("cosine vector magnitude must be positive")

    denominator = math.sqrt(left_magnitude_squared) * math.sqrt(right_magnitude_squared)

    _require_finite_result_value(
        denominator,
        "cosine denominator",
    )

    if denominator <= 0.0:
        raise FloatingPointError("cosine denominator must be positive")

    score = dot_product / denominator

    _require_finite_result_value(
        score,
        "cosine score",
    )

    return score


def _resolve_query_words(
    query_words: Sequence[str],
    preprocessing: Word2VecPreprocessing,
) -> tuple[list[_RecognizedQueryWord], list[str]]:
    """Resolve Query Word positions without trimming, splitting, or deduplicating."""
    recognized: list[_RecognizedQueryWord] = []
    warnings: list[str] = []

    for submitted_word in query_words:
        if not isinstance(submitted_word, str):
            raise TypeError("every Query Word must be a string")

        lookup_word = submitted_word.lower()
        vocabulary_index = preprocessing.token_indices.get(lookup_word)

        if vocabulary_index is None:
            bpe_tokens = apply_merges(
                lookup_word,
                preprocessing.merges,
            )
            warnings.append(
                f'"{submitted_word}" is not a single BPE token — '
                f"it splits into [{', '.join(bpe_tokens)}]"
            )
            continue

        recognized.append(
            _RecognizedQueryWord(
                submitted=submitted_word,
                index=vocabulary_index,
            )
        )

    return (
        recognized,
        warnings,
    )


def _build_selected_embeddings(
    recognized_words: Sequence[_RecognizedQueryWord],
    public_vectors: Sequence[Sequence[float]],
) -> list[WordEmbedding]:
    """Build selected Word Embeddings in recognized positional order."""
    return [
        {
            "word": recognized_word.submitted,
            "vector": list(public_vectors[recognized_word.index]),
        }
        for recognized_word in recognized_words
    ]


def _build_nearest_neighbors(
    recognized_words: Sequence[_RecognizedQueryWord],
    vocabulary: Sequence[str],
    public_vectors: Sequence[Sequence[float]],
) -> list[NearestNeighborGroup]:
    """Build stable top-five Nearest Neighbor groups from public vectors."""
    neighbor_groups: list[NearestNeighborGroup] = []

    for recognized_word in recognized_words:
        query_vector = public_vectors[recognized_word.index]
        candidates: list[NearestNeighborCandidate] = []

        for candidate_index, candidate_word in enumerate(vocabulary):
            if candidate_index == recognized_word.index:
                continue

            score = round_typescript_decimal(
                _cosine_similarity(
                    query_vector,
                    public_vectors[candidate_index],
                ),
                PUBLIC_SCORE_DIGITS,
            )

            candidates.append(
                {
                    "word": candidate_word,
                    "score": score,
                }
            )

        candidates.sort(
            key=lambda candidate: candidate["score"],
            reverse=True,
        )

        neighbor_groups.append(
            {
                "word": recognized_word.submitted,
                "nearest": [
                    {
                        "word": candidate["word"],
                        "score": candidate["score"],
                    }
                    for candidate in candidates[:NEAREST_NEIGHBOR_LIMIT]
                ],
            }
        )

    return neighbor_groups


def _build_similarity_pairs(
    recognized_words: Sequence[_RecognizedQueryWord],
    public_vectors: Sequence[Sequence[float]],
) -> list[SimilarityPair]:
    """Build every recognized positional Similarity Pair in nested-loop order."""
    similarities: list[SimilarityPair] = []

    for left_position, left_word in enumerate(recognized_words):
        for right_position in range(
            left_position + 1,
            len(recognized_words),
        ):
            right_word = recognized_words[right_position]
            score = round_typescript_decimal(
                _cosine_similarity(
                    public_vectors[left_word.index],
                    public_vectors[right_word.index],
                ),
                PUBLIC_SCORE_DIGITS,
            )

            similarities.append(
                {
                    "a": left_word.submitted,
                    "b": right_word.submitted,
                    "score": score,
                }
            )

    return similarities


def _build_vector_analogies(
    completion: CompletedEmbeddingTraining,
    preprocessing: Word2VecPreprocessing,
    public_vectors: Sequence[Sequence[float]],
) -> list[VectorAnalogy]:
    """Evaluate predefined raw-source and public-candidate Vector Analogies."""
    analogies: list[VectorAnalogy] = []

    for first_word, subtracted_word, added_word in ANALOGY_DEFINITIONS:
        first_index = preprocessing.token_indices.get(first_word)
        subtracted_index = preprocessing.token_indices.get(subtracted_word)
        added_index = preprocessing.token_indices.get(added_word)

        if first_index is None or subtracted_index is None or added_index is None:
            continue

        result_vector: list[float] = []

        for dimension in range(completion.dimensions):
            coordinate = (
                float(completion.input_weights[first_index, dimension])
                - float(completion.input_weights[subtracted_index, dimension])
                + float(completion.input_weights[added_index, dimension])
            )

            _require_finite_result_value(
                coordinate,
                "analogy query coordinate",
            )

            result_vector.append(coordinate)

        excluded_indices = {
            first_index,
            subtracted_index,
            added_index,
        }
        best_index: int | None = None
        best_score = -math.inf

        for candidate_index, _candidate_word in enumerate(preprocessing.vocabulary):
            if candidate_index in excluded_indices:
                continue

            score = _cosine_similarity(
                result_vector,
                public_vectors[candidate_index],
            )

            if score > best_score:
                best_score = score
                best_index = candidate_index

        if best_index is None:
            continue

        _require_finite_result_value(
            best_score,
            "selected analogy score",
        )

        analogies.append(
            {
                "query": f"{first_word} - {subtracted_word} + {added_word}",
                "result": preprocessing.vocabulary[best_index],
                "score": round_typescript_decimal(
                    best_score,
                    PUBLIC_SCORE_DIGITS,
                ),
            }
        )

    return analogies


def build_embedding_result(
    completion: CompletedEmbeddingTraining,
    preprocessing: Word2VecPreprocessing,
    query_words: Sequence[str],
) -> EmbeddingResult:
    """Convert one completed run into the exact frontend Embedding Result."""
    public_vectors = _build_public_vectors(
        completion,
        preprocessing,
    )
    recognized_words, warnings = _resolve_query_words(
        query_words,
        preprocessing,
    )

    return {
        "embeddings": _build_selected_embeddings(
            recognized_words,
            public_vectors,
        ),
        "neighbors": _build_nearest_neighbors(
            recognized_words,
            preprocessing.vocabulary,
            public_vectors,
        ),
        "similarities": _build_similarity_pairs(
            recognized_words,
            public_vectors,
        ),
        "analogies": _build_vector_analogies(
            completion,
            preprocessing,
            public_vectors,
        ),
        "warnings": list(warnings),
    }


def build_saved_embedding_model(
    completion: CompletedEmbeddingTraining,
    preprocessing: Word2VecPreprocessing,
) -> SavedEmbeddingModel:
    """Convert one completed run into the complete persistence-ready model."""
    public_vectors = _build_public_vectors(
        completion,
        preprocessing,
    )

    merges: list[SavedEmbeddingMerge] = [
        {
            "pair": list(merge.pair),
            "merged": merge.merged,
        }
        for merge in preprocessing.merges
    ]

    embeddings = {
        token: list(public_vectors[index]) for index, token in enumerate(preprocessing.vocabulary)
    }

    return {
        "type": "word2vec-skipgram",
        "dimensions": completion.dimensions,
        "vocab": list(preprocessing.vocabulary),
        "merges": merges,
        "embeddings": embeddings,
    }
