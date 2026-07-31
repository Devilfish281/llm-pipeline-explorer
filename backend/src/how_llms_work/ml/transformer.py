# src/how_llms_work/ml/transformer.py
"""Immutable Transformer preprocessing, layouts, initialization, and mathematics."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Real
from threading import Event, Lock
from types import MappingProxyType
from typing import Final, Literal, Mapping, TypeAlias, TypedDict, cast

import numpy as np
import numpy.typing as npt
from how_llms_work.ml.bpe import Merge, apply_merges, count_words, train_bpe
from how_llms_work.ml.math_utils import Mulberry32, round_typescript_decimal
from how_llms_work.ml.matrix import (
    matmul,
    matmul_transposed_left,
    matmul_transposed_right,
    scalar_multiply,
    stable_row_softmax,
)

__all__ = [
    "LOGICAL_TRAINING_SHARD_COUNT",
    "TRANSFORMER_BPE_MERGE_LIMIT",
    "TRANSFORMER_GENERATION_SEED_LENGTH",
    "TRANSFORMER_SEQUENCE_LENGTH",
    "TRANSFORMER_TRAINING_CORPUS",
    "LogicalTrainingShard",
    "TransformerPreprocessingSnapshot",
    "TransformerTrainingSequence",
    "build_logical_training_shards",
    "get_transformer_preprocessing",
    "TRANSFORMER_ATTENTION_HEAD_COUNT",
    "TRANSFORMER_ATTENTION_SCALE",
    "TRANSFORMER_CONTEXT_LENGTH",
    "TRANSFORMER_CROSS_ENTROPY_EPSILON",
    "TRANSFORMER_EMBEDDING_DIMENSION",
    "TRANSFORMER_FEED_FORWARD_DIMENSION",
    "TRANSFORMER_HEAD_DIMENSION",
    "TRANSFORMER_LAYER_NORMALIZATION_EPSILON",
    "TRANSFORMER_MAX_LAYER_COUNT",
    "TRANSFORMER_MIN_LAYER_COUNT",
    "InitializedTransformerParameters",
    "LogicalTrainingShardResult",
    "TransformerAttentionHeadCache",
    "TransformerBackwardResult",
    "TransformerBlockForwardCache",
    "TransformerBlockParameterViews",
    "TransformerForwardResult",
    "TransformerGradientBuffer",
    "TransformerLayerNormalizationCache",
    "TransformerParameterLayout",
    "TransformerParameterLayoutRecord",
    "TransformerParameterViews",
    "TransformerSequenceResult",
    "build_transformer_parameter_layout",
    "build_transformer_parameter_views",
    "calculate_logical_training_shard",
    "calculate_transformer_backward",
    "calculate_transformer_cross_entropy",
    "calculate_transformer_forward",
    "calculate_transformer_sequence",
    "create_transformer_gradient_buffer",
    "initialize_transformer_parameters",
    "transformer_parameter_count",
    "TransformerEpochObservation",
    "TransformerEpochUpdate",
    "TransformerTrainingRun",
    "build_transformer_report_epochs",
    "create_transformer_training_run",
    "GeneratedTextSample",
    "SavedTransformerPromptError",
    "EmptySavedTransformerPromptError",
    "UnsupportedSavedTransformerPromptError",
    "SavedTransformerPromptTooLongError",
    "PreparedSavedTransformerPrompt",
    "SavedTransformerBlockWeights",
    "SavedTransformerConfig",
    "SavedTransformerMerge",
    "SavedTransformerModel",
    "SavedTransformerWeights",
    "build_saved_transformer_model",
    "evaluate_transformer_final_loss",
    "generate_transformer_text",
    "generate_saved_transformer_text",
    "prepare_saved_transformer_prompt",
]


TRANSFORMER_BPE_MERGE_LIMIT: Final = 1_000
TRANSFORMER_SEQUENCE_LENGTH: Final = 16
TRANSFORMER_GENERATION_SEED_LENGTH: Final = 3
LOGICAL_TRAINING_SHARD_COUNT: Final = 4

TRANSFORMER_CONTEXT_LENGTH: Final = 32
TRANSFORMER_EMBEDDING_DIMENSION: Final = 32
TRANSFORMER_ATTENTION_HEAD_COUNT: Final = 2
TRANSFORMER_HEAD_DIMENSION: Final = 16
TRANSFORMER_FEED_FORWARD_DIMENSION: Final = 128
TRANSFORMER_MIN_LAYER_COUNT: Final = 1
TRANSFORMER_MAX_LAYER_COUNT: Final = 6

TRANSFORMER_ATTENTION_SCALE: Final = 0.25
TRANSFORMER_LAYER_NORMALIZATION_EPSILON: Final = 1e-5
TRANSFORMER_CROSS_ENTROPY_EPSILON: Final = 1e-10

_TRANSFORMER_ADAM_LEARNING_RATE: Final = 0.001
_TRANSFORMER_ADAM_BETA1: Final = 0.9
_TRANSFORMER_ADAM_BETA2: Final = 0.999
_TRANSFORMER_ADAM_EPSILON: Final = 1e-8

_TRANSFORMER_FLOAT_ITEMSIZE: Final[int] = int(np.dtype(np.float32).itemsize)
_TRANSFORMER_BLOCK_PARAMETER_KEYS: Final[tuple[str, ...]] = (
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

_Float32Array: TypeAlias = npt.NDArray[np.float32]
_Float64Array: TypeAlias = npt.NDArray[np.float64]
_ParameterShape: TypeAlias = tuple[int, ...]

# The existing Python BPE seam uses ASCII-style word pre-tokenization and does
# not accept a custom regex. Transformer preprocessing needs the TypeScript
# leading-space behavior: /\s\w+|[^\w\s]/g.
#
# Each leading space is temporarily represented as "_" while calling the
# existing public count_words(), train_bpe(), and apply_merges() operations.
# The fixed corpus contains no underscores, which makes this transformation
# reversible and keeps BPE behavior in the established shared BPE module.
_TRANSFORMER_SPACE_SENTINEL: Final = "_"
_TRANSFORMER_PRE_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\s[A-Za-z0-9_]+|[^A-Za-z0-9_\s]"
)

TRANSFORMER_TRAINING_CORPUS: Final[tuple[str, ...]] = (
    "once upon a time there was a king who ruled a great kingdom. the king was a tall and powerful man. he sat on a golden throne in a big castle. one day the king said i need a queen to rule with me. so he found a wise and kind woman and she became queen. the queen wore a silver crown and sat on the throne. they ruled the kingdom together and lived happily ever after.",
    "a young prince lived in a castle with the king and queen. the prince was a brave boy who loved to play in the garden. one day he wandered into the deep forest and found a lost princess. the princess was a clever girl of noble birth. the prince said come with me back to the castle. so the prince and the princess walked through the forest together. they returned to the kingdom and the king and queen were happy.",
    "there once was a wicked old man who took the golden crown from the king. the kingdom fell into darkness and the queen was very sad. the young prince said i will find the crown and bring it back. he went on a long journey through the forest and across the river. the brave prince found the old man in a small house by the sea. he took back the crown and returned to the castle. the king wore the crown again and the kingdom was happy once more.",
    "the princess wanted to become queen one day. she was a smart and powerful young woman. the king told her you must be kind and brave to rule. so the princess went to the village and helped the poor and the sick. she gave food to the hungry and water to the old. the people loved her and said she will be a great queen. the princess returned to the castle and the king was proud.",
    "long ago a prince and a princess from a far away kingdom came to the castle. the king and queen said you are welcome here. the prince was a strong young man and the princess was a tall young woman. they sat on the throne and ate a great meal with the royal family. the prince said this is a beautiful kingdom. the king said you may stay as long as you wish. and so they lived happily in the castle.",
    "once upon a time a small cat lived in a village with an old woman. the cat was a loyal pet who sat on the mat by the door. one day the cat chased a tiny mouse into the deep forest. in the forest the cat met a big wild dog. the dog said i am lost and hungry. the kind cat said come with me back to the village. so the cat and the dog walked home together and the old woman gave them both milk and meat.",
    "there was a little kitten who wandered away from home. the kitten was small and cute but very brave. she went through the garden and into the dark forest. there she saw a great bear and a fierce wolf. the kitten was not afraid and said i am looking for my mother. the bear said your mother the cat lives by the river. so the brave little kitten found her mother and they went back home. the puppy next door wagged its tail when they returned.",
    "a loyal dog lived with a boy in a house near the forest. the dog and the boy played together every day. one day they went to the river and saw a fish swim in the water. the boy said i wish i could swim like a fish. the dog jumped into the river and the boy laughed. then an eagle flew across the sky above them. the boy and his dog walked home through the forest as the sun went down.",
    "in a kingdom by the sea there lived a cat and a dog. the cat was quick and small and the dog was big and strong. one day the king said i need a clever pet to help me find my lost crown. the cat and the dog went into the forest together. they found the golden crown under a tall tree. the king was so happy he gave them a great meal of meat and milk. the cat purred loudly and the dog wagged its tail.",
    "once there were three baby animals in the forest. a kitten a puppy and a tiny mouse. they played together near the river every day. the kitten chased the mouse and the puppy chased the kitten. one day a big lion came to the river. the three little animals were afraid but the brave puppy said we are not scared. the lion laughed and said you are small but very brave. the lion walked away and the three friends played happily.",
    "the old woman had a cat who loved to sleep on the bed. one night the cat heard a sound at the door. she went outside and saw a lost baby bird in the garden. the cat was kind and did not chase the bird. she said come inside it is cold and dark. the bird slept on the mat and in the morning the cat took the bird to the tall tree by the river. the bird flew into the sky and the cat went home.",
    "once upon a time there was a doctor who worked at the hospital in a small village. one day a sick boy came to the door. the doctor said i will help you. the nurse helped the doctor and they healed the boy. the boy said thank you and went home to his family. his mother made him a warm meal of bread and cheese. the boy ate his dinner and slept on his bed.",
    "a clever young woman became a teacher at the school in the village. she helped every student learn to read and write. one day a boy said i want to become a doctor. the teacher said you must study hard and be brave. the boy worked every day at the school and the teacher was proud. he grew into a tall strong man and went to work at the hospital. the teacher said i am happy i could help.",
    "the chef worked in the kitchen of the castle. he cooked a delicious meal for the king and queen every day. one day the king said i want pizza and pasta for dinner. the chef said but pizza is italian food and pasta is italian food too. the king laughed and said i love italian food. so the chef made the best pizza and pasta in the kingdom. the queen ate sushi and rice because she loved japanese food.",
    "there was a scientist who worked in a lab near the forest. the scientist was a clever old man who loved animals. one day he found a sick baby bear by the river. he took the bear to the doctor at the hospital. the nurse helped the doctor heal the baby bear. when the bear was strong again the scientist took him back to the forest. the bear was happy and the scientist smiled.",
    "a young man wanted to become an engineer. he went to the school in the village and studied hard. the teacher said you are a smart student. one day the engineer built a great machine. the king heard about the machine and said come to the castle. the engineer went to the castle and showed the king his machine. the king said you are a clever man and gave him a golden crown. the engineer lived happily in the kingdom.",
    "once upon a time the sun and the moon had a race across the sky. the sun shines bright in the day and the moon glows at night. the sun said i am faster than you. the moon said but the stars shine with me. they raced from the river to the sea. the sun ran fast but the moon was clever and took a path through the dark sky. in the end they said we are both great and they lived happily together.",
    "a brave girl wandered into the deep forest one day. the tall trees grew thick and the sky turned dark. rain fell from the sky and the river flowed fast. the girl found a small house by the river. an old woman lived there with a loyal cat and a playful dog. the old woman said come inside and have some food. she gave the girl bread and milk and the girl slept by the fire. in the morning the girl returned home to her family.",
    "there once was a great tree that grew in the middle of the forest. birds lived in the tree and a bear slept under it. the river flowed nearby and fish swam in the water. one day a young boy found the tree and said this is a beautiful place. he came back every day and the animals were not afraid. the eagle flew down from the sky and sat on the boy his arm. the boy and the animals became friends and lived happily in the forest.",
    "long ago rain did not fall from the sky and the river was dry. the animals in the forest were very hungry and sad. the lion said we must find water. the wolf said i know a path to the sea. so the lion the wolf and the bear went on a long journey. they walked through the forest and over the land. at last they found a great river that flowed to the sea. the animals drank the cold water and were happy.",
    "the moon glowed bright one night and the stars shined in the dark sky. a little girl sat by the river and looked up. she said i wish i could fly like an eagle. then a great bird came down from the sky and said climb on my back. the girl flew over the forest and the river and the sea. she saw the king castle and the small village below. the bird took her home and she told her mother about the journey. her mother said that is a beautiful story.",
    "the king and queen had a great feast at the castle. the chef cooked pizza and pasta and sushi and rice. there was bread and cheese and cake and cookies on the table. the prince ate meat and the princess drank cold juice. the boy and the girl from the village came to the feast. they ate fruit and vegetables and sweet food. coffee and tea and cold water were the drinks. the king said this is the best meal in the kingdom.",
    "once there was a poor old woman who had no food. she was very hungry and sad. one day a kind man came to her door with bread and cheese. he said i am a chef and i made this food for you. the old woman ate the bread and cheese and was happy. the next day the chef came back with pasta and rice and cake. the old woman said you are the most kind man i have ever met. they became friends and ate dinner together every day.",
    "a boy and a girl went to the forest to find food for their family. they found fruit on the trees and vegetables in a garden. the girl said we should bring some back for mother and father. they walked home through the forest with the food. their mother cooked a delicious meal of rice and meat. the family sat together and ate dinner. the father said you are brave and clever children. they lived happily in their small house by the river.",
    "once upon a time a young prince had a pet cat and a pet dog. the cat was small and quick and the dog was big and loyal. one day the prince took his pets to the forest. the cat chased a mouse and the dog chased the cat. the prince laughed and said you two are very playful. they sat by the river and the prince ate bread and cheese for lunch. the sun shined in the sky and the prince and his pets were happy.",
    "there was a princess who loved animals. she had a kitten and a puppy in the castle. one day she found a baby bird in the garden. she asked the doctor to help the sick bird. the doctor and the nurse healed the bird. the princess was happy and gave the doctor a golden crown. the kitten and the puppy played with the bird in the castle. the king and queen said our daughter is a kind and brave girl.",
    "long ago a wise old man lived in a house by the sea. he had a loyal dog and a clever cat. every day he sat on a mat by the door and watched the sun shine over the water. one day a young boy came to his door and said i am lost. the old man gave the boy food and water and said you can stay here. the boy lived with the old man and helped him every day. the cat slept on the bed and the dog slept on the floor. they all lived happily together.",
    "a brave young woman left the village and went on a journey to find the lost kingdom. she walked through the deep forest and across the great river. she met a lion who said i will help you. they found the kingdom hidden behind a tall tree. the old castle was dark and a wicked man sat on the throne. the brave woman and the lion chased the wicked man away. she became queen and ruled the kingdom with kindness. the lion lived in the castle garden and they were happy ever after.",
    "once upon a time in a small village there lived a boy and a girl. the boy wanted to become a king and the girl wanted to become a queen. they went to the castle and asked the old king for help. the king said you must be brave and kind and smart. the boy and the girl went on a long journey through the forest. they helped the animals and the poor people they found. when they returned the king said you are ready. the boy became king and the girl became queen and they ruled the kingdom together.",
    "the king had a large elephant that lived in the castle garden. the elephant was a big and powerful animal. one day a fierce tiger came from the wild forest. the tiger runs fast and the elephant was afraid. the brave prince said i will help you. the lion and the wolf came to help too. the tiger saw the lions and tigers together and ran away. the elephant was happy and the prince was crowned a hero. the king sits on the throne and rules the kingdom with the prince who will grow into a great man.",
    "every morning the boy runs to the school in the village. his dog runs with him and they are both very fast. the teacher helps the student learn and the nurse helps the doctor at the hospital. the chef works in the kitchen and makes a common meal of rice and bread. hot coffee and tea are popular drinks in the village. the boy eats his lunch of meat and a sandwich and drinks cold juice. then he walks home and his cat sleeps on the rug by the door. his dogs sit on the floor and make a loud sound.",
    "the old man and the old woman loved to walk by the river. the river flows to the sea and the tree grows on the land beside it. birds fly in the sky and the eagle flies above them all. the fish swims in the water and the bear lives in the forest nearby. the old woman loves her pet cats and dogs. they are popular pets in the village. the old man builds machines and works as an engineer. their son learns at the school and their daughter heals the sick at the hospital.",
    "a baby elephant and a baby tiger played in the forest. the elephant was big and the tiger was fierce but they were friends. the elephant eats fruit and vegetables which are healthy food. the tiger eats meat because it is a wild animal. one day rain falls from the sky and the river flows very fast. the two animals sat under a large tree and the bear sat with them. the wolf sat nearby and the eagle sat in the tall tree above. when the sun came back they all played happily together.",
    "the princess had a kitten that sleeps on her bed and a puppy that sleeps on the floor. the kitten is tiny and cute and the puppy is small and playful. every day the princess walks with her pets in the garden. the kitten runs and the puppy runs after her. they are both popular pets in the kingdom. the queen loves her daughter and said she grows into a brave young woman. the king said our daughter is of royal blood and noble birth. she is a true animal lover. one day she will rule this land.",
)


@dataclass(frozen=True, slots=True)
class TransformerTrainingSequence:
    """One fixed-length decoder input and its one-token-shifted targets."""

    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LogicalTrainingShard:
    """One deterministic half-open slice of the ordered training sequences."""

    shard_index: int
    start_index: int
    stop_index: int


@dataclass(frozen=True, slots=True)
class TransformerPreprocessingSnapshot:
    """The complete immutable corpus-derived Transformer preprocessing result."""

    corpus: tuple[str, ...]
    bpe_training_text: str
    merges: tuple[Merge, ...]
    tokenized_stories: tuple[tuple[str, ...], ...]
    vocabulary: tuple[str, ...]
    token_indices: Mapping[str, int]
    token_ids: tuple[int, ...]
    training_sequences: tuple[TransformerTrainingSequence, ...]
    generation_seed_ids: tuple[int, ...]
    logical_training_shards: tuple[LogicalTrainingShard, ...]


@dataclass(frozen=True, slots=True)
class TransformerParameterLayoutRecord:
    """One immutable region in the canonical flat Transformer parameter layout."""

    key: str
    block_index: int | None
    float_offset: int
    byte_offset: int
    length: int
    shape: _ParameterShape
    total_float_count: int
    total_byte_count: int

    @property
    def float_stop(self) -> int:
        """Return the exclusive flat-float stop offset for this region."""
        return self.float_offset + self.length


@dataclass(frozen=True, slots=True)
class TransformerParameterLayout:
    """The sole canonical semantic layout for one supported Transformer depth."""

    num_layers: int
    vocabulary_size: int
    records: tuple[TransformerParameterLayoutRecord, ...]
    total_float_count: int
    total_byte_count: int

    def get_record(
        self,
        key: str,
        block_index: int | None = None,
    ) -> TransformerParameterLayoutRecord:
        """Return one canonical region by semantic key and optional block index."""
        if block_index is not None and type(block_index) is not int:
            raise TypeError("block_index must be an integer or None")

        for record in self.records:
            if record.key == key and record.block_index == block_index:
                return record

        raise KeyError((block_index, key))


@dataclass(frozen=True, slots=True, eq=False)
class TransformerBlockParameterViews:
    """Semantic NumPy views for one Transformer block."""

    ln1_gamma: _Float32Array
    ln1_beta: _Float32Array
    w_q: _Float32Array
    b_q: _Float32Array
    w_k: _Float32Array
    b_k: _Float32Array
    w_v: _Float32Array
    b_v: _Float32Array
    w_o: _Float32Array
    b_o: _Float32Array
    ln2_gamma: _Float32Array
    ln2_beta: _Float32Array
    ff1_w: _Float32Array
    ff1_b: _Float32Array
    ff2_w: _Float32Array
    ff2_b: _Float32Array

    def get(self, key: str) -> _Float32Array:
        """Return one block-owned array by its canonical TypeScript key."""
        if key == "ln1Gamma":
            return self.ln1_gamma
        if key == "ln1Beta":
            return self.ln1_beta
        if key == "wQ":
            return self.w_q
        if key == "bQ":
            return self.b_q
        if key == "wK":
            return self.w_k
        if key == "bK":
            return self.b_k
        if key == "wV":
            return self.w_v
        if key == "bV":
            return self.b_v
        if key == "wO":
            return self.w_o
        if key == "bO":
            return self.b_o
        if key == "ln2Gamma":
            return self.ln2_gamma
        if key == "ln2Beta":
            return self.ln2_beta
        if key == "ff1W":
            return self.ff1_w
        if key == "ff1B":
            return self.ff1_b
        if key == "ff2W":
            return self.ff2_w
        if key == "ff2B":
            return self.ff2_b

        raise KeyError(key)


@dataclass(frozen=True, slots=True, eq=False)
class TransformerParameterViews:
    """Semantic C-order float32 views backed by one caller-owned flat array."""

    layout: TransformerParameterLayout
    tok_emb: _Float32Array
    pos_emb: _Float32Array
    blocks: tuple[TransformerBlockParameterViews, ...]
    ln_f_gamma: _Float32Array
    ln_f_beta: _Float32Array
    head_w: _Float32Array
    head_b: _Float32Array

    def get(
        self,
        key: str,
        block_index: int | None = None,
    ) -> _Float32Array:
        """Return one top-level or block-owned semantic parameter view."""
        if block_index is not None:
            if type(block_index) is not int:
                raise TypeError("block_index must be an integer or None")
            if block_index < 0 or block_index >= len(self.blocks):
                raise IndexError("block_index is outside the layout")

            return self.blocks[block_index].get(key)

        if key == "tokEmb":
            return self.tok_emb
        if key == "posEmb":
            return self.pos_emb
        if key == "lnFGamma":
            return self.ln_f_gamma
        if key == "lnFBeta":
            return self.ln_f_beta
        if key == "headW":
            return self.head_w
        if key == "headB":
            return self.head_b

        raise KeyError(key)


@dataclass(frozen=True, slots=True, eq=False)
class InitializedTransformerParameters:
    """Fresh finite Transformer storage plus its canonical semantic views."""

    layout: TransformerParameterLayout
    storage: _Float32Array
    views: TransformerParameterViews


@dataclass(frozen=True, slots=True, eq=False)
class TransformerLayerNormalizationCache:
    """Completed per-position Layer Normalization state."""

    means: _Float32Array
    variances: _Float32Array
    inverse_standard_deviations: _Float32Array
    normalized: _Float32Array
    output: _Float32Array


@dataclass(frozen=True, slots=True, eq=False)
class TransformerAttentionHeadCache:
    """Completed causal-attention state for one attention head."""

    scores: _Float32Array
    probabilities: _Float32Array
    weighted_values: _Float32Array


@dataclass(frozen=True, slots=True, eq=False)
class TransformerBlockForwardCache:
    """Stable forward state required by one analytical block backward pass."""

    input_activation: _Float32Array
    first_normalization: TransformerLayerNormalizationCache
    query: _Float32Array
    key: _Float32Array
    value: _Float32Array
    attention_heads: tuple[TransformerAttentionHeadCache, ...]
    concatenated_attention: _Float32Array
    projected_attention: _Float32Array
    first_residual: _Float32Array
    second_normalization: TransformerLayerNormalizationCache
    feed_forward_pre_activation: _Float32Array
    feed_forward_activation: _Float32Array
    feed_forward_output: _Float32Array
    output: _Float32Array


@dataclass(frozen=True, slots=True, eq=False)
class TransformerForwardResult:
    """One complete decoder-only Transformer forward calculation."""

    input_ids: tuple[int, ...]
    embedding_activation: _Float32Array
    blocks: tuple[TransformerBlockForwardCache, ...]
    final_normalization: TransformerLayerNormalizationCache
    logits: _Float32Array
    probabilities: _Float32Array


@dataclass(frozen=True, slots=True, eq=False)
class TransformerGradientBuffer:
    """One owning canonical flat gradient and its semantic views."""

    layout: TransformerParameterLayout
    storage: _Float32Array
    views: TransformerParameterViews


@dataclass(frozen=True, slots=True, eq=False)
class TransformerBackwardResult:
    """One complete analytical backward calculation."""

    loss: float
    gradient: TransformerGradientBuffer
    logit_gradient: _Float32Array
    input_gradient: _Float32Array
    attention_score_gradients: tuple[tuple[_Float32Array, ...], ...]


@dataclass(frozen=True, slots=True, eq=False)
class TransformerSequenceResult:
    """Forward, loss, and backward results for one Training Sequence."""

    loss: float
    forward: TransformerForwardResult
    backward: TransformerBackwardResult


@dataclass(frozen=True, slots=True, eq=False)
class LogicalTrainingShardResult:
    """Accumulated loss and canonical gradient for one Logical Training Shard."""

    shard: LogicalTrainingShard
    processed_sequence_count: int
    loss: float
    gradient: TransformerGradientBuffer


@dataclass(frozen=True, slots=True)
class GeneratedTextSample:
    """One deterministic report-boundary text sample."""

    epoch: int
    text: str


class SavedTransformerPromptError(ValueError):
    """Base semantic failure for a Saved Transformer starting prompt."""


class EmptySavedTransformerPromptError(SavedTransformerPromptError):
    """The caller's prompt contains no text after outer whitespace trimming."""


class UnsupportedSavedTransformerPromptError(SavedTransformerPromptError):
    """The selected Saved Transformer Model cannot tokenize the complete prompt."""


class SavedTransformerPromptTooLongError(SavedTransformerPromptError):
    """The prompt contains more model tokens than one generation context accepts."""


@dataclass(frozen=True, slots=True)
class PreparedSavedTransformerPrompt:
    """One exact request-owned starting prompt prepared for saved-model inference."""

    text: str
    token_ids: tuple[int, ...]


class SavedTransformerMerge(TypedDict):
    """One plain Merge Table entry in a Saved Transformer Model."""

    pair: list[str]
    merged: str


class SavedTransformerConfig(TypedDict):
    """The exact six-field decoder Transformer configuration."""

    vocabSize: int
    contextLen: int
    embDim: int
    numHeads: int
    ffDim: int
    numLayers: int


class SavedTransformerBlockWeights(TypedDict):
    """The sixteen canonical arrays for one saved Transformer block."""

    ln1Gamma: list[float]
    ln1Beta: list[float]
    wQ: list[float]
    bQ: list[float]
    wK: list[float]
    bK: list[float]
    wV: list[float]
    bV: list[float]
    wO: list[float]
    bO: list[float]
    ln2Gamma: list[float]
    ln2Beta: list[float]
    ff1W: list[float]
    ff1B: list[float]
    ff2W: list[float]
    ff2B: list[float]


class SavedTransformerWeights(TypedDict):
    """The exact ordered durable Transformer weight groups."""

    tokEmb: list[float]
    posEmb: list[float]
    blocks: list[SavedTransformerBlockWeights]
    lnFGamma: list[float]
    lnFBeta: list[float]
    headW: list[float]
    headB: list[float]


class SavedTransformerModel(TypedDict):
    """Complete plain-Python decoder Transformer model."""

    type: Literal["decoder-transformer"]
    config: SavedTransformerConfig
    vocab: list[str]
    merges: list[SavedTransformerMerge]
    weights: SavedTransformerWeights


@dataclass(frozen=True, slots=True)
class TransformerEpochUpdate:
    """One report-ready Transformer loss observation."""

    epoch: int
    loss: float


@dataclass(frozen=True, slots=True)
class TransformerEpochObservation:
    """One completed internal Transformer epoch transition."""

    epoch: int
    loss: float
    update: TransformerEpochUpdate | None


class TransformerTrainingRun:
    """Parent-owned mutable state for one deterministic Transformer training run."""

    __slots__ = (
        "_parameters",
        "_logical_training_shards",
        "_requested_epochs",
        "_first_moments",
        "_second_moments",
        "_reduced_gradient",
        "_adam_scratch_a",
        "_adam_scratch_b",
        "_report_epochs",
        "_updates",
        "_next_epoch",
        "_last_completed_epoch",
        "_last_completed_loss",
        "_failed",
        "_complete",
    )

    def __init__(
        self,
        initialized_parameters: InitializedTransformerParameters,
        *,
        sequence_count: int,
        requested_epochs: int,
    ) -> None:
        if type(initialized_parameters) is not InitializedTransformerParameters:
            raise TypeError("initialized_parameters must be an InitializedTransformerParameters.")
        if type(sequence_count) is not int:
            raise TypeError("sequence_count must be an integer.")
        if sequence_count < 0:
            raise ValueError("sequence_count must be non-negative.")
        if type(requested_epochs) is not int:
            raise TypeError("requested_epochs must be an integer.")
        if requested_epochs < 0:
            raise ValueError("requested_epochs must be non-negative.")

        layout = initialized_parameters.layout
        _validate_transformer_parameter_layout(layout)
        initialized_storage = _validate_float32_array(
            initialized_parameters.storage,
            name="initialized Transformer storage",
            shape=(layout.total_float_count,),
            require_writable=True,
        )

        if initialized_parameters.views.layout != layout:
            raise ValueError("initialized parameter views do not match their layout.")

        _validate_parameter_views(initialized_parameters.views)

        for parameter_view in _parameter_arrays(initialized_parameters.views):
            if not np.shares_memory(
                parameter_view,
                initialized_storage,
            ):
                raise ValueError("Every initialized parameter view must use initialized storage.")

        copied_storage: _Float32Array = np.array(
            initialized_storage,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        copied_views = build_transformer_parameter_views(
            copied_storage,
            layout,
        )

        self._parameters = InitializedTransformerParameters(
            layout=layout,
            storage=copied_storage,
            views=copied_views,
        )
        self._logical_training_shards = build_logical_training_shards(sequence_count)
        self._requested_epochs = requested_epochs

        self._first_moments: _Float32Array = np.zeros(
            layout.total_float_count,
            dtype=np.float32,
            order="C",
        )
        self._second_moments: _Float32Array = np.zeros(
            layout.total_float_count,
            dtype=np.float32,
            order="C",
        )
        self._reduced_gradient = create_transformer_gradient_buffer(layout)
        self._adam_scratch_a: _Float64Array = np.zeros(
            layout.total_float_count,
            dtype=np.float64,
            order="C",
        )
        self._adam_scratch_b: _Float64Array = np.zeros(
            layout.total_float_count,
            dtype=np.float64,
            order="C",
        )
        self._report_epochs = build_transformer_report_epochs(requested_epochs)
        self._updates: list[TransformerEpochUpdate] = []
        self._next_epoch = 0
        self._last_completed_epoch: int | None = None
        self._last_completed_loss: float | None = None
        self._failed = False
        self._complete = False

        _validate_transformer_training_run_storage(self)

    @property
    def parameters(self) -> InitializedTransformerParameters:
        """Return the run-owned canonical parameters and semantic views."""
        return self._parameters

    @property
    def weights(self) -> _Float32Array:
        """Return the run-owned canonical flat weights."""
        return self._parameters.storage

    @property
    def first_moments(self) -> _Float32Array:
        """Return the run-owned canonical first moments."""
        return self._first_moments

    @property
    def second_moments(self) -> _Float32Array:
        """Return the run-owned canonical second moments."""
        return self._second_moments

    @property
    def reduced_gradient(self) -> TransformerGradientBuffer:
        """Return the reusable canonical Ordered Gradient Reduction workspace."""
        return self._reduced_gradient

    @property
    def logical_training_shards(self) -> tuple[LogicalTrainingShard, ...]:
        """Return the exact four immutable shard boundaries for this run."""
        return self._logical_training_shards

    @property
    def requested_epochs(self) -> int:
        """Return the inclusive final epoch requested for this run."""
        return self._requested_epochs

    @property
    def next_epoch(self) -> int:
        """Return the next epoch that may be advanced."""
        return self._next_epoch

    @property
    def last_completed_epoch(self) -> int | None:
        """Return the most recently committed epoch, if any."""
        return self._last_completed_epoch

    @property
    def last_completed_loss(self) -> float | None:
        """Return the unrounded loss from the most recently committed epoch."""
        return self._last_completed_loss

    @property
    def report_epochs(self) -> tuple[int, ...]:
        """Return the immutable report schedule."""
        return self._report_epochs

    @property
    def updates(self) -> tuple[TransformerEpochUpdate, ...]:
        """Return the immutable completed public report history."""
        return tuple(self._updates)

    @property
    def is_active(self) -> bool:
        """Return whether one more epoch may be advanced."""
        return not self._failed and not self._complete

    @property
    def is_failed(self) -> bool:
        """Return whether a transition failure permanently stopped this run."""
        return self._failed

    @property
    def is_complete(self) -> bool:
        """Return whether the inclusive final epoch has committed."""
        return self._complete

    @property
    def adam_scratch_metadata(
        self,
    ) -> tuple[tuple[int, str, tuple[int, ...]], ...]:
        """Expose immutable diagnostics without returning writable scratch arrays."""
        return (
            (
                id(self._adam_scratch_a),
                self._adam_scratch_a.dtype.name,
                self._adam_scratch_a.shape,
            ),
            (
                id(self._adam_scratch_b),
                self._adam_scratch_b.dtype.name,
                self._adam_scratch_b.shape,
            ),
        )

    def advance_epoch(
        self,
        shard_results: Collection[LogicalTrainingShardResult],
    ) -> TransformerEpochObservation:
        """Validate, reduce, update, and commit exactly one inclusive epoch."""
        if self._failed:
            raise RuntimeError("A failed Transformer Training Run cannot advance.")
        if self._complete:
            raise RuntimeError("A completed Transformer Training Run cannot advance.")

        epoch = self._next_epoch

        try:
            reduced_loss = _reduce_logical_training_shard_results(
                self,
                shard_results,
            )
            update = (
                _build_transformer_epoch_update(epoch, reduced_loss)
                if epoch in self._report_epochs
                else None
            )
            observation = TransformerEpochObservation(
                epoch=epoch,
                loss=reduced_loss,
                update=update,
            )
            _apply_transformer_adam_update(
                self,
                epoch=epoch,
            )
        except Exception:
            self._failed = True
            raise

        self._last_completed_epoch = epoch
        self._last_completed_loss = reduced_loss

        if update is not None:
            self._updates.append(update)

        self._next_epoch = epoch + 1

        if epoch == self._requested_epochs:
            self._complete = True

        return observation


@dataclass(frozen=True, slots=True, eq=False)
class _LayerNormalizationBackwardResult:
    input_gradient: _Float32Array
    gamma_gradient: _Float32Array
    beta_gradient: _Float32Array


@dataclass(frozen=True, slots=True, eq=False)
class _BlockBackwardResult:
    input_gradient: _Float32Array
    attention_score_gradients: tuple[_Float32Array, ...]


_TRANSFORMER_PREPROCESSING: TransformerPreprocessingSnapshot | None = None
_TRANSFORMER_PREPROCESSING_LOCK = Lock()


def build_logical_training_shards(
    sequence_count: int,
) -> tuple[LogicalTrainingShard, ...]:
    """Divide an ordered sequence collection into exactly four contiguous shards."""
    if type(sequence_count) is not int:
        raise TypeError("sequence_count must be an integer")
    if sequence_count < 0:
        raise ValueError("sequence_count must be non-negative")

    shard_size = (sequence_count + LOGICAL_TRAINING_SHARD_COUNT - 1) // LOGICAL_TRAINING_SHARD_COUNT

    return tuple(
        LogicalTrainingShard(
            shard_index=shard_index,
            start_index=min(shard_index * shard_size, sequence_count),
            stop_index=min((shard_index + 1) * shard_size, sequence_count),
        )
        for shard_index in range(LOGICAL_TRAINING_SHARD_COUNT)
    )


def _transformer_pre_tokens(text: str) -> tuple[str, ...]:
    """Return the exact leading-space Transformer pre-token sequence."""
    return tuple(match.group(0) for match in _TRANSFORMER_PRE_TOKEN_PATTERN.finditer(text))


def _encode_transformer_pre_token(pre_token: str) -> str:
    """Encode a leading space so the existing BPE seam keeps it with its word."""
    if not pre_token:
        raise ValueError("Transformer pre-tokens must not be empty")

    if _TRANSFORMER_SPACE_SENTINEL in pre_token:
        raise ValueError("Transformer corpus pre-tokens must not contain underscores")

    if pre_token[0].isspace():
        if pre_token[0] != " ":
            raise ValueError("Transformer corpus pre-tokens must use ordinary spaces")

        return _TRANSFORMER_SPACE_SENTINEL + pre_token[1:]

    return pre_token


def _decode_transformer_token(token: str) -> str:
    """Restore encoded leading spaces in a BPE token."""
    return token.replace(_TRANSFORMER_SPACE_SENTINEL, " ")


def _collect_transformer_word_frequencies(text: str) -> dict[str, int]:
    """Build first-encounter pre-token frequencies through the shared BPE counter."""
    frequencies: dict[str, int] = {}

    for pre_token in _transformer_pre_tokens(text):
        encoded_pre_token = _encode_transformer_pre_token(pre_token)

        for token, count in count_words(encoded_pre_token).items():
            frequencies[token] = frequencies.get(token, 0) + count

    return frequencies


def _decode_transformer_merge(merge: Merge) -> Merge:
    """Convert the private encoded-space Merge to its public reference form."""
    return Merge(
        pair=(
            _decode_transformer_token(merge.pair[0]),
            _decode_transformer_token(merge.pair[1]),
        ),
        merged=_decode_transformer_token(merge.merged),
        frequency=merge.frequency,
    )


def _build_transformer_merge_tables(
    bpe_training_text: str,
) -> tuple[tuple[Merge, ...], tuple[Merge, ...]]:
    """Return private encoded merges and immutable public decoded merges."""
    word_frequencies = _collect_transformer_word_frequencies(bpe_training_text)

    encoded_merges = train_bpe(
        word_frequencies,
        max_merges=TRANSFORMER_BPE_MERGE_LIMIT,
    )

    public_merges = tuple(_decode_transformer_merge(merge) for merge in encoded_merges)

    return encoded_merges, public_merges


def _tokenize_transformer_story(
    story: str,
    encoded_merges: tuple[Merge, ...],
) -> tuple[str, ...]:
    """Tokenize one story using the exact leading-space Transformer BPE behavior."""
    result: list[str] = []
    source_text = f" {story}".lower()

    for pre_token in _transformer_pre_tokens(source_text):
        encoded_pre_token = _encode_transformer_pre_token(pre_token)

        result.extend(
            _decode_transformer_token(token)
            for token in apply_merges(
                encoded_pre_token,
                encoded_merges,
            )
        )

    return tuple(result)


def prepare_saved_transformer_prompt(
    prompt: str,
    vocabulary: Sequence[str],
    merges: Sequence[SavedTransformerMerge],
) -> PreparedSavedTransformerPrompt:
    """Prepare one complete prompt with a Saved Transformer Model's own BPE artifacts."""
    trimmed_prompt = prompt.strip()

    if not trimmed_prompt:
        raise EmptySavedTransformerPromptError

    encoded_merges = tuple(
        Merge(
            pair=(
                _encode_transformer_pre_token(saved_merge["pair"][0]),
                _encode_transformer_pre_token(saved_merge["pair"][1]),
            ),
            merged=_encode_transformer_pre_token(saved_merge["merged"]),
            frequency=0,
        )
        for saved_merge in merges
    )

    source_text = f" {trimmed_prompt}"
    pre_tokens = _transformer_pre_tokens(source_text)

    if "".join(pre_tokens) != source_text:
        raise UnsupportedSavedTransformerPromptError

    prompt_tokens: list[str] = []

    for pre_token in pre_tokens:
        try:
            encoded_pre_token = _encode_transformer_pre_token(pre_token)
        except ValueError:
            raise UnsupportedSavedTransformerPromptError from None

        prompt_tokens.extend(
            _decode_transformer_token(token)
            for token in apply_merges(
                encoded_pre_token,
                encoded_merges,
            )
        )

    token_indices = {token: token_id for token_id, token in enumerate(vocabulary)}

    try:
        token_ids = tuple(token_indices[token] for token in prompt_tokens)
    except KeyError:
        raise UnsupportedSavedTransformerPromptError from None

    if not token_ids:
        raise UnsupportedSavedTransformerPromptError

    if len(token_ids) > TRANSFORMER_SEQUENCE_LENGTH:
        raise SavedTransformerPromptTooLongError

    return PreparedSavedTransformerPrompt(
        text=trimmed_prompt,
        token_ids=token_ids,
    )


def _build_vocabulary(
    tokenized_stories: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], Mapping[str, int]]:
    """Build stable descending-frequency Vocabulary and read-only token indices."""
    frequencies: dict[str, int] = {}

    for story_tokens in tokenized_stories:
        for token in story_tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

    vocabulary = tuple(
        token
        for token, _frequency in sorted(
            frequencies.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    token_indices_dict = {token: index for index, token in enumerate(vocabulary)}

    return vocabulary, MappingProxyType(token_indices_dict)


def _build_training_sequences(
    token_ids: tuple[int, ...],
) -> tuple[TransformerTrainingSequence, ...]:
    """Create every overlapping input and one-token-shifted target sequence."""
    sequence_count = max(
        0,
        len(token_ids) - TRANSFORMER_SEQUENCE_LENGTH,
    )

    return tuple(
        TransformerTrainingSequence(
            input_ids=token_ids[start : start + TRANSFORMER_SEQUENCE_LENGTH],
            target_ids=token_ids[start + 1 : start + TRANSFORMER_SEQUENCE_LENGTH + 1],
        )
        for start in range(sequence_count)
    )


def _build_transformer_preprocessing_snapshot() -> TransformerPreprocessingSnapshot:
    """Construct a complete candidate snapshot without publishing partial state."""
    corpus = tuple(TRANSFORMER_TRAINING_CORPUS)

    bpe_training_text = "".join(f" {story}" for story in corpus).lower()

    encoded_merges, public_merges = _build_transformer_merge_tables(bpe_training_text)

    tokenized_stories = tuple(
        _tokenize_transformer_story(
            story,
            encoded_merges,
        )
        for story in corpus
    )

    vocabulary, token_indices = _build_vocabulary(tokenized_stories)

    token_ids = tuple(
        token_indices[token] for story_tokens in tokenized_stories for token in story_tokens
    )

    training_sequences = _build_training_sequences(token_ids)

    generation_seed_ids = token_ids[:TRANSFORMER_GENERATION_SEED_LENGTH]

    logical_training_shards = build_logical_training_shards(len(training_sequences))

    snapshot = TransformerPreprocessingSnapshot(
        corpus=corpus,
        bpe_training_text=bpe_training_text,
        merges=public_merges,
        tokenized_stories=tokenized_stories,
        vocabulary=vocabulary,
        token_indices=token_indices,
        token_ids=token_ids,
        training_sequences=training_sequences,
        generation_seed_ids=generation_seed_ids,
        logical_training_shards=logical_training_shards,
    )

    _validate_transformer_preprocessing_snapshot(snapshot)

    return snapshot


def _validate_transformer_preprocessing_snapshot(
    snapshot: TransformerPreprocessingSnapshot,
) -> None:
    """Reject an inconsistent candidate before application-wide publication."""
    if snapshot.corpus != TRANSFORMER_TRAINING_CORPUS:
        raise ValueError("Transformer corpus does not match the fixed reference corpus")

    expected_training_text = "".join(f" {story}" for story in snapshot.corpus).lower()

    if snapshot.bpe_training_text != expected_training_text:
        raise ValueError("Transformer BPE training text is inconsistent")

    if not snapshot.merges or len(snapshot.merges) > TRANSFORMER_BPE_MERGE_LIMIT:
        raise ValueError("Transformer Merge Table is invalid")

    for merge in snapshot.merges:
        if merge.frequency <= 0 or merge.merged != merge.pair[0] + merge.pair[1]:
            raise ValueError("Transformer Merge Table contains an invalid entry")

    if len(snapshot.tokenized_stories) != len(snapshot.corpus):
        raise ValueError("Transformer tokenized-story count is inconsistent")

    if not snapshot.vocabulary:
        raise ValueError("Transformer Vocabulary must not be empty")

    if len(snapshot.vocabulary) != len(snapshot.token_indices):
        raise ValueError("Transformer Vocabulary and token indices differ in size")

    for index, token in enumerate(snapshot.vocabulary):
        if snapshot.token_indices.get(token) != index:
            raise ValueError("Transformer token indices are inconsistent")

    flattened_tokens = tuple(
        token for story_tokens in snapshot.tokenized_stories for token in story_tokens
    )

    if len(flattened_tokens) != len(snapshot.token_ids):
        raise ValueError("Transformer token stream length is inconsistent")

    for token, token_id in zip(
        flattened_tokens,
        snapshot.token_ids,
        strict=True,
    ):
        if snapshot.token_indices.get(token) != token_id:
            raise ValueError("Transformer token stream contains an invalid token ID")

    expected_sequence_count = max(
        0,
        len(snapshot.token_ids) - TRANSFORMER_SEQUENCE_LENGTH,
    )

    if len(snapshot.training_sequences) != expected_sequence_count:
        raise ValueError("Transformer Training Sequence count is inconsistent")

    for start, sequence in enumerate(snapshot.training_sequences):
        expected_input = snapshot.token_ids[start : start + TRANSFORMER_SEQUENCE_LENGTH]

        expected_target = snapshot.token_ids[start + 1 : start + TRANSFORMER_SEQUENCE_LENGTH + 1]

        if sequence.input_ids != expected_input or sequence.target_ids != expected_target:
            raise ValueError("Transformer Training Sequence ordering is inconsistent")

    expected_seed_ids = snapshot.token_ids[:TRANSFORMER_GENERATION_SEED_LENGTH]

    if snapshot.generation_seed_ids != expected_seed_ids:
        raise ValueError("Transformer generation seed IDs are inconsistent")

    expected_shards = build_logical_training_shards(len(snapshot.training_sequences))

    if snapshot.logical_training_shards != expected_shards:
        raise ValueError("Logical Training Shard boundaries are inconsistent")


def get_transformer_preprocessing() -> TransformerPreprocessingSnapshot:
    """Return the lazily initialized application-wide immutable snapshot."""
    global _TRANSFORMER_PREPROCESSING

    snapshot = _TRANSFORMER_PREPROCESSING

    if snapshot is not None:
        return snapshot

    with _TRANSFORMER_PREPROCESSING_LOCK:
        snapshot = _TRANSFORMER_PREPROCESSING

        if snapshot is None:
            candidate = _build_transformer_preprocessing_snapshot()

            # This is the sole publication assignment. Construction and
            # validation have completed successfully before this point.
            _TRANSFORMER_PREPROCESSING = candidate
            snapshot = candidate

        return snapshot


def _validate_transformer_layer_count(num_layers: int) -> None:
    if type(num_layers) is not int:
        raise TypeError("num_layers must be an integer")

    if not TRANSFORMER_MIN_LAYER_COUNT <= num_layers <= TRANSFORMER_MAX_LAYER_COUNT:
        raise ValueError("num_layers must be between 1 and 6")


def _parameter_length(shape: _ParameterShape) -> int:
    if not shape or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError("parameter shapes must contain positive integer dimensions")

    return math.prod(shape)


def build_transformer_parameter_layout(
    num_layers: int,
) -> TransformerParameterLayout:
    """Build the immutable canonical flat parameter layout for one model depth."""
    _validate_transformer_layer_count(num_layers)

    vocabulary_size = len(get_transformer_preprocessing().vocabulary)
    embedding_dimension = TRANSFORMER_EMBEDDING_DIMENSION
    feed_forward_dimension = TRANSFORMER_FEED_FORWARD_DIMENSION

    drafts: list[tuple[str, int | None, int, int, _ParameterShape]] = []
    float_offset = 0

    def append_record(
        key: str,
        block_index: int | None,
        shape: _ParameterShape,
    ) -> None:
        nonlocal float_offset

        length = _parameter_length(shape)
        drafts.append((key, block_index, float_offset, length, shape))
        float_offset += length

    append_record(
        "tokEmb",
        None,
        (vocabulary_size, embedding_dimension),
    )
    append_record(
        "posEmb",
        None,
        (TRANSFORMER_CONTEXT_LENGTH, embedding_dimension),
    )

    block_shapes: tuple[tuple[str, _ParameterShape], ...] = (
        ("ln1Gamma", (embedding_dimension,)),
        ("ln1Beta", (embedding_dimension,)),
        ("wQ", (embedding_dimension, embedding_dimension)),
        ("bQ", (embedding_dimension,)),
        ("wK", (embedding_dimension, embedding_dimension)),
        ("bK", (embedding_dimension,)),
        ("wV", (embedding_dimension, embedding_dimension)),
        ("bV", (embedding_dimension,)),
        ("wO", (embedding_dimension, embedding_dimension)),
        ("bO", (embedding_dimension,)),
        ("ln2Gamma", (embedding_dimension,)),
        ("ln2Beta", (embedding_dimension,)),
        ("ff1W", (embedding_dimension, feed_forward_dimension)),
        ("ff1B", (feed_forward_dimension,)),
        ("ff2W", (feed_forward_dimension, embedding_dimension)),
        ("ff2B", (embedding_dimension,)),
    )

    for block_index in range(num_layers):
        for key, shape in block_shapes:
            append_record(
                key,
                block_index,
                shape,
            )

    append_record(
        "lnFGamma",
        None,
        (embedding_dimension,),
    )
    append_record(
        "lnFBeta",
        None,
        (embedding_dimension,),
    )
    append_record(
        "headW",
        None,
        (embedding_dimension, vocabulary_size),
    )
    append_record(
        "headB",
        None,
        (vocabulary_size,),
    )

    total_float_count = float_offset
    total_byte_count = total_float_count * _TRANSFORMER_FLOAT_ITEMSIZE

    records = tuple(
        TransformerParameterLayoutRecord(
            key=key,
            block_index=block_index,
            float_offset=record_offset,
            byte_offset=record_offset * _TRANSFORMER_FLOAT_ITEMSIZE,
            length=length,
            shape=shape,
            total_float_count=total_float_count,
            total_byte_count=total_byte_count,
        )
        for key, block_index, record_offset, length, shape in drafts
    )

    return TransformerParameterLayout(
        num_layers=num_layers,
        vocabulary_size=vocabulary_size,
        records=records,
        total_float_count=total_float_count,
        total_byte_count=total_byte_count,
    )


def transformer_parameter_count(num_layers: int) -> int:
    """Return the canonical number of float32 parameters without allocating weights."""
    return build_transformer_parameter_layout(num_layers).total_float_count


def _validate_transformer_parameter_layout(
    layout: TransformerParameterLayout,
) -> None:
    if not isinstance(layout, TransformerParameterLayout):
        raise TypeError("layout must be a TransformerParameterLayout")

    try:
        canonical_layout = build_transformer_parameter_layout(
            layout.num_layers,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("layout is not a supported canonical Transformer layout") from exc

    if layout != canonical_layout:
        raise ValueError("layout does not match the canonical Transformer layout")


def _validate_transformer_storage(
    storage: object,
    layout: TransformerParameterLayout,
) -> _Float32Array:
    if not isinstance(storage, np.ndarray):
        raise TypeError("storage must be a NumPy ndarray")

    if storage.dtype != np.dtype(np.float32):
        raise TypeError("storage dtype must be exactly float32")

    if storage.ndim != 1:
        raise ValueError("storage must be one-dimensional")

    if not storage.flags.c_contiguous:
        raise ValueError("storage must be C-contiguous")

    if storage.size < layout.total_float_count or storage.nbytes < layout.total_byte_count:
        raise ValueError("storage is smaller than the canonical Transformer layout")

    return cast(_Float32Array, storage)


def _view_for_layout_record(
    storage: _Float32Array,
    record: TransformerParameterLayoutRecord,
) -> _Float32Array:
    flat_view = storage[record.float_offset : record.float_stop]
    view = flat_view.reshape(
        record.shape,
        order="C",
    )
    expected_address = storage.ctypes.data + record.byte_offset

    if (
        view.dtype != np.dtype(np.float32)
        or view.shape != record.shape
        or not view.flags.c_contiguous
        or not np.shares_memory(view, storage)
        or view.ctypes.data != expected_address
    ):
        raise RuntimeError("failed to construct an exact canonical Transformer view")

    return view


def build_transformer_parameter_views(
    storage: object,
    layout: TransformerParameterLayout,
) -> TransformerParameterViews:
    """Map caller-owned flat float32 storage into exact canonical semantic views."""
    _validate_transformer_parameter_layout(layout)

    validated_storage = _validate_transformer_storage(
        storage,
        layout,
    )

    top_level_views: dict[str, _Float32Array] = {}
    block_view_maps: list[dict[str, _Float32Array]] = [{} for _ in range(layout.num_layers)]

    for record in layout.records:
        view = _view_for_layout_record(
            validated_storage,
            record,
        )

        if record.block_index is None:
            top_level_views[record.key] = view
        else:
            block_view_maps[record.block_index][record.key] = view

    blocks: list[TransformerBlockParameterViews] = []

    for block_view_map in block_view_maps:
        if tuple(block_view_map) != _TRANSFORMER_BLOCK_PARAMETER_KEYS:
            raise RuntimeError("canonical Transformer block view construction is incomplete")

        blocks.append(
            TransformerBlockParameterViews(
                ln1_gamma=block_view_map["ln1Gamma"],
                ln1_beta=block_view_map["ln1Beta"],
                w_q=block_view_map["wQ"],
                b_q=block_view_map["bQ"],
                w_k=block_view_map["wK"],
                b_k=block_view_map["bK"],
                w_v=block_view_map["wV"],
                b_v=block_view_map["bV"],
                w_o=block_view_map["wO"],
                b_o=block_view_map["bO"],
                ln2_gamma=block_view_map["ln2Gamma"],
                ln2_beta=block_view_map["ln2Beta"],
                ff1_w=block_view_map["ff1W"],
                ff1_b=block_view_map["ff1B"],
                ff2_w=block_view_map["ff2W"],
                ff2_b=block_view_map["ff2B"],
            )
        )

    return TransformerParameterViews(
        layout=layout,
        tok_emb=top_level_views["tokEmb"],
        pos_emb=top_level_views["posEmb"],
        blocks=tuple(blocks),
        ln_f_gamma=top_level_views["lnFGamma"],
        ln_f_beta=top_level_views["lnFBeta"],
        head_w=top_level_views["headW"],
        head_b=top_level_views["headB"],
    )


def _fill_deterministic_transformer_parameters(
    views: TransformerParameterViews,
) -> None:
    for block in views.blocks:
        block.ln1_gamma.fill(np.float32(1.0))
        block.ln2_gamma.fill(np.float32(1.0))

        block.ln1_beta.fill(np.float32(0.0))
        block.b_q.fill(np.float32(0.0))
        block.b_k.fill(np.float32(0.0))
        block.b_v.fill(np.float32(0.0))
        block.b_o.fill(np.float32(0.0))
        block.ln2_beta.fill(np.float32(0.0))
        block.ff1_b.fill(np.float32(0.0))
        block.ff2_b.fill(np.float32(0.0))

    views.ln_f_gamma.fill(np.float32(1.0))
    views.ln_f_beta.fill(np.float32(0.0))
    views.head_b.fill(np.float32(0.0))


def _fill_xavier_uniform(
    destination: _Float32Array,
    fan_in: int,
    fan_out: int,
    generator: Mulberry32,
) -> int:
    if not destination.flags.writeable:
        raise ValueError("initialization storage must be writable")

    limit = math.sqrt(6.0 / (fan_in + fan_out))
    flat_destination = destination.reshape(
        -1,
        order="C",
    )

    for index in range(flat_destination.size):
        random_value = generator.random()
        value = (random_value * 2.0 - 1.0) * limit
        flat_destination[index] = np.float32(value)

    return flat_destination.size


def _validate_initialized_transformer_storage(
    storage: _Float32Array,
    layout: TransformerParameterLayout,
) -> None:
    if storage.dtype != np.dtype(np.float32):
        raise RuntimeError("initialized Transformer storage has the wrong dtype")

    if storage.ndim != 1 or not storage.flags.c_contiguous:
        raise RuntimeError("initialized Transformer storage has an invalid layout")

    if storage.size != layout.total_float_count or storage.nbytes != layout.total_byte_count:
        raise RuntimeError("initialized Transformer storage has the wrong size")

    if not np.isfinite(storage).all():
        raise FloatingPointError("initialized Transformer storage contains a non-finite value")


def initialize_transformer_parameters(
    layout: TransformerParameterLayout,
    generator: Mulberry32,
) -> InitializedTransformerParameters:
    """Allocate and initialize one fresh finite Transformer parameter block."""
    _validate_transformer_parameter_layout(layout)

    if not isinstance(generator, Mulberry32):
        raise TypeError("generator must be a Mulberry32 instance")

    storage: _Float32Array = np.empty(
        layout.total_float_count,
        dtype=np.float32,
        order="C",
    )
    storage.fill(np.float32(np.nan))

    views = build_transformer_parameter_views(
        storage,
        layout,
    )
    starting_draw_count = generator.draw_count

    _fill_deterministic_transformer_parameters(
        views,
    )

    if generator.draw_count != starting_draw_count:
        raise RuntimeError("deterministic Transformer fills consumed random values")

    random_coordinate_count = 0
    embedding_dimension = TRANSFORMER_EMBEDDING_DIMENSION
    feed_forward_dimension = TRANSFORMER_FEED_FORWARD_DIMENSION

    for block in views.blocks:
        random_coordinate_count += _fill_xavier_uniform(
            block.w_q,
            embedding_dimension,
            embedding_dimension,
            generator,
        )
        random_coordinate_count += _fill_xavier_uniform(
            block.w_k,
            embedding_dimension,
            embedding_dimension,
            generator,
        )
        random_coordinate_count += _fill_xavier_uniform(
            block.w_v,
            embedding_dimension,
            embedding_dimension,
            generator,
        )
        random_coordinate_count += _fill_xavier_uniform(
            block.w_o,
            embedding_dimension,
            embedding_dimension,
            generator,
        )
        random_coordinate_count += _fill_xavier_uniform(
            block.ff1_w,
            embedding_dimension,
            feed_forward_dimension,
            generator,
        )
        random_coordinate_count += _fill_xavier_uniform(
            block.ff2_w,
            feed_forward_dimension,
            embedding_dimension,
            generator,
        )

    random_coordinate_count += _fill_xavier_uniform(
        views.tok_emb,
        layout.vocabulary_size,
        embedding_dimension,
        generator,
    )
    random_coordinate_count += _fill_xavier_uniform(
        views.pos_emb,
        TRANSFORMER_CONTEXT_LENGTH,
        embedding_dimension,
        generator,
    )
    random_coordinate_count += _fill_xavier_uniform(
        views.head_w,
        embedding_dimension,
        layout.vocabulary_size,
        generator,
    )

    if generator.draw_count - starting_draw_count != random_coordinate_count:
        raise RuntimeError("Transformer initialization consumed an unexpected number of draws")

    _validate_initialized_transformer_storage(
        storage,
        layout,
    )

    return InitializedTransformerParameters(
        layout=layout,
        storage=storage,
        views=views,
    )


def _validate_float32_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    allow_negative_infinity: bool = False,
    require_writable: bool = False,
) -> _Float32Array:
    """Validate one completed Transformer-owned float32 array."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a NumPy ndarray.")

    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32.")

    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")

    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")

    if require_writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable.")

    if allow_negative_infinity:
        if np.isnan(value).any() or np.isposinf(value).any():
            raise FloatingPointError(f"{name} contains an unsupported non-finite value.")
    elif not np.isfinite(value).all():
        raise FloatingPointError(f"{name} contains a non-finite value.")

    return value


def _validate_float64_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    require_writable: bool = False,
) -> _Float64Array:
    """Validate one run-owned finite C-contiguous float64 scratch array."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a NumPy ndarray.")

    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64.")

    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")

    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")

    if require_writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable.")

    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} contains a non-finite value.")

    return cast(_Float64Array, value)


def _validate_transformer_training_run_storage(
    run: TransformerTrainingRun,
) -> None:
    """Validate all persistent parent-owned numerical storage and ownership."""
    shape = (run._parameters.layout.total_float_count,)
    float32_arrays = (
        run._parameters.storage,
        run._first_moments,
        run._second_moments,
        run._reduced_gradient.storage,
    )
    float32_names = (
        "Transformer weights",
        "Transformer first moments",
        "Transformer second moments",
        "Transformer reduced gradient",
    )

    for name, array in zip(float32_names, float32_arrays, strict=True):
        _validate_float32_array(
            array,
            name=name,
            shape=shape,
            require_writable=True,
        )

    _validate_float64_array(
        run._adam_scratch_a,
        name="Transformer Adam scratch A",
        shape=shape,
        require_writable=True,
    )
    _validate_float64_array(
        run._adam_scratch_b,
        name="Transformer Adam scratch B",
        shape=shape,
        require_writable=True,
    )

    if run._reduced_gradient.layout != run._parameters.layout:
        raise ValueError("The reduction workspace layout does not match the run layout.")

    if run._reduced_gradient.views.layout != run._parameters.layout:
        raise ValueError("The reduction workspace views do not match the run layout.")

    _validate_parameter_views(run._parameters.views)
    _validate_parameter_views(run._reduced_gradient.views)

    for parameter_view in _parameter_arrays(run._parameters.views):
        if not np.shares_memory(parameter_view, run._parameters.storage):
            raise ValueError("Every run parameter view must use the run weight storage.")

    for gradient_view in _parameter_arrays(run._reduced_gradient.views):
        if not np.shares_memory(gradient_view, run._reduced_gradient.storage):
            raise ValueError("Every reduction view must use the reduction workspace.")

    owned_arrays: tuple[npt.NDArray[np.generic], ...] = (
        *float32_arrays,
        run._adam_scratch_a,
        run._adam_scratch_b,
    )

    for left_index, left in enumerate(owned_arrays):
        for right in owned_arrays[left_index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError("Transformer Training Run storage must not overlap.")

    if np.any(run._second_moments < np.float32(0.0)):
        raise FloatingPointError("Transformer second moments cannot be negative.")


def _canonicalize_logical_training_shard_results(
    run: TransformerTrainingRun,
    shard_results: Collection[LogicalTrainingShardResult],
) -> tuple[
    LogicalTrainingShardResult,
    LogicalTrainingShardResult,
    LogicalTrainingShardResult,
    LogicalTrainingShardResult,
]:
    """Validate a complete result set and return exact shard order zero through three."""
    if not isinstance(shard_results, Collection):
        raise TypeError("shard_results must be a finite collection.")

    if len(shard_results) != LOGICAL_TRAINING_SHARD_COUNT:
        raise ValueError("Exactly four Logical Training Shard results are required.")

    by_shard_index: dict[int, LogicalTrainingShardResult] = {}
    gradient_storages: list[_Float32Array] = []
    parent_arrays: tuple[npt.NDArray[np.generic], ...] = (
        run._parameters.storage,
        run._first_moments,
        run._second_moments,
        run._reduced_gradient.storage,
        run._adam_scratch_a,
        run._adam_scratch_b,
    )
    expected_shape = (run._parameters.layout.total_float_count,)

    for result in shard_results:
        if type(result) is not LogicalTrainingShardResult:
            raise TypeError("Every shard_results entry must be a LogicalTrainingShardResult.")

        if type(result.shard) is not LogicalTrainingShard:
            raise TypeError("Every result shard must be a LogicalTrainingShard.")

        shard_index = result.shard.shard_index

        if type(shard_index) is not int:
            raise TypeError("Every shard index must be an integer.")

        if shard_index < 0 or shard_index >= LOGICAL_TRAINING_SHARD_COUNT:
            raise ValueError("Every shard index must be between zero and three.")

        if shard_index in by_shard_index:
            raise ValueError("Every shard index must occur exactly once.")

        expected_shard = run._logical_training_shards[shard_index]

        if result.shard != expected_shard:
            raise ValueError("Shard metadata does not match the run boundary.")

        if result.shard.start_index > result.shard.stop_index:
            raise ValueError("A shard start index cannot exceed its stop index.")

        expected_sequence_count = expected_shard.stop_index - expected_shard.start_index

        if type(result.processed_sequence_count) is not int:
            raise TypeError("processed_sequence_count must be an integer.")

        if result.processed_sequence_count != expected_sequence_count:
            raise ValueError("processed_sequence_count does not match the represented shard range.")

        if isinstance(result.loss, (bool, np.bool_)) or not isinstance(
            result.loss,
            Real,
        ):
            raise TypeError("Every shard loss must be a real Python numeric value.")

        loss = float(result.loss)

        if not math.isfinite(loss):
            raise FloatingPointError("Every shard loss must be finite.")

        if type(result.gradient) is not TransformerGradientBuffer:
            raise TypeError("Every result gradient must be a TransformerGradientBuffer.")

        if result.gradient.layout != run._parameters.layout:
            raise ValueError("Every shard gradient must use the run's canonical layout.")

        if result.gradient.views.layout != result.gradient.layout:
            raise ValueError("Shard gradient views do not match their layout.")

        gradient_storage = _validate_float32_array(
            result.gradient.storage,
            name=f"Logical Training Shard {shard_index} gradient",
            shape=expected_shape,
        )
        _validate_parameter_views(result.gradient.views)

        for semantic_view in _parameter_arrays(result.gradient.views):
            if not np.shares_memory(semantic_view, gradient_storage):
                raise ValueError("Every shard gradient view must use its gradient storage.")

        for parent_array in parent_arrays:
            if np.shares_memory(gradient_storage, parent_array):
                raise ValueError("Shard gradients must not overlap parent-owned storage.")

        for previous_gradient in gradient_storages:
            if np.shares_memory(gradient_storage, previous_gradient):
                raise ValueError("Distinct shard gradients must not overlap.")

        if expected_sequence_count == 0:
            if loss != 0.0:
                raise ValueError("An empty shard must have exact zero loss.")

            if np.any(gradient_storage != np.float32(0.0)):
                raise ValueError("An empty shard must have an all-zero gradient.")

        by_shard_index[shard_index] = result
        gradient_storages.append(gradient_storage)

    return (
        by_shard_index[0],
        by_shard_index[1],
        by_shard_index[2],
        by_shard_index[3],
    )


def _reduce_logical_training_shard_results(
    run: TransformerTrainingRun,
    shard_results: Collection[LogicalTrainingShardResult],
) -> float:
    """Reduce exactly four validated results in canonical shard order."""
    canonical_results = _canonicalize_logical_training_shard_results(
        run,
        shard_results,
    )
    workspace = run._reduced_gradient.storage
    workspace.fill(np.float32(0.0))
    reduced_loss = 0.0

    with np.errstate(over="raise", invalid="raise"):
        for result in canonical_results:
            reduced_loss += float(result.loss)

            if not math.isfinite(reduced_loss):
                raise FloatingPointError("Ordered reduced loss is not finite.")

            np.add(
                workspace,
                result.gradient.storage,
                out=workspace,
            )

            if not np.isfinite(workspace).all():
                raise FloatingPointError("Ordered reduced gradient is not finite.")

    if not math.isfinite(reduced_loss):
        raise FloatingPointError("Ordered reduced loss is not finite.")

    if not np.isfinite(workspace).all():
        raise FloatingPointError("Ordered reduced gradient is not finite.")

    return reduced_loss


def _build_transformer_epoch_update(
    epoch: int,
    loss: float,
) -> TransformerEpochUpdate:
    """Create one six-decimal report-ready Transformer Epoch Update."""
    if type(epoch) is not int:
        raise TypeError("epoch must be an integer.")

    if epoch < 0:
        raise ValueError("epoch must be non-negative.")

    if isinstance(loss, (bool, np.bool_)) or not isinstance(loss, Real):
        raise TypeError("loss must be a real Python numeric value.")

    internal_loss = float(loss)

    if not math.isfinite(internal_loss):
        raise FloatingPointError("Transformer epoch loss is not finite.")

    public_loss = round_typescript_decimal(internal_loss, 6)

    if not math.isfinite(public_loss):
        raise FloatingPointError("Transformer public epoch loss is not finite.")

    return TransformerEpochUpdate(
        epoch=epoch,
        loss=public_loss,
    )


def _apply_transformer_adam_update(
    run: TransformerTrainingRun,
    *,
    epoch: int,
) -> None:
    """Calculate all Adam candidates and commit persistent state once."""
    if type(epoch) is not int:
        raise TypeError("epoch must be an integer.")

    if epoch < 0:
        raise ValueError("epoch must be non-negative.")

    if epoch != run._next_epoch:
        raise ValueError("epoch does not match the run's next expected epoch.")

    step = epoch + 1

    if step <= 0:
        raise ValueError("Adam step must be strictly positive.")

    _validate_transformer_training_run_storage(run)

    first_bias_denominator = 1.0 - (_TRANSFORMER_ADAM_BETA1**step)
    second_bias_denominator = 1.0 - (_TRANSFORMER_ADAM_BETA2**step)

    if (
        not math.isfinite(first_bias_denominator)
        or first_bias_denominator <= 0.0
        or not math.isfinite(second_bias_denominator)
        or second_bias_denominator <= 0.0
    ):
        raise FloatingPointError("Adam bias-correction denominators are invalid.")

    scratch_a = run._adam_scratch_a
    scratch_b = run._adam_scratch_b
    reduced_gradient = run._reduced_gradient.storage

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        # Complete first-moment candidate in float64 scratch A.
        np.copyto(
            scratch_a,
            run._first_moments,
            casting="safe",
        )
        scratch_a *= _TRANSFORMER_ADAM_BETA1

        np.copyto(
            scratch_b,
            reduced_gradient,
            casting="safe",
        )
        scratch_b *= 1.0 - _TRANSFORMER_ADAM_BETA1

        np.add(
            scratch_a,
            scratch_b,
            out=scratch_a,
        )

        # Materialize the completed first moment once as float32.
        first_moment_candidate = _materialize_float32(
            scratch_a,
            name="Transformer first-moment candidate",
        )
        _validate_float32_array(
            first_moment_candidate,
            name="Transformer first-moment candidate",
            shape=run._first_moments.shape,
        )

        # Complete second-moment candidate in the same two scratches.
        np.copyto(
            scratch_a,
            reduced_gradient,
            casting="safe",
        )
        np.square(
            scratch_a,
            out=scratch_a,
        )
        scratch_a *= 1.0 - _TRANSFORMER_ADAM_BETA2

        np.copyto(
            scratch_b,
            run._second_moments,
            casting="safe",
        )
        scratch_b *= _TRANSFORMER_ADAM_BETA2

        np.add(
            scratch_b,
            scratch_a,
            out=scratch_b,
        )

        # Materialize the completed second moment once as float32.
        second_moment_candidate = _materialize_float32(
            scratch_b,
            name="Transformer second-moment candidate",
        )
        _validate_float32_array(
            second_moment_candidate,
            name="Transformer second-moment candidate",
            shape=run._second_moments.shape,
        )

        if np.any(second_moment_candidate < np.float32(0.0)):
            raise FloatingPointError("Transformer second-moment candidate cannot be negative.")

        # Bias correction consumes the completed float32 candidates.
        np.copyto(
            scratch_a,
            first_moment_candidate,
            casting="safe",
        )
        scratch_a /= first_bias_denominator

        np.copyto(
            scratch_b,
            second_moment_candidate,
            casting="safe",
        )
        scratch_b /= second_bias_denominator

        if np.any(scratch_b < 0.0):
            raise FloatingPointError(
                "Bias-corrected Transformer second moments cannot be negative."
            )

        np.sqrt(
            scratch_b,
            out=scratch_b,
        )
        scratch_b += _TRANSFORMER_ADAM_EPSILON

        if not np.isfinite(scratch_b).all() or np.any(scratch_b <= 0.0):
            raise FloatingPointError("Transformer Adam denominator is invalid.")

        # Scratch A becomes the complete parameter delta.
        np.divide(
            scratch_a,
            scratch_b,
            out=scratch_a,
        )
        scratch_a *= _TRANSFORMER_ADAM_LEARNING_RATE

        # Scratch B becomes the complete weight candidate.
        np.copyto(
            scratch_b,
            run._parameters.storage,
            casting="safe",
        )
        np.subtract(
            scratch_b,
            scratch_a,
            out=scratch_b,
        )

        weight_candidate = _materialize_float32(
            scratch_b,
            name="Transformer weight candidate",
        )
        _validate_float32_array(
            weight_candidate,
            name="Transformer weight candidate",
            shape=run._parameters.storage.shape,
        )

    # No persistent state is modified before this final commit boundary.
    np.copyto(
        run._first_moments,
        first_moment_candidate,
        casting="no",
    )
    np.copyto(
        run._second_moments,
        second_moment_candidate,
        casting="no",
    )
    np.copyto(
        run._parameters.storage,
        weight_candidate,
        casting="no",
    )


def build_transformer_report_epochs(
    requested_epochs: int,
) -> tuple[int, ...]:
    """Return the exact unique Transformer report schedule."""
    if type(requested_epochs) is not int:
        raise TypeError("requested_epochs must be an integer.")

    if requested_epochs < 0:
        raise ValueError("requested_epochs must be non-negative.")

    report_step = max(1, requested_epochs // 50)

    return tuple(
        epoch
        for epoch in range(requested_epochs + 1)
        if epoch % report_step == 0 or epoch == requested_epochs
    )


def create_transformer_training_run(
    initialized_parameters: InitializedTransformerParameters,
    *,
    sequence_count: int,
    requested_epochs: int,
) -> TransformerTrainingRun:
    """Create one fresh independent parent-owned Transformer Training Run."""
    return TransformerTrainingRun(
        initialized_parameters,
        sequence_count=sequence_count,
        requested_epochs=requested_epochs,
    )


def _materialize_float32(
    value: npt.ArrayLike,
    *,
    name: str,
    allow_negative_infinity: bool = False,
) -> _Float32Array:
    """Materialize one new independent C-contiguous float32 array."""
    result = np.array(
        value,
        dtype=np.float32,
        order="C",
        copy=True,
    )

    if allow_negative_infinity:
        if np.isnan(result).any() or np.isposinf(result).any():
            raise FloatingPointError(f"{name} contains an unsupported non-finite value.")
    elif not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains a non-finite value.")

    return result


def _validate_token_ids(
    token_ids: object,
    *,
    name: str,
    vocabulary_size: int,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    """Validate one immutable ordered token-ID sequence."""
    if type(token_ids) is not tuple:
        raise TypeError(f"{name} must be a tuple.")

    if not token_ids:
        raise ValueError(f"{name} must contain at least one token ID.")

    if len(token_ids) > TRANSFORMER_CONTEXT_LENGTH:
        raise ValueError(f"{name} cannot exceed {TRANSFORMER_CONTEXT_LENGTH} positions.")

    if expected_length is not None and len(token_ids) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} IDs.")

    validated: list[int] = []

    for token_id in token_ids:
        if type(token_id) is not int:
            raise TypeError(f"Every {name} entry must be an integer.")

        if token_id < 0 or token_id >= vocabulary_size:
            raise ValueError(f"Every {name} entry must be between 0 and {vocabulary_size - 1}.")

        validated.append(token_id)

    return tuple(validated)


def _parameter_arrays(
    parameters: TransformerParameterViews,
) -> tuple[_Float32Array, ...]:
    """Return every parameter array in canonical semantic order."""
    arrays: list[_Float32Array] = [
        parameters.tok_emb,
        parameters.pos_emb,
    ]

    for block in parameters.blocks:
        arrays.extend(
            (
                block.ln1_gamma,
                block.ln1_beta,
                block.w_q,
                block.b_q,
                block.w_k,
                block.b_k,
                block.w_v,
                block.b_v,
                block.w_o,
                block.b_o,
                block.ln2_gamma,
                block.ln2_beta,
                block.ff1_w,
                block.ff1_b,
                block.ff2_w,
                block.ff2_b,
            )
        )

    arrays.extend(
        (
            parameters.ln_f_gamma,
            parameters.ln_f_beta,
            parameters.head_w,
            parameters.head_b,
        )
    )

    return tuple(arrays)


def _validate_parameter_views(
    parameters: TransformerParameterViews,
) -> int:
    """Validate the complete semantic Transformer parameter boundary."""
    if type(parameters) is not TransformerParameterViews:
        raise TypeError("parameters must be TransformerParameterViews.")

    if len(parameters.blocks) != parameters.layout.num_layers:
        raise ValueError("parameters.blocks must match the canonical layout layer count.")

    vocabulary_size = int(parameters.tok_emb.shape[0])

    if vocabulary_size <= 0:
        raise ValueError("The Transformer Vocabulary cannot be empty.")

    _validate_float32_array(
        parameters.tok_emb,
        name="parameters.tok_emb",
        shape=(
            vocabulary_size,
            TRANSFORMER_EMBEDDING_DIMENSION,
        ),
    )
    _validate_float32_array(
        parameters.pos_emb,
        name="parameters.pos_emb",
        shape=(
            TRANSFORMER_CONTEXT_LENGTH,
            TRANSFORMER_EMBEDDING_DIMENSION,
        ),
    )

    for block_index, block in enumerate(parameters.blocks):
        vector_shape = (TRANSFORMER_EMBEDDING_DIMENSION,)
        matrix_shape = (
            TRANSFORMER_EMBEDDING_DIMENSION,
            TRANSFORMER_EMBEDDING_DIMENSION,
        )

        _validate_float32_array(
            block.ln1_gamma,
            name=f"parameters.blocks[{block_index}].ln1_gamma",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.ln1_beta,
            name=f"parameters.blocks[{block_index}].ln1_beta",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.w_q,
            name=f"parameters.blocks[{block_index}].w_q",
            shape=matrix_shape,
        )
        _validate_float32_array(
            block.b_q,
            name=f"parameters.blocks[{block_index}].b_q",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.w_k,
            name=f"parameters.blocks[{block_index}].w_k",
            shape=matrix_shape,
        )
        _validate_float32_array(
            block.b_k,
            name=f"parameters.blocks[{block_index}].b_k",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.w_v,
            name=f"parameters.blocks[{block_index}].w_v",
            shape=matrix_shape,
        )
        _validate_float32_array(
            block.b_v,
            name=f"parameters.blocks[{block_index}].b_v",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.w_o,
            name=f"parameters.blocks[{block_index}].w_o",
            shape=matrix_shape,
        )
        _validate_float32_array(
            block.b_o,
            name=f"parameters.blocks[{block_index}].b_o",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.ln2_gamma,
            name=f"parameters.blocks[{block_index}].ln2_gamma",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.ln2_beta,
            name=f"parameters.blocks[{block_index}].ln2_beta",
            shape=vector_shape,
        )
        _validate_float32_array(
            block.ff1_w,
            name=f"parameters.blocks[{block_index}].ff1_w",
            shape=(
                TRANSFORMER_EMBEDDING_DIMENSION,
                TRANSFORMER_FEED_FORWARD_DIMENSION,
            ),
        )
        _validate_float32_array(
            block.ff1_b,
            name=f"parameters.blocks[{block_index}].ff1_b",
            shape=(TRANSFORMER_FEED_FORWARD_DIMENSION,),
        )
        _validate_float32_array(
            block.ff2_w,
            name=f"parameters.blocks[{block_index}].ff2_w",
            shape=(
                TRANSFORMER_FEED_FORWARD_DIMENSION,
                TRANSFORMER_EMBEDDING_DIMENSION,
            ),
        )
        _validate_float32_array(
            block.ff2_b,
            name=f"parameters.blocks[{block_index}].ff2_b",
            shape=vector_shape,
        )

    _validate_float32_array(
        parameters.ln_f_gamma,
        name="parameters.ln_f_gamma",
        shape=(TRANSFORMER_EMBEDDING_DIMENSION,),
    )
    _validate_float32_array(
        parameters.ln_f_beta,
        name="parameters.ln_f_beta",
        shape=(TRANSFORMER_EMBEDDING_DIMENSION,),
    )
    _validate_float32_array(
        parameters.head_w,
        name="parameters.head_w",
        shape=(
            TRANSFORMER_EMBEDDING_DIMENSION,
            vocabulary_size,
        ),
    )
    _validate_float32_array(
        parameters.head_b,
        name="parameters.head_b",
        shape=(vocabulary_size,),
    )

    return vocabulary_size


def _add_matrices(
    left: _Float32Array,
    right: _Float32Array,
    *,
    name: str,
) -> _Float32Array:
    """Add two exact-shape matrices through float64 scratch precision."""
    if left.shape != right.shape:
        raise ValueError(f"{name} operands must have identical shapes.")

    candidate = left.astype(np.float64) + right.astype(np.float64)

    return _materialize_float32(
        candidate,
        name=name,
    )


def _add_vector_bias(
    matrix: _Float32Array,
    bias: _Float32Array,
    *,
    name: str,
) -> _Float32Array:
    """Add one exact column vector bias to every matrix row."""
    if matrix.ndim != 2:
        raise ValueError(f"{name} matrix must have rank two.")

    if bias.shape != (matrix.shape[1],):
        raise ValueError(f"{name} bias must have shape ({matrix.shape[1]},).")

    candidate = matrix.astype(np.float64) + bias.astype(np.float64)[np.newaxis, :]

    return _materialize_float32(
        candidate,
        name=name,
    )


def _sum_rows(
    matrix: _Float32Array,
    *,
    name: str,
) -> _Float32Array:
    """Sum matrix rows into one float32 parameter-gradient vector."""
    candidate = np.sum(
        matrix,
        axis=0,
        dtype=np.float64,
    )

    return _materialize_float32(
        candidate,
        name=name,
    )


def _copy_column_slice(
    matrix: _Float32Array,
    start: int,
    stop: int,
    *,
    name: str,
) -> _Float32Array:
    """Return one independent C-contiguous feature-column slice."""
    return _materialize_float32(
        matrix[:, start:stop],
        name=name,
    )


def _embedding_forward(
    input_ids: tuple[int, ...],
    parameters: TransformerParameterViews,
) -> _Float32Array:
    """Combine learned token and absolute positional embeddings."""
    candidate = np.empty(
        (
            len(input_ids),
            TRANSFORMER_EMBEDDING_DIMENSION,
        ),
        dtype=np.float64,
        order="C",
    )

    for position, token_id in enumerate(input_ids):
        candidate[position, :] = parameters.tok_emb[token_id, :].astype(
            np.float64
        ) + parameters.pos_emb[position, :].astype(np.float64)

    return _materialize_float32(
        candidate,
        name="embedding activation",
    )


def _layer_normalization_forward(
    source: _Float32Array,
    gamma: _Float32Array,
    beta: _Float32Array,
    *,
    name: str,
) -> TransformerLayerNormalizationCache:
    """Apply per-position population-variance Layer Normalization."""
    sequence_length = source.shape[0]

    _validate_float32_array(
        source,
        name=f"{name} source",
        shape=(
            sequence_length,
            TRANSFORMER_EMBEDDING_DIMENSION,
        ),
    )
    _validate_float32_array(
        gamma,
        name=f"{name} gamma",
        shape=(TRANSFORMER_EMBEDDING_DIMENSION,),
    )
    _validate_float32_array(
        beta,
        name=f"{name} beta",
        shape=(TRANSFORMER_EMBEDDING_DIMENSION,),
    )

    source_float64 = source.astype(np.float64)
    means_float64 = np.mean(
        source_float64,
        axis=1,
        dtype=np.float64,
    )
    centered_float64 = source_float64 - means_float64[:, np.newaxis]
    variances_float64 = np.mean(
        centered_float64 * centered_float64,
        axis=1,
        dtype=np.float64,
    )
    inverse_standard_deviations_float64 = 1.0 / np.sqrt(
        variances_float64 + TRANSFORMER_LAYER_NORMALIZATION_EPSILON
    )
    normalized_float64 = centered_float64 * inverse_standard_deviations_float64[:, np.newaxis]
    output_float64 = (
        normalized_float64 * gamma.astype(np.float64)[np.newaxis, :]
        + beta.astype(np.float64)[np.newaxis, :]
    )

    return TransformerLayerNormalizationCache(
        means=_materialize_float32(
            means_float64,
            name=f"{name} means",
        ),
        variances=_materialize_float32(
            variances_float64,
            name=f"{name} variances",
        ),
        inverse_standard_deviations=_materialize_float32(
            inverse_standard_deviations_float64,
            name=f"{name} inverse standard deviations",
        ),
        normalized=_materialize_float32(
            normalized_float64,
            name=f"{name} normalized activation",
        ),
        output=_materialize_float32(
            output_float64,
            name=f"{name} output",
        ),
    )


def _layer_normalization_backward(
    output_gradient: _Float32Array,
    cache: TransformerLayerNormalizationCache,
    gamma: _Float32Array,
    *,
    name: str,
) -> _LayerNormalizationBackwardResult:
    """Apply the analytical per-position Layer Normalization reverse path."""
    sequence_length = output_gradient.shape[0]
    feature_count = TRANSFORMER_EMBEDDING_DIMENSION

    _validate_float32_array(
        output_gradient,
        name=f"{name} output gradient",
        shape=(sequence_length, feature_count),
    )
    _validate_float32_array(
        cache.normalized,
        name=f"{name} cached normalized activation",
        shape=(sequence_length, feature_count),
    )
    _validate_float32_array(
        cache.inverse_standard_deviations,
        name=f"{name} cached inverse standard deviations",
        shape=(sequence_length,),
    )
    _validate_float32_array(
        gamma,
        name=f"{name} gamma",
        shape=(feature_count,),
    )

    output_gradient_float64 = output_gradient.astype(np.float64)
    normalized_float64 = cache.normalized.astype(np.float64)
    gamma_float64 = gamma.astype(np.float64)
    inverse_standard_deviations_float64 = cache.inverse_standard_deviations.astype(np.float64)

    normalized_gradient_float64 = output_gradient_float64 * gamma_float64[np.newaxis, :]
    row_gradient_sums = np.sum(
        normalized_gradient_float64,
        axis=1,
        dtype=np.float64,
    )
    row_gradient_normalized_sums = np.sum(
        normalized_gradient_float64 * normalized_float64,
        axis=1,
        dtype=np.float64,
    )

    input_gradient_float64 = (
        inverse_standard_deviations_float64[:, np.newaxis] / float(feature_count)
    ) * (
        float(feature_count) * normalized_gradient_float64
        - row_gradient_sums[:, np.newaxis]
        - normalized_float64 * row_gradient_normalized_sums[:, np.newaxis]
    )

    gamma_gradient_float64 = np.sum(
        output_gradient_float64 * normalized_float64,
        axis=0,
        dtype=np.float64,
    )
    beta_gradient_float64 = np.sum(
        output_gradient_float64,
        axis=0,
        dtype=np.float64,
    )

    return _LayerNormalizationBackwardResult(
        input_gradient=_materialize_float32(
            input_gradient_float64,
            name=f"{name} input gradient",
        ),
        gamma_gradient=_materialize_float32(
            gamma_gradient_float64,
            name=f"{name} gamma gradient",
        ),
        beta_gradient=_materialize_float32(
            beta_gradient_float64,
            name=f"{name} beta gradient",
        ),
    )


def _linear_forward(
    source: _Float32Array,
    weight: _Float32Array,
    bias: _Float32Array,
    *,
    name: str,
) -> _Float32Array:
    """Apply one matrix projection followed by an exact vector bias."""
    projected = matmul(
        source,
        weight,
    )

    return _add_vector_bias(
        projected,
        bias,
        name=name,
    )


def _forward_transformer_block(
    source: _Float32Array,
    parameters: TransformerBlockParameterViews,
    *,
    block_index: int,
) -> TransformerBlockForwardCache:
    """Calculate one pre-normalized causal Transformer block."""
    first_normalization = _layer_normalization_forward(
        source,
        parameters.ln1_gamma,
        parameters.ln1_beta,
        name=f"block {block_index} first normalization",
    )

    query = _linear_forward(
        first_normalization.output,
        parameters.w_q,
        parameters.b_q,
        name=f"block {block_index} query",
    )
    key = _linear_forward(
        first_normalization.output,
        parameters.w_k,
        parameters.b_k,
        name=f"block {block_index} key",
    )
    value = _linear_forward(
        first_normalization.output,
        parameters.w_v,
        parameters.b_v,
        name=f"block {block_index} value",
    )

    sequence_length = source.shape[0]
    attention_heads: list[TransformerAttentionHeadCache] = []
    weighted_heads: list[_Float32Array] = []

    for head_index in range(TRANSFORMER_ATTENTION_HEAD_COUNT):
        feature_start = head_index * TRANSFORMER_HEAD_DIMENSION
        feature_stop = feature_start + TRANSFORMER_HEAD_DIMENSION

        query_head = _copy_column_slice(
            query,
            feature_start,
            feature_stop,
            name=f"block {block_index} head {head_index} query",
        )
        key_head = _copy_column_slice(
            key,
            feature_start,
            feature_stop,
            name=f"block {block_index} head {head_index} key",
        )
        value_head = _copy_column_slice(
            value,
            feature_start,
            feature_stop,
            name=f"block {block_index} head {head_index} value",
        )

        scores = scalar_multiply(
            matmul_transposed_right(
                query_head,
                key_head,
            ),
            TRANSFORMER_ATTENTION_SCALE,
        )

        future_mask = np.triu(
            np.ones(
                (
                    sequence_length,
                    sequence_length,
                ),
                dtype=np.bool_,
            ),
            k=1,
        )
        scores[future_mask] = -np.inf

        _validate_float32_array(
            scores,
            name=f"block {block_index} head {head_index} scores",
            shape=(sequence_length, sequence_length),
            allow_negative_infinity=True,
        )

        probabilities = stable_row_softmax(scores)

        if not np.array_equal(
            probabilities[future_mask],
            np.zeros(
                int(np.count_nonzero(future_mask)),
                dtype=np.float32,
            ),
        ):
            raise FloatingPointError("Causal attention produced a nonzero future probability.")

        weighted_values = matmul(
            probabilities,
            value_head,
        )

        attention_heads.append(
            TransformerAttentionHeadCache(
                scores=scores,
                probabilities=probabilities,
                weighted_values=weighted_values,
            )
        )
        weighted_heads.append(weighted_values)

    concatenated_attention = _materialize_float32(
        np.concatenate(
            weighted_heads,
            axis=1,
        ),
        name=f"block {block_index} concatenated attention",
    )
    projected_attention = _linear_forward(
        concatenated_attention,
        parameters.w_o,
        parameters.b_o,
        name=f"block {block_index} projected attention",
    )
    first_residual = _add_matrices(
        source,
        projected_attention,
        name=f"block {block_index} first residual",
    )

    second_normalization = _layer_normalization_forward(
        first_residual,
        parameters.ln2_gamma,
        parameters.ln2_beta,
        name=f"block {block_index} second normalization",
    )
    feed_forward_pre_activation = _linear_forward(
        second_normalization.output,
        parameters.ff1_w,
        parameters.ff1_b,
        name=f"block {block_index} feed-forward first projection",
    )
    feed_forward_activation = _materialize_float32(
        np.maximum(
            feed_forward_pre_activation,
            np.float32(0.0),
        ),
        name=f"block {block_index} feed-forward ReLU",
    )
    feed_forward_output = _linear_forward(
        feed_forward_activation,
        parameters.ff2_w,
        parameters.ff2_b,
        name=f"block {block_index} feed-forward second projection",
    )
    output = _add_matrices(
        first_residual,
        feed_forward_output,
        name=f"block {block_index} second residual",
    )

    return TransformerBlockForwardCache(
        input_activation=_materialize_float32(
            source,
            name=f"block {block_index} cached input",
        ),
        first_normalization=first_normalization,
        query=query,
        key=key,
        value=value,
        attention_heads=tuple(attention_heads),
        concatenated_attention=concatenated_attention,
        projected_attention=projected_attention,
        first_residual=first_residual,
        second_normalization=second_normalization,
        feed_forward_pre_activation=feed_forward_pre_activation,
        feed_forward_activation=feed_forward_activation,
        feed_forward_output=feed_forward_output,
        output=output,
    )


def _validate_forward_result(
    result: TransformerForwardResult,
    *,
    vocabulary_size: int,
    expected_block_count: int,
) -> None:
    """Validate one completed public forward result before backward use."""
    if type(result) is not TransformerForwardResult:
        raise TypeError("forward must be a TransformerForwardResult.")

    sequence_length = len(result.input_ids)

    _validate_token_ids(
        result.input_ids,
        name="forward.input_ids",
        vocabulary_size=vocabulary_size,
    )
    _validate_float32_array(
        result.embedding_activation,
        name="forward.embedding_activation",
        shape=(
            sequence_length,
            TRANSFORMER_EMBEDDING_DIMENSION,
        ),
    )

    if len(result.blocks) != expected_block_count:
        raise ValueError("forward.blocks must match the parameter layer count.")

    if not result.blocks:
        raise ValueError("forward.blocks cannot be empty.")

    for block_index, block in enumerate(result.blocks):
        if len(block.attention_heads) != TRANSFORMER_ATTENTION_HEAD_COUNT:
            raise ValueError(
                f"forward.blocks[{block_index}] must contain exactly "
                f"{TRANSFORMER_ATTENTION_HEAD_COUNT} attention heads."
            )

        for head_index, head in enumerate(block.attention_heads):
            _validate_float32_array(
                head.scores,
                name=(f"forward.blocks[{block_index}].attention_heads[{head_index}].scores"),
                shape=(sequence_length, sequence_length),
                allow_negative_infinity=True,
            )
            _validate_float32_array(
                head.probabilities,
                name=(f"forward.blocks[{block_index}].attention_heads[{head_index}].probabilities"),
                shape=(sequence_length, sequence_length),
            )

            future_mask = np.triu(
                np.ones(
                    (
                        sequence_length,
                        sequence_length,
                    ),
                    dtype=np.bool_,
                ),
                k=1,
            )

            if not np.isneginf(head.scores[future_mask]).all():
                raise ValueError("Every future attention score must be negative infinity.")

            if np.any(head.probabilities[future_mask] != np.float32(0.0)):
                raise ValueError("Every future attention probability must be exactly zero.")

    _validate_float32_array(
        result.final_normalization.output,
        name="forward.final_normalization.output",
        shape=(
            sequence_length,
            TRANSFORMER_EMBEDDING_DIMENSION,
        ),
    )
    _validate_float32_array(
        result.logits,
        name="forward.logits",
        shape=(sequence_length, vocabulary_size),
    )
    _validate_float32_array(
        result.probabilities,
        name="forward.probabilities",
        shape=(sequence_length, vocabulary_size),
    )


def calculate_transformer_forward(
    input_ids: tuple[int, ...],
    parameters: TransformerParameterViews,
) -> TransformerForwardResult:
    """Calculate one complete decoder-only Transformer forward pass."""
    vocabulary_size = _validate_parameter_views(parameters)
    validated_input_ids = _validate_token_ids(
        input_ids,
        name="input_ids",
        vocabulary_size=vocabulary_size,
    )

    activation = _embedding_forward(
        validated_input_ids,
        parameters,
    )
    embedding_activation = activation
    block_caches: list[TransformerBlockForwardCache] = []

    for block_index, block_parameters in enumerate(parameters.blocks):
        block_cache = _forward_transformer_block(
            activation,
            block_parameters,
            block_index=block_index,
        )
        block_caches.append(block_cache)
        activation = block_cache.output

    final_normalization = _layer_normalization_forward(
        activation,
        parameters.ln_f_gamma,
        parameters.ln_f_beta,
        name="final normalization",
    )
    logits = _linear_forward(
        final_normalization.output,
        parameters.head_w,
        parameters.head_b,
        name="Vocabulary logits",
    )
    probabilities = stable_row_softmax(logits)

    result = TransformerForwardResult(
        input_ids=validated_input_ids,
        embedding_activation=embedding_activation,
        blocks=tuple(block_caches),
        final_normalization=final_normalization,
        logits=logits,
        probabilities=probabilities,
    )

    _validate_forward_result(
        result,
        vocabulary_size=vocabulary_size,
        expected_block_count=len(parameters.blocks),
    )

    return result


def calculate_transformer_cross_entropy(
    probabilities: _Float32Array,
    target_ids: tuple[int, ...],
) -> float:
    """Calculate average next-token cross-entropy for one sequence."""
    if type(probabilities) is not np.ndarray:
        raise TypeError("probabilities must be a NumPy ndarray.")

    if probabilities.dtype != np.dtype(np.float32):
        raise TypeError("probabilities must have dtype float32.")

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have rank two.")

    sequence_length, vocabulary_size = probabilities.shape

    if sequence_length <= 0 or vocabulary_size <= 0:
        raise ValueError("probabilities cannot have an empty dimension.")

    _validate_float32_array(
        probabilities,
        name="probabilities",
        shape=(sequence_length, vocabulary_size),
    )
    validated_target_ids = _validate_token_ids(
        target_ids,
        name="target_ids",
        vocabulary_size=vocabulary_size,
        expected_length=sequence_length,
    )

    row_sums = np.sum(
        probabilities,
        axis=1,
        dtype=np.float64,
    )

    if not np.allclose(
        row_sums,
        np.ones(sequence_length, dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    ):
        raise FloatingPointError("Every probability row must sum to one.")

    if np.any(probabilities < np.float32(0.0)):
        raise FloatingPointError("Probabilities cannot contain negative values.")

    accumulated_loss = 0.0

    for position, target_id in enumerate(validated_target_ids):
        target_probability = float(probabilities[position, target_id])
        accumulated_loss += -math.log(target_probability + TRANSFORMER_CROSS_ENTROPY_EPSILON)

    average_loss = accumulated_loss / float(sequence_length)

    if not math.isfinite(average_loss):
        raise FloatingPointError("Transformer cross-entropy loss is not finite.")

    return average_loss


def create_transformer_gradient_buffer(
    layout: TransformerParameterLayout,
) -> TransformerGradientBuffer:
    """Allocate one fresh all-zero canonical Transformer gradient."""
    if type(layout) is not TransformerParameterLayout:
        raise TypeError("layout must be a TransformerParameterLayout.")

    storage = np.zeros(
        layout.total_float_count,
        dtype=np.float32,
        order="C",
    )
    views = build_transformer_parameter_views(
        storage,
        layout,
    )

    return TransformerGradientBuffer(
        layout=layout,
        storage=cast(_Float32Array, storage),
        views=views,
    )


def _accumulate_array_in_place(
    destination: _Float32Array,
    contribution: _Float32Array,
    *,
    name: str,
) -> None:
    """Transactionally accumulate one same-shape float32 contribution."""
    if destination.shape != contribution.shape:
        raise ValueError(f"{name} destination and contribution shapes must match.")

    if not destination.flags.writeable:
        raise ValueError(f"{name} destination must be writable.")

    candidate = _materialize_float32(
        destination.astype(np.float64) + contribution.astype(np.float64),
        name=f"{name} candidate",
    )

    destination[...] = candidate


def _causal_softmax_backward(
    probability_gradient: _Float32Array,
    probabilities: _Float32Array,
    *,
    name: str,
) -> _Float32Array:
    """Reverse causal row softmax while retaining exact future zeros."""
    if probability_gradient.shape != probabilities.shape:
        raise ValueError(f"{name} gradient and probability shapes must match.")

    sequence_length = probabilities.shape[0]
    score_gradient = np.zeros(
        probabilities.shape,
        dtype=np.float32,
        order="C",
    )

    for row_index in range(sequence_length):
        probability_dot_gradient = 0.0

        for column_index in range(row_index + 1):
            probability_dot_gradient += float(probabilities[row_index, column_index]) * float(
                probability_gradient[
                    row_index,
                    column_index,
                ]
            )

        for column_index in range(row_index + 1):
            score_gradient[row_index, column_index] = np.float32(
                float(probabilities[row_index, column_index])
                * (
                    float(
                        probability_gradient[
                            row_index,
                            column_index,
                        ]
                    )
                    - probability_dot_gradient
                )
            )

    if not np.isfinite(score_gradient).all():
        raise FloatingPointError(f"{name} is not finite.")

    future_mask = np.triu(
        np.ones(
            (
                sequence_length,
                sequence_length,
            ),
            dtype=np.bool_,
        ),
        k=1,
    )

    if np.any(score_gradient[future_mask] != np.float32(0.0)):
        raise FloatingPointError(f"{name} contains a nonzero future gradient.")

    return score_gradient


def _backward_transformer_block(
    output_gradient: _Float32Array,
    cache: TransformerBlockForwardCache,
    parameters: TransformerBlockParameterViews,
    gradient: TransformerBlockParameterViews,
    *,
    block_index: int,
) -> _BlockBackwardResult:
    """Calculate and accumulate one complete Transformer block backward pass."""
    feed_forward_output_gradient = _materialize_float32(
        output_gradient,
        name=f"block {block_index} feed-forward output gradient",
    )

    _accumulate_array_in_place(
        gradient.ff2_w,
        matmul_transposed_left(
            cache.feed_forward_activation,
            feed_forward_output_gradient,
        ),
        name=f"block {block_index} ff2_w gradient",
    )
    _accumulate_array_in_place(
        gradient.ff2_b,
        _sum_rows(
            feed_forward_output_gradient,
            name=f"block {block_index} ff2_b contribution",
        ),
        name=f"block {block_index} ff2_b gradient",
    )

    feed_forward_activation_gradient = matmul_transposed_right(
        feed_forward_output_gradient,
        parameters.ff2_w,
    )
    feed_forward_pre_activation_gradient = _materialize_float32(
        feed_forward_activation_gradient.astype(np.float64)
        * (cache.feed_forward_pre_activation > np.float32(0.0)).astype(np.float64),
        name=(f"block {block_index} feed-forward pre-activation gradient"),
    )

    _accumulate_array_in_place(
        gradient.ff1_w,
        matmul_transposed_left(
            cache.second_normalization.output,
            feed_forward_pre_activation_gradient,
        ),
        name=f"block {block_index} ff1_w gradient",
    )
    _accumulate_array_in_place(
        gradient.ff1_b,
        _sum_rows(
            feed_forward_pre_activation_gradient,
            name=f"block {block_index} ff1_b contribution",
        ),
        name=f"block {block_index} ff1_b gradient",
    )

    second_normalization_output_gradient = matmul_transposed_right(
        feed_forward_pre_activation_gradient,
        parameters.ff1_w,
    )
    second_normalization_backward = _layer_normalization_backward(
        second_normalization_output_gradient,
        cache.second_normalization,
        parameters.ln2_gamma,
        name=f"block {block_index} second normalization",
    )

    _accumulate_array_in_place(
        gradient.ln2_gamma,
        second_normalization_backward.gamma_gradient,
        name=f"block {block_index} ln2_gamma gradient",
    )
    _accumulate_array_in_place(
        gradient.ln2_beta,
        second_normalization_backward.beta_gradient,
        name=f"block {block_index} ln2_beta gradient",
    )

    first_residual_gradient = _add_matrices(
        output_gradient,
        second_normalization_backward.input_gradient,
        name=f"block {block_index} first residual gradient",
    )

    _accumulate_array_in_place(
        gradient.w_o,
        matmul_transposed_left(
            cache.concatenated_attention,
            first_residual_gradient,
        ),
        name=f"block {block_index} w_o gradient",
    )
    _accumulate_array_in_place(
        gradient.b_o,
        _sum_rows(
            first_residual_gradient,
            name=f"block {block_index} b_o contribution",
        ),
        name=f"block {block_index} b_o gradient",
    )

    concatenated_attention_gradient = matmul_transposed_right(
        first_residual_gradient,
        parameters.w_o,
    )

    query_gradient = np.zeros_like(
        cache.query,
        dtype=np.float32,
        order="C",
    )
    key_gradient = np.zeros_like(
        cache.key,
        dtype=np.float32,
        order="C",
    )
    value_gradient = np.zeros_like(
        cache.value,
        dtype=np.float32,
        order="C",
    )
    attention_score_gradients: list[_Float32Array] = []

    for head_index, head_cache in enumerate(cache.attention_heads):
        feature_start = head_index * TRANSFORMER_HEAD_DIMENSION
        feature_stop = feature_start + TRANSFORMER_HEAD_DIMENSION

        query_head = _copy_column_slice(
            cache.query,
            feature_start,
            feature_stop,
            name=(f"block {block_index} head {head_index} cached query"),
        )
        key_head = _copy_column_slice(
            cache.key,
            feature_start,
            feature_stop,
            name=(f"block {block_index} head {head_index} cached key"),
        )
        value_head = _copy_column_slice(
            cache.value,
            feature_start,
            feature_stop,
            name=(f"block {block_index} head {head_index} cached value"),
        )
        weighted_value_gradient = _copy_column_slice(
            concatenated_attention_gradient,
            feature_start,
            feature_stop,
            name=(f"block {block_index} head {head_index} weighted-value gradient"),
        )

        probability_gradient = matmul_transposed_right(
            weighted_value_gradient,
            value_head,
        )
        head_value_gradient = matmul_transposed_left(
            head_cache.probabilities,
            weighted_value_gradient,
        )
        score_gradient = _causal_softmax_backward(
            probability_gradient,
            head_cache.probabilities,
            name=(f"block {block_index} head {head_index} attention-score gradient"),
        )
        head_query_gradient = scalar_multiply(
            matmul(
                score_gradient,
                key_head,
            ),
            TRANSFORMER_ATTENTION_SCALE,
        )
        head_key_gradient = scalar_multiply(
            matmul_transposed_left(
                score_gradient,
                query_head,
            ),
            TRANSFORMER_ATTENTION_SCALE,
        )

        query_gradient[
            :,
            feature_start:feature_stop,
        ] = head_query_gradient
        key_gradient[
            :,
            feature_start:feature_stop,
        ] = head_key_gradient
        value_gradient[
            :,
            feature_start:feature_stop,
        ] = head_value_gradient

        attention_score_gradients.append(score_gradient)

    _accumulate_array_in_place(
        gradient.w_q,
        matmul_transposed_left(
            cache.first_normalization.output,
            query_gradient,
        ),
        name=f"block {block_index} w_q gradient",
    )
    _accumulate_array_in_place(
        gradient.b_q,
        _sum_rows(
            query_gradient,
            name=f"block {block_index} b_q contribution",
        ),
        name=f"block {block_index} b_q gradient",
    )
    _accumulate_array_in_place(
        gradient.w_k,
        matmul_transposed_left(
            cache.first_normalization.output,
            key_gradient,
        ),
        name=f"block {block_index} w_k gradient",
    )
    _accumulate_array_in_place(
        gradient.b_k,
        _sum_rows(
            key_gradient,
            name=f"block {block_index} b_k contribution",
        ),
        name=f"block {block_index} b_k gradient",
    )
    _accumulate_array_in_place(
        gradient.w_v,
        matmul_transposed_left(
            cache.first_normalization.output,
            value_gradient,
        ),
        name=f"block {block_index} w_v gradient",
    )
    _accumulate_array_in_place(
        gradient.b_v,
        _sum_rows(
            value_gradient,
            name=f"block {block_index} b_v contribution",
        ),
        name=f"block {block_index} b_v gradient",
    )

    first_normalization_output_gradient = _add_matrices(
        _add_matrices(
            matmul_transposed_right(
                query_gradient,
                parameters.w_q,
            ),
            matmul_transposed_right(
                key_gradient,
                parameters.w_k,
            ),
            name=(f"block {block_index} query-key input gradient"),
        ),
        matmul_transposed_right(
            value_gradient,
            parameters.w_v,
        ),
        name=(f"block {block_index} complete attention input gradient"),
    )

    first_normalization_backward = _layer_normalization_backward(
        first_normalization_output_gradient,
        cache.first_normalization,
        parameters.ln1_gamma,
        name=(f"block {block_index} first normalization"),
    )

    _accumulate_array_in_place(
        gradient.ln1_gamma,
        first_normalization_backward.gamma_gradient,
        name=f"block {block_index} ln1_gamma gradient",
    )
    _accumulate_array_in_place(
        gradient.ln1_beta,
        first_normalization_backward.beta_gradient,
        name=f"block {block_index} ln1_beta gradient",
    )

    input_gradient = _add_matrices(
        first_residual_gradient,
        first_normalization_backward.input_gradient,
        name=f"block {block_index} input gradient",
    )

    return _BlockBackwardResult(
        input_gradient=input_gradient,
        attention_score_gradients=tuple(attention_score_gradients),
    )


def calculate_transformer_backward(
    forward: TransformerForwardResult,
    target_ids: tuple[int, ...],
    parameters: TransformerParameterViews,
) -> TransformerBackwardResult:
    """Calculate one complete analytical canonical Transformer gradient."""
    vocabulary_size = _validate_parameter_views(parameters)
    _validate_forward_result(
        forward,
        vocabulary_size=vocabulary_size,
        expected_block_count=len(parameters.blocks),
    )
    validated_target_ids = _validate_token_ids(
        target_ids,
        name="target_ids",
        vocabulary_size=vocabulary_size,
        expected_length=len(forward.input_ids),
    )

    gradient = create_transformer_gradient_buffer(parameters.layout)
    loss = calculate_transformer_cross_entropy(
        forward.probabilities,
        validated_target_ids,
    )

    sequence_length = len(validated_target_ids)
    logit_gradient = _materialize_float32(
        forward.probabilities,
        name="initial logit gradient",
    )

    for position, target_id in enumerate(validated_target_ids):
        logit_gradient[position, target_id] = np.float32(
            float(logit_gradient[position, target_id]) - 1.0
        )

    logit_gradient = scalar_multiply(
        logit_gradient,
        1.0 / float(sequence_length),
    )

    _accumulate_array_in_place(
        gradient.views.head_w,
        matmul_transposed_left(
            forward.final_normalization.output,
            logit_gradient,
        ),
        name="head_w gradient",
    )
    _accumulate_array_in_place(
        gradient.views.head_b,
        _sum_rows(
            logit_gradient,
            name="head_b contribution",
        ),
        name="head_b gradient",
    )

    final_normalization_output_gradient = matmul_transposed_right(
        logit_gradient,
        parameters.head_w,
    )
    final_normalization_backward = _layer_normalization_backward(
        final_normalization_output_gradient,
        forward.final_normalization,
        parameters.ln_f_gamma,
        name="final normalization",
    )

    _accumulate_array_in_place(
        gradient.views.ln_f_gamma,
        final_normalization_backward.gamma_gradient,
        name="ln_f_gamma gradient",
    )
    _accumulate_array_in_place(
        gradient.views.ln_f_beta,
        final_normalization_backward.beta_gradient,
        name="ln_f_beta gradient",
    )

    activation_gradient = final_normalization_backward.input_gradient
    reverse_attention_score_gradients: list[tuple[_Float32Array, ...]] = []

    for block_index in range(len(parameters.blocks) - 1, -1, -1):
        block_backward = _backward_transformer_block(
            activation_gradient,
            forward.blocks[block_index],
            parameters.blocks[block_index],
            gradient.views.blocks[block_index],
            block_index=block_index,
        )
        activation_gradient = block_backward.input_gradient
        reverse_attention_score_gradients.append(block_backward.attention_score_gradients)

    attention_score_gradients = tuple(reversed(reverse_attention_score_gradients))

    for position, token_id in enumerate(forward.input_ids):
        _accumulate_array_in_place(
            gradient.views.tok_emb[token_id, :],
            activation_gradient[position, :],
            name=(f"token embedding gradient for token ID {token_id}"),
        )
        _accumulate_array_in_place(
            gradient.views.pos_emb[position, :],
            activation_gradient[position, :],
            name=(f"position embedding gradient for position {position}"),
        )

    _validate_float32_array(
        gradient.storage,
        name="complete canonical gradient",
        shape=(gradient.layout.total_float_count,),
        require_writable=True,
    )
    _validate_float32_array(
        activation_gradient,
        name="complete input gradient",
        shape=(
            sequence_length,
            TRANSFORMER_EMBEDDING_DIMENSION,
        ),
    )

    for parameter_array in _parameter_arrays(parameters):
        if np.shares_memory(
            gradient.storage,
            parameter_array,
        ):
            raise ValueError("Gradient storage must not overlap parameter storage.")

    return TransformerBackwardResult(
        loss=loss,
        gradient=gradient,
        logit_gradient=logit_gradient,
        input_gradient=activation_gradient,
        attention_score_gradients=attention_score_gradients,
    )


def calculate_transformer_sequence(
    sequence: TransformerTrainingSequence,
    parameters: TransformerParameterViews,
) -> TransformerSequenceResult:
    """Calculate one complete Training Sequence forward and backward pass."""
    if type(sequence) is not TransformerTrainingSequence:
        raise TypeError("sequence must be a TransformerTrainingSequence.")

    if len(sequence.input_ids) != TRANSFORMER_SEQUENCE_LENGTH:
        raise ValueError(
            f"A Transformer Training Sequence input must contain {TRANSFORMER_SEQUENCE_LENGTH} IDs."
        )

    if len(sequence.target_ids) != TRANSFORMER_SEQUENCE_LENGTH:
        raise ValueError(
            "A Transformer Training Sequence target must contain "
            f"{TRANSFORMER_SEQUENCE_LENGTH} IDs."
        )

    forward = calculate_transformer_forward(
        sequence.input_ids,
        parameters,
    )
    backward = calculate_transformer_backward(
        forward,
        sequence.target_ids,
        parameters,
    )

    return TransformerSequenceResult(
        loss=backward.loss,
        forward=forward,
        backward=backward,
    )


def calculate_logical_training_shard(
    sequences: tuple[TransformerTrainingSequence, ...],
    shard: LogicalTrainingShard,
    parameters: TransformerParameterViews,
) -> LogicalTrainingShardResult:
    """Calculate one fixed-order Logical Training Shard."""
    if type(sequences) is not tuple:
        raise TypeError("sequences must be an immutable tuple.")

    if type(shard) is not LogicalTrainingShard:
        raise TypeError("shard must be a LogicalTrainingShard.")

    if shard.shard_index < 0 or shard.shard_index >= LOGICAL_TRAINING_SHARD_COUNT:
        raise ValueError("shard.shard_index is outside the four-shard boundary.")

    if (
        shard.start_index < 0
        or shard.stop_index < shard.start_index
        or shard.stop_index > len(sequences)
    ):
        raise ValueError("The Logical Training Shard range is invalid.")

    expected_shard = build_logical_training_shards(len(sequences))[shard.shard_index]

    if shard != expected_shard:
        raise ValueError("shard must match the canonical boundary for the supplied sequences.")

    for sequence in sequences:
        if type(sequence) is not TransformerTrainingSequence:
            raise TypeError("Every sequences entry must be a TransformerTrainingSequence.")

    _validate_parameter_views(parameters)

    shard_gradient = create_transformer_gradient_buffer(parameters.layout)
    accumulated_loss = 0.0
    processed_sequence_count = 0

    for sequence_index in range(shard.start_index, shard.stop_index):
        sequence_result = calculate_transformer_sequence(
            sequences[sequence_index],
            parameters,
        )

        accumulated_loss += sequence_result.loss

        if not math.isfinite(accumulated_loss):
            raise FloatingPointError("Logical Training Shard loss is not finite.")

        _accumulate_array_in_place(
            shard_gradient.storage,
            sequence_result.backward.gradient.storage,
            name="Logical Training Shard canonical gradient",
        )
        processed_sequence_count += 1

    _validate_float32_array(
        shard_gradient.storage,
        name="Logical Training Shard canonical gradient",
        shape=(shard_gradient.layout.total_float_count,),
        require_writable=True,
    )

    return LogicalTrainingShardResult(
        shard=shard,
        processed_sequence_count=processed_sequence_count,
        loss=accumulated_loss,
        gradient=shard_gradient,
    )


def _completion_cancelled(cancellation_event: Event) -> bool:
    """Read one Event-compatible cooperative-cancellation signal."""
    is_set = getattr(cancellation_event, "is_set", None)

    if not callable(is_set):
        raise TypeError("cancellation_event must provide an is_set() method.")

    cancelled = is_set()

    if type(cancelled) is not bool:
        raise TypeError("cancellation_event.is_set() must return a bool.")

    return cancelled


def _raise_if_transformer_completion_cancelled(
    cancellation_event: Event,
) -> None:
    """Prevent a cancelled completion operation from returning success."""
    if _completion_cancelled(cancellation_event):
        raise RuntimeError("Transformer completion was cancelled.")


def _validate_transformer_completion_preprocessing(
    preprocessing: TransformerPreprocessingSnapshot,
) -> int:
    """Validate immutable preprocessing shared by completion operations."""
    if type(preprocessing) is not TransformerPreprocessingSnapshot:
        raise TypeError("preprocessing must be a TransformerPreprocessingSnapshot.")

    if type(preprocessing.vocabulary) is not tuple or not preprocessing.vocabulary:
        raise ValueError("preprocessing.vocabulary must be a non-empty tuple.")

    if any(type(token) is not str for token in preprocessing.vocabulary):
        raise TypeError("Every preprocessing Vocabulary entry must be a string.")

    if type(preprocessing.merges) is not tuple:
        raise TypeError("preprocessing.merges must be an immutable tuple.")

    for merge in preprocessing.merges:
        if type(merge) is not Merge:
            raise TypeError("Every preprocessing Merge Table entry must be a Merge.")

        if type(merge.pair) is not tuple or len(merge.pair) != 2:
            raise ValueError("Every preprocessing Merge pair must contain exactly two tokens.")

        if any(type(token) is not str for token in merge.pair):
            raise TypeError("Every preprocessing Merge pair entry must be a string.")

        if type(merge.merged) is not str:
            raise TypeError("Every preprocessing merged token must be a string.")

    vocabulary_size = len(preprocessing.vocabulary)

    _validate_token_ids(
        preprocessing.generation_seed_ids,
        name="preprocessing.generation_seed_ids",
        vocabulary_size=vocabulary_size,
        expected_length=TRANSFORMER_GENERATION_SEED_LENGTH,
    )

    if type(preprocessing.training_sequences) is not tuple:
        raise TypeError("preprocessing.training_sequences must be an immutable tuple.")

    for sequence in preprocessing.training_sequences:
        if type(sequence) is not TransformerTrainingSequence:
            raise TypeError(
                "Every preprocessing.training_sequences entry must be a "
                "TransformerTrainingSequence."
            )

    if type(preprocessing.logical_training_shards) is not tuple:
        raise TypeError("preprocessing.logical_training_shards must be an immutable tuple.")

    expected_shards = build_logical_training_shards(len(preprocessing.training_sequences))

    if preprocessing.logical_training_shards != expected_shards:
        raise ValueError("preprocessing logical shards do not match its Training Sequences.")

    return vocabulary_size


def _validate_transformer_completion_parameters(
    parameters: InitializedTransformerParameters,
    preprocessing: TransformerPreprocessingSnapshot,
) -> int:
    """Validate one finite canonical parameter owner against preprocessing."""
    if type(parameters) is not InitializedTransformerParameters:
        raise TypeError("parameters must be an InitializedTransformerParameters.")

    vocabulary_size = _validate_transformer_completion_preprocessing(preprocessing)

    _validate_transformer_parameter_layout(parameters.layout)

    if parameters.layout.vocabulary_size != vocabulary_size:
        raise ValueError("parameter layout Vocabulary size does not match preprocessing.")

    storage = _validate_float32_array(
        parameters.storage,
        name="Transformer completion parameter storage",
        shape=(parameters.layout.total_float_count,),
    )

    if parameters.views.layout != parameters.layout:
        raise ValueError("parameter views do not match their canonical layout.")

    parameter_vocabulary_size = _validate_parameter_views(parameters.views)

    if parameter_vocabulary_size != vocabulary_size:
        raise ValueError("parameter Vocabulary size does not match preprocessing.")

    for parameter_view in _parameter_arrays(parameters.views):
        if not np.shares_memory(parameter_view, storage):
            raise ValueError("Every completion parameter view must use parameter storage.")

    return vocabulary_size


def _validate_completed_transformer_run(
    run: TransformerTrainingRun,
    preprocessing: TransformerPreprocessingSnapshot,
) -> int:
    """Validate the common successful-run boundary for final completion work."""
    if type(run) is not TransformerTrainingRun:
        raise TypeError("run must be a TransformerTrainingRun.")

    if run.is_failed:
        raise RuntimeError("A failed Transformer Training Run cannot be completed.")

    if not run.is_complete:
        raise RuntimeError("The Transformer Training Run must be complete.")

    if run.last_completed_epoch != run.requested_epochs:
        raise RuntimeError("The inclusive final Transformer epoch has not committed.")

    _validate_transformer_training_run_storage(run)

    vocabulary_size = _validate_transformer_completion_parameters(
        run.parameters,
        preprocessing,
    )

    if run.logical_training_shards != preprocessing.logical_training_shards:
        raise ValueError("The run sequence ownership does not match preprocessing.")

    return vocabulary_size


def _validate_transformer_generation_arguments(
    *,
    epoch: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[int, float, float, int]:
    """Validate exact public generation scalar types and ranges."""
    if type(epoch) is not int:
        raise TypeError("epoch must be an integer.")

    if epoch < 0:
        raise ValueError("epoch must be non-negative.")

    if type(temperature) is not float:
        raise TypeError("temperature must be a float.")

    if not math.isfinite(temperature):
        raise FloatingPointError("temperature must be finite.")

    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.1 and 2.0.")

    if type(top_p) is not float:
        raise TypeError("top_p must be a float.")

    if not math.isfinite(top_p):
        raise FloatingPointError("top_p must be finite.")

    if not 0.1 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.1 and 1.0.")

    if type(max_tokens) is not int:
        raise TypeError("max_tokens must be an integer.")

    if not 3 <= max_tokens <= 500:
        raise ValueError("max_tokens must be between 3 and 500.")

    return epoch, temperature, top_p, max_tokens


def _validate_saved_transformer_generation_state(
    parameters: InitializedTransformerParameters,
    vocabulary: Sequence[str],
    prompt: PreparedSavedTransformerPrompt,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Validate request-owned saved-model state before deterministic generation."""
    if type(parameters) is not InitializedTransformerParameters:
        raise TypeError("parameters must be an InitializedTransformerParameters.")

    if isinstance(vocabulary, (str, bytes)) or not isinstance(vocabulary, Sequence):
        raise TypeError("vocabulary must be an ordered sequence of strings.")

    validated_vocabulary = tuple(vocabulary)

    if not validated_vocabulary:
        raise ValueError("vocabulary must not be empty.")

    if any(type(token) is not str for token in validated_vocabulary):
        raise TypeError("Every vocabulary entry must be a string.")

    vocabulary_size = len(validated_vocabulary)

    _validate_transformer_parameter_layout(parameters.layout)

    if parameters.layout.vocabulary_size != vocabulary_size:
        raise ValueError("parameter layout Vocabulary size does not match vocabulary.")

    storage = _validate_float32_array(
        parameters.storage,
        name="Saved Transformer generation parameter storage",
        shape=(parameters.layout.total_float_count,),
    )

    if parameters.views.layout != parameters.layout:
        raise ValueError("parameter views do not match their canonical layout.")

    parameter_vocabulary_size = _validate_parameter_views(parameters.views)

    if parameter_vocabulary_size != vocabulary_size:
        raise ValueError("parameter Vocabulary size does not match vocabulary.")

    for parameter_view in _parameter_arrays(parameters.views):
        if not np.shares_memory(parameter_view, storage):
            raise ValueError("Every generation parameter view must use parameter storage.")

    if type(prompt) is not PreparedSavedTransformerPrompt:
        raise TypeError("prompt must be a PreparedSavedTransformerPrompt.")

    if type(prompt.text) is not str:
        raise TypeError("prompt.text must be a string.")

    if not prompt.text or prompt.text != prompt.text.strip():
        raise ValueError("prompt.text must be one non-empty outer-trimmed string.")

    if type(prompt.token_ids) is not tuple:
        raise TypeError("prompt.token_ids must be an immutable tuple.")

    if not 1 <= len(prompt.token_ids) <= TRANSFORMER_SEQUENCE_LENGTH:
        raise ValueError("prompt.token_ids must contain between one and sixteen IDs.")

    for token_id in prompt.token_ids:
        if type(token_id) is not int:
            raise TypeError("Every prompt token ID must be an integer.")

        if token_id < 0 or token_id >= vocabulary_size:
            raise ValueError("Every prompt token ID must exist in the vocabulary.")

    return validated_vocabulary, prompt.token_ids


def _sample_transformer_nucleus_token(
    probabilities: _Float32Array,
    *,
    top_p: float,
    generator: Mulberry32,
) -> int:
    """Sample one Vocabulary ID from the minimum stable top-p prefix."""
    if type(probabilities) is not np.ndarray:
        raise TypeError("probabilities must be a NumPy ndarray.")

    if probabilities.dtype != np.dtype(np.float32):
        raise TypeError("probabilities must have dtype float32.")

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be one non-empty row.")

    if not probabilities.flags.c_contiguous:
        raise ValueError("probabilities must be C-contiguous.")

    if not np.isfinite(probabilities).all():
        raise FloatingPointError("probabilities contain a non-finite value.")

    if np.any(probabilities < np.float32(0.0)):
        raise FloatingPointError("probabilities cannot contain negative values.")

    ranked_indices = sorted(
        range(int(probabilities.size)),
        key=lambda token_id: (
            -float(probabilities[token_id]),
            token_id,
        ),
    )

    nucleus: list[tuple[int, float]] = []
    nucleus_probability = 0.0

    for token_id in ranked_indices:
        probability = float(probabilities[token_id])
        nucleus.append((token_id, probability))
        nucleus_probability += probability

        if not math.isfinite(nucleus_probability):
            raise FloatingPointError("top-p nucleus probability is not finite.")

        if nucleus_probability >= top_p:
            break

    if not nucleus or nucleus_probability <= 0.0:
        raise FloatingPointError("top-p nucleus has no positive probability.")

    threshold = generator.random() * nucleus_probability
    cumulative_probability = 0.0

    for token_id, probability in nucleus:
        cumulative_probability += probability

        if threshold < cumulative_probability:
            return token_id

    return nucleus[-1][0]


def _generate_transformer_token_ids(
    parameters: InitializedTransformerParameters,
    initial_token_ids: tuple[int, ...],
    *,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    cancellation_event: Event,
) -> tuple[int, ...]:
    """Generate deterministic token IDs through the shared decoder-only sampling path."""
    generated_ids = list(initial_token_ids)
    generator = Mulberry32(seed)

    for _ in range(max_tokens):
        _raise_if_transformer_completion_cancelled(cancellation_event)

        context_ids = tuple(generated_ids[-TRANSFORMER_SEQUENCE_LENGTH:])

        forward = calculate_transformer_forward(
            context_ids,
            parameters.views,
        )

        final_logits = forward.logits[-1]

        scaled_values = (
            np.asarray(
                final_logits,
                dtype=np.float64,
            )
            / temperature
        )

        if not np.isfinite(scaled_values).all():
            raise FloatingPointError("temperature-scaled logits are not finite.")

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            scaled_row: _Float32Array = np.array(
                scaled_values,
                dtype=np.float32,
                order="C",
                copy=True,
            ).reshape(
                1,
                parameters.layout.vocabulary_size,
            )

        if not np.isfinite(scaled_row).all():
            raise FloatingPointError("temperature-scaled float32 logits are not finite.")

        probabilities: _Float32Array = stable_row_softmax(scaled_row)[0]

        next_token_id = _sample_transformer_nucleus_token(
            probabilities,
            top_p=top_p,
            generator=generator,
        )

        generated_ids.append(next_token_id)

    _raise_if_transformer_completion_cancelled(cancellation_event)

    return tuple(generated_ids)


def generate_transformer_text(
    parameters: InitializedTransformerParameters,
    preprocessing: TransformerPreprocessingSnapshot,
    *,
    epoch: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    cancellation_event: Event,
) -> GeneratedTextSample:
    """Generate one deterministic epoch-seeded Transformer text sample."""
    (
        validated_epoch,
        validated_temperature,
        validated_top_p,
        validated_max_tokens,
    ) = _validate_transformer_generation_arguments(
        epoch=epoch,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    _validate_transformer_completion_parameters(
        parameters,
        preprocessing,
    )

    generated_ids = _generate_transformer_token_ids(
        parameters,
        preprocessing.generation_seed_ids,
        seed=(42 + validated_epoch) & 0xFFFFFFFF,
        temperature=validated_temperature,
        top_p=validated_top_p,
        max_tokens=validated_max_tokens,
        cancellation_event=cancellation_event,
    )

    text = "".join(preprocessing.vocabulary[token_id] for token_id in generated_ids)

    return GeneratedTextSample(
        epoch=validated_epoch,
        text=text,
    )


def generate_saved_transformer_text(
    parameters: InitializedTransformerParameters,
    vocabulary: Sequence[str],
    prompt: PreparedSavedTransformerPrompt,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    cancellation_event: Event,
) -> str:
    """Generate one deterministic complete continuation from a prepared saved-model prompt."""
    (
        _validated_epoch,
        validated_temperature,
        validated_top_p,
        validated_max_tokens,
    ) = _validate_transformer_generation_arguments(
        epoch=0,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    validated_vocabulary, prompt_token_ids = _validate_saved_transformer_generation_state(
        parameters,
        vocabulary,
        prompt,
    )

    generated_ids = _generate_transformer_token_ids(
        parameters,
        prompt_token_ids,
        seed=42,
        temperature=validated_temperature,
        top_p=validated_top_p,
        max_tokens=validated_max_tokens,
        cancellation_event=cancellation_event,
    )

    continuation_ids = generated_ids[len(prompt_token_ids) :]

    continuation = "".join(validated_vocabulary[token_id] for token_id in continuation_ids)

    return f"{prompt.text}{continuation}"


def evaluate_transformer_final_loss(
    run: TransformerTrainingRun,
    preprocessing: TransformerPreprocessingSnapshot,
    *,
    cancellation_event: Event,
) -> float:
    """Recompute exact fixed-order loss from final post-Adam parameters."""
    _validate_completed_transformer_run(
        run,
        preprocessing,
    )

    if not preprocessing.training_sequences:
        raise ValueError("Final evaluation requires at least one Training Sequence.")

    accumulated_loss = 0.0

    for sequence in preprocessing.training_sequences:
        _raise_if_transformer_completion_cancelled(cancellation_event)

        forward = calculate_transformer_forward(
            sequence.input_ids,
            run.parameters.views,
        )

        sequence_loss = calculate_transformer_cross_entropy(
            forward.probabilities,
            sequence.target_ids,
        )

        if not math.isfinite(sequence_loss):
            raise FloatingPointError("Final Transformer sequence loss is not finite.")

        accumulated_loss += sequence_loss

        if not math.isfinite(accumulated_loss):
            raise FloatingPointError("Accumulated final Transformer loss is not finite.")

    average_loss = accumulated_loss / float(len(preprocessing.training_sequences))

    if not math.isfinite(average_loss):
        raise FloatingPointError("Final Transformer average loss is not finite.")

    _raise_if_transformer_completion_cancelled(cancellation_event)

    return round_typescript_decimal(
        average_loss,
        6,
    )


def _saved_transformer_parameter_values(
    values: _Float32Array,
    *,
    record: TransformerParameterLayoutRecord,
    name: str,
) -> list[float]:
    """Flatten, validate, and round one canonical durable parameter array."""
    validated = _validate_float32_array(
        values,
        name=name,
        shape=record.shape,
    )

    flat_values = validated.reshape(
        -1,
        order="C",
    )

    if int(flat_values.size) != record.length:
        raise ValueError(f"{name} length does not match its canonical layout record.")

    return [
        round_typescript_decimal(
            float(value),
            6,
        )
        for value in flat_values
    ]


def _build_saved_transformer_block_weights(
    block: TransformerBlockParameterViews,
    *,
    block_index: int,
    layout: TransformerParameterLayout,
) -> SavedTransformerBlockWeights:
    """Build one exact canonical block dictionary in stable key order."""
    saved_values: dict[str, list[float]] = {}

    for key in _TRANSFORMER_BLOCK_PARAMETER_KEYS:
        saved_values[key] = _saved_transformer_parameter_values(
            block.get(key),
            record=layout.get_record(
                key,
                block_index,
            ),
            name=f"blocks[{block_index}].{key}",
        )

    return cast(
        SavedTransformerBlockWeights,
        saved_values,
    )


def build_saved_transformer_model(
    run: TransformerTrainingRun,
    preprocessing: TransformerPreprocessingSnapshot,
) -> SavedTransformerModel:
    """Construct one fresh exact persistence-ready Saved Transformer Model."""
    vocabulary_size = _validate_completed_transformer_run(
        run,
        preprocessing,
    )

    layout = run.parameters.layout
    views = run.parameters.views

    config: SavedTransformerConfig = {
        "vocabSize": vocabulary_size,
        "contextLen": TRANSFORMER_CONTEXT_LENGTH,
        "embDim": TRANSFORMER_EMBEDDING_DIMENSION,
        "numHeads": TRANSFORMER_ATTENTION_HEAD_COUNT,
        "ffDim": TRANSFORMER_FEED_FORWARD_DIMENSION,
        "numLayers": layout.num_layers,
    }

    saved_blocks: list[SavedTransformerBlockWeights] = [
        _build_saved_transformer_block_weights(
            block,
            block_index=block_index,
            layout=layout,
        )
        for block_index, block in enumerate(views.blocks)
    ]

    weights: SavedTransformerWeights = {
        "tokEmb": _saved_transformer_parameter_values(
            views.tok_emb,
            record=layout.get_record("tokEmb"),
            name="tokEmb",
        ),
        "posEmb": _saved_transformer_parameter_values(
            views.pos_emb,
            record=layout.get_record("posEmb"),
            name="posEmb",
        ),
        "blocks": saved_blocks,
        "lnFGamma": _saved_transformer_parameter_values(
            views.ln_f_gamma,
            record=layout.get_record("lnFGamma"),
            name="lnFGamma",
        ),
        "lnFBeta": _saved_transformer_parameter_values(
            views.ln_f_beta,
            record=layout.get_record("lnFBeta"),
            name="lnFBeta",
        ),
        "headW": _saved_transformer_parameter_values(
            views.head_w,
            record=layout.get_record("headW"),
            name="headW",
        ),
        "headB": _saved_transformer_parameter_values(
            views.head_b,
            record=layout.get_record("headB"),
            name="headB",
        ),
    }

    merges: list[SavedTransformerMerge] = [
        {
            "pair": list(merge.pair),
            "merged": merge.merged,
        }
        for merge in preprocessing.merges
    ]

    return {
        "type": "decoder-transformer",
        "config": config,
        "vocab": list(preprocessing.vocabulary),
        "merges": merges,
        "weights": weights,
    }
