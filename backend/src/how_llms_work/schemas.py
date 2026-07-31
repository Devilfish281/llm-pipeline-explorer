# backend/src/how_llms_work/schemas.py

from typing import Annotated

from how_llms_work.ml.neural_net import NetworkMode
from pydantic import BaseModel, ConfigDict, Field

QueryWord = Annotated[str, Field(strict=True, min_length=1)]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class NeuralNetRequest(BaseModel):
    mode: NetworkMode
    epochs: Annotated[int, Field(strict=True, ge=100, le=100_000)] = 5_000


class TrainEmbedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    words: Annotated[list[QueryWord], Field(min_length=1, max_length=10)]
    epochs: Annotated[int, Field(strict=True, ge=10, le=10_000)] = 10_000
    dimensions: Annotated[int, Field(strict=True, ge=4, le=64)] = 32
    window_size: Annotated[
        int,
        Field(
            alias="windowSize",
            strict=True,
            ge=1,
            le=5,
        ),
    ] = 2
    negative_samples: Annotated[
        int,
        Field(
            alias="negativeSamples",
            strict=True,
            ge=1,
            le=10,
        ),
    ] = 5


class TrainTransformerRequest(BaseModel):
    """Strict public request for one Transformer Training Run."""

    model_config = ConfigDict(extra="ignore")

    epochs: Annotated[
        int,
        Field(
            strict=True,
            ge=50,
            le=2_000,
        ),
    ] = 300

    temperature: Annotated[
        float,
        Field(
            strict=True,
            allow_inf_nan=False,
            ge=0.1,
            le=2.0,
        ),
    ] = 0.8

    top_p: Annotated[
        float,
        Field(
            alias="topP",
            strict=True,
            allow_inf_nan=False,
            ge=0.1,
            le=1.0,
        ),
    ] = 0.9

    num_layers: Annotated[
        int,
        Field(
            alias="numLayers",
            strict=True,
            ge=1,
            le=6,
        ),
    ] = 2

    max_tokens: Annotated[
        int,
        Field(
            alias="maxTokens",
            strict=True,
            ge=3,
            le=500,
        ),
    ] = 40


class LoadTransformerRequest(BaseModel):
    """Strict public request for one named Saved Transformer Generation Run."""

    model_config = ConfigDict(extra="ignore")

    model_file: Annotated[
        str | None,
        Field(
            alias="modelFile",
            strict=True,
            min_length=1,
        ),
    ]

    prompt: Annotated[
        str,
        Field(strict=True),
    ]

    temperature: Annotated[
        float,
        Field(
            strict=True,
            allow_inf_nan=False,
            ge=0.1,
            le=2.0,
        ),
    ]

    top_p: Annotated[
        float,
        Field(
            alias="topP",
            strict=True,
            allow_inf_nan=False,
            ge=0.1,
            le=1.0,
        ),
    ]

    max_tokens: Annotated[
        int,
        Field(
            alias="maxTokens",
            strict=True,
            ge=3,
            le=500,
        ),
    ]
