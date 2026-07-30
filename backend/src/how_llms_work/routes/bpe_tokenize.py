# backend/src/how_llms_work/routes/bpe_tokenize.py

"""FastAPI route for frontend-compatible BPE tokenization streams."""

import asyncio
from collections.abc import AsyncIterator, Sequence

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from how_llms_work.ml.bpe import Merge, apply_merges, count_words, train_bpe
from how_llms_work.schemas import ChatRequest
from how_llms_work.sse import create_sse_response, format_sse

router = APIRouter()


async def stream_bpe_tokenization(
    message: str,
    word_count: int,
    characters: Sequence[str],
    merges: Sequence[Merge],
) -> AsyncIterator[str]:
    yield format_sse(
        "init",
        {
            "corpus": message,
            "characters": list(characters[:200]),
            "charCount": len(characters),
            "wordCount": word_count,
        },
    )

    await asyncio.sleep(0.8)

    vocabulary = set(characters)
    total_token_count = len(characters)

    for step, merge in enumerate(merges, start=1):
        vocabulary.add(merge.merged)
        total_token_count -= merge.frequency

        yield format_sse(
            "merge",
            {
                "step": step,
                "pair": list(merge.pair),
                "frequency": merge.frequency,
                "newToken": merge.merged,
                "vocabSize": len(vocabulary),
                "tokenCount": total_token_count,
            },
        )

    input_tokens = apply_merges(message, merges)
    compression_ratio = f"{len(message) / len(input_tokens):.1f}x" if message else "N/A"

    yield format_sse(
        "result",
        {
            "inputTokens": input_tokens,
            "tokenCount": len(input_tokens),
            "originalCharCount": len(message),
            "compressionRatio": compression_ratio,
        },
    )


@router.post("/bpe-tokenize")
async def bpe_tokenize(
    request: ChatRequest,
) -> StreamingResponse:
    message = request.message
    word_frequencies = count_words(message)
    characters = list(message)
    merges = train_bpe(word_frequencies)

    return create_sse_response(
        stream_bpe_tokenization(
            message=message,
            word_count=len(word_frequencies),
            characters=characters,
            merges=merges,
        )
    )
