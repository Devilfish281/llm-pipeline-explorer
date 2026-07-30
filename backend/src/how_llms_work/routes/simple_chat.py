# backend/src/how_llms_work/routes/simple_chat.py

import asyncio
import random
import re
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from how_llms_work.schemas import ChatRequest
from how_llms_work.sse import create_sse_response, format_sse

router = APIRouter()


def get_simple_chat_response(message: str) -> str:
    message_lower = message.lower()
    words = [re.sub(r"""[!?.;,:"']""", "", word) for word in message_lower.split()]

    greetings = {"hello", "hi", "hey"}

    if greetings.intersection(words):
        return "Hello! How can I help you today?"

    if message_lower.startswith("i feel "):
        feeling = message_lower[len("i feel ") :].replace(" i ", " you ")
        return f"Why do you feel {feeling}?"

    if "my" in words:
        subject_index = words.index("my") + 1
        if subject_index < len(words):
            return f"Tell me more about your {words[subject_index]}."

    if "worried" in message_lower:
        return "How long have you been worried about this?"

    continuations = [
        "Please tell me more.",
        "How does that make you feel?",
        "Why do you think that is?",
    ]
    return random.choice(continuations)


async def stream_chat(message: str) -> AsyncIterator[str]:
    response = get_simple_chat_response(message)

    yield format_sse("start", {})
    await asyncio.sleep(1)

    for word in response.split():
        yield format_sse("word", {"word": word})
        await asyncio.sleep(0.2)

    yield format_sse("done", {})


@router.post("/simple-chat")
async def simple_chat(request: ChatRequest) -> StreamingResponse:
    return create_sse_response(stream_chat(request.message))
