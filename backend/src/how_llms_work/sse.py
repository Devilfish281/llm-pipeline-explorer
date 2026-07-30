# backend/src/how_llms_work/sse.py

import json
from collections.abc import AsyncIterable
from typing import Final

from fastapi.responses import StreamingResponse

SSE_MEDIA_TYPE: Final = "text/event-stream"
SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def format_sse(event: str, data: dict[str, object]) -> str:
    serialized_data = json.dumps(data)
    return f"event: {event}\ndata: {serialized_data}\n\n"


def create_sse_response(stream: AsyncIterable[str]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )
