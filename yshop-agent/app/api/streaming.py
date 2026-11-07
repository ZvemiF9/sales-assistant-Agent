"""Public SSE framing around one owned, committed chat operation."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

import anyio
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from app.agent.schemas import SessionChatResponse
from app.errors import ShopError


def event(kind: str, data: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def stream_chat(
    work: Callable[[], Awaitable[SessionChatResponse]], *, heartbeat_seconds: float = 15,
) -> AsyncIterator[str]:
    cancellation = anyio.CancelScope()

    async def owned_work():
        with cancellation:
            return await work()

    task = asyncio.ensure_future(owned_work())
    try:
        yield event("status", {"status": "processing"})
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
            if not done:
                yield ": heartbeat\n\n"
        try:
            result = task.result().model_dump(mode="json")
        except Exception as error:
            public = error if isinstance(error, ShopError) else ShopError("model_unavailable")
            payload = {"code": public.code, "message": public.message}
            if public.outcome is not None:
                payload["outcome"] = public.outcome
            yield event("error", {"error": payload})
            return
        yield event("text", {"text": result["answer"]})
        yield event("products", {"products": result["products"]})
        yield event("sources", {"sources": result["sources"], "notice": result["notice"]})
        if result["pending_action"] is not None:
            yield event("confirmation", {"pending_action": result["pending_action"]})
        yield event("done", result)
    finally:
        # Starlette closes the generator inside a cancelled AnyIO scope.
        # Keep cleanup owned and awaited even in that scope.
        with anyio.CancelScope(shield=True):
            if not task.done():
                cancellation.cancel()
            await asyncio.gather(task, return_exceptions=True)


class ChatStreamingResponse(StreamingResponse):
    def __init__(self, work: Callable[[], Awaitable[SessionChatResponse]]) -> None:
        self._stream = stream_chat(work)
        super().__init__(
            self._stream, media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A send-side disconnect can leave the iterator suspended at yield.
            with anyio.CancelScope(shield=True):
                await self._stream.aclose()
