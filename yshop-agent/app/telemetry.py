"""Request-scoped, metadata-only correlation for HTTP, tools and action history."""

from contextvars import ContextVar
import logging
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import MutableHeaders


_REQUEST_ID: ContextVar[str | None] = ContextVar("agent_request_id", default=None)
logger = logging.getLogger(__name__)
_ROUTES = {
    "/health": "health",
    "/agent/chat": "chat",
    "/agent/actions/{action_id}": "action",
    "/agent/actions/{action_id}/decision": "decision",
    "/agent/actions/{action_id}/history": "history",
    "/agent/cart": "cart",
    "/agent/me": "identity",
    "/agent/products": "products",
    "/agent/products/{product_id}": "product",
    "/agent/policies": "policies",
    "/docs": "docs",
    "/openapi.json": "openapi",
}


def request_id() -> str | None:
    return _REQUEST_ID.get()


def metadata_log(target: logging.Logger, message: str, *args) -> None:
    target.info(message + " request_id=%s", *args, request_id())


class RequestMetadataMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        token = _REQUEST_ID.set(uuid4().hex)
        started = perf_counter()
        status = 500

        async def correlated_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id()
            await send(message)

        try:
            await self.app(scope, receive, correlated_send)
        finally:
            route = _ROUTES.get(getattr(scope.get("route"), "path", None), "other")
            method = scope.get("method")
            method = method if method in {"GET", "POST", "OPTIONS", "HEAD", "PUT", "DELETE", "PATCH"} else "OTHER"
            try:
                metadata_log(logger, "agent request route=%s method=%s status=%d elapsed_ms=%d",
                             route, method, status, max(0, int((perf_counter() - started) * 1000)))
            finally:
                _REQUEST_ID.reset(token)
