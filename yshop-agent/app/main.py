from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.routes import router
from app.agent.config import AgentSettings
from app.agent.model import OpenAIChatModel
from app.config import Settings
from app.errors import ShopError
from app.rag.config import RagSettings
from app.rag.service import PolicyService
from app.tools.yshop import YShopClient
from app.storage.config import SessionSettings
from app.storage.sessions import SessionStore
from app.storage.actions import CartActionStore
from app.storage.checkpoints import open_checkpoints
from app.telemetry import RequestMetadataMiddleware


class AgentAPI(FastAPI):
    def build_middleware_stack(self):
        # Correlate responses produced by Starlette's outer 500 handler too.
        return RequestMetadataMiddleware(super().build_middleware_stack())


def error_response(error: ShopError) -> JSONResponse:
    body = {"code": error.code, "message": error.message}
    if error.outcome is not None:
        body["outcome"] = error.outcome
    headers = {"WWW-Authenticate": "Bearer"} if error.http_status == 401 else None
    return JSONResponse({"error": body}, status_code=error.http_status, headers=headers)


def create_app(settings: Settings | None = None, *, rag_settings: RagSettings | None = None,
               agent_settings: AgentSettings | None = None,
               session_settings: SessionSettings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()
    agent_settings = agent_settings if agent_settings is not None else AgentSettings()
    session_settings = session_settings if session_settings is not None else SessionSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.sessions = await run_in_threadpool(CartActionStore, session_settings.db_path)
        except ShopError:
            try:
                application.state.sessions = await run_in_threadpool(SessionStore, session_settings.db_path)
            except ShopError:
                application.state.sessions = None
        async with AsyncExitStack() as resources:
            application.state.checkpoints = None
            try:
                application.state.checkpoints = await resources.enter_async_context(
                    open_checkpoints(session_settings.checkpoint_path))
            except ShopError:
                pass
            async with httpx.AsyncClient(
            base_url=settings.yshop_api_url,
            timeout=settings.http_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            ) as http:
                application.state.shop = YShopClient(http)
                async with httpx.AsyncClient(
                timeout=agent_settings.llm_timeout_seconds,
                proxy=agent_settings.llm_proxy_url,
                follow_redirects=False,
                trust_env=False,
                ) as model_http:
                    application.state.chat_model = OpenAIChatModel(agent_settings, model_http)
                    try:
                        yield
                    finally:
                        if application.state.sessions is not None:
                            await run_in_threadpool(application.state.sessions.close)

    application = AgentAPI(title="YShop Agent", lifespan=lifespan)
    # This creates only lightweight objects: no Chroma client or model load.
    application.state.policy = PolicyService(rag_settings)

    @application.exception_handler(ShopError)
    async def handle_shop_error(request: Request, error: ShopError) -> JSONResponse:
        return error_response(error)

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return error_response(ShopError("invalid_request"))

    if settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=["X-Request-ID"],
        )
    application.include_router(router)
    return application


app = create_app()
