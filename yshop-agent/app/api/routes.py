import asyncio
import json
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, Path, Query, Request
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.api.dependencies import RequestIdentity, require_identity
from app.api.streaming import ChatStreamingResponse
from app.agent.conversation import ConversationAgent
from app.agent.schemas import ChatRequest, SessionChatResponse
from app.agent.tools import QueryTools
from app.errors import ShopError
from app.schemas import CartSnapshot, ProductDetail, ProductSearchInput, ProductSearchResult, UserIdentity
from app.rag.schemas import PolicyAnswer, PolicyQuery
from app.actions.schemas import ActionDecision, ActionResponse
from app.actions.history import ActionHistory
from app.agent.cart_action import CartActionService
from app.storage.actions import CartActionStore
from app.tools.yshop import _unique_json_object


router = APIRouter()
Identity = Annotated[RequestIdentity, Depends(require_identity)]


def action_tools(request: Request, identity: RequestIdentity) -> QueryTools:
    model = getattr(request.app.state, "chat_model", None)
    key = getattr(getattr(model, "settings", None), "llm_api_key", None)
    secrets = (key.get_secret_value(),) if key is not None else ()
    return QueryTools(request.app.state.shop, request.app.state.policy, identity.authorization,
                      public_secrets=secrets, public_text_secrets=(identity.user.nickname,))


def action_service(request: Request) -> CartActionService:
    sessions = getattr(request.app.state, "sessions", None)
    if not isinstance(sessions, CartActionStore):
        raise ShopError("session_unavailable")
    return CartActionService(sessions, getattr(request.app.state, "checkpoints", None))


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "yshop-agent"}


@router.post("/agent/chat", response_model=SessionChatResponse)
async def chat(query: ChatRequest, request: Request, identity: Identity) -> SessionChatResponse:
    model = getattr(request.app.state, "chat_model", None)
    sessions = getattr(request.app.state, "sessions", None)
    if sessions is None:
        raise ShopError("session_unavailable")
    session_id = query.session_id
    if session_id is None:
        session_id = await run_in_threadpool(sessions.create, identity.user.user_id)
    lease = None
    try:
        # A direct task cancellation must not discard a lease returned by a worker.
        with anyio.CancelScope(shield=True):
            claiming = asyncio.create_task(run_in_threadpool(sessions.claim, session_id, identity.user.user_id))
            try:
                lease = await asyncio.shield(claiming)
            except asyncio.CancelledError:
                lease = await claiming
                raise
        tools = action_tools(request, identity)
        if isinstance(sessions, CartActionStore):
            action = await action_service(request).prepare_chat(
                lease, identity.user.user_id, query.message, identity.authorization, tools)
            if action is not None:
                return SessionChatResponse(**action.model_dump(), model_rounds=0)
        if model is None:
            raise ShopError("model_not_configured")
        result, state = await ConversationAgent(model, tools).run(query.message, lease.state)
        response = SessionChatResponse(**result.model_dump(), session_id=session_id)
        await run_in_threadpool(sessions.commit, lease, state)
        return response
    finally:
        try:
            if lease is not None:
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(sessions.release, lease)
        except ShopError:
            # Preserve the original model/storage error. A failed release expires.
            pass


@router.get("/agent/actions/{action_id}", response_model=ActionResponse)
async def get_action(action_id: str, request: Request, identity: Identity) -> ActionResponse:
    return await action_service(request).get(action_id, identity.user.user_id, action_tools(request, identity))


@router.post("/agent/chat/stream")
async def chat_stream(query: ChatRequest, request: Request, identity: Identity) -> ChatStreamingResponse:
    return ChatStreamingResponse(lambda: chat(query, request, identity))


@router.get("/agent/actions/{action_id}/history", response_model=ActionHistory)
async def action_history(action_id: str, request: Request, identity: Identity) -> ActionHistory:
    return await run_in_threadpool(action_service(request).store.history, action_id, identity.user.user_id)


@router.post("/agent/actions/{action_id}/decision", response_model=ActionResponse)
async def decide_action(action_id: str, request: Request, identity: Identity) -> ActionResponse:
    try:
        body = await request.body()
        if len(body) > 1024 or request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise ValueError("invalid decision content")
        decoded = json.loads(body, object_pairs_hook=_unique_json_object)
        decision = ActionDecision.model_validate(decoded)
    except (ValueError, UnicodeError, ValidationError):
        raise ShopError("invalid_request") from None
    return await action_service(request).decide(
        action_id, identity.user.user_id, identity.authorization, action_tools(request, identity), decision)


@router.get("/agent/policies", response_model=PolicyAnswer)
async def policies(
    query: Annotated[PolicyQuery, Query()], request: Request, identity: Identity,
) -> PolicyAnswer:
    # Identity is verified for this request, and is never passed to retrieval.
    return await run_in_threadpool(request.app.state.policy.answer, query.question)


@router.get("/agent/me", response_model=UserIdentity)
async def current_user(identity: Identity) -> UserIdentity:
    return identity.user


@router.get("/agent/products/{product_id}", response_model=ProductDetail)
async def product_detail(
    product_id: Annotated[int, Path(gt=0)], request: Request, identity: Identity
) -> ProductDetail:
    return await request.app.state.shop.get_product_detail(product_id, identity.authorization)


@router.get("/agent/products", response_model=ProductSearchResult)
async def search_products(
    query: Annotated[ProductSearchInput, Query()], request: Request, identity: Identity,
) -> ProductSearchResult:
    return await request.app.state.shop.search_products(query, identity.authorization)


@router.get("/agent/cart", response_model=CartSnapshot)
async def cart(request: Request, identity: Identity) -> CartSnapshot:
    return await request.app.state.shop.get_cart(identity.authorization)
