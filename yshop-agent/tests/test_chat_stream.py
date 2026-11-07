"""Real authenticated API and owned stream work; external transport is simulated."""

import asyncio
import json
import threading
from uuid import uuid4

import pytest
import anyio
from langchain_core.messages import AIMessage

from app.agent.schemas import SessionChatResponse
from app.config import SERVICE_ENV_FILE
from app.errors import ShopError
from tests.test_agent_graph import AUTH, ScriptedModel, call
from tests.test_cart_action_api import client_for, selected_session
from tests.test_chat_api import chat_client


def events(body):
    result = []
    for frame in body.split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        lines = frame.splitlines()
        result.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return result


@pytest.mark.parametrize("authorization", [None, "Basic invalid", "Bearer", "Bearer a b"])
def test_stream_rejects_identity_before_work(authorization):
    model = ScriptedModel([])
    with chat_client(model=model) as (client, app, calls):
        headers = {} if authorization is None else {"Authorization": authorization}
        response = client.post("/agent/chat/stream", json={"message": "query"}, headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "login_required"
        assert calls == [] and model.seen == []


@pytest.mark.parametrize("body", [{}, {"message": " "}, {"message": "x" * 1001},
                                  {"message": "query", "token": "private"}])
def test_stream_validates_same_request_before_work(body):
    with chat_client(model=ScriptedModel([])) as (client, app, calls):
        response = client.post("/agent/chat/stream", json=body, headers={"Authorization": AUTH})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert "private" not in response.text


def test_stream_returns_same_committed_public_reply_and_headers():
    replies = [AIMessage(content="", tool_calls=[call("search_products", {"keyword": "phone"})]),
               AIMessage(content="")]
    with chat_client(model=ScriptedModel(replies * 2)) as (client, app, calls):
        ordinary = client.post("/agent/chat", json={"message": "phone"}, headers={"Authorization": AUTH}).json()
        streamed = client.post("/agent/chat/stream", json={"message": "phone"}, headers={"Authorization": AUTH})
        assert streamed.status_code == 200
        assert streamed.headers["Content-Type"].startswith("text/event-stream")
        assert streamed.headers["Cache-Control"] == "no-cache"
        assert streamed.headers["X-Accel-Buffering"] == "no"
        assert len(streamed.headers["X-Request-ID"]) == 32
        observed = events(streamed.text)
        assert [kind for kind, _ in observed] == ["status", "text", "products", "sources", "done"]
        assert observed[0] == ("status", {"status": "processing"})
        final = observed[-1][1]
        assert app.state.sessions.load(final["session_id"], 33).candidates
        ordinary.pop("session_id")
        comparable = dict(final)
        comparable.pop("session_id")
        for value in (ordinary, comparable):
            for trace in value["tool_calls"]:
                trace.pop("elapsed_ms")
        assert comparable == ordinary
        assert "private" not in streamed.text and AUTH not in streamed.text
        assert all(request.method == "GET" for request in calls)


def test_stream_pending_card_is_saved_without_any_cart_write():
    database = SERVICE_ENV_FILE.parent / ".tmp" / f"day8-stream-card-{uuid4().hex}" / "sessions.sqlite3"
    with client_for(database) as (client, app, shop):
        sid = selected_session(app)
        response = client.post("/agent/chat/stream", json={
            "message": "\u8fd9\u4e2a\u89c4\u683c\u52a0\u5165\u4e00\u4ef6\u8d2d\u7269\u8f66", "session_id": sid,
        }, headers={"Authorization": AUTH})
        observed = events(response.text)
        final = observed[-1][1]
        assert final["status"] == "awaiting_confirmation"
        assert ("confirmation", {"pending_action": final["pending_action"]}) in observed
        recovered = client.get(f"/agent/actions/{final['pending_action']['action_id']}",
                               headers={"Authorization": AUTH}).json()
        assert recovered["pending_action"] == final["pending_action"]
        assert shop.writes == []


def test_stream_cors_and_expired_identity():
    with chat_client(cors=True, expired=True, model=ScriptedModel([])) as (client, app, calls):
        headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "Authorization, Content-Type"}
        assert client.options("/agent/chat/stream", headers=headers).status_code == 200
        response = client.post("/agent/chat/stream", json={"message": "query"},
                               headers={"Authorization": AUTH, "Origin": headers["Origin"]})
        assert response.status_code == 401
        assert response.headers["Access-Control-Allow-Origin"] == headers["Origin"]
        assert len(calls) == 1


@pytest.mark.parametrize("error", [ShopError("model_timeout"), RuntimeError("private exception")])
def test_stream_errors_are_fixed_public_and_never_done(error):
    with chat_client(model=ScriptedModel([error])) as (client, app, calls):
        response = client.post("/agent/chat/stream", json={"message": "query"}, headers={"Authorization": AUTH})
        observed = events(response.text)
        assert [kind for kind, _ in observed] == ["status", "error"]
        assert observed[-1][1]["error"]["code"] in {"model_timeout", "model_unavailable"}
        assert "private" not in response.text and AUTH not in response.text
        assert len(calls) == 1


def public_reply():
    return SessionChatResponse(session_id="a" * 32, answer="verified", status="needs_input", model_rounds=0)


async def test_processing_arrives_before_work_finishes_and_heartbeat_does_not_cancel_work():
    from app.api.streaming import stream_chat

    released = asyncio.Event()
    completed = asyncio.Event()

    async def work():
        await released.wait()
        completed.set()
        return public_reply()

    stream = stream_chat(work, heartbeat_seconds=0.01)
    assert events(await anext(stream)) == [("status", {"status": "processing"})]
    assert (await anext(stream)).startswith(": heartbeat")
    assert not completed.is_set()
    released.set()
    rest = "".join([frame async for frame in stream])
    assert events(rest)[-1][1] == public_reply().model_dump(mode="json")


async def test_closing_stream_cancels_and_awaits_owned_work():
    from app.api.streaming import stream_chat

    started, cleaned = asyncio.Event(), asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            with anyio.CancelScope(shield=True):
                await asyncio.sleep(0)
                cleaned.set()

    stream = stream_chat(work)
    await anext(stream)
    await started.wait()
    await stream.aclose()
    assert cleaned.is_set()


async def test_stream_cancellation_releases_actual_session_lease():
    from app.api.routes import chat
    from app.api.dependencies import RequestIdentity
    from app.api.streaming import stream_chat
    from app.schemas import UserIdentity
    from starlette.requests import Request

    database = SERVICE_ENV_FILE.parent / ".tmp" / f"day8-stream-lease-{uuid4().hex}" / "sessions.sqlite3"
    started = asyncio.Event()

    class WaitingModel:
        async def ainvoke(self, messages, tools):
            started.set()
            await asyncio.Event().wait()

    with client_for(database) as (client, app, shop):
        sid = app.state.sessions.create(33)
        app.state.chat_model = WaitingModel()
        request = Request({"type": "http", "app": app})
        identity = RequestIdentity(user=UserIdentity(user_id=33, nickname="test"), authorization=AUTH)
        from app.agent.schemas import ChatRequest
        stream = stream_chat(lambda: chat(ChatRequest(message="query", session_id=sid), request, identity))
        await anext(stream)
        await asyncio.wait_for(started.wait(), 2)
        await stream.aclose()
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.release(lease)


async def test_simultaneous_streams_do_not_share_results():
    from app.api.streaming import stream_chat

    async def collect(sid):
        async def work():
            await asyncio.sleep(0)
            return public_reply().model_copy(update={"session_id": sid})
        return events("".join([part async for part in stream_chat(work)]))[-1][1]

    values = await asyncio.gather(collect("a" * 32), collect("b" * 32))
    assert [value["session_id"] for value in values] == ["a" * 32, "b" * 32]


async def test_cancel_while_claiming_does_not_orphan_the_actual_lease():
    from app.api.routes import chat
    from app.api.dependencies import RequestIdentity
    from app.api.streaming import stream_chat
    from app.agent.schemas import ChatRequest
    from app.schemas import UserIdentity
    from starlette.requests import Request

    database = SERVICE_ENV_FILE.parent / ".tmp" / f"day8-stream-claim-{uuid4().hex}" / "sessions.sqlite3"
    claimed, unblock = threading.Event(), threading.Event()
    with client_for(database) as (client, app, shop):
        sid = app.state.sessions.create(33)
        original = app.state.sessions.claim

        def delayed_claim(*args):
            lease = original(*args)
            claimed.set()
            assert unblock.wait(5)
            return lease

        app.state.sessions.claim = delayed_claim
        request = Request({"type": "http", "app": app})
        identity = RequestIdentity(user=UserIdentity(user_id=33), authorization=AUTH)
        stream = stream_chat(lambda: chat(ChatRequest(message="query", session_id=sid), request, identity))
        await anext(stream)
        assert await asyncio.to_thread(claimed.wait, 2)
        closing = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        unblock.set()
        await asyncio.wait_for(closing, 2)
        app.state.sessions.claim = original
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.release(lease)


async def test_starlette_disconnect_waits_for_real_lease_release():
    from app.api.routes import chat_stream
    from app.api.dependencies import RequestIdentity
    from app.agent.schemas import ChatRequest
    from app.schemas import UserIdentity
    from starlette.requests import Request

    started, disconnected = asyncio.Event(), asyncio.Event()
    releasing, unblock = threading.Event(), threading.Event()
    database = SERVICE_ENV_FILE.parent / ".tmp" / f"day8-stream-disconnect-{uuid4().hex}" / "sessions.sqlite3"

    class WaitingModel:
        async def ainvoke(self, messages, tools):
            started.set()
            await asyncio.Event().wait()

    with client_for(database) as (client, app, shop):
        sid = app.state.sessions.create(33)
        app.state.chat_model = WaitingModel()
        original = app.state.sessions.release

        def delayed_release(lease):
            releasing.set()
            assert unblock.wait(5)
            original(lease)

        app.state.sessions.release = delayed_release
        scope = {"type": "http", "app": app, "asgi": {"spec_version": "2.0"}}
        request = Request(scope)
        identity = RequestIdentity(user=UserIdentity(user_id=33), authorization=AUTH)
        response = await chat_stream(ChatRequest(message="query", session_id=sid), request, identity)

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        sent = []

        async def send(message):
            sent.append(message)

        running = asyncio.create_task(response(scope, receive, send))
        await asyncio.wait_for(started.wait(), 2)
        disconnected.set()
        try:
            assert await asyncio.to_thread(releasing.wait, 2)
            assert not running.done()
        finally:
            unblock.set()
            await asyncio.wait_for(running, 2)
        app.state.sessions.release = original
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.release(lease)
        assert b"event: done" not in b"".join(value.get("body", b"") for value in sent)


async def test_starlette_send_side_disconnect_closes_work_before_response_returns():
    from app.api.routes import chat_stream
    from app.api.dependencies import RequestIdentity
    from app.agent.schemas import ChatRequest
    from app.schemas import UserIdentity
    from starlette.requests import Request
    from starlette.requests import ClientDisconnect

    started = asyncio.Event()
    cleaned = threading.Event()
    database = SERVICE_ENV_FILE.parent / ".tmp" / f"day8-stream-send-{uuid4().hex}" / "sessions.sqlite3"

    class WaitingModel:
        async def ainvoke(self, messages, tools):
            started.set()
            await asyncio.Event().wait()

    with client_for(database) as (client, app, shop):
        sid = app.state.sessions.create(33)
        app.state.chat_model = WaitingModel()
        original = app.state.sessions.release

        def tracked_release(lease):
            original(lease)
            cleaned.set()

        app.state.sessions.release = tracked_release
        scope = {"type": "http", "app": app, "asgi": {"spec_version": "2.4"}}
        request = Request(scope)
        identity = RequestIdentity(user=UserIdentity(user_id=33), authorization=AUTH)
        response = await chat_stream(ChatRequest(message="query", session_id=sid), request, identity)

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                await asyncio.wait_for(started.wait(), 2)
                raise OSError("simulated disconnect")

        try:
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
            assert cleaned.is_set(), "send-side disconnect must await lease cleanup"
        finally:
            await response.body_iterator.aclose()
        app.state.sessions.release = original
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.release(lease)
        assert shop.writes == []


@pytest.mark.parametrize("operation", ["commit", "prepare"])
async def test_disconnect_during_storage_awaits_the_original_work(operation):
    from app.api.routes import chat
    from app.api.dependencies import RequestIdentity
    from app.api.streaming import stream_chat
    from app.agent.schemas import ChatRequest
    from app.schemas import UserIdentity
    from starlette.requests import Request

    committing, unblock, committed = threading.Event(), threading.Event(), threading.Event()
    database = SERVICE_ENV_FILE.parent / ".tmp" / f"day8-stream-commit-{uuid4().hex}" / "sessions.sqlite3"
    with client_for(database) as (client, app, shop):
        sid = selected_session(app) if operation == "prepare" else app.state.sessions.create(33)
        app.state.chat_model = ScriptedModel([AIMessage(content="verified")])
        original = getattr(app.state.sessions, operation)

        def delayed_commit(*args):
            committing.set()
            assert unblock.wait(5)
            original(*args)
            committed.set()

        setattr(app.state.sessions, operation, delayed_commit)
        request = Request({"type": "http", "app": app})
        identity = RequestIdentity(user=UserIdentity(user_id=33), authorization=AUTH)
        message = "\u8fd9\u4e2a\u89c4\u683c\u52a0\u5165\u4e00\u4ef6\u8d2d\u7269\u8f66" if operation == "prepare" else "query"
        stream = stream_chat(lambda: chat(ChatRequest(message=message, session_id=sid), request, identity))
        await anext(stream)
        assert await asyncio.to_thread(committing.wait, 2)
        closing = asyncio.create_task(stream.aclose())
        try:
            await asyncio.sleep(0.05)
            assert not closing.done(), "response closure must own the in-flight commit"
        finally:
            unblock.set()
            await asyncio.wait_for(closing, 2)
        assert committed.is_set()
        setattr(app.state.sessions, operation, original)
        lease = app.state.sessions.claim(sid, 33)
        app.state.sessions.release(lease)
        assert shop.writes == []
