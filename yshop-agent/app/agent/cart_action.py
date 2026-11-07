"""Owned immutable cart actions with request-local durable interrupt/resume graphs."""

import asyncio
import json
import math
import time
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.actions.intent import resolve_cart_intent
from app.actions.schemas import (
    ACTION_TTL_SECONDS, ActionDecision, ActionResponse, CartActionCard, CartActionResult,
)
from app.agent.state import ConversationState, MAX_HISTORY_MESSAGES, MAX_SNAPSHOT_BYTES, PublicMessage
from app.agent.tools import QueryTools
from app.errors import ShopError
from app.schemas import CartAddInput, CartSnapshot, ProductDetail
from app.storage.actions import ActionRecord, CartActionStore
from app.storage.sessions import SessionLease


class CartGraphState(TypedDict, total=False):
    action_id: str
    card: dict
    decision: dict
    response: dict


_ANSWERS = {
    "awaiting_confirmation": "请确认是否将此商品规格和数量加入购物车。",
    "executing": "加购正在处理，请勿重复提交。",
    "succeeded": "已加入购物车。",
    "cancelled": "已取消加购。",
    "failed": "加购未执行或未成功，请检查商品和购物车。",
    "unknown": "加购结果暂时无法确认，请查看购物车，勿重复提交。",
}
_INPUT = {
    "product_selection": "请明确选择最近展示的一个商品。",
    "sku_selection": "请明确选择该商品的一个规格。",
    "quantity": "请明确加购数量，数量须为1至9999的整数。",
}


def _bounded_json(value) -> None:
    def visit(item, depth=0):
        if depth > 12:
            raise ValueError("checkpoint depth")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is str and len(item) <= 16384:
            return
        if type(item) is list and len(item) <= 64:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict and len(item) <= 64 and all(type(key) is str for key in item):
            for key, child in item.items():
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        raise ValueError("non-public checkpoint")
    visit(value)
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > 32768:
        raise ValueError("checkpoint size")


class CartActionService:
    def __init__(self, store: CartActionStore, checkpointer):
        self.store = store
        self.checkpointer = checkpointer

    @staticmethod
    def _config(action_id):
        return {"configurable": {"thread_id": action_id}}

    @staticmethod
    def _response(record: ActionRecord, tools: QueryTools) -> ActionResponse:
        if record.status == "pending":
            response = ActionResponse(session_id=record.card.session_id, status="awaiting_confirmation",
                                      answer=_ANSWERS["awaiting_confirmation"], pending_action=record.card)
        else:
            result = record.result or CartActionResult(
                action_id=record.card.action_id, session_id=record.card.session_id, status="executing")
            response = ActionResponse(session_id=record.card.session_id, status=result.status,
                                      answer=_ANSWERS[result.status], action_result=result)
        return tools.public_model(response)

    def graph(self, user_id: int, authorization: str, tools: QueryTools, *,
              lease: SessionLease | None = None, action_id: str | None = None,
              approved: ActionDecision | None = None):
        """Transport identity and execution rights live only in these fresh closures."""
        def owned_id(state):
            value = state["action_id"]
            if action_id is not None and value != action_id:
                raise ShopError("action_conflict")
            return value

        async def prepare(state):
            record = await run_in_threadpool(self.store.get, owned_id(state), user_id)
            if record.status != "pending":
                raise ShopError("action_conflict")
            card = tools.public_model(record.card).model_dump(mode="json")
            _bounded_json(card)
            return {"card": card}

        def wait(state):
            return {"decision": interrupt(state["card"])}

        def choose(state):
            try:
                decision = ActionDecision.model_validate(state["decision"])
                if approved is None or decision != approved or lease is None:
                    raise ValueError("no current authenticated approval")
            except (ValueError, TypeError):
                raise ShopError("invalid_request") from None
            return "execute" if decision.decision == "confirm" else "cancel"

        async def execute(state):
            response = await self._execute(owned_id(state), user_id, authorization, tools, lease)
            return {"response": response.model_dump(mode="json")}

        async def cancel(state):
            record = await run_in_threadpool(self.store.get, owned_id(state), user_id)
            if record.status != "pending":
                raise ShopError("action_conflict")
            record = await self._finish(lease, user_id, record.card, "cancelled")
            return {"response": self._response(record, tools).model_dump(mode="json")}

        builder = StateGraph(CartGraphState)
        builder.add_node("prepare", prepare)
        builder.add_node("wait", wait)
        builder.add_node("execute", execute)
        builder.add_node("cancel", cancel)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "wait")
        builder.add_conditional_edges("wait", choose, ["execute", "cancel"])
        builder.add_edge("execute", END)
        builder.add_edge("cancel", END)
        return builder.compile(checkpointer=self.checkpointer)

    async def _ensure_wait(self, record, compiled, tools):
        if record.status != "pending":
            raise ShopError("action_conflict")
        action_id = record.card.action_id
        config = self._config(action_id)
        card = tools.public_model(record.card).model_dump(mode="json")
        _bounded_json(card)

        def waiting(snapshot):
            try:
                _bounded_json(snapshot.values)
            except ValueError:
                return False
            return (
                snapshot.values == {"action_id": action_id, "card": card}
                and snapshot.next == ("wait",)
                and len(snapshot.tasks) == 1
                and snapshot.tasks[0].name == "wait"
                and len(snapshot.tasks[0].interrupts) == 1
                and snapshot.tasks[0].interrupts[0].value == card
            )

        try:
            snapshot = await compiled.aget_state(config)
            if not waiting(snapshot):
                # Only an owned pending action can discard stale decisions and restart at prepare.
                await self.checkpointer.adelete_thread(action_id)
                await compiled.ainvoke({"action_id": action_id}, config)
                if not waiting(await compiled.aget_state(config)):
                    raise ValueError("missing durable interrupt")
        except ShopError:
            raise
        except Exception:
            raise ShopError("session_unavailable") from None

    async def _pause(self, record, user_id, authorization, tools, *, lease=None):
        if record.status != "pending":
            return self._response(record, tools)
        if self.checkpointer is None:
            raise ShopError("session_unavailable")
        owns_lease = lease is None
        if owns_lease:
            lease = await run_in_threadpool(self.store.claim, record.card.session_id, user_id)
        try:
            record = await run_in_threadpool(self.store.get, record.card.action_id, user_id)
            if record.status == "pending":
                compiled = self.graph(user_id, authorization, tools, action_id=record.card.action_id)
                await self._ensure_wait(record, compiled, tools)
            return self._response(record, tools)
        finally:
            if owns_lease:
                try:
                    await asyncio.shield(run_in_threadpool(self.store.release, lease))
                except ShopError:
                    pass

    @staticmethod
    def _cart_quantity(cart: CartSnapshot, product_id: int, unique_id: str):
        if not isinstance(cart, CartSnapshot):
            raise ShopError("shop_invalid_response")
        cart = CartSnapshot.model_validate(cart.model_dump(mode="json"))
        valid = [item for item in cart.valid if item.product_id == product_id and item.unique_id == unique_id]
        invalid = [item for item in cart.invalid if item.product_id == product_id and item.unique_id == unique_id]
        if invalid:
            raise ShopError("cart_item_invalid")
        if len(valid) > 1:
            raise ShopError("cart_state_ambiguous")
        return valid[0].quantity if valid else 0

    async def _preflight(self, product_id, unique_id, quantity, authorization, tools):
        product = await tools.get_cart_product(product_id, unique_id, authorization)
        if not isinstance(product, ProductDetail):
            raise ShopError("shop_invalid_response")
        product = ProductDetail.model_validate(product.model_dump(mode="json"))
        matches = [sku for sku in product.skus if sku.unique_id == unique_id]
        if product.product_id != product_id or len(matches) != 1:
            raise ShopError("sku_unavailable")
        cart = tools.public_model(await tools._shop.get_cart(authorization))
        previous = self._cart_quantity(cart, product_id, unique_id)
        if previous + quantity > matches[0].stock:
            raise ShopError("insufficient_stock")
        safe_product = tools.public_model(product)
        safe_sku = next((sku for sku in safe_product.skus if sku.unique_id == unique_id), None)
        if safe_sku is None:
            raise ShopError("sku_unavailable")
        return safe_product, safe_sku, previous

    @staticmethod
    def _card(session_id, product, sku, quantity):
        try:
            return CartActionCard(action_id=uuid4().hex, session_id=session_id,
                product_id=product.product_id, product_name=product.name, unique_id=sku.unique_id,
                sku_name=sku.name, quantity=quantity, unit_price=sku.price, total_price=sku.price * quantity,
                image=sku.image or product.image, expires_at=time.time() + ACTION_TTL_SECONDS)
        except ValidationError:
            raise ShopError("shop_invalid_response") from None

    @staticmethod
    def _history(state, message, answer, tools):
        state = tools.public_model(state, error_code="invalid_request")
        state.history = [*state.history, PublicMessage(role="user", content=tools.public_data(message)),
                         PublicMessage(role="assistant", content=answer)][-MAX_HISTORY_MESSAGES:]
        while len(state.history) > 2 and len(state.model_dump_json().encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            state.history = state.history[2:]
        return ConversationState.model_validate(state.model_dump(mode="json"))

    async def prepare_chat(self, lease, user_id, message, authorization, tools):
        active = await run_in_threadpool(self.store.pending, lease.session_id, user_id)
        if active is not None:
            if active.status == "executing":
                active = await self._finish(
                    lease, user_id, active.card, "unknown", error_code="cart_result_unknown")
                return self._response(active, tools)
            return await self._pause(active, user_id, authorization, tools, lease=lease)
        state = tools.public_model(lease.state, error_code="invalid_request")
        intent = resolve_cart_intent(tools.public_data(message), state)
        if intent is None:
            return None
        if intent.reason:
            response = ActionResponse(session_id=lease.session_id, status="needs_input", answer=_INPUT[intent.reason])
            response = tools.public_model(response)
            state = self._history(state, message, response.answer, tools)
            await run_in_threadpool(self.store.commit, lease, state)
            return response
        if self.checkpointer is None:
            raise ShopError("session_unavailable")
        product, sku, _ = await self._preflight(intent.product_id, intent.unique_id, intent.quantity,
                                               authorization, tools)
        card = self._card(lease.session_id, product, sku, intent.quantity)
        state.selected_product_id = product.product_id
        state.selected_sku_id = sku.unique_id
        state = self._history(state, message, _ANSWERS["awaiting_confirmation"], tools)
        record = await run_in_threadpool(self.store.prepare, lease, user_id, state, card)
        return await self._pause(record, user_id, authorization, tools)

    async def _finish(self, lease, user_id, card, status, *, error_code=None, cart_id=None):
        result = CartActionResult(action_id=card.action_id, session_id=card.session_id,
                                  status=status, error_code=error_code, cart_id=cart_id)
        return await run_in_threadpool(self.store.finish, lease, user_id, card.action_id, result)

    async def _execute(self, action_id, user_id, authorization, tools, lease):
        record = await run_in_threadpool(self.store.get, action_id, user_id)
        if record.status != "pending":
            raise ShopError("action_conflict")
        card = record.card
        try:
            product, sku, previous = await self._preflight(
                card.product_id, card.unique_id, card.quantity, authorization, tools)
        except ShopError as error:
            record = await self._finish(lease, user_id, card, "failed", error_code=error.code)
            if error.code == "login_required":
                raise error
            return self._response(record, tools)
        except Exception:
            record = await self._finish(lease, user_id, card, "failed", error_code="shop_invalid_response")
            return self._response(record, tools)
        if sku.price != card.unit_price or sku.name != tools.public_model(card).sku_name:
            fresh = self._card(card.session_id, product, sku, card.quantity)
            state = tools.public_model(lease.state, error_code="invalid_request")
            record = await run_in_threadpool(self.store.replace, lease, user_id, card.action_id, state, fresh)
            return self._response(record, tools)
        # This durable CAS is the write fence. No code path retries cart/add.
        await run_in_threadpool(self.store.begin_execution, lease, user_id, action_id)
        status, code, cart_id = "unknown", "cart_result_unknown", None
        try:
            added = await tools._shop.add_to_cart(CartAddInput(
                product_id=card.product_id, unique_id=card.unique_id, quantity=card.quantity), authorization)
            cart = await tools._shop.get_cart(authorization)
            expected_quantity = previous + card.quantity
            actual_quantity = self._cart_quantity(cart, card.product_id, card.unique_id)
            matches = [item for item in cart.valid if item.cart_id == added.cart_id
                       and item.product_id == card.product_id and item.unique_id == card.unique_id
                       and item.quantity == expected_quantity]
            if actual_quantity != expected_quantity or len(matches) != 1:
                raise ShopError("cart_verification_failed", outcome="unknown")
            status, code, cart_id = "succeeded", None, added.cart_id
        except ShopError as error:
            status = "failed" if error.outcome == "failed" else "unknown"
            code = error.code
        except asyncio.CancelledError:
            try:
                await asyncio.shield(self._finish(lease, user_id, card, "unknown", error_code="cart_result_unknown"))
            except Exception:
                pass
            raise
        except Exception:
            pass
        record = await self._finish(lease, user_id, card, status, error_code=code, cart_id=cart_id)
        return self._response(record, tools)

    async def get(self, action_id, user_id, tools):
        record = await run_in_threadpool(self.store.get, action_id, user_id)
        return self._response(record, tools)

    async def decide(self, action_id, user_id, authorization, tools, decision: ActionDecision):
        record = await run_in_threadpool(self.store.get, action_id, user_id)
        if record.status not in {"pending", "executing"}:
            return self._response(record, tools)
        lease = await run_in_threadpool(self.store.claim, record.card.session_id, user_id)
        try:
            record = await run_in_threadpool(self.store.get, action_id, user_id)
            if record.status == "executing":
                record = await self._finish(lease, user_id, record.card, "unknown", error_code="cart_result_unknown")
                return self._response(record, tools)
            if record.status != "pending":
                return self._response(record, tools)
            if record.card.expires_at <= time.time():
                await self._finish(lease, user_id, record.card, "failed", error_code="action_expired")
                raise ShopError("action_expired")
            if self.checkpointer is None:
                raise ShopError("session_unavailable")
            compiled = self.graph(user_id, authorization, tools, lease=lease,
                                  action_id=action_id, approved=decision)
            try:
                await self._ensure_wait(record, compiled, tools)
                await compiled.ainvoke(Command(resume=decision.model_dump(mode="json")), self._config(action_id))
            except ShopError:
                raise
            except Exception:
                raise ShopError("session_unavailable") from None
            # The action table, not a possibly stale final checkpoint, supplies the outcome.
            current = await run_in_threadpool(self.store.get, action_id, user_id)
            active = await run_in_threadpool(self.store.pending, record.card.session_id, user_id)
            if current.status == "cancelled" and active is not None and active.card.action_id != action_id:
                return await self._pause(active, user_id, authorization, tools)
            return self._response(current, tools)
        finally:
            try:
                await asyncio.shield(run_in_threadpool(self.store.release, lease))
            except ShopError:
                pass
