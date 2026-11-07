"""Bounded async OpenAI Chat Completions adapter with explicit data allowlists."""

import json
import re
from typing import Protocol

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.config import AgentSettings
from app.errors import ShopError


MAX_TOOL_CALLS = 3
MAX_ARGUMENT_CHARACTERS = 8192
CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


class ChatModel(Protocol):
    async def ainvoke(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage:
        """Return a sanitized assistant message or a public model error."""
        ...


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError("non-finite JSON number")


def _loads(value: str | bytes) -> object:
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _contains_credential(value: object, credential: str) -> bool:
    if isinstance(value, str):
        return credential in value
    if isinstance(value, dict):
        return any(_contains_credential(key, credential) or _contains_credential(item, credential)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_credential(item, credential) for item in value)
    return False


def _parse_tool_calls(calls: object) -> list[dict]:
    if not isinstance(calls, list) or len(calls) > MAX_TOOL_CALLS:
        raise ValueError("invalid tool calls")
    result = []
    seen = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("invalid function call")
        call_id = call.get("id")
        function = call.get("function")
        if not isinstance(call_id, str) or not CALL_ID.fullmatch(call_id) or call_id in seen:
            raise ValueError("invalid call ID")
        if not isinstance(function, dict):
            raise ValueError("invalid function")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not FUNCTION_NAME.fullmatch(name):
            raise ValueError("invalid function name")
        if not isinstance(arguments, str) or len(arguments) > MAX_ARGUMENT_CHARACTERS:
            raise ValueError("invalid function arguments")
        args = _loads(arguments)
        if not isinstance(args, dict):
            raise ValueError("function arguments must be an object")
        # Canonicalization also rejects numeric overflow such as JSON 1e999.
        json.dumps(args, ensure_ascii=False, allow_nan=False)
        seen.add(call_id)
        result.append({"id": call_id, "name": name, "args": args, "type": "tool_call"})
    return result


def _serialize_messages(messages: list[BaseMessage], credential: str) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    result = []
    for message in messages:
        if not isinstance(message, BaseMessage) or not isinstance(message.content, str):
            raise ValueError("only text messages are supported")
        if isinstance(message, SystemMessage):
            item = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            item = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            item = {"role": "assistant", "content": message.content}
            if message.invalid_tool_calls:
                raise ValueError("invalid assistant tool calls")
            if message.tool_calls:
                calls = [{"id": call["id"], "type": "function", "function": {
                    "name": call["name"], "arguments": json.dumps(
                        call["args"], ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                }} for call in message.tool_calls]
                if _contains_credential(_parse_tool_calls(calls), credential):
                    raise ValueError("credential present in assistant tool arguments")
                item["tool_calls"] = calls
        elif isinstance(message, ToolMessage):
            if not CALL_ID.fullmatch(message.tool_call_id):
                raise ValueError("invalid tool response ID")
            item = {"role": "tool", "content": message.content,
                    "tool_call_id": message.tool_call_id}
        else:
            raise ValueError("unsupported message type")
        result.append(item)
    return result


def _parse_usage(usage: object) -> dict | None:
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise ValueError("invalid token usage")
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens
                             if type(input_tokens) is int and type(output_tokens) is int else None)
    if any(type(value) is not int or value < 0
           for value in (input_tokens, output_tokens, total_tokens)):
        raise ValueError("invalid token usage")
    if total_tokens != input_tokens + output_tokens:
        raise ValueError("inconsistent token usage")
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def _parse_reply(payload: object, credential: str) -> AIMessage:
    if not isinstance(payload, dict):
        raise ValueError("invalid completion envelope")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected one completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("invalid assistant message")
    raw_calls = message.get("tool_calls")
    calls = [] if raw_calls is None else _parse_tool_calls(raw_calls)
    content = message.get("content")
    if content is None and calls:
        content = ""
    if not isinstance(content, str):
        raise ValueError("invalid assistant content")
    if _contains_credential(content, credential) or _contains_credential(calls, credential):
        raise ValueError("credential echoed in assistant fields")
    return AIMessage(content=content, tool_calls=calls, usage_metadata=_parse_usage(payload.get("usage")))


class OpenAIChatModel:
    def __init__(self, settings: AgentSettings, http: httpx.AsyncClient) -> None:
        if http.trust_env:
            raise ValueError("Model HTTP client must have trust_env=False")
        self.settings = settings
        self.http = http

    async def ainvoke(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage:
        if not self.settings.configured:
            raise ShopError("model_not_configured") from None
        try:
            credential = self.settings.llm_api_key.get_secret_value()
            payload = {"model": self.settings.llm_model,
                       "messages": _serialize_messages(messages, credential)}
            if self.settings.llm_enable_thinking is not None:
                payload["enable_thinking"] = self.settings.llm_enable_thinking
            if tools:
                payload.update(tools=tools, tool_choice="auto")
            if _contains_credential(payload, credential) or credential in self.settings.llm_base_url:
                raise ValueError("credential present outside Authorization")
            request = httpx.Request(
                "POST", self.settings.llm_base_url + "chat/completions", json=payload,
                headers={"Authorization": "Bearer " + credential},
                extensions={"timeout": httpx.Timeout(self.settings.llm_timeout_seconds).as_dict()},
            )
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise ShopError("invalid_request") from None
        try:
            # A fresh Request avoids inheriting cookies, query parameters or shop headers.
            response = await self.http.send(request, auth=None, follow_redirects=False)
        except httpx.TimeoutException:
            raise ShopError("model_timeout") from None
        except httpx.RequestError:
            raise ShopError("model_unavailable") from None
        if not 200 <= response.status_code < 300:
            raise ShopError("model_rejected") from None
        try:
            return _parse_reply(_loads(response.content), credential)
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
            raise ShopError("model_invalid_response") from None
