"""Optional provider thinking configuration at the real HTTP adapter boundary."""
import json

import httpx
import pytest
from pydantic import ValidationError

from app.agent.config import AgentSettings
from app.errors import ShopError
from tests.test_agent_model import SECRET, completion, invoke, settings


@pytest.mark.parametrize("enabled", [False, True])
async def test_explicit_thinking_mode_is_a_json_boolean_on_the_wire(enabled):
    requests = []

    def handler(request):
        requests.append(request)
        payload = json.loads(request.content)
        assert payload["enable_thinking"] is enabled
        assert "reasoning_content" not in payload
        return httpx.Response(200, json=completion())

    await invoke(handler, config=settings(llm_enable_thinking=enabled))
    assert len(requests) == 1


async def test_unconfigured_thinking_mode_preserves_generic_provider_wire(monkeypatch):
    monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
    config = settings()
    assert config.llm_enable_thinking is None

    def handler(request):
        assert set(json.loads(request.content)) == {"model", "messages", "tools", "tool_choice"}
        return httpx.Response(200, json=completion())

    await invoke(handler, config=config)


@pytest.mark.parametrize("value,expected", [("false", False), ("TRUE", True), (" false ", False), ("", None)])
def test_thinking_mode_reads_optional_environment_boolean(monkeypatch, value, expected):
    monkeypatch.setenv("LLM_ENABLE_THINKING", value)
    assert AgentSettings(_env_file=None).llm_enable_thinking is expected


@pytest.mark.parametrize("value", [0, 1, 0.0, [], {}, "maybe", SECRET])
def test_thinking_mode_rejects_invalid_values_without_exposing_input(value):
    with pytest.raises(ValidationError) as caught:
        settings(llm_enable_thinking=value)
    assert SECRET not in str(caught.value)


async def test_thinking_off_still_rejects_bad_tool_json_without_retry():
    requests = []

    def handler(request):
        requests.append(request)
        assert json.loads(request.content)["enable_thinking"] is False
        return httpx.Response(200, json=completion({"role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "search_products", "arguments": "{"}}]}))

    with pytest.raises(ShopError) as caught:
        await invoke(handler, config=settings(llm_enable_thinking=False))
    assert caught.value.code == "model_invalid_response"
    assert len(requests) == 1
