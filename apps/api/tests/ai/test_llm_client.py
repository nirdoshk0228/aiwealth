"""OpenAI-compatible client against httpx.MockTransport: request shape, parsing, retries."""

import json

import httpx
import pytest

from app.ai.llm.base import LLMError, Message, QuotaExhausted, ToolCall
from app.ai.llm.cache import CachedLLMClient
from app.ai.llm.fake import FakeLLMClient, call, reply
from app.ai.llm.openai_compat import OpenAICompatibleClient

KEY = "sk-test-secret-value"


def ok_body(message: dict[str, object]) -> dict[str, object]:
    return {
        "model": "m",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def client(handler, sleeps: list[float] | None = None, retries: int = 3) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://llm.example/v1/",
        api_key=KEY,
        model="m",
        max_retries=retries,
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_request_shape_and_tool_call_parsing() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json=ok_body(
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "private chain of thought",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "get_risk_summary",
                                "arguments": '{"confidence":0.99}',
                            },
                        }
                    ],
                }
            ),
        )

    tools = [{"type": "function", "function": {"name": "get_risk_summary", "parameters": {}}}]
    res = client(handler).chat([Message("user", "hi")], tools=tools)
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    body = seen["body"]
    assert body["model"] == "m" and body["temperature"] == 0.0
    assert body["tools"] == tools and body["tool_choice"] == "auto"
    assert "response_format" not in body
    assert res.message.tool_calls == (ToolCall("c1", "get_risk_summary", '{"confidence":0.99}'),)
    assert res.message.tool_calls[0].parsed() == {"confidence": 0.99}
    assert (res.usage.prompt_tokens, res.usage.completion_tokens) == (11, 7)
    assert "reasoning" not in json.dumps(res.message.to_api())  # dropped, never sent back


def test_json_mode_sets_response_format() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.content)["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=ok_body({"role": "assistant", "content": "{}"}))

    assert client(handler).chat([Message("user", "x")], json_mode=True).message.content == "{}"


def test_retries_429_with_retry_after_then_succeeds() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "2.5"}, json={"error": "rate"})
        return httpx.Response(200, json=ok_body({"role": "assistant", "content": "done"}))

    res = client(handler, sleeps).chat([Message("user", "x")])
    assert res.message.content == "done" and calls["n"] == 3
    assert sleeps == [2.5, 2.5]


def test_429_after_all_retries_is_quota_exhausted_and_hides_the_key() -> None:
    sleeps: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": f"key {KEY} over quota"}})

    with pytest.raises(QuotaExhausted) as info:
        client(handler, sleeps, retries=2).chat([Message("user", "x")])
    assert KEY not in str(info.value)
    assert sleeps == [1.0, 2.0]  # exponential backoff without retry-after


def test_400_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(LLMError, match="HTTP 400"):
        client(handler).chat([Message("user", "x")])
    assert calls["n"] == 1


def test_unexpected_shape_is_an_llm_error() -> None:
    with pytest.raises(LLMError, match="unexpected response shape"):
        client(lambda r: httpx.Response(200, json={"nope": 1})).chat([Message("user", "x")])


def test_disk_cache_hits_without_calling_the_model(tmp_path) -> None:
    fake = FakeLLMClient([reply("first"), call("get_factor_exposures")])
    cached = CachedLLMClient(fake, tmp_path)
    msgs = [Message("user", "hello")]
    a = cached.chat(msgs)
    b = cached.chat(msgs)  # same request: from disk
    assert a.message.content == b.message.content == "first"
    assert (cached.hits, cached.misses, len(fake.requests)) == (1, 1, 1)
    assert b.cached and not a.cached
    assert cached.tokens_used == 15
    c = cached.chat([Message("user", "other")])  # different request: calls the model
    assert c.message.tool_calls[0].name == "get_factor_exposures"
    again = cached.chat([Message("user", "other")])
    assert again.message.tool_calls == c.message.tool_calls


def test_tool_use_failed_is_a_distinct_error_with_the_reason() -> None:
    from app.ai.llm.base import ToolUseFailed

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": "tool_use_failed", "message": f"schema mismatch ({KEY})"}},
        )

    with pytest.raises(ToolUseFailed) as info:
        client(handler).chat([Message("user", "x")])
    assert "schema mismatch" in str(info.value) and KEY not in str(info.value)
