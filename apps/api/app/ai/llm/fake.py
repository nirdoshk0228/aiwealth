"""Scripted LLMClient for tests: returns queued responses and records every request."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.ai.llm.base import LLMResponse, Message, ToolCall, Usage

Responder = Callable[[list[Message], list[dict[str, Any]] | None], LLMResponse]


def reply(content: str) -> LLMResponse:
    return LLMResponse(Message("assistant", content), "stop", Usage(10, 5), "fake")


def call(name: str, arguments: str = "{}", call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        Message("assistant", None, (ToolCall(call_id, name, arguments),)),
        "tool_calls",
        Usage(10, 5),
        "fake",
    )


@dataclass
class FakeLLMClient:
    """Each `chat` call pops the next scripted item (a response, or a function of the request).
    Raises AssertionError when the script runs out, so a test sees unexpected extra calls."""

    script: list[LLMResponse | Responder]
    model: str = "fake-model"
    requests: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.requests.append({"messages": list(messages), "tools": tools, "json_mode": json_mode})
        if not self.script:
            raise AssertionError("FakeLLMClient: no scripted response left")
        item = self.script.pop(0)
        return item(messages, tools) if callable(item) else item
