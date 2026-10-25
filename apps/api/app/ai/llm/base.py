"""The one interface the AI layer uses to talk to a chat model.

Messages and tools use the OpenAI chat-completions shapes, which Groq, OpenAI, Gemini's and
Anthropic's OpenAI-compatible endpoints and local servers all accept, so swapping provider is a
configuration change (base URL, model id, key), not a code change. Tests use `FakeLLMClient`.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model produced it (may be invalid JSON)

    def parsed(self) -> dict[str, Any]:
        """Arguments as a dict; raises ValueError on invalid JSON or a non-object."""
        value = json.loads(self.arguments or "{}")
        if not isinstance(value, dict):
            raise ValueError("tool arguments must be a JSON object")
        return value


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # role == "tool": which call this answers

    def to_api(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {"name": t.name, "arguments": t.arguments},
                }
                for t in self.tool_calls
            ]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        return out


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMResponse:
    message: Message  # role "assistant": content and/or tool calls
    finish_reason: str | None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    cached: bool = False


class LLMError(RuntimeError):
    """The provider failed after retries (or answered with something unusable)."""


class QuotaExhausted(LLMError):
    """Rate limit or quota still exceeded after the retries (HTTP 429)."""


class ToolUseFailed(LLMError):
    """The provider rejected the model's own tool call (e.g. Groq validates tool arguments
    against the schema and answers 400 `tool_use_failed`). Recoverable: tell the model."""


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...
