"""Assistant chat: request and the server-sent events streamed back.

Events are part of the OpenAPI spec (as the 200 response schema of the chat route), so the web
client's types for them are generated like every other API type.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.portfolio import PortfolioIn

MAX_MESSAGE_CHARS = 2_000
MAX_HISTORY = 12


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    """The conversation so far (last message from the user) and the portfolio on screen."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_HISTORY)
    portfolio: PortfolioIn | None = None


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    id: str
    name: str
    ok: bool
    output: dict[str, Any]


class Citation(BaseModel):
    document: str
    title: str
    page: int
    url: str | None = None


class AnswerEvent(BaseModel):
    type: Literal["answer"] = "answer"
    content: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = Field(description="Every number in the answer was found in a source.")
    ungrounded_numbers: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list, description="Guardrail actions taken.")
    disclaimer: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    trace_id: str
    steps: int
    prompt_tokens: int
    completion_tokens: int


ChatEvent = Annotated[
    ToolCallEvent | ToolResultEvent | AnswerEvent | ErrorEvent | DoneEvent,
    Field(discriminator="type"),
]


class ChatEventEnvelope(BaseModel):
    """One SSE `data:` payload."""

    event: ChatEvent


class ScenarioParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=3, max_length=1_000, examples=["IT stocks fall 15%, market -5%"])
