"""LLMClient for any OpenAI-compatible chat-completions endpoint (Groq by default).

Plain httpx (already a dependency), no vendor SDK. Retries with exponential backoff on HTTP 429
and 5xx, honouring `retry-after` when the provider sends it; gives up with QuotaExhausted /
LLMError so callers can tell "out of free quota" from a bug. The API key is only placed in the
Authorization header and never logged or included in an exception message.
"""

import time
from collections.abc import Callable
from typing import Any

import httpx

from app.ai.llm.base import (
    LLMError,
    LLMResponse,
    Message,
    QuotaExhausted,
    ToolCall,
    ToolUseFailed,
    Usage,
)

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 30.0


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_retries: int = 4,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._sleep = sleep
        self._key = api_key
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

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
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return _parse(self._post(body), self.model)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last_status, last_detail = None, ""
        for attempt in range(self._max_retries + 1):
            try:
                r = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise LLMError(f"LLM provider unreachable: {type(exc).__name__}") from None
                self._sleep(min(2.0**attempt, MAX_BACKOFF_SECONDS))
                continue
            if r.status_code == 200:
                return r.json()
            last_status = r.status_code
            last_detail = self._detail(r)
            if r.status_code == 400 and "tool_use_failed" in r.text:
                raise ToolUseFailed(last_detail)
            if r.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                break
            self._sleep(_wait_seconds(r, attempt))
        if last_status == 429:
            suffix = f" ({last_detail})" if last_detail else ""
            raise QuotaExhausted(f"LLM rate limit or free quota exhausted; try again later{suffix}")
        suffix = f": {last_detail}" if last_detail else ""
        raise LLMError(f"LLM provider error (HTTP {last_status}){suffix}")

    def _detail(self, r: httpx.Response) -> str:
        """The provider's error message, truncated, with the API key scrubbed."""
        try:
            err = r.json().get("error", {})
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
        except ValueError:
            msg = r.text
        return str(msg).replace(self._key, "***")[:300]


def _wait_seconds(r: httpx.Response, attempt: int) -> float:
    header = r.headers.get("retry-after")
    if header is not None:
        try:
            return min(float(header), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    return min(2.0**attempt, MAX_BACKOFF_SECONDS)


def _parse(data: dict[str, Any], model: str) -> LLMResponse:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("unexpected response shape from the LLM provider") from exc
    calls = tuple(
        ToolCall(
            id=str(c.get("id", "")),
            name=str(c.get("function", {}).get("name", "")),
            arguments=str(c.get("function", {}).get("arguments", "") or ""),
        )
        for c in (msg.get("tool_calls") or [])
    )
    usage = data.get("usage") or {}
    return LLMResponse(
        # The provider's private `reasoning` field is dropped: never shown, never sent back.
        message=Message(role="assistant", content=msg.get("content"), tool_calls=calls),
        finish_reason=choice.get("finish_reason"),
        usage=Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        ),
        model=str(data.get("model") or model),
    )
