"""Disk cache around an LLMClient, so evaluation re-runs cost no free-tier quota.

Key = sha256 of the full request (model, messages, tools, mode, temperature). A hit returns the
stored response with `cached=True`. Files live in a gitignored directory (default
<repo>/.cache/llm). Only for offline evaluation; the chat endpoint never caches answers.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.ai.llm.base import LLMClient, LLMResponse, Message, ToolCall, Usage


class CachedLLMClient:
    def __init__(self, inner: LLMClient, directory: Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.directory = directory
        self.hits = 0
        self.misses = 0
        self.tokens_used = 0  # tokens of non-cached calls only

    def _key(self, payload: dict[str, Any]) -> str:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
        payload = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "tools": tools,
            "tool_choice": tool_choice,
            "json_mode": json_mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        path = self.directory / f"{self._key(payload)}.json"
        if path.is_file():
            self.hits += 1
            return _load(json.loads(path.read_text(encoding="utf-8")))
        self.misses += 1
        res = self.inner.chat(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.tokens_used += res.usage.total
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(res), ensure_ascii=False), encoding="utf-8")
        return res


def _load(d: dict[str, Any]) -> LLMResponse:
    m = d["message"]
    return LLMResponse(
        message=Message(
            role=m["role"],
            content=m["content"],
            tool_calls=tuple(ToolCall(**t) for t in m["tool_calls"]),
            tool_call_id=m["tool_call_id"],
        ),
        finish_reason=d["finish_reason"],
        usage=Usage(**d["usage"]),
        model=d["model"],
        cached=True,
    )
