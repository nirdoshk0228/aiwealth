"""Trace of one assistant turn: every prompt, model response, tool call and retrieved chunk.

Written as JSON lines to <ai_trace_dir>/ai_trace.jsonl (gitignored) and kept in memory so the
chat endpoint can return it for the trace view. Secrets are never part of what is recorded
(requests are recorded as messages and tool calls, not HTTP headers).
"""

import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("aiwealth.ai.trace")
_write_lock = threading.Lock()


class Trace:
    def __init__(self, directory: Path | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []
        self._path = directory / "ai_trace.jsonl" if directory is not None else None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def record(self, kind: str, **data: Any) -> None:
        event = {"trace_id": self.id, "at": datetime.now(UTC).isoformat(), "kind": kind, **data}
        self.events.append(event)
        logger.debug("%s", kind)
        if self._path is not None:
            line = json.dumps(event, ensure_ascii=False, default=str)
            try:
                with _write_lock:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    with self._path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")
            except OSError:
                logger.warning("could not write the AI trace file")

    def add_usage(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
