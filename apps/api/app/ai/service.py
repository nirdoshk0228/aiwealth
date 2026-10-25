"""Entry points of the AI layer used by the router: streamed chat and scenario parsing."""

import json
from collections.abc import Iterator

from app.ai.agent.loop import Finalizer, default_finalize, run_agent
from app.ai.agent.tools import ToolContext
from app.ai.llm.base import LLMClient
from app.ai.rag.kb import KnowledgeBase
from app.ai.scenario_parser import ParsedScenario, parse_scenario
from app.ai.trace import Trace
from app.core.config import Settings
from app.schemas.chat import ChatEventEnvelope, ChatRequest, ErrorEvent
from app.services.market import MarketDataStore


def chat_events(
    client: LLMClient,
    req: ChatRequest,
    store: MarketDataStore,
    settings: Settings,
    *,
    knowledge: KnowledgeBase | None = None,
    finalize: Finalizer = default_finalize,
) -> Iterator[ChatEventEnvelope]:
    if req.messages[-1].role != "user":
        yield ChatEventEnvelope(event=ErrorEvent(message="the last message must be the user's"))
        return
    ctx = ToolContext(
        market=store.get,
        portfolio=req.portfolio,
        knowledge=knowledge.get if knowledge is not None else None,
        rag_mode=settings.rag_mode,
    )
    trace = Trace(settings.ai_trace_dir)
    for event in run_agent(client, req.messages, ctx, trace=trace, finalize=finalize):
        yield ChatEventEnvelope(event=event)


def to_sse(events: Iterator[ChatEventEnvelope]) -> Iterator[str]:
    """Server-sent events: one `data:` line of JSON per event, blank line between events."""
    for env in events:
        payload = env.model_dump(mode="json")
        yield f"event: {env.event.type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def parse(
    client: LLMClient, text: str, store: MarketDataStore, settings: Settings
) -> ParsedScenario:
    md = store.get()
    return parse_scenario(client, text, md.universe, trace=Trace(settings.ai_trace_dir))
