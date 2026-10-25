"""Assistant routes: streamed chat (SSE), scenario parsing, availability. Rate-limited."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ai import service
from app.ai.llm.base import LLMClient
from app.ai.llm.factory import get_llm_client
from app.api.deps import get_store
from app.core.config import Settings, get_settings
from app.core.ratelimit import RateLimiter
from app.schemas.chat import ChatRequest, ScenarioParseRequest
from app.schemas.stress import HypotheticalScenarioSpec
from app.services.market import MarketDataStore

router = APIRouter(prefix="/ai", tags=["assistant"])

StoreDep = Annotated[MarketDataStore, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ClientDep = Annotated[LLMClient, Depends(get_llm_client)]


def client_key(request: Request) -> str:
    """The client IP; behind the host's proxy the first X-Forwarded-For entry is the client."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    limiter: RateLimiter = request.app.state.chat_limiter
    limiter.check(client_key(request))


class AssistantStatus(BaseModel):
    available: bool
    model: str | None


@router.get("/status", response_model=AssistantStatus)
def assistant_status(settings: SettingsDep) -> AssistantStatus:
    configured = settings.llm_api_key is not None
    return AssistantStatus(available=configured, model=settings.llm_model if configured else None)


@router.post(
    "/chat",
    dependencies=[Depends(rate_limit)],
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events; each `data:` line is a ChatEventEnvelope.",
            "content": {
                "text/event-stream": {"schema": {"$ref": "#/components/schemas/ChatEventEnvelope"}}
            },
        }
    },
)
def chat(
    body: ChatRequest,
    request: Request,
    client: ClientDep,
    store: StoreDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Ask the assistant. Streams tool calls, tool results, the checked answer, then `done`."""
    events = service.chat_events(
        client, body, store, settings, knowledge=request.app.state.knowledge
    )
    return StreamingResponse(
        service.to_sse(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/parse-scenario",
    response_model=HypotheticalScenarioSpec,
    dependencies=[Depends(rate_limit)],
)
def parse_scenario(
    body: ScenarioParseRequest, client: ClientDep, store: StoreDep, settings: SettingsDep
) -> HypotheticalScenarioSpec:
    """Turn a scenario described in words into validated shocks (not run; see /stress/run)."""
    return service.parse(client, body.text, store, settings).spec
