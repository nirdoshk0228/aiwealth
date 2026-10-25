"""The tool-calling loop with a scripted model: tool selection, messages, limits, errors."""

import json

import pytest

from app.ai.agent.loop import run_agent
from app.ai.agent.prompts import SYSTEM_PROMPT
from app.ai.agent.tools import TOOLS, ToolContext
from app.ai.llm.base import LLMError, Message
from app.ai.llm.fake import FakeLLMClient, call, reply
from app.ai.trace import Trace
from app.schemas.chat import ChatMessage
from app.schemas.portfolio import PortfolioIn
from app.services.market import MarketData

Q = [ChatMessage(role="user", content="What is my 99% VaR?")]


@pytest.fixture
def ctx(market: MarketData, fund_mix: dict[str, object]) -> ToolContext:
    return ToolContext(market=lambda: market, portfolio=PortfolioIn.model_validate(fund_mix))


def types(events: list) -> list[str]:
    return [e.type for e in events]


def test_tool_call_then_answer(ctx: ToolContext, tmp_path) -> None:
    def quote_var(messages, tools):  # a well-behaved model copies the tool's display string
        var = json.loads(messages[-1].content)["var"]["historical"]["var"]
        return reply(f"Your 1-day 99% historical VaR is {var}.")

    fake = FakeLLMClient([call("get_risk_summary", '{"confidence": 0.99}', "c1"), quote_var])
    trace = Trace(tmp_path)
    events = list(run_agent(fake, Q, ctx, trace=trace))
    assert types(events) == ["tool_call", "tool_result", "answer", "done"]
    assert events[0].name == "get_risk_summary" and events[0].arguments == {"confidence": 0.99}
    assert events[1].ok and events[1].output["confidence"] == "99%"
    assert events[2].content.startswith("Your 1-day 99% historical VaR is ")
    assert events[2].grounded and events[2].ungrounded_numbers == []
    assert events[3].steps == 2 and events[3].prompt_tokens == 20

    first, second = fake.requests
    assert first["messages"][0] == Message("system", SYSTEM_PROMPT)
    context = first["messages"][1]
    assert context.role == "system" and context.content.startswith(
        "(Context) The user's portfolio is loaded"
    )
    assert "8 holdings: Parag Parikh Flexi Cap Fund;" in context.content
    assert first["messages"][2] == Message("user", "What is my 99% VaR?")
    assert {t["function"]["name"] for t in first["tools"]} == {t.name for t in TOOLS}
    # The second request carries the assistant's tool call and the tool result for it.
    tool_msg = second["messages"][-1]
    assert tool_msg.role == "tool" and tool_msg.tool_call_id == "c1"
    assert json.loads(tool_msg.content)["confidence"] == "99%"
    # Trace written as JSON lines: prompt, responses, tool result, answer.
    kinds = [
        json.loads(line)["kind"]
        for line in (tmp_path / "ai_trace.jsonl").read_text("utf-8").splitlines()
    ]
    assert kinds == ["prompt", "model_response", "tool_result", "model_response", "answer"]


def test_tool_error_goes_back_to_the_model(ctx: ToolContext) -> None:
    fake = FakeLLMClient(
        [
            call("run_stress_scenario", '{"scenario_id": "mars"}', "c1"),
            call("run_stress_scenario", '{"scenario_id": "covid_crash_2020"}', "c2"),
            reply("In the COVID replay ..."),
        ]
    )
    events = list(run_agent(fake, Q, ctx))
    results = [e for e in events if e.type == "tool_result"]
    assert [r.ok for r in results] == [False, True]
    assert "unknown scenario" in json.loads(fake.requests[1]["messages"][-1].content)["error"]


def test_several_tool_calls_in_one_turn(ctx: ToolContext) -> None:
    from app.ai.llm.base import LLMResponse, ToolCall, Usage

    both = LLMResponse(
        Message(
            "assistant",
            None,
            (
                ToolCall("a", "get_factor_exposures", "{}"),
                ToolCall("b", "get_risk_contributions", "{}"),
            ),
        ),
        "tool_calls",
        Usage(1, 1),
    )
    fake = FakeLLMClient([both, reply("done")])
    events = list(run_agent(fake, Q, ctx))
    assert types(events) == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "answer",
        "done",
    ]
    assert [m.tool_call_id for m in fake.requests[1]["messages"][-2:]] == ["a", "b"]


def test_step_limit(ctx: ToolContext) -> None:
    fake = FakeLLMClient([call("list_stress_scenarios", "{}", f"c{i}") for i in range(3)])
    events = list(run_agent(fake, Q, ctx, max_steps=3))
    assert types(events)[-2:] == ["error", "done"]
    assert "too many steps" in events[-2].message


def test_llm_failure_becomes_an_error_event(ctx: ToolContext) -> None:
    def boom(messages, tools):
        raise LLMError("LLM provider error (HTTP 503)")

    events = list(run_agent(FakeLLMClient([boom]), Q, ctx))
    assert types(events) == ["error", "done"] and "503" in events[0].message


def test_empty_answer(ctx: ToolContext) -> None:
    events = list(run_agent(FakeLLMClient([reply("  ")]), Q, ctx))
    assert types(events) == ["error", "done"]


def test_custom_finalizer_is_applied(ctx: ToolContext) -> None:
    from app.schemas.chat import AnswerEvent

    seen = {}

    def finalize(answer, outputs, c):
        seen["outputs"] = outputs
        return AnswerEvent(content=answer.upper(), grounded=True, disclaimer="d")

    fake = FakeLLMClient([call("get_factor_exposures", "{}"), reply("beta is 0.5")])
    events = list(run_agent(fake, Q, ctx, finalize=finalize))
    assert events[2].content == "BETA IS 0.5"
    assert [o["tool"] for o in seen["outputs"]] == ["get_factor_exposures"]


def test_rejected_tool_call_is_fed_back_once(ctx: ToolContext) -> None:
    from app.ai.llm.base import ToolUseFailed

    def rejected(messages, tools):
        raise ToolUseFailed("parameters for tool run_custom_stress did not match schema")

    fake = FakeLLMClient(
        [rejected, call("get_factor_exposures", "{}"), reply("Your market beta is shown above.")]
    )
    events = list(run_agent(fake, Q, ctx))
    assert types(events) == ["tool_call", "tool_result", "answer", "done"]
    note = fake.requests[1]["messages"][-1]
    assert note.role == "user" and "did not match schema" in note.content
    # A second rejection in the same turn is reported, not retried forever.
    twice = FakeLLMClient([rejected, rejected])
    assert types(list(run_agent(twice, Q, ctx))) == ["error", "done"]


def test_context_note_without_a_portfolio(market) -> None:
    from app.ai.agent.loop import initial_messages

    empty = ToolContext(market=lambda: market, portfolio=None)
    msgs = initial_messages(Q, empty)
    assert msgs[1].content.startswith("(Context) No portfolio is loaded")


def test_context_note_counts_as_a_grounding_source(ctx: ToolContext) -> None:
    from app.ai.agent.loop import initial_messages

    initial_messages(Q, ctx)
    fake = FakeLLMClient([reply("Your portfolio is worth Rs 10,00,000 across 8 holdings.")])
    events = list(run_agent(fake, Q, ctx))
    assert events[0].type == "answer" and events[0].grounded
