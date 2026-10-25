"""The tool-calling loop: plain Python, no orchestration framework.

    messages = [system, *history]
    repeat (at most max_steps):
        response = model(messages, tools)
        if it asked for tools: run each, append the results, loop
        else: the content is the answer -> finalize (guardrails) -> stop

Events are yielded as they happen (tool call, tool result, answer, done), so the chat endpoint
can stream progress. The answer is sent whole, after the guardrails have checked it: a numeric
grounding check needs the complete text, so token-by-token streaming of unchecked text is
deliberately not done.
"""

import json
from collections.abc import Callable, Iterator
from typing import Any

from app.ai.agent.display import inr
from app.ai.agent.prompts import SYSTEM_PROMPT
from app.ai.agent.tools import TOOLS, ToolContext, execute
from app.ai.guardrails import advice_sentences, check_grounding, redact_advice, source_text
from app.ai.llm.base import LLMClient, LLMError, Message, ToolUseFailed
from app.ai.rag.citations import extract_citations
from app.ai.trace import Trace
from app.schemas.chat import (
    AnswerEvent,
    ChatMessage,
    DoneEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
)

MAX_STEPS = 6
DISCLAIMER = (
    "For information only, not investment advice. Figures are model estimates from historical "
    "data under the stated assumptions."
)

# finalize(answer, tool outputs, ctx) -> AnswerEvent: the guardrail hook (Phase 9).
Finalizer = Callable[[str, list[dict[str, Any]], ToolContext], AnswerEvent]


def default_finalize(answer: str, outputs: list[dict[str, Any]], ctx: ToolContext) -> AnswerEvent:
    """Guardrails on the final answer: numeric grounding, compliance, citations, disclaimer."""
    sources = source_text(outputs, ctx.retrieved, [*ctx.user_texts, ctx.context_note])
    grounding = check_grounding(answer, sources)
    advice = advice_sentences(answer)
    citations, invented = extract_citations(answer, ctx.retrieved)
    flags = [f"advice: {a}" for a in advice]
    flags += [f"citation not among the retrieved passages: {r}" for r in invented]
    if ctx.sanitized_passages:
        flags.append(
            f"instruction-like text removed from {ctx.sanitized_passages} retrieved passage(s)"
        )
    return AnswerEvent(
        content=answer,
        citations=citations,
        grounded=grounding.grounded,
        ungrounded_numbers=grounding.ungrounded,
        flags=flags,
        disclaimer=DISCLAIMER,
    )


def portfolio_context(ctx: ToolContext) -> str:
    """Tell the model whether a portfolio is loaded (the tools receive it from the request, so
    without this note the model may assume there is none and decline). Holding names and the
    total value count as sources for the grounding check."""
    p = ctx.portfolio
    if p is None:
        return (
            "(Context) No portfolio is loaded. Portfolio tools will fail until the user adds one."
        )
    market = None
    try:
        market = ctx.market()
    except Exception:  # names are a convenience; never fail the chat over them
        market = None
    names = [
        (market.name(h.asset_id).split(" - ")[0] if market is not None else h.asset_id)
        for h in p.holdings
    ]
    return (
        f"(Context) The user's portfolio is loaded and every portfolio tool uses it "
        f"automatically; do not ask for holdings. Value {inr(p.portfolio_value)}, "
        f"{len(names)} holdings: {'; '.join(names)}."
    )


def initial_messages(history: list[ChatMessage], ctx: ToolContext) -> list[Message]:
    """System prompt, portfolio context, then the conversation. Shared with the evaluation."""
    ctx.context_note = portfolio_context(ctx)
    return [Message("system", SYSTEM_PROMPT), Message("system", ctx.context_note)] + [
        Message(m.role, m.content) for m in history
    ]


def repair_tool_call(messages: list[Message], exc: Exception) -> None:
    """Append the note that asks the model to redo a provider-rejected tool call."""
    messages.append(
        Message(
            "user",
            f"(System note) Your last tool call was rejected: {exc}. Call the tool "
            "again with arguments that match its JSON schema exactly.",
        )
    )


def needs_repair(event: AnswerEvent) -> bool:
    return not event.grounded or any(f.startswith("advice: ") for f in event.flags)


def repair_note(event: AnswerEvent) -> str:
    parts = []
    if event.ungrounded_numbers:
        parts.append(
            "these numbers do not appear in any tool output or passage: "
            + ", ".join(event.ungrounded_numbers)
            + ". Do not compute, sum or estimate numbers; use only figures exactly as the tools "
            "gave them, or leave the figure out"
        )
    advice = [f[len("advice: ") :] for f in event.flags if f.startswith("advice: ")]
    if advice:
        parts.append(
            "these sentences tell the user what to do with their money, which is not allowed: "
            + " | ".join(advice)
            + ". Describe what the numbers show instead"
        )
    return "(System note) Rewrite your previous answer: " + "; ".join(parts) + "."


def _final(event: AnswerEvent) -> AnswerEvent:
    """Last resort after the repair turn: redact advice sentences, keep flags visible."""
    if any(f.startswith("advice: ") for f in event.flags):
        return event.model_copy(update={"content": redact_advice(event.content)})
    return event


def run_agent(
    client: LLMClient,
    history: list[ChatMessage],
    ctx: ToolContext,
    *,
    trace: Trace | None = None,
    finalize: Finalizer = default_finalize,
    max_steps: int = MAX_STEPS,
) -> Iterator[ToolCallEvent | ToolResultEvent | AnswerEvent | ErrorEvent | DoneEvent]:
    trace = trace or Trace()
    schemas = [t.schema() for t in TOOLS]
    messages = initial_messages(history, ctx)
    outputs: list[dict[str, Any]] = []
    ctx.user_texts = [m.content for m in history if m.role == "user"]
    answer_repaired = False
    trace.record("prompt", model=client.model, messages=[m.to_api() for m in messages])

    steps = 0
    repaired = False
    try:
        while steps < max_steps:
            steps += 1
            try:
                res = client.chat(messages, tools=schemas)
            except ToolUseFailed as exc:
                # The provider rejected the model's tool call against the schema. Tell the model
                # once; a second failure is reported to the user.
                trace.record("tool_use_failed", step=steps, message=str(exc))
                if repaired:
                    raise
                repaired = True
                repair_tool_call(messages, exc)
                continue
            trace.add_usage(res.usage.prompt_tokens, res.usage.completion_tokens)
            msg = res.message
            trace.record(
                "model_response",
                step=steps,
                content=msg.content,
                tool_calls=[
                    {"id": c.id, "name": c.name, "arguments": c.arguments} for c in msg.tool_calls
                ],
                finish_reason=res.finish_reason,
                usage={
                    "prompt": res.usage.prompt_tokens,
                    "completion": res.usage.completion_tokens,
                },
            )
            if msg.tool_calls:
                messages.append(msg)
                for call in msg.tool_calls:
                    try:
                        shown_args = call.parsed()
                    except ValueError:
                        shown_args = {"_raw": call.arguments}
                    yield ToolCallEvent(id=call.id, name=call.name, arguments=shown_args)
                    output = execute(call.name, call.arguments, ctx)
                    ok = "error" not in output
                    if ok:
                        outputs.append({"tool": call.name, "output": output})
                    trace.record("tool_result", id=call.id, name=call.name, ok=ok, output=output)
                    yield ToolResultEvent(id=call.id, name=call.name, ok=ok, output=output)
                    messages.append(
                        Message(
                            "tool",
                            json.dumps(output, ensure_ascii=False),
                            tool_call_id=call.id,
                        )
                    )
                continue
            answer = (msg.content or "").strip()
            if not answer:
                yield ErrorEvent(message="The model returned an empty answer; please rephrase.")
                break
            event = finalize(answer, outputs, ctx)
            if needs_repair(event) and not answer_repaired:
                answer_repaired = True
                note = repair_note(event)
                trace.record(
                    "guardrail_repair",
                    step=steps,
                    flags=event.flags,
                    ungrounded=event.ungrounded_numbers,
                )
                messages += [Message("assistant", answer), Message("user", note)]
                continue
            event = _final(event)
            trace.record(
                "answer",
                content=event.content,
                grounded=event.grounded,
                ungrounded=event.ungrounded_numbers,
                flags=event.flags,
            )
            yield event
            break
        else:
            yield ErrorEvent(
                message="The assistant needed too many steps for this question; try asking for "
                "one thing at a time."
            )
    except LLMError as exc:
        trace.record("error", message=str(exc))
        yield ErrorEvent(message=str(exc))
    yield DoneEvent(
        trace_id=trace.id,
        steps=steps,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
    )
