"""The agent loop: one employee message in, one Hebrew reply out, with tool calls in between.

The LLM only converses and chooses tools. Every number and decision comes from the tools
(app.agent.tool_registry -> app.agent.tools -> Steps 1-3). The loop adds Python-side guarantees:
- a hard round limit;
- every number in the final reply must appear in a tool result or the employee's own messages;
  otherwise the reply is replaced by a deterministic one built from the engine's result;
- an estimated comparison is always disclosed (a fixed notice is added if the reply does not say so).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from app.agent.llm import LLMProvider, Message, ProviderError, ToolResult
from app.agent.output_check import discloses_estimate, unsupported_numbers
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.state import ComparisonStatus, ConversationState, with_messages
from app.agent.tool_registry import SPECS, execute, start_turn
from app.agent.tools import ComparisonSummary, ToolContext

MAX_TOOL_ROUNDS = 6  # provider replies with tool calls, per employee message

MAX_ROUNDS_REPLY = "מצטער/ת, לא הצלחתי להשלים את הבדיקה. אפשר לנסח שוב או לפרט מה תרצה/י לבדוק?"
PROVIDER_ERROR_REPLY = "מצטער/ת, יש תקלה זמנית בחיבור ליועץ. אפשר לנסות שוב בעוד רגע."
UNSUPPORTED_NUMBERS_REPLY = "מצטער/ת, לא הצלחתי לנסח תשובה מבוססת על הנתונים. אפשר לשאול שוב?"
ESTIMATE_NOTICE = (
    "שימו לב: ההשתתפות ברכב החלופי היא ברירת מחדל מהקטלוג, כלומר הערכה ולא נתון מאושר. "
    "אפשר לבקש מנציג התחבורה את הסכום המדויק."
)


class TurnStop(StrEnum):
    REPLIED = "replied"
    REPLACED_UNSUPPORTED_NUMBERS = "replaced_unsupported_numbers"  # the model's reply had numbers not from tools
    MAX_ROUNDS = "max_rounds"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True, slots=True)
class RoundTiming:
    provider_seconds: float  # one provider.complete() call
    tool_seconds: float | None  # all local tool executions of that round; None when the reply had no tool calls


@dataclass(frozen=True, slots=True)
class TurnResult:
    state: ConversationState
    reply: str
    stop: TurnStop
    tool_rounds: int
    timings: tuple[RoundTiming, ...] = ()  # one per provider call, in order
    total_seconds: float = 0.0


def run_turn(
    state: ConversationState,
    user_text: str,
    provider: LLMProvider,
    ctx: ToolContext | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TurnResult:
    started = clock()
    ctx = ctx or ToolContext.load()
    state = with_messages(start_turn(state, user_text, ctx), Message.user(user_text))
    comparison: ComparisonSummary | None = None  # the last comparison calculated in this turn
    timings: list[RoundTiming] = []

    def finish(text: str, stop: TurnStop, rounds: int) -> TurnResult:
        final = with_messages(state, Message.assistant(text))
        return TurnResult(final, text, stop, rounds, tuple(timings), clock() - started)

    for rounds in range(MAX_TOOL_ROUNDS + 1):
        call_started = clock()
        try:
            reply = provider.complete(SYSTEM_PROMPT, state.messages, SPECS)
        except ProviderError:
            timings.append(RoundTiming(clock() - call_started, None))
            return finish(PROVIDER_ERROR_REPLY, TurnStop.PROVIDER_ERROR, rounds)
        provider_seconds = clock() - call_started

        if not reply.tool_calls:
            timings.append(RoundTiming(provider_seconds, None))
            text, stop = _checked_reply(reply.text, state, comparison)
            return finish(text, stop, rounds)
        if rounds == MAX_TOOL_ROUNDS:
            timings.append(RoundTiming(provider_seconds, None))
            break  # not executed and not recorded: a call without its result would break the history

        tools_started = clock()
        state = with_messages(state, reply)
        results = []
        for call in reply.tool_calls:
            execution = execute(call, state, ctx)
            state = execution.state
            results.append(ToolResult(call.call_id, call.name, execution.content))
            if execution.comparison is not None:
                comparison = execution.comparison
        state = with_messages(state, Message.tool(results))
        timings.append(RoundTiming(provider_seconds, clock() - tools_started))

    return finish(MAX_ROUNDS_REPLY, TurnStop.MAX_ROUNDS, MAX_TOOL_ROUNDS)


def _checked_reply(text: str, state: ConversationState, comparison: ComparisonSummary | None) -> tuple[str, TurnStop]:
    """The model's reply if every number in it is supported, else a deterministic replacement."""
    if unsupported_numbers(text, state):
        replacement = comparison_reply(comparison) if comparison else UNSUPPORTED_NUMBERS_REPLY
        return replacement, TurnStop.REPLACED_UNSUPPORTED_NUMBERS
    if comparison and comparison.status == ComparisonStatus.ESTIMATE and not discloses_estimate(text):
        text = f"{text}\n\n{ESTIMATE_NOTICE}" if text else ESTIMATE_NOTICE
    return text, TurnStop.REPLIED


def comparison_reply(summary: ComparisonSummary) -> str:
    """A plain Hebrew summary built only from the engine's displayed figures."""
    alternative = summary.result.alternative_situation
    name = "ויתור על הרכב" if alternative.kind == "waiver" else alternative.commercial_name
    lines = [
        f"נטו היום: {summary.current_net:,} ₪",
        f"נטו בחלופה ({name}): {summary.alternative_net:,} ₪",
        f"הפרש: {summary.net_difference:+,} ₪ בחודש",
    ]
    if summary.status == ComparisonStatus.ESTIMATE:
        lines.append(ESTIMATE_NOTICE)
    return "\n".join(lines)

