"""The agent loop: one employee message in, one Hebrew reply out, with tool calls in between.

The LLM only converses and chooses tools. Every number and decision comes from the tools
(app.agent.tool_registry -> app.agent.tools -> Steps 1-3). The loop adds Python-side guarantees:
- a hard round limit;
- every number in the final reply must appear in a tool result or the employee's own messages;
  otherwise the reply is replaced by a deterministic one built from the engine's result;
- an estimated comparison is always disclosed (a fixed notice is added if the reply does not say so);
- after the first tool round, a recognised outcome (app.agent.render) is put into words by Python,
  saving the provider call that would only have verbalised it. The rendered reply must pass the same
  numbers check; if it does not, or the round is not recognised, the provider writes the reply.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from app.agent.llm import LLMProvider, Message, ProviderError, ToolResult
from app.agent.outcome import TurnOutcome, round_outcome
from app.agent.output_check import discloses_estimate, unsupported_numbers
from app.agent.prompt import system_prompt
from app.agent.render import ReplyKind, RoundCall, deterministic_reply
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
    RENDERED = "rendered"  # written by Python from the first tool round's results (app.agent.render)
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
    outcome: TurnOutcome | None = None  # the turn's last decisive tool outcome, as data (app.agent.outcome)


def run_turn(
    state: ConversationState,
    user_text: str,
    provider: LLMProvider,
    ctx: ToolContext | None = None,
    clock: Callable[[], float] = time.perf_counter,
    events: Callable[[str], None] | None = None,
) -> TurnResult:
    """events, if given, receives live progress lines as they happen: round numbers, timings,
    tool names with ok / error code, and provider error kinds only - never values or prompts."""
    emit = events or (lambda line: None)
    started = clock()
    ctx = ctx or ToolContext.load()
    state = with_messages(start_turn(state, user_text, ctx), Message.user(user_text))
    system = system_prompt(state, ctx)  # the representative's answers do not change during a turn
    comparison: ComparisonSummary | None = None  # the last comparison calculated in this turn
    timings: list[RoundTiming] = []
    outcome: TurnOutcome | None = None

    def finish(text: str, stop: TurnStop, rounds: int) -> TurnResult:
        final = with_messages(state, Message.assistant(text))
        return TurnResult(final, text, stop, rounds, tuple(timings), clock() - started, outcome)

    for rounds in range(MAX_TOOL_ROUNDS + 1):
        number = rounds + 1
        emit(f"round {number}: provider call started")
        call_started = clock()
        try:
            reply = provider.complete(system, state.messages, SPECS)
        except ProviderError as e:
            timings.append(RoundTiming(clock() - call_started, None))
            emit(
                f"round {number}: provider error category={e.category.value} status={e.status_code or '-'} "
                f"retryable={'yes' if e.retryable else 'no'} after {timings[-1].provider_seconds:.2f}s"
            )
            return finish(PROVIDER_ERROR_REPLY, TurnStop.PROVIDER_ERROR, rounds)
        provider_seconds = clock() - call_started
        returned = f"{len(reply.tool_calls)} tool calls" if reply.tool_calls else "final reply"
        emit(f"round {number}: provider returned in {provider_seconds:.2f}s ({returned})")

        if not reply.tool_calls:
            timings.append(RoundTiming(provider_seconds, None))
            text, stop = _checked_reply(reply.text, state, comparison)
            return finish(text, stop, rounds)
        if rounds == MAX_TOOL_ROUNDS:
            timings.append(RoundTiming(provider_seconds, None))
            emit(f"round {number}: round limit reached; tool calls not executed")
            break  # not executed and not recorded: a call without its result would break the history

        tools_started = clock()
        state = with_messages(state, reply)
        results = []
        round_calls = []
        for call in reply.tool_calls:
            execution = execute(call, state, ctx)
            state = execution.state
            results.append(ToolResult(call.call_id, call.name, execution.content))  # answers the call as made
            tool = execution.tool or call.name  # what actually ran (find_car "alternative" runs a comparison)
            round_calls.append(
                RoundCall(
                    tool, execution.content, execution.comparison,
                    via=call.name if tool != call.name else None, error=execution.error,
                )
            )
            if execution.comparison is not None:
                comparison = execution.comparison
        outcome = round_outcome(round_calls, outcome)
        state = with_messages(state, Message.tool(results))
        timings.append(RoundTiming(provider_seconds, clock() - tools_started))
        outcomes = ", ".join(
            f"{c.via}→{c.name}={_outcome(c.content)}" if c.via else f"{c.name}={_outcome(c.content)}" for c in round_calls
        )
        emit(f"round {number}: tools {timings[-1].tool_seconds:.2f}s: {outcomes}")

        if rounds == 0:
            rendered = _rendered_reply(user_text, round_calls, state, ctx, emit, number)
            if rendered is not None:
                return finish(rendered, TurnStop.RENDERED, rounds + 1)

    return finish(MAX_ROUNDS_REPLY, TurnStop.MAX_ROUNDS, MAX_TOOL_ROUNDS)


def _rendered_reply(
    user_text: str, round_calls: list[RoundCall], state: ConversationState, ctx: ToolContext,
    emit: Callable[[str], None], number: int,
) -> str | None:
    """Python's reply for a recognised first round, or None to ask the provider as before."""
    rendered = deterministic_reply(user_text, round_calls, state, ctx.rules)
    if rendered is None:
        return None
    estimate = rendered.kind == ReplyKind.COMPARISON and any(
        c.comparison is not None and c.comparison.status == ComparisonStatus.ESTIMATE for c in round_calls
    )
    if unsupported_numbers(rendered.text, state) or (estimate and not discloses_estimate(rendered.text)):
        emit(f"round {number}: deterministic reply not used ({rendered.kind.value}); asking the provider")
        return None
    emit(f"round {number}: deterministic reply ({rendered.kind.value})")
    return rendered.text


def _outcome(content) -> str:
    """ok, or the error code: never the result itself."""
    return "ok" if content.get("ok") else str(content.get("error", {}).get("code", "error"))


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

