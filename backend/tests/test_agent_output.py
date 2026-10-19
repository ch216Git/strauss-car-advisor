"""Final-reply safety: estimate disclosure, numbers only from tools, deterministic replacement, timing."""

import itertools

import pytest

from app.agent.agent import (
    ESTIMATE_NOTICE,
    UNSUPPORTED_NUMBERS_REPLY,
    TurnStop,
    comparison_reply,
    run_turn,
)
from app.agent.llm import Message, Role, ToolCall
from app.agent.output_check import discloses_estimate, unsupported_numbers
from app.agent.state import ConversationState
from app.agent.tool_registry import CHANGE_LABELS
from app.agent.tools import ToolContext

_ids = itertools.count()


class ScriptedProvider:
    name = "scripted"

    def __init__(self, *steps: Message) -> None:
        self._steps = list(steps)

    def complete(self, system, messages, tools) -> Message:
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"o-{next(_ids)}", n, a) for n, a in name_and_arguments])


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


E001 = "אני מנהלת מוצר, 18,500 ברוטו בלי עמלות ובונוס, פנסיה 6%, השתלמות 2.5%, 2.75 נקודות. יש לי קורולה קרוס עם השתתפות 300."
E001_DETAILS = {
    "occupation": "מנהל/ת מוצר", "base_salary": 18500, "commissions": 0, "bonus": 0,
    "pension_percent": 6, "training_fund_percent": 2.5, "credit_points": 2.75,
}


@pytest.fixture(scope="module")
def e001(ctx: ToolContext) -> ConversationState:
    provider = ScriptedProvider(
        calls(("save_employee_details", E001_DETAILS), ("find_car", {"query": "קורולה קרוס", "purpose": "information"})),
        calls(("set_current_car", {"car_id": "CAR-09", "participation": 300})),
        Message.assistant("איזו חלופה?"),
    )
    return run_turn(ConversationState("c-1"), E001, provider, ctx).state


def atto_turn(ctx: ToolContext, state: ConversationState, final: str, participation: int | None = None):
    arguments = {"car_id": "CAR-10"} if participation is None else {"car_id": "CAR-10", "participation": participation}
    text = "ומה עם אטו 3?" if participation is None else f"ומה עם אטו 3 בהשתתפות {participation}?"
    provider = ScriptedProvider(
        calls(("find_car", {"query": "אטו 3", "purpose": "information"})), calls(("compare_alternative", arguments)), Message.assistant(final)
    )
    return run_turn(state, text, provider, ctx)


def comparison_result(turn) -> dict:
    return dict([m for m in turn.state.messages if m.role == Role.TOOL][-1].tool_results[0].content)


# --- Estimate disclosure ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "ההשתתפות מוערכת",
        "סכום מוערך",
        "השתתפות מוערכים",
        "זו הערכה בלבד",
        "לפי הערכת הקטלוג",
        "להערכתי",
        "נטו משוער",
        "התוצאה משוערת",
        "לפי ברירת מחדל",
        "לפי ברירת-מחדל",
        "בברירת המחדל",
        "אומדן",
    ],
)
def test_hebrew_estimate_forms_are_recognized(text: str) -> None:
    assert discloses_estimate(text)


@pytest.mark.parametrize("text", ["הנטו יעלה ב-661 ₪.", "ערך השימוש יורד", "הערות: אין"])
def test_replies_without_disclosure(text: str) -> None:
    assert not discloses_estimate(text)


def test_gemini_style_disclosure_gets_no_duplicate_notice(ctx: ToolContext, e001: ConversationState) -> None:
    final = "נטו היום 11,032 ₪, עם ATTO 3 11,693 ₪ (+661 ₪). ההשתתפות ברכב החדש מוערכת לפי הקטלוג."

    turn = atto_turn(ctx, e001, final)

    assert (turn.stop, turn.reply) == (TurnStop.REPLIED, final)
    assert ESTIMATE_NOTICE not in turn.reply


# --- Numbers only from tool results ------------------------------------------------------------

def test_the_440_figure_is_in_the_tool_result_with_a_label(ctx: ToolContext, e001: ConversationState) -> None:
    turn = atto_turn(ctx, e001, "הטבת ברוטו של 440 ש״ח לרכב חשמלי, כלומר 250 ₪ נטו. זו הערכה.")
    result = comparison_result(turn)

    electric = next(c for c in result["changes"] if c["item"] == "electric_gross")
    assert (electric["alternative"], electric["label"]) == (440, CHANGE_LABELS["electric_gross"])
    assert result["alternative"]["electric_net_bonus"] == ctx.rules.electric.net_bonus
    assert turn.stop == TurnStop.REPLIED  # 440 and 250 are both in the tool result


def test_invented_number_replaces_the_reply_with_engine_figures(ctx: ToolContext, e001: ConversationState) -> None:
    turn = atto_turn(ctx, e001, "תחסכי 1,234 ₪ בשנה, הערכה.")  # 1,234 is not in any tool result

    assert turn.stop == TurnStop.REPLACED_UNSUPPORTED_NUMBERS
    assert turn.reply == "\n".join([
        "נטו היום: 11,032 ₪",
        "נטו בחלופה (ATTO 3): 11,693 ₪",
        "הפרש: +661 ₪ בחודש",
        ESTIMATE_NOTICE,
    ])
    assert turn.state.messages[-1].text == turn.reply


def test_derived_number_is_not_allowed(ctx: ToolContext, e001: ConversationState) -> None:
    # 661 x 12 = 7,932 a year: a calculation by the model, not an engine figure.
    turn = atto_turn(ctx, e001, "+661 ₪ בחודש, כלומר 7,932 ₪ בשנה (הערכה).")

    assert turn.stop == TurnStop.REPLACED_UNSUPPORTED_NUMBERS


def test_confirmed_replacement_has_no_estimate_notice(ctx: ToolContext, e001: ConversationState) -> None:
    turn = atto_turn(ctx, e001, "בערך 700 ₪.", participation=300)

    assert turn.stop == TurnStop.REPLACED_UNSUPPORTED_NUMBERS
    assert ESTIMATE_NOTICE not in turn.reply and "הפרש: +661 ₪ בחודש" in turn.reply


def test_invented_number_without_a_comparison_gets_a_generic_reply(ctx: ToolContext) -> None:
    provider = ScriptedProvider(Message.assistant("כמה ילדים יש לך עד גיל 18?"))

    turn = run_turn(ConversationState("c-1"), "שלום", provider, ctx)

    assert (turn.stop, turn.reply) == (TurnStop.REPLACED_UNSUPPORTED_NUMBERS, UNSUPPORTED_NUMBERS_REPLY)


def test_family_rule_numbers_come_from_the_tool_result(ctx: ToolContext, e001: ConversationState) -> None:
    rule = ctx.rules.family_track
    question = f"כמה ילדים עד גיל {rule.child_max_age} יש לך? צריך לפחות {rule.min_children}."
    provider = ScriptedProvider(
        calls(("find_car", {"query": "קרניבל", "purpose": "information"})), calls(("compare_alternative", {"car_id": "CAR-19"})), Message.assistant(question)
    )

    turn = run_turn(e001, "ומה עם קרניבל?", provider, ctx)

    assert (turn.stop, turn.reply) == (TurnStop.REPLIED, question)


def test_employee_numbers_and_list_markers_may_be_repeated(ctx: ToolContext, e001: ConversationState) -> None:
    turn = atto_turn(ctx, e001, "1. ברוטו 18,500 ופנסיה 6%\n2. נטו היום 11,032 ₪, בחלופה 11,693 ₪ (הערכה)")

    assert turn.stop == TurnStop.REPLIED


def test_unsupported_numbers_lists_only_the_unsupported(e001: ConversationState) -> None:
    assert unsupported_numbers("18,500 ו-6% ו-999", e001) == (999.0,)
    assert unsupported_numbers("18.5 אלף", e001) == ()  # read as 18,500


def test_comparison_reply_for_a_waiver(ctx: ToolContext, e001: ConversationState) -> None:
    # Comparing in round 2 keeps the provider's reply (rendering is for round 1 only), so the replacement runs.
    provider = ScriptedProvider(
        calls(("list_alternatives", {})), calls(("compare_alternative", {"waiver": True})), Message.assistant("x 123456")
    )

    turn = run_turn(e001, "ואם אוותר על הרכב?", provider, ctx)

    assert turn.stop == TurnStop.REPLACED_UNSUPPORTED_NUMBERS
    assert turn.reply.splitlines()[1].startswith("נטו בחלופה (ויתור על הרכב):")


def test_every_change_item_has_a_label(ctx: ToolContext, e001: ConversationState) -> None:
    changes = comparison_result(atto_turn(ctx, e001, "הערכה"))["changes"]

    assert all(change["label"] == CHANGE_LABELS[change["item"]] for change in changes)


def test_comparison_reply_uses_only_engine_figures(ctx: ToolContext, e001: ConversationState) -> None:
    turn = atto_turn(ctx, e001, "999999")

    assert unsupported_numbers(turn.reply, turn.state) == ()
    assert turn.reply.startswith("נטו היום: 11,032 ₪")


# --- Timing --------------------------------------------------------------------------------------

def test_timings_per_provider_call_and_round(ctx: ToolContext) -> None:
    ticks = itertools.count()
    provider = ScriptedProvider(calls(("find_car", {"query": "טוסון", "purpose": "information"})), Message.assistant("מצאתי"))

    turn = run_turn(ConversationState("c-1"), "יש לי טוסון", provider, ctx, clock=lambda: float(next(ticks)))

    first, final = turn.timings
    assert first.provider_seconds == 1.0 and first.tool_seconds == 1.0
    assert final.provider_seconds == 1.0 and final.tool_seconds is None
    assert turn.total_seconds > first.provider_seconds + first.tool_seconds + final.provider_seconds


# --- Live events --------------------------------------------------------------------------------

class RecordingProvider:
    """Records how many events existed when each call started, to prove "started" comes first."""

    name = "recording"

    def __init__(self, events: list[str], *steps) -> None:
        self.events, self._steps, self.events_at_call = events, list(steps), []

    def complete(self, system, messages, tools) -> Message:
        self.events_at_call.append(list(self.events))
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def test_started_event_is_emitted_before_each_provider_call(ctx: ToolContext) -> None:
    events: list[str] = []
    provider = RecordingProvider(events, calls(("find_car", {"query": "טוסון", "purpose": "information"})), Message.assistant("מצאתי"))

    run_turn(ConversationState("c-1"), "יש לי טוסון", provider, ctx, events=events.append)

    assert provider.events_at_call[0][-1] == "round 1: provider call started"
    assert provider.events_at_call[1][-1] == "round 2: provider call started"
    assert events[1].startswith("round 1: provider returned in ") and events[1].endswith("(1 tool calls)")
    assert events[2].startswith("round 1: tools ") and events[2].endswith(": find_car=ok")
    assert events[4].endswith("(final reply)")


@pytest.mark.parametrize(
    "category, status, kind, expected",
    [
        ("server_error", 503, "unavailable", "category=server_error status=503 retryable=yes"),
        ("rate_limited", 429, "rate_limited", "category=rate_limited status=429 retryable=no"),
        ("timeout", None, "unavailable", "category=timeout status=- retryable=yes"),
        ("auth_error", 403, "authentication", "category=auth_error status=403 retryable=no"),
        ("invalid_request", 400, "request_failed", "category=invalid_request status=400 retryable=no"),
    ],
)
def test_provider_error_event_shows_only_safe_diagnostics(ctx: ToolContext, category, status, kind, expected) -> None:
    from app.agent.llm import ProviderError, ProviderErrorKind, ProviderFailure

    events: list[str] = []
    error = ProviderError(
        "gemini", ProviderErrorKind(kind), "raw detail: request body and key=AIza-SECRET",
        category=ProviderFailure(category), status_code=status,
    )

    turn = run_turn(ConversationState("c-1"), "שלום", RecordingProvider(events, error), ctx, events=events.append)

    assert turn.stop == TurnStop.PROVIDER_ERROR
    assert events[0] == "round 1: provider call started"
    assert events[1].startswith(f"round 1: provider error {expected} after ")
    assert "raw detail" not in events[1] and "AIza" not in events[1]
    assert turn.reply == "מצטער/ת, יש תקלה זמנית בחיבור ליועץ. אפשר לנסות שוב בעוד רגע."  # generic for the employee


def test_events_carry_no_employee_values(ctx: ToolContext) -> None:
    events: list[str] = []
    provider = RecordingProvider(
        events,
        calls(("save_employee_details", E001_DETAILS), ("find_car", {"query": "קורולה קרוס", "purpose": "information"})),
        Message.assistant("תודה"),
    )

    run_turn(ConversationState("c-1"), E001, provider, ctx, events=events.append)

    text = "\n".join(events)
    assert "save_employee_details=ok, find_car=ok" in text
    for value in ("18500", "18,500", "2.75", "מנהל", "קורולה", "CAR-09"):
        assert value not in text


def test_no_events_by_default(ctx: ToolContext) -> None:
    provider = ScriptedProvider(Message.assistant("שלום"))

    assert run_turn(ConversationState("c-1"), "שלום", provider, ctx).reply == "שלום"
