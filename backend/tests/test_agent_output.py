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
        calls(("save_employee_details", E001_DETAILS), ("find_car", {"query": "קורולה קרוס"})),
        calls(("set_current_car", {"car_id": "CAR-09", "participation": 300})),
        Message.assistant("איזו חלופה?"),
    )
    return run_turn(ConversationState("c-1"), E001, provider, ctx).state


def atto_turn(ctx: ToolContext, state: ConversationState, final: str, participation: int | None = None):
    arguments = {"car_id": "CAR-10"} if participation is None else {"car_id": "CAR-10", "participation": participation}
    text = "ומה עם אטו 3?" if participation is None else f"ומה עם אטו 3 בהשתתפות {participation}?"
    provider = ScriptedProvider(
        calls(("find_car", {"query": "אטו 3"})), calls(("compare_alternative", arguments)), Message.assistant(final)
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
        calls(("find_car", {"query": "קרניבל"})), calls(("compare_alternative", {"car_id": "CAR-19"})), Message.assistant(question)
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
    provider = ScriptedProvider(calls(("compare_alternative", {"waiver": True})), Message.assistant("x 123456"))

    turn = run_turn(e001, "ואם אוותר על הרכב?", provider, ctx)

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
    provider = ScriptedProvider(calls(("find_car", {"query": "טוסון"})), Message.assistant("מצאתי"))

    turn = run_turn(ConversationState("c-1"), "יש לי טוסון", provider, ctx, clock=lambda: float(next(ticks)))

    first, final = turn.timings
    assert first.provider_seconds == 1.0 and first.tool_seconds == 1.0
    assert final.provider_seconds == 1.0 and final.tool_seconds is None
    assert turn.total_seconds > first.provider_seconds + first.tool_seconds + final.provider_seconds
