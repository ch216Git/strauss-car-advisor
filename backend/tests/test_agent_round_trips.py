"""Fewer provider round trips without weaker validation: occupation wording and cars in the employee's words."""

import itertools

import pytest

from app.agent.agent import TurnStop, run_turn
from app.agent.grounding import matching_occupations
from app.agent.llm import Message, Role, ToolCall
from app.agent.state import ConversationState, ConversationStatus, conversation_status
from app.agent.tools import ToolContext, save_employee_details
from app.comparison import CompanyCar

_ids = itertools.count()


class CountingProvider:
    name = "counting"

    def __init__(self, *steps: Message) -> None:
        self._steps, self.calls = list(steps), 0

    def complete(self, system, messages, tools) -> Message:
        self.calls += 1
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"r-{next(_ids)}", n, a) for n, a in name_and_arguments])


def results(state: ConversationState) -> list[dict]:
    return [dict(r.content) for m in state.messages if m.role == Role.TOOL for r in m.tool_results]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


# The real first message, as the employee wrote it.
E001 = (
    "אני מנהלת מוצר, 18,500 ברוטו בלי עמלות ובונוס, פנסיה 6%, השתלמות 2.5%, 2.75 נקודות. "
    "יש לי קורולה קרוס עם השתתפות 300. כמה ישתנה הנטו עם אטו 3?"
)
E001_DETAILS = {  # occupation in the employee's wording, as Gemini sent it in round 1
    "occupation": "מנהלת מוצר", "base_salary": 18500, "commissions": 0, "bonus": 0,
    "pension_percent": 6, "training_fund_percent": 2.5, "credit_points": 2.75,
}


# --- Occupation: employee wording -> the one workbook name ----------------------------------

@pytest.mark.parametrize(
    "wording, name",
    [
        ("מנהלת מוצר", "מנהל/ת מוצר"),
        ("מנהל מוצר", "מנהל/ת מוצר"),
        ("מנהלת המוצר", "מנהל/ת מוצר"),
        ("נציגת מכירות שטח", "נציג/ת מכירות שטח"),
        ("אנליסטית נתונים", "אנליסט/ית נתונים"),
        ("מנהלת משאבי אנוש", "מנהל/ת משאבי אנוש"),
    ],
)
def test_employee_wording_resolves_to_one_workbook_name(ctx: ToolContext, wording: str, name: str) -> None:
    assert matching_occupations(wording, ctx.catalog.occupations) == (name,)


@pytest.mark.parametrize("wording", ["מנהלת", "מנהלת מוצר בכירה", "מתכנתת", "מנהלת מוצרים ופיתוח", "product manager"])
def test_partial_extra_or_other_words_resolve_to_nothing(ctx: ToolContext, wording: str) -> None:
    assert matching_occupations(wording, ctx.catalog.occupations) == ()


def test_tool_saves_the_workbook_name(ctx: ToolContext) -> None:
    outcome = save_employee_details(ConversationState("c-1"), ctx, occupation="מנהלת מוצר")

    assert outcome.state.details.occupation == "מנהל/ת מוצר"


def test_round_1_occupation_is_now_saved(ctx: ToolContext) -> None:
    provider = CountingProvider(calls(("save_employee_details", {"occupation": "מנהלת מוצר"})), Message.assistant("ok"))

    turn = run_turn(ConversationState("c-1"), "אני מנהלת מוצר", provider, ctx)

    assert results(turn.state)[0]["ok"]
    assert turn.state.details.occupation == "מנהל/ת מוצר"


def test_grounding_still_refuses_an_occupation_the_employee_did_not_state(ctx: ToolContext) -> None:
    provider = CountingProvider(calls(("save_employee_details", {"occupation": "מנהלת פיתוח"})), Message.assistant("ok"))

    turn = run_turn(ConversationState("c-1"), "אני מנהלת מוצר", provider, ctx)

    assert results(turn.state)[0]["error"]["code"] == "not_stated_by_employee"
    assert turn.state.details.occupation is None


# --- The whole first message in one tool round --------------------------------------------------

def test_first_message_needs_one_provider_call(ctx: ToolContext) -> None:
    # The comparison is calculated in round 1 and written by Python (app.agent.render): no second call.
    provider = CountingProvider(
        calls(
            ("save_employee_details", E001_DETAILS),
            ("set_current_car", {"car": "קורולה קרוס", "participation": 300}),
            ("compare_alternative", {"car": "אטו 3"}),
        ),
    )

    turn = run_turn(ConversationState("c-1"), E001, provider, ctx)

    assert (provider.calls, turn.tool_rounds, turn.stop) == (1, 1, TurnStop.RENDERED)
    saved, current, compared = results(turn.state)
    assert saved["ok"] and current["ok"] and compared["ok"]
    assert (compared["current"]["net"], compared["alternative"]["net"], compared["net_difference"]) == (11032, 11693, 661)
    assert turn.state.details.current == CompanyCar("CAR-09", 300)


def test_missing_details_still_reported_in_the_same_round(ctx: ToolContext) -> None:
    provider = CountingProvider(
        calls(
            ("save_employee_details", {"occupation": "מנהלת מוצר"}),
            ("set_current_car", {"car": "קורולה קרוס", "participation": 300}),
            ("compare_alternative", {"car": "אטו 3"}),
        ),
    )

    turn = run_turn(ConversationState("c-1"), "אני מנהלת מוצר, יש לי קורולה קרוס בהשתתפות 300. ואטו 3?", provider, ctx)

    _, current, compared = results(turn.state)
    assert current["ok"]
    assert compared["error"]["code"] == "missing_details" and "base_salary" in compared["error"]["missing"]
    assert (provider.calls, turn.stop) == (1, TurnStop.RENDERED)  # the question is written by Python


# --- `car` keeps every guard of find_car --------------------------------------------------------

def test_ambiguous_car_is_not_set_and_candidates_come_back(ctx: ToolContext) -> None:
    provider = CountingProvider(calls(("set_current_car", {"car": "קורולה", "participation": 250})), Message.assistant("איזו?"))

    turn = run_turn(ConversationState("c-1"), "יש לי קורולה, השתתפות 250", provider, ctx)

    (result,) = results(turn.state)
    assert result["error"]["code"] == "car_needs_choice"
    assert [car["commercial_name"] for car in result["error"]["matches"]["cars"]] == ["COROLLA", "COROLLA CROSS"]
    assert turn.state.details.current is None
    assert {"CAR-07", "CAR-09"} <= turn.state.offered_car_ids

    # The employee names one: it can now be set.
    after = run_turn(
        turn.state, "קורולה קרוס", CountingProvider(calls(("set_current_car", {"car": "קורולה קרוס", "participation": 250})), Message.assistant("נשמר")), ctx
    )
    assert after.state.details.current == CompanyCar("CAR-09", 250)


def test_car_words_the_employee_did_not_write_are_refused(ctx: ToolContext) -> None:
    provider = CountingProvider(calls(("set_current_car", {"car": "קורולה קרוס", "participation": 300})), Message.assistant("?"))

    turn = run_turn(ConversationState("c-1"), "יש לי קורולה, השתתפות 300", provider, ctx)

    assert results(turn.state)[0]["error"]["code"] == "car_not_confirmed"
    assert turn.state.details.current is None


def test_car_not_found_keeps_the_representative_flow(ctx: ToolContext) -> None:
    provider = CountingProvider(
        calls(("set_current_car", {"car": "מאזדה CX-5 2019", "participation": 300})),
        calls(("ask_representative", {"topic": "current_car_not_in_catalogue", "question": "מאזדה CX-5 2019"})),
        Message.assistant("שאלתי את הנציג"),
    )

    turn = run_turn(ConversationState("c-1"), "יש לי מאזדה CX-5 2019 בהשתתפות 300", provider, ctx)

    assert results(turn.state)[0]["error"]["code"] == "car_not_found"
    assert conversation_status(turn.state, ctx.rules) == ConversationStatus.WAITING_FOR_REPRESENTATIVE


def test_participation_with_car_words_is_still_grounded(ctx: ToolContext) -> None:
    provider = CountingProvider(calls(("set_current_car", {"car": "קורולה קרוס", "participation": 300})), Message.assistant("?"))

    turn = run_turn(ConversationState("c-1"), "יש לי קורולה קרוס", provider, ctx)

    assert results(turn.state)[0]["error"]["code"] == "not_stated_by_employee"


@pytest.mark.parametrize(
    "arguments",
    [
        {"car": "קורולה קרוס", "car_id": "CAR-09", "participation": 300},
        {"car": "קורולה קרוס", "waiver": True},
    ],
)
def test_car_with_car_id_or_waiver_is_refused(ctx: ToolContext, arguments) -> None:
    provider = CountingProvider(calls(("set_current_car", arguments)), Message.assistant("?"))

    turn = run_turn(ConversationState("c-1"), "קורולה קרוס בהשתתפות 300, ויתור", provider, ctx)

    assert results(turn.state)[0]["error"]["code"] == "invalid_arguments"
