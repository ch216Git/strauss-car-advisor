"""Provenance: the model cannot invent employee facts or choose a car the employee did not name."""

import pytest

from app.agent.agent import run_turn
from app.agent.grounding import number_grounded, numbers_in, occupation_grounded, waiver_grounded
from app.agent.llm import Message, Role, ToolCall
from app.agent.state import ConversationState
from app.agent.tools import ToolContext
from app.comparison import CompanyCar, Waiver

_ids = iter(range(10_000))


class ScriptedProvider:
    name = "scripted"

    def __init__(self, *steps: Message) -> None:
        self._steps = list(steps)

    def complete(self, system, messages, tools) -> Message:
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"g-{next(_ids)}", n, a) for n, a in name_and_arguments])


def turn(state: ConversationState, user_text: str, ctx: ToolContext, *tool_rounds: Message):
    return run_turn(state, user_text, ScriptedProvider(*tool_rounds, Message.assistant("ok")), ctx)


def last_results(state: ConversationState, count: int) -> list[dict]:
    contents = [dict(r.content) for m in state.messages if m.role == Role.TOOL for r in m.tool_results]
    return contents[-count:]


def code(result: dict) -> str | None:
    return None if result.get("ok") else result["error"]["code"]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


# --- Reading numbers and words ----------------------------------------------------------

@pytest.mark.parametrize(
    "text, value",
    [
        ("שכר 18,500 ברוטו", 18500),
        ("שכר 18500", 18500),
        ("18.5 אלף", 18500),
        ("18 אלף", 18000),
        ("18.5k", 18500),
        ("השתתפות 300.", 300),
        ("פנסיה 6%", 6),
        ("2.75 נקודות זיכוי", 2.75),
        ("יש לי חמישה ילדים", 5),
        ("ושלושה ילדים", 3),
        ("ארבע בנות", 4),
        ("בונוס של אלפיים", 2000),
    ],
)
def test_numbers_the_employee_wrote(text: str, value: float) -> None:
    assert value in numbers_in([text])


@pytest.mark.parametrize("text", ["אין לי עמלות", "בלי בונוס", "ללא קרן השתלמות", "לא מקבלת בונוס", "בונוס 0", "אפס"])
def test_zero_from_a_none_word(text: str) -> None:
    assert number_grounded(0, [text])


def test_zero_needs_a_none_word() -> None:
    assert not number_grounded(0, ["השכר שלי 18,500"])


@pytest.mark.parametrize(
    "occupation, text, grounded",
    [
        ("מנהל/ת מוצר", "אני מנהלת מוצר", True),
        ("מנהל/ת מוצר", "אני מנהל המוצר", True),  # prefix ה
        ("אנליסט/ית נתונים", "אנליסטית נתונים", True),
        ("נציג/ת מכירות שטח", "נציגת מכירות בשטח", True),  # prefix ב
        ("מנהל/ת מוצר", "אני מנהלת פיתוח", False),
        ("מפתח/ת תוכנה", "אני מתכנתת", False),  # synonym: not matched
        ("נציג/ת מכירות שטח", "נציגת מכירות", False),  # one word missing
    ],
)
def test_occupation_words(occupation: str, text: str, grounded: bool) -> None:
    assert occupation_grounded(occupation, [text]) is grounded


@pytest.mark.parametrize(
    "text, grounded",
    [("אני מקבלת אחזקת רכב", True), ("ויתרתי על הרכב", True), ("אני מוותרת על הרכב", True), ("יש לי רכב", False)],
)
def test_waiver_words(text: str, grounded: bool) -> None:
    assert waiver_grounded([text]) is grounded


# --- Invented employee facts are refused --------------------------------------------------

def test_invented_salary_is_refused_and_nothing_is_saved(ctx: ToolContext) -> None:
    result = turn(
        ConversationState("c-1"), "אני מנהלת מוצר", ctx,
        calls(("save_employee_details", {"occupation": "מנהל/ת מוצר", "base_salary": 18500})),
    )

    (saved,) = last_results(result.state, 1)
    assert code(saved) == "not_stated_by_employee"
    assert "base_salary" in saved["error"]["detail"] and "occupation" not in saved["error"]["detail"]
    assert result.state.details.occupation is None  # all-or-nothing


def test_stated_salary_is_saved(ctx: ToolContext) -> None:
    result = turn(
        ConversationState("c-1"), "אני מנהלת מוצר ומרוויחה 18.5 אלף", ctx,
        calls(("save_employee_details", {"occupation": "מנהל/ת מוצר", "base_salary": 18500})),
    )

    assert (result.state.details.occupation, result.state.details.base_salary) == ("מנהל/ת מוצר", 18500)


def test_invented_occupation_is_refused(ctx: ToolContext) -> None:
    result = turn(
        ConversationState("c-1"), "אני מפתחת תוכנה", ctx,
        calls(("save_employee_details", {"occupation": "מנהל/ת מוצר"})),
    )

    assert code(last_results(result.state, 1)[0]) == "not_stated_by_employee"
    assert result.state.details.occupation is None


@pytest.mark.parametrize("text, saved", [("יש לי 4 ילדים", 4), ("יש לי ארבעה ילדים", 4), ("שאלה על קרניבל", None)])
def test_children_must_be_stated(ctx: ToolContext, text: str, saved: int | None) -> None:
    result = turn(ConversationState("c-1"), text, ctx, calls(("save_employee_details", {"eligible_children": 4})))

    assert result.state.details.eligible_children == saved


def test_zero_commissions_need_the_employee_to_say_none(ctx: ToolContext) -> None:
    unsaid = turn(ConversationState("c-1"), "השכר שלי 18,500", ctx, calls(("save_employee_details", {"commissions": 0})))
    said = turn(ConversationState("c-1"), "אין לי עמלות", ctx, calls(("save_employee_details", {"commissions": 0})))

    assert unsaid.state.details.commissions is None
    assert said.state.details.commissions == 0


def test_invented_current_participation_is_refused(ctx: ToolContext) -> None:
    result = turn(
        ConversationState("c-1"), "יש לי קורולה קרוס", ctx,
        calls(("find_car", {"query": "קורולה קרוס"})),
        calls(("set_current_car", {"car_id": "CAR-09", "participation": 300})),
    )

    assert code(last_results(result.state, 1)[0]) == "not_stated_by_employee"
    assert result.state.details.current is None


def test_invented_current_waiver_is_refused(ctx: ToolContext) -> None:
    unsaid = turn(ConversationState("c-1"), "אני מנהלת מוצר", ctx, calls(("set_current_car", {"waiver": True})))
    said = turn(ConversationState("c-1"), "היום אני מקבלת אחזקת רכב", ctx, calls(("set_current_car", {"waiver": True})))

    assert unsaid.state.details.current is None
    assert said.state.details.current == Waiver()


E001 = "אני מנהלת מוצר, 18,500 ברוטו בלי עמלות ובונוס, פנסיה 6%, השתלמות 2.5%, 2.75 נקודות. יש לי קורולה קרוס עם השתתפות 300."
E001_DETAILS = {
    "occupation": "מנהל/ת מוצר", "base_salary": 18500, "commissions": 0, "bonus": 0,
    "pension_percent": 6, "training_fund_percent": 2.5, "credit_points": 2.75,
}


@pytest.fixture(scope="module")
def e001(ctx: ToolContext) -> ConversationState:
    return turn(
        ConversationState("c-1"), E001, ctx,
        calls(("save_employee_details", E001_DETAILS), ("find_car", {"query": "קורולה קרוס"})),
        calls(("set_current_car", {"car_id": "CAR-09", "participation": 300})),
    ).state


def test_facts_from_an_earlier_message_count(e001: ConversationState) -> None:
    assert e001.details.base_salary == 18500 and e001.details.current == CompanyCar("CAR-09", 300)


def test_invented_alternative_participation_is_refused(ctx: ToolContext, e001: ConversationState) -> None:
    invented = turn(
        e001, "כמה עם אטו 3?", ctx,
        calls(("find_car", {"query": "אטו 3"})),
        calls(("compare_alternative", {"car_id": "CAR-10", "participation": 450})),
    )
    stated = turn(
        e001, "כמה עם אטו 3 אם ההשתתפות 450?", ctx,
        calls(("find_car", {"query": "אטו 3"})),
        calls(("compare_alternative", {"car_id": "CAR-10", "participation": 450})),
    )

    assert code(last_results(invented.state, 1)[0]) == "not_stated_by_employee"
    assert last_results(stated.state, 1)[0]["status"] == "confirmed"


# --- Cars: only what the employee named ---------------------------------------------------------

def test_car_identified_from_the_models_own_query_is_not_usable(ctx: ToolContext) -> None:
    # The employee said "קורולה"; the model searched for "קורולה קרוס" itself.
    result = turn(
        ConversationState("c-1"), "יש לי קורולה, השתתפות 300", ctx,
        calls(("find_car", {"query": "קורולה קרוס"})),
        calls(("set_current_car", {"car_id": "CAR-09", "participation": 300})),
    )

    found, refused = last_results(result.state, 2)
    assert found["status"] == "identified"
    assert code(refused) == "car_not_confirmed"
    assert "CAR-09" not in result.state.usable_car_ids


def ambiguous_corolla(ctx: ToolContext) -> ConversationState:
    return turn(ConversationState("c-1"), "יש לי קורולה, השתתפות 250", ctx, calls(("find_car", {"query": "קורולה"}))).state


@pytest.mark.parametrize(
    "reply, chosen",
    [
        ("קורולה קרוס", "CAR-09"),
        ("הקורולה הרגילה", "CAR-07"),  # only COROLLA's whole name is in the reply
        ("זו מ-2026", "CAR-07"),  # the year tells them apart
    ],
)
def test_candidate_must_be_named_in_the_reply(ctx: ToolContext, reply: str, chosen: str) -> None:
    other = {"CAR-07": "CAR-09", "CAR-09": "CAR-07"}[chosen]

    wrong = turn(ambiguous_corolla(ctx), reply, ctx, calls(("set_current_car", {"car_id": other, "participation": 250})))
    right = turn(ambiguous_corolla(ctx), reply, ctx, calls(("set_current_car", {"car_id": chosen, "participation": 250})))

    assert code(last_results(wrong.state, 1)[0]) == "car_not_confirmed"
    assert right.state.details.current == CompanyCar(chosen, 250)


@pytest.mark.parametrize("reply", ["הרגילה", "כן", "את הראשונה"])
def test_vague_reply_confirms_no_candidate(ctx: ToolContext, reply: str) -> None:
    result = turn(
        ambiguous_corolla(ctx), reply, ctx,
        calls(("set_current_car", {"car_id": "CAR-07", "participation": 250}), ("set_current_car", {"car_id": "CAR-09", "participation": 250})),
    )

    assert [code(r) for r in last_results(result.state, 2)] == ["car_not_confirmed", "car_not_confirmed"]
    assert result.state.details.current is None


def test_unconfirmed_candidates_stay_offered(ctx: ToolContext) -> None:
    vague = turn(ambiguous_corolla(ctx), "רגע, מה ההבדל?", ctx)
    named = turn(vague.state, "קורולה קרוס", ctx, calls(("set_current_car", {"car_id": "CAR-09", "participation": 250})))

    assert named.state.details.current == CompanyCar("CAR-09", 250)


def test_only_the_named_alternative_becomes_usable(ctx: ToolContext, e001: ConversationState) -> None:
    listed = turn(e001, "מה החלופות?", ctx, calls(("list_alternatives", {}))).state
    assert {"CAR-10", "CAR-11"} <= listed.offered_car_ids

    result = turn(
        listed, "את האטו 3 בבקשה", ctx,
        calls(("compare_alternative", {"car_id": "CAR-11"}), ("compare_alternative", {"car_id": "CAR-10"})),
    )

    ev3, atto = last_results(result.state, 2)
    assert code(ev3) == "car_not_confirmed"
    assert atto["ok"] and atto["alternative"]["commercial_name"] == "ATTO 3"
