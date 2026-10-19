"""Smart questions: facts that decide eligibility are asked first, and a refusal ends the questions.

The verdict always comes from Step 3 (comparison.check_eligibility, the same rule compare() uses).
Offline: scripted providers only.
"""

import itertools

import pytest

from app.agent.agent import TurnStop, run_turn
from app.agent.llm import Message, Role, ToolCall
from app.agent.output_check import unsupported_numbers
from app.agent.render import RoundCall, deterministic_reply, not_eligible_text
from app.agent.state import ConversationState, RepTopic
from app.agent.tools import (
    ComparisonSummary,
    ToolContext,
    ToolError,
    ToolErrorCode,
    ask_representative,
    compare_alternative,
    save_employee_details,
    set_current_car,
)
from app.comparison import CompanyCar, Employee, Waiver, check_eligibility, compare
from app.errors import NotEligibleError

_ids = itertools.count()

# Questions about money: none may be asked before eligibility is decided.
FINANCIAL = ("שכר", "עמלות", "בונוס", "פנסיה", "השתלמות", "נקודות הזיכוי", "השתתפות")


class StrictProvider:
    name = "strict"

    def __init__(self, *steps: Message) -> None:
        self._steps, self.calls = list(steps), 0

    def complete(self, system, messages, tools) -> Message:
        self.calls += 1
        assert self._steps, "the agent asked the provider for more replies than scripted"
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"s-{next(_ids)}", n, a) for n, a in name_and_arguments])


def results(state: ConversationState) -> list[dict]:
    return [dict(r.content) for m in state.messages if m.role == Role.TOOL for r in m.tool_results]


def conversation(ctx: ToolContext, *turns: tuple[str, Message]):
    """Runs the turns with one provider that allows exactly one call per turn."""
    provider = StrictProvider(*(step for _, step in turns))
    state, replies = ConversationState("c-1"), []
    for text, _ in turns:
        turn = run_turn(state, text, provider, ctx)
        assert turn.stop == TurnStop.RENDERED, turn.reply
        assert unsupported_numbers(turn.reply, turn.state) == ()
        replies.append(turn.reply)
        state = turn.state
    return state, replies, provider


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


def sources(ctx: ToolContext) -> dict:
    return {"rules": ctx.rules, "catalog": ctx.catalog}


# --- Step 3: check_eligibility is compare()'s rule ------------------------------------------------

def _verdict(call) -> str:
    try:
        return call()
    except NotEligibleError as e:
        return f"refused:{e.reason}"


@pytest.mark.parametrize("children", [0, 5])
def test_eligibility_matches_compare_for_every_occupation_and_alternative(ctx: ToolContext, children: int) -> None:
    current = CompanyCar("CAR-09", 300)
    for occupation in ctx.catalog.occupations:
        employee = Employee(occupation, 18500, 0, 0, 0.06, 0.025, 2.75, children, current)
        for alternative in [*(CompanyCar(car_id) for car_id in ctx.catalog.cars), Waiver()]:
            expected = _verdict(lambda: compare(employee, alternative, repository=ctx.repository, lookup=ctx.lookup, **sources(ctx)).eligibility_basis)
            check = lambda: check_eligibility(
                alternative, occupation_name=occupation, eligible_children=children, current=current, **sources(ctx)
            )
            actual = _verdict(lambda: check().basis)
            assert actual == expected, (occupation, alternative)
            if not actual.startswith("refused"):
                assert check().missing == ()


@pytest.mark.parametrize(
    "alternative, missing, basis",
    [
        (Waiver(), ("occupation",), None),  # some occupations may not waive
        (CompanyCar("CAR-14"), ("occupation",), None),  # group ד: only some groups may choose it
        (CompanyCar("CAR-10"), ("occupation",), None),  # group ג
        (CompanyCar("CAR-01"), (), "own_group_or_lower"),  # group א: every occupation may choose it
        (CompanyCar("CAR-19"), ("eligible_children",), None),  # family track: children decide, not occupation
    ],
)
def test_unknown_facts_are_asked_only_when_they_decide(ctx: ToolContext, alternative, missing, basis) -> None:
    check = check_eligibility(alternative, occupation_name=None, eligible_children=None, **sources(ctx))

    assert (check.missing, check.basis) == (missing, basis)


def test_children_decide_the_family_track_without_the_occupation(ctx: ToolContext) -> None:
    rule = ctx.rules.family_track
    with pytest.raises(NotEligibleError) as refused:
        check_eligibility(CompanyCar("CAR-19"), occupation_name=None, eligible_children=rule.min_children - 1, **sources(ctx))
    assert refused.value.reason == "not_enough_children"

    check = check_eligibility(CompanyCar("CAR-19"), occupation_name=None, eligible_children=rule.min_children, **sources(ctx))
    assert (check.basis, check.missing) == ("family_track", ())


@pytest.mark.parametrize(
    "current, alternative",
    [(CompanyCar("CAR-14", 500), CompanyCar("CAR-14")), (CompanyCar("CAR-19", 500), CompanyCar("CAR-19", 600))],
)
def test_keeping_the_current_car_is_never_gated(ctx: ToolContext, current, alternative) -> None:
    check = check_eligibility(alternative, occupation_name="מנהל/ת מוצר", eligible_children=None, current=current, **sources(ctx))

    assert (check.basis, check.missing) == ("keeps_current_situation", ())


def test_compare_still_raises_value_error_for_unknown_children(ctx: ToolContext) -> None:
    employee = Employee("מנהל/ת מוצר", 18500, 0, 0, 0.06, 0.025, 2.75, None, CompanyCar("CAR-09", 300))

    with pytest.raises(ValueError, match="The number of children is needed to check a family-track car; it is unknown"):
        compare(employee, CompanyCar("CAR-19"), repository=ctx.repository, lookup=ctx.lookup, **sources(ctx))


# --- The tool: blocking first, then the gate, then calculation details ------------------------------

def test_representative_blocking_comes_before_the_gate(ctx: ToolContext) -> None:
    state = ask_representative(ConversationState("c-1"), ctx, RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE.value, "מאזדה CX-5").state

    error = compare_alternative(state, ctx, waiver=True).result

    assert isinstance(error, ToolError) and error.code == ToolErrorCode.BLOCKED_BY_REPRESENTATIVE


def test_gate_asks_only_the_deciding_fact(ctx: ToolContext) -> None:
    error = compare_alternative(ConversationState("c-1"), ctx, waiver=True).result

    assert (error.code, error.missing, error.stage) == (ToolErrorCode.MISSING_DETAILS, ("occupation",), "eligibility")


def test_after_the_gate_every_calculation_detail_is_asked_together(ctx: ToolContext) -> None:
    state = save_employee_details(ConversationState("c-1"), ctx, occupation="מפתח/ת תוכנה").state

    error = compare_alternative(state, ctx, waiver=True).result

    assert error.stage is None
    assert error.missing == ("base_salary", "commissions", "bonus", "pension_rate", "training_fund_rate", "credit_points", "current")


def test_keeping_the_current_family_car_does_not_need_children(ctx: ToolContext) -> None:
    state = save_employee_details(
        ConversationState("c-1"), ctx, occupation="מנהל/ת מוצר", base_salary=18500, commissions=0, bonus=0,
        pension_percent=6, training_fund_percent=2.5, credit_points=2.75,
    ).state
    state = set_current_car(state, ctx, car_id="CAR-19", participation=500).state

    summary = compare_alternative(state, ctx, car_id="CAR-19", participation=600).result

    assert isinstance(summary, ComparisonSummary)
    assert summary.result.eligibility_basis == "keeps_current_situation"


def test_refusal_carries_its_facts_from_data(ctx: ToolContext) -> None:
    state = save_employee_details(ConversationState("c-1"), ctx, occupation="מנהל/ת מוצר").state

    error = compare_alternative(state, ctx, car_id="CAR-14").result

    assert (error.code, error.reason) == (ToolErrorCode.NOT_ELIGIBLE, "group_too_high")
    assert error.facts == {
        "new_selection": True, "occupation": "מנהל/ת מוצר", "employee_group": "ג",
        "car_id": "CAR-14", "commercial_name": "MODEL 3", "car_group": "ד", "choose_from": ctx.rules.choose_from,
    }


# --- Conversations: no financial questions before a decisive gate -------------------------------------

def test_waiver_for_an_occupation_that_cannot_waive(ctx: ToolContext) -> None:
    state, replies, provider = conversation(
        ctx,
        ("אני רוצה לוותר על הרכב. כמה זה ישנה לי בנטו?", calls(("compare_alternative", {"waiver": True}))),
        ("אני נהג הפצה", calls(("save_employee_details", {"occupation": "נהג הפצה"}), ("compare_alternative", {"waiver": True}))),
    )

    assert replies == [
        "כדי לבדוק אם החלופה הזו אפשרית עבורך, צריך קודם לדעת:\n• התפקיד שלך",
        "לפי טבלת התפקידים, בתפקיד נהג/ת הפצה אי אפשר לבחור כעת בוויתור על הרכב. אפשר לבדוק חלופה אחרת.",
    ]
    assert provider.calls == 2
    assert not any(word in reply for reply in replies for word in FINANCIAL)
    assert state.details.base_salary is None and state.last_comparison is None


def test_waiver_for_an_occupation_that_can_waive_asks_the_rest_together(ctx: ToolContext) -> None:
    _, replies, _ = conversation(
        ctx,
        ("אני רוצה לוותר על הרכב. כמה זה ישנה לי בנטו?", calls(("compare_alternative", {"waiver": True}))),
        ("אני מפתחת תוכנה", calls(("save_employee_details", {"occupation": "מפתחת תוכנה"}), ("compare_alternative", {"waiver": True}))),
    )

    rest = replies[1]
    assert rest.startswith("כדי לחשב את השינוי בנטו חסרים לי עוד כמה פרטים:")
    for wanted in ("רכב חברה", "שכר הבסיס", "עמלות ובונוס", "לפנסיה ולקרן השתלמות", "נקודות הזיכוי"):
        assert wanted in rest
    assert "ילד" not in rest


def test_family_car_with_too_few_children(ctx: ToolContext) -> None:
    rule = ctx.rules.family_track
    state, replies, provider = conversation(
        ctx,
        ("אפשר לקבל קרניבל?", calls(("compare_alternative", {"car": "קרניבל"}))),
        ("יש לי 3 ילדים", calls(("save_employee_details", {"eligible_children": 3}), ("compare_alternative", {"car": "קרניבל"}))),
    )

    assert replies[0] == (
        "כדי לבדוק אם החלופה הזו אפשרית עבורך, צריך קודם לדעת:\n"
        f"• מספר הילדים שלך עד גיל {rule.child_max_age} (נדרש לרכב מהמסלול המשפחתי)"
    )
    assert replies[1] == (
        f"אי אפשר לבחור ב-CARNIVAL כבחירה חדשה במסלול המשפחתי: המסלול מיועד למי שיש לו לפחות {rule.min_children} "
        f"ילדים עד גיל {rule.child_max_age}, ולפי מה שמסרת יש לך 3. אפשר לבדוק חלופה אחרת."
    )
    assert provider.calls == 2
    assert "התפקיד" not in replies[0]  # the occupation does not decide the family track
    assert not any(word in reply for reply in replies for word in FINANCIAL)


def test_family_car_with_enough_children_asks_the_rest_together(ctx: ToolContext) -> None:
    _, replies, _ = conversation(
        ctx,
        ("אפשר לקבל קרניבל?", calls(("compare_alternative", {"car": "קרניבל"}))),
        ("יש לי 5 ילדים", calls(("save_employee_details", {"eligible_children": 5}), ("compare_alternative", {"car": "קרניבל"}))),
    )

    assert replies[1].startswith("כדי לחשב את השינוי בנטו חסרים לי עוד כמה פרטים:")
    assert "התפקיד שלך" in replies[1] and "שכר הבסיס" in replies[1] and "ילד" not in replies[1]


def test_car_above_the_group(ctx: ToolContext) -> None:
    _, replies, provider = conversation(
        ctx,
        ("אני רוצה טסלה מודל 3", calls(("compare_alternative", {"car": "טסלה מודל 3"}))),
        ("אני מנהלת מוצר", calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), ("compare_alternative", {"car": "טסלה מודל 3"}))),
    )

    assert replies == [
        "כדי לבדוק אם החלופה הזו אפשרית עבורך, צריך קודם לדעת:\n• התפקיד שלך",
        "אי אפשר לבחור ב-MODEL 3 כבחירה חדשה: הרכב בקבוצה ד, והתפקיד שלך בקבוצה ג, "
        "שמאפשרת לבחור רכב מהקבוצה שלך או מקבוצה נמוכה ממנה. אפשר לבדוק חלופה אחרת.",
    ]
    assert provider.calls == 2
    assert not any(word in reply for reply in replies for word in FINANCIAL)


# --- Unknown refusals stay with the provider ----------------------------------------------------------

def test_unknown_reason_or_missing_facts_are_not_rendered(ctx: ToolContext) -> None:
    assert not_eligible_text({"reason": "something_new", "facts": {"occupation": "x"}}) is None
    assert not_eligible_text({"reason": "group_too_high", "facts": {"commercial_name": "X"}}) is None
    assert not_eligible_text({"reason": "group_too_high", "facts": {
        "commercial_name": "X", "car_group": "ד", "employee_group": "ג", "choose_from": "some_future_rule"
    }}) is None

    unknown = RoundCall("compare_alternative", {"ok": False, "error": {"code": "not_eligible", "reason": "something_new"}})
    assert deterministic_reply("ומה עם X?", [unknown], ConversationState("c-1"), ctx.rules) is None


def test_unknown_reason_goes_to_the_provider(ctx: ToolContext, monkeypatch) -> None:
    from app.agent import tools as tools_module

    def refuse(*args, **kwargs):
        raise NotEligibleError("some_future_reason", "a rule the renderer does not know")

    monkeypatch.setattr(tools_module, "check_eligibility", refuse)
    provider = StrictProvider(calls(("compare_alternative", {"waiver": True})), Message.assistant("תשובת המודל"))

    turn = run_turn(ConversationState("c-1"), "ואם אוותר על הרכב?", provider, ctx)

    assert results(turn.state)[-1]["error"]["reason"] == "some_future_reason"
    assert (provider.calls, turn.stop, turn.reply) == (2, TurnStop.REPLIED, "תשובת המודל")
