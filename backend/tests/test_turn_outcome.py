"""TurnResult.outcome (app.agent.outcome): what a turn decided, as data. Scripted providers only."""

import itertools

import pytest

from app.agent import render
from app.agent.agent import PROVIDER_ERROR_REPLY, TurnStop, run_turn
from app.agent.llm import Message, ProviderError, ProviderErrorKind, Role, ToolCall
from app.agent.outcome import MissingStage, OutcomeKind
from app.agent.state import ComparisonStatus, ConversationState
from app.agent.tools import ToolContext
from app.agent.views import EXPLAINED_CHANGES

_ids = itertools.count()


class StrictProvider:
    name = "strict"

    def __init__(self, *steps) -> None:
        self._steps = list(steps)

    def complete(self, system, messages, tools) -> Message:
        assert self._steps, "the agent asked the provider for more replies than scripted"
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"o-{next(_ids)}", n, a) for n, a in name_and_arguments])


def results(state: ConversationState) -> list[dict]:
    return [dict(r.content) for m in state.messages if m.role == Role.TOOL for r in m.tool_results]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


CARNIVAL = ("compare_alternative", {"car": "קרניבל"})
CURRENT = ("set_current_car", {"car": "קורולה קרוס", "participation": 300})
ATTO = ("compare_alternative", {"car": "אטו 3"})
E001 = (
    ("שלום, אני מנהלת מוצר ויש לי היום קורולה קרוס עם השתתפות של 300 ש״ח. כמה ישתנה לי הנטו אם אעבור לאטו 3?",
     calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), CURRENT, ATTO)),
    ("השכר שלי 18,500 ברוטו, אין לי עמלות ובלי בונוס.",
     calls(("save_employee_details", {"base_salary": 18500, "commissions": 0, "bonus": 0}), CURRENT, ATTO)),
    ("יש לי 2.75 נקודות זיכוי, פנסיה 6% וקרן השתלמות 2.5%.",
     calls(("save_employee_details", {"credit_points": 2.75, "pension_percent": 6, "training_fund_percent": 2.5}), CURRENT, ATTO)),
)


def converse(ctx, *turns, state=None):
    state = state or ConversationState("c-1")
    out = []
    for text, *steps in turns:
        turn = run_turn(state, text, StrictProvider(*steps), ctx)
        out.append(turn)
        state = turn.state
    return out


def test_eligibility_question_is_the_eligibility_stage(ctx: ToolContext) -> None:
    (turn,) = converse(ctx, ("אפשר לקבל קרניבל?", calls(CARNIVAL)))

    outcome = turn.outcome
    assert (outcome.kind, outcome.stage, outcome.missing) == (
        OutcomeKind.MISSING_DETAILS, MissingStage.ELIGIBILITY, ("eligible_children",)
    )
    assert outcome.car_id in ctx.rules.family_track.cars and not outcome.waiver
    assert outcome.comparison is None


def test_not_eligible_keeps_reason_and_facts(ctx: ToolContext) -> None:
    _, turn = converse(
        ctx,
        ("אפשר לקבל קרניבל?", calls(CARNIVAL)),
        ("יש לי 3 ילדים", calls(("save_employee_details", {"eligible_children": 3}), CARNIVAL)),
    )

    outcome = turn.outcome
    assert (outcome.kind, outcome.reason) == (OutcomeKind.NOT_ELIGIBLE, "not_enough_children")
    rule = ctx.rules.family_track
    assert {k: outcome.facts[k] for k in ("min_children", "child_max_age", "eligible_children")} == {
        "min_children": rule.min_children, "child_max_age": rule.child_max_age, "eligible_children": 3,
    }
    assert outcome.car_id in rule.cars


def test_calculation_details_are_the_calculation_stage(ctx: ToolContext) -> None:
    turn = converse(ctx, E001[0])[0]

    outcome = turn.outcome
    assert (outcome.kind, outcome.stage) == (OutcomeKind.MISSING_DETAILS, MissingStage.CALCULATION)
    assert "base_salary" in outcome.missing and outcome.car_id == "CAR-10"


def test_e001_comparison_is_captured(ctx: ToolContext) -> None:
    turn = converse(ctx, *E001)[-1]

    outcome = turn.outcome
    assert (outcome.kind, outcome.car_id) == (OutcomeKind.COMPARISON, "CAR-10")
    assert outcome.comparison.status == ComparisonStatus.ESTIMATE
    assert (outcome.comparison.current_net, outcome.comparison.alternative_net, outcome.comparison.net_difference) == (
        11032, 11693, 661
    )


def test_a_later_call_decides_but_the_comparison_is_kept(ctx: ToolContext) -> None:
    *_, ready = converse(ctx, *E001)
    ask = ("ask_representative", {"topic": "alternative_participation", "question": "ההשתתפות ב-ATTO 3?", "car_id": "CAR-10"})
    (turn,) = converse(
        ctx, ("ומה עם אטו 3? ותשאל את הנציג", calls(("compare_alternative", {"car_id": "CAR-10"}), ask), Message.assistant("שאלתי")),
        state=ready.state,
    )

    outcome = turn.outcome
    assert (outcome.kind, outcome.question_topic, outcome.already_open) == (
        OutcomeKind.QUESTION_SENT, "alternative_participation", False
    )
    assert outcome.question_id == turn.state.rep_questions[-1].question_id
    assert outcome.comparison is not None and outcome.comparison.net_difference == 661


def test_car_choice_for_the_current_car(ctx: ToolContext) -> None:
    (turn,) = converse(ctx, ("יש לי קורולה, השתתפות 250", calls(("set_current_car", {"car": "קורולה", "participation": 250}))))

    outcome = turn.outcome
    assert (outcome.kind, outcome.choice_target) == (OutcomeKind.CAR_CHOICE, "current")
    assert set(outcome.choice_car_ids) == {"CAR-07", "CAR-09"}


def test_blocked_by_representative(ctx: ToolContext) -> None:
    unknown = "אני מנהלת מוצר, יש לי מאזדה CX-5 2019 בהשתתפות 300"
    asked, blocked = converse(
        ctx,
        (unknown,
         calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), ("set_current_car", {"car": "מאזדה CX-5 2019", "participation": 300})),
         calls(("ask_representative", {"topic": "current_car_not_in_catalogue", "question": "מאזדה CX-5 2019"})),
         Message.assistant("הרכב לא בקטלוג; שאלתי את הנציג.")),
        ("ומה עם אטו 3?", calls(ATTO)),
    )

    assert asked.outcome.kind == OutcomeKind.QUESTION_SENT  # decided in the second round
    assert (blocked.outcome.kind, blocked.outcome.car_id) == (OutcomeKind.BLOCKED, "CAR-10")


def test_no_decisive_call_gives_no_outcome(ctx: ToolContext) -> None:
    (turn,) = converse(ctx, ("מה זה טוסון?", calls(("find_car", {"query": "טוסון", "purpose": "information"})), Message.assistant("רכב")))

    assert turn.outcome is None


def test_provider_error_has_no_outcome(ctx: ToolContext) -> None:
    (turn,) = converse(ctx, ("שלום", ProviderError("fake", ProviderErrorKind.UNAVAILABLE, "down")))

    assert (turn.stop, turn.reply, turn.outcome) == (TurnStop.PROVIDER_ERROR, PROVIDER_ERROR_REPLY, None)


def test_the_model_sees_the_same_tool_results(ctx: ToolContext) -> None:
    """The alternative recorded for screens never reaches the tool JSON the model reads."""
    (carnival,) = converse(ctx, ("אפשר לקבל קרניבל?", calls(CARNIVAL)))
    first = converse(ctx, E001[0])[0]

    assert set(results(carnival.state)[-1]["error"]) == {"code", "detail", "missing", "stage", "family_track_rule"}
    assert set(results(first.state)[-1]["error"]) == {"code", "detail", "missing"}


def test_explained_changes_follow_the_replies() -> None:
    assert EXPLAINED_CHANGES == tuple(item for item in render._CHANGE_ORDER if item != "electric_gross")
