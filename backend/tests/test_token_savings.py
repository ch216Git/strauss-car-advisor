"""Fewer provider calls: a round that only saved details, while a comparison waits for them, is
answered by Python (app.agent.render, pattern E). Offline: scripted providers only."""

import itertools

import pytest

from app.agent.agent import TurnStop, run_turn
from app.agent.llm import Message, ToolCall
from app.agent.render import ReplyKind
from app.agent.state import ConversationState
from app.agent.tools import ToolContext

_ids = itertools.count()
MODEL_REPLY = "תשובת המודל"


class Strict:
    name = "strict"

    def __init__(self, *steps: Message) -> None:
        self._steps, self.calls = list(steps), 0

    def complete(self, system, messages, tools) -> Message:
        self.calls += 1
        assert self._steps, "the agent asked the provider for more replies than scripted"
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"s-{next(_ids)}", n, a) for n, a in name_and_arguments])


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


FIRST = "שלום, אני מנהלת מוצר ויש לי היום קורולה קרוס עם השתתפות של 300 ש״ח. כמה ישתנה לי הנטו אם אעבור לאטו 3?"
FIRST_ROUND = calls(
    ("save_employee_details", {"occupation": "מנהלת מוצר"}),
    ("set_current_car", {"car": "קורולה קרוס", "participation": 300}),
    ("compare_alternative", {"car": "אטו 3"}),
)
SALARY = ("save_employee_details", {"base_salary": 18500})


@pytest.fixture
def interviewing(ctx: ToolContext) -> ConversationState:
    """E-001 after the first message: eligibility passed, the comparison waits for calculation details."""
    result = run_turn(ConversationState("c-1"), FIRST, Strict(FIRST_ROUND), ctx)
    assert result.outcome.kind.value == "missing_details" and result.outcome.stage.value == "calculation"
    return result.state


def test_saving_details_during_the_interview_needs_one_provider_call(ctx: ToolContext, interviewing) -> None:
    events: list[str] = []
    provider = Strict(calls(SALARY))  # a second call fails the test

    result = run_turn(interviewing, "השכר שלי 18,500 ברוטו", provider, ctx, events=events.append)

    assert (provider.calls, result.stop) == (1, TurnStop.RENDERED)
    assert any(e.endswith(f"deterministic reply ({ReplyKind.DETAILS_SAVED.value})") for e in events)
    assert result.reply == "\n".join([
        "תודה, שמרתי.",
        "כדי לחשב את השינוי בנטו חסרים לי עוד כמה פרטים:",
        "• עמלות ובונוס חודשיים (אם אין, אפשר לכתוב שאין)",
        "• אחוז ההפרשה שלך לפנסיה ולקרן השתלמות",
        "• מספר נקודות הזיכוי שלך",
    ])


@pytest.mark.parametrize(
    "text",
    [
        "השכר שלי 18,500 ברוטו. מה זה שווי שימוש",  # a question word
        "השכר שלי 18,500 ברוטו, זה מספיק?",  # a question mark
        "השכר שלי 18,500, ספר לי גם על אטו 3",  # a request for information
    ],
)
def test_a_message_that_also_asks_goes_to_the_provider(ctx: ToolContext, interviewing, text: str) -> None:
    provider = Strict(calls(SALARY), Message.assistant(MODEL_REPLY))
    result = run_turn(interviewing, text, provider, ctx)
    assert (provider.calls, result.stop, result.reply) == (2, TurnStop.REPLIED, MODEL_REPLY)


def test_no_salary_interview_before_a_comparison_asked_for_it(ctx: ToolContext) -> None:
    # Eligibility comes first: details given before any car was checked are left to the provider.
    provider = Strict(calls(SALARY), Message.assistant(MODEL_REPLY))
    result = run_turn(ConversationState("c-1"), "השכר שלי 18,500 ברוטו", provider, ctx)
    assert (provider.calls, result.stop) == (2, TurnStop.REPLIED)


def test_not_during_the_eligibility_stage(ctx: ToolContext) -> None:
    state = run_turn(ConversationState("c-1"), "אפשר לקבל קרניבל?", Strict(calls(("compare_alternative", {"car": "קרניבל"}))), ctx).state
    provider = Strict(calls(SALARY), Message.assistant(MODEL_REPLY))
    result = run_turn(state, "השכר שלי 18,500 ברוטו", provider, ctx)
    assert (provider.calls, result.stop) == (2, TurnStop.REPLIED)


def test_when_nothing_is_missing_the_provider_takes_the_next_step(ctx: ToolContext, interviewing) -> None:
    everything = ("save_employee_details", {
        "base_salary": 18500, "commissions": 0, "bonus": 0, "credit_points": 2.75, "pension_percent": 6, "training_fund_percent": 2.5,
    })
    text = "השכר שלי 18,500 ברוטו, אין עמלות ובונוס, 2.75 נקודות זיכוי, פנסיה 6% וקרן השתלמות 2.5%"
    provider = Strict(calls(everything), calls(("compare_alternative", {"car": "אטו 3"})), Message.assistant(MODEL_REPLY))

    result = run_turn(interviewing, text, provider, ctx)

    assert provider.calls == 3 and result.state.last_comparison is not None  # the comparison was not skipped
