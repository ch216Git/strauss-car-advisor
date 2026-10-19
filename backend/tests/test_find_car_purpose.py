"""find_car's required purpose: "alternative" runs the existing compare_alternative(car=query) path.

Uncertain -> provider, certain -> deterministic Python: a comparison reached through find_car is
rendered only for a plain request. Offline: scripted providers only.
"""

import itertools

import pytest

from app.agent.agent import TurnStop, run_turn
from app.agent.llm import Message, Role, ToolCall
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.providers.gemini import to_gemini_request
from app.agent.outcome import OutcomeKind
from app.agent.render import choice_request, plain_request
from app.agent.state import ConversationState
from app.agent.tool_registry import SPECS, execute
from app.agent.tools import ToolContext

_ids = itertools.count()
MODEL_REPLY = "תשובת המודל"


class StrictProvider:
    name = "strict"

    def __init__(self, *steps: Message) -> None:
        self._steps, self.calls = list(steps), 0

    def complete(self, system, messages, tools) -> Message:
        self.calls += 1
        assert self._steps, "the agent asked the provider for more replies than scripted"
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"p-{next(_ids)}", n, a) for n, a in name_and_arguments])


def find(query: str, purpose: str) -> tuple[str, dict]:
    return ("find_car", {"query": query, "purpose": purpose})


def results(state: ConversationState) -> list[dict]:
    return [dict(r.content) for m in state.messages if m.role == Role.TOOL for r in m.tool_results]


def turn(ctx, text: str, *steps: Message, state: ConversationState | None = None):
    provider = StrictProvider(*steps)
    return run_turn(state or ConversationState("c-1"), text, provider, ctx), provider


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


# --- 1. The family question: one provider call, children only ----------------------------------------

def test_alternative_purpose_reaches_the_eligibility_gate_in_one_call(ctx: ToolContext) -> None:
    result, provider = turn(ctx, "אפשר לקבל קרניבל?", calls(find("קרניבל", "alternative")))

    assert (provider.calls, result.stop) == (1, TurnStop.RENDERED)
    assert result.reply == (
        "כדי לבדוק אם החלופה הזו אפשרית עבורך, צריך קודם לדעת:\n"
        f"• מספר הילדים שלך עד גיל {ctx.rules.family_track.child_max_age} (נדרש לרכב מהמסלול המשפחתי)"
    )
    (content,) = results(result.state)
    assert content["ran"] == "compare_alternative"
    assert (content["error"]["code"], content["error"]["missing"], content["error"]["stage"]) == (
        "missing_details", ["eligible_children"], "eligibility"
    )
    assert "CAR-19" in result.state.usable_car_ids


# --- 2-4. Information questions never become a deterministic interview ----------------------------------

def test_information_purpose_does_not_compare(ctx: ToolContext) -> None:
    result, provider = turn(ctx, "מה זה קרניבל?", calls(find("קרניבל", "information")), Message.assistant(MODEL_REPLY))

    (content,) = results(result.state)
    assert content["status"] == "identified" and "ran" not in content
    assert (provider.calls, result.stop, result.reply) == (2, TurnStop.REPLIED, MODEL_REPLY)
    assert result.state.alternative is None and result.state.last_comparison is None


def test_wrong_alternative_purpose_on_an_information_question_goes_to_the_provider(ctx: ToolContext) -> None:
    result, provider = turn(ctx, "מה זה קרניבל?", calls(find("קרניבל", "alternative")), Message.assistant(MODEL_REPLY))

    assert results(result.state)[0]["ran"] == "compare_alternative"  # it ran, and only informs the model
    assert (provider.calls, result.stop, result.reply) == (2, TurnStop.REPLIED, MODEL_REPLY)


@pytest.mark.parametrize("text", ["ספר לי על קרניבל", "תסביר לי על הקרניבל", "אפשר פרטים על קרניבל"])
def test_requests_for_information_are_not_rendered(ctx: ToolContext, text: str) -> None:
    result, provider = turn(ctx, text, calls(find("קרניבל", "alternative")), Message.assistant(MODEL_REPLY))

    assert (provider.calls, result.stop) == (2, TurnStop.REPLIED)


@pytest.mark.parametrize("purpose", ["information", "alternative"])
def test_browsing_and_a_car_question_get_no_deterministic_comparison(ctx: ToolContext, purpose: str) -> None:
    result, provider = turn(
        ctx, "תראה לי איזה רכבים יש, ואגב מה זה קרניבל?",
        calls(("list_alternatives", {}), find("קרניבל", purpose)), Message.assistant(MODEL_REPLY),
    )

    assert (provider.calls, result.stop, result.reply) == (2, TurnStop.REPLIED, MODEL_REPLY)


# --- 5. Today's car is not compared ------------------------------------------------------------------

def test_current_purpose_does_not_compare(ctx: ToolContext) -> None:
    result, provider = turn(ctx, "יש לי קרניבל", calls(find("קרניבל", "current")), Message.assistant(MODEL_REPLY))

    (content,) = results(result.state)
    assert content["status"] == "identified" and "ran" not in content
    assert (provider.calls, result.stop) == (2, TurnStop.REPLIED)
    assert result.state.alternative is None and result.state.details.current is None


# --- 6-8. Ambiguous, not found, several cars ---------------------------------------------------------

def test_ambiguous_alternative_keeps_the_car_choice_flow(ctx: ToolContext) -> None:
    result, provider = turn(ctx, "אפשר לקבל קורולה?", calls(find("קורולה", "alternative")))

    (content,) = results(result.state)
    assert content["error"]["code"] == "car_needs_choice" and content["ran"] == "compare_alternative"
    assert (provider.calls, result.stop) == (1, TurnStop.RENDERED)
    assert "הרכב החלופי" in result.reply and "COROLLA (2026)" in result.reply and "COROLLA CROSS (2025)" in result.reply
    assert {"CAR-07", "CAR-09"} <= result.state.offered_car_ids
    assert not ({"CAR-07", "CAR-09"} & result.state.usable_car_ids)  # nothing guessed
    assert result.state.alternative is None


COROLLA_CHOICE = ("COROLLA (2026)", "COROLLA CROSS (2025)")


@pytest.mark.parametrize(
    "text",
    ["כמה ישתנה לי הנטו אם אעבור לקורולה?", "איזו קורולה אפשר לקבל?", "האם אפשר לעבור לקורולה?", "מה יוצא לי אם אעבור לקורולה?"],
)
def test_a_question_about_switching_renders_the_car_choice_after_one_call(ctx: ToolContext, text: str) -> None:
    # Regression: a single question word made the via-find_car guard reject the round, and the second
    # provider call (rate limited in the real flow) turned a ready car choice into a temporary error.
    events: list[str] = []
    provider = StrictProvider(calls(find("קורולה", "alternative")))  # a second call fails the test
    result = run_turn(ConversationState("c-1"), text, provider, ctx, events=events.append)

    assert any(e.endswith(": find_car→compare_alternative=car_needs_choice") for e in events)
    assert (provider.calls, result.stop, result.tool_rounds) == (1, TurnStop.RENDERED, 1)
    assert "הרכב החלופי" in result.reply and all(name in result.reply for name in COROLLA_CHOICE)
    assert (result.outcome.kind, result.outcome.choice_target) == (OutcomeKind.CAR_CHOICE, "alternative")
    assert set(result.outcome.choice_car_ids) == {"CAR-07", "CAR-09"}
    assert not ({"CAR-07", "CAR-09"} & result.state.usable_car_ids)


@pytest.mark.parametrize(
    "text",
    [
        "מה זה קורולה?",  # what the car is
        "ספר לי על הקורולה",
        "למה אני לא יכול לקבל קורולה?",  # an explanation
        "כמה ישתנה לי הנטו אם אעבור לקורולה, ומה עם טוסון?",  # two questions
        "איך עוברים לקורולה? כמה זה עולה?",
    ],
)
def test_information_and_compound_requests_with_a_car_choice_go_to_the_provider(ctx: ToolContext, text: str) -> None:
    result, provider = turn(ctx, text, calls(find("קורולה", "alternative")), Message.assistant(MODEL_REPLY))

    assert (provider.calls, result.stop, result.reply) == (2, TurnStop.REPLIED, MODEL_REPLY)
    assert result.outcome.kind == OutcomeKind.CAR_CHOICE  # the decision is still known as data


def test_a_comparison_via_find_car_still_needs_a_plain_request(ctx: ToolContext) -> None:
    # The car-choice relaxation does not reach other outcomes: one question word still asks the provider.
    result, provider = turn(
        ctx, "כמה ישתנה לי הנטו אם אעבור לקרניבל?", calls(find("קרניבל", "alternative")), Message.assistant(MODEL_REPLY)
    )

    assert results(result.state)[0]["error"]["code"] == "missing_details"
    assert (provider.calls, result.stop) == (2, TurnStop.REPLIED)


@pytest.mark.parametrize("text", ["כמה ישתנה לי הנטו אם אעבור לקורולה?", "מה זה קורולה?"])
def test_direct_compare_alternative_car_choice_is_unchanged(ctx: ToolContext, text: str) -> None:
    result, provider = turn(ctx, text, calls(("compare_alternative", {"car": "קורולה"})))

    assert (provider.calls, result.stop) == (1, TurnStop.RENDERED)
    assert "הרכב החלופי" in result.reply and all(name in result.reply for name in COROLLA_CHOICE)


def test_unknown_car_keeps_the_not_found_flow(ctx: ToolContext) -> None:
    result, provider = turn(ctx, "אפשר לקבל פרארי?", calls(find("פרארי", "alternative")), Message.assistant(MODEL_REPLY))

    assert results(result.state)[0]["error"]["code"] == "car_not_found"
    assert (provider.calls, result.stop, result.reply) == (2, TurnStop.REPLIED, MODEL_REPLY)


def test_car_the_employee_did_not_name_is_not_used(ctx: ToolContext) -> None:
    result, provider = turn(ctx, "אפשר לקבל רכב משפחתי?", calls(find("קרניבל", "alternative")), Message.assistant(MODEL_REPLY))

    assert results(result.state)[0]["error"]["code"] == "car_not_confirmed"
    assert (provider.calls, result.stop) == (2, TurnStop.REPLIED)


def test_two_alternatives_in_one_round_go_to_the_provider(ctx: ToolContext) -> None:
    result, provider = turn(
        ctx, "אפשר לקבל קרניבל או טוסון", calls(find("קרניבל", "alternative"), find("טוסון", "alternative")),
        Message.assistant(MODEL_REPLY),
    )

    assert [c["ran"] for c in results(result.state)] == ["compare_alternative", "compare_alternative"]
    assert (provider.calls, result.stop) == (2, TurnStop.REPLIED)


# --- Arguments --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({"query": "קרניבל"}, "missing required arguments ['purpose']"),
        ({"query": "קרניבל", "purpose": "compare"}, "purpose must be one of"),
        ({"query": "קרניבל", "purpose": "alternative", "participation": 450}, "unknown arguments ['participation']"),
    ],
)
def test_purpose_is_required_and_participation_cannot_pass(ctx: ToolContext, arguments, fragment) -> None:
    execution = execute(ToolCall("x", "find_car", arguments), ConversationState("c-1"), ctx)

    assert execution.content["error"]["code"] == "invalid_arguments" and fragment in execution.content["error"]["detail"]
    assert execution.tool == "find_car"


# --- 9. Provider history keeps the original call -------------------------------------------------------

def test_result_answers_the_find_car_call_while_the_comparison_is_recorded(ctx: ToolContext) -> None:
    state = run_turn(ConversationState("c-1"), "אפשר לקבל קרניבל?", StrictProvider(Message.assistant("x")), ctx).state
    call = ToolCall("fc-42", "find_car", {"query": "קרניבל", "purpose": "alternative"})

    execution = execute(call, state, ctx)
    assert execution.tool == "compare_alternative"

    result, _ = turn(ctx, "אפשר לקבל קרניבל?", Message.assistant(tool_calls=[call]))
    assistant, tool = result.state.messages[1], result.state.messages[2]
    assert (tool.tool_results[0].call_id, tool.tool_results[0].name) == ("fc-42", "find_car")
    assert assistant.tool_calls[0] == call

    contents, _ = to_gemini_request(SYSTEM_PROMPT, result.state.messages, SPECS)
    function_call = contents[1]["parts"][0]["function_call"]
    function_response = contents[2]["parts"][0]["function_response"]
    assert (function_call["name"], function_call["id"]) == (function_response["name"], function_response["id"]) == ("find_car", "fc-42")


def test_debug_line_names_both_tools_without_values(ctx: ToolContext) -> None:
    events: list[str] = []

    run_turn(ConversationState("c-1"), "אפשר לקבל קרניבל?", StrictProvider(calls(find("קרניבל", "alternative"))), ctx, events=events.append)

    assert any(e.endswith(": find_car→compare_alternative=missing_details") for e in events)
    assert not any(value in e for e in events for value in ("קרניבל", "CAR-19", "CARNIVAL"))


# --- 10. E-001 is unchanged ------------------------------------------------------------------------

E001 = (
    ("שלום, אני מנהלת מוצר ויש לי היום קורולה קרוס עם השתתפות של 300 ש״ח. כמה ישתנה לי הנטו אם אעבור לאטו 3?",
     {"occupation": "מנהלת מוצר"}),
    ("השכר שלי 18,500 ברוטו, אין לי עמלות ובלי בונוס.", {"base_salary": 18500, "commissions": 0, "bonus": 0}),
    ("יש לי 2.75 נקודות זיכוי, פנסיה 6% וקרן השתלמות 2.5%.", {"credit_points": 2.75, "pension_percent": 6, "training_fund_percent": 2.5}),
)


def test_e001_still_uses_three_provider_calls_with_the_same_reply(ctx: ToolContext) -> None:
    current = ("set_current_car", {"car": "קורולה קרוס", "participation": 300})
    provider = StrictProvider(*(calls(("save_employee_details", d), current, ("compare_alternative", {"car": "אטו 3"})) for _, d in E001))
    state = ConversationState("c-1")
    for text, _ in E001:
        result = run_turn(state, text, provider, ctx)
        assert result.stop == TurnStop.RENDERED
        state = result.state

    assert provider.calls == 3
    assert result.reply == "\n".join([
        "נטו היום (COROLLA CROSS): 11,032 ₪",
        "נטו בחלופה (ATTO 3): 11,693 ₪",
        "הפרש: +661 ₪ בחודש",
        "מה משתנה: שווי שימוש 4,070 → 3,130 ₪ · +250 ₪ נטו החזר חשמל (מגולם)",
        "שימו לב: ההשתתפות ברכב החלופי (300 ₪) היא ברירת מחדל מהקטלוג, כלומר הערכה ולא נתון מאושר.",
        "אפשר לבקש מנציג התחבורה את הסכום המדויק.",
    ])


# --- The plain-request check itself --------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, plain",
    [
        ("אפשר לקבל קרניבל?", True),
        ("אני רוצה טוסון", True),
        ("מה זה קרניבל?", False),
        ("ומה עם קרניבל?", False),  # a question word: the provider decides (a safe false negative)
        ("ספר לי על קרניבל", False),
        ("תסביר את הקרניבל", False),
    ],
)
def test_plain_request(text: str, plain: bool) -> None:
    assert plain_request(text) is plain


@pytest.mark.parametrize(
    "text, choice",
    [
        ("אני רוצה לעבור לקורולה", True),
        ("כמה ישתנה לי הנטו אם אעבור לקורולה?", True),  # "כמה" is read as itself, not כ + "מה"
        ("איזו קורולה אפשר לקבל?", True),
        ("ומה יוצא לי מקורולה?", True),
        ("מה זה קורולה?", False),
        ("ומה זו הקורולה?", False),
        ("מהי קורולה?", False),
        ("למה אי אפשר קורולה?", False),
        ("איך מקבלים קורולה?", False),
        ("ספר לי על קורולה", False),
        ("אפשר פרטים על קורולה", False),
    ],
)
def test_choice_request(text: str, choice: bool) -> None:
    assert choice_request(text) is choice
