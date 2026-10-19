"""Deterministic replies after the first tool round (app.agent.render): offline, scripted providers only."""

import itertools

import pytest

from app.agent import agent as agent_module
from app.agent.agent import TurnStop, run_turn
from app.agent.llm import Message, Role, ToolCall
from app.agent.output_check import discloses_estimate, unsupported_numbers
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.providers.gemini import to_gemini_request
from app.agent.render import BLOCKED_TEXT, ReplyKind, Rendered, electric_benefit, may_ask_more, missing_text
from app.agent.state import ComparisonStatus, ConversationState
from app.agent.tool_registry import SPECS
from app.agent.tools import ToolContext

_ids = itertools.count()


class StrictProvider:
    """Returns the scripted steps in order; asking for more than scripted fails the test."""

    name = "strict"

    def __init__(self, *steps: Message) -> None:
        self._steps, self.calls = list(steps), 0

    def complete(self, system, messages, tools) -> Message:
        self.calls += 1
        assert self._steps, "the agent asked the provider for more replies than scripted"
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"d-{next(_ids)}", n, a) for n, a in name_and_arguments])


def results(state: ConversationState) -> list[dict]:
    return [dict(r.content) for m in state.messages if m.role == Role.TOOL for r in m.tool_results]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


# The real E2E scenario (workbook E-001), as the employee typed it and as Gemini called the tools.
E001_MESSAGES = (
    "שלום, אני מנהלת מוצר ויש לי היום קורולה קרוס עם השתתפות של 300 ש״ח. כמה ישתנה לי הנטו אם אעבור לאטו 3?",
    "השכר שלי 18,500 ברוטו, אין לי עמלות ובלי בונוס.",
    "יש לי 2.75 נקודות זיכוי, פנסיה 6% וקרן השתלמות 2.5%.",
)
CURRENT = ("set_current_car", {"car": "קורולה קרוס", "participation": 300})
ATTO = ("compare_alternative", {"car": "אטו 3"})
E001_ROUNDS = (
    calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), CURRENT, ATTO),
    calls(("save_employee_details", {"base_salary": 18500, "commissions": 0, "bonus": 0}), CURRENT, ATTO),
    calls(("save_employee_details", {"credit_points": 2.75, "pension_percent": 6, "training_fund_percent": 2.5}), CURRENT, ATTO),
)


@pytest.fixture(scope="module")
def e001_turns(ctx: ToolContext):
    provider = StrictProvider(*E001_ROUNDS)  # one step per turn: a second call per turn would fail
    state, turns = ConversationState("c-1"), []
    for text in E001_MESSAGES:
        turn = run_turn(state, text, provider, ctx)
        turns.append(turn)
        state = turn.state
    return turns, provider


@pytest.fixture(scope="module")
def ready(e001_turns) -> ConversationState:
    return e001_turns[0][-1].state


def one_turn(ctx: ToolContext, state: ConversationState, text: str, *steps: Message):
    provider = StrictProvider(*steps)
    return run_turn(state, text, provider, ctx), provider


# --- E-001: 3 turns, 3 provider calls --------------------------------------------------------

def test_e001_uses_exactly_three_provider_calls(e001_turns) -> None:
    turns, provider = e001_turns

    assert provider.calls == 3  # was 6: one verbalisation call per turn is gone
    assert [t.stop for t in turns] == [TurnStop.RENDERED] * 3
    assert [t.tool_rounds for t in turns] == [1, 1, 1]
    assert [len(t.timings) for t in turns] == [1, 1, 1]


def test_e001_turn_1_asks_only_for_what_is_missing(e001_turns) -> None:
    text = e001_turns[0][0].reply

    for wanted in ("שכר הבסיס", "עמלות ובונוס", "לפנסיה ולקרן השתלמות", "נקודות הזיכוי"):
        assert wanted in text
    for not_wanted in ("ילד", "התפקיד", "השתתפות עצמית", "רכב חברה"):  # given already, or not relevant
        assert not_wanted not in text


def test_e001_turn_2_asks_for_the_rest(e001_turns) -> None:
    text = e001_turns[0][1].reply

    assert "לפנסיה ולקרן השתלמות" in text and "נקודות הזיכוי" in text
    assert "שכר הבסיס" not in text and "עמלות" not in text and "ילד" not in text


def test_e001_comparison_is_rendered_exactly(ctx: ToolContext, e001_turns) -> None:
    turn = e001_turns[0][2]

    assert turn.reply == "\n".join([
        "נטו היום (COROLLA CROSS): 11,032 ₪",
        "נטו בחלופה (ATTO 3): 11,693 ₪",
        "הפרש: +661 ₪ בחודש",
        "מה משתנה: שווי שימוש 4,070 → 3,130 ₪ · +250 ₪ נטו החזר חשמל (מגולם)",
        "שימו לב: ההשתתפות ברכב החלופי (300 ₪) היא ברירת מחדל מהקטלוג, כלומר הערכה ולא נתון מאושר.",
        "אפשר לבקש מנציג התחבורה את הסכום המדויק.",
    ])
    compared = results(turn.state)[-1]
    assert (compared["current"]["net"], compared["alternative"]["net"], compared["net_difference"]) == (11032, 11693, 661)
    assert turn.state.last_comparison.status == ComparisonStatus.ESTIMATE
    assert unsupported_numbers(turn.reply, turn.state) == ()
    assert discloses_estimate(turn.reply)


def test_electric_wording_comes_from_the_rules(ctx: ToolContext, e001_turns) -> None:
    turn = e001_turns[0][2]
    phrase = electric_benefit(ctx.rules)

    assert phrase == f"+{ctx.rules.electric.net_bonus} ₪ נטו החזר חשמל (מגולם)"
    assert phrase in turn.reply
    assert results(turn.state)[-1]["alternative"]["electric_benefit"] == phrase  # the same text for the LLM path


def test_rendered_reply_is_appended_and_history_stays_valid(e001_turns) -> None:
    turns, _ = e001_turns
    state = turns[-1].state

    assert [m.role for m in state.messages] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT] * 3
    assert [state.messages[i].text for i in (3, 7, 11)] == [t.reply for t in turns]
    contents, _ = to_gemini_request(SYSTEM_PROMPT, state.messages, SPECS)  # next turn's request builds
    assert [c["role"] for c in contents][-1] == "model"


def test_rendered_turn_does_not_change_its_input_state(ctx: ToolContext, ready: ConversationState) -> None:
    before = ready
    one_turn(ctx, ready, "ומה עם טוסון?", calls(("compare_alternative", {"car": "טוסון"})))

    assert ready == before


# --- A. Comparisons -------------------------------------------------------------------------

def test_confirmed_participation_has_no_estimate_notice(ctx: ToolContext, ready: ConversationState) -> None:
    turn, provider = one_turn(
        ctx, ready, "ומה אם ההשתתפות באטו 3 היא 450?", calls(("compare_alternative", {"car": "אטו 3", "participation": 450}))
    )

    assert (provider.calls, turn.stop) == (1, TurnStop.RENDERED)
    assert turn.state.last_comparison.status == ComparisonStatus.CONFIRMED
    assert "השתתפות 300 → 450 ₪" in turn.reply
    assert not discloses_estimate(turn.reply) and "נציג" not in turn.reply
    compared = results(turn.state)[-1]
    assert f"{compared['alternative']['net']:,} ₪" in turn.reply
    assert unsupported_numbers(turn.reply, turn.state) == ()


def test_non_electric_car_has_no_electric_benefit(ctx: ToolContext, ready: ConversationState) -> None:
    turn, _ = one_turn(ctx, ready, "ומה עם טוסון?", calls(("compare_alternative", {"car": "טוסון"})))

    assert turn.stop == TurnStop.RENDERED
    assert "נטו בחלופה (TUCSON)" in turn.reply and "שווי שימוש 4,070 → " in turn.reply
    assert "החזר חשמל" not in turn.reply
    assert "electric_benefit" not in results(turn.state)[-1]["alternative"]
    assert discloses_estimate(turn.reply)  # catalogue participation: still an estimate


def test_waiver_comparison(ctx: ToolContext, ready: ConversationState) -> None:
    turn, _ = one_turn(ctx, ready, "ואם אוותר על הרכב?", calls(("compare_alternative", {"waiver": True})))

    lines = turn.reply.splitlines()
    assert turn.stop == TurnStop.RENDERED
    assert lines[1].startswith("נטו בחלופה (ויתור על הרכב): ")
    assert "שווי שימוש 4,070 → 0 ₪" in turn.reply and "אחזקת רכב 0 → " in turn.reply and "השתתפות 300 → 0 ₪" in turn.reply
    assert not discloses_estimate(turn.reply)  # a waiver has no participation to estimate
    assert unsupported_numbers(turn.reply, turn.state) == ()


def test_family_track_asks_children_then_shows_the_share(ctx: ToolContext, ready: ConversationState) -> None:
    asked, _ = one_turn(ctx, ready, "ומה עם קרניבל?", calls(("compare_alternative", {"car": "קרניבל"})))

    assert asked.stop == TurnStop.RENDERED
    assert asked.reply == "\n".join([  # children decide eligibility, so they are asked first, alone
        "כדי לבדוק אם החלופה הזו אפשרית עבורך, צריך קודם לדעת:",
        f"• מספר הילדים שלך עד גיל {ctx.rules.family_track.child_max_age} (נדרש לרכב מהמסלול המשפחתי)",
    ])
    assert unsupported_numbers(asked.reply, asked.state) == ()

    shown, _ = one_turn(
        ctx, asked.state, "יש לי 5 ילדים",
        calls(("save_employee_details", {"eligible_children": 5}), ("compare_alternative", {"car": "קרניבל"})),
    )
    assert shown.stop == TurnStop.RENDERED
    assert "חלק במסלול המשפחתי 0 → " in shown.reply and "החזר חשמל" not in shown.reply
    assert unsupported_numbers(shown.reply, shown.state) == ()


def test_open_participation_question_is_mentioned(ctx: ToolContext, ready: ConversationState) -> None:
    ask = ("ask_representative", {"topic": "alternative_participation", "question": "ההשתתפות ב-ATTO 3?", "car_id": "CAR-10"})
    sent, _ = one_turn(ctx, ready, "תשאל את הנציג מה ההשתתפות באטו 3", calls(ask))
    again, _ = one_turn(ctx, sent.state, "תשאל שוב את הנציג", calls(ask))
    compared, _ = one_turn(ctx, again.state, "ומה עם אטו 3?", calls(ATTO))

    assert (sent.stop, again.stop, compared.stop) == (TurnStop.RENDERED,) * 3
    assert sent.reply.startswith("שלחתי לנציג התחבורה שאלה על ההשתתפות")
    assert again.reply.startswith("כבר נשלחה לנציג התחבורה שאלה")
    assert compared.reply.endswith("כבר נשלחה שאלה לנציג התחבורה על הסכום המדויק.")


# --- B. Missing details -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "missing, lines",
    [
        (["commissions"], ["עמלות חודשיות (אם אין, אפשר לכתוב שאין)"]),
        (["bonus", "commissions"], ["עמלות ובונוס חודשיים (אם אין, אפשר לכתוב שאין)"]),
        (["training_fund_rate"], ["אחוז ההפרשה שלך לקרן השתלמות (אם אין, אפשר לכתוב שאין)"]),
        (["pension_rate", "credit_points"], ["אחוז ההפרשה שלך לפנסיה", "מספר נקודות הזיכוי שלך"]),
        (["credit_points", "base_salary", "occupation"], ["התפקיד שלך", "שכר הבסיס החודשי (ברוטו)", "מספר נקודות הזיכוי שלך"]),
        (["current.participation"], ["כמה השתתפות עצמית את/ה משלם/ת היום על הרכב, בחודש"]),
    ],
)
def test_missing_fields_are_asked_in_a_fixed_order(ctx: ToolContext, missing, lines) -> None:
    text = missing_text(missing, ctx.rules)

    assert text.splitlines()[1:] == [f"• {line}" for line in lines]
    assert "ילד" not in text


@pytest.mark.parametrize("missing", [[], ["salary_net"], ["base_salary", "something_new"]])
def test_unknown_or_no_missing_fields_are_not_rendered(ctx: ToolContext, missing) -> None:
    assert missing_text(missing, ctx.rules) is None


def test_named_car_without_participation_asks_only_for_participation(ctx: ToolContext) -> None:
    turn, provider = one_turn(
        ctx, ConversationState("c-1"), "אני מנהלת מוצר ויש לי קורולה קרוס. כמה ישתנה עם אטו 3?",
        calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), ("set_current_car", {"car": "קורולה קרוס"}), ATTO),
    )

    assert (provider.calls, turn.stop) == (1, TurnStop.RENDERED)
    assert "כמה השתתפות עצמית" in turn.reply and "רכב חברה" not in turn.reply


# --- C. Representative and car choice -----------------------------------------------------------

def test_blocked_comparison(ctx: ToolContext) -> None:
    unknown = "אני מנהלת מוצר, יש לי מאזדה CX-5 2019 בהשתתפות 300"
    asked, _ = one_turn(  # car_not_found stays on the provider path
        ctx, ConversationState("c-1"), unknown,
        calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), ("set_current_car", {"car": "מאזדה CX-5 2019", "participation": 300})),
        calls(("ask_representative", {"topic": "current_car_not_in_catalogue", "question": "מאזדה CX-5 2019"})),
        Message.assistant("הרכב לא בקטלוג; שאלתי את הנציג."),
    )
    assert asked.stop == TurnStop.REPLIED

    turn, provider = one_turn(ctx, asked.state, "ומה עם אטו 3?", calls(ATTO))

    assert results(turn.state)[-1]["error"]["code"] == "blocked_by_representative"
    assert (provider.calls, turn.stop, turn.reply) == (1, TurnStop.RENDERED, BLOCKED_TEXT)


def test_ambiguous_car_lists_the_candidates(ctx: ToolContext) -> None:
    turn, provider = one_turn(
        ctx, ConversationState("c-1"), "יש לי קורולה, השתתפות 250", calls(("set_current_car", {"car": "קורולה", "participation": 250}))
    )

    assert (provider.calls, turn.stop) == (1, TurnStop.RENDERED)
    lines = turn.reply.splitlines()
    assert "הרכב הנוכחי" in lines[0]
    assert lines[1:] == ["• טויוטה טורקיה COROLLA (2026)", "• טויוטה יפן COROLLA CROSS (2025)"]
    assert turn.state.details.current is None and {"CAR-07", "CAR-09"} <= turn.state.offered_car_ids
    assert unsupported_numbers(turn.reply, turn.state) == ()


# --- Everything else stays with the provider -----------------------------------------------------

FALLBACKS = {
    "two question marks": ("ready", "כמה ישתנה עם אטו 3? ומה זה שווי שימוש?", [ATTO]),
    "two question words": ("ready", "כמה ישתנה עם אטו 3 ואיך מחושב שווי שימוש", [ATTO]),
    "unknown occupation": ("new", "אני מנכ\"ל", [("save_employee_details", {"occupation": "מנכ\"ל"})]),
    "car not found": ("new", "יש לי מאזדה CX-5 2019 בהשתתפות 300", [("set_current_car", {"car": "מאזדה CX-5 2019", "participation": 300})]),
    "not stated by the employee": ("ready", "ומה עם אטו 3?", [("compare_alternative", {"car": "אטו 3", "participation": 450})]),
    "find_car only": ("ready", "ומה עם אטו 3?", [("find_car", {"query": "אטו 3", "purpose": "information"})]),
    "list_alternatives": ("ready", "מה האפשרויות שלי?", [("list_alternatives", {})]),
    "two comparisons": ("ready", "אטו 3 או טוסון?", [ATTO, ("compare_alternative", {"car": "טוסון"})]),
    "state changed after the comparison": ("ready", "ומה עם אטו 3?", [ATTO, ("save_employee_details", {"base_salary": 18500})]),
    "details saved, nothing compared": ("new", "אני מנהלת מוצר", [("save_employee_details", {"occupation": "מנהלת מוצר"})]),
}


@pytest.mark.parametrize("start, text, round_1", FALLBACKS.values(), ids=FALLBACKS.keys())
def test_other_rounds_go_to_the_provider(ctx: ToolContext, ready: ConversationState, start, text, round_1) -> None:
    state = ready if start == "ready" else ConversationState("c-1")

    turn, provider = one_turn(ctx, state, text, calls(*round_1), Message.assistant("תשובת המודל"))

    assert (provider.calls, turn.stop) == (2, TurnStop.REPLIED)
    assert turn.reply.startswith("תשובת המודל")  # plus the usual estimate notice when an estimate was calculated


def test_a_comparison_in_a_later_round_keeps_the_provider_reply(ctx: ToolContext, ready: ConversationState) -> None:
    turn, provider = one_turn(
        ctx, ready, "ומה עם אטו 3?", calls(("find_car", {"query": "אטו 3", "purpose": "information"})), calls(ATTO), Message.assistant("לפי הערכה: +661 ₪.")
    )

    assert (provider.calls, turn.stop, turn.reply) == (3, TurnStop.REPLIED, "לפי הערכה: +661 ₪.")


def test_rendered_reply_failing_the_numbers_check_falls_back(ctx: ToolContext, ready: ConversationState, monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "deterministic_reply", lambda *a: Rendered(ReplyKind.COMPARISON, "הפרש: 999,999 ₪"))
    events: list[str] = []

    provider = StrictProvider(calls(ATTO), Message.assistant("לפי הערכה: +661 ₪."))
    turn = run_turn(ready, "ומה עם אטו 3?", provider, ctx, events=events.append)

    assert (provider.calls, turn.stop, turn.reply) == (2, TurnStop.REPLIED, "לפי הערכה: +661 ₪.")
    assert "round 1: deterministic reply not used (comparison); asking the provider" in events


@pytest.mark.parametrize(
    "text, more",
    [
        (E001_MESSAGES[0], False),
        (E001_MESSAGES[1], False),
        ("ומה עם קרניבל?", False),
        ("כמה ישתנה? ולמה?", True),
        ("כמה ישתנה ומה זה שווי שימוש", True),
    ],
)
def test_question_counting(text: str, more: bool) -> None:
    assert may_ask_more(text) is more


# --- Debug events --------------------------------------------------------------------------------

def test_rendered_turn_event_has_no_values(ctx: ToolContext, ready: ConversationState) -> None:
    events: list[str] = []

    run_turn(ready, "ומה עם אטו 3?", StrictProvider(calls(ATTO)), ctx, events=events.append)

    assert events[-1] == "round 1: deterministic reply (comparison)"
    text = "\n".join(events)
    for value in ("11,032", "11032", "661", "ATTO", "CAR-10"):
        assert value not in text
