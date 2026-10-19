"""The agent loop with scripted fake providers only: no Gemini, no network, no quota."""

from collections.abc import Callable, Sequence

import pytest

from app.agent.agent import (
    ESTIMATE_NOTICE,
    MAX_ROUNDS_REPLY,
    MAX_TOOL_ROUNDS,
    PROVIDER_ERROR_REPLY,
    TurnStop,
    run_turn,
)
from app.agent.llm import Message, ProviderError, ProviderErrorKind, Role, ToolCall, ToolSpec
from app.agent.prompt import SYSTEM_PROMPT
from app.agent.state import ComparisonStatus, ConversationState, ConversationStatus, RepTopic, conversation_status
from app.agent.tool_registry import SPECS
from app.agent.tools import ToolContext
from app.comparison import CompanyCar

Step = Message | Exception | Callable[[Sequence[Message]], Message]


class ScriptedProvider:
    """Returns the scripted steps in order and records every request."""

    name = "scripted"

    def __init__(self, *steps: Step) -> None:
        self._steps = list(steps)
        self.requests: list[tuple[str, tuple[Message, ...], tuple[ToolSpec, ...]]] = []

    def complete(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Message:
        self.requests.append((system, tuple(messages), tuple(tools)))
        assert self._steps, "the agent asked the provider for more replies than scripted"
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step


class EndlessToolCaller:
    name = "endless"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, messages, tools) -> Message:
        self.calls += 1
        return calls(("find_car", {"query": "קורולה קרוס"}))


_ids = iter(range(10_000))


def calls(*name_and_arguments: tuple[str, dict]) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"call-{next(_ids)}", name, args) for name, args in name_and_arguments])


def reply(text: str) -> Message:
    return Message.assistant(text)


def results(state: ConversationState) -> list[dict]:
    """Contents of every tool result in the transcript, oldest first."""
    return [dict(r.content) for m in state.messages if m.role == Role.TOOL for r in m.tool_results]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


# E-001 as the employee would state it (workbook 'עובדים לדוגמה').
E001_DETAILS = {
    "occupation": "מנהל/ת מוצר", "base_salary": 18500, "commissions": 0, "bonus": 0,
    "pension_percent": 6, "training_fund_percent": 2.5, "credit_points": 2.75,
}
E001_MESSAGE = "אני מנהלת מוצר, 18,500 ברוטו בלי עמלות ובונוס, פנסיה 6%, השתלמות 2.5%, 2.75 נקודות. יש לי קורולה קרוס עם השתתפות 300. כמה ישתנה הנטו עם אטו 3?"


def e001_ready(ctx: ToolContext) -> ConversationState:
    """E-001 with details and current car saved through the loop."""
    provider = ScriptedProvider(
        calls(("save_employee_details", E001_DETAILS), ("find_car", {"query": "קורולה קרוס"})),
        calls(("set_current_car", {"car_id": "CAR-09", "participation": 300})),
        reply("מעולה, איזו חלופה לבדוק?"),
    )
    return run_turn(ConversationState("c-1"), E001_MESSAGE, provider, ctx).state


# --- Normal flow ------------------------------------------------------------------------

def test_full_turn_with_workbook_numbers(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(  # several tool calls in one round
            ("save_employee_details", E001_DETAILS),
            ("find_car", {"query": "קורולה קרוס"}),
            ("find_car", {"query": "אטו 3"}),
        ),
        calls(("set_current_car", {"car_id": "CAR-09", "participation": 300})),
        calls(("compare_alternative", {"car_id": "CAR-10"})),
        reply("היום 11,032 ₪ נטו, עם ATTO 3 כ-11,693 ₪ (הערכה, השתתפות לפי ברירת מחדל): +661 ₪."),
    )

    turn = run_turn(ConversationState("c-1"), E001_MESSAGE, provider, ctx)

    assert (turn.stop, turn.tool_rounds) == (TurnStop.REPLIED, 3)
    assert turn.reply.startswith("היום 11,032")
    comparison = results(turn.state)[-1]
    assert comparison["ok"] and (comparison["current"]["net"], comparison["alternative"]["net"]) == (11032, 11693)
    assert comparison["net_difference"] == 661
    assert turn.state.last_comparison.status == ComparisonStatus.ESTIMATE
    roles = [m.role for m in turn.state.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]


def test_provider_gets_prompt_tools_and_history(ctx: ToolContext) -> None:
    provider = ScriptedProvider(calls(("find_car", {"query": "טוסון"})), reply("מצאתי"))

    run_turn(ConversationState("c-1"), "יש לי טוסון", provider, ctx)

    system, messages, tools = provider.requests[1]
    assert system == SYSTEM_PROMPT
    assert tools == SPECS
    assert [t.name for t in tools] == [
        "save_employee_details", "find_car", "set_current_car", "list_alternatives", "compare_alternative", "ask_representative"
    ]
    assert [m.role for m in messages] == [Role.USER, Role.ASSISTANT, Role.TOOL]


def test_results_answer_each_call_in_order(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(("find_car", {"query": "טוסון"}), ("find_car", {"query": "פרארי"})), reply("ok")
    )

    turn = run_turn(ConversationState("c-1"), "x", provider, ctx)

    assistant, tool = turn.state.messages[1], turn.state.messages[2]
    assert [r.call_id for r in tool.tool_results] == [c.call_id for c in assistant.tool_calls]
    assert [r.content["status"] for r in tool.tool_results] == ["identified", "not_found"]


def test_input_state_is_not_changed(ctx: ToolContext) -> None:
    start = ConversationState("c-1")

    run_turn(start, "x", ScriptedProvider(calls(("save_employee_details", {"base_salary": 1})), reply("ok")), ctx)

    assert start == ConversationState("c-1")


# --- Tool validation ---------------------------------------------------------------------

def test_unknown_tool_is_reported_not_run(ctx: ToolContext) -> None:
    provider = ScriptedProvider(calls(("calculate_net", {"salary": 18500})), reply("סליחה"))

    turn = run_turn(ConversationState("c-1"), "x", provider, ctx)

    (result,) = results(turn.state)
    assert result == {"ok": False, "error": {"code": "unknown_tool", "detail": result["error"]["detail"]}}
    assert turn.stop == TurnStop.REPLIED


@pytest.mark.parametrize(
    "name, arguments, fragment",
    [
        ("save_employee_details", {"base_salary": "18500"}, "base_salary must be a number"),
        ("save_employee_details", {"pension_percent": True}, "must be a number"),
        ("save_employee_details", {"eligible_children": 2.5}, "must be a integer"),
        ("save_employee_details", {"base_salary": -1}, "must be >= 0"),
        ("save_employee_details", {"net": 12000}, "unknown arguments ['net']"),
        ("find_car", {}, "missing required arguments ['query']"),
        ("ask_representative", {"topic": "salary", "question": "?"}, "topic must be one of"),
        ("compare_alternative", {"waiver": True, "base_salary": 99999}, "unknown arguments ['base_salary']"),
        ("set_current_car", {"waiver": "yes"}, "waiver must be a boolean"),
    ],
)
def test_invalid_arguments_are_refused_before_running(ctx: ToolContext, name, arguments, fragment) -> None:
    turn = run_turn(ConversationState("c-1"), "x", ScriptedProvider(calls((name, arguments)), reply("ok")), ctx)

    (result,) = results(turn.state)
    assert result["error"]["code"] == "invalid_arguments"
    assert fragment in result["error"]["detail"]
    assert turn.state.details == ConversationState("c-1").details


def test_whole_number_float_is_accepted_for_integers(ctx: ToolContext) -> None:
    provider = ScriptedProvider(calls(("save_employee_details", {"eligible_children": 5.0})), reply("ok"))

    turn = run_turn(ConversationState("c-1"), "יש לי 5 ילדים", provider, ctx)

    assert turn.state.details.eligible_children == 5


def test_null_arguments_mean_not_given(ctx: ToolContext) -> None:
    provider = ScriptedProvider(calls(("save_employee_details", {"base_salary": 18500, "bonus": None})), reply("ok"))

    turn = run_turn(ConversationState("c-1"), "אני מרוויחה 18,500", provider, ctx)

    assert (turn.state.details.base_salary, turn.state.details.bonus) == (18500, None)


# --- Round limit and provider errors ------------------------------------------------------

def test_max_rounds_stops_the_loop(ctx: ToolContext) -> None:
    provider = EndlessToolCaller()

    turn = run_turn(ConversationState("c-1"), "x", provider, ctx)

    assert (turn.stop, turn.reply, turn.tool_rounds) == (TurnStop.MAX_ROUNDS, MAX_ROUNDS_REPLY, MAX_TOOL_ROUNDS)
    assert provider.calls == MAX_TOOL_ROUNDS + 1
    # Every recorded tool call has its results; the unexecuted last request is not recorded.
    with_calls = [m for m in turn.state.messages if m.tool_calls]
    tool_messages = [m for m in turn.state.messages if m.role == Role.TOOL]
    assert len(with_calls) == len(tool_messages) == MAX_TOOL_ROUNDS
    assert turn.state.messages[-1] == Message.assistant(MAX_ROUNDS_REPLY)


def test_provider_error_gives_a_fixed_reply(ctx: ToolContext) -> None:
    provider = ScriptedProvider(ProviderError("scripted", ProviderErrorKind.RATE_LIMITED, "slow down"))

    turn = run_turn(ConversationState("c-1"), "שלום", provider, ctx)

    assert (turn.stop, turn.reply) == (TurnStop.PROVIDER_ERROR, PROVIDER_ERROR_REPLY)
    assert [m.role for m in turn.state.messages] == [Role.USER, Role.ASSISTANT]


# --- Car selection: never auto-selected ------------------------------------------------------

def test_ambiguous_car_needs_the_employee_to_choose(ctx: ToolContext) -> None:
    first = ScriptedProvider(
        calls(("find_car", {"query": "קורולה"})),
        calls(("set_current_car", {"car_id": "CAR-07", "participation": 250})),  # picks a candidate itself
        reply("יש שתי קורולות, איזו?"),
    )

    turn = run_turn(ConversationState("c-1"), "יש לי קורולה, השתתפות 250", first, ctx)

    found, refused = results(turn.state)
    assert found["status"] == "candidates" and len(found["cars"]) == 2
    assert refused["error"]["code"] == "car_not_confirmed"
    assert turn.state.details.current is None

    # After the employee answers, the shown candidate can be chosen.
    second = ScriptedProvider(calls(("set_current_car", {"car_id": "CAR-07", "participation": 250})), reply("נשמר"))
    after = run_turn(turn.state, "הקורולה הרגילה", second, ctx)

    assert after.state.details.current == CompanyCar("CAR-07", 250)


def test_listed_alternatives_are_choosable_only_after_the_employee_replies(ctx: ToolContext) -> None:
    state = e001_ready(ctx)
    first = ScriptedProvider(
        calls(("list_alternatives", {})),
        calls(("compare_alternative", {"car_id": "CAR-10"})),  # chooses for the employee
        reply("אלה החלופות"),
    )
    turn = run_turn(state, "מה אפשר?", first, ctx)
    assert results(turn.state)[-1]["error"]["code"] == "car_not_confirmed"

    second = ScriptedProvider(calls(("compare_alternative", {"car_id": "CAR-10"})), reply("הנה, זו הערכה"))
    after = run_turn(turn.state, "את האטו 3", second, ctx)

    assert results(after.state)[-1]["ok"]


def test_identified_car_is_usable_at_once(ctx: ToolContext) -> None:
    state = e001_ready(ctx)

    assert "CAR-09" in state.usable_car_ids
    assert state.details.current == CompanyCar("CAR-09", 300)


def test_unknown_current_car_is_not_guessed_and_goes_to_the_representative(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(("save_employee_details", E001_DETAILS), ("find_car", {"query": "מאזדה CX-5 2019"})),
        calls(("set_current_car", {"car_id": "CAR-13", "participation": 300})),  # a guess: Mazda 3
        calls(("ask_representative", {"topic": "current_car_not_in_catalogue", "question": "מאזדה CX-5 2019, השתתפות 300"})),
        calls(("compare_alternative", {"waiver": True})),
        reply("הרכב שלך לא בקטלוג, שאלתי את הנציג."),
    )

    turn = run_turn(ConversationState("c-1"), E001_MESSAGE.replace("קורולה קרוס", "מאזדה CX-5 2019"), provider, ctx)

    _, not_found, guessed, asked, blocked = results(turn.state)
    assert not_found["status"] == "not_found"
    assert guessed["error"]["code"] == "car_not_confirmed"
    assert asked["ok"] and asked["question"]["topic"] == RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE.value
    assert blocked["error"]["code"] == "blocked_by_representative"
    assert turn.state.details.current is None
    assert conversation_status(turn.state, ctx.rules) == ConversationStatus.WAITING_FOR_REPRESENTATIVE


def test_participation_question_needs_a_confirmed_car(ctx: ToolContext) -> None:
    state = e001_ready(ctx)
    provider = ScriptedProvider(
        calls(("ask_representative", {"topic": "alternative_participation", "question": "?", "car_id": "CAR-14"})),
        calls(("find_car", {"query": "אטו 3"})),
        calls(("ask_representative", {"topic": "alternative_participation", "question": "מה ההשתתפות?", "car_id": "car-10"})),
        reply("שאלתי את הנציג"),
    )

    turn = run_turn(state, "תברר את ההשתתפות באטו 3", provider, ctx)

    refused, _, asked = results(turn.state)[-3:]
    assert refused["error"]["code"] == "car_not_confirmed"
    assert asked["ok"] and asked["question"]["car_id"] == "CAR-10"


# --- Estimate vs confirmed -----------------------------------------------------------------

def test_undisclosed_estimate_gets_the_notice(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(("find_car", {"query": "אטו 3"})),
        calls(("compare_alternative", {"car_id": "CAR-10"})),
        reply("הנטו יעלה ב-661 ₪."),  # does not say it is an estimate
    )

    turn = run_turn(e001_ready(ctx), "ואטו 3?", provider, ctx)

    assert results(turn.state)[-1]["status"] == "estimate"
    assert results(turn.state)[-1]["alternative"]["participation_is_estimate"] is True
    assert turn.reply == f"הנטו יעלה ב-661 ₪.\n\n{ESTIMATE_NOTICE}"
    assert turn.state.messages[-1].text == turn.reply


def test_disclosed_estimate_gets_no_duplicate_notice(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(("find_car", {"query": "אטו 3"})),
        calls(("compare_alternative", {"car_id": "CAR-10"})),
        reply("לפי הערכה: +661 ₪."),
    )

    assert run_turn(e001_ready(ctx), "ואטו 3?", provider, ctx).reply == "לפי הערכה: +661 ₪."


def test_confirmed_comparison_has_no_notice(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(("find_car", {"query": "אטו 3"})),
        calls(("compare_alternative", {"car_id": "CAR-10", "participation": 300})),
        reply("+661 ₪ בחודש."),
    )

    turn = run_turn(e001_ready(ctx), "ואטו 3 עם השתתפות 300?", provider, ctx)

    assert results(turn.state)[-1]["status"] == "confirmed"
    assert turn.reply == "+661 ₪ בחודש."


# --- Needed information only ------------------------------------------------------------------

def test_children_are_asked_only_for_a_family_track_car(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(("find_car", {"query": "אטו 3"}), ("find_car", {"query": "קרניבל"})),
        calls(("compare_alternative", {"car_id": "CAR-10"}), ("compare_alternative", {"car_id": "CAR-19"})),
        reply("כמה ילדים?"),
    )

    turn = run_turn(e001_ready(ctx), "אטו 3 או קרניבל?", provider, ctx)

    regular, family = results(turn.state)[-2:]
    assert regular["ok"]
    assert (family["error"]["code"], family["error"]["missing"]) == ("missing_details", ["eligible_children"])
    # The rule the question depends on comes from rules.json, so the model can quote it without inventing it.
    assert family["error"]["family_track_rule"] == {
        "min_children": ctx.rules.family_track.min_children, "child_max_age": ctx.rules.family_track.child_max_age
    }


def test_missing_details_are_reported_to_the_model(ctx: ToolContext) -> None:
    provider = ScriptedProvider(calls(("save_employee_details", {"base_salary": 18500})), reply("מה התפקיד?"))

    turn = run_turn(ConversationState("c-1"), "אני מרוויחה 18,500", provider, ctx)

    assert "occupation" in results(turn.state)[0]["missing"]


# --- The model cannot bypass business validation ------------------------------------------------

def test_ineligible_car_is_refused_by_the_engine(ctx: ToolContext) -> None:
    provider = ScriptedProvider(
        calls(("find_car", {"query": "טסלה מודל 3"})),
        calls(("compare_alternative", {"car_id": "CAR-14"})),  # group ד; E-001 is group ג
        reply("לא זכאית"),
    )

    turn = run_turn(e001_ready(ctx), "וטסלה?", provider, ctx)

    error = results(turn.state)[-1]["error"]
    assert (error["code"], error["reason"]) == ("not_eligible", "group_too_high")
    assert turn.state.last_comparison is None


def test_unknown_occupation_is_refused(ctx: ToolContext) -> None:
    provider = ScriptedProvider(calls(("save_employee_details", {"occupation": "מנכ\"ל"})), reply("?"))

    turn = run_turn(ConversationState("c-1"), "אני מנכ\"ל", provider, ctx)

    assert results(turn.state)[0]["error"]["code"] == "unknown_occupation"
    assert turn.state.details.occupation is None


def test_comparison_numbers_come_only_from_the_engine(ctx: ToolContext) -> None:
    # The model cannot pass salary or net figures to the comparison, and the result is pre-rounded.
    provider = ScriptedProvider(
        calls(("compare_alternative", {"waiver": True, "net_difference": 5000})),
        calls(("compare_alternative", {"waiver": True})),
        reply("לפי החישוב"),
    )

    turn = run_turn(e001_ready(ctx), "ואם אוותר?", provider, ctx)

    refused, computed = results(turn.state)[-2:]
    assert refused["error"]["code"] == "invalid_arguments"
    assert computed["status"] == "confirmed"
    assert all(isinstance(change[key], int) for change in computed["changes"] for key in ("current", "alternative", "difference"))
