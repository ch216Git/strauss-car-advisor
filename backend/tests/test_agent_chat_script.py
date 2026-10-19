"""The manual chat script's loop and safe debug output, with a fake provider (no Gemini)."""

import re

from app.agent.llm import Message, ToolCall
from app.agent.tools import ToolContext
from scripts.agent_chat import INTERNAL_ERROR_REPLY, chat

MESSAGE = "אני מנהלת מוצר, 18,500 ברוטו בלי עמלות ובונוס. יש לי קורולה קרוס עם השתתפות 300."


class ScriptedProvider:
    name = "scripted"

    def __init__(self, *steps) -> None:
        self._steps = list(steps)

    def complete(self, system, messages, tools) -> Message:
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def run_chat(provider, inputs: list[str], debug: bool = False):
    lines = iter(inputs)
    written: list[str] = []
    debugged: list[str] = []

    def read(prompt: str) -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    state = chat(provider, ToolContext.load(), read, written.append, debugged.append if debug else None)
    return state, written, debugged


def details_then_reply() -> list[Message]:
    return [
        Message.assistant(tool_calls=[
            ToolCall("c1", "save_employee_details", {"occupation": "מנהל/ת מוצר", "base_salary": 18500, "bonus": 0}),
            ToolCall("c2", "find_car", {"query": "קורולה קרוס"}),
            ToolCall("c3", "set_current_car", {"car_id": "CAR-09", "participation": 999}),  # invented: refused
        ]),
        Message.assistant("כמה אחוז פנסיה?"),
    ]


def test_keeps_state_between_turns_and_prints_only_replies() -> None:
    provider = ScriptedProvider(*details_then_reply(), Message.assistant("תודה"))

    state, written, _ = run_chat(provider, [MESSAGE, "  ", "6%", "exit", "never read"])

    assert written == ["כמה אחוז פנסיה?", "תודה"]
    assert state.details.base_salary == 18500
    assert len([m for m in state.messages if m.text == "6%"]) == 1


def test_exit_words_and_end_of_input_stop_the_chat() -> None:
    for inputs in (["יציאה"], ["EXIT"], []):
        _, written, _ = run_chat(ScriptedProvider(), inputs)
        assert written == []


def test_debug_is_live_and_shows_no_values() -> None:
    _, _, debugged = run_chat(ScriptedProvider(*details_then_reply()), [MESSAGE], debug=True)

    patterns = [
        r"\[debug\] round 1: provider call started",
        r"\[debug\] round 1: provider returned in \d+\.\d\ds \(3 tool calls\)",
        r"\[debug\] round 1: tools \d+\.\d\ds: save_employee_details=ok, find_car=ok, set_current_car=not_stated_by_employee",
        r"\[debug\] round 2: provider call started",
        r"\[debug\] round 2: provider returned in \d+\.\d\ds \(final reply\)",
        r"\[debug\] turn: stop=replied tool_rounds=1 provider=\d+\.\d\ds tools=\d+\.\d\ds total=\d+\.\d\ds status=collecting",
    ]
    assert len(debugged) == len(patterns)
    for line, pattern in zip(debugged, patterns):
        assert re.fullmatch(pattern, line), line
    text = "\n".join(debugged)
    for secret in ("18500", "18,500", "999", "300", "מנהל", "קורולה", "CAR-09"):
        assert secret not in text


def test_debug_is_off_by_default() -> None:
    _, _, debugged = run_chat(ScriptedProvider(*details_then_reply()), [MESSAGE])

    assert debugged == []


def test_unexpected_error_shows_no_details() -> None:
    provider = ScriptedProvider(RuntimeError("salary 18500 leaked"), Message.assistant("שוב שלום"))

    state, written, debugged = run_chat(provider, ["שלום", "שלום"], debug=True)

    assert written == [INTERNAL_ERROR_REPLY, "שוב שלום"]
    assert debugged[:2] == ["[debug] round 1: provider call started", "[debug] internal error: RuntimeError"]
    assert all("18500" not in line for line in written + debugged)
    assert [m.text for m in state.messages] == ["שלום", "שוב שלום"]  # the failed turn left no trace


def test_provider_retry_events_are_prefixed_with_the_round() -> None:
    class RetryingProvider(ScriptedProvider):
        def set_events(self, events) -> None:
            self.events = events

        def complete(self, system, messages, tools) -> Message:
            self.events("provider attempt 1 failed category=server_error status=503 retryable=yes")
            self.events("provider retrying in 1.0s")
            self.events("provider attempt 2 started")
            return super().complete(system, messages, tools)

    _, written, debugged = run_chat(RetryingProvider(Message.assistant("שלום")), ["שלום"], debug=True)

    assert written == ["שלום"]
    assert debugged[:4] == [
        "[debug] round 1: provider call started",
        "[debug] round 1: provider attempt 1 failed category=server_error status=503 retryable=yes",
        "[debug] round 1: provider retrying in 1.0s",
        "[debug] round 1: provider attempt 2 started",
    ]
