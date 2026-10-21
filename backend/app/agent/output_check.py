"""Checks on the advisor's final reply, in Python: numbers must come from tools, estimates must be disclosed.

No LLM and no fuzzy matching: numbers are read with fixed rules (app.agent.grounding) and compared
exactly; disclosure is recognized by Hebrew word stems.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from app.agent.grounding import displayed_numbers, employee_texts
from app.agent.llm import Role
from app.agent.state import ConversationState, QuestionStatus
from app.car_search import normalize, prefix_stripped

# A list marker at the start of a line ("1." / "2)") is layout, not a figure.
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")

# Word stems that disclose an estimate: מוערך/מוערכת/מוערכים, הערכה/הערכת/להערכתי, משוער/משוערת,
# אומדן; "ברירת מחדל" is matched as a phrase. Compared after final-letter normalization.
_ESTIMATE_STEMS = ("מוערכ", "הערכ", "משוער", "אומדנ", "estimat")
_DEFAULT_PHRASES = (("ברירת", "מחדל"), ("ברירתמחדל",), ("default",))


def discloses_estimate(text: str) -> bool:
    tokens = normalize(text)
    for token in tokens:
        if any(form.startswith(_ESTIMATE_STEMS) for form in (token, *prefix_stripped(token))):
            return True
    forms = [{token, *prefix_stripped(token)} for token in tokens]
    for phrase in _DEFAULT_PHRASES:
        for i in range(len(forms) - len(phrase) + 1):
            if all(word in forms[i + k] for k, word in enumerate(phrase)):
                return True
    return False


def unsupported_numbers(reply: str, state: ConversationState) -> tuple[float, ...]:
    """Numbers in reply that appear neither in a tool result, nor in the employee's own messages, nor in
    the transport representative's answers."""
    allowed = _numbers_in_tool_results(state) | _numbers_in_answers(state) | {
        abs(n) for text in employee_texts(state) for n in displayed_numbers(text)
    }
    shown = displayed_numbers(_LIST_MARKER.sub(" ", reply))
    return tuple(dict.fromkeys(n for n in shown if not any(abs(abs(n) - a) < 1e-9 for a in allowed)))


def _numbers_in_tool_results(state: ConversationState) -> set[float]:
    found: set[float] = set()
    for message in state.messages:
        if message.role == Role.TOOL:
            for result in message.tool_results:
                found.update(_json_numbers(result.content))
    return found


def _numbers_in_answers(state: ConversationState) -> set[float]:
    found: set[float] = set()
    for question in state.rep_questions:
        if question.status == QuestionStatus.ANSWERED:
            found.update(abs(n) for n in displayed_numbers(question.answer or ""))
            if question.participation is not None:
                found.add(abs(float(question.participation)))
    return found


def _json_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield abs(float(value))
    elif isinstance(value, str):
        yield from (abs(n) for n in displayed_numbers(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _json_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _json_numbers(item)
