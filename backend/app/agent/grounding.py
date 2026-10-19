"""Provenance checks: employee facts the LLM saves must appear in the employee's own messages.

Deterministic, no LLM, no fuzzy matching. A value is grounded only when it can be read back from
what the employee typed (digits, "18,500", "18.5 אלף", Hebrew number words, the occupation's words,
waiver words). The LLM supplying a well-typed value is not enough.
"""

import re
from collections.abc import Iterable, Sequence

from app.agent.llm import Role
from app.agent.state import ConversationState
from app.car_search import normalize, prefix_stripped

_SEPARATED_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # 18,500 -> 18500
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)(?:\s*(אלף|אלפים|k)\b)?", re.IGNORECASE)  # 300, 2.75, 18.5 אלף, 18.5k

# Hebrew number words (after final-letter normalization) and their values.
_NUMBER_WORDS = {
    "אפס": 0,
    "אחד": 1, "אחת": 1,
    "שניימ": 2, "שתיימ": 2, "שני": 2, "שתי": 2,
    "שלושה": 3, "שלוש": 3, "שלושת": 3,
    "ארבעה": 4, "ארבע": 4, "ארבעת": 4,
    "חמישה": 5, "חמש": 5, "חמשת": 5,
    "שישה": 6, "שש": 6, "ששת": 6,
    "שבעה": 7, "שבע": 7, "שבעת": 7,
    "שמונה": 8, "שמונת": 8,
    "תשעה": 9, "תשע": 9, "תשעת": 9,
    "עשרה": 10, "עשר": 10, "עשרת": 10,
    "אלפ": 1000, "אלפיימ": 2000,
}
# Words that state an amount of zero ("אין עמלות", "בלי בונוס", "ללא קרן השתלמות").
_ZERO_WORDS = {"אינ", "בלי", "ללא", "לא", "אפס", "none", "no", "zero"}
# Word starts that say the car is waived / car maintenance is received instead.
_WAIVER_STARTS = ("ויתור", "ויתרתי", "ויתר", "וויתר", "מוותר", "אוותר", "אחזק")


def employee_texts(state: ConversationState) -> tuple[str, ...]:
    """Everything the employee has written in this conversation, oldest first."""
    return tuple(m.text for m in state.messages if m.role == Role.USER)


def numbers_in(texts: Iterable[str]) -> frozenset[float]:
    found: set[float] = set()
    for text in texts:
        plain = _SEPARATED_THOUSANDS.sub("", text)
        for digits, thousands in _NUMBER.findall(plain):
            value = float(digits)
            found.add(value)
            if thousands:
                found.add(value * 1000)
        for token in _tokens(text):
            if token in _NUMBER_WORDS:
                found.add(float(_NUMBER_WORDS[token]))
    return frozenset(found)


def displayed_numbers(text: str) -> list[float]:
    """Each number written with digits, as its reader takes it: "11,693" = 11693, "18.5 אלף" = 18500."""
    return [
        float(digits) * (1000 if thousands else 1)
        for digits, thousands in _NUMBER.findall(_SEPARATED_THOUSANDS.sub("", text))
    ]


def number_grounded(value: float, texts: Sequence[str]) -> bool:
    """value was written by the employee; 0 also by a "none" word (אין / בלי / ללא / לא)."""
    if any(abs(value - n) < 1e-9 for n in numbers_in(texts)):
        return True
    return value == 0 and any(token in _ZERO_WORDS for text in texts for token in _tokens(text))


def occupation_grounded(occupation: str, texts: Sequence[str]) -> bool:
    """Every word of the occupation name, in either gender form ("מנהל/ת" = מנהל or מנהלת), was written."""
    written = {token for text in texts for token in _tokens(text)}
    return all(any(form in written for form in _forms(word)) for word in occupation.split())


def waiver_grounded(texts: Sequence[str]) -> bool:
    return any(token.startswith(_WAIVER_STARTS) for text in texts for token in _tokens(text))


def _forms(word: str) -> tuple[str, ...]:
    base, _, suffix = word.partition("/")
    return tuple(dict.fromkeys(token for form in (base, base + suffix) for token in normalize(form)))


def _tokens(text: str) -> set[str]:
    """Normalized words of text, each also without its Hebrew prefix letters (ה, ב, ל, מ, ו, ש, כ)."""
    tokens: set[str] = set()
    for token in normalize(text):
        tokens.add(token)
        tokens.update(prefix_stripped(token))
    return tokens
