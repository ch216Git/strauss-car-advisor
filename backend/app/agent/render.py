"""Deterministic employee replies for tool rounds whose outcome Python already knows (presentation only).

After the FIRST tool round of a turn, if every tool call matches one recognised pattern, the reply is
written here from the tools' own results instead of asking the provider to put them into words:
  A. one comparison was calculated;
  B. the comparison needs details the employee has not given yet;
  C1. the comparison is blocked until the representative answers about today's car;
  C2. a question was sent to the representative;
  C3. a car the employee named matches several catalogue cars;
  D. the alternative cannot be newly selected (a known not-eligible reason);
  E. details were only saved, and the tools report which are still missing.

Nothing is decided or calculated here: eligibility, missing fields, estimate status and every figure
come from the tools (Steps 1-3); this module only formats them. Any other outcome, or an employee
message that may ask more than one thing, gives None and the provider writes the reply as before.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.agent.state import (
    REQUIRED_DETAILS,
    ComparisonStatus,
    ConversationState,
    RepTopic,
    open_questions,
)
from app.agent.tools import ComparisonSummary, ToolError
from app.car_search import normalize, prefix_stripped
from app.comparison import SituationDetails, display_shekels
from app.rules import Rules


@dataclass(frozen=True, slots=True)
class RoundCall:
    """One executed tool call of a round: its name, JSON result, and the comparison if one was calculated."""

    name: str  # the tool that ran
    content: Mapping[str, Any]
    comparison: ComparisonSummary | None = None
    via: str | None = None  # the tool the model called, when it ran as another (find_car -> compare_alternative)
    error: ToolError | None = None  # the structured error behind content, when the tool itself refused


class ReplyKind(StrEnum):
    COMPARISON = "comparison"  # A
    MISSING = "missing"  # B
    BLOCKED = "blocked"  # C1
    QUESTION = "question"  # C2
    CANDIDATES = "candidates"  # C3
    NOT_ELIGIBLE = "not_eligible"  # D
    DETAILS_SAVED = "details_saved"  # E


@dataclass(frozen=True, slots=True)
class Rendered:
    kind: ReplyKind
    text: str


def electric_benefit(rules: Rules) -> str:
    """The official wording of the electric benefit (workbook: "+250 ₪ נטו החזר חשמל (מגולם)"), from rules.json."""
    return f"+{display_shekels(rules.electric.net_bonus):,} ₪ נטו החזר חשמל (מגולם)"


# --- Which rounds are rendered ------------------------------------------------------------

# (tool, outcome) pairs a rendered round may contain; anything else goes to the provider.
_RENDERABLE = {
    ("save_employee_details", "ok"),
    ("set_current_car", "ok"),
    ("set_current_car", "missing_details"),
    ("set_current_car", "car_needs_choice"),
    ("compare_alternative", "ok"),
    ("compare_alternative", "missing_details"),
    ("compare_alternative", "blocked_by_representative"),
    ("compare_alternative", "car_needs_choice"),
    ("compare_alternative", "not_eligible"),
    ("ask_representative", "ok"),
}


def deterministic_reply(
    user_text: str, calls: Sequence[RoundCall], state: ConversationState, rules: Rules
) -> Rendered | None:
    """The reply for a recognised round, or None to let the provider write it."""
    if not calls or may_ask_more(user_text):
        return None
    # A round the model reached by looking a car up (find_car -> compare_alternative): a comparison is
    # rendered only when nothing else is asked; a car choice also for one question about choosing that car
    # or its effect, since that question can only be answered once the car is known.
    choosing = any(_outcome(call) == "car_needs_choice" for call in calls)
    if any(call.via for call in calls) and not (choice_request if choosing else plain_request)(user_text):
        return None
    if any((call.name, _outcome(call)) not in _RENDERABLE for call in calls):
        return None

    choices = [call for call in calls if _outcome(call) == "car_needs_choice"]
    questions = [call for call in calls if call.name == "ask_representative"]
    compares = [call for call in calls if call.name == "compare_alternative"]

    if choices:  # C3: ask which car first; other missing details come after the choice
        others_ok = all(
            call is choices[0] or _is(call, "save_employee_details", "ok") or _is(call, "set_current_car", "ok")
            or _is(call, "compare_alternative", "missing_details")
            for call in calls
        )
        text = _candidates_text(choices[0]) if len(choices) == 1 and others_ok else None
        return Rendered(ReplyKind.CANDIDATES, text) if text else None

    if questions:  # C2
        if len(questions) != 1:
            return None
        question = questions[0]
        topic = question.content["question"]["topic"]
        others_ok = all(
            call is question or _is(call, "save_employee_details", "ok")
            or (topic == RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE and _is(call, "compare_alternative", "blocked_by_representative"))
            for call in calls
        )
        return Rendered(ReplyKind.QUESTION, _question_text(topic, question.content["already_open"])) if others_ok else None

    if not compares:  # E
        return _details_saved(user_text, calls, state, rules)

    # A, B, C1, D: exactly one comparison, and it is the round's last call (nothing changed the state after it).
    if len(compares) != 1 or calls[-1] is not compares[0]:
        return None
    compare, before = compares[0], calls[:-1]
    outcome = _outcome(compare)
    if outcome == "ok":
        if compare.comparison is None or not all(
            _is(call, "save_employee_details", "ok") or _is(call, "set_current_car", "ok") for call in before
        ):
            return None
        return Rendered(ReplyKind.COMPARISON, comparison_text(compare.comparison, state, rules))
    if outcome == "missing_details":
        if not all(call.name in ("save_employee_details", "set_current_car") for call in before):
            return None
        if compare.content["error"].get("stage") == "eligibility":  # only the facts that decide eligibility
            text = missing_text(_missing(compare), rules, eligibility=True)
            return Rendered(ReplyKind.MISSING, text) if text else None
        missing = [field for call in (*before, compare) if _outcome(call) == "missing_details" for field in _missing(call)]
        if "current.participation" in missing:  # the car was named; only its participation is missing
            missing = [field for field in missing if field != "current"]
        text = missing_text(missing, rules)
        return Rendered(ReplyKind.MISSING, text) if text else None
    if outcome == "blocked_by_representative":
        if not all(_is(call, "save_employee_details", "ok") for call in before):
            return None
        return Rendered(ReplyKind.BLOCKED, BLOCKED_TEXT)
    if outcome == "not_eligible":
        if not all(_is(call, "save_employee_details", "ok") or _is(call, "set_current_car", "ok") for call in before):
            return None
        text = not_eligible_text(compare.content["error"])
        return Rendered(ReplyKind.NOT_ELIGIBLE, text) if text else None
    return None


# Hebrew question words. A message with two of them (or two question marks) may ask something beyond
# what a rendered reply answers, so the provider writes that reply instead.
_QUESTION_WORDS = frozenset(normalize("מה כמה האם איך למה מדוע מתי איפה היכן מי איזה איזו אילו מהו מהי מהם מהן"))


def may_ask_more(text: str) -> bool:
    if text.count("?") > 1:
        return True
    asked = sum(
        1 for token in normalize(text) if any(form in _QUESTION_WORDS for form in (token, *prefix_stripped(token)))
    )
    return asked > 1


# Word starts that ask for information about something ("ספר לי", "תסביר", "מידע", "פרטים") rather than
# choosing it. Only ever used to hand a reply to the provider, never to render one; not exhaustive.
_INFORMATION_STARTS = ("ספר", "תספר", "הסבר", "תסביר", "מידע", "פרט", "פירוט")


def plain_request(text: str) -> bool:
    """No question word and no request for information: the stricter check for a comparison reached via
    find_car, where the model's stated purpose is the only sign that a comparison was wanted."""
    for token in normalize(text):
        forms = (token, *prefix_stripped(token))
        if any(form in _QUESTION_WORDS or form.startswith(_INFORMATION_STARTS) for form in forms):
            return False
    return True


# Question words a car choice answers first: what choosing the car changes, whether it is possible, which
# one ("כמה ישתנה לי הנטו אם אעבור לקורולה?", "איזו קורולה אפשר לקבל?"). Any other question word
# (why, how, when, who...) goes to the provider.
_CHOICE_QUESTION_WORDS = frozenset(normalize("מה כמה האם איזה איזו אילו"))
_DEMONSTRATIVES = frozenset(normalize("זה זו זאת אלה"))


def choice_request(text: str) -> bool:
    """No request for information and at most choice-related question words: the check for a car choice
    reached via find_car. "מה זה/זו ..." asks what a car is, so it is not a choice request."""
    tokens = normalize(text)
    for i, token in enumerate(tokens):
        forms = (token, *prefix_stripped(token))
        if any(form.startswith(_INFORMATION_STARTS) for form in forms):
            return False
        asked = next((form for form in forms if form in _QUESTION_WORDS), None)  # the token itself first
        if asked is None:
            continue
        if asked not in _CHOICE_QUESTION_WORDS:
            return False
        if asked == "מה" and tokens[i + 1:i + 2] and tokens[i + 1] in _DEMONSTRATIVES:
            return False
    return True


def _outcome(call: RoundCall) -> str:
    return "ok" if call.content.get("ok") else str(call.content.get("error", {}).get("code", "error"))


def _is(call: RoundCall, name: str, outcome: str) -> bool:
    return call.name == name and _outcome(call) == outcome


def _missing(call: RoundCall) -> list[str]:
    return list(call.content.get("error", {}).get("missing", ()))


# --- E. Details saved -------------------------------------------------------------------------

DETAILS_SAVED_TEXT = "תודה, שמרתי."


def _details_saved(user_text: str, calls: Sequence[RoundCall], state: ConversationState, rules: Rules) -> Rendered | None:
    """Only details were saved while a comparison waits for them: ask for what the last call reports
    still missing. Not when the message may ask anything (a question mark or a question word), not before
    a comparison asked for calculation details (eligibility comes first: no salary interview for a car
    not yet checked), and not when nothing is missing: then the next step is the provider's."""
    if "?" in user_text or not plain_request(user_text) or not _collecting_for_a_comparison(state):
        return None
    if not all(_is(call, "save_employee_details", "ok") or _is(call, "set_current_car", "ok") for call in calls):
        return None
    missing = list(calls[-1].content.get("missing", ()))
    text = missing_text(missing, rules) if missing else None
    return Rendered(ReplyKind.DETAILS_SAVED, f"{DETAILS_SAVED_TEXT}\n{text}") if text else None


def _collecting_for_a_comparison(state: ConversationState) -> bool:
    """The conversation's last comparison stopped for calculation details (eligibility already passed)."""
    for message in reversed(state.messages):
        for result in reversed(message.tool_results):
            if result.name == "compare_alternative" or result.content.get("ran") == "compare_alternative":
                error = result.content.get("error") or {}
                return error.get("code") == "missing_details" and error.get("stage") != "eligibility"
    return False


# --- A. Comparison ----------------------------------------------------------------------------

def comparison_text(summary: ComparisonSummary, state: ConversationState, rules: Rules) -> str:
    result = summary.result
    lines = [
        f"נטו היום ({_name(result.current_situation)}): {summary.current_net:,} ₪",
        f"נטו בחלופה ({_name(result.alternative_situation)}): {summary.alternative_net:,} ₪",
        f"הפרש: {summary.net_difference:+,} ₪ בחודש",
    ]
    changes = _change_parts(summary, rules)
    if changes:
        lines.append("מה משתנה: " + " · ".join(changes))
    if summary.status == ComparisonStatus.ESTIMATE:
        alternative = result.alternative_situation
        lines.append(
            f"שימו לב: ההשתתפות ברכב החלופי ({display_shekels(alternative.participation):,} ₪) היא ברירת מחדל "
            "מהקטלוג, כלומר הערכה ולא נתון מאושר."
        )
        asked = any(q.car_id == alternative.car_id for q in open_questions(state, RepTopic.ALTERNATIVE_PARTICIPATION))
        lines.append(
            "כבר נשלחה שאלה לנציג התחבורה על הסכום המדויק." if asked
            else "אפשר לבקש מנציג התחבורה את הסכום המדויק."
        )
    return "\n".join(lines)


def _name(details: SituationDetails) -> str:
    return "ויתור על הרכב" if details.kind == "waiver" else details.commercial_name


# The workbook's explanation items, in its order; each shown only when the engine reports a change.
_CHANGE_TEXT = {
    "use_value": "שווי שימוש",
    "maintenance": "אחזקת רכב",
    "participation": "השתתפות",
    "family_share": "חלק במסלול המשפחתי",
}
_CHANGE_ORDER = ("use_value", "maintenance", "electric_gross", "participation", "family_share")


def _change_parts(summary: ComparisonSummary, rules: Rules) -> list[str]:
    result = summary.result
    shown = {
        change.item: (display_shekels(change.current), display_shekels(change.alternative))
        for change in result.changes
        if display_shekels(change.current) != display_shekels(change.alternative)
    }
    parts = []
    for item in _CHANGE_ORDER:
        if item not in shown:
            continue
        current, alternative = shown[item]
        if item == "electric_gross":
            if result.alternative_situation.electric:
                parts.append(electric_benefit(rules))
            else:  # today's electric car is left
                parts.append(f"בלי החזר החשמל ({display_shekels(rules.electric.net_bonus):,} ₪ נטו)")
        else:
            parts.append(f"{_CHANGE_TEXT[item]} {current:,} → {alternative:,} ₪")
    return parts


# --- B. Missing details -----------------------------------------------------------------------

# Presentation order and wording; a pair is asked in one line when both are missing.
_MISSING_LINES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("occupation",), "התפקיד שלך"),
    (("current",), "איזה רכב חברה יש לך היום ומה ההשתתפות העצמית החודשית בו (או שוויתרת על רכב)"),
    (("current.participation",), "כמה השתתפות עצמית את/ה משלם/ת היום על הרכב, בחודש"),
    (("base_salary",), "שכר הבסיס החודשי (ברוטו)"),
    (("commissions", "bonus"), "עמלות ובונוס חודשיים (אם אין, אפשר לכתוב שאין)"),
    (("commissions",), "עמלות חודשיות (אם אין, אפשר לכתוב שאין)"),
    (("bonus",), "בונוס חודשי (אם אין, אפשר לכתוב שאין)"),
    (("pension_rate", "training_fund_rate"), "אחוז ההפרשה שלך לפנסיה ולקרן השתלמות"),
    (("pension_rate",), "אחוז ההפרשה שלך לפנסיה"),
    (("training_fund_rate",), "אחוז ההפרשה שלך לקרן השתלמות (אם אין, אפשר לכתוב שאין)"),
    (("credit_points",), "מספר נקודות הזיכוי שלך"),
    (("eligible_children",), ""),  # from rules.json, see _line
)
assert {fields[0] for fields, _ in _MISSING_LINES if len(fields) == 1} == {
    *REQUIRED_DETAILS, "current.participation", "eligible_children"
}


def missing_text(missing: Sequence[str], rules: Rules, eligibility: bool = False) -> str | None:
    """A question for exactly the missing fields, or None if a field is not one we know how to ask about.

    eligibility: the fields decide whether the alternative is possible at all, so they come first.
    """
    remaining = set(missing)
    lines = []
    for fields, text in _MISSING_LINES:
        if set(fields) <= remaining:
            lines.append(_line(fields, text, rules))
            remaining -= set(fields)
    if remaining or not lines:
        return None
    if eligibility:
        header = "כדי לבדוק אם החלופה הזו אפשרית עבורך, צריך קודם לדעת:"
    elif len(lines) == 1:
        header = "כדי לחשב את השינוי בנטו חסר לי עוד פרט אחד:"
    else:
        header = "כדי לחשב את השינוי בנטו חסרים לי עוד כמה פרטים:"
    return "\n".join([header, *(f"• {line}" for line in lines)])


def _line(fields: tuple[str, ...], text: str, rules: Rules) -> str:
    if fields == ("eligible_children",):
        return f"מספר הילדים שלך עד גיל {rules.family_track.child_max_age} (נדרש לרכב מהמסלול המשפחתי)"
    return text


# --- D. Not eligible ---------------------------------------------------------------------------

# Hebrew for rules.json chooseFrom values; a value without wording is left to the provider.
_CHOOSE_FROM = {"own_group_or_lower": "רכב מהקבוצה שלך או מקבוצה נמוכה ממנה"}
_ANOTHER = "אפשר לבדוק חלופה אחרת."


def not_eligible_text(error: Mapping[str, Any]) -> str | None:
    """The refusal in Hebrew from the tool's facts (catalogue, CSV, rules.json), or None if not recognised.

    Eligibility concerns a NEW selection, so the wording restricts choosing the alternative now and
    never says anything about a car the employee already has.
    """
    facts = error.get("facts") or {}
    reason = error.get("reason")
    try:
        if reason == "waiver_not_allowed":
            return f"לפי טבלת התפקידים, בתפקיד {facts['occupation']} אי אפשר לבחור כעת בוויתור על הרכב. {_ANOTHER}"
        if reason == "group_too_high":
            return (
                f"אי אפשר לבחור ב-{facts['commercial_name']} כבחירה חדשה: הרכב בקבוצה {facts['car_group']}, "
                f"והתפקיד שלך בקבוצה {facts['employee_group']}, שמאפשרת לבחור {_CHOOSE_FROM[facts['choose_from']]}. {_ANOTHER}"
            )
        if reason == "not_enough_children" and facts["eligible_children"] is not None:
            return (
                f"אי אפשר לבחור ב-{facts['commercial_name']} כבחירה חדשה במסלול המשפחתי: המסלול מיועד למי שיש לו "
                f"לפחות {facts['min_children']} ילדים עד גיל {facts['child_max_age']}, ולפי מה שמסרת יש לך "
                f"{facts['eligible_children']}. {_ANOTHER}"
            )
    except KeyError:
        return None
    return None


# --- C. Representative and car choice ---------------------------------------------------------

BLOCKED_TEXT = (
    "הרכב הנוכחי שלך אינו בקטלוג, ולכן אי אפשר לחשב את ההשוואה כרגע. "
    "השאלה הועברה לנציג התחבורה, ונמשיך כשתגיע תשובה."
)

_QUESTION_SENT = {
    RepTopic.ALTERNATIVE_PARTICIPATION: "שלחתי לנציג התחבורה שאלה על ההשתתפות ברכב החלופי. כשתגיע תשובה נוכל לעדכן את ההשוואה.",
    RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE: (
        "הרכב הנוכחי שלך אינו בקטלוג, ולכן אי אפשר לחשב כרגע. שלחתי שאלה לנציג התחבורה, ונמשיך כשתגיע תשובה."
    ),
    RepTopic.OTHER: "שלחתי את השאלה לנציג התחבורה. נמשיך כשתגיע תשובה.",
}
_ALREADY_OPEN = "כבר נשלחה לנציג התחבורה שאלה בנושא הזה, והיא עדיין ממתינה לתשובה."


def _question_text(topic: str, already_open: bool) -> str:
    return _ALREADY_OPEN if already_open else _QUESTION_SENT[RepTopic(topic)]


def _candidates_text(call: RoundCall) -> str | None:
    cars = call.content.get("error", {}).get("matches", {}).get("cars", [])
    if len(cars) < 2:
        return None
    which = "הרכב הנוכחי" if call.name == "set_current_car" else "הרכב החלופי"
    lines = [f"כמה רכבים בקטלוג מתאימים למה שכתבת על {which}. לאיזה מהם התכוונת? אפשר לכתוב את שם הדגם."]
    lines += [f"• {car['manufacturer']} {car['commercial_name']} ({car['registration_year']})" for car in cars]
    return "\n".join(lines)
