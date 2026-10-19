"""What a turn decided, as data, for screens that must not read the Hebrew reply (presentation only).

run_turn records, across all of a turn's tool rounds, the last decisive tool outcome:
  COMPARISON          compare_alternative calculated a comparison;
  MISSING_DETAILS     compare_alternative needs facts first: stage ELIGIBILITY (only the facts that
                      decide whether the alternative may be selected) or CALCULATION (eligibility
                      passed; details the calculation needs);
  NOT_ELIGIBLE        the alternative cannot be newly selected;
  BLOCKED             the comparison waits for the representative (today's car is not in the catalogue);
  QUESTION_SENT       a question was opened for the representative;
  CAR_CHOICE          several catalogue cars match what the employee wrote.

Nothing is decided here: every kind, field, stage, reason, fact and figure is read from the tools'
own structured results (app.agent.tools), exactly as the tools produced them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from app.agent.render import RoundCall
from app.agent.tools import ComparisonSummary
from app.comparison import CompanyCar, Waiver


class OutcomeKind(StrEnum):
    COMPARISON = "comparison"
    MISSING_DETAILS = "missing_details"
    NOT_ELIGIBLE = "not_eligible"
    BLOCKED = "blocked"
    QUESTION_SENT = "question_sent"
    CAR_CHOICE = "car_choice"


class MissingStage(StrEnum):
    ELIGIBILITY = "eligibility"
    CALCULATION = "calculation"


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    kind: OutcomeKind
    car_id: str | None = None  # the alternative car the outcome is about; None for a waiver or unknown
    waiver: bool = False  # the alternative is a waiver
    missing: tuple[str, ...] = ()  # MISSING_DETAILS: field names, in the tool's order
    stage: MissingStage | None = None  # MISSING_DETAILS
    reason: str | None = None  # NOT_ELIGIBLE: NotEligibleError reason
    facts: Mapping[str, object] | None = None  # NOT_ELIGIBLE: what the verdict rests on, as the tool gave it
    question_id: str | None = None  # QUESTION_SENT
    question_topic: str | None = None  # QUESTION_SENT: RepTopic value
    already_open: bool = False  # QUESTION_SENT: an identical open question existed
    choice_target: str | None = None  # CAR_CHOICE: "current" or "alternative"
    choice_car_ids: tuple[str, ...] = ()  # CAR_CHOICE: the matching catalogue cars, in the tool's order
    # The turn's last calculated comparison, also when a later call (e.g. a question) decided the turn.
    comparison: ComparisonSummary | None = None


def round_outcome(calls: Sequence[RoundCall], previous: TurnOutcome | None) -> TurnOutcome | None:
    """The turn's outcome after one more round: its last decisive call, else the previous outcome."""
    latest = previous
    comparison = previous.comparison if previous else None
    for call in calls:
        if call.comparison is not None:
            comparison = call.comparison
        decided = _decisive(call)
        if decided is not None:
            latest = decided
    if latest is None:
        return None
    return replace(latest, comparison=comparison)


def _decisive(call: RoundCall) -> TurnOutcome | None:
    code = _code(call.content)
    if code == "car_needs_choice" and call.name in ("compare_alternative", "set_current_car"):
        cars = call.content["error"].get("matches", {}).get("cars", [])
        return TurnOutcome(
            OutcomeKind.CAR_CHOICE,
            choice_target="current" if call.name == "set_current_car" else "alternative",
            choice_car_ids=tuple(str(car["car_id"]) for car in cars),
        )
    if call.name == "ask_representative" and code == "ok":
        question = call.content.get("question", {})
        return TurnOutcome(
            OutcomeKind.QUESTION_SENT,
            car_id=question.get("car_id"),
            question_id=question.get("question_id"),
            question_topic=question.get("topic"),
            already_open=bool(call.content.get("already_open")),
        )
    if call.name != "compare_alternative":
        return None
    if code == "ok" and call.comparison is not None:
        details = call.comparison.result.alternative_situation
        return TurnOutcome(OutcomeKind.COMPARISON, car_id=details.car_id, waiver=details.kind == "waiver")
    error = call.error
    if error is None:
        return None
    car_id, waiver = _alternative(error.alternative)
    if code == "missing_details":
        stage = MissingStage.ELIGIBILITY if error.stage == "eligibility" else MissingStage.CALCULATION
        return TurnOutcome(OutcomeKind.MISSING_DETAILS, car_id, waiver, missing=error.missing, stage=stage)
    if code == "not_eligible":
        return TurnOutcome(OutcomeKind.NOT_ELIGIBLE, car_id, waiver, reason=error.reason, facts=error.facts)
    if code == "blocked_by_representative":
        return TurnOutcome(OutcomeKind.BLOCKED, car_id, waiver)
    return None


def _code(content: Mapping[str, Any]) -> str:
    return "ok" if content.get("ok") else str(content.get("error", {}).get("code", "error"))


def _alternative(situation: CompanyCar | Waiver | None) -> tuple[str | None, bool]:
    if isinstance(situation, CompanyCar):
        return situation.car_id, False
    return None, isinstance(situation, Waiver)
