"""Conversation state for the Step 4 agent: what the employee has told us, and what is still open.

Immutable: tools return a new state instead of changing one. A field is None until the employee
gives it; it is never filled with a guess.
"""

from dataclasses import dataclass, replace
from enum import StrEnum

from app.agent.llm import Message
from app.comparison import CompanyCar, ComparisonResult, Situation
from app.rules import Rules


class RepTopic(StrEnum):
    ALTERNATIVE_PARTICIPATION = "alternative_participation"
    CURRENT_CAR_NOT_IN_CATALOGUE = "current_car_not_in_catalogue"
    OTHER = "other"


class QuestionStatus(StrEnum):
    OPEN = "open"
    ANSWERED = "answered"


class ConversationStatus(StrEnum):
    COLLECTING = "collecting"  # details are still missing
    WAITING_FOR_REPRESENTATIVE = "waiting_for_representative"  # an open question blocks the comparison
    READY = "ready"  # a comparison can be calculated


class ComparisonStatus(StrEnum):
    CONFIRMED = "confirmed"  # every input was stated by the employee (or the representative)
    ESTIMATE = "estimate"  # at least one input is a catalogue default; see estimated_fields


@dataclass(frozen=True, slots=True)
class EmployeeDetails:
    occupation: str | None = None  # workbook occupations table; gives the group and waiver amount
    base_salary: float | None = None
    commissions: float | None = None  # an explicit "none" is saved as 0
    bonus: float | None = None
    pension_rate: float | None = None  # fraction, 0.06 = 6%
    training_fund_rate: float | None = None  # fraction
    credit_points: float | None = None
    eligible_children: int | None = None  # children up to rules familyTrack.childMaxAge
    current: Situation | None = None  # a catalogue car with stated participation, or a waiver


@dataclass(frozen=True, slots=True)
class RepQuestion:
    question_id: str
    topic: RepTopic
    text: str
    car_id: str | None = None  # the alternative car, for alternative_participation questions
    status: QuestionStatus = QuestionStatus.OPEN
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class ShownComparison:
    """The last comparison the employee was shown."""

    alternative: Situation
    status: ComparisonStatus
    estimated_fields: tuple[str, ...]
    result: ComparisonResult


@dataclass(frozen=True, slots=True)
class ConversationState:
    conversation_id: str
    details: EmployeeDetails = EmployeeDetails()
    alternative: Situation | None = None  # the alternative last compared
    rep_questions: tuple[RepQuestion, ...] = ()
    last_comparison: ShownComparison | None = None
    messages: tuple[Message, ...] = ()  # provider-neutral transcript, oldest first
    # Car selection guard (enforced by the agent's tool registry, not the prompt):
    # usable = named by the employee's own words (identified, or chosen from what was offered);
    # offered = shown as candidates/alternatives, usable once a later employee message names exactly one.
    usable_car_ids: frozenset[str] = frozenset()
    offered_car_ids: frozenset[str] = frozenset()


def with_messages(state: ConversationState, *messages: Message) -> ConversationState:
    """A new state with messages appended to the transcript."""
    return replace(state, messages=(*state.messages, *messages))


# Details every comparison needs; eligible_children only for a new family-track alternative.
REQUIRED_DETAILS = (
    "occupation",
    "base_salary",
    "commissions",
    "bonus",
    "pension_rate",
    "training_fund_rate",
    "credit_points",
    "current",
)


def missing_details(state: ConversationState, rules: Rules, alternative: Situation | None = None) -> tuple[str, ...]:
    """EmployeeDetails fields still needed to compare with alternative (or with any alternative)."""
    details = state.details
    missing = [name for name in REQUIRED_DETAILS if getattr(details, name) is None]
    keeps_current_car = isinstance(details.current, CompanyCar) and details.current.car_id == getattr(alternative, "car_id", None)
    if (
        isinstance(alternative, CompanyCar)
        and alternative.car_id in rules.family_track.cars
        and details.eligible_children is None
        and not keeps_current_car  # children decide only a new family-track selection
    ):
        missing.append("eligible_children")
    return tuple(missing)


def open_questions(state: ConversationState, topic: RepTopic | None = None) -> tuple[RepQuestion, ...]:
    return tuple(
        q for q in state.rep_questions if q.status == QuestionStatus.OPEN and (topic is None or q.topic == topic)
    )


def blocking_questions(state: ConversationState) -> tuple[RepQuestion, ...]:
    """Open questions without which nothing can be calculated: an unknown current car."""
    if state.details.current is not None:
        return ()
    return open_questions(state, RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE)


def conversation_status(state: ConversationState, rules: Rules) -> ConversationStatus:
    if blocking_questions(state):
        return ConversationStatus.WAITING_FOR_REPRESENTATIVE
    if missing_details(state, rules, state.alternative):
        return ConversationStatus.COLLECTING
    return ConversationStatus.READY

