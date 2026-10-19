"""Step 4 agent tools: thin wrappers over Steps 1-3 that read and update ConversationState.

Each tool takes the current state plus arguments and returns ToolOutcome(new state, typed result).
Tools never raise for bad input or ineligible choices: they return a ToolError the agent can explain.
No salary, eligibility or comparison logic lives here; it is all in comparison.py and below.
"""

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from typing import Generic, TypeVar

from app.agent.state import (
    ComparisonStatus,
    ConversationState,
    ConversationStatus,
    EmployeeDetails,
    QuestionStatus,
    RepQuestion,
    RepTopic,
    ShownComparison,
    blocking_questions,
    conversation_status,
    missing_details,
    open_questions,
)
from app.agent.grounding import matching_occupations
from app.car_search import CarSearch, MatchStatus, build_car_search, normalize
from app.catalog import Catalog, CatalogCar, load_catalog, lookup_vehicle
from app.comparison import (
    CompanyCar,
    ComparisonResult,
    Employee,
    Situation,
    Waiver,
    check_eligibility,
    compare,
    display_shekels,
    eligible_alternatives,
)
from app.errors import NotEligibleError
from app.lookup_config import LookupConfig, load_lookup_config
from app.rules import Rules, load_rules
from app.vehicle_repository import VehicleRepository

T = TypeVar("T")


# --- Context and envelopes ----------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolContext:
    """The data sources the tools read; load() gives the project defaults."""

    rules: Rules
    catalog: Catalog
    lookup: LookupConfig
    repository: VehicleRepository
    car_search: CarSearch

    @staticmethod
    def load() -> "ToolContext":
        return _default_context()


@cache
def _default_context() -> ToolContext:
    rules, catalog, lookup, repository = load_rules(), load_catalog(), load_lookup_config(), VehicleRepository()
    return ToolContext(rules, catalog, lookup, repository, build_car_search(catalog, repository, lookup, rules))


class ToolErrorCode(StrEnum):
    INVALID_ARGUMENT = "invalid_argument"
    UNKNOWN_OCCUPATION = "unknown_occupation"
    NOT_IN_CATALOGUE = "not_in_catalogue"
    MISSING_DETAILS = "missing_details"
    BLOCKED_BY_REPRESENTATIVE = "blocked_by_representative"
    NOT_ELIGIBLE = "not_eligible"
    CANNOT_CALCULATE = "cannot_calculate"


@dataclass(frozen=True, slots=True)
class ToolError:
    code: ToolErrorCode
    detail: str
    missing: tuple[str, ...] = ()  # for MISSING_DETAILS
    reason: str | None = None  # NotEligibleError reason, for NOT_ELIGIBLE
    question_ids: tuple[str, ...] = ()  # for BLOCKED_BY_REPRESENTATIVE
    stage: str | None = None  # MISSING_DETAILS: "eligibility" when only the deciding facts are asked
    facts: dict[str, object] | None = None  # NOT_ELIGIBLE: the data the verdict rests on (catalogue, rules)
    # compare_alternative: the alternative the error is about, for screens (app.agent.outcome).
    # Not part of the tool's JSON result, so the model sees exactly what it saw before.
    alternative: Situation | None = None


@dataclass(frozen=True, slots=True)
class ToolOutcome(Generic[T]):
    state: ConversationState
    result: T | ToolError


# --- Result types -------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DetailsSaved:
    saved: tuple[str, ...]  # EmployeeDetails fields set by this call
    missing: tuple[str, ...]
    status: ConversationStatus


@dataclass(frozen=True, slots=True)
class CarSummary:
    car_id: str
    manufacturer: str  # from the Tax Authority CSV
    commercial_name: str  # from the Tax Authority CSV
    registration_year: int
    group: str
    drive: str | None  # rules valueOfUse.driveField name
    use_value: int  # Step 1


@dataclass(frozen=True, slots=True)
class CarMatches:
    status: MatchStatus  # identified: use cars[0].car_id; candidates: let the employee choose; not_found
    cars: tuple[CarSummary, ...]
    unrecognized_words: tuple[str, ...]  # words that matched nothing; a reason to confirm, not to guess

    @property
    def found(self) -> bool:
        return self.status != MatchStatus.NOT_FOUND


@dataclass(frozen=True, slots=True)
class CurrentSituationSaved:
    current: Situation
    missing: tuple[str, ...]
    status: ConversationStatus


@dataclass(frozen=True, slots=True)
class AlternativeCar:
    car: CarSummary
    basis: str  # eligibility basis
    default_participation: int  # catalogue default: an estimate, never confirmed data
    participation_is_estimate: bool = True


@dataclass(frozen=True, slots=True)
class AlternativeWaiver:
    maintenance: int  # the occupation's amount (workbook)


@dataclass(frozen=True, slots=True)
class Alternatives:
    cars: tuple[AlternativeCar, ...]
    waiver: AlternativeWaiver | None  # None when the occupation does not allow waiving
    family_track_needs_children: bool  # family-track cars left out because the number of children is unknown


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    status: ComparisonStatus
    estimated_fields: tuple[str, ...]  # e.g. ("alternative.participation",); empty when confirmed
    current_net: int  # whole shekels, as displayed
    alternative_net: int
    net_difference: int
    result: ComparisonResult  # full structured detail from Step 3


@dataclass(frozen=True, slots=True)
class QuestionSent:
    question: RepQuestion
    already_open: bool  # an identical open question existed; no duplicate was created
    status: ConversationStatus


# --- Tools --------------------------------------------------------------------

# Tool argument -> (EmployeeDetails field, conversion). Percentages arrive as percent (6 = 6%).
_DETAIL_ARGUMENTS = {
    "occupation": "occupation",
    "base_salary": "base_salary",
    "commissions": "commissions",
    "bonus": "bonus",
    "pension_percent": "pension_rate",
    "training_fund_percent": "training_fund_rate",
    "credit_points": "credit_points",
    "eligible_children": "eligible_children",
}


def save_employee_details(state: ConversationState, ctx: ToolContext, **arguments: object) -> ToolOutcome[DetailsSaved]:
    """Save whichever employee details were given; None means "not given" and changes nothing.

    All-or-nothing: if any argument is invalid, nothing is saved.
    """
    unknown = sorted(set(arguments) - set(_DETAIL_ARGUMENTS))
    if unknown:
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, f"Unknown arguments {unknown}; known: {list(_DETAIL_ARGUMENTS)}")

    updates: dict[str, object] = {}
    for argument, value in arguments.items():
        if value is None:
            continue
        converted = _convert_detail(argument, value, ctx)
        if isinstance(converted, ToolError):
            return ToolOutcome(state, converted)
        updates[_DETAIL_ARGUMENTS[argument]] = converted

    new_state = _with_details(state, replace(state.details, **updates)) if updates else state
    return ToolOutcome(
        new_state,
        DetailsSaved(
            saved=tuple(updates),
            missing=missing_details(new_state, ctx.rules, new_state.alternative),
            status=conversation_status(new_state, ctx.rules),
        ),
    )


def find_car(state: ConversationState, ctx: ToolContext, query: str) -> ToolOutcome[CarMatches]:
    """Catalogue cars matching what the employee typed, in Hebrew or English (see app.car_search).

    identified: exactly one car, every word recognized. candidates: the employee must choose or
    confirm. not_found: the car is not in the catalogue (offer the representative).
    """
    if not normalize(query):
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, "query must name the manufacturer, model or year")
    found = ctx.car_search.search(query)
    cars = tuple(_car_summary(ctx.catalog.cars[car_id], ctx) for car_id in found.car_ids)
    return ToolOutcome(state, CarMatches(found.status, cars, found.unrecognized_words))


def set_current_car(
    state: ConversationState,
    ctx: ToolContext,
    car_id: str | None = None,
    participation: float | None = None,
    waiver: bool = False,
) -> ToolOutcome[CurrentSituationSaved]:
    """Record today's situation: a catalogue car with the employee's stated participation, or a waiver."""
    if waiver == (car_id is not None):
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, "give either car_id (with participation) or waiver=true")

    car_id = _catalogue_id(car_id, ctx)
    if waiver:
        current: Situation = Waiver()
    else:
        if car_id not in ctx.catalog.cars:
            return _error(
                state, ToolErrorCode.NOT_IN_CATALOGUE,
                f"{car_id!r} is not a catalogue car; a car outside the catalogue cannot be calculated yet",
            )
        if participation is None:
            return _error(
                state, ToolErrorCode.MISSING_DETAILS,
                "the current car's participation must be stated by the employee", missing=("current.participation",),
            )
        if participation < 0:
            return _error(state, ToolErrorCode.INVALID_ARGUMENT, f"participation must not be negative: {participation}")
        current = CompanyCar(car_id, participation)

    new_state = _with_details(state, replace(state.details, current=current))
    return ToolOutcome(
        new_state,
        CurrentSituationSaved(
            current=current,
            missing=missing_details(new_state, ctx.rules, new_state.alternative),
            status=conversation_status(new_state, ctx.rules),
        ),
    )


def list_alternatives(state: ConversationState, ctx: ToolContext) -> ToolOutcome[Alternatives]:
    """Every alternative the employee may newly select, with Step 1 use values."""
    occupation_name = state.details.occupation
    if occupation_name is None:
        return _error(state, ToolErrorCode.MISSING_DETAILS, "the occupation decides the eligible cars", missing=("occupation",))

    children = state.details.eligible_children
    options = eligible_alternatives(occupation_name, children, rules=ctx.rules, catalog=ctx.catalog)
    cars = []
    waiver = None
    for option in options:
        if isinstance(option.alternative, Waiver):
            waiver = AlternativeWaiver(maintenance=ctx.catalog.occupations[occupation_name].waiver_maintenance)
            continue
        car = ctx.catalog.cars[option.alternative.car_id]
        cars.append(AlternativeCar(_car_summary(car, ctx), option.basis, car.default_participation))
    return ToolOutcome(
        state,
        Alternatives(
            cars=tuple(cars),
            waiver=waiver,
            family_track_needs_children=children is None and bool(ctx.rules.family_track.cars),
        ),
    )


def compare_alternative(
    state: ConversationState,
    ctx: ToolContext,
    car_id: str | None = None,
    waiver: bool = False,
    participation: float | None = None,
) -> ToolOutcome[ComparisonSummary]:
    """Compare today with one alternative using only saved details.

    participation is the confirmed amount for the alternative car (from the employee or the
    representative). Without it the catalogue default is used and the result is an ESTIMATE.
    """
    if waiver == (car_id is not None):
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, "give either car_id or waiver=true")
    if waiver and participation is not None:
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, "a waiver has no participation")
    car_id = _catalogue_id(car_id, ctx)
    if car_id is not None and car_id not in ctx.catalog.cars:
        return _error(state, ToolErrorCode.NOT_IN_CATALOGUE, f"{car_id!r} is not a catalogue car")
    if participation is not None and participation < 0:
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, f"participation must not be negative: {participation}")

    alternative: Situation = Waiver() if waiver else CompanyCar(car_id, participation)

    blocking = blocking_questions(state)
    if blocking:
        return ToolOutcome(
            state,
            ToolError(
                ToolErrorCode.BLOCKED_BY_REPRESENTATIVE,
                "today's car is not in the catalogue; waiting for the representative",
                question_ids=tuple(q.question_id for q in blocking),
                alternative=alternative,
            ),
        )

    # Eligibility first, on the facts known so far (Step 3 decides): no salary questions for an
    # alternative the employee cannot select, and only the facts that can change the verdict.
    details = state.details
    try:
        gate = check_eligibility(
            alternative, occupation_name=details.occupation, eligible_children=details.eligible_children,
            current=details.current, rules=ctx.rules, catalog=ctx.catalog,
        )
    except NotEligibleError as e:
        return ToolOutcome(
            state,
            ToolError(
                ToolErrorCode.NOT_ELIGIBLE, str(e), reason=e.reason,
                facts=_eligibility_facts(e.reason, alternative, state, ctx), alternative=alternative,
            ),
        )
    if gate.missing:
        return ToolOutcome(
            state,
            ToolError(
                ToolErrorCode.MISSING_DETAILS, "details that decide eligibility are needed first",
                missing=gate.missing, stage="eligibility", alternative=alternative,
            ),
        )

    missing = missing_details(state, ctx.rules, alternative)
    if missing:
        return ToolOutcome(
            state,
            ToolError(ToolErrorCode.MISSING_DETAILS, "details needed before comparing", missing=missing, alternative=alternative),
        )

    employee = Employee(
        occupation=details.occupation,
        base_salary=details.base_salary,
        commissions=details.commissions,
        bonus=details.bonus,
        pension_rate=details.pension_rate,
        training_fund_rate=details.training_fund_rate,
        credit_points=details.credit_points,
        eligible_children=details.eligible_children,
        current=details.current,
    )
    try:
        result = compare(
            employee, alternative, rules=ctx.rules, catalog=ctx.catalog, lookup=ctx.lookup, repository=ctx.repository
        )
    except NotEligibleError as e:
        return ToolOutcome(state, ToolError(ToolErrorCode.NOT_ELIGIBLE, str(e), reason=e.reason, alternative=alternative))
    except ValueError as e:
        return _error(state, ToolErrorCode.CANNOT_CALCULATE, str(e))

    estimated = (
        ("alternative.participation",)
        if result.alternative_situation.participation_source == "catalogue_default"
        else ()
    )
    status = ComparisonStatus.ESTIMATE if estimated else ComparisonStatus.CONFIRMED
    new_state = replace(
        state,
        alternative=alternative,
        last_comparison=ShownComparison(alternative, status, estimated, result),
    )
    return ToolOutcome(
        new_state,
        ComparisonSummary(
            status=status,
            estimated_fields=estimated,
            current_net=display_shekels(result.current.net),
            alternative_net=display_shekels(result.alternative.net),
            net_difference=result.net_difference,
            result=result,
        ),
    )


def ask_representative(
    state: ConversationState, ctx: ToolContext, topic: str, question: str, car_id: str | None = None
) -> ToolOutcome[QuestionSent]:
    """Open a question to the representative (answered on the representative screen, Step 5)."""
    try:
        rep_topic = RepTopic(topic)
    except ValueError:
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, f"topic must be one of {[t.value for t in RepTopic]}")
    car_id = _catalogue_id(car_id, ctx)
    text = question.strip()
    if not text:
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, "question must not be empty")
    if rep_topic == RepTopic.ALTERNATIVE_PARTICIPATION and car_id not in ctx.catalog.cars:
        return _error(state, ToolErrorCode.INVALID_ARGUMENT, "a participation question needs the catalogue car_id")

    for existing in open_questions(state, rep_topic):
        if existing.car_id == car_id and (rep_topic != RepTopic.OTHER or existing.text == text):
            return ToolOutcome(state, QuestionSent(existing, True, conversation_status(state, ctx.rules)))

    new_question = RepQuestion(
        question_id=f"Q-{len(state.rep_questions) + 1}",
        topic=rep_topic,
        text=text,
        car_id=car_id,
        status=QuestionStatus.OPEN,
    )
    new_state = replace(state, rep_questions=(*state.rep_questions, new_question))
    return ToolOutcome(new_state, QuestionSent(new_question, False, conversation_status(new_state, ctx.rules)))


# --- Helpers ------------------------------------------------------------------

def _error(state: ConversationState, code: ToolErrorCode, detail: str, missing: tuple[str, ...] = ()) -> ToolOutcome:
    return ToolOutcome(state, ToolError(code, detail, missing=missing))


def _catalogue_id(car_id: str | None, ctx: ToolContext) -> str | None:
    """The stable catalogue ID for a loosely written one ("car-10" -> "CAR-10"); unknown IDs pass unchanged."""
    if car_id is None or car_id in ctx.catalog.cars:
        return car_id
    return ctx.car_search.car_id(car_id) or car_id


def _eligibility_facts(reason: str, alternative: Situation, state: ConversationState, ctx: ToolContext) -> dict[str, object]:
    """What a refusal rests on, read from the workbook, the CSV and rules.json; nothing is decided here."""
    facts: dict[str, object] = {"new_selection": True}  # eligibility applies to a newly selected alternative
    occupation = ctx.catalog.occupations.get(state.details.occupation or "")
    if occupation is not None:
        facts.update(occupation=occupation.name, employee_group=occupation.group)
    if isinstance(alternative, CompanyCar):
        car = ctx.catalog.cars[alternative.car_id]
        facts.update(car_id=car.car_id, commercial_name=_car_summary(car, ctx).commercial_name, car_group=car.group)
    if reason == "group_too_high":
        facts["choose_from"] = ctx.rules.choose_from
    if reason == "not_enough_children":
        family = ctx.rules.family_track
        facts.update(
            min_children=family.min_children, child_max_age=family.child_max_age, eligible_children=state.details.eligible_children
        )
    return facts


def _with_details(state: ConversationState, details: EmployeeDetails) -> ConversationState:
    """New details make the last comparison stale, so it is dropped."""
    return replace(state, details=details, last_comparison=None)


def _convert_detail(argument: str, value: object, ctx: ToolContext) -> object:
    if argument == "occupation":
        if isinstance(value, str) and value in ctx.catalog.occupations:
            return value
        # The employee's wording ("מנהלת מוצר") -> the one workbook name with the same words ("מנהל/ת מוצר").
        matches = matching_occupations(value, ctx.catalog.occupations) if isinstance(value, str) else ()
        if len(matches) == 1:
            return matches[0]
        return ToolError(
            ToolErrorCode.UNKNOWN_OCCUPATION,
            f"{value!r} is not in the occupations table; known: {list(ctx.catalog.occupations)}",
        )
    if argument == "eligible_children":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return ToolError(ToolErrorCode.INVALID_ARGUMENT, f"eligible_children must be a whole number >= 0: {value!r}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return ToolError(ToolErrorCode.INVALID_ARGUMENT, f"{argument} must be a number >= 0: {value!r}")
    if argument.endswith("_percent"):
        if value > 100:
            return ToolError(ToolErrorCode.INVALID_ARGUMENT, f"{argument} is a percentage (6 = 6%): {value!r}")
        return value / 100
    return value


def _car_summary(car: CatalogCar, ctx: ToolContext) -> CarSummary:
    record = lookup_vehicle(car, ctx.repository, ctx.lookup)
    return CarSummary(
        car_id=car.car_id,
        manufacturer=record.manufacturer,
        commercial_name=record.commercial_name,
        registration_year=car.registration_year,
        group=car.group,
        drive=ctx.rules.value_of_use.drive_field.get(record.propulsion_code),
        use_value=record.use_value,
    )

