"""The public HTTP contract (Step 5). Every response field is listed here; nothing else is sent.

No tool, provider or debug data, no timings and no exception text ever appear in these models.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGE_LENGTH = 2000
SESSION_ID_PATTERN = r"^[0-9a-f]{32}$"


class UiState(StrEnum):
    """Stable screen states. "calculating" is not one: the browser shows it while a request is pending."""

    IDLE = "idle"  # a new conversation; only POST /api/sessions returns it
    ELIGIBILITY_CHECK = "eligibility_check"  # only the facts that decide eligibility are asked
    NOT_ELIGIBLE = "not_eligible"
    COLLECTING = "collecting"  # eligibility passed; details the calculation needs are asked
    ESTIMATE = "estimate"  # a comparison with at least one catalogue-default input
    CONFIRMED = "confirmed"  # a comparison with every input stated
    REPRESENTATIVE = "representative"  # a question is open with the transport representative
    CAR_CHOICE = "car_choice"  # several catalogue cars match what the employee wrote
    ANSWERED = "answered"  # a reply with no decisive outcome (information, follow-up)
    INCOMPLETE = "incomplete"  # the advisor could not complete a reply grounded in the data
    TEMPORARY_ERROR = "temporary_error"  # the advisor service failed; nothing was saved; retry


# --- Requests -------------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    message: str

    @field_validator("message")
    @classmethod
    def _message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        if len(value) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"message must be at most {MAX_MESSAGE_LENGTH} characters")
        return value


# --- Responses ------------------------------------------------------------------------------

class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CarOut(_Out):
    car_id: str
    manufacturer: str
    commercial_name: str
    registration_year: int
    group: str
    electric: bool
    family_track: bool


class AlternativeOut(_Out):
    kind: Literal["company_car", "waiver"]
    car: CarOut | None = None


class EligibilityFactsOut(_Out):
    """What a refusal rests on, exactly as the backend gave it."""

    occupation: str | None = None
    employee_group: str | None = None
    car_group: str | None = None
    choose_from: str | None = None
    min_children: int | None = None
    child_max_age: int | None = None
    eligible_children: int | None = None


class EligibilityOut(_Out):
    missing: list[str] = []
    reason: str | None = None
    facts: EligibilityFactsOut | None = None


class DetailsOut(_Out):
    missing: list[str]
    provided: list[str]


class SideOut(_Out):
    kind: Literal["company_car", "waiver"]
    car: CarOut | None = None
    net: int
    use_value: int
    participation: int
    participation_source: Literal["stated", "catalogue_default"] | None = None
    maintenance: int
    family_share: int | None = None


class ChangeOut(_Out):
    item: Literal["use_value", "maintenance", "participation", "family_share"]
    current: int
    alternative: int


class ElectricBenefitOut(_Out):
    net: int
    grossed_up: bool
    applies_to: Literal["alternative", "current"]


class ComparisonOut(_Out):
    status: Literal["estimate", "confirmed"]
    estimated_fields: list[str]
    current: SideOut
    alternative: SideOut
    net_difference: int
    changes: list[ChangeOut]
    electric_benefit: ElectricBenefitOut | None = None


class QuestionOut(_Out):
    question_id: str
    topic: Literal["alternative_participation", "current_car_not_in_catalogue", "other"]
    status: Literal["open", "answered"]


class LatestQuestionOut(_Out):
    question_id: str
    topic: Literal["alternative_participation", "current_car_not_in_catalogue", "other"]
    already_open: bool


class RepresentativeOut(_Out):
    questions: list[QuestionOut]
    latest: LatestQuestionOut | None = None


class CarChoiceOut(_Out):
    target: Literal["current", "alternative"]
    options: list[CarOut]


class ChatResponse(_Out):
    session_id: str
    turn: int  # committed turns so far; unchanged by a temporary error
    reply: str  # the advisor's reply, verbatim; shown as text
    ui_state: UiState
    retryable: bool = False
    alternative: AlternativeOut | None = None
    current: CarOut | None = None
    eligibility: EligibilityOut | None = None
    details: DetailsOut | None = None
    comparison: ComparisonOut | None = None
    representative: RepresentativeOut | None = None
    car_choice: CarChoiceOut | None = None


class SessionCreated(_Out):
    session_id: str
    ui_state: UiState = UiState.IDLE


class HealthOut(_Out):
    status: Literal["ok"] = "ok"


class ErrorDetail(_Out):
    code: str
    message: str  # Hebrew, for the employee; never exception text
    fields: list[str] | None = None  # invalid_request: which request fields were invalid (names only)


class ErrorOut(_Out):
    error: ErrorDetail
