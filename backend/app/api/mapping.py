"""TurnResult + server-side state -> the public ChatResponse. Field mapping only.

The screen state comes from the turn's structured outcome (TurnResult.outcome, TurnResult.stop),
never from the reply text and never from earlier responses. Every figure comes from the backend's
views (app.agent.views); nothing is decided or calculated here.
"""

from collections.abc import Mapping

from app.agent.agent import TurnResult, TurnStop
from app.agent.outcome import MissingStage, OutcomeKind, TurnOutcome
from app.agent.state import ComparisonStatus, ConversationState
from app.agent.tools import ComparisonSummary, ToolContext
from app.comparison import display_shekels
from app.agent.views import (
    CarView,
    ComparisonView,
    SideView,
    car_view,
    comparison_view,
    current_car_id,
    provided_details,
)
from app.api.schemas import (
    AlternativeOut,
    CarChoiceOut,
    CatalogueOut,
    CarOut,
    ChangeOut,
    ChatResponse,
    ComparisonOut,
    DetailsOut,
    ElectricBenefitOut,
    EligibilityFactsOut,
    EligibilityOut,
    LatestQuestionOut,
    QuestionOut,
    QueueQuestionOut,
    RepresentativeOut,
    SideOut,
    UiState,
)

_FACT_FIELDS = tuple(EligibilityFactsOut.model_fields)


def ui_state(turn: TurnResult) -> UiState:
    if turn.stop == TurnStop.PROVIDER_ERROR:
        return UiState.TEMPORARY_ERROR
    if turn.stop == TurnStop.MAX_ROUNDS:
        return UiState.INCOMPLETE
    outcome = turn.outcome
    if turn.stop == TurnStop.REPLACED_UNSUPPORTED_NUMBERS:
        # The reply was replaced: by the turn's comparison when there was one, else by an apology.
        return _comparison_state(outcome.comparison) if outcome and outcome.comparison else UiState.INCOMPLETE
    if outcome is None:
        return UiState.ANSWERED
    if outcome.kind == OutcomeKind.COMPARISON and outcome.comparison is not None:
        return _comparison_state(outcome.comparison)
    if outcome.kind == OutcomeKind.MISSING_DETAILS:
        return UiState.ELIGIBILITY_CHECK if outcome.stage == MissingStage.ELIGIBILITY else UiState.COLLECTING
    return {
        OutcomeKind.NOT_ELIGIBLE: UiState.NOT_ELIGIBLE,
        OutcomeKind.BLOCKED: UiState.REPRESENTATIVE,
        OutcomeKind.QUESTION_SENT: UiState.REPRESENTATIVE,
        OutcomeKind.CAR_CHOICE: UiState.CAR_CHOICE,
    }.get(outcome.kind, UiState.ANSWERED)


def chat_response(session_id: str, turn_number: int, turn: TurnResult, ctx: ToolContext) -> ChatResponse:
    """The response for a committed turn: turn.state is the session's state from now on."""
    state = ui_state(turn)
    outcome = turn.outcome
    conversation = turn.state
    response = ChatResponse(session_id=session_id, turn=turn_number, reply=turn.reply, ui_state=state)

    alternative = _alternative(outcome, ctx)
    current = _car(car_view(current_car_id(conversation.details), ctx))
    eligibility = details = choice = None
    if outcome is not None and state == UiState.ELIGIBILITY_CHECK:
        eligibility = EligibilityOut(missing=list(outcome.missing))
    elif outcome is not None and state == UiState.NOT_ELIGIBLE:
        eligibility = EligibilityOut(reason=outcome.reason, facts=_facts(outcome.facts))
    elif outcome is not None and state == UiState.COLLECTING:
        details = DetailsOut(missing=list(outcome.missing), provided=list(provided_details(conversation.details)))
    elif outcome is not None and state == UiState.CAR_CHOICE:
        options = [view for view in (car_view(car_id, ctx) for car_id in outcome.choice_car_ids) if view]
        choice = CarChoiceOut(target=outcome.choice_target, options=[_car(view) for view in options])

    shown = conversation.last_comparison
    return response.model_copy(update={
        "alternative": alternative,
        "current": current,
        "eligibility": eligibility,
        "details": details,
        "comparison": _comparison(comparison_view(shown, ctx)) if shown is not None else None,
        "representative": _representative(conversation, outcome),
        "car_choice": choice,
    })


def error_response(session_id: str, turn_number: int, turn: TurnResult) -> ChatResponse:
    """A temporary failure: the reply only. Nothing from the failed turn is kept or shown."""
    return ChatResponse(
        session_id=session_id, turn=turn_number, reply=turn.reply, ui_state=UiState.TEMPORARY_ERROR, retryable=True
    )


# --- Helpers ---------------------------------------------------------------------------------

def _comparison_state(summary: ComparisonSummary) -> UiState:
    return UiState.ESTIMATE if summary.status == ComparisonStatus.ESTIMATE else UiState.CONFIRMED


def _alternative(outcome: TurnOutcome | None, ctx: ToolContext) -> AlternativeOut | None:
    if outcome is None or outcome.kind in (OutcomeKind.CAR_CHOICE, OutcomeKind.QUESTION_SENT):
        return None
    if outcome.waiver:
        return AlternativeOut(kind="waiver")
    view = car_view(outcome.car_id, ctx)
    return AlternativeOut(kind="company_car", car=_car(view)) if view else None


def _car(view: CarView | None) -> CarOut | None:
    if view is None:
        return None
    return CarOut(
        car_id=view.car_id, manufacturer=view.manufacturer, commercial_name=view.commercial_name,
        registration_year=view.registration_year, group=view.group, electric=view.electric,
        family_track=view.family_track,
    )


def _facts(facts: Mapping[str, object] | None) -> EligibilityFactsOut | None:
    if not facts:
        return None
    return EligibilityFactsOut(**{name: facts[name] for name in _FACT_FIELDS if facts.get(name) is not None})


def _side(side: SideView) -> SideOut:
    return SideOut(
        kind=side.kind, car=_car(side.car), net=side.net, use_value=side.use_value, participation=side.participation,
        participation_source=side.participation_source, maintenance=side.maintenance, family_share=side.family_share,
    )


def _comparison(view: ComparisonView) -> ComparisonOut:
    electric = view.electric_benefit
    return ComparisonOut(
        status=view.status,
        estimated_fields=list(view.estimated_fields),
        current=_side(view.current),
        alternative=_side(view.alternative),
        net_difference=view.net_difference,
        changes=[ChangeOut(item=c.item, current=c.current, alternative=c.alternative) for c in view.changes],
        electric_benefit=(
            ElectricBenefitOut(net=electric.net, grossed_up=electric.grossed_up, applies_to=electric.applies_to)
            if electric else None
        ),
    )


def representative_status(state: ConversationState) -> RepresentativeOut:
    """The employee's questions and the representative's answers, between turns (polled by the screen)."""
    return _representative(state, None) or RepresentativeOut(questions=[])


def queue_questions(session_id: str, state: ConversationState, ctx: ToolContext) -> list[QueueQuestionOut]:
    return [
        QueueQuestionOut(
            session_id=session_id, question_id=q.question_id, topic=q.topic.value, status=q.status.value,
            question=q.text, car=_car(car_view(q.car_id, ctx)), answer=q.answer,
            participation=display_shekels(q.participation) if q.participation is not None else None,
        )
        for q in state.rep_questions
    ]


def catalogue(ctx: ToolContext) -> CatalogueOut:
    views = (car_view(car_id, ctx) for car_id in ctx.catalog.cars)
    return CatalogueOut(cars=[_car(view) for view in sorted(filter(None, views), key=lambda v: (v.commercial_name, v.registration_year))])


def _representative(state: ConversationState, outcome: TurnOutcome | None) -> RepresentativeOut | None:
    if not state.rep_questions:
        return None
    latest = None
    if outcome is not None and outcome.kind == OutcomeKind.QUESTION_SENT and outcome.question_id:
        latest = LatestQuestionOut(
            question_id=outcome.question_id, topic=outcome.question_topic, already_open=outcome.already_open
        )
    return RepresentativeOut(
        questions=[
            QuestionOut(question_id=q.question_id, topic=q.topic.value, status=q.status.value, answer=q.answer)
            for q in state.rep_questions
        ],
        latest=latest,
    )
