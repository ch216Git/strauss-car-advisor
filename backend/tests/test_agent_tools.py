from dataclasses import replace

import pytest

from app.agent.state import (
    ComparisonStatus,
    ConversationState,
    ConversationStatus,
    QuestionStatus,
    RepTopic,
)
from app.agent.tools import (
    Alternatives,
    CarMatches,
    ComparisonSummary,
    CurrentSituationSaved,
    DetailsSaved,
    QuestionSent,
    ToolContext,
    ToolError,
    ToolErrorCode,
    ask_representative,
    compare_alternative,
    find_car,
    list_alternatives,
    save_employee_details,
    set_current_car,
)
from app.car_search import MatchStatus
from app.catalog import lookup_vehicle
from app.comparison import CompanyCar, Waiver


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


def ok(outcome, result_type):
    assert isinstance(outcome.result, result_type), outcome.result
    return outcome.state, outcome.result


def error(outcome, code: ToolErrorCode) -> ToolError:
    assert isinstance(outcome.result, ToolError), outcome.result
    assert outcome.result.code == code, outcome.result
    return outcome.result


def car_id(ctx: ToolContext, commercial_name: str) -> str:
    _, matches = ok(find_car(ConversationState("x"), ctx, commercial_name), CarMatches)
    assert len(matches.cars) == 1, matches
    return matches.cars[0].car_id


# --- Workbook employees through the tools ('עובדים לדוגמה' + 'חישובים') -------
# Hand-copied from the workbook, as the employee would say it (percent, commercial names).
E001 = dict(occupation="מנהל/ת מוצר", base_salary=18500, commissions=0, bonus=0, pension_percent=6,
            training_fund_percent=2.5, credit_points=2.75, eligible_children=2)
E002 = dict(occupation="נציג/ת מכירות שטח", base_salary=9800, commissions=2400, bonus=0, pension_percent=6,
            training_fund_percent=0, credit_points=2.25, eligible_children=5)
E005 = dict(occupation="מנהל/ת פיתוח", base_salary=27000, commissions=0, bonus=0, pension_percent=7,
            training_fund_percent=2.5, credit_points=2.75, eligible_children=1)


def employee_state(ctx: ToolContext, details: dict, current_car: str, participation: float) -> ConversationState:
    state, _ = ok(save_employee_details(ConversationState("c-1"), ctx, **details), DetailsSaved)
    state, _ = ok(set_current_car(state, ctx, car_id=car_id(ctx, current_car), participation=participation), CurrentSituationSaved)
    return state


@pytest.mark.parametrize(
    "details, current_car, participation, alternative, today, after, difference",
    [
        (E001, "COROLLA CROSS", 300, "ATTO 3", 11032, 11693, 661),
        (E002, "NIRO", 250, "CARNIVAL", 8600, 5078, -3522),
        (E005, "MAZDA 3", 350, "WAIVER", 14244, 17806, 3562),
    ],
)
def test_workbook_employee_through_tools(ctx, details, current_car, participation, alternative, today, after, difference) -> None:
    state = employee_state(ctx, details, current_car, participation)
    arguments = {"waiver": True} if alternative == "WAIVER" else {"car_id": car_id(ctx, alternative)}

    _, summary = ok(compare_alternative(state, ctx, **arguments), ComparisonSummary)

    assert (summary.current_net, summary.alternative_net, summary.net_difference) == (today, after, difference)


@pytest.fixture(scope="module")
def e001(ctx: ToolContext) -> ConversationState:
    return employee_state(ctx, E001, "COROLLA CROSS", 300)


# --- save_employee_details ----------------------------------------------------

def test_save_converts_percent_and_reports_missing(ctx: ToolContext) -> None:
    start = ConversationState("c-1")

    state, saved = ok(save_employee_details(start, ctx, base_salary=18500, pension_percent=6), DetailsSaved)

    assert saved.saved == ("base_salary", "pension_rate")
    assert (state.details.base_salary, state.details.pension_rate) == (18500, 0.06)
    assert "occupation" in saved.missing and "base_salary" not in saved.missing
    assert saved.status == ConversationStatus.COLLECTING
    assert start.details.base_salary is None  # the original state is untouched


def test_none_means_not_given(ctx: ToolContext) -> None:
    state, saved = ok(save_employee_details(ConversationState("c-1"), ctx, base_salary=None, bonus=0), DetailsSaved)

    assert saved.saved == ("bonus",)
    assert (state.details.base_salary, state.details.bonus) == (None, 0)


def test_unknown_occupation_is_refused(ctx: ToolContext) -> None:
    outcome = save_employee_details(ConversationState("c-1"), ctx, occupation="אסטרונאוט/ית")

    error(outcome, ToolErrorCode.UNKNOWN_OCCUPATION)


@pytest.mark.parametrize(
    "arguments",
    [
        {"base_salary": -1},
        {"base_salary": "18500"},
        {"bonus": True},
        {"pension_percent": 106},
        {"eligible_children": 2.5},
        {"salary": 18500},
    ],
)
def test_invalid_details_save_nothing(ctx: ToolContext, arguments) -> None:
    start = ConversationState("c-1")

    outcome = save_employee_details(start, ctx, credit_points=2.25, **arguments)

    assert isinstance(outcome.result, ToolError)
    assert outcome.state == start  # all-or-nothing: the valid credit_points was not saved either


def test_changing_details_drops_the_last_comparison(ctx: ToolContext, e001: ConversationState) -> None:
    compared, _ = ok(compare_alternative(e001, ctx, car_id=car_id(ctx, "ATTO 3")), ComparisonSummary)

    changed, _ = ok(save_employee_details(compared, ctx, bonus=1000), DetailsSaved)

    assert compared.last_comparison is not None
    assert changed.last_comparison is None


# --- find_car -------------------------------------------------------------------

def test_find_car_by_model_returns_csv_names_and_step1_use_value(ctx: ToolContext) -> None:
    _, matches = ok(find_car(ConversationState("x"), ctx, "corolla cross 2025"), CarMatches)

    assert matches.found and len(matches.cars) == 1
    car = matches.cars[0]
    record = lookup_vehicle(ctx.catalog.cars[car.car_id], ctx.repository, ctx.lookup)
    assert (car.commercial_name, car.manufacturer, car.use_value) == (record.commercial_name, record.manufacturer, record.use_value)


def test_find_car_returns_every_match_for_an_ambiguous_query(ctx: ToolContext) -> None:
    _, matches = ok(find_car(ConversationState("x"), ctx, "קיה"), CarMatches)

    assert len(matches.cars) > 1
    assert all("קיה" in car.manufacturer for car in matches.cars)


@pytest.mark.parametrize(
    "query, commercial_name",
    [("קורולה קרוס", "COROLLA CROSS"), ("BYD אטו 3", "ATTO 3"), ("טסלה מודל 3", "MODEL 3"), ("ראב 4", "RAV4")],
)
def test_find_car_identifies_hebrew_and_english_input(ctx: ToolContext, query: str, commercial_name: str) -> None:
    _, matches = ok(find_car(ConversationState("x"), ctx, query), CarMatches)

    assert matches.status == MatchStatus.IDENTIFIED
    assert [car.commercial_name for car in matches.cars] == [commercial_name]


def test_find_car_returns_candidates_for_ambiguous_hebrew(ctx: ToolContext) -> None:
    _, matches = ok(find_car(ConversationState("x"), ctx, "קורולה"), CarMatches)

    assert matches.status == MatchStatus.CANDIDATES
    assert [car.commercial_name for car in matches.cars] == ["COROLLA", "COROLLA CROSS"]


def test_find_car_reports_unrecognized_words(ctx: ToolContext) -> None:
    _, matches = ok(find_car(ConversationState("x"), ctx, "מאזדה CX-5"), CarMatches)

    assert matches.status == MatchStatus.CANDIDATES
    assert matches.unrecognized_words == ("cx", "5")


def test_loosely_written_car_id_is_normalized(ctx: ToolContext) -> None:
    state, saved = ok(set_current_car(ConversationState("c-1"), ctx, car_id="car-09", participation=300), CurrentSituationSaved)

    assert state.details.current == CompanyCar("CAR-09", 300)


def test_hebrew_names_through_the_whole_flow(ctx: ToolContext) -> None:
    # E-001 as the employee would write it in Hebrew: no English model names needed.
    state, _ = ok(save_employee_details(ConversationState("c-1"), ctx, **E001), DetailsSaved)
    _, current = ok(find_car(state, ctx, "קורולה קרוס"), CarMatches)
    _, alternative = ok(find_car(state, ctx, "האטו 3"), CarMatches)
    state, _ = ok(set_current_car(state, ctx, car_id=current.cars[0].car_id, participation=300), CurrentSituationSaved)

    _, summary = ok(compare_alternative(state, ctx, car_id=alternative.cars[0].car_id), ComparisonSummary)

    assert (summary.current_net, summary.alternative_net, summary.net_difference) == (11032, 11693, 661)


def test_car_not_found_keeps_the_representative_flow(ctx: ToolContext) -> None:
    state, _ = ok(save_employee_details(ConversationState("c-1"), ctx, **E001), DetailsSaved)
    _, matches = ok(find_car(state, ctx, "מאזדה CX-5 2019"), CarMatches)
    assert matches.status == MatchStatus.NOT_FOUND

    state, sent = ok(
        ask_representative(state, ctx, topic="current_car_not_in_catalogue", question="מאזדה CX-5 2019"), QuestionSent
    )

    assert sent.status == ConversationStatus.WAITING_FOR_REPRESENTATIVE
    error(compare_alternative(state, ctx, car_id=car_id(ctx, "ATTO 3")), ToolErrorCode.BLOCKED_BY_REPRESENTATIVE)


def test_find_car_year_must_match(ctx: ToolContext) -> None:
    _, matches = ok(find_car(ConversationState("x"), ctx, "corolla cross 2019"), CarMatches)

    assert (matches.found, matches.cars) == (False, ())


def test_car_outside_catalogue_is_not_found(ctx: ToolContext) -> None:
    _, matches = ok(find_car(ConversationState("x"), ctx, "פרארי"), CarMatches)

    assert not matches.found


def test_empty_query_is_refused(ctx: ToolContext) -> None:
    error(find_car(ConversationState("x"), ctx, "  -- "), ToolErrorCode.INVALID_ARGUMENT)


# --- set_current_car --------------------------------------------------------------

def test_set_current_car(ctx: ToolContext) -> None:
    state, saved = ok(set_current_car(ConversationState("c-1"), ctx, car_id="CAR-09", participation=300), CurrentSituationSaved)

    assert state.details.current == saved.current == CompanyCar("CAR-09", 300)


def test_current_car_participation_is_never_defaulted(ctx: ToolContext) -> None:
    start = ConversationState("c-1")

    err = error(set_current_car(start, ctx, car_id="CAR-09"), ToolErrorCode.MISSING_DETAILS)

    assert err.missing == ("current.participation",)


def test_current_car_outside_catalogue_is_refused(ctx: ToolContext) -> None:
    start = ConversationState("c-1")
    outcome = set_current_car(start, ctx, car_id="CX-5", participation=300)

    error(outcome, ToolErrorCode.NOT_IN_CATALOGUE)
    assert outcome.state == start


def test_set_current_waiver(ctx: ToolContext) -> None:
    state, saved = ok(set_current_car(ConversationState("c-1"), ctx, waiver=True), CurrentSituationSaved)

    assert state.details.current == Waiver()


@pytest.mark.parametrize("arguments", [{}, {"car_id": "CAR-09", "participation": 300, "waiver": True}])
def test_current_needs_exactly_one_of_car_or_waiver(ctx: ToolContext, arguments) -> None:
    error(set_current_car(ConversationState("c-1"), ctx, **arguments), ToolErrorCode.INVALID_ARGUMENT)


# --- list_alternatives -----------------------------------------------------------

def test_alternatives_need_the_occupation(ctx: ToolContext) -> None:
    err = error(list_alternatives(ConversationState("c-1"), ctx), ToolErrorCode.MISSING_DETAILS)

    assert err.missing == ("occupation",)


def test_alternatives_respect_group_and_mark_participation_as_estimate(ctx: ToolContext, e001: ConversationState) -> None:
    _, alternatives = ok(list_alternatives(e001, ctx), Alternatives)
    order = ctx.rules.group_order
    own_group = ctx.catalog.occupations[e001.details.occupation].group

    assert alternatives.cars
    assert all(order.index(option.car.group) <= order.index(own_group) for option in alternatives.cars)
    assert all(option.participation_is_estimate for option in alternatives.cars)
    assert all(
        option.default_participation == ctx.catalog.cars[option.car.car_id].default_participation for option in alternatives.cars
    )


def test_alternatives_include_waiver_amount_from_occupation(ctx: ToolContext, e001: ConversationState) -> None:
    _, alternatives = ok(list_alternatives(e001, ctx), Alternatives)

    assert alternatives.waiver is not None
    assert alternatives.waiver.maintenance == ctx.catalog.occupations[e001.details.occupation].waiver_maintenance


def test_no_waiver_for_a_mandatory_car_occupation(ctx: ToolContext) -> None:
    state, _ = ok(save_employee_details(ConversationState("c-1"), ctx, occupation=E002["occupation"]), DetailsSaved)

    _, alternatives = ok(list_alternatives(state, ctx), Alternatives)

    assert alternatives.waiver is None


def test_family_cars_need_a_known_number_of_children(ctx: ToolContext) -> None:
    unknown, _ = ok(save_employee_details(ConversationState("c-1"), ctx, occupation=E002["occupation"]), DetailsSaved)
    known, _ = ok(save_employee_details(unknown, ctx, eligible_children=5), DetailsSaved)
    family = set(ctx.rules.family_track.cars)

    _, without_children = ok(list_alternatives(unknown, ctx), Alternatives)
    _, with_children = ok(list_alternatives(known, ctx), Alternatives)

    assert without_children.family_track_needs_children
    assert not family & {option.car.car_id for option in without_children.cars}
    assert not with_children.family_track_needs_children
    assert family <= {option.car.car_id for option in with_children.cars}


# --- compare_alternative -----------------------------------------------------------

def test_default_participation_gives_an_estimate(ctx: ToolContext, e001: ConversationState) -> None:
    state, summary = ok(compare_alternative(e001, ctx, car_id=car_id(ctx, "ATTO 3")), ComparisonSummary)

    assert summary.status == ComparisonStatus.ESTIMATE
    assert summary.estimated_fields == ("alternative.participation",)
    assert summary.result.alternative_situation.participation_source == "catalogue_default"
    assert state.last_comparison.status == ComparisonStatus.ESTIMATE


def test_stated_participation_gives_a_confirmed_result(ctx: ToolContext, e001: ConversationState) -> None:
    _, summary = ok(compare_alternative(e001, ctx, car_id=car_id(ctx, "ATTO 3"), participation=320), ComparisonSummary)

    assert summary.status == ComparisonStatus.CONFIRMED
    assert summary.estimated_fields == ()
    assert summary.result.alternative.participation == 320


def test_waiver_comparison_is_confirmed(ctx: ToolContext, e001: ConversationState) -> None:
    _, summary = ok(compare_alternative(e001, ctx, waiver=True), ComparisonSummary)

    assert summary.status == ComparisonStatus.CONFIRMED


def test_comparison_is_stored_in_state(ctx: ToolContext, e001: ConversationState) -> None:
    atto = car_id(ctx, "ATTO 3")

    state, summary = ok(compare_alternative(e001, ctx, car_id=atto), ComparisonSummary)

    assert state.alternative == CompanyCar(atto)
    assert state.last_comparison.result == summary.result
    assert e001.last_comparison is None


def test_missing_details_are_listed_not_guessed(ctx: ToolContext) -> None:
    state, _ = ok(save_employee_details(ConversationState("c-1"), ctx, occupation=E001["occupation"], base_salary=18500), DetailsSaved)

    err = error(compare_alternative(state, ctx, car_id=car_id(ctx, "ATTO 3")), ToolErrorCode.MISSING_DETAILS)

    assert set(err.missing) == {"commissions", "bonus", "pension_rate", "training_fund_rate", "credit_points", "current"}


def test_family_car_with_unknown_children_asks_for_them(ctx: ToolContext, e001: ConversationState) -> None:
    unknown_children = replace(e001, details=replace(e001.details, eligible_children=None))

    err = error(compare_alternative(unknown_children, ctx, car_id=car_id(ctx, "CARNIVAL")), ToolErrorCode.MISSING_DETAILS)

    assert err.missing == ("eligible_children",)


def test_ineligible_alternative_gives_the_reason(ctx: ToolContext, e001: ConversationState) -> None:
    outcome = compare_alternative(e001, ctx, car_id=car_id(ctx, "MODEL 3"))  # group ד; E-001 is group ג

    err = error(outcome, ToolErrorCode.NOT_ELIGIBLE)
    assert err.reason == "group_too_high"
    assert outcome.state == e001


def test_unknown_current_car_blocks_the_comparison(ctx: ToolContext) -> None:
    state, _ = ok(save_employee_details(ConversationState("c-1"), ctx, **E001), DetailsSaved)
    state, sent = ok(
        ask_representative(state, ctx, topic="current_car_not_in_catalogue", question="מאזדה CX-5 2019, השתתפות 300"),
        QuestionSent,
    )

    err = error(compare_alternative(state, ctx, car_id=car_id(ctx, "ATTO 3")), ToolErrorCode.BLOCKED_BY_REPRESENTATIVE)

    assert err.question_ids == (sent.question.question_id,)
    assert sent.status == ConversationStatus.WAITING_FOR_REPRESENTATIVE


def test_current_waiver_without_an_amount_cannot_be_calculated(ctx: ToolContext) -> None:
    # Known limitation: a mandatory-car occupation has no maintenance amount for a current waiver.
    details = {**E002, "eligible_children": 0}
    state, _ = ok(save_employee_details(ConversationState("c-1"), ctx, **details), DetailsSaved)
    state, _ = ok(set_current_car(state, ctx, waiver=True), CurrentSituationSaved)

    error(compare_alternative(state, ctx, car_id=car_id(ctx, "NIRO")), ToolErrorCode.CANNOT_CALCULATE)


@pytest.mark.parametrize(
    "arguments, code",
    [
        ({}, ToolErrorCode.INVALID_ARGUMENT),
        ({"car_id": "CAR-10", "waiver": True}, ToolErrorCode.INVALID_ARGUMENT),
        ({"waiver": True, "participation": 300}, ToolErrorCode.INVALID_ARGUMENT),
        ({"car_id": "CAR-10", "participation": -1}, ToolErrorCode.INVALID_ARGUMENT),
        ({"car_id": "CX-5"}, ToolErrorCode.NOT_IN_CATALOGUE),
    ],
)
def test_invalid_comparison_arguments(ctx: ToolContext, e001: ConversationState, arguments, code) -> None:
    error(compare_alternative(e001, ctx, **arguments), code)


# --- ask_representative ---------------------------------------------------------------

def test_participation_question_is_recorded(ctx: ToolContext, e001: ConversationState) -> None:
    atto = car_id(ctx, "ATTO 3")

    state, sent = ok(
        ask_representative(e001, ctx, topic="alternative_participation", question="מה ההשתתפות ב-ATTO 3?", car_id=atto),
        QuestionSent,
    )

    assert sent.question in state.rep_questions
    assert (sent.question.question_id, sent.question.topic, sent.question.status) == (
        "Q-1", RepTopic.ALTERNATIVE_PARTICIPATION, QuestionStatus.OPEN
    )
    assert sent.question.car_id == atto
    assert not sent.already_open
    assert sent.status == ConversationStatus.READY  # a participation question does not block
    assert e001.rep_questions == ()


def test_same_open_question_is_not_duplicated(ctx: ToolContext, e001: ConversationState) -> None:
    atto = car_id(ctx, "ATTO 3")
    state, first = ok(ask_representative(e001, ctx, topic="alternative_participation", question="?", car_id=atto), QuestionSent)

    state, again = ok(ask_representative(state, ctx, topic="alternative_participation", question="שוב?", car_id=atto), QuestionSent)
    state, other_car = ok(
        ask_representative(state, ctx, topic="alternative_participation", question="?", car_id=car_id(ctx, "EV3")), QuestionSent
    )

    assert again.already_open and again.question == first.question
    assert other_car.question.question_id == "Q-2"
    assert len(state.rep_questions) == 2


@pytest.mark.parametrize(
    "arguments",
    [
        {"topic": "salary", "question": "?"},
        {"topic": "other", "question": "   "},
        {"topic": "alternative_participation", "question": "?"},
        {"topic": "alternative_participation", "question": "?", "car_id": "CX-5"},
    ],
)
def test_invalid_questions_are_refused(ctx: ToolContext, e001: ConversationState, arguments) -> None:
    outcome = ask_representative(e001, ctx, **arguments)

    error(outcome, ToolErrorCode.INVALID_ARGUMENT)
    assert outcome.state == e001
