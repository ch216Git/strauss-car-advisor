from dataclasses import FrozenInstanceError, replace

import pytest

from app.agent.state import (
    REQUIRED_DETAILS,
    ConversationState,
    ConversationStatus,
    EmployeeDetails,
    RepQuestion,
    RepTopic,
    blocking_questions,
    conversation_status,
    missing_details,
)
from app.comparison import CompanyCar, Waiver
from app.rules import Rules, load_rules


@pytest.fixture(scope="module")
def rules() -> Rules:
    return load_rules()


COMPLETE = EmployeeDetails(
    occupation="מנהל/ת מוצר", base_salary=18500, commissions=0, bonus=0, pension_rate=0.06,
    training_fund_rate=0.025, credit_points=2.75, current=CompanyCar("CAR-09", 300),
)


def test_new_conversation_is_collecting_everything(rules: Rules) -> None:
    state = ConversationState("c-1")

    assert missing_details(state, rules) == REQUIRED_DETAILS
    assert conversation_status(state, rules) == ConversationStatus.COLLECTING


def test_complete_details_are_ready(rules: Rules) -> None:
    state = ConversationState("c-1", details=COMPLETE)

    assert missing_details(state, rules) == ()
    assert conversation_status(state, rules) == ConversationStatus.READY


def test_children_needed_only_for_a_family_track_alternative(rules: Rules) -> None:
    state = ConversationState("c-1", details=COMPLETE)
    family_car = rules.family_track.cars[0]

    assert missing_details(state, rules, CompanyCar("CAR-10")) == ()
    assert missing_details(state, rules, Waiver()) == ()
    assert missing_details(state, rules, CompanyCar(family_car)) == ("eligible_children",)
    assert missing_details(replace(state, details=replace(COMPLETE, eligible_children=5)), rules, CompanyCar(family_car)) == ()


def test_unknown_current_car_question_blocks(rules: Rules) -> None:
    question = RepQuestion("Q-1", RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE, "Mazda CX-5 2019")
    waiting = ConversationState("c-1", details=replace(COMPLETE, current=None), rep_questions=(question,))

    assert blocking_questions(waiting) == (question,)
    assert conversation_status(waiting, rules) == ConversationStatus.WAITING_FOR_REPRESENTATIVE
    # Once a catalogue current car is known, the old question no longer blocks.
    resolved = replace(waiting, details=COMPLETE)
    assert blocking_questions(resolved) == ()
    assert conversation_status(resolved, rules) == ConversationStatus.READY


def test_participation_question_does_not_block(rules: Rules) -> None:
    question = RepQuestion("Q-1", RepTopic.ALTERNATIVE_PARTICIPATION, "participation for ATTO 3?", car_id="CAR-10")
    state = ConversationState("c-1", details=COMPLETE, rep_questions=(question,))

    assert conversation_status(state, rules) == ConversationStatus.READY


def test_state_is_immutable() -> None:
    state = ConversationState("c-1")

    with pytest.raises(FrozenInstanceError):
        state.details = COMPLETE  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.details.base_salary = 1  # type: ignore[misc]
