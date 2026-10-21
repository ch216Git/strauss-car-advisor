"""The transport representative answers an open question (Step 5's representative screen).

Not a model tool: the representative's answer is applied by Python, checked against the question's
topic and the catalogue, and only then becomes part of the conversation:
  alternative_participation      the confirmed monthly participation for the question's car; the next
                                 compare_alternative for that car uses it (a confirmed comparison);
  current_car_not_in_catalogue   optionally, the catalogue car that stands for today's car, with today's
                                 participation; it becomes the employee's current situation;
  other                          the answer in words only.
The answer's words reach the model as data in the system instruction (app.agent.prompt).
"""

from dataclasses import replace

from app.agent.state import ConversationState, QuestionStatus, RepQuestion, RepTopic
from app.agent.tools import ToolContext, _catalogue_id
from app.comparison import CompanyCar

MAX_ANSWER_LENGTH = 1000


class AnswerError(ValueError):
    """The answer cannot be applied. code is a stable API error code; fields names what is wrong."""

    def __init__(self, code: str, fields: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code, self.fields = code, fields


def answer_question(
    state: ConversationState,
    ctx: ToolContext,
    question_id: str,
    answer: str = "",
    participation: float | None = None,
    car_id: str | None = None,
) -> ConversationState:
    """The state with question_id answered. Raises AnswerError; the state is then unchanged."""
    question = next((q for q in state.rep_questions if q.question_id == question_id), None)
    if question is None:
        raise AnswerError("question_not_found")
    if question.status != QuestionStatus.OPEN:
        raise AnswerError("already_answered")
    text = answer.strip()
    if len(text) > MAX_ANSWER_LENGTH:
        raise AnswerError("invalid_answer", ("answer",))
    if participation is not None and participation < 0:
        raise AnswerError("invalid_answer", ("participation",))

    details = state.details
    if question.topic == RepTopic.ALTERNATIVE_PARTICIPATION:
        if participation is None:
            raise AnswerError("invalid_answer", ("participation",))
        if car_id is not None:
            raise AnswerError("invalid_answer", ("car_id",))
    elif question.topic == RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE:
        if (car_id is None) != (participation is None):  # the car and today's participation come together
            raise AnswerError("invalid_answer", ("participation",) if participation is None else ("car_id",))
        if car_id is not None:
            car_id = _catalogue_id(car_id, ctx)
            if car_id not in ctx.catalog.cars:
                raise AnswerError("invalid_answer", ("car_id",))
            details = replace(details, current=CompanyCar(car_id, participation))
        elif not text:
            raise AnswerError("invalid_answer", ("answer",))
    else:
        if participation is not None or car_id is not None:
            raise AnswerError("invalid_answer", ("participation",) if participation is not None else ("car_id",))
        if not text:
            raise AnswerError("invalid_answer", ("answer",))

    answered = replace(question, status=QuestionStatus.ANSWERED, answer=text or None, participation=participation)
    if question.topic == RepTopic.CURRENT_CAR_NOT_IN_CATALOGUE:
        answered = replace(answered, car_id=car_id)
    questions = tuple(answered if q.question_id == question_id else q for q in state.rep_questions)
    return replace(state, details=details, rep_questions=questions)


def answered_questions(state: ConversationState) -> tuple[RepQuestion, ...]:
    return tuple(q for q in state.rep_questions if q.status == QuestionStatus.ANSWERED)
