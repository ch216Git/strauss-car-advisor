"""The transport representative: the "no grounded answer" prompt rule, answering questions, and the
answers flowing back into the conversation. Offline: scripted providers only."""

import itertools
import time

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import TurnStop, run_turn
from app.agent.llm import Message, ToolCall
from app.agent.output_check import unsupported_numbers
from app.agent.prompt import SYSTEM_PROMPT, system_prompt
from app.agent.representative import AnswerError, answer_question
from app.agent.snapshot import decode_token, encode_token
from app.agent.state import ComparisonStatus, ConversationState, QuestionStatus, RepTopic, conversation_status
from app.agent.tools import ToolContext, ask_representative, compare_alternative
from app.api.main import create_app
from app.api.sessions import SessionStore
from app.comparison import CompanyCar

_ids = itertools.count()


class Strict:
    name = "strict"

    def __init__(self, *steps: Message) -> None:
        self._steps, self.systems = list(steps), []

    def add(self, *steps: Message) -> None:
        self._steps.extend(steps)

    def complete(self, system, messages, tools) -> Message:
        self.systems.append(system)
        assert self._steps, "the agent asked the provider for more replies than scripted"
        return self._steps.pop(0)


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"r-{next(_ids)}", n, a) for n, a in name_and_arguments])


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


CURRENT = ("set_current_car", {"car": "קורולה קרוס", "participation": 300})
ATTO = ("compare_alternative", {"car": "אטו 3"})
E001 = (
    ("שלום, אני מנהלת מוצר ויש לי היום קורולה קרוס עם השתתפות של 300 ש״ח. כמה ישתנה לי הנטו אם אעבור לאטו 3?",
     {"occupation": "מנהלת מוצר"}),
    ("השכר שלי 18,500 ברוטו, אין לי עמלות ובלי בונוס.", {"base_salary": 18500, "commissions": 0, "bonus": 0}),
    ("יש לי 2.75 נקודות זיכוי, פנסיה 6% וקרן השתלמות 2.5%.", {"credit_points": 2.75, "pension_percent": 6, "training_fund_percent": 2.5}),
)
PARTICIPATION_QUESTION = ("ask_representative", {"topic": "alternative_participation", "question": "מה ההשתתפות ב-ATTO 3?", "car_id": "CAR-10"})


def e001_estimate(ctx: ToolContext) -> ConversationState:
    state = ConversationState("c-1")
    provider = Strict(*(calls(("save_employee_details", d), CURRENT, ATTO) for _, d in E001))
    for text, _ in E001:
        result = run_turn(state, text, provider, ctx)
        state = result.state
    assert state.last_comparison.status == ComparisonStatus.ESTIMATE
    return state


def with_participation_question(ctx: ToolContext) -> ConversationState:
    result = run_turn(e001_estimate(ctx), "תשאל את הנציג מה ההשתתפות באטו 3", Strict(calls(PARTICIPATION_QUESTION)), ctx)
    assert result.stop == TurnStop.RENDERED and result.state.rep_questions[0].status == QuestionStatus.OPEN
    return result.state


# --- The prompt rule -----------------------------------------------------------------------------

def test_the_prompt_says_to_offer_the_representative_instead_of_guessing() -> None:
    rule = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith("8."))
    assert "אל תנחש/י ואל תמציא/י" in rule and "נציג התחבורה" in rule
    assert "ask_representative" in rule and "topic=other" in rule


def test_without_answers_the_system_prompt_is_unchanged(ctx: ToolContext) -> None:
    assert system_prompt(with_participation_question(ctx), ctx) == SYSTEM_PROMPT


def test_an_unanswerable_question_goes_to_the_representative_as_other(ctx: ToolContext) -> None:
    question = ("ask_representative", {"topic": "other", "question": "האם אפשר להחליף רכב באמצע הליסינג?"})
    result = run_turn(ConversationState("c-1"), "כן, תעביר/י לנציג", Strict(calls(question)), ctx)

    assert result.stop == TurnStop.RENDERED and result.reply == "שלחתי את השאלה לנציג התחבורה. נמשיך כשתגיע תשובה."
    assert (result.state.rep_questions[0].topic, result.state.rep_questions[0].status) == (RepTopic.OTHER, QuestionStatus.OPEN)


# --- Answering ---------------------------------------------------------------------------------

def test_a_confirmed_participation_turns_the_estimate_into_a_confirmed_comparison(ctx: ToolContext) -> None:
    state = answer_question(with_participation_question(ctx), ctx, "Q-1", "אושר מול הספק", participation=450)
    provider = Strict(calls(("compare_alternative", {"car_id": "CAR-10"})))

    result = run_turn(state, "קיבלתי תשובה מהנציג, אפשר לעדכן?", provider, ctx)

    assert result.stop == TurnStop.RENDERED
    comparison = result.state.last_comparison
    assert (comparison.status, comparison.estimated_fields) == (ComparisonStatus.CONFIRMED, ())
    assert comparison.result.alternative_situation.participation == 450
    assert "הערכה" not in result.reply and "הפרש:" in result.reply
    assert "450" in provider.systems[0] and "אושר מול הספק" in provider.systems[0]  # the model knows the answer


def test_an_explicit_employee_participation_still_wins(ctx: ToolContext) -> None:
    state = answer_question(with_participation_question(ctx), ctx, "Q-1", participation=450)
    result = compare_alternative(state, ctx, car_id="CAR-10", participation=300).result
    assert result.result.alternative_situation.participation == 300


def test_the_representatives_numbers_may_be_quoted(ctx: ToolContext) -> None:
    state = answer_question(with_participation_question(ctx), ctx, "Q-1", "ההשתתפות היא 450 ₪, החל מ-1.12", participation=450)
    assert unsupported_numbers("לפי הנציג ההשתתפות היא 450 ₪", state) == ()
    assert unsupported_numbers("ההשתתפות היא 999 ₪", state) == (999,)


def test_a_current_car_answer_unblocks_the_comparison(ctx: ToolContext) -> None:
    state = ConversationState("c-1")
    state = ask_representative(state, ctx, "current_car_not_in_catalogue", "הרכב של העובדת: סקודה סופרב 2018").state
    assert compare_alternative(state, ctx, car_id="CAR-10").result.code == "blocked_by_representative"

    state = answer_question(state, ctx, "Q-1", "מקביל לקורולה קרוס", participation=300, car_id="car-09")

    assert state.details.current == CompanyCar("CAR-09", 300)
    assert conversation_status(state, ctx.rules).value == "collecting"  # no longer waiting
    assert state.rep_questions[0].car_id == "CAR-09"
    assert "הרכב של היום נשמר כ-COROLLA CROSS" in system_prompt(state, ctx)


@pytest.mark.parametrize(
    "topic, kwargs, code, fields",
    [
        ("alternative_participation", {"answer": "בערך 400"}, "invalid_answer", ("participation",)),
        ("alternative_participation", {"participation": -1}, "invalid_answer", ("participation",)),
        ("alternative_participation", {"participation": 400, "car_id": "CAR-09"}, "invalid_answer", ("car_id",)),
        ("current_car_not_in_catalogue", {"car_id": "CAR-09"}, "invalid_answer", ("participation",)),
        ("current_car_not_in_catalogue", {"participation": 300}, "invalid_answer", ("car_id",)),
        ("current_car_not_in_catalogue", {"car_id": "CAR-999", "participation": 300}, "invalid_answer", ("car_id",)),
        ("current_car_not_in_catalogue", {}, "invalid_answer", ("answer",)),
        ("other", {"answer": "  "}, "invalid_answer", ("answer",)),
        ("other", {"answer": "כן", "participation": 1}, "invalid_answer", ("participation",)),
    ],
)
def test_answers_are_checked_against_the_topic(ctx: ToolContext, topic, kwargs, code, fields) -> None:
    state = ask_representative(ConversationState("c-1"), ctx, topic, "שאלה", car_id="CAR-10").state
    with pytest.raises(AnswerError) as raised:
        answer_question(state, ctx, "Q-1", **kwargs)
    assert (raised.value.code, raised.value.fields) == (code, fields)


def test_unknown_and_answered_questions_are_refused(ctx: ToolContext) -> None:
    state = ask_representative(ConversationState("c-1"), ctx, "other", "שאלה").state
    with pytest.raises(AnswerError, match="question_not_found"):
        answer_question(state, ctx, "Q-9", "כן")
    state = answer_question(state, ctx, "Q-1", "כן")
    with pytest.raises(AnswerError, match="already_answered"):
        answer_question(state, ctx, "Q-1", "לא")


def test_a_resume_snapshot_keeps_the_confirmed_participation(ctx: ToolContext) -> None:
    state = answer_question(with_participation_question(ctx), ctx, "Q-1", participation=450)
    token = encode_token(state, ctx.rules.version, 4, 0, b"k" * 32)
    restored, _ = decode_token(token, b"k" * 32, ctx, ctx.rules.version, 0)
    assert restored.rep_questions[0].participation == 450
    assert restored.rep_questions[0].status == QuestionStatus.ANSWERED


# --- The API -----------------------------------------------------------------------------------

@pytest.fixture
def client(ctx: ToolContext):
    provider = Strict()
    return TestClient(create_app(provider, ctx, SessionStore(), signing_key=b"k" * 32)), provider


def say(client: TestClient, session_id: str, text: str) -> dict:
    response = client.post("/api/chat", json={"session_id": session_id, "message": text})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_representative_answers_and_the_employee_continues(client) -> None:
    client, provider = client
    session_id = client.post("/api/sessions").json()["session_id"]
    for text, details in E001:
        provider.add(calls(("save_employee_details", details), CURRENT, ATTO))
        say(client, session_id, text)
    provider.add(calls(PARTICIPATION_QUESTION))
    say(client, session_id, "תשאל את הנציג מה ההשתתפות באטו 3")

    queue = client.get("/api/representative/questions").json()["questions"]
    assert queue == [{
        "session_id": session_id, "question_id": "Q-1", "topic": "alternative_participation", "status": "open",
        "question": "מה ההשתתפות ב-ATTO 3?", "car": queue[0]["car"],
    }]
    assert queue[0]["car"]["car_id"] == "CAR-10"
    status = client.get(f"/api/sessions/{session_id}/representative").json()
    assert status.pop("resume_token")
    assert status == {"questions": [{"question_id": "Q-1", "topic": "alternative_participation", "status": "open"}]}

    answered = client.post("/api/representative/answers", json={
        "session_id": session_id, "question_id": "Q-1", "answer": "אושר", "participation": 450,
    })
    assert answered.status_code == 200
    assert (answered.json()["status"], answered.json()["participation"]) == ("answered", 450)
    status = client.get(f"/api/sessions/{session_id}/representative").json()
    assert status["questions"][0] == {"question_id": "Q-1", "topic": "alternative_participation", "status": "answered", "answer": "אושר"}
    saved, _ = decode_token(status["resume_token"], b"k" * 32, ToolContext.load(), ToolContext.load().rules.version, int(time.time()))
    assert (saved.rep_questions[0].status, saved.rep_questions[0].participation) == (QuestionStatus.ANSWERED, 450)
    assert saved.conversation_id == session_id

    provider.add(calls(("compare_alternative", {"car_id": "CAR-10"})))
    body = say(client, session_id, "קיבלתי תשובה מהנציג, אפשר לעדכן?")
    assert body["ui_state"] == "confirmed"
    assert body["comparison"]["alternative"]["participation"] == 450
    assert body["representative"]["questions"][0]["answer"] == "אושר"


def test_answer_errors(client) -> None:
    client, provider = client
    session_id = client.post("/api/sessions").json()["session_id"]
    provider.add(calls(("ask_representative", {"topic": "other", "question": "האם יש רכב 7 מקומות חשמלי?"})))
    say(client, session_id, "תעביר לנציג")

    def post(**body):
        return client.post("/api/representative/answers", json={"session_id": session_id, "question_id": "Q-1", **body})

    assert post(answer="", participation=None).json()["error"] == {
        "code": "invalid_answer", "message": "התשובה אינה מתאימה לשאלה. בדקו את השדות המסומנים.", "fields": ["answer"],
    }
    assert post(answer="x", unknown=1).status_code == 422
    assert client.post("/api/representative/answers", json={"session_id": "0" * 32, "question_id": "Q-1", "answer": "x"}).status_code == 404
    assert post(question_id="Q-7", answer="x").json()["error"]["code"] == "question_not_found"
    assert post(answer="לא כרגע").status_code == 200
    assert post(answer="שוב").status_code == 409
    assert client.get(f"/api/sessions/{'0' * 32}/representative").status_code == 404


def test_the_catalogue_for_the_representative(client) -> None:
    client, _ = client
    cars = client.get("/api/representative/cars").json()["cars"]
    assert {"CAR-09", "CAR-10"} <= {car["car_id"] for car in cars}
    assert set(cars[0]) == {"car_id", "manufacturer", "commercial_name", "registration_year", "group", "electric", "family_track"}
