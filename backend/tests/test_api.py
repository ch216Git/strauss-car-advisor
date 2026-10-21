"""The HTTP API (app.api) with scripted fake providers: no Gemini, no network, no quota."""

import itertools
import json
import re

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import PROVIDER_ERROR_REPLY
from app.agent.llm import Message, ProviderError, ProviderErrorKind, ToolCall
from app.agent.tools import ToolContext
from app.api import main as api_main
from app.api.main import create_app
from app.api.schemas import (
    AlternativeOut,
    CarChoiceOut,
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
    RepresentativeOut,
    SideOut,
    UiState,
)
from app.api.sessions import SessionStore

_ids = itertools.count()


class Scripted:
    """Returns the scripted steps in order (an exception step is raised) and records each request."""

    name = "scripted"

    def __init__(self, *steps) -> None:
        self._steps = list(steps)
        self.requests: list[tuple] = []

    def add(self, *steps) -> None:
        self._steps.extend(steps)

    def complete(self, system, messages, tools) -> Message:
        self.requests.append(tuple(messages))
        assert self._steps, "the agent asked the provider for more replies than scripted"
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class Endless:
    name = "endless"

    def complete(self, system, messages, tools) -> Message:
        return calls(("find_car", {"query": "קורולה קרוס", "purpose": "information"}))


def calls(*name_and_arguments) -> Message:
    return Message.assistant(tool_calls=[ToolCall(f"a-{next(_ids)}", n, a) for n, a in name_and_arguments])


def down() -> ProviderError:
    return ProviderError("fake", ProviderErrorKind.UNAVAILABLE, "service unavailable")


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


class Api:
    def __init__(self, ctx: ToolContext, provider) -> None:
        self.provider = provider
        self.store = SessionStore()
        self.client = TestClient(create_app(provider, ctx, self.store))
        self.responses: list[dict] = []  # every chat response body, for the allow-list check

    def start(self) -> str:
        response = self.client.post("/api/sessions")
        assert response.status_code == 201
        return response.json()["session_id"]

    def say(self, session_id: str, text: str, status: int = 200) -> dict:
        response = self.client.post("/api/chat", json={"session_id": session_id, "message": text})
        assert response.status_code == status, response.text
        body = response.json()
        self.responses.append(body)
        return body


@pytest.fixture
def api(ctx: ToolContext) -> Api:
    return Api(ctx, Scripted())


CARNIVAL = ("compare_alternative", {"car": "קרניבל"})
CURRENT = ("set_current_car", {"car": "קורולה קרוס", "participation": 300})
ATTO = ("compare_alternative", {"car": "אטו 3"})
E001 = (
    ("שלום, אני מנהלת מוצר ויש לי היום קורולה קרוס עם השתתפות של 300 ש״ח. כמה ישתנה לי הנטו אם אעבור לאטו 3?",
     calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), CURRENT, ATTO)),
    ("השכר שלי 18,500 ברוטו, אין לי עמלות ובלי בונוס.",
     calls(("save_employee_details", {"base_salary": 18500, "commissions": 0, "bonus": 0}), CURRENT, ATTO)),
    ("יש לי 2.75 נקודות זיכוי, פנסיה 6% וקרן השתלמות 2.5%.",
     calls(("save_employee_details", {"credit_points": 2.75, "pension_percent": 6, "training_fund_percent": 2.5}), CURRENT, ATTO)),
)


def run_e001(api: Api, session_id: str) -> list[dict]:
    bodies = []
    for text, step in E001:
        api.provider.add(step)
        bodies.append(api.say(session_id, text))
    return bodies


# --- Sessions and health ----------------------------------------------------------------------

def test_health_leaks_nothing_and_needs_no_provider(ctx: ToolContext, monkeypatch) -> None:
    def no_provider():
        raise AssertionError("health must not build the provider")

    monkeypatch.setattr(api_main, "_gemini", no_provider)
    client = TestClient(create_app(None, ctx))

    response = client.get("/api/health")

    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_new_session_is_idle(api: Api) -> None:
    response = api.client.post("/api/sessions")

    assert response.status_code == 201
    body = response.json()
    assert body["ui_state"] == "idle" and re.fullmatch(r"[0-9a-f]{32}", body["session_id"])
    assert set(body) == {"session_id", "ui_state"}


def test_reset_deletes_the_old_conversation(api: Api) -> None:
    old = api.start()
    api.provider.add(Message.assistant("שלום"))
    api.say(old, "שלום")

    assert api.client.delete(f"/api/sessions/{old}").status_code == 204
    assert api.client.delete(f"/api/sessions/{old}").status_code == 204  # already gone: still fine
    new = api.start()

    assert new != old
    assert api.say(old, "שלום", status=404)["error"]["code"] == "session_not_found"


def test_unknown_session_is_404(api: Api) -> None:
    body = api.say("0" * 32, "שלום", status=404)

    assert body == {"error": {"code": "session_not_found", "message": api_main.MESSAGES["session_not_found"]}}
    assert api.provider.requests == []


def test_concurrent_turn_is_409(api: Api) -> None:
    session_id = api.start()
    session = api.store.get(session_id)
    session.lock.acquire()  # a turn is running
    try:
        body = api.say(session_id, "שלום", status=409)
    finally:
        session.lock.release()

    assert body["error"]["code"] == "turn_in_progress"
    assert api.provider.requests == [] and session.turn == 0


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"session_id": "a" * 32, "message": ""}, "message"),
        ({"session_id": "a" * 32, "message": "   "}, "message"),
        ({"session_id": "a" * 32, "message": "א" * 2001}, "message"),
        ({"session_id": "not-a-session", "message": "שלום"}, "session_id"),
        ({"session_id": "a" * 32}, "message"),
        ({"session_id": "a" * 32, "message": "שלום", "state": {}}, "state"),
    ],
)
def test_validation_is_422_without_echo(api: Api, payload: dict, field: str) -> None:
    response = api.client.post("/api/chat", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_request" and field in body["error"]["fields"]
    assert "שלום" not in response.text and "input" not in body["error"]


def test_session_store_expires_and_caps() -> None:
    now = [0.0]
    store = SessionStore(ttl_seconds=10, max_sessions=2, clock=lambda: now[0])
    first = store.create()
    now[0] = 5
    second = store.create()
    now[0] = 8
    third = store.create()  # the cap drops the least recently used

    assert store.get(first.session_id) is None and len(store) == 2
    now[0] = 16
    assert store.get(second.session_id) is None  # idle for 11 > 10
    assert store.get(third.session_id) is third


# --- Screen states ----------------------------------------------------------------------------

def test_carnival_eligibility_question(api: Api, ctx: ToolContext) -> None:
    session_id = api.start()
    api.provider.add(calls(CARNIVAL))

    body = api.say(session_id, "אפשר לקבל קרניבל?")

    assert body["ui_state"] == "eligibility_check" and body["turn"] == 1
    assert body["eligibility"] == {"missing": ["eligible_children"]}
    car = body["alternative"]["car"]
    assert body["alternative"]["kind"] == "company_car"
    assert car["family_track"] is True and car["car_id"] in ctx.rules.family_track.cars
    assert car["commercial_name"] == "CARNIVAL"
    assert body["reply"].startswith("כדי לבדוק אם החלופה הזו אפשרית עבורך")
    for absent in ("comparison", "details", "representative", "car_choice", "current"):
        assert absent not in body


def test_carnival_not_eligible(api: Api, ctx: ToolContext) -> None:
    session_id = api.start()
    api.provider.add(calls(CARNIVAL), calls(("save_employee_details", {"eligible_children": 3}), CARNIVAL))
    api.say(session_id, "אפשר לקבל קרניבל?")

    body = api.say(session_id, "יש לי 3 ילדים")

    rule = ctx.rules.family_track
    car_id = body["alternative"]["car"]["car_id"]
    assert body["ui_state"] == "not_eligible" and body["turn"] == 2
    assert body["eligibility"] == {
        "missing": [],
        "reason": "not_enough_children",
        "facts": {
            "car_group": ctx.catalog.cars[car_id].group,  # the catalogue's, as the backend gave it
            "min_children": rule.min_children, "child_max_age": rule.child_max_age, "eligible_children": 3,
        },
    }
    assert body["alternative"]["car"]["commercial_name"] == "CARNIVAL"
    assert "comparison" not in body  # no financial result for Carnival


def test_eligibility_check_is_generic(api: Api) -> None:
    """The occupation decides here, not children: the same state, another field."""
    session_id = api.start()
    api.provider.add(calls(("compare_alternative", {"car": "טסלה מודל 3"})))

    body = api.say(session_id, "אני רוצה טסלה מודל 3")

    assert (body["ui_state"], body["eligibility"]) == ("eligibility_check", {"missing": ["occupation"]})


def test_carnival_with_enough_children_collects_calculation_details(api: Api) -> None:
    session_id = api.start()
    api.provider.add(calls(CARNIVAL), calls(("save_employee_details", {"eligible_children": 5}), CARNIVAL))
    api.say(session_id, "אפשר לקבל קרניבל?")

    body = api.say(session_id, "יש לי 5 ילדים")

    assert body["ui_state"] == "collecting"
    assert "occupation" in body["details"]["missing"] and "base_salary" in body["details"]["missing"]
    assert body["details"]["provided"] == ["eligible_children"]
    assert "comparison" not in body and "eligibility" not in body


def test_calculation_details_collection(api: Api) -> None:
    session_id = api.start()
    first, second, _ = run_e001(api, session_id)

    assert first["ui_state"] == second["ui_state"] == "collecting"
    assert first["details"]["missing"] == [
        "base_salary", "commissions", "bonus", "pension_rate", "training_fund_rate", "credit_points"
    ]
    assert set(first["details"]["provided"]) == {"occupation", "current"}
    assert second["details"]["missing"] == ["pension_rate", "training_fund_rate", "credit_points"]
    assert first["current"]["commercial_name"] == "COROLLA CROSS"
    assert first["alternative"]["car"]["electric"] is True


def test_e001_estimate_payload(api: Api) -> None:
    session_id = api.start()
    body = run_e001(api, session_id)[-1]

    assert (body["ui_state"], body["turn"]) == ("estimate", 3)
    comparison = body["comparison"]
    assert comparison["status"] == "estimate"
    assert comparison["estimated_fields"] == ["alternative.participation"]
    current, alternative = comparison["current"], comparison["alternative"]
    assert (current["net"], current["use_value"]) == (11032, 4070)
    assert (current["participation"], current["participation_source"]) == (300, "stated")
    assert current["car"]["commercial_name"] == "COROLLA CROSS"
    assert (alternative["net"], alternative["use_value"]) == (11693, 3130)
    assert (alternative["participation"], alternative["participation_source"]) == (300, "catalogue_default")
    assert alternative["car"]["electric"] is True
    assert comparison["net_difference"] == 661
    assert comparison["changes"] == [{"item": "use_value", "current": 4070, "alternative": 3130}]
    assert comparison["electric_benefit"] == {"net": 250, "grossed_up": True, "applies_to": "alternative"}
    assert "הערכה" in body["reply"]  # the backend's own disclosure is passed through as written


def test_confirmed_comparison_uses_the_same_payload(api: Api) -> None:
    session_id = api.start()
    run_e001(api, session_id)
    api.provider.add(calls(("compare_alternative", {"car_id": "CAR-10", "participation": 300})))

    body = api.say(session_id, "ואטו 3 עם השתתפות 300?")

    assert body["ui_state"] == "confirmed"
    comparison = body["comparison"]
    assert (comparison["status"], comparison["estimated_fields"]) == ("confirmed", [])
    assert comparison["alternative"]["participation_source"] == "stated"
    assert (comparison["current"]["net"], comparison["alternative"]["net"], comparison["net_difference"]) == (11032, 11693, 661)


def test_representative_after_the_estimate(api: Api) -> None:
    session_id = api.start()
    run_e001(api, session_id)
    ask = ("ask_representative", {"topic": "alternative_participation", "question": "ההשתתפות ב-ATTO 3?", "car_id": "CAR-10"})
    api.provider.add(calls(ask))

    body = api.say(session_id, "תשאל את הנציג מה ההשתתפות באטו 3")

    assert body["ui_state"] == "representative"
    assert body["representative"] == {
        "questions": [{"question_id": "Q-1", "topic": "alternative_participation", "status": "open"}],
        "latest": {"question_id": "Q-1", "topic": "alternative_participation", "already_open": False},
    }
    assert body["comparison"]["status"] == "estimate"  # the last comparison stays on screen
    assert "alternative" not in body


def test_representative_when_todays_car_is_unknown(api: Api) -> None:
    session_id = api.start()
    api.provider.add(
        calls(("save_employee_details", {"occupation": "מנהלת מוצר"}), ("set_current_car", {"car": "מאזדה CX-5 2019", "participation": 300})),
        calls(("ask_representative", {"topic": "current_car_not_in_catalogue", "question": "מאזדה CX-5 2019"})),
        Message.assistant("הרכב לא בקטלוג; שאלתי את הנציג."),
        calls(ATTO),
    )
    asked = api.say(session_id, "אני מנהלת מוצר, יש לי מאזדה CX-5 2019 בהשתתפות 300")

    blocked = api.say(session_id, "ומה עם אטו 3?")

    assert asked["ui_state"] == blocked["ui_state"] == "representative"
    assert blocked["representative"]["questions"] == [
        {"question_id": "Q-1", "topic": "current_car_not_in_catalogue", "status": "open"}
    ]
    assert "latest" not in blocked["representative"] and "comparison" not in blocked
    assert blocked["alternative"]["car"]["car_id"] == "CAR-10"


def test_car_choice(api: Api) -> None:
    session_id = api.start()
    api.provider.add(calls(("set_current_car", {"car": "קורולה", "participation": 250})))

    body = api.say(session_id, "יש לי קורולה, השתתפות 250")

    assert body["ui_state"] == "car_choice"
    assert body["car_choice"]["target"] == "current"
    assert [car["commercial_name"] for car in body["car_choice"]["options"]] == ["COROLLA", "COROLLA CROSS"]
    assert "current" not in body and "alternative" not in body


def test_car_choice_via_find_car_needs_one_provider_call(api: Api) -> None:
    session_id = api.start()
    api.provider.add(calls(("find_car", {"query": "קורולה", "purpose": "alternative"})))  # no second step scripted

    body = api.say(session_id, "כמה ישתנה לי הנטו אם אעבור לקורולה?")

    assert len(api.provider.requests) == 1
    assert (body["ui_state"], body["retryable"], body["turn"]) == ("car_choice", False, 1)
    assert body["car_choice"]["target"] == "alternative"
    assert [car["commercial_name"] for car in body["car_choice"]["options"]] == ["COROLLA", "COROLLA CROSS"]


def test_answered(api: Api) -> None:
    session_id = api.start()
    api.provider.add(calls(("find_car", {"query": "טוסון", "purpose": "information"})), Message.assistant("טוסון הוא רכב בקטלוג."))

    body = api.say(session_id, "מה זה טוסון?")

    assert isinstance(body.pop("resume_token"), str)  # a committed turn carries a token; checked elsewhere
    assert body == {"session_id": session_id, "turn": 1, "reply": "טוסון הוא רכב בקטלוג.", "ui_state": "answered",
                    "retryable": False}


def test_incomplete_after_the_round_limit(ctx: ToolContext) -> None:
    api = Api(ctx, Endless())
    session_id = api.start()

    body = api.say(session_id, "יש לי קורולה קרוס")

    assert (body["ui_state"], body["turn"]) == ("incomplete", 1)


def test_incomplete_when_the_reply_was_replaced(api: Api) -> None:
    session_id = api.start()
    api.provider.add(Message.assistant("הנטו שלך יעלה ב-9,999 ₪"))  # a number no tool gave

    body = api.say(session_id, "כמה ארוויח?")

    assert body["ui_state"] == "incomplete" and "9,999" not in body["reply"]


# --- Provider failure: atomic turns -------------------------------------------------------------

def test_temporary_provider_error(api: Api) -> None:
    session_id = api.start()
    api.provider.add(Message.assistant("שלום"), down())
    api.say(session_id, "שלום")
    before = api.store.get(session_id).state

    body = api.say(session_id, "אפשר לקבל קרניבל?")

    assert body == {"session_id": session_id, "turn": 1, "reply": PROVIDER_ERROR_REPLY, "ui_state": "temporary_error",
                    "retryable": True}
    session = api.store.get(session_id)
    assert session.state is before and session.turn == 1  # nothing committed, turn not advanced


def test_retry_after_provider_error_has_clean_history(api: Api) -> None:
    session_id = api.start()
    api.provider.add(down(), calls(CARNIVAL))
    api.say(session_id, "אפשר לקבל קרניבל?")

    body = api.say(session_id, "אפשר לקבל קרניבל?")

    assert (body["ui_state"], body["turn"]) == ("eligibility_check", 1)
    retried = api.provider.requests[1]  # what the provider saw on the retry
    texts = [m.text for m in retried]
    assert texts == ["אפשר לקבל קרניבל?"]  # one copy of the message, no error reply
    history = [m.text for m in api.store.get(session_id).state.messages if m.text]
    assert PROVIDER_ERROR_REPLY not in history and history.count("אפשר לקבל קרניבל?") == 1


def test_internal_error_is_500_without_details(ctx: ToolContext) -> None:
    api = Api(ctx, Scripted(RuntimeError("secret salary 18500")))
    session_id = api.start()

    response = api.client.post("/api/chat", json={"session_id": session_id, "message": "שלום"})

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal_error", "message": api_main.MESSAGES["internal_error"]}}
    assert "18500" not in response.text and "RuntimeError" not in response.text
    assert api.store.get(session_id).turn == 0 and api.store.get(session_id).state.messages == ()


def test_advisor_unavailable_without_configuration(ctx: ToolContext, monkeypatch) -> None:
    def no_key():
        raise ProviderError("gemini", ProviderErrorKind.CONFIGURATION, "GEMINI_API_KEY is not set")

    monkeypatch.setattr(api_main, "_gemini", no_key)
    client = TestClient(create_app(None, ctx))
    session_id = client.post("/api/sessions").json()["session_id"]

    response = client.post("/api/chat", json={"session_id": session_id, "message": "שלום"})

    assert response.status_code == 503 and response.json()["error"]["code"] == "advisor_unavailable"
    assert "GEMINI" not in response.text and "gemini" not in response.text


# --- The contract ----------------------------------------------------------------------------

def test_ui_states_are_stable() -> None:
    assert [s.value for s in UiState] == [
        "idle", "eligibility_check", "not_eligible", "collecting", "estimate", "confirmed",
        "representative", "car_choice", "answered", "incomplete", "temporary_error",
    ]


ALLOWED_KEYS = set().union(*(model.model_fields for model in (
    ChatResponse, AlternativeOut, CarOut, EligibilityOut, EligibilityFactsOut, DetailsOut, SideOut, ChangeOut,
    ElectricBenefitOut, ComparisonOut, QuestionOut, LatestQuestionOut, RepresentativeOut, CarChoiceOut,
)))
TOOL_NAMES = ("save_employee_details", "find_car", "set_current_car", "list_alternatives", "compare_alternative",
              "ask_representative")
INTERNAL_WORDS = ("tool", "provider", "gemini", "stop", "timing", "seconds", "round", "debug", "call",
                  "exception", "prompt", "text", "label", "ran", "via")
INTERNAL_KEYS = {"detail", "code", "matches", "family_track_rule", "drive", "eligibility_basis", "state", "messages"}


def keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(v) for v in value)) if value else set()
    return set()


def test_responses_carry_only_the_contract(api: Api) -> None:
    """Every state's response: only declared keys, and no tool, provider or debug internals."""
    session_id = api.start()
    run_e001(api, session_id)
    api.provider.add(
        calls(("ask_representative", {"topic": "alternative_participation", "question": "ההשתתפות ב-ATTO 3?", "car_id": "CAR-10"})),
        down(),
    )
    api.say(session_id, "תשאל את הנציג מה ההשתתפות באטו 3")
    api.say(session_id, "ומה עוד?")
    other = api.start()
    api.provider.add(calls(CARNIVAL), calls(("save_employee_details", {"eligible_children": 3}), CARNIVAL),
                     calls(("set_current_car", {"car": "קורולה", "participation": 250})))
    api.say(other, "אפשר לקבל קרניבל?")
    api.say(other, "יש לי 3 ילדים")
    api.say(other, "יש לי קורולה, השתתפות 250")

    states = {body["ui_state"] for body in api.responses}
    assert states == {"collecting", "estimate", "representative", "temporary_error", "eligibility_check",
                      "not_eligible", "car_choice"}
    for body in api.responses:
        found = keys(body)
        assert found <= ALLOWED_KEYS, found - ALLOWED_KEYS
        assert not [k for k in found for word in INTERNAL_WORDS if word in k.lower()]
        assert not found & INTERNAL_KEYS
        dumped = json.dumps(body, ensure_ascii=False)
        assert not [name for name in TOOL_NAMES if name in dumped]
        assert "netfree" not in dumped.lower() and "<script" not in dumped.lower()


# --- Resume without a database (history) --------------------------------------------------------

RESUME_KEY = b"api-test-signing-key-32-bytes!!!"


class ResumableApi(Api):
    """An Api with a fixed signing key, so a token survives a simulated restart (a fresh app+store)."""

    def __init__(self, ctx: ToolContext, provider, store: SessionStore | None = None) -> None:
        self.provider = provider
        self.store = store if store is not None else SessionStore()
        self.client = TestClient(create_app(provider, ctx, self.store, signing_key=RESUME_KEY))
        self.responses = []


def test_committed_turn_carries_a_token_but_temporary_error_does_not(ctx: ToolContext) -> None:
    api = ResumableApi(ctx, Scripted())
    session_id = api.start()
    api.provider.add(Message.assistant("שלום"), down())

    ok = api.say(session_id, "שלום")
    failed = api.say(session_id, "שוב")

    assert isinstance(ok["resume_token"], str) and ok["resume_token"]
    assert "resume_token" not in failed  # a temporary error commits nothing, so no token


def test_committed_turn_sets_cache_control_no_store(ctx: ToolContext) -> None:
    api = ResumableApi(ctx, Scripted())
    session_id = api.start()
    api.provider.add(Message.assistant("שלום"))

    response = api.client.post("/api/chat", json={"session_id": session_id, "message": "שלום"})

    assert response.headers.get("cache-control") == "no-store"


def test_restore_after_a_restart_continues_the_conversation(ctx: ToolContext) -> None:
    api = ResumableApi(ctx, Scripted())
    session_id = api.start()
    token = run_e001(api, session_id)[-1]["resume_token"]
    assert token

    # A restart: a brand-new app and store, the same signing key, the old session gone.
    fresh = ResumableApi(ctx, Scripted())
    restored = fresh.client.post("/api/sessions/restore", json={"resume_token": token})

    assert restored.status_code == 201
    body = restored.json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["session_id"]) and body["session_id"] != session_id
    assert (body["turn"], body["ui_state"]) == (3, "idle")

    # The restored session continues: a confirmed comparison uses the restored, grounded details.
    fresh.provider.add(calls(("compare_alternative", {"car_id": "CAR-10", "participation": 300})))
    nxt = fresh.say(body["session_id"], "ואטו 3 עם השתתפות 300?")
    assert (nxt["ui_state"], nxt["turn"]) == ("confirmed", 4)
    assert nxt["comparison"]["alternative"]["participation_source"] == "stated"


def test_grounding_still_enforced_after_restore(ctx: ToolContext) -> None:
    api = ResumableApi(ctx, Scripted())
    session_id = api.start()
    token = run_e001(api, session_id)[-1]["resume_token"]
    fresh = ResumableApi(ctx, Scripted())
    new_id = fresh.client.post("/api/sessions/restore", json={"resume_token": token}).json()["session_id"]

    # A participation the employee never stated must be refused by the grounding guard, even post-restore:
    # the model cannot invent 999; the only stated participation is 300 (preserved in the messages).
    fresh.provider.add(
        calls(("compare_alternative", {"car_id": "CAR-10", "participation": 999})),
        Message.assistant("אפשר לאשר את סכום ההשתתפות?"),
    )
    body = fresh.say(new_id, "ואם ההשתת99?")

    assert body["ui_state"] == "answered"  # the invented number was refused; no confirmed comparison
    assert "999" not in body["reply"]


def test_restore_rejects_a_bad_token_with_410(ctx: ToolContext) -> None:
    api = ResumableApi(ctx, Scripted())

    response = api.client.post("/api/sessions/restore", json={"resume_token": "not-a-real-token"})

    assert response.status_code == 410
    assert response.json() == {"error": {"code": "resume_unavailable", "message": api_main.MESSAGES["resume_unavailable"]}}


def test_restore_rejects_a_token_from_a_different_key(ctx: ToolContext) -> None:
    api = ResumableApi(ctx, Scripted())
    session_id = api.start()
    api.provider.add(Message.assistant("שלום"))
    token = api.say(session_id, "שלום")["resume_token"]

    other = ResumableApi(ctx, Scripted(), store=SessionStore())
    other.client = TestClient(create_app(Scripted(), ctx, other.store, signing_key=b"a-completely-different-key-32!!!"))
    response = other.client.post("/api/sessions/restore", json={"resume_token": token})

    assert response.status_code == 410 and response.json()["error"]["code"] == "resume_unavailable"


def test_oversized_resume_token_is_422(ctx: ToolContext) -> None:
    api = ResumableApi(ctx, Scripted())

    response = api.client.post("/api/sessions/restore", json={"resume_token": "x" * (64 * 1024 + 1)})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_signing_key_is_read_after_env_files_are_loaded(ctx: ToolContext, monkeypatch, tmp_path) -> None:
    """A key in backend/.env must be seen: history_signing_key loads .env before reading the variable."""
    import app.env_file as env_file

    env_path = tmp_path / ".env"
    env_path.write_text("HISTORY_SIGNING_KEY=from-the-env-file\n", encoding="utf-8")
    monkeypatch.setattr(env_file, "DEFAULT_ENV_FILES", (env_path,))
    monkeypatch.delenv("HISTORY_SIGNING_KEY", raising=False)

    key = api_main.history_signing_key()

    assert key == b"from-the-env-file"


def test_missing_signing_key_warns_and_is_ephemeral(ctx: ToolContext, monkeypatch, tmp_path, caplog) -> None:
    import app.env_file as env_file

    monkeypatch.setattr(env_file, "DEFAULT_ENV_FILES", (tmp_path / "absent.env",))
    monkeypatch.delenv("HISTORY_SIGNING_KEY", raising=False)

    with caplog.at_level("WARNING"):
        first = api_main.history_signing_key()
        second = api_main.history_signing_key()

    assert first != second and len(first) == 32  # ephemeral: a new random key each call
    assert any("HISTORY_SIGNING_KEY" in r.message and "ephemeral" in r.message for r in caplog.records)


def test_logs_carry_no_token_or_message_text(ctx: ToolContext, caplog) -> None:
    api = ResumableApi(ctx, Scripted())
    session_id = api.start()
    token = run_e001(api, session_id)[-1]["resume_token"]
    fresh = ResumableApi(ctx, Scripted())

    with caplog.at_level("WARNING"):
        fresh.client.post("/api/sessions/restore", json={"resume_token": "tampered." + token})

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert token not in logged and "מנהלת מוצר" not in logged
    assert "category=" in logged  # a category is logged, never the cause or the token
