"""GeminiProvider with fakes only: no google-genai import, no network, no quota."""

import base64
import sys
from types import SimpleNamespace

import pytest

from app.agent.llm import LLMProvider, Message, ProviderError, ProviderErrorKind, Role, ToolCall, ToolResult, ToolSpec
from app.agent.providers.gemini import (
    DEFAULT_MODEL,
    LOCAL_CALL_ID_PREFIX,
    GeminiProvider,
    GeminiSettings,
    from_gemini_response,
    to_gemini_request,
)

FAKE_KEY = "AIza-FAKE-KEY-never-real-0123456789"


class FakeModels:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response, self.error, self.calls = response, error, []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return self.response


def fake_client(response=None, error: Exception | None = None):
    return SimpleNamespace(models=FakeModels(response, error))


def response(*parts, finish_reason="STOP"):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)), finish_reason=finish_reason)]
    )


def text_part(text, thought=False):
    return SimpleNamespace(text=text, thought=thought, function_call=None, thought_signature=None)


def call_part(name, args, call_id=None, signature=None):
    return SimpleNamespace(
        text=None, thought=None, thought_signature=signature,
        function_call=SimpleNamespace(name=name, args=args, id=call_id),
    )


class FakeAPIError(Exception):
    """Shaped like google.genai.errors.APIError: an HTTP code and a message that may echo the request."""

    def __init__(self, code: int) -> None:
        super().__init__(f"{code} ERROR. request with key={FAKE_KEY}")
        self.code = code


SETTINGS = GeminiSettings(api_key=FAKE_KEY)
FIND_CAR = ToolSpec(
    "find_car", "Find a catalogue car", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
)


# --- Settings from the environment -----------------------------------------------------

def test_settings_from_env() -> None:
    settings = GeminiSettings.from_env({"GEMINI_API_KEY": f"  {FAKE_KEY} ", "GEMINI_MODEL": "gemini-2.5-flash"})

    assert (settings.api_key, settings.model) == (FAKE_KEY, "gemini-2.5-flash")


def test_default_model() -> None:
    assert GeminiSettings.from_env({"GEMINI_API_KEY": FAKE_KEY}).model == DEFAULT_MODEL


@pytest.mark.parametrize("environ", [{}, {"GEMINI_API_KEY": "   "}])
def test_missing_key_is_a_configuration_error(environ) -> None:
    with pytest.raises(ProviderError) as info:
        GeminiSettings.from_env(environ)

    assert info.value.kind == ProviderErrorKind.CONFIGURATION
    assert "GEMINI_API_KEY is not set" in str(info.value)


def test_settings_from_real_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert GeminiSettings.from_env().api_key == FAKE_KEY


def test_key_is_not_in_reprs() -> None:
    provider = GeminiProvider(SETTINGS, client=fake_client())

    assert FAKE_KEY not in repr(SETTINGS)
    assert FAKE_KEY not in repr(provider)


def test_provider_satisfies_the_protocol() -> None:
    provider: LLMProvider = GeminiProvider(SETTINGS, client=fake_client())

    assert provider.name == "gemini"


# --- Request conversion --------------------------------------------------------------------

def test_request_has_system_tools_and_no_automatic_calling() -> None:
    contents, config = to_gemini_request("אתה יועץ רכב", [Message.user("שלום")], [FIND_CAR])

    assert contents == [{"role": "user", "parts": [{"text": "שלום"}]}]
    assert config["system_instruction"] == "אתה יועץ רכב"
    assert config["automatic_function_calling"] == {"disable": True}
    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration == {
        "name": "find_car",
        "description": "Find a catalogue car",
        "parameters_json_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }
    assert type(declaration["parameters_json_schema"]) is dict  # plain dicts for the SDK, not read-only views


def test_request_without_tools_has_no_tools_entry() -> None:
    _, config = to_gemini_request("", [Message.user("x")], [])

    assert "tools" not in config and "system_instruction" not in config


def test_tool_call_round_trip_maps_roles_ids_and_signature() -> None:
    signature = b"\x01\x02opaque"
    messages = [
        Message.user("יש לי קורולה קרוס"),
        Message.assistant(
            "בודק",
            [ToolCall("fc-7", "find_car", {"query": "קורולה קרוס"}, base64.b64encode(signature).decode())],
        ),
        Message.tool([ToolResult("fc-7", "find_car", {"status": "identified", "cars": [{"car_id": "CAR-09"}]})]),
    ]

    contents, _ = to_gemini_request("", messages, [FIND_CAR])

    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [
        {"text": "בודק"},
        {"function_call": {"name": "find_car", "args": {"query": "קורולה קרוס"}, "id": "fc-7"}, "thought_signature": signature},
    ]
    assert contents[2]["parts"] == [
        {"function_response": {"name": "find_car", "response": {"status": "identified", "cars": [{"car_id": "CAR-09"}]}, "id": "fc-7"}}
    ]


def test_locally_made_call_ids_are_not_sent_to_gemini() -> None:
    local_id = f"{LOCAL_CALL_ID_PREFIX}0-find_car"
    messages = [
        Message.assistant(tool_calls=[ToolCall(local_id, "find_car", {"query": "x"})]),
        Message.tool([ToolResult(local_id, "find_car", {})]),
    ]

    contents, _ = to_gemini_request("", messages, [])

    assert "id" not in contents[0]["parts"][0]["function_call"]
    assert "id" not in contents[1]["parts"][0]["function_response"]
    assert "thought_signature" not in contents[0]["parts"][0]


# --- Response conversion ----------------------------------------------------------------------

def test_text_reply() -> None:
    message = from_gemini_response(response(text_part("שלום, "), text_part("מה התפקיד שלך?")))

    assert message == Message.assistant("שלום, מה התפקיד שלך?")


def test_function_calls_with_ids_arguments_and_signature() -> None:
    reply = response(
        text_part("בודק", thought=True),  # model reasoning: not shown as text
        call_part("find_car", {"query": "קורולה"}, call_id="fc-1", signature=b"sig"),
        call_part("list_alternatives", None),
    )

    message = from_gemini_response(reply)

    assert message.role == Role.ASSISTANT and message.text == ""
    first, second = message.tool_calls
    assert (first.call_id, first.name, dict(first.arguments)) == ("fc-1", "find_car", {"query": "קורולה"})
    assert base64.b64decode(first.provider_state) == b"sig"
    assert (second.call_id, dict(second.arguments), second.provider_state) == (
        f"{LOCAL_CALL_ID_PREFIX}1-list_alternatives", {}, None
    )


def test_no_candidates_is_an_invalid_response() -> None:
    blocked = SimpleNamespace(candidates=[], prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY")))

    with pytest.raises(ProviderError) as info:
        from_gemini_response(blocked)

    assert info.value.kind == ProviderErrorKind.INVALID_RESPONSE
    assert "SAFETY" in str(info.value)


def test_empty_reply_is_an_invalid_response() -> None:
    with pytest.raises(ProviderError) as info:
        from_gemini_response(response(text_part("thinking...", thought=True), finish_reason="MAX_TOKENS"))

    assert info.value.kind == ProviderErrorKind.INVALID_RESPONSE
    assert "MAX_TOKENS" in str(info.value)


def test_function_call_without_name_is_an_invalid_response() -> None:
    with pytest.raises(ProviderError):
        from_gemini_response(response(call_part("", {})))


# --- complete() with a fake client --------------------------------------------------------------

def test_complete_sends_model_contents_and_config() -> None:
    client = fake_client(response(text_part("שלום")))
    provider = GeminiProvider(GeminiSettings(api_key=FAKE_KEY, model="gemini-test"), client=client)

    message = provider.complete("system", [Message.user("היי")], [FIND_CAR])

    assert message == Message.assistant("שלום")
    (sent,) = client.models.calls
    assert sent["model"] == "gemini-test"
    assert sent["contents"] == [{"role": "user", "parts": [{"text": "היי"}]}]
    assert sent["config"]["tools"][0]["function_declarations"][0]["name"] == "find_car"
    assert FAKE_KEY not in repr(sent)  # the key is never part of the request payload


@pytest.mark.parametrize(
    "error, kind",
    [
        (FakeAPIError(401), ProviderErrorKind.AUTHENTICATION),
        (FakeAPIError(403), ProviderErrorKind.AUTHENTICATION),
        (FakeAPIError(429), ProviderErrorKind.RATE_LIMITED),
        (FakeAPIError(503), ProviderErrorKind.UNAVAILABLE),
        (FakeAPIError(400), ProviderErrorKind.REQUEST_FAILED),
        (TimeoutError(f"timed out, key={FAKE_KEY}"), ProviderErrorKind.UNAVAILABLE),
        (type("ConnectError", (Exception,), {})(f"key={FAKE_KEY}"), ProviderErrorKind.UNAVAILABLE),
        (RuntimeError(f"boom key={FAKE_KEY}"), ProviderErrorKind.REQUEST_FAILED),
    ],
)
def test_sdk_errors_become_provider_errors_without_the_key(error: Exception, kind: ProviderErrorKind) -> None:
    provider = GeminiProvider(SETTINGS, client=fake_client(error=error))

    with pytest.raises(ProviderError) as info:
        provider.complete("", [Message.user("x")], [])

    raised = info.value
    assert raised.kind == kind
    assert FAKE_KEY not in str(raised) and FAKE_KEY not in repr(raised)
    assert all(FAKE_KEY not in str(arg) for arg in raised.args)
    assert raised.__cause__ is None and raised.__suppress_context__  # the SDK exception is not chained


def test_missing_sdk_is_a_configuration_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)  # makes "from google import genai" fail
    provider = GeminiProvider(SETTINGS)  # no client injected: the real one would be created

    with pytest.raises(ProviderError) as info:
        provider.complete("", [Message.user("x")], [])

    assert info.value.kind == ProviderErrorKind.CONFIGURATION
    assert "pip install google-genai" in str(info.value)
    assert FAKE_KEY not in str(info.value)


def test_client_creation_failure_hides_the_key(monkeypatch) -> None:
    def failing_client(*, api_key):
        raise ValueError(f"bad key {api_key}")

    fake_genai = SimpleNamespace(Client=failing_client)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))

    with pytest.raises(ProviderError) as info:
        GeminiProvider(SETTINGS).complete("", [Message.user("x")], [])

    assert info.value.kind == ProviderErrorKind.CONFIGURATION
    assert FAKE_KEY not in str(info.value)


def test_real_client_is_created_lazily_with_the_settings_key(monkeypatch) -> None:
    created = []

    def recording_client(*, api_key):
        created.append(api_key)
        return fake_client(response(text_part("ok")))

    fake_genai = SimpleNamespace(Client=recording_client)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    provider = GeminiProvider(SETTINGS)

    assert created == []  # nothing created until the first request
    provider.complete("", [Message.user("x")], [])
    provider.complete("", [Message.user("y")], [])
    assert created == [FAKE_KEY]
