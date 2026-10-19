"""GeminiProvider with fakes only: no google-genai import, no network, no quota."""

import base64
import sys
from types import SimpleNamespace

import pytest

from app.agent.llm import LLMProvider, Message, ProviderError, ProviderErrorKind, Role, ToolCall, ToolResult, ToolSpec
from app.agent.providers.gemini import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
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
    def failing_client(*, api_key, http_options):
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

    def recording_client(*, api_key, http_options):
        created.append((api_key, http_options))
        return fake_client(response(text_part("ok")))

    fake_genai = SimpleNamespace(Client=recording_client)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    provider = GeminiProvider(SETTINGS)

    assert created == []  # nothing created until the first request
    provider.complete("", [Message.user("x")], [])
    provider.complete("", [Message.user("y")], [])
    assert created == [(FAKE_KEY, {"timeout": 60_000})]


# --- Request timeout ----------------------------------------------------------------------------

def test_default_timeout() -> None:
    assert GeminiSettings.from_env({"GEMINI_API_KEY": FAKE_KEY}).timeout_seconds == DEFAULT_TIMEOUT_SECONDS == 60.0


def test_timeout_from_env() -> None:
    settings = GeminiSettings.from_env({"GEMINI_API_KEY": FAKE_KEY, "GEMINI_TIMEOUT_SECONDS": " 15 "})

    assert settings.timeout_seconds == 15.0


@pytest.mark.parametrize("value", ["abc", "0", "-5", "nan"])
def test_invalid_timeout_is_a_configuration_error(value: str) -> None:
    with pytest.raises(ProviderError) as info:
        GeminiSettings.from_env({"GEMINI_API_KEY": FAKE_KEY, "GEMINI_TIMEOUT_SECONDS": value})

    assert info.value.kind == ProviderErrorKind.CONFIGURATION
    assert "GEMINI_TIMEOUT_SECONDS" in str(info.value) and FAKE_KEY not in str(info.value)


@pytest.mark.parametrize("seconds, milliseconds", [(60.0, 60_000), (15.0, 15_000), (2.5, 2_500)])
def test_timeout_is_set_on_the_sdk_client_in_milliseconds(monkeypatch, seconds: float, milliseconds: int) -> None:
    created = []

    def recording_client(*, api_key, http_options):
        created.append(http_options)
        return fake_client(response(text_part("ok")))

    fake_genai = SimpleNamespace(Client=recording_client)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))

    GeminiProvider(GeminiSettings(api_key=FAKE_KEY, timeout_seconds=seconds)).complete("", [Message.user("x")], [])

    assert created == [{"timeout": milliseconds}]


def named_error(name: str) -> Exception:
    return type(name, (Exception,), {})(f"timed out; key={FAKE_KEY}")


@pytest.mark.parametrize(
    "error",
    [
        named_error("ReadTimeout"),  # httpx, when the request exceeds the timeout
        named_error("WriteTimeout"),
        named_error("PoolTimeout"),
        named_error("ConnectTimeout"),
        named_error("TimeoutException"),
        FakeAPIError(408),
        FakeAPIError(504),  # the server gave up (X-Server-Timeout)
    ],
)
def test_timeouts_are_retryable_unavailable_errors(error: Exception) -> None:
    provider = GeminiProvider(SETTINGS, client=fake_client(error=error))

    with pytest.raises(ProviderError) as info:
        provider.complete("", [Message.user("x")], [])

    assert info.value.kind == ProviderErrorKind.UNAVAILABLE and info.value.retryable
    assert FAKE_KEY not in str(info.value) and info.value.__cause__ is None


def test_repr_shows_timeout_but_not_key() -> None:
    text = repr(GeminiProvider(GeminiSettings(api_key=FAKE_KEY, timeout_seconds=30), client=fake_client()))

    assert "timeout_seconds=30" in text and FAKE_KEY not in text


# --- Failure classification (google-genai ClientError/ServerError and httpx shapes) -------------

from app.agent.llm import ProviderFailure  # noqa: E402
from app.agent.providers.gemini import classify_failure  # noqa: E402


class FakeTransportError(Exception):  # httpx class tree, by name: classification never imports httpx
    pass


FakeTransportError.__name__ = "TransportError"
FakeTimeoutException = type("TimeoutException", (FakeTransportError,), {})
FakeNetworkError = type("NetworkError", (FakeTransportError,), {})
FakeProtocolError = type("ProtocolError", (FakeTransportError,), {})


def transport(cls_name: str, base: type) -> Exception:
    return type(cls_name, (base,), {})(f"connection to https://example.invalid?key={FAKE_KEY} failed")


@pytest.mark.parametrize(
    "error, category, status, kind, retryable",
    [
        # HTTP statuses from google.genai.errors.ClientError / ServerError (.code)
        (FakeAPIError(400), ProviderFailure.INVALID_REQUEST, 400, ProviderErrorKind.REQUEST_FAILED, False),
        (FakeAPIError(404), ProviderFailure.INVALID_REQUEST, 404, ProviderErrorKind.REQUEST_FAILED, False),
        (FakeAPIError(401), ProviderFailure.AUTH_ERROR, 401, ProviderErrorKind.AUTHENTICATION, False),
        (FakeAPIError(403), ProviderFailure.AUTH_ERROR, 403, ProviderErrorKind.AUTHENTICATION, False),
        (FakeAPIError(429), ProviderFailure.RATE_LIMITED, 429, ProviderErrorKind.RATE_LIMITED, False),
        (FakeAPIError(408), ProviderFailure.TIMEOUT, 408, ProviderErrorKind.UNAVAILABLE, True),
        (FakeAPIError(504), ProviderFailure.TIMEOUT, 504, ProviderErrorKind.UNAVAILABLE, True),
        (FakeAPIError(500), ProviderFailure.SERVER_ERROR, 500, ProviderErrorKind.UNAVAILABLE, True),
        (FakeAPIError(502), ProviderFailure.SERVER_ERROR, 502, ProviderErrorKind.UNAVAILABLE, True),
        (FakeAPIError(503), ProviderFailure.SERVER_ERROR, 503, ProviderErrorKind.UNAVAILABLE, True),
        # Transport problems (no HTTP status)
        (transport("ReadTimeout", FakeTimeoutException), ProviderFailure.TIMEOUT, None, ProviderErrorKind.UNAVAILABLE, True),
        (transport("ConnectTimeout", FakeTimeoutException), ProviderFailure.TIMEOUT, None, ProviderErrorKind.UNAVAILABLE, True),
        (transport("ConnectError", FakeNetworkError), ProviderFailure.NETWORK_ERROR, None, ProviderErrorKind.UNAVAILABLE, True),
        (transport("ReadError", FakeNetworkError), ProviderFailure.NETWORK_ERROR, None, ProviderErrorKind.UNAVAILABLE, True),
        (transport("RemoteProtocolError", FakeProtocolError), ProviderFailure.NETWORK_ERROR, None, ProviderErrorKind.UNAVAILABLE, True),
        (TimeoutError(f"key={FAKE_KEY}"), ProviderFailure.TIMEOUT, None, ProviderErrorKind.UNAVAILABLE, True),
        (ConnectionResetError(f"key={FAKE_KEY}"), ProviderFailure.NETWORK_ERROR, None, ProviderErrorKind.UNAVAILABLE, True),
        # Anything else
        (RuntimeError(f"key={FAKE_KEY}"), ProviderFailure.UNKNOWN, None, ProviderErrorKind.REQUEST_FAILED, False),
    ],
)
def test_failures_are_classified_deterministically(error, category, status, kind, retryable) -> None:
    provider = GeminiProvider(SETTINGS, client=fake_client(error=error))

    with pytest.raises(ProviderError) as info:
        provider.complete("", [Message.user("x")], [])

    raised = info.value
    assert (raised.category, raised.status_code, raised.kind, raised.retryable) == (category, status, kind, retryable)
    assert FAKE_KEY not in str(raised) and "example.invalid" not in str(raised) and "ERROR." not in str(raised)
    assert raised.__cause__ is None


@pytest.mark.parametrize("code", ["503", 999, 42, True, None])
def test_only_real_http_statuses_are_reported(code) -> None:
    error = Exception("x")
    error.code = code

    category, status = classify_failure(error)

    assert status is None and category == ProviderFailure.UNKNOWN


def test_adapter_errors_get_a_category_from_their_kind() -> None:
    assert ProviderError("gemini", ProviderErrorKind.CONFIGURATION, "x").category == ProviderFailure.CONFIGURATION
    assert ProviderError("gemini", ProviderErrorKind.INVALID_RESPONSE, "x").category == ProviderFailure.INVALID_RESPONSE
    assert ProviderError("gemini", ProviderErrorKind.CONFIGURATION, "x").status_code is None


# --- Bounded retry ---------------------------------------------------------------------------------

from app.agent.providers import gemini as gemini_module  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch) -> list[float]:
    """No test in this module ever waits: retries record their delays instead."""
    slept: list[float] = []
    monkeypatch.setattr(gemini_module, "_sleep", slept.append)
    return slept


class SequenceModels:
    """Answers each generate_content call with the next outcome: an exception is raised, a response returned."""

    def __init__(self, *outcomes) -> None:
        self._outcomes, self.calls = list(outcomes), []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def retrying_provider(*outcomes):
    models = SequenceModels(*outcomes)
    slept: list[float] = []
    events: list[str] = []
    provider = GeminiProvider(SETTINGS, client=SimpleNamespace(models=models), sleep=slept.append, events=events.append)
    return provider, models, slept, events


OK = response(text_part("שלום"))


@pytest.mark.parametrize(
    "failures",
    [
        [FakeAPIError(503)],
        [FakeAPIError(503), FakeAPIError(503)],
        [FakeAPIError(500), FakeAPIError(502)],
        [transport("ConnectError", FakeNetworkError)],
        [transport("ReadTimeout", FakeTimeoutException)],
        [FakeAPIError(504)],
    ],
)
def test_retryable_failures_then_success(failures) -> None:
    provider, models, slept, _ = retrying_provider(*failures, OK)

    assert provider.complete("s", [Message.user("x")], [FIND_CAR]) == Message.assistant("שלום")
    assert len(models.calls) == len(failures) + 1
    assert slept == [1.0, 2.0][: len(failures)]


def test_three_503s_raise_after_three_attempts() -> None:
    provider, models, slept, _ = retrying_provider(FakeAPIError(503), FakeAPIError(503), FakeAPIError(503))

    with pytest.raises(ProviderError) as info:
        provider.complete("s", [Message.user("x")], [])

    assert (info.value.category, info.value.status_code, info.value.retryable) == (ProviderFailure.SERVER_ERROR, 503, True)
    assert len(models.calls) == 3 and slept == [1.0, 2.0]
    assert info.value.__cause__ is None and FAKE_KEY not in str(info.value)


def test_rate_limited_is_not_retried() -> None:
    provider, models, slept, events = retrying_provider(FakeAPIError(429), OK)

    with pytest.raises(ProviderError) as info:
        provider.complete("s", [Message.user("x")], [])

    assert (info.value.category, info.value.status_code, info.value.retryable) == (ProviderFailure.RATE_LIMITED, 429, False)
    assert len(models.calls) == 1 and slept == []
    assert events == ["provider attempt 1 failed category=rate_limited status=429 retryable=no"]


def test_429_after_a_503_stops_the_retries() -> None:
    provider, models, slept, _ = retrying_provider(FakeAPIError(503), FakeAPIError(429), OK)

    with pytest.raises(ProviderError) as info:
        provider.complete("s", [Message.user("x")], [])

    assert info.value.category == ProviderFailure.RATE_LIMITED
    assert len(models.calls) == 2 and slept == [1.0]


@pytest.mark.parametrize("error", [FakeAPIError(400), FakeAPIError(401), FakeAPIError(403), FakeAPIError(404), RuntimeError("x")])
def test_non_retryable_failures_are_not_retried(error) -> None:
    provider, models, slept, _ = retrying_provider(error, OK)

    with pytest.raises(ProviderError) as info:
        provider.complete("s", [Message.user("x")], [])

    assert not info.value.retryable
    assert len(models.calls) == 1 and slept == []


def test_invalid_response_is_not_retried() -> None:
    blocked = SimpleNamespace(candidates=[], prompt_feedback=None)
    provider, models, slept, _ = retrying_provider(blocked, OK)

    with pytest.raises(ProviderError) as info:
        provider.complete("s", [Message.user("x")], [])

    assert info.value.kind == ProviderErrorKind.INVALID_RESPONSE
    assert len(models.calls) == 1 and slept == []


def test_configuration_error_is_not_retried(monkeypatch, no_real_sleep) -> None:
    monkeypatch.setitem(sys.modules, "google.genai", None)

    with pytest.raises(ProviderError) as info:
        GeminiProvider(SETTINGS).complete("", [Message.user("x")], [])

    assert info.value.category == ProviderFailure.CONFIGURATION
    assert no_real_sleep == []


def test_every_attempt_sends_the_identical_request() -> None:
    history = [
        Message.user("יש לי קורולה קרוס"),
        Message.assistant(tool_calls=[ToolCall("fc-1", "find_car", {"query": "קורולה קרוס"}, base64.b64encode(b"sig").decode())]),
        Message.tool([ToolResult("fc-1", "find_car", {"status": "identified"})]),
    ]
    provider, models, _, _ = retrying_provider(FakeAPIError(503), FakeAPIError(503), OK)

    provider.complete("system", history, [FIND_CAR])

    first, second, third = models.calls
    assert first == second == third
    assert first["contents"] is second["contents"] is third["contents"]  # built once, not rebuilt


def test_function_call_reply_after_a_retry_is_the_same_as_without_one() -> None:
    reply = response(call_part("compare_alternative", {"car": "אטו 3"}, call_id="fc-9", signature=b"sig"))
    direct = GeminiProvider(SETTINGS, client=fake_client(reply)).complete("s", [Message.user("x")], [FIND_CAR])
    provider, _, _, _ = retrying_provider(FakeAPIError(503), reply)

    retried = provider.complete("s", [Message.user("x")], [FIND_CAR])

    assert retried == direct
    assert retried.tool_calls[0].name == "compare_alternative" and dict(retried.tool_calls[0].arguments) == {"car": "אטו 3"}


def test_retry_events_are_safe() -> None:
    provider, _, _, events = retrying_provider(FakeAPIError(503), transport("ReadTimeout", FakeTimeoutException), OK)

    provider.complete("s", [Message.user("שכר 18,500")], [FIND_CAR])

    assert events == [
        "provider attempt 1 failed category=server_error status=503 retryable=yes",
        "provider retrying in 1.0s",
        "provider attempt 2 started",
        "provider attempt 2 failed category=timeout status=- retryable=yes",
        "provider retrying in 2.0s",
        "provider attempt 3 started",
    ]
    text = "\n".join(events)
    for secret in (FAKE_KEY, "example.invalid", "18,500", "find_car", "ERROR."):
        assert secret not in text


def test_no_events_and_no_retry_on_success() -> None:
    provider, models, slept, events = retrying_provider(OK)

    provider.complete("s", [Message.user("x")], [])

    assert (len(models.calls), slept, events) == (1, [], [])
