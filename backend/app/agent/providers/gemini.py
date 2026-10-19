"""Gemini adapter for LLMProvider, using the official google-genai SDK (Gemini Developer API).

Gemini-specific shapes stay in this module: requests are built as the plain dicts the SDK accepts,
and responses are read back into app.agent.llm types. The SDK is imported only when a real client
is created, so the rest of the project (and its tests) does not need it.

The API key comes from the environment (GEMINI_API_KEY) and is never put in messages, errors,
reprs or conversation state.
"""

import base64
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.agent.llm import Message, ProviderError, ProviderErrorKind, ProviderFailure, Role, ToolCall, ToolSpec

PROVIDER = "gemini"
API_KEY_VARIABLE = "GEMINI_API_KEY"
MODEL_VARIABLE = "GEMINI_MODEL"
TIMEOUT_VARIABLE = "GEMINI_TIMEOUT_SECONDS"
DEFAULT_MODEL = "gemini-flash-latest"
# Per request. Without it google-genai builds its HTTP client with timeout=None (wait forever) and,
# with no retry_options, makes a single attempt - so this bounds every generate_content call.
DEFAULT_TIMEOUT_SECONDS = 60.0
# Retries of one generate_content request after a retryable failure (5xx, timeout, network; not 429):
# 3 attempts in all, waiting 1s then 2s. Each attempt is bounded by the timeout above.
RETRY_DELAYS_SECONDS = (1.0, 2.0)
LOCAL_CALL_ID_PREFIX = "local-"  # call IDs we made up because Gemini sent none; not sent back


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @staticmethod
    def from_env(environ: Mapping[str, str] | None = None) -> "GeminiSettings":
        environ = os.environ if environ is None else environ
        api_key = environ.get(API_KEY_VARIABLE, "").strip()
        if not api_key:
            raise ProviderError(PROVIDER, ProviderErrorKind.CONFIGURATION, f"{API_KEY_VARIABLE} is not set")
        model = environ.get(MODEL_VARIABLE, "").strip() or DEFAULT_MODEL
        return GeminiSettings(api_key=api_key, model=model, timeout_seconds=_timeout(environ))


class GeminiProvider:
    """LLMProvider for Gemini. Pass client= (and sleep=) to inject fakes in tests (no network, no quota).

    events, if set, receives value-free retry lines: attempt numbers, failure category, HTTP status,
    retryable, and the wait - never requests, messages, keys or SDK text.
    """

    name = PROVIDER

    def __init__(
        self,
        settings: GeminiSettings,
        client: Any = None,
        *,
        sleep: Callable[[float], None] | None = None,
        events: Callable[[str], None] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._sleep = sleep
        self._events = events

    def set_events(self, events: Callable[[str], None] | None) -> None:
        self._events = events

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self._settings.model!r}, timeout_seconds={self._settings.timeout_seconds})"

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def timeout_seconds(self) -> float:
        return self._settings.timeout_seconds

    def complete(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Message:
        """One generate_content request, retried on retryable failures with the very same request."""
        contents, config = to_gemini_request(system, messages, tools)  # built once, sent unchanged each attempt
        attempts = len(RETRY_DELAYS_SECONDS) + 1
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                self._emit(f"provider attempt {attempt} started")
            try:
                response = self._get_client().models.generate_content(
                    model=self._settings.model, contents=contents, config=config
                )
            except ProviderError:
                raise  # our own configuration errors: never retried
            except Exception as e:  # SDK and transport errors; their text is not reused (see ProviderError)
                error = _request_error(e)
            else:
                return from_gemini_response(response)

            self._emit(
                f"provider attempt {attempt} failed category={error.category.value} "
                f"status={error.status_code or '-'} retryable={'yes' if error.retryable else 'no'}"
            )
            if not error.retryable or attempt == attempts:
                raise error from None
            delay = RETRY_DELAYS_SECONDS[attempt - 1]
            self._emit(f"provider retrying in {delay:.1f}s")
            (self._sleep or _sleep)(delay)
        raise AssertionError("unreachable")

    def _emit(self, line: str) -> None:
        if self._events:
            self._events(line)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _create_client(self._settings.api_key, self._settings.timeout_seconds)
        return self._client


# --- Request ----------------------------------------------------------------------

def to_gemini_request(
    system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(contents, config) dicts for models.generate_content."""
    config: dict[str, Any] = {"automatic_function_calling": {"disable": True}}
    if system:
        config["system_instruction"] = system
    if tools:
        config["tools"] = [
            {
                "function_declarations": [
                    {"name": t.name, "description": t.description, "parameters_json_schema": _plain(t.parameters)}
                    for t in tools
                ]
            }
        ]
    return [_content(message) for message in messages], config


def _content(message: Message) -> dict[str, Any]:
    if message.role == Role.USER:
        return {"role": "user", "parts": [{"text": message.text}]}
    if message.role == Role.ASSISTANT:
        parts: list[dict[str, Any]] = [{"text": message.text}] if message.text else []
        parts.extend(_function_call_part(call) for call in message.tool_calls)
        return {"role": "model", "parts": parts}
    return {
        "role": "user",
        "parts": [
            {"function_response": _with_gemini_id({"name": r.name, "response": _plain(r.content)}, r.call_id)}
            for r in message.tool_results
        ],
    }


def _function_call_part(call: ToolCall) -> dict[str, Any]:
    part: dict[str, Any] = {"function_call": _with_gemini_id({"name": call.name, "args": _plain(call.arguments)}, call.call_id)}
    if call.provider_state:
        part["thought_signature"] = base64.b64decode(call.provider_state)
    return part


def _with_gemini_id(payload: dict[str, Any], call_id: str) -> dict[str, Any]:
    if not call_id.startswith(LOCAL_CALL_ID_PREFIX):
        payload["id"] = call_id
    return payload


def _plain(value: Any) -> Any:
    """Read-only mappings (from llm types) back to plain dicts/lists for the SDK."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --- Response ---------------------------------------------------------------------

def from_gemini_response(response: Any) -> Message:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        feedback = getattr(response, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None)
        raise ProviderError(
            PROVIDER, ProviderErrorKind.INVALID_RESPONSE, f"no candidates (block reason: {_enum_name(reason)})"
        )
    candidate = candidates[0]
    parts = getattr(getattr(candidate, "content", None), "parts", None) or []

    texts: list[str] = []
    calls: list[ToolCall] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            calls.append(_tool_call(function_call, getattr(part, "thought_signature", None), len(calls)))
        elif getattr(part, "text", None) and not getattr(part, "thought", False):
            texts.append(part.text)

    text = "".join(texts).strip()
    if not text and not calls:
        reason = _enum_name(getattr(candidate, "finish_reason", None))
        raise ProviderError(PROVIDER, ProviderErrorKind.INVALID_RESPONSE, f"empty reply (finish reason: {reason})")
    return Message.assistant(text, calls)


def _tool_call(function_call: Any, thought_signature: bytes | None, index: int) -> ToolCall:
    name = getattr(function_call, "name", None)
    if not name:
        raise ProviderError(PROVIDER, ProviderErrorKind.INVALID_RESPONSE, "function call without a name")
    return ToolCall(
        call_id=getattr(function_call, "id", None) or f"{LOCAL_CALL_ID_PREFIX}{index}-{name}",
        name=name,
        arguments=dict(getattr(function_call, "args", None) or {}),
        provider_state=base64.b64encode(thought_signature).decode("ascii") if thought_signature else None,
    )


def _enum_name(value: Any) -> str:
    return getattr(value, "name", None) or (str(value) if value is not None else "none")


# --- Client and errors ------------------------------------------------------------

def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _timeout(environ: Mapping[str, str]) -> float:
    raw = environ.get(TIMEOUT_VARIABLE, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        seconds = float(raw)
    except ValueError:
        seconds = 0.0
    if not seconds > 0:
        raise ProviderError(PROVIDER, ProviderErrorKind.CONFIGURATION, f"{TIMEOUT_VARIABLE} must be a number of seconds > 0")
    return seconds


def _create_client(api_key: str, timeout_seconds: float) -> Any:
    try:
        from google import genai
    except ImportError:
        raise ProviderError(
            PROVIDER, ProviderErrorKind.CONFIGURATION, "the google-genai package is not installed (pip install google-genai)"
        ) from None
    try:
        # HttpOptions.timeout is in milliseconds; set on the client, it applies to every request.
        return genai.Client(api_key=api_key, http_options={"timeout": round(timeout_seconds * 1000)})
    except Exception as e:
        raise ProviderError(
            PROVIDER, ProviderErrorKind.CONFIGURATION, f"could not create the client ({type(e).__name__})"
        ) from None


# Transport failures raised by the SDK's HTTP client (httpx) before any HTTP status exists, recognized
# by class names in the exception's hierarchy so httpx need not be imported. Timeouts are checked first:
# httpx.TimeoutException is itself a TransportError.
_TIMEOUT_CLASSES = {"TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutError"}
_NETWORK_CLASSES = {"TransportError", "NetworkError", "ConnectError", "ReadError", "WriteError", "CloseError",
                    "ProtocolError", "RemoteProtocolError", "LocalProtocolError", "ProxyError", "ConnectionError"}

_KIND = {
    ProviderFailure.TIMEOUT: ProviderErrorKind.UNAVAILABLE,
    ProviderFailure.SERVER_ERROR: ProviderErrorKind.UNAVAILABLE,
    ProviderFailure.NETWORK_ERROR: ProviderErrorKind.UNAVAILABLE,
    ProviderFailure.RATE_LIMITED: ProviderErrorKind.RATE_LIMITED,
    ProviderFailure.AUTH_ERROR: ProviderErrorKind.AUTHENTICATION,
    ProviderFailure.INVALID_REQUEST: ProviderErrorKind.REQUEST_FAILED,
    ProviderFailure.UNKNOWN: ProviderErrorKind.REQUEST_FAILED,
}


def classify_failure(error: Exception) -> tuple[ProviderFailure, int | None]:
    """(category, HTTP status) for an SDK or transport exception. Its message is never read.

    google-genai raises errors.ClientError (4xx) / ServerError (5xx), both APIError with the HTTP
    status in .code; transport problems arrive as httpx exceptions without a status.
    """
    code = getattr(error, "code", None)
    status = code if isinstance(code, int) and not isinstance(code, bool) and 100 <= code <= 599 else None
    if status is not None:
        if status in (401, 403):
            return ProviderFailure.AUTH_ERROR, status
        if status == 429:
            return ProviderFailure.RATE_LIMITED, status
        if status in (408, 504):
            return ProviderFailure.TIMEOUT, status
        if status >= 500:
            return ProviderFailure.SERVER_ERROR, status
        if status >= 400:
            return ProviderFailure.INVALID_REQUEST, status
        return ProviderFailure.UNKNOWN, status
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & _TIMEOUT_CLASSES:
        return ProviderFailure.TIMEOUT, None
    if names & _NETWORK_CLASSES:
        return ProviderFailure.NETWORK_ERROR, None
    return ProviderFailure.UNKNOWN, None


def _request_error(error: Exception) -> ProviderError:
    category, status = classify_failure(error)
    detail = f"{category.value}, HTTP {status}" if status is not None else category.value
    return ProviderError(
        PROVIDER, _KIND[category], f"generate_content failed ({detail})", category=category, status_code=status
    )
