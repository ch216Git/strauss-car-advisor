"""Gemini adapter for LLMProvider, using the official google-genai SDK (Gemini Developer API).

Gemini-specific shapes stay in this module: requests are built as the plain dicts the SDK accepts,
and responses are read back into app.agent.llm types. The SDK is imported only when a real client
is created, so the rest of the project (and its tests) does not need it.

The API key comes from the environment (GEMINI_API_KEY) and is never put in messages, errors,
reprs or conversation state.
"""

import base64
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.agent.llm import Message, ProviderError, ProviderErrorKind, Role, ToolCall, ToolSpec

PROVIDER = "gemini"
API_KEY_VARIABLE = "GEMINI_API_KEY"
MODEL_VARIABLE = "GEMINI_MODEL"
TIMEOUT_VARIABLE = "GEMINI_TIMEOUT_SECONDS"
DEFAULT_MODEL = "gemini-flash-latest"
# Per request. Without it google-genai builds its HTTP client with timeout=None (wait forever) and,
# with no retry_options, makes a single attempt - so this bounds every generate_content call.
DEFAULT_TIMEOUT_SECONDS = 60.0
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
    """LLMProvider for Gemini. Pass client= to inject a fake in tests (no network, no quota)."""

    name = PROVIDER

    def __init__(self, settings: GeminiSettings, client: Any = None) -> None:
        self._settings = settings
        self._client = client

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self._settings.model!r}, timeout_seconds={self._settings.timeout_seconds})"

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def timeout_seconds(self) -> float:
        return self._settings.timeout_seconds

    def complete(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Message:
        contents, config = to_gemini_request(system, messages, tools)
        try:
            response = self._get_client().models.generate_content(
                model=self._settings.model, contents=contents, config=config
            )
        except ProviderError:
            raise
        except Exception as e:  # SDK and transport errors; their text is not reused (see ProviderError)
            raise _request_error(e) from None
        return from_gemini_response(response)

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


# Transport failures from the SDK's HTTP client, recognized by name so httpx need not be imported.
_TRANSPORT_ERRORS = (
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException", "NetworkError",
)


def _request_error(error: Exception) -> ProviderError:
    """Classify by HTTP status (google.genai.errors.APIError.code); the SDK's own text is not reused."""
    code = getattr(error, "code", None)
    if code in (401, 403):
        kind = ProviderErrorKind.AUTHENTICATION
    elif code == 429:
        kind = ProviderErrorKind.RATE_LIMITED
    elif code == 408 or (isinstance(code, int) and code >= 500):  # 408/504: request timed out
        kind = ProviderErrorKind.UNAVAILABLE
    elif isinstance(error, (ConnectionError, TimeoutError)) or type(error).__name__.endswith(_TRANSPORT_ERRORS):
        kind = ProviderErrorKind.UNAVAILABLE
    else:
        kind = ProviderErrorKind.REQUEST_FAILED
    status = f"HTTP {code}" if isinstance(code, int) else type(error).__name__
    return ProviderError(PROVIDER, kind, f"generate_content failed ({status})")
