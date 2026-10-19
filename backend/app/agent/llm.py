"""Provider-neutral LLM types: messages, tool calls/results, tool specs, and the LLMProvider protocol.

Nothing here knows about Gemini, Groq or OpenRouter; each provider adapter converts to and from these.
The LLM only talks and picks tools: salary, eligibility and comparisons are done by the tools.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # results of the assistant's tool calls


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]  # JSON-compatible
    # Opaque data a provider needs back with this call on the next request (e.g. a Gemini thought
    # signature). Only the adapter that created it reads it.
    provider_state: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _frozen_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str  # the ToolCall this answers
    name: str
    content: Mapping[str, Any]  # JSON-compatible tool output

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _frozen_mapping(self.content))


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant only
    tool_results: tuple[ToolResult, ...] = ()  # tool only

    def __post_init__(self) -> None:
        if self.role == Role.USER and (self.tool_calls or self.tool_results or not self.text):
            raise ValueError("A user message has text only")
        if self.role == Role.ASSISTANT and (self.tool_results or not (self.text or self.tool_calls)):
            raise ValueError("An assistant message has text and/or tool calls, and no tool results")
        if self.role == Role.TOOL and (self.text or self.tool_calls or not self.tool_results):
            raise ValueError("A tool message has tool results only")

    @staticmethod
    def user(text: str) -> "Message":
        return Message(Role.USER, text=text)

    @staticmethod
    def assistant(text: str = "", tool_calls: Sequence[ToolCall] = ()) -> "Message":
        return Message(Role.ASSISTANT, text=text, tool_calls=tuple(tool_calls))

    @staticmethod
    def tool(results: Sequence[ToolResult]) -> "Message":
        return Message(Role.TOOL, tool_results=tuple(results))


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})  # JSON Schema

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))


class ProviderErrorKind(StrEnum):
    CONFIGURATION = "configuration"  # missing key, SDK not installed, bad settings
    AUTHENTICATION = "authentication"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"  # blocked, empty or unreadable reply
    REQUEST_FAILED = "request_failed"


# Only transient service/network failures. A rate-limited (429) request is not retried: retrying
# at once adds requests without restoring quota.
_RETRYABLE = {ProviderErrorKind.UNAVAILABLE}


class ProviderFailure(StrEnum):
    """What went wrong, for diagnostics. kind (above) decides behaviour; this explains it."""

    TIMEOUT = "timeout"  # the request took too long (client timeout, HTTP 408/504)
    RATE_LIMITED = "rate_limited"  # HTTP 429: quota or rate limit
    SERVER_ERROR = "server_error"  # HTTP 5xx, e.g. 503 when the model is overloaded
    NETWORK_ERROR = "network_error"  # connection failed or dropped before an HTTP status
    AUTH_ERROR = "auth_error"  # HTTP 401/403
    INVALID_REQUEST = "invalid_request"  # other HTTP 4xx: the request itself was rejected
    CONFIGURATION = "configuration"  # our side: missing key, SDK not installed
    INVALID_RESPONSE = "invalid_response"  # a reply arrived but was blocked, empty or unreadable
    UNKNOWN = "unknown"


_DEFAULT_FAILURE = {
    ProviderErrorKind.CONFIGURATION: ProviderFailure.CONFIGURATION,
    ProviderErrorKind.AUTHENTICATION: ProviderFailure.AUTH_ERROR,
    ProviderErrorKind.RATE_LIMITED: ProviderFailure.RATE_LIMITED,
    ProviderErrorKind.UNAVAILABLE: ProviderFailure.SERVER_ERROR,
    ProviderErrorKind.INVALID_RESPONSE: ProviderFailure.INVALID_RESPONSE,
    ProviderErrorKind.REQUEST_FAILED: ProviderFailure.UNKNOWN,
}


class ProviderError(Exception):
    """A provider failure, described without secrets or provider-specific types.

    Adapters must build the message themselves: never from raw SDK exception text, which may echo
    request details. category and status_code (an HTTP status, when the provider gave one) are safe
    to show in diagnostics.
    """

    def __init__(
        self,
        provider: str,
        kind: ProviderErrorKind,
        message: str,
        *,
        category: ProviderFailure | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"{provider}: {kind.value}: {message}")
        self.provider = provider
        self.kind = kind
        self.category = category or _DEFAULT_FAILURE[kind]
        self.status_code = status_code if isinstance(status_code, int) and 100 <= status_code <= 599 else None

    @property
    def retryable(self) -> bool:
        return self.kind in _RETRYABLE


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Message:
        """The assistant's next message: text, tool calls, or both. Raises ProviderError."""
        ...
