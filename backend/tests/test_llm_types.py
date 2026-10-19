from dataclasses import FrozenInstanceError

import pytest

from app.agent.llm import (
    Message,
    ProviderError,
    ProviderErrorKind,
    Role,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from app.agent.state import ConversationState, with_messages


def call(name: str = "find_car", **arguments) -> ToolCall:
    return ToolCall(call_id="c1", name=name, arguments=arguments)


# --- Message shapes ------------------------------------------------------------

def test_message_constructors() -> None:
    assert Message.user("שלום") == Message(Role.USER, text="שלום")
    assert Message.assistant("היי").role == Role.ASSISTANT
    assert Message.assistant(tool_calls=[call()]).tool_calls == (call(),)
    assert Message.tool([ToolResult("c1", "find_car", {"status": "identified"})]).role == Role.TOOL


@pytest.mark.parametrize(
    "build",
    [
        lambda: Message(Role.USER),  # no text
        lambda: Message(Role.USER, text="x", tool_calls=(call(),)),
        lambda: Message(Role.ASSISTANT),  # neither text nor calls
        lambda: Message(Role.ASSISTANT, text="x", tool_results=(ToolResult("c1", "n", {}),)),
        lambda: Message(Role.TOOL),  # no results
        lambda: Message(Role.TOOL, text="x", tool_results=(ToolResult("c1", "n", {}),)),
    ],
)
def test_invalid_message_shapes_are_refused(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_messages_and_payloads_are_read_only() -> None:
    arguments = {"query": "קורולה"}
    tool_call = ToolCall("c1", "find_car", arguments)
    arguments["query"] = "changed"  # the caller's dict is copied, not shared

    with pytest.raises(FrozenInstanceError):
        Message.user("x").text = "y"  # type: ignore[misc]
    with pytest.raises(TypeError):
        tool_call.arguments["query"] = "y"  # type: ignore[index]
    with pytest.raises(TypeError):
        ToolResult("c1", "n", {"a": 1}).content["a"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        ToolSpec("n", "d", {"type": "object"}).parameters["type"] = "x"  # type: ignore[index]
    assert tool_call.arguments["query"] == "קורולה"


def test_tool_spec_defaults_to_no_parameters() -> None:
    assert dict(ToolSpec("list_alternatives", "d").parameters) == {"type": "object", "properties": {}}


# --- ProviderError ------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind, retryable",
    [
        (ProviderErrorKind.RATE_LIMITED, True),
        (ProviderErrorKind.UNAVAILABLE, True),
        (ProviderErrorKind.AUTHENTICATION, False),
        (ProviderErrorKind.CONFIGURATION, False),
        (ProviderErrorKind.INVALID_RESPONSE, False),
        (ProviderErrorKind.REQUEST_FAILED, False),
    ],
)
def test_provider_error_retryable(kind: ProviderErrorKind, retryable: bool) -> None:
    error = ProviderError("gemini", kind, "x")

    assert (error.provider, error.kind, error.retryable) == ("gemini", kind, retryable)
    assert str(error) == f"gemini: {kind.value}: x"


# --- Messages in ConversationState --------------------------------------------------

def test_state_starts_with_no_messages() -> None:
    assert ConversationState("c-1").messages == ()


def test_with_messages_appends_without_changing_the_old_state() -> None:
    first = with_messages(ConversationState("c-1"), Message.user("שלום"))

    second = with_messages(first, Message.assistant(tool_calls=[call()]), Message.tool([ToolResult("c1", "find_car", {})]))

    assert [m.role for m in first.messages] == [Role.USER]
    assert [m.role for m in second.messages] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    with pytest.raises(FrozenInstanceError):
        second.messages = ()  # type: ignore[misc]
