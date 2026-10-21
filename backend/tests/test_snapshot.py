"""Snapshot/token round trips and strict restore validation (app.agent.snapshot). No network, no DB."""

import json
import zlib

import pytest

from app.agent.llm import Message, ToolCall, ToolResult
from app.agent.snapshot import (
    SCHEMA_VERSION,
    TOKEN_MAX_AGE_SECONDS,
    SnapshotError,
    decode_token,
    encode_token,
    snapshot,
)
from app.agent.state import (
    ConversationState,
    EmployeeDetails,
    QuestionStatus,
    RepQuestion,
    RepTopic,
    with_messages,
)
from app.agent.tools import ToolContext
from app.comparison import CompanyCar, Waiver

KEY = b"test-signing-key-32-bytes-long!!"
OTHER_KEY = b"another-key-entirely-different!!"
RULES_VERSION = "mock-2026-10-12"
NOW = 1_700_000_000


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext.load()


@pytest.fixture(scope="module")
def rules_version(ctx: ToolContext) -> str:
    return ctx.rules.version


def built_state() -> ConversationState:
    """A realistic committed state: stated details, a current car, an alternative, a question, cars."""
    state = ConversationState(
        "sess-1",
        details=EmployeeDetails(
            occupation="מנהל/ת מוצר",
            base_salary=18500,
            commissions=0,
            bonus=0,
            pension_rate=0.06,
            training_fund_rate=0.025,
            credit_points=2.75,
            eligible_children=2,
            current=CompanyCar("CAR-09", 300),
        ),
        alternative=CompanyCar("CAR-10", None),
        rep_questions=(
            RepQuestion("Q-1", RepTopic.ALTERNATIVE_PARTICIPATION, "ההשתתפות ב-ATTO 3?", car_id="CAR-10"),
        ),
        usable_car_ids=frozenset({"CAR-09"}),
        offered_car_ids=frozenset({"CAR-10"}),
    )
    # A full transcript: user + assistant text, plus tool calls/results that must be dropped.
    return with_messages(
        state,
        Message.user("אני מנהלת מוצר עם קורולה קרוס, השתתפות 300"),
        Message.assistant(tool_calls=[ToolCall("c1", "save_employee_details", {"occupation": "מנהלת מוצר"})]),
        Message.tool([ToolResult("c1", "save_employee_details", {"ok": True})]),
        Message.assistant("כמה אחוז פנסיה?"),
        Message.user("פנסיה 6%"),
    )


# --- Round trip --------------------------------------------------------------------------------

def test_round_trip_restores_the_resumable_state(ctx: ToolContext, rules_version: str) -> None:
    state = built_state()

    token = encode_token(state, rules_version, turn=3, issued_at=NOW, key=KEY)
    restored, turn = decode_token(token, KEY, ctx, rules_version, NOW)

    assert turn == 3
    assert restored.details == state.details
    assert restored.alternative == state.alternative
    assert restored.rep_questions == state.rep_questions
    assert restored.usable_car_ids == state.usable_car_ids
    assert restored.offered_car_ids == state.offered_car_ids


def test_only_user_and_assistant_text_survive(ctx: ToolContext, rules_version: str) -> None:
    token = encode_token(built_state(), rules_version, turn=1, issued_at=NOW, key=KEY)
    restored, _ = decode_token(token, KEY, ctx, rules_version, NOW)

    assert [(m.role.value, m.text) for m in restored.messages] == [
        ("user", "אני מנהלת מוצר עם קורולה קרוס, השתתפות 300"),  # preserved exactly: grounding reads it
        ("assistant", "כמה אחוז פנסיה?"),
        ("user", "פנסיה 6%"),
    ]
    assert all(not m.tool_calls and not m.tool_results for m in restored.messages)


def test_snapshot_drops_tool_data_and_last_comparison(rules_version: str) -> None:
    data = snapshot(built_state(), rules_version, turn=2, issued_at=NOW)

    assert data["v"] == SCHEMA_VERSION
    assert "last_comparison" not in data and "tool_calls" not in json.dumps(data)
    assert [m["role"] for m in data["messages"]] == ["user", "assistant", "user"]


def test_waiver_current_round_trips(ctx: ToolContext, rules_version: str) -> None:
    state = with_messages(
        ConversationState("s", details=EmployeeDetails(occupation="מנהל/ת מוצר", current=Waiver())),
        Message.user("אני מוותר על הרכב"),
    )
    token = encode_token(state, rules_version, turn=1, issued_at=NOW, key=KEY)
    restored, _ = decode_token(token, KEY, ctx, rules_version, NOW)

    assert isinstance(restored.details.current, Waiver)


# --- Rejection ---------------------------------------------------------------------------------

def test_tampered_token_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    token = encode_token(built_state(), rules_version, turn=1, issued_at=NOW, key=KEY)
    signature, _, body = token.partition(".")
    tampered = f"{signature}.{body[:-2]}XY"

    with pytest.raises(SnapshotError):
        decode_token(tampered, KEY, ctx, rules_version, NOW)


def test_wrong_key_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    token = encode_token(built_state(), rules_version, turn=1, issued_at=NOW, key=KEY)

    with pytest.raises(SnapshotError) as info:
        decode_token(token, OTHER_KEY, ctx, rules_version, NOW)
    assert info.value.category == "bad_signature"


def test_schema_version_mismatch_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    token = _forge({**snapshot(built_state(), rules_version, 1, NOW), "v": SCHEMA_VERSION + 1})

    with pytest.raises(SnapshotError) as info:
        decode_token(token, KEY, ctx, rules_version, NOW)
    assert info.value.category == "schema_version"


def test_rules_version_mismatch_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    token = encode_token(built_state(), "some-other-rules-version", turn=1, issued_at=NOW, key=KEY)

    with pytest.raises(SnapshotError) as info:
        decode_token(token, KEY, ctx, rules_version, NOW)
    assert info.value.category == "rules_version"


def test_expired_token_is_rejected_with_an_injected_clock(ctx: ToolContext, rules_version: str) -> None:
    token = encode_token(built_state(), rules_version, turn=1, issued_at=NOW, key=KEY)

    # Exactly at the limit still restores; one second past it is expired.
    decode_token(token, KEY, ctx, rules_version, NOW + TOKEN_MAX_AGE_SECONDS)
    with pytest.raises(SnapshotError) as info:
        decode_token(token, KEY, ctx, rules_version, NOW + TOKEN_MAX_AGE_SECONDS + 1)
    assert info.value.category == "expired"


def test_oversized_token_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    with pytest.raises(SnapshotError) as info:
        decode_token("x" * (64 * 1024 + 1), KEY, ctx, rules_version, NOW)
    assert info.value.category == "oversized"


def test_unknown_car_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    data = snapshot(built_state(), rules_version, 1, NOW)
    data["details"]["current"] = {"kind": "company_car", "car_id": "CAR-999", "participation": 300}

    with pytest.raises(SnapshotError) as info:
        decode_token(_forge(data), KEY, ctx, rules_version, NOW)
    assert info.value.category == "car_id"


def test_unknown_occupation_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    data = snapshot(built_state(), rules_version, 1, NOW)
    data["details"]["occupation"] = "אסטרונאוט/ית"

    with pytest.raises(SnapshotError) as info:
        decode_token(_forge(data), KEY, ctx, rules_version, NOW)
    assert info.value.category == "occupation"


def test_out_of_range_number_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    data = snapshot(built_state(), rules_version, 1, NOW)
    data["details"]["pension_rate"] = 2.0  # a fraction must be <= 1

    with pytest.raises(SnapshotError) as info:
        decode_token(_forge(data), KEY, ctx, rules_version, NOW)
    assert info.value.category == "pension_rate"


def test_empty_message_text_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    data = snapshot(built_state(), rules_version, 1, NOW)
    data["messages"].append({"role": "user", "text": "   "})

    with pytest.raises(SnapshotError) as info:
        decode_token(_forge(data), KEY, ctx, rules_version, NOW)
    assert info.value.category == "message_text"


def test_bad_role_is_rejected(ctx: ToolContext, rules_version: str) -> None:
    data = snapshot(built_state(), rules_version, 1, NOW)
    data["messages"].append({"role": "tool", "text": "secret"})

    with pytest.raises(SnapshotError) as info:
        decode_token(_forge(data), KEY, ctx, rules_version, NOW)
    assert info.value.category == "role"


def _forge(data: dict) -> str:
    """Sign arbitrary snapshot data with the test KEY, as encode_token would, to test restore paths."""
    from app.agent.snapshot import _b64encode, _sign  # test-only reach into the signing primitives

    payload = zlib.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return f"{_b64encode(_sign(payload, KEY))}.{_b64encode(payload)}"
