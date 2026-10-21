"""Resume a conversation without a database: ConversationState <-> a signed, compressed token.

The browser keeps the token (history); the server keeps nothing on disk. A token is JSON -> zlib ->
base64url, prefixed with an HMAC-SHA256 of the payload (stdlib hmac, compare_digest). It is SIGNED,
not encrypted: it carries the employee's own words, so never log it or its contents.

What is stored: the facts the employee stated (details, with current as a car_id+participation or a
waiver), the alternative last compared, the representative questions, the car-selection guard sets,
the committed turn number, rules.version, an issued-at and a schema version, and the transcript
restricted to user and assistant TEXT. Tool calls, tool results and provider_state are dropped;
last_comparison is dropped too (it is recomputed on the next comparison). Grounding reads the
employee's own messages, so those are preserved exactly.

Even though a token is signed, restore() re-validates every field against the catalogue and the
rules before trusting it: a signing key could be shared across environments, and a bug must not turn
a malformed token into a bad state.
"""

import base64
import hmac
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from app.agent.grounding import matching_occupations
from app.agent.llm import Message, Role
from app.agent.state import (
    ConversationState,
    EmployeeDetails,
    QuestionStatus,
    RepQuestion,
    RepTopic,
)
from app.agent.tools import ToolContext
from app.comparison import CompanyCar, Situation, Waiver

SCHEMA_VERSION = 1
_SESSION_ID = re.compile(r"[0-9a-f]{32}")
TOKEN_MAX_AGE_SECONDS = 30 * 24 * 60 * 60  # 30 days; older tokens are refused as expired

# Size caps: a token is employee-authored, so bound it before trusting any of it.
MAX_TOKEN_CHARS = 64 * 1024  # the base64url string the client sends back
MAX_SNAPSHOT_BYTES = 64 * 1024  # the inflated JSON, to bound decompression
MAX_MESSAGES = 200  # a long but finite transcript
MAX_MESSAGE_CHARS = 4000  # two employee messages back to back never approach this

# Numeric ranges, mirroring the tools' own checks (app.agent.tools._convert_detail): fractions for
# rates, counts >= 0, money >= 0. Validation only; nothing is calculated here.
_MAX_RATE = 1.0  # pension / training-fund fractions (6% is stored as 0.06)
_MAX_MONEY = 10_000_000  # base salary, commissions, bonus, participation: a sane upper bound
_MAX_CHILDREN = 50
_MAX_CREDIT_POINTS = 100


class SnapshotError(Exception):
    """A token could not be turned into a trustworthy state. The reason is a category, never shown."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


# --- Serialize ----------------------------------------------------------------------------------

def snapshot(state: ConversationState, rules_version: str, turn: int, issued_at: int) -> dict[str, Any]:
    """A plain JSON-compatible dict of the resumable parts of state (see module docstring)."""
    return {
        "v": SCHEMA_VERSION,
        "rules_version": rules_version,
        "iat": issued_at,
        "turn": turn,  # committed turns so far; travels with the state so a restore resumes the count
        # The session it was issued for: a restore reuses it when free, so the representative's queue
        # keeps the same conversation. Signed like the rest, so a token cannot claim another session.
        "sid": state.conversation_id,
        "details": _details_out(state.details),
        "alternative": _situation_out(state.alternative),
        "rep_questions": [_question_out(q) for q in state.rep_questions],
        "usable_car_ids": sorted(state.usable_car_ids),
        "offered_car_ids": sorted(state.offered_car_ids),
        "messages": [
            {"role": m.role.value, "text": m.text}
            for m in state.messages
            if m.role in (Role.USER, Role.ASSISTANT) and m.text
        ],
    }


def _details_out(details: EmployeeDetails) -> dict[str, Any]:
    return {
        "occupation": details.occupation,
        "base_salary": details.base_salary,
        "commissions": details.commissions,
        "bonus": details.bonus,
        "pension_rate": details.pension_rate,
        "training_fund_rate": details.training_fund_rate,
        "credit_points": details.credit_points,
        "eligible_children": details.eligible_children,
        "current": _situation_out(details.current),
    }


def _situation_out(situation: Situation | None) -> dict[str, Any] | None:
    if situation is None:
        return None
    if isinstance(situation, Waiver):
        return {"kind": "waiver"}
    return {"kind": "company_car", "car_id": situation.car_id, "participation": situation.participation}


def _question_out(q: RepQuestion) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "topic": q.topic.value,
        "text": q.text,
        "car_id": q.car_id,
        "status": q.status.value,
        "answer": q.answer,
        "participation": q.participation,
    }


# --- Token --------------------------------------------------------------------------------------

def encode_token(state: ConversationState, rules_version: str, turn: int, issued_at: int, key: bytes) -> str | None:
    """A signed, compressed token for state, or None if it would exceed MAX_TOKEN_CHARS (omit it)."""
    data = snapshot(state, rules_version, turn, issued_at)
    payload = zlib.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    body = _b64encode(payload)
    token = f"{_b64encode(_sign(payload, key))}.{body}"
    return token if len(token) <= MAX_TOKEN_CHARS else None


def decode_token(token: str, key: bytes, ctx: ToolContext, rules_version: str, now: int) -> tuple[ConversationState, int]:
    """(state, turn) from a valid token, or raise SnapshotError with a category (never the cause)."""
    if not token or len(token) > MAX_TOKEN_CHARS:
        raise SnapshotError("oversized")
    signature_b64, _, body_b64 = token.partition(".")
    if not signature_b64 or not body_b64:
        raise SnapshotError("malformed")
    try:
        signature = _b64decode(signature_b64)
        payload = _b64decode(body_b64)
    except (ValueError, TypeError):
        raise SnapshotError("malformed") from None
    if not hmac.compare_digest(signature, _sign(payload, key)):
        raise SnapshotError("bad_signature")
    data = _inflate(payload)
    return _restore(data, ctx, rules_version, now)


def _sign(payload: bytes, key: bytes) -> bytes:
    return hmac.new(key, payload, sha256).digest()


def _inflate(payload: bytes) -> dict[str, Any]:
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(payload, MAX_SNAPSHOT_BYTES + 1)
    except zlib.error:
        raise SnapshotError("malformed") from None
    if len(raw) > MAX_SNAPSHOT_BYTES or decompressor.unconsumed_tail:
        raise SnapshotError("oversized")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise SnapshotError("malformed") from None
    if not isinstance(data, dict):
        raise SnapshotError("malformed")
    return data


# --- Restore ------------------------------------------------------------------------------------

def _restore(data: Mapping[str, Any], ctx: ToolContext, rules_version: str, now: int) -> tuple[ConversationState, int]:
    if data.get("v") != SCHEMA_VERSION:
        raise SnapshotError("schema_version")
    if data.get("rules_version") != rules_version:
        raise SnapshotError("rules_version")
    issued_at = _int(data.get("iat"), "iat", minimum=0)
    if now - issued_at > TOKEN_MAX_AGE_SECONDS:
        raise SnapshotError("expired")

    turn = _int(data.get("turn"), "turn", minimum=0)
    details = _restore_details(data.get("details"), ctx)
    alternative = _restore_situation(data.get("alternative"), ctx)
    questions = _restore_questions(data.get("rep_questions"))
    usable = _restore_car_ids(data.get("usable_car_ids"), ctx)
    offered = _restore_car_ids(data.get("offered_car_ids"), ctx)
    messages = _restore_messages(data.get("messages"))

    # conversation_id: the session the token was issued for (empty for older tokens); the caller
    # reuses it when no live session has it, else gives the state a new id.
    sid = data.get("sid")
    state = ConversationState(
        conversation_id=sid if isinstance(sid, str) and _SESSION_ID.fullmatch(sid) else "",
        details=details,
        alternative=alternative,
        rep_questions=questions,
        messages=messages,
        usable_car_ids=usable,
        offered_car_ids=offered,
    )
    return state, turn


def _restore_details(raw: Any, ctx: ToolContext) -> EmployeeDetails:
    if not isinstance(raw, Mapping):
        raise SnapshotError("details")
    occupation = raw.get("occupation")
    if occupation is not None:
        occupation = _resolve_occupation(occupation, ctx)
    return EmployeeDetails(
        occupation=occupation,
        base_salary=_opt_money(raw.get("base_salary"), "base_salary"),
        commissions=_opt_money(raw.get("commissions"), "commissions"),
        bonus=_opt_money(raw.get("bonus"), "bonus"),
        pension_rate=_opt_rate(raw.get("pension_rate"), "pension_rate"),
        training_fund_rate=_opt_rate(raw.get("training_fund_rate"), "training_fund_rate"),
        credit_points=_opt_number(raw.get("credit_points"), "credit_points", _MAX_CREDIT_POINTS),
        eligible_children=_opt_int(raw.get("eligible_children"), "eligible_children", _MAX_CHILDREN),
        current=_restore_situation(raw.get("current"), ctx),
    )


def _resolve_occupation(value: Any, ctx: ToolContext) -> str:
    """The canonical workbook name, exactly as the tools accept it (app.agent.tools._convert_detail)."""
    if isinstance(value, str) and value in ctx.catalog.occupations:
        return value
    matches = matching_occupations(value, ctx.catalog.occupations) if isinstance(value, str) else ()
    if len(matches) == 1:
        return matches[0]
    raise SnapshotError("occupation")


def _restore_situation(raw: Any, ctx: ToolContext) -> Situation | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise SnapshotError("situation")
    kind = raw.get("kind")
    if kind == "waiver":
        return Waiver()
    if kind == "company_car":
        car_id = raw.get("car_id")
        if not isinstance(car_id, str) or car_id not in ctx.catalog.cars:
            raise SnapshotError("car_id")
        return CompanyCar(car_id, _opt_money(raw.get("participation"), "participation"))
    raise SnapshotError("situation")


def _restore_questions(raw: Any) -> tuple[RepQuestion, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SnapshotError("rep_questions")
    questions = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise SnapshotError("rep_questions")
        questions.append(
            RepQuestion(
                question_id=_text(item.get("question_id"), "question_id"),
                topic=_enum(item.get("topic"), RepTopic, "topic"),
                text=_text(item.get("text"), "question_text"),
                car_id=_opt_text(item.get("car_id"), "question_car_id"),
                status=_enum(item.get("status"), QuestionStatus, "status"),
                answer=_opt_text(item.get("answer"), "answer"),
                participation=_opt_money(item.get("participation"), "question_participation"),
            )
        )
    return tuple(questions)


def _restore_car_ids(raw: Any, ctx: ToolContext) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SnapshotError("car_ids")
    ids = set()
    for car_id in raw:
        if not isinstance(car_id, str) or car_id not in ctx.catalog.cars:
            raise SnapshotError("car_id")
        ids.add(car_id)
    return frozenset(ids)


def _restore_messages(raw: Any) -> tuple[Message, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SnapshotError("messages")
    if len(raw) > MAX_MESSAGES:
        raise SnapshotError("too_many_messages")
    messages = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise SnapshotError("messages")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            raise SnapshotError("message_text")
        role = item.get("role")
        if role == Role.USER.value:
            messages.append(Message.user(text))
        elif role == Role.ASSISTANT.value:
            messages.append(Message.assistant(text))
        else:
            raise SnapshotError("role")
    return tuple(messages)


# --- Scalars ------------------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _opt_number(value: Any, where: str, maximum: float) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or value < 0 or value > maximum:
        raise SnapshotError(where)
    return float(value)


def _opt_money(value: Any, where: str) -> float | None:
    return _opt_number(value, where, _MAX_MONEY)


def _opt_rate(value: Any, where: str) -> float | None:
    return _opt_number(value, where, _MAX_RATE)


def _opt_int(value: Any, where: str, maximum: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
        raise SnapshotError(where)
    return value


def _int(value: Any, where: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SnapshotError(where)
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_MESSAGE_CHARS:
        raise SnapshotError(where)
    return value


def _opt_text(value: Any, where: str) -> str | None:
    return None if value is None else _text(value, where)


def _enum(value: Any, enum: type, where: str):
    try:
        return enum(value)
    except ValueError:
        raise SnapshotError(where) from None


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)
