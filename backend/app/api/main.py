"""The HTTP API (Step 5): sessions, one chat turn per request, the representative's queue, health.

Run from backend/:  uvicorn app.api.main:app

The agent loop (run_turn) is called unchanged; every decision, figure and reply comes from it.
This layer keeps conversations in memory, takes one turn at a time per session, commits a turn's
state atomically (never after a provider failure) and maps the result to the public contract.
Responses carry no tool, provider or debug data and no exception text; logs carry no values.
"""

import logging
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import replace

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.agent import TurnStop, run_turn
from app.agent.llm import LLMProvider, ProviderError
from app.agent.outcome import carried_forward
from app.agent.snapshot import SnapshotError, decode_token, encode_token
from app.agent.tools import ToolContext
from app.agent.document import FIELDS, MAX_FILE_BYTES, Document, DocumentError, DocumentReader, read_details
from app.agent.representative import AnswerError, answer_question
from app.api.mapping import catalogue, chat_response, error_response, queue_questions, representative_status
from app.api.schemas import (
    AnswerRequest,
    CatalogueOut,
    ChatRequest,
    ChatResponse,
    DocumentOut,
    ErrorDetail,
    FoundDetailOut,
    ErrorOut,
    HealthOut,
    QueueOut,
    QueueQuestionOut,
    RepresentativeStatusOut,
    RestoreRequest,
    SessionCreated,
    SessionRestored,
    UnclearDetailOut,
)
from app.api.sessions import SessionStore
from app.env_file import load_env_files
from pathlib import Path
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("app.api")

DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")  # the Vite dev server
HISTORY_KEY_VARIABLE = "HISTORY_SIGNING_KEY"

MESSAGES = {
    "session_not_found": "השיחה לא נמצאה. אפשר להתחיל שיחה חדשה.",
    "turn_in_progress": "ההודעה הקודמת עדיין בטיפול. אפשר לשלוח שוב בעוד רגע.",
    "invalid_request": "הבקשה אינה תקינה.",
    "advisor_unavailable": "היועץ אינו זמין כרגע.",
    "internal_error": "מצטער/ת, הייתה תקלה פנימית. אפשר לנסות שוב.",
    "resume_unavailable": "לא ניתן לשחזר את השיחה הזו. אפשר להתחיל שיחה חדשה.",
    "question_not_found": "השאלה לא נמצאה.",
    "already_answered": "כבר נשלחה תשובה לשאלה הזו.",
    "invalid_answer": "התשובה אינה מתאימה לשאלה. בדקו את השדות המסומנים.",
    "file_too_large": "הקובץ גדול מדי. אפשר להעלות קובץ עד 5MB.",
    "unsupported_file": "סוג הקובץ אינו נתמך. אפשר להעלות PDF, תמונה (PNG/JPG/WEBP), TXT, CSV או Excel.",
    "unreadable_file": "לא הצלחנו לקרוא את הקובץ. אפשר לנסות קובץ אחר או לכתוב את הפרטים בשיחה.",
    "document_unavailable": "קריאת הקובץ נכשלה כרגע. אפשר לנסות שוב בעוד רגע או לכתוב את הפרטים בשיחה.",
}


def history_signing_key() -> bytes:
    """The HMAC key for resume tokens: from the env (loaded from .env first), else ephemeral.

    load_env_files() must run before reading the variable, or a key in backend/.env is never seen
    and every restart invalidates existing history. An ephemeral key is per-process by design.
    """
    load_env_files()
    configured = os.environ.get(HISTORY_KEY_VARIABLE, "").strip()
    if configured:
        return configured.encode("utf-8")
    log.warning("%s is not set: resume tokens use an ephemeral key and will not survive a restart", HISTORY_KEY_VARIABLE)
    return secrets.token_bytes(32)


class ApiError(Exception):
    def __init__(self, status: int, code: str, fields: list[str] | None = None) -> None:
        super().__init__(code)
        self.status, self.code, self.fields = status, code, fields


def _gemini_reader() -> DocumentReader:
    from app.agent.providers.gemini import GeminiDocumentReader, GeminiSettings

    load_env_files()
    return GeminiDocumentReader(GeminiSettings.from_env(), events=lambda line: log.warning("gemini: %s", line))


def _gemini() -> LLMProvider:
    from app.agent.providers.gemini import GeminiProvider, GeminiSettings
    from app.env_file import load_env_files

    load_env_files()
    # Value-free lines only: retries, failure categories and token counts.
    return GeminiProvider(GeminiSettings.from_env(), events=lambda line: log.warning("gemini: %s", line))


class _Lazy:
    """Built on first use, once: the app starts (and answers health) without a key or loaded data."""

    def __init__(self, build: Callable[[], object]) -> None:
        self._build, self._value, self._lock = build, None, threading.Lock()

    def get(self):
        with self._lock:
            if self._value is None:
                self._value = self._build()
            return self._value


def create_app(
    provider: LLMProvider | None = None,
    ctx: ToolContext | None = None,
    store: SessionStore | None = None,
    cors_origins: tuple[str, ...] = DEV_ORIGINS,
    signing_key: bytes | None = None,
    document_reader: DocumentReader | None = None,
    clock: Callable[[], int] = lambda: int(time.time()),
) -> FastAPI:
    providers = _Lazy(lambda: provider if provider is not None else _gemini())
    contexts = _Lazy(lambda: ctx if ctx is not None else ToolContext.load())
    readers = _Lazy(lambda: document_reader if document_reader is not None else _gemini_reader())
    sessions = store if store is not None else SessionStore()
    # Resolved once at start-up, after .env is loaded (see history_signing_key); never logged.
    key = signing_key if signing_key is not None else history_signing_key()

    app = FastAPI(title="Strauss car advisor API", version="1")
    app.state.sessions = sessions
    app.add_middleware(
        CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(ApiError)
    def _api_error(request: Request, error: ApiError) -> JSONResponse:
        return _error_json(error.status, error.code, error.fields)

    @app.exception_handler(RequestValidationError)
    def _invalid(request: Request, error: RequestValidationError) -> JSONResponse:
        # Field names only: the default body would echo the request, including the employee's message.
        names = sorted({".".join(str(part) for part in e["loc"] if part != "body") or "body" for e in error.errors()})
        return _error_json(422, "invalid_request", names)

    @app.get("/api/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut()

    @app.post("/api/sessions", response_model=SessionCreated, status_code=201)
    def create_session() -> SessionCreated:
        return SessionCreated(session_id=sessions.create().session_id)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        sessions.delete(session_id)
        return Response(status_code=204)

    @app.post(
        "/api/chat", response_model=ChatResponse, response_model_exclude_none=True,
        responses={404: {"model": ErrorOut}, 409: {"model": ErrorOut}, 422: {"model": ErrorOut},
                   500: {"model": ErrorOut}, 503: {"model": ErrorOut}},
    )
    def chat(request: ChatRequest, response: Response) -> ChatResponse:
        session = sessions.get(request.session_id)
        if session is None:
            raise ApiError(404, "session_not_found")
        if not session.lock.acquire(blocking=False):
            raise ApiError(409, "turn_in_progress")
        try:
            try:
                advisor = providers.get()
            except ProviderError:  # configuration (no key, no SDK): not the employee's to retry
                log.warning("advisor unavailable: provider could not be created")
                raise ApiError(503, "advisor_unavailable") from None
            try:
                tool_context = contexts.get()
                turn = run_turn(
                    session.state,
                    request.message,
                    advisor,
                    tool_context,
                    events=lambda event: log.warning("agent: %s", event),
                )
                if turn.stop == TurnStop.PROVIDER_ERROR:
                    # Atomic turn: the state run_turn returned (with this message and the error reply)
                    # is dropped and the turn number stays, so a retry starts from the same state.
                    return error_response(session.session_id, session.turn, turn)
                if turn.outcome is None and turn.stop in (TurnStop.REPLIED, TurnStop.RENDERED):
                    # Details saved without a new decision: still show what is missing, as of now.
                    carried = carried_forward(session.outcome, session.state.details, turn.state, tool_context.rules)
                    turn = replace(turn, outcome=carried)
                committed_turn = session.turn + 1
                result = chat_response(session.session_id, committed_turn, turn, tool_context)
                token = encode_token(turn.state, tool_context.rules.version, committed_turn, clock(), key)
            except Exception as e:  # never the exception text: it may contain employee values
                log.error("internal error in turn: %s", type(e).__name__)
                raise ApiError(500, "internal_error") from None
            session.state = turn.state  # committed only once the whole response is built
            session.turn = committed_turn
            session.outcome = turn.outcome
            if token is not None:  # omitted when oversized: the client marks it not resumable
                response.headers["Cache-Control"] = "no-store"  # a token is employee state; never cache it
                return result.model_copy(update={"resume_token": token})
            return result
        finally:
            sessions.touch(session)
            session.lock.release()

    @app.post(
        "/api/sessions/restore", response_model=SessionRestored, status_code=201,
        responses={410: {"model": ErrorOut}, 422: {"model": ErrorOut}},
    )
    def restore_session(request: RestoreRequest) -> SessionRestored:
        tool_context = contexts.get()
        try:
            state, turn = decode_token(request.resume_token, key, tool_context, tool_context.rules.version, clock())
        except SnapshotError as error:
            log.warning("resume rejected: category=%s", error.category)  # category only; never the token
            raise ApiError(410, "resume_unavailable") from None
        session = sessions.create(state=state, turn=turn)
        return SessionRestored(session_id=session.session_id, turn=session.turn)

    # --- Details from an uploaded file ----------------------------------------------------------

    @app.post(
        "/api/sessions/{session_id}/document", response_model=DocumentOut,
        responses={404: {"model": ErrorOut}, 413: {"model": ErrorOut}, 415: {"model": ErrorOut},
                   422: {"model": ErrorOut}, 503: {"model": ErrorOut}},
    )
    def read_document(session_id: str, response: Response, file: UploadFile = File(...)) -> DocumentOut:
        """Reads the file and returns only what is certain. Saves nothing: the employee confirms in the chat.
        The file is never stored, and neither it nor its content is logged."""
        if sessions.get(session_id) is None:
            raise ApiError(404, "session_not_found")
        data = file.file.read(MAX_FILE_BYTES + 1)
        document = Document(file.filename or "", data)
        try:
            reader = readers.get()
        except ProviderError:
            log.warning("document reader unavailable: provider could not be created")
            raise ApiError(503, "advisor_unavailable") from None
        try:
            details = read_details(document, reader, contexts.get())
        except DocumentError as error:
            status = {"file_too_large": 413, "unsupported_file": 415}.get(error.code, 422)
            log.warning("document rejected: %s", error.code)
            raise ApiError(status, error.code) from None
        except ProviderError as error:
            log.warning("document read failed: category=%s", error.category.value)
            raise ApiError(503, "document_unavailable") from None
        log.warning("document read: found=%d unclear=%d", len(details.found), len(details.unclear))  # counts only
        response.headers["Cache-Control"] = "no-store"
        return DocumentOut(
            found=[FoundDetailOut(field=f.field, label=f.label, value=f.value) for f in details.found],
            unclear=[UnclearDetailOut(field=f, label=FIELDS[f]) for f in details.unclear],
            from_image=details.from_image,
        )

    # --- The transport representative ---------------------------------------------------------
    # No authentication, like the rest of this hackathon API: before production these routes need
    # a representative login, since the queue shows questions from every conversation.

    @app.get("/api/sessions/{session_id}/representative", response_model=RepresentativeStatusOut,
             response_model_exclude_none=True, responses={404: {"model": ErrorOut}})
    def representative(session_id: str, response: Response) -> RepresentativeStatusOut:
        """The employee's screen polls this between turns to see the representative's answers (and opens
        with it to learn whether the server still has the conversation)."""
        session = sessions.get(session_id)
        if session is None:
            raise ApiError(404, "session_not_found")
        response.headers["Cache-Control"] = "no-store"  # the token is employee state; never cache it
        state, rules = session.state, contexts.get().rules
        status = representative_status(state)
        token = encode_token(state, rules.version, session.turn, clock(), key)
        return RepresentativeStatusOut(**status.model_dump(), resume_token=token)

    @app.get("/api/representative/questions", response_model=QueueOut, response_model_exclude_none=True)
    def representative_queue(response: Response) -> QueueOut:
        tool_context = contexts.get()
        questions = [q for s in sessions.live() for q in queue_questions(s.session_id, s.state, tool_context)]
        response.headers["Cache-Control"] = "no-store"
        return QueueOut(questions=sorted(questions, key=lambda q: q.status != "open"))  # stable: keeps recency

    @app.get("/api/representative/cars", response_model=CatalogueOut)
    def representative_cars() -> CatalogueOut:
        return catalogue(contexts.get())

    @app.post(
        "/api/representative/answers", response_model=QueueQuestionOut, response_model_exclude_none=True,
        responses={404: {"model": ErrorOut}, 409: {"model": ErrorOut}, 422: {"model": ErrorOut}},
    )
    def representative_answer(request: AnswerRequest) -> QueueQuestionOut:
        session = sessions.get(request.session_id)
        if session is None:
            raise ApiError(404, "session_not_found")
        if not session.lock.acquire(blocking=False):  # never while the employee's turn is running
            raise ApiError(409, "turn_in_progress")
        try:
            tool_context = contexts.get()
            try:
                state = answer_question(
                    session.state, tool_context, request.question_id, request.answer,
                    participation=request.participation, car_id=request.car_id,
                )
            except AnswerError as error:
                status = {"question_not_found": 404, "already_answered": 409}.get(error.code, 422)
                raise ApiError(status, error.code, list(error.fields) or None) from None
            session.state = state
            log.warning("representative answered a question")  # no values
            return next(q for q in queue_questions(session.session_id, state, tool_context)
                        if q.question_id == request.question_id)
        finally:
            sessions.touch(session)
            session.lock.release()

    return app


def _error_json(status: int, code: str, fields: list[str] | None = None) -> JSONResponse:
    body = ErrorOut(error=ErrorDetail(code=code, message=MESSAGES[code], fields=fields))
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


app = create_app()

_WEB_DIR = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
