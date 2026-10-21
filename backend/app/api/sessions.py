"""In-memory conversations for one backend process (hackathon scope): no disk, no sharing.

The browser holds only the session ID; the ConversationState stays here. A session runs one turn
at a time: its lock is taken without waiting, so a second request during a turn is refused.
"""

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from app.agent.outcome import TurnOutcome
from app.agent.state import ConversationState

DEFAULT_TTL_SECONDS = 2 * 60 * 60
DEFAULT_MAX_SESSIONS = 500


@dataclass(eq=False)
class Session:
    session_id: str
    state: ConversationState
    touched: float
    turn: int = 0  # committed turns
    outcome: TurnOutcome | None = None  # the last committed turn's outcome, as shown (app.api.main)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._guard = threading.Lock()

    def create(self, state: ConversationState | None = None, turn: int = 0) -> Session:
        """A new session. A restored state keeps its conversation_id when free (else a fresh id) and turn."""
        with self._guard:
            self._drop_expired()
            while len(self._sessions) >= self._max:  # drop the least recently used idle session
                idle = [s for s in self._sessions.values() if not s.lock.locked()]
                if not idle:
                    break
                del self._sessions[min(idle, key=lambda s: s.touched).session_id]
            wanted = state.conversation_id if state is not None else ""
            # A restored conversation keeps its id when no live session has it (the representative's
            # queue then keeps pointing at it); otherwise, and for a new one, a fresh id.
            session_id = wanted if wanted and wanted not in self._sessions else uuid.uuid4().hex
            seeded = ConversationState(session_id) if state is None else replace(state, conversation_id=session_id)
            session = Session(session_id, seeded, self._clock(), turn=turn)
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._guard:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if self._expired(session):
                del self._sessions[session_id]
                return None
            return session

    def delete(self, session_id: str) -> None:
        with self._guard:
            self._sessions.pop(session_id, None)

    def live(self) -> list[Session]:
        """Every unexpired session, most recently used first (the representative's queue)."""
        with self._guard:
            self._drop_expired()
            return sorted(self._sessions.values(), key=lambda s: s.touched, reverse=True)

    def touch(self, session: Session) -> None:
        session.touched = self._clock()

    def __len__(self) -> int:
        return len(self._sessions)

    def _expired(self, session: Session) -> bool:
        return not session.lock.locked() and self._clock() - session.touched > self._ttl

    def _drop_expired(self) -> None:
        for session_id in [s.session_id for s in self._sessions.values() if self._expired(s)]:
            del self._sessions[session_id]
