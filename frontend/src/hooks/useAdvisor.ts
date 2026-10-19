// The conversation on the client: a session ID, the visible transcript and the latest decision.
// Only the session ID is stored (sessionStorage); the conversation itself lives on the server.

import { useCallback, useEffect, useRef, useState } from "react";
import { createSession, deleteSession, sendChat } from "../api/client";
import type { ChatResponse, UiState } from "../api/types";

export const SESSION_KEY = "advisor.session_id";

export type Entry =
  | { id: number; role: "employee"; text: string }
  | { id: number; role: "advisor"; text: string }
  | { id: number; role: "error"; text: string; retryText?: string };

type NewEntry = Entry extends infer E ? (E extends Entry ? Omit<E, "id"> : never) : never;

export type SessionStatus = "starting" | "ready" | "failed";

// States that change what the decision panel shows; the others leave the last decision in place.
const DECISIVE: ReadonlySet<UiState> = new Set<UiState>([
  "eligibility_check", "not_eligible", "collecting", "estimate", "confirmed", "representative", "car_choice",
]);

export const TEXT = {
  sessionLost: "השיחה הקודמת כבר לא זמינה, ולכן התחלנו שיחה חדשה. אפשר לשלוח את ההודעה שוב.",
  invalid: "ההודעה לא נשלחה כי היא אינה תקינה. אפשר לקצר אותה או לנסח מחדש.",
  busy: "ההודעה הקודמת עדיין בטיפול. אפשר לנסות שוב בעוד רגע.",
  unreachable: "אין חיבור לשרת כרגע. אפשר לנסות שוב.",
  failed: "מצטער/ת, הייתה תקלה. אפשר לנסות שוב.",
};

function readStoredSession(): string | null {
  try {
    return window.sessionStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function storeSession(sessionId: string | null): void {
  try {
    if (sessionId) window.sessionStorage.setItem(SESSION_KEY, sessionId);
    else window.sessionStorage.removeItem(SESSION_KEY);
  } catch {
    // Storage blocked: the session still works for this page.
  }
}

export function useAdvisor() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<SessionStatus>("starting");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [pending, setPending] = useState(false);
  const [blocked, setBlocked] = useState(false); // 409: sending stays off until a retry succeeds
  const [decision, setDecision] = useState<ChatResponse | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const nextId = useRef(1);
  const generation = useRef(0); // bumped on reset: late responses for an old conversation are dropped
  const started = useRef(false);
  const sessionRef = useRef<string | null>(null);
  const pendingRef = useRef(false);

  const add = useCallback((entry: NewEntry) => {
    const full = { ...entry, id: nextId.current++ } as Entry;
    setEntries((current) => [...current, full]);
  }, []);

  const clearConversation = useCallback(() => {
    generation.current += 1;
    setEntries([]);
    setDecision(null);
    setPending(false);
    pendingRef.current = false;
    setBlocked(false);
  }, []);

  const startSession = useCallback(async () => {
    const mine = generation.current;
    setStatus("starting");
    const result = await createSession();
    if (mine !== generation.current) return;
    if (!result.ok) {
      setStatus("failed");
      return;
    }
    sessionRef.current = result.data.session_id;
    setSessionId(result.data.session_id);
    storeSession(result.data.session_id);
    setStatus("ready");
  }, []);

  useEffect(() => {
    if (started.current) return; // once, also under StrictMode's double effects
    started.current = true;
    const stored = readStoredSession();
    if (stored) {
      sessionRef.current = stored;
      setSessionId(stored);
      setStatus("ready");
    } else {
      void startSession();
    }
  }, [startSession]);

  const run = useCallback(async (text: string, isRetry: boolean) => {
    const session = sessionRef.current;
    if (!session || pendingRef.current) return;
    const mine = generation.current;
    if (!isRetry) add({ role: "employee", text });
    setNotice(null);
    setPending(true);
    pendingRef.current = true;

    const result = await sendChat(session, text);
    if (mine !== generation.current) return; // the conversation was reset meanwhile
    setPending(false);
    pendingRef.current = false;

    if (result.ok) {
      const response = result.data;
      setBlocked(false);
      if (response.ui_state === "temporary_error") {
        add({ role: "error", text: response.reply, retryText: response.retryable ? text : undefined });
        return;
      }
      add({ role: "advisor", text: response.reply });
      if (DECISIVE.has(response.ui_state)) setDecision(response);
      return;
    }

    if (result.status === 404) {
      clearConversation();
      storeSession(null);
      sessionRef.current = null;
      setSessionId(null);
      setNotice(TEXT.sessionLost);
      setDraft(text);
      await startSession();
      return;
    }
    if (result.status === 422) {
      if (!isRetry) setEntries((current) => current.slice(0, -1)); // it was never processed
      setNotice(TEXT.invalid);
      setDraft(text);
      return;
    }
    if (result.status === 409) {
      setBlocked(true);
      add({ role: "error", text: result.message ?? TEXT.busy, retryText: text });
      return;
    }
    const fallback = result.status === 0 ? TEXT.unreachable : TEXT.failed;
    add({ role: "error", text: result.message ?? fallback, retryText: text });
  }, [add, clearConversation, startSession]);

  const send = useCallback((raw: string) => {
    const text = raw.trim();
    if (!text || blocked) return;
    setDraft("");
    void run(text, false);
  }, [blocked, run]);

  const retry = useCallback((entryId: number) => {
    const entry = entries.find((e) => e.id === entryId);
    if (!entry || entry.role !== "error" || !entry.retryText) return;
    setEntries((current) => current.filter((e) => e.id !== entryId)); // the failed attempt is shown once
    void run(entry.retryText, true);
  }, [entries, run]);

  const newConversation = useCallback(async () => {
    const old = sessionRef.current;
    clearConversation();
    setNotice(null);
    setDraft("");
    sessionRef.current = null;
    setSessionId(null);
    storeSession(null);
    if (old) void deleteSession(old); // best effort: a failure changes nothing for the employee
    await startSession();
  }, [clearConversation, startSession]);

  return {
    sessionId, status, entries, pending, blocked, decision, notice, draft,
    setDraft, send, retry, newConversation, retrySession: startSession,
    canSend: status === "ready" && !!sessionId && !pending && !blocked,
  };
}

export type Advisor = ReturnType<typeof useAdvisor>;
