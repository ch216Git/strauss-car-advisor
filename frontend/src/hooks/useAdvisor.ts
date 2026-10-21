// The conversation on the client: a session ID, the visible transcript and the latest decision.
// The server keeps the live session; this browser also keeps a local history (lib/history) so past
// conversations survive reloads and backend restarts. Only the active session ID is per tab
// (sessionStorage); the history list is shared across tabs (localStorage) and refreshed on changes.

import { useCallback, useEffect, useRef, useState } from "react";
import { createSession, deleteSession, restoreSession, sendChat } from "../api/client";
import type { ChatResponse, UiState } from "../api/types";
import {
  type Conversation,
  type HistoryEntry,
  HISTORY_KEY,
  deleteConversation,
  clearHistory,
  findConversation,
  historyEnabled,
  loadConversations,
  saveConversation,
  setHistoryEnabled,
  titleFrom,
} from "../lib/history";

export const SESSION_KEY = "advisor.session_id";
export const ACTIVE_KEY = "advisor.active_conversation_id";

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
  notResumable: "השיחה הזו שמורה לצפייה בלבד ואי אפשר להמשיך אותה. אפשר להתחיל שיחה חדשה.",
  storageFull: "אחסון ההיסטוריה מלא, ולכן שיחות ישנות עשויות להימחק. השיחה הנוכחית ממשיכה כרגיל.",
};

function readStored(key: string): string | null {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key: string, value: string | null): void {
  try {
    if (value) window.sessionStorage.setItem(key, value);
    else window.sessionStorage.removeItem(key);
  } catch {
    // Storage blocked: the session still works for this page.
  }
}

function newConversationId(): string {
  try {
    return crypto.randomUUID();
  } catch {
    return `c-${Date.now()}-${Math.random().toString(36).slice(2)}`; // older engines / no crypto
  }
}

function toHistoryEntries(entries: Entry[]): HistoryEntry[] {
  // Error entries are never persisted; only the employee/advisor transcript is.
  return entries.flatMap((e) => (e.role === "employee" || e.role === "advisor") ? [{ role: e.role, text: e.text }] : []);
}

export function useAdvisor() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<SessionStatus>("starting");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [pending, setPending] = useState(false);
  const [blocked, setBlocked] = useState(false); // 409: sending stays off until a retry succeeds
  const [readOnly, setReadOnly] = useState(false); // a non-resumable conversation: viewing only
  const [decision, setDecision] = useState<ChatResponse | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [enabled, setEnabled] = useState(true);

  const nextId = useRef(1);
  const generation = useRef(0); // bumped on reset: late responses for an old conversation are dropped
  const started = useRef(false);
  const sessionRef = useRef<string | null>(null);
  const pendingRef = useRef(false);
  const activeRef = useRef<string | null>(null); // the active conversation id (per tab)
  const tokenRef = useRef<string | undefined>(undefined); // latest resume token for the active conversation
  const resumableRef = useRef(true);
  const createdAtRef = useRef<number>(Date.now());
  // The decision currently on screen (the "sticky" one). Mirrors setDecision so persist and the 404
  // path use what is shown, never a stale closure value.
  const decisionRef = useRef<ChatResponse | null>(null);

  // Set the on-screen decision and keep the ref in step; pass null to clear it.
  const commitDecision = useCallback((next: ChatResponse | null) => {
    decisionRef.current = next;
    setDecision(next);
  }, []);

  const refreshList = useCallback(() => setConversations(loadConversations()), []);

  const add = useCallback((entry: NewEntry) => {
    const full = { ...entry, id: nextId.current++ } as Entry;
    setEntries((current) => [...current, full]);
  }, []);

  // One atomic history write per committed turn: the whole conversation as it now stands. The saved
  // decision is the sticky one on screen (decisionRef), so history matches the panel exactly.
  const persist = useCallback((allEntries: Entry[]) => {
    if (!historyEnabled() || !activeRef.current || !sessionRef.current) return;
    const historyEntries = toHistoryEntries(allEntries);
    if (historyEntries.length === 0) return; // never save an empty conversation
    const now = Date.now();
    const conversation: Conversation = {
      id: activeRef.current,
      title: titleFrom(historyEntries),
      createdAt: createdAtRef.current,
      updatedAt: now,
      sessionId: sessionRef.current,
      resumeToken: tokenRef.current,
      resumable: resumableRef.current && tokenRef.current !== undefined,
      entries: historyEntries,
      decision: decisionRef.current,
    };
    const result = saveConversation(conversation, activeRef.current);
    if (result === "quota" || result === "unavailable") setNotice(TEXT.storageFull);
    refreshList();
  }, [refreshList]);

  const resetLocal = useCallback(() => {
    generation.current += 1;
    setEntries([]);
    commitDecision(null);
    setPending(false);
    pendingRef.current = false;
    setBlocked(false);
    setReadOnly(false);
    resumableRef.current = true;
    tokenRef.current = undefined;
  }, [commitDecision]);

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
    store(SESSION_KEY, result.data.session_id);
    setStatus("ready");
  }, []);

  // Begin a brand-new conversation (fresh active id + server session).
  const beginFresh = useCallback(async () => {
    const id = newConversationId();
    activeRef.current = id;
    createdAtRef.current = Date.now();
    store(ACTIVE_KEY, id);
    sessionRef.current = null;
    setSessionId(null);
    store(SESSION_KEY, null);
    await startSession();
  }, [startSession]);

  useEffect(() => {
    if (started.current) return; // once, also under StrictMode's double effects
    started.current = true;
    setEnabled(historyEnabled());
    refreshList();

    const activeId = readStored(ACTIVE_KEY);
    const saved = activeId ? findConversation(activeId) : null;
    if (saved) {
      // Open the stored conversation with no network call: the panel appears instantly.
      activeRef.current = saved.id;
      createdAtRef.current = saved.createdAt;
      tokenRef.current = saved.resumeToken;
      resumableRef.current = saved.resumable;
      sessionRef.current = saved.sessionId;
      setSessionId(saved.sessionId);
      setEntries(saved.entries.map((e) => ({ ...e, id: nextId.current++ })));
      commitDecision(saved.decision);
      setReadOnly(!saved.resumable);
      setStatus("ready");
      return;
    }
    const stored = readStored(SESSION_KEY);
    if (stored) {
      // A live session from this tab (possibly from before history existed): reuse it, no network.
      const id = activeId ?? newConversationId();
      activeRef.current = id;
      store(ACTIVE_KEY, id);
      sessionRef.current = stored;
      setSessionId(stored);
      setStatus("ready");
    } else {
      void beginFresh();
    }
  }, [beginFresh, commitDecision, refreshList]);

  // Another tab changed the shared history: refresh the list (and the toggle).
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === HISTORY_KEY || event.key === null) refreshList();
      if (event.key === null) setEnabled(historyEnabled());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [refreshList]);

  const run = useCallback(async (text: string, isRetry: boolean) => {
    const session = sessionRef.current;
    if (!session || pendingRef.current || readOnly) return;
    const mine = generation.current;
    if (!isRetry) add({ role: "employee", text });
    setNotice(null);
    setPending(true);
    pendingRef.current = true;

    let result = await sendChat(session, text);

    // Session gone on the server: if we can resume from the stored token, do it once and retry.
    if (!result.ok && result.status === 404 && tokenRef.current) {
      const restored = await restoreSession(tokenRef.current);
      if (mine !== generation.current) return;
      if (restored.ok) {
        sessionRef.current = restored.data.session_id;
        setSessionId(restored.data.session_id);
        store(SESSION_KEY, restored.data.session_id);
        result = await sendChat(restored.data.session_id, text);
      } else if (restored.status === 410) {
        setPending(false);
        pendingRef.current = false;
        resumableRef.current = false;
        setReadOnly(true);
        setNotice(TEXT.notResumable);
        return;
      }
    }

    if (mine !== generation.current) return; // the conversation was switched/reset meanwhile
    setPending(false);
    pendingRef.current = false;

    if (result.ok) {
      const response = result.data;
      setBlocked(false);
      if (response.ui_state === "temporary_error") {
        // A temporary error commits nothing and keeps the previous token; nothing new is persisted.
        add({ role: "error", text: response.reply, retryText: response.retryable ? text : undefined });
        return;
      }
      const advisorEntry: Entry = { id: nextId.current++, role: "advisor", text: response.reply };
      // A decisive turn updates the sticky decision; others keep the one already on screen (the ref).
      if (DECISIVE.has(response.ui_state)) commitDecision(response);
      if (response.resume_token !== undefined) tokenRef.current = response.resume_token;
      else resumableRef.current = false; // the server omitted a token (oversized): mark not resumable
      setEntries((current) => {
        const all = [...current, advisorEntry];
        persist(all); // one atomic write: employee message + reply + the on-screen decision + token
        return all;
      });
      return;
    }

    if (result.status === 404) {
      // No token to resume from: start a new server session and ask the employee to resend. Keep the
      // decision that was on screen (from the ref, or the saved conversation) instead of blanking it.
      const kept = decisionRef.current ?? (activeRef.current ? findConversation(activeRef.current)?.decision ?? null : null);
      resetLocal();
      commitDecision(kept);
      void beginFresh();
      setNotice(TEXT.sessionLost);
      setDraft(text);
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
  }, [add, beginFresh, commitDecision, persist, readOnly, resetLocal]);

  const send = useCallback((raw: string) => {
    const text = raw.trim();
    if (!text || blocked || readOnly) return;
    setDraft("");
    void run(text, false);
  }, [blocked, readOnly, run]);

  const retry = useCallback((entryId: number) => {
    const entry = entries.find((e) => e.id === entryId);
    if (!entry || entry.role !== "error" || !entry.retryText) return;
    setEntries((current) => current.filter((e) => e.id !== entryId)); // the failed attempt is shown once
    void run(entry.retryText, true);
  }, [entries, run]);

  const newConversation = useCallback(async () => {
    if (pendingRef.current) return; // disabled while a request is in flight
    const old = sessionRef.current;
    resetLocal();
    setNotice(null);
    setDraft("");
    await beginFresh();
    if (old) void deleteSession(old); // best effort: a failure changes nothing for the employee
  }, [beginFresh, resetLocal]);

  // Open a stored conversation: no network call, panel and transcript from local data.
  const openConversation = useCallback((id: string) => {
    if (pendingRef.current) return;
    const saved = findConversation(id);
    if (!saved) return;
    resetLocal();
    setNotice(null);
    setDraft("");
    activeRef.current = saved.id;
    createdAtRef.current = saved.createdAt;
    tokenRef.current = saved.resumeToken;
    resumableRef.current = saved.resumable;
    sessionRef.current = saved.sessionId;
    setSessionId(saved.sessionId);
    store(ACTIVE_KEY, saved.id);
    store(SESSION_KEY, saved.sessionId);
    setEntries(saved.entries.map((e) => ({ ...e, id: nextId.current++ })));
    commitDecision(saved.decision);
    setReadOnly(!saved.resumable);
    setStatus("ready");
  }, [commitDecision, resetLocal]);

  const removeConversation = useCallback((id: string) => {
    if (pendingRef.current) return;
    const saved = findConversation(id);
    deleteConversation(id);
    refreshList();
    if (saved?.sessionId) void deleteSession(saved.sessionId); // best effort
    if (id === activeRef.current) void newConversation(); // the open one was deleted: start fresh
  }, [newConversation, refreshList]);

  const removeAll = useCallback(() => {
    if (pendingRef.current) return;
    const sessions = loadConversations().map((c) => c.sessionId);
    clearHistory();
    refreshList();
    for (const s of sessions) if (s) void deleteSession(s); // best effort
    void newConversation();
  }, [newConversation, refreshList]);

  const toggleHistory = useCallback((on: boolean) => {
    setHistoryEnabled(on);
    setEnabled(on);
    if (!on) {
      clearHistory(); // turning off stops writes and removes what is stored
      refreshList();
    } else {
      persist(entries); // turning on saves the current conversation, if any
    }
  }, [entries, persist, refreshList]);

  return {
    sessionId, status, entries, pending, blocked, readOnly, decision, notice, draft,
    conversations, activeId: activeRef.current, historyEnabled: enabled,
    setDraft, send, retry, newConversation, retrySession: startSession,
    openConversation, removeConversation, removeAll, toggleHistory,
    canSend: status === "ready" && !!sessionId && !pending && !blocked && !readOnly,
    busy: pending,
  };
}

export type Advisor = ReturnType<typeof useAdvisor>;
