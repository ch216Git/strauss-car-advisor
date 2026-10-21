// The representative's queue, kept in this browser (localStorage) as well, so it survives a server
// restart. The server stays the source of truth: a question it no longer has (its conversation is not
// reconnected yet) is shown as offline and cannot be answered until the employee's screen restores it.

import type { QueueQuestion } from "../api/types";

export const QUEUE_KEY = "advisor.representative_queue";
export const MAX_CACHED = 200;
export const CACHE_DAYS = 7;

export type CachedQuestion = QueueQuestion & { seenAt: number }; // last time the server listed it
export type ShownQuestion = QueueQuestion & { offline: boolean };

export const questionKey = (q: Pick<QueueQuestion, "session_id" | "question_id">) => `${q.session_id}/${q.question_id}`;

export function loadQueue(): CachedQuestion[] {
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(QUEUE_KEY) ?? "[]");
    return Array.isArray(parsed)
      ? parsed.filter((q): q is CachedQuestion =>
        !!q && typeof q.session_id === "string" && typeof q.question_id === "string" && typeof q.question === "string"
        && typeof q.seenAt === "number")
      : [];
  } catch {
    return []; // storage blocked or unreadable: the screen still works from the server
  }
}

function saveQueue(items: CachedQuestion[]): void {
  try {
    window.localStorage.setItem(QUEUE_KEY, JSON.stringify(items));
  } catch {
    // Storage full or blocked: the live queue still works.
  }
}

/** The server's questions (live) followed by cached ones it no longer lists (offline); saves the result. */
export function mergeQueue(server: QueueQuestion[], cached: CachedQuestion[], now = Date.now()): ShownQuestion[] {
  const live = server.map((q) => ({ ...q, seenAt: now }));
  const listed = new Set(live.map(questionKey));
  const oldest = now - CACHE_DAYS * 24 * 60 * 60 * 1000;
  const offline = cached.filter((q) => !listed.has(questionKey(q)) && q.seenAt >= oldest);
  const kept = [...live, ...offline].slice(0, MAX_CACHED);
  saveQueue(kept);
  return kept.map((q) => shown(q, !listed.has(questionKey(q))));
}

/** Before the server answers (or when it cannot be reached): everything cached, as offline. */
export function cachedQueue(): ShownQuestion[] {
  return loadQueue().map((q) => shown(q, true));
}

function shown(q: CachedQuestion, offline: boolean): ShownQuestion {
  const copy: Partial<CachedQuestion> = { ...q };
  delete copy.seenAt; // bookkeeping for this browser, not part of the question
  return { ...(copy as QueueQuestion), offline };
}
