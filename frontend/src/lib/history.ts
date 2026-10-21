// Local conversation history, kept only in this browser profile (localStorage). No server storage.
// Every access is wrapped: storage may be blocked, full, or hold corrupt/old data; none of that
// may crash the app. Unknown or wrong-version data is ignored. The token stored here is signed
// server state (the employee's own words) - it never leaves this browser except back to the API.

import type { ChatResponse } from "../api/types";

export const HISTORY_KEY = "advisor.history.v1";
export const ENABLED_KEY = "advisor.history.enabled.v1";
export const HISTORY_VERSION = 1;
export const MAX_CONVERSATIONS = 30;
export const TITLE_MAX = 40;

// A stored turn, employee and advisor text only (error entries are never persisted).
export type HistoryEntry =
  | { role: "employee"; text: string }
  | { role: "advisor"; text: string };

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  sessionId: string;
  resumeToken?: string;
  resumable: boolean; // false once a token was missing (oversized) or a restore was refused
  entries: HistoryEntry[];
  decision: ChatResponse | null; // the last decisive response, for an instant panel on open
}

interface Store {
  version: number;
  conversations: Conversation[];
}

// --- Enabled toggle (default on) ---------------------------------------------------------------

export function historyEnabled(): boolean {
  try {
    return window.localStorage.getItem(ENABLED_KEY) !== "0";
  } catch {
    return true; // storage blocked: behave as enabled; writes below will simply no-op
  }
}

export function setHistoryEnabled(enabled: boolean): void {
  try {
    window.localStorage.setItem(ENABLED_KEY, enabled ? "1" : "0");
  } catch {
    // Nothing we can do; the in-memory UI state still reflects the choice.
  }
}

// --- Read --------------------------------------------------------------------------------------

export function loadConversations(): Conversation[] {
  const store = readStore();
  return store ? [...store.conversations].sort((a, b) => b.updatedAt - a.updatedAt) : [];
}

export function findConversation(id: string): Conversation | null {
  return loadConversations().find((c) => c.id === id) ?? null;
}

// --- Write -------------------------------------------------------------------------------------

export type SaveResult = "saved" | "skipped" | "quota" | "unavailable";

/** Insert or replace a conversation and persist. Never saves an empty one. Returns what happened. */
export function saveConversation(conversation: Conversation, activeId: string | null): SaveResult {
  if (!historyEnabled()) return "skipped";
  if (conversation.entries.length === 0) return "skipped"; // never persist an empty conversation

  const current = readStore()?.conversations ?? [];
  const without = current.filter((c) => c.id !== conversation.id);
  let next = [conversation, ...without].sort((a, b) => b.updatedAt - a.updatedAt);
  if (next.length > MAX_CONVERSATIONS) next = evict(next, MAX_CONVERSATIONS, activeId);

  // On quota, drop the oldest non-active conversations one at a time and retry.
  for (;;) {
    const result = writeStore({ version: HISTORY_VERSION, conversations: next });
    if (result !== "quota") return result;
    const trimmed = evict(next, next.length - 1, activeId);
    if (trimmed.length === next.length) return "quota"; // nothing left to drop but the active one
    next = trimmed;
  }
}

export function deleteConversation(id: string): void {
  const store = readStore();
  if (!store) return;
  writeStore({ version: HISTORY_VERSION, conversations: store.conversations.filter((c) => c.id !== id) });
}

export function clearHistory(): void {
  try {
    window.localStorage.removeItem(HISTORY_KEY);
  } catch {
    // Storage blocked: there was nothing we could have written anyway.
  }
}

/** First employee message, trimmed to a short single line. */
export function titleFrom(entries: HistoryEntry[]): string {
  const first = entries.find((e) => e.role === "employee")?.text ?? "";
  const line = first.replace(/\s+/g, " ").trim();
  return line.length > TITLE_MAX ? `${line.slice(0, TITLE_MAX).trimEnd()}…` : line;
}

// --- Internals ---------------------------------------------------------------------------------

function evict(conversations: Conversation[], target: number, activeId: string | null): Conversation[] {
  // Keep the active conversation and the most recently updated; drop the oldest others first.
  const ordered = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const kept = ordered.filter((c) => c.id === activeId);
  for (const c of ordered) {
    if (kept.length >= target) break;
    if (c.id !== activeId) kept.push(c);
  }
  return kept.sort((a, b) => b.updatedAt - a.updatedAt);
}

function readStore(): Store | null {
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(HISTORY_KEY);
  } catch {
    return null; // storage blocked
  }
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isStore(parsed)) return null; // corrupt or a future/older version: ignore safely
    return parsed;
  } catch {
    return null; // not JSON
  }
}

function writeStore(store: Store): SaveResult {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(store));
    return "saved";
  } catch (error) {
    if (isQuotaError(error)) return "quota";
    return "unavailable"; // storage blocked or otherwise unwritable
  }
}

function isQuotaError(error: unknown): boolean {
  // Named differently across browsers (and a DOMException is not always an Error): match by name/code.
  if (typeof error !== "object" || error === null) return false;
  const { name, code } = error as { name?: string; code?: number };
  return name === "QuotaExceededError" || name === "NS_ERROR_DOM_QUOTA_REACHED" || code === 22 || code === 1014;
}

// --- Shape guards (hand-written: never trust stored JSON) --------------------------------------

function isStore(value: unknown): value is Store {
  if (!isObject(value)) return false;
  if (value.version !== HISTORY_VERSION) return false;
  return Array.isArray(value.conversations) && value.conversations.every(isConversation);
}

export function isConversation(value: unknown): value is Conversation {
  if (!isObject(value)) return false;
  return (
    typeof value.id === "string" &&
    typeof value.title === "string" &&
    typeof value.createdAt === "number" &&
    typeof value.updatedAt === "number" &&
    typeof value.sessionId === "string" &&
    (value.resumeToken === undefined || typeof value.resumeToken === "string") &&
    typeof value.resumable === "boolean" &&
    Array.isArray(value.entries) &&
    value.entries.every(isEntry) &&
    (value.decision === null || isObject(value.decision))
  );
}

function isEntry(value: unknown): value is HistoryEntry {
  return isObject(value) && (value.role === "employee" || value.role === "advisor") && typeof value.text === "string";
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
