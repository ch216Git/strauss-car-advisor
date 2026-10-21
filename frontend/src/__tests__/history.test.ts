import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type Conversation,
  type HistoryEntry,
  ENABLED_KEY,
  HISTORY_KEY,
  MAX_CONVERSATIONS,
  clearHistory,
  deleteConversation,
  historyEnabled,
  loadConversations,
  saveConversation,
  setHistoryEnabled,
  titleFrom,
} from "../lib/history";

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function make(id: string, over: Partial<Conversation> = {}): Conversation {
  const entries: HistoryEntry[] = over.entries ?? [{ role: "employee", text: `שאלה ${id}` }, { role: "advisor", text: "תשובה" }];
  return {
    id, title: titleFrom(entries), createdAt: 1000, updatedAt: 1000, sessionId: `s-${id}`,
    resumeToken: "tok", resumable: true, entries, decision: null, ...over,
  };
}

describe("history store", () => {
  it("saves and reads back a conversation", () => {
    expect(saveConversation(make("a"), "a")).toBe("saved");
    const list = loadConversations();
    expect(list).toHaveLength(1);
    expect(list[0]).toMatchObject({ id: "a", title: "שאלה a", resumable: true });
  });

  it("orders by updatedAt, newest first", () => {
    saveConversation(make("a", { updatedAt: 100 }), null);
    saveConversation(make("b", { updatedAt: 300 }), null);
    saveConversation(make("c", { updatedAt: 200 }), null);
    expect(loadConversations().map((c) => c.id)).toEqual(["b", "c", "a"]);
  });

  it("derives the title from the first employee message, trimmed", () => {
    const long = "א".repeat(60);
    expect(titleFrom([{ role: "advisor", text: "קודם" }, { role: "employee", text: long }]))
      .toBe(`${"א".repeat(40)}…`);
  });

  it("never saves an empty conversation", () => {
    expect(saveConversation(make("a", { entries: [] }), "a")).toBe("skipped");
    expect(loadConversations()).toEqual([]);
  });

  it("replaces an existing conversation by id, not duplicates it", () => {
    saveConversation(make("a", { updatedAt: 100 }), "a");
    saveConversation(make("a", { updatedAt: 200, decision: null }), "a");
    expect(loadConversations()).toHaveLength(1);
    expect(loadConversations()[0]?.updatedAt).toBe(200);
  });

  it("caps the number of conversations, keeping the newest", () => {
    for (let i = 0; i < MAX_CONVERSATIONS + 5; i++) saveConversation(make(`c${i}`, { updatedAt: i }), null);
    const list = loadConversations();
    expect(list).toHaveLength(MAX_CONVERSATIONS);
    expect(list[0]?.id).toBe(`c${MAX_CONVERSATIONS + 4}`); // newest kept
    expect(list.some((c) => c.id === "c0")).toBe(false); // oldest evicted
  });

  it("deletes one conversation", () => {
    saveConversation(make("a"), null);
    saveConversation(make("b"), null);
    deleteConversation("a");
    expect(loadConversations().map((c) => c.id)).toEqual(["b"]);
  });

  it("clears all history", () => {
    saveConversation(make("a"), null);
    clearHistory();
    expect(loadConversations()).toEqual([]);
  });

  it("ignores corrupt data without crashing", () => {
    window.localStorage.setItem(HISTORY_KEY, "{ not json");
    expect(loadConversations()).toEqual([]);
  });

  it("ignores data from an unknown version", () => {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify({ version: 999, conversations: [make("a")] }));
    expect(loadConversations()).toEqual([]);
  });

  it("ignores a conversation whose shape is wrong", () => {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify({ version: 1, conversations: [{ id: 1 }] }));
    expect(loadConversations()).toEqual([]);
  });

  it("returns nothing and does not throw when storage reads throw", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
    expect(loadConversations()).toEqual([]);
  });

  it("reports unavailable when storage writes throw (not quota)", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("blocked"); });
    expect(saveConversation(make("a"), "a")).toBe("unavailable");
  });

  it("evicts the oldest non-active conversation on quota, keeping the active one", () => {
    // Seed two; then a quota-throwing write that succeeds only once something is dropped.
    const stored = { version: 1, conversations: [make("old", { updatedAt: 1 }), make("active", { updatedAt: 2 })] };
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(stored));
    let calls = 0;
    const real = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (this: Storage, key: string, value: string) {
      calls += 1;
      const parsed = JSON.parse(value) as { conversations: Conversation[] };
      // Fail while "old" is still present; succeed once it has been evicted.
      if (key === HISTORY_KEY && parsed.conversations.some((c) => c.id === "old")) {
        const err = new DOMException("full", "QuotaExceededError");
        throw err;
      }
      real.call(this, key, value);
    });

    const result = saveConversation(make("active", { updatedAt: 3 }), "active");

    expect(result).toBe("saved");
    expect(calls).toBeGreaterThan(1); // retried after eviction
    expect(loadConversations().map((c) => c.id)).toEqual(["active"]);
  });

  it("toggle persists and gates reads", () => {
    expect(historyEnabled()).toBe(true); // default on
    setHistoryEnabled(false);
    expect(window.localStorage.getItem(ENABLED_KEY)).toBe("0");
    expect(historyEnabled()).toBe(false);
  });

  it("does not write when history is disabled", () => {
    setHistoryEnabled(false);
    expect(saveConversation(make("a"), "a")).toBe("skipped");
    expect(window.localStorage.getItem(HISTORY_KEY)).toBeNull();
  });
});
