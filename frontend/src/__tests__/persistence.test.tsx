// What survives a server restart: the employee's conversation (restored at once from its token, under
// its own session id) and the representative's queue (kept in the representative's browser).

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, { REPRESENTATIVE_HASH } from "../App";
import type { QueueQuestion } from "../api/types";
import { ACTIVE_KEY } from "../hooks/useAdvisor";
import { REPRESENTATIVE_POLL_MS } from "../hooks/useRepresentativeAnswers";
import { type Conversation, findConversation, saveConversation } from "../lib/history";
import { CACHE_DAYS, MAX_CACHED, QUEUE_KEY, loadQueue, mergeQueue } from "../lib/repQueue";
import { ATTO_3, SESSION, TOKEN, e001, mockServer, response } from "../test/server";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  window.location.hash = "";
});

const OPEN_Q = { question_id: "Q-1", topic: "alternative_participation" as const, status: "open" as const };

function saved(over: Partial<Conversation> = {}): Conversation {
  return {
    id: "conv-1", title: "אטו 3", createdAt: 1, updatedAt: 2, sessionId: SESSION, resumeToken: TOKEN, resumable: true,
    entries: [{ role: "employee", text: "תשאל את הנציג" }, { role: "advisor", text: "שלחתי לנציג התחבורה שאלה." }],
    decision: response({
      ui_state: "representative", comparison: e001(),
      representative: { questions: [OPEN_Q], latest: { question_id: "Q-1", topic: "alternative_participation", already_open: false } },
    }),
    ...over,
  };
}

function openSaved(conversation = saved()) {
  saveConversation(conversation, conversation.id);
  window.sessionStorage.setItem(ACTIVE_KEY, conversation.id);
}

describe("employee: the conversation survives a server restart", () => {
  it("is restored at once when the server lost it, under the same session id", async () => {
    openSaved();
    const server = mockServer();
    server.reply("GET /api/sessions/:id", 404, { error: { code: "session_not_found", message: "x" } });
    server.restore(SESSION, 4); // the token's own id was free

    render(<App />);

    await waitFor(() => expect(server.restores()).toHaveLength(1));
    expect(server.restores()[0]?.body).toEqual({ resume_token: TOKEN });
    expect(server.chats()).toEqual([]); // restored without the employee writing anything
    expect(screen.getByText("שלחתי לנציג התחבורה שאלה.")).toBeInTheDocument();
  });

  it("does nothing more when the server still has it", async () => {
    openSaved();
    const server = mockServer();
    server.reply("GET /api/sessions/:id", 200, { questions: [OPEN_Q], resume_token: TOKEN });

    render(<App />);

    await waitFor(() => expect(server.requests).toHaveLength(1));
    expect(server.restores()).toEqual([]);
  });

  it("keeps the fresh token that holds the representative's answer", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    openSaved();
    const server = mockServer();
    server.reply("GET /api/sessions/:id", 200, { questions: [OPEN_Q], resume_token: TOKEN }); // the check on open
    server.reply("GET /api/sessions/:id", 200, {
      questions: [{ ...OPEN_Q, status: "answered", answer: "450" }], resume_token: "token-with-the-answer",
    });
    render(<App />);
    await waitFor(() => expect(server.requests).toHaveLength(1));

    await act(() => vi.advanceTimersByTimeAsync(REPRESENTATIVE_POLL_MS));

    await waitFor(() => expect(findConversation("conv-1")?.resumeToken).toBe("token-with-the-answer"));
  });
});

// --- The representative's queue ----------------------------------------------------------------------

const QUESTION: QueueQuestion = {
  session_id: SESSION, question_id: "Q-1", topic: "alternative_participation", status: "open",
  question: "מה ההשתתפות ב-ATTO 3?", car: ATTO_3,
};

describe("the representative's queue in this browser", () => {
  it("marks questions the server no longer lists as offline, and keeps the rest live", () => {
    const now = 1_000_000_000_000;
    const other = { ...QUESTION, question_id: "Q-2", question: "שאלה כללית" };
    const shown = mergeQueue([QUESTION], [{ ...other, seenAt: now - 1000 }], now);
    expect(shown.map((q) => [q.question_id, q.offline])).toEqual([["Q-1", false], ["Q-2", true]]);
    expect(loadQueue().map((q) => q.question_id)).toEqual(["Q-1", "Q-2"]); // saved
  });

  it("forgets questions not seen for a week, and keeps at most MAX_CACHED", () => {
    const now = 1_000_000_000_000;
    const old = { ...QUESTION, question_id: "Q-9", seenAt: now - (CACHE_DAYS + 1) * 24 * 60 * 60 * 1000 };
    expect(mergeQueue([], [old], now)).toEqual([]);
    const many = Array.from({ length: MAX_CACHED + 5 }, (_, i) => ({ ...QUESTION, question_id: `Q-${i}` }));
    expect(mergeQueue(many, [], now)).toHaveLength(MAX_CACHED);
  });

  it("survives unreadable storage", () => {
    window.localStorage.setItem(QUEUE_KEY, "{not json");
    expect(loadQueue()).toEqual([]);
  });

  it("after a restart shows the cached question at once, offline and not answerable, until it is back", async () => {
    window.localStorage.setItem(QUEUE_KEY, JSON.stringify([{ ...QUESTION, seenAt: Date.now() }]));
    const server = mockServer();
    server.reply("GET /api/representative/questions", 200, { questions: [] }); // the server restarted
    server.reply("GET /api/representative/cars", 200, { cars: [ATTO_3] });
    window.location.hash = REPRESENTATIVE_HASH;
    const user = userEvent.setup();
    render(<App />);

    const card = await screen.findByRole("article", { name: QUESTION.question });
    await waitFor(() => expect(server.requests.some((r) => r.path === "/api/representative/questions")).toBe(true));
    expect(within(card).getByRole("note")).toHaveTextContent("השיחה הזו לא מחוברת כרגע לשרת");
    expect(within(card).queryByRole("button", { name: "שליחת תשובה" })).not.toBeInTheDocument();

    server.reply("GET /api/representative/questions", 200, { questions: [QUESTION] }); // the employee came back
    await user.click(screen.getByRole("button", { name: "רענון" }));

    await waitFor(() => expect(within(card).queryByRole("note")).not.toBeInTheDocument());
    expect(within(card).getByRole("button", { name: "שליחת תשובה" })).toBeInTheDocument();
  });
});
