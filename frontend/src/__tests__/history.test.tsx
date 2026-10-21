import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { HISTORY_KEY } from "../lib/history";
import { E001_REPLY, SESSION_2, TOKEN, e001, mockServer, response } from "../test/server";

afterEach(() => {
  vi.unstubAllGlobals();
});

async function start(server = mockServer()) {
  server.session();
  const user = userEvent.setup();
  const view = render(<App />);
  const input = await screen.findByLabelText("הודעה ליועץ");
  await waitFor(() => expect(input).toBeEnabled());
  return { server, user, view };
}

async function say(user: ReturnType<typeof userEvent.setup>, text: string) {
  const input = screen.getByLabelText("הודעה ליועץ");
  await user.clear(input);
  await user.type(input, `${text}{Enter}`);
}

async function openHistory(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "היסטוריה" }));
  return screen.getByRole("dialog", { name: "שיחות קודמות" });
}

/** A committed, resumable estimate turn. */
function committed(reply = E001_REPLY) {
  return response({ ui_state: "estimate", reply, comparison: e001(), resume_token: TOKEN });
}

describe("history persistence", () => {
  it("persists a committed turn and restores it across a remount", async () => {
    const { server, user, view } = await start();
    server.chat(committed());
    await say(user, "כמה ישתנה הנטו?");
    await screen.findByText("הערכת שינוי בנטו");

    view.unmount();
    const server2 = mockServer(); // a fresh "page load": the stored session id is reused, no new session
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("הודעה ליועץ")).toBeEnabled());
    expect(within(screen.getByRole("log", { name: "הודעות" })).getByText("כמה ישתנה הנטו?")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "ההחלטה שלך" })).getByText("הערכת שינוי בנטו")).toBeInTheDocument();
    // Shown without waiting for the network; only a background check that the server still has it.
    expect(server2.requests.every((r) => r.method === "GET" && r.path.endsWith("/representative"))).toBe(true);
  });

  it("keeps the previous decision after a non-decisive turn, across a remount", async () => {
    const { server, user, view } = await start();
    // A decisive turn (estimate) sets the sticky decision...
    server.chat(
      committed(),
      // ...then a non-decisive turn (answered) must NOT clear it, and must be what history stores.
      response({ ui_state: "answered", turn: 2, reply: "שווי שימוש הוא הסכום שנזקף לשכר.", resume_token: TOKEN }),
    );
    await say(user, "כמה ישתנה הנטו?");
    await screen.findByText("הערכת שינוי בנטו");
    await say(user, "מה זה שווי שימוש?");
    await screen.findByText("שווי שימוש הוא הסכום שנזקף לשכר.");
    // Still shown live after the answered turn.
    expect(within(screen.getByRole("region", { name: "ההחלטה שלך" })).getByText("הערכת שינוי בנטו")).toBeInTheDocument();

    view.unmount();
    mockServer(); // fresh page load; the stored conversation is reopened from localStorage
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("הודעה ליועץ")).toBeEnabled());
    // The previous decision is restored from history, not blanked.
    expect(within(screen.getByRole("region", { name: "ההחלטה שלך" })).getByText("הערכת שינוי בנטו")).toBeInTheDocument();
    expect(screen.queryByText("ההחלטה שלך תופיע כאן")).not.toBeInTheDocument();
  });

  it("lists past conversations by title, newest first", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "answered", reply: "א", resume_token: TOKEN }));
    await say(user, "שאלה ראשונה");
    await screen.findByText("א");
    server.deleted();
    server.session(SESSION_2);
    await user.click(screen.getByRole("button", { name: "שיחה חדשה" }));
    server.chat(response({ ui_state: "answered", reply: "ב", resume_token: TOKEN }));
    await say(user, "שאלה שנייה");
    await screen.findByText("ב");

    const dialog = await openHistory(user);

    const titles = within(dialog).getAllByText(/שאלה/).map((n) => n.textContent);
    expect(titles).toEqual(["שאלה שנייה", "שאלה ראשונה"]);
  });

  it("opening a past conversation needs no network request to show it", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "answered", reply: "א", resume_token: TOKEN }));
    await say(user, "שאלה ראשונה");
    await screen.findByText("א");
    server.deleted();
    server.session(SESSION_2);
    await user.click(screen.getByRole("button", { name: "שיחה חדשה" }));
    server.chat(response({ ui_state: "answered", reply: "ב", resume_token: TOKEN }));
    await say(user, "שאלה שנייה");
    await screen.findByText("ב");

    const dialog = await openHistory(user);
    const before = server.requests.length;
    await user.click(within(dialog).getByText("שאלה ראשונה")); // the row's open button (title text)

    await screen.findByText("א"); // the first conversation's reply is back on screen
    // No session is created or restored and nothing is sent: only the background check.
    expect(server.requests.slice(before).every((r) => r.method === "GET" && r.path.endsWith("/representative"))).toBe(true);
  });

  it("never saves an empty conversation", async () => {
    const { user } = await start();

    await openHistory(user);

    expect(screen.getByText("אין עדיין שיחות שמורות.")).toBeInTheDocument();
    expect(window.localStorage.getItem(HISTORY_KEY)).toBeNull();
  });
});

describe("resume", () => {
  it("on 404 restores from the token and retries the send", async () => {
    const { server, user } = await start();
    server.chat(committed());
    await say(user, "כמה ישתנה הנטו?");
    await screen.findByText("הערכת שינוי בנטו");

    server.chatStatus(404, "session_not_found"); // the server forgot the session
    server.restore(SESSION_2, 1); // restore gives a new session id
    server.chat(response({ ui_state: "answered", turn: 2, reply: "אחרי שחזור", resume_token: TOKEN }));

    await say(user, "ועכשיו?");

    expect(await screen.findByText("אחרי שחזור")).toBeInTheDocument();
    expect(server.restores()).toHaveLength(1);
    const chats = server.chats();
    expect(chats[chats.length - 1]?.body).toEqual({ session_id: SESSION_2, message: "ועכשיו?" });
  });

  it("when restore is refused (410), the conversation becomes read-only", async () => {
    const { server, user } = await start();
    server.chat(committed());
    await say(user, "כמה ישתנה הנטו?");
    await screen.findByText("הערכת שינוי בנטו");

    server.chatStatus(404, "session_not_found");
    server.restoreGone();

    await say(user, "ועכשיו?");

    expect(await screen.findByText(/שמורה לצפייה בלבד/)).toBeInTheDocument();
    expect(screen.getByLabelText("הודעה ליועץ")).toBeDisabled();
    expect(server.restores()).toHaveLength(1);
  });
});

describe("delete and toggle", () => {
  it("deletes one conversation from the drawer", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "answered", reply: "א", resume_token: TOKEN }));
    await say(user, "שאלה אחת");
    await screen.findByText("א");
    server.deleted();
    server.session(SESSION_2);
    await user.click(screen.getByRole("button", { name: "שיחה חדשה" }));
    server.chat(response({ ui_state: "answered", reply: "ב", resume_token: TOKEN }));
    await say(user, "שאלה שתיים");
    await screen.findByText("ב");

    const dialog = await openHistory(user);
    server.deleted();
    await user.click(within(dialog).getByRole("button", { name: /מחיקת השיחה: שאלה אחת/ }));
    await user.click(within(dialog).getByRole("button", { name: "למחוק" }));

    await waitFor(() => expect(within(dialog).queryByText("שאלה אחת")).not.toBeInTheDocument());
    expect(within(dialog).getByText("שאלה שתיים")).toBeInTheDocument();
  });

  it("deletes all history", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "answered", reply: "א", resume_token: TOKEN }));
    await say(user, "שאלה אחת");
    await screen.findByText("א");

    const dialog = await openHistory(user);
    server.deleted();
    server.session(SESSION_2);
    await user.click(within(dialog).getByRole("button", { name: "מחיקת כל ההיסטוריה" }));
    await user.click(within(dialog).getByRole("button", { name: "למחוק הכול" }));

    await waitFor(() => expect(screen.getByText("אין עדיין שיחות שמורות.")).toBeInTheDocument());
    expect(window.localStorage.getItem(HISTORY_KEY)).toBeFalsy();
  });

  it("turning the toggle off writes nothing", async () => {
    const { server, user } = await start();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);

    let dialog = await openHistory(user);
    await user.click(within(dialog).getByRole("checkbox", { name: "שמור שיחות במכשיר הזה" }));
    await user.keyboard("{Escape}");

    server.chat(response({ ui_state: "answered", reply: "א", resume_token: TOKEN }));
    await say(user, "שאלה בזמן שהשמירה כבויה");
    await screen.findByText("א");

    expect(window.localStorage.getItem(HISTORY_KEY)).toBeFalsy(); // disabled: nothing persisted
    dialog = await openHistory(user);
    expect(within(dialog).getByText("אין עדיין שיחות שמורות.")).toBeInTheDocument();
    confirm.mockRestore();
  });

  it("history and new-conversation controls are disabled while a request is pending", async () => {
    const { server, user } = await start();
    const resolve = server.chatDeferred();

    await say(user, "רגע אחד");

    expect(screen.getByRole("button", { name: "היסטוריה" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "שיחה חדשה" })).toBeDisabled();
    resolve(response({ ui_state: "answered", reply: "סיום", resume_token: TOKEN }));
    await screen.findByText("סיום");
    expect(screen.getByRole("button", { name: "היסטוריה" })).toBeEnabled();
  });
});
