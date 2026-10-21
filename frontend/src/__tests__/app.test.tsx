import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { ACTIVE_KEY, SESSION_KEY, TEXT } from "../hooks/useAdvisor";
import {
  ATTO_3,
  CARNIVAL,
  COROLLA,
  COROLLA_CROSS,
  E001_REPLY,
  SESSION,
  SESSION_2,
  e001,
  mockServer,
  response,
} from "../test/server";

afterEach(() => {
  vi.unstubAllGlobals();
});

async function start(server = mockServer()) {
  server.session();
  const user = userEvent.setup();
  render(<App />);
  const input = await screen.findByLabelText("הודעה ליועץ");
  await waitFor(() => expect(input).toBeEnabled());
  return { server, user, input };
}

async function say(user: ReturnType<typeof userEvent.setup>, text: string) {
  const input = screen.getByLabelText("הודעה ליועץ");
  await user.clear(input);
  await user.type(input, `${text}{Enter}`); // keyboard submit
}

function decision() {
  return screen.getByRole("region", { name: "ההחלטה שלך" });
}

function conversation() {
  return screen.getByRole("log", { name: "הודעות" });
}

describe("session", () => {
  it("creates a session and keeps only its ID in sessionStorage", async () => {
    const { server } = await start();

    expect(server.requests[0]).toMatchObject({ method: "POST", path: "/api/sessions" });
    expect(window.sessionStorage.getItem(SESSION_KEY)).toBe(SESSION);
    // Only the session ID and the per-tab active-conversation ID live in sessionStorage; no state.
    expect(new Set(Object.keys(window.sessionStorage))).toEqual(new Set([SESSION_KEY, ACTIVE_KEY]));
  });

  it("reuses a stored session without creating another", async () => {
    window.sessionStorage.setItem(SESSION_KEY, SESSION_2);
    const server = mockServer();
    render(<App />);

    await waitFor(() => expect(screen.getByLabelText("הודעה ליועץ")).toBeEnabled());
    expect(server.requests).toEqual([]);
  });

  it("starts a new conversation: deletes the old session, creates one, clears everything", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "estimate", reply: E001_REPLY, comparison: e001() }));
    await say(user, "כמה ישתנה הנטו?");
    await screen.findByText("הערכת שינוי בנטו");
    server.deleted();
    server.session(SESSION_2);

    await user.click(screen.getByRole("button", { name: "שיחה חדשה" }));

    await waitFor(() => expect(window.sessionStorage.getItem(SESSION_KEY)).toBe(SESSION_2));
    expect(server.requests.some((r) => r.method === "DELETE" && r.path === `/api/sessions/${SESSION}`)).toBe(true);
    expect(within(conversation()).queryByText("כמה ישתנה הנטו?")).not.toBeInTheDocument();
    expect(within(decision()).getByText("ההחלטה שלך תופיע כאן")).toBeInTheDocument();
  });

  it("recreates the session after 404 and says so", async () => {
    const { server, user } = await start();
    server.chatStatus(404, "session_not_found");
    server.session(SESSION_2);

    await say(user, "שלום");

    expect(await screen.findByText(TEXT.sessionLost)).toBeInTheDocument();
    await waitFor(() => expect(window.sessionStorage.getItem(SESSION_KEY)).toBe(SESSION_2));
    expect(within(conversation()).queryByText("שלום", { selector: "p" })).not.toBeInTheDocument();
    expect(screen.getByLabelText("הודעה ליועץ")).toHaveValue("שלום"); // ready to send again
  });
});

describe("chat", () => {
  it("sends only the session ID and the message", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "answered", reply: "טוסון הוא רכב בקטלוג." }));

    await say(user, "מה זה טוסון?");

    expect(await screen.findByText("טוסון הוא רכב בקטלוג.")).toBeInTheDocument();
    expect(server.chats()[0]).toEqual({ method: "POST", path: "/api/chat", body: { session_id: SESSION, message: "מה זה טוסון?" } });
    expect(within(conversation()).getByText("מה זה טוסון?")).toBeInTheDocument();
  });

  it("shows a neutral pending state while the request is in flight", async () => {
    const { server, user } = await start();
    const resolve = server.chatDeferred();

    await say(user, "אפשר לקבל קרניבל?");

    expect(within(conversation()).getByRole("status")).toHaveTextContent("בודקים…");
    expect(screen.getByLabelText("הודעה ליועץ")).toBeDisabled();
    expect(screen.queryByText(/מחשבים/)).not.toBeInTheDocument();
    resolve(response({ ui_state: "answered", reply: "בסדר" }));
    await screen.findByText("בסדר");
    expect(screen.queryByText("בודקים…")).not.toBeInTheDocument();
  });

  it("answered keeps the last decision on screen", async () => {
    const { server, user } = await start();
    server.chat(
      response({ ui_state: "estimate", reply: E001_REPLY, comparison: e001() }),
      response({ ui_state: "answered", turn: 2, reply: "שווי שימוש הוא הסכום שנזקף לשכר." }),
    );
    await say(user, "כמה ישתנה?");
    await screen.findByText("הערכת שינוי בנטו");

    await say(user, "מה זה שווי שימוש?");

    expect(await screen.findByText("שווי שימוש הוא הסכום שנזקף לשכר.")).toBeInTheDocument();
    expect(within(decision()).getByText("הערכת שינוי בנטו")).toBeInTheDocument();
  });

  it("incomplete shows the backend reply and leaves the panel as it was", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "incomplete", reply: "מצטער/ת, לא הצלחתי להשלים את הבדיקה." }));

    await say(user, "משהו");

    expect(await screen.findByText("מצטער/ת, לא הצלחתי להשלים את הבדיקה.")).toBeInTheDocument();
    expect(within(decision()).getByText("ההחלטה שלך תופיע כאן")).toBeInTheDocument();
  });

  it("shows before → after runs left to right without changing the text", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "estimate", reply: E001_REPLY, comparison: e001() }));

    await say(user, "כמה?");

    const run = await screen.findByText("4,070 → 3,130");
    expect(run.tagName).toBe("BDI");
    expect(run).toHaveAttribute("dir", "ltr");
    expect(run.closest("p")).toHaveTextContent("מה משתנה: שווי שימוש 4,070 → 3,130 ₪");
  });
});

describe("decision states", () => {
  it("eligibility_check asks only for the decisive field, generically", async () => {
    const { server, user } = await start();
    server.chat(
      response({
        ui_state: "eligibility_check",
        reply: "כדי לבדוק אם החלופה הזו אפשרית עבורך, צריך קודם לדעת:\n• מספר הילדים שלך עד גיל 23 (נדרש לרכב מהמסלול המשפחתי)",
        alternative: { kind: "company_car", car: CARNIVAL },
        eligibility: { missing: ["eligible_children"] },
      }),
      response({ ui_state: "answered", turn: 2, reply: "ok" }),
    );

    await say(user, "אפשר לקבל קרניבל?");

    const panel = decision();
    expect(await within(panel).findByText("בודקים קודם התאמה למסלול")).toBeInTheDocument();
    expect(within(panel).getByText("מספר הילדים הזכאים")).toBeInTheDocument();
    expect(within(panel).getByText("מסלול משפחתי")).toBeInTheDocument();
    expect(within(panel).queryByText(/נטו משוער/)).not.toBeInTheDocument();

    await user.click(within(panel).getByRole("button", { name: "יותר" }));
    await user.click(within(panel).getByRole("button", { name: "יותר" }));
    await user.click(within(panel).getByRole("button", { name: "שליחה" }));
    await waitFor(() => expect(server.chats()[1]?.body).toEqual({ session_id: SESSION, message: "יש לי 2 ילדים" }));
  });

  it("eligibility_check can ask for the occupation instead", async () => {
    const { server, user } = await start();
    server.chat(
      response({ ui_state: "eligibility_check", reply: "…", alternative: { kind: "company_car", car: ATTO_3 }, eligibility: { missing: ["occupation"] } }),
    );

    await say(user, "אני רוצה אטו 3");

    const panel = decision();
    expect(await within(panel).findByText("התפקיד")).toBeInTheDocument();
    expect(within(panel).getByLabelText("התפקיד שלך")).toBeInTheDocument();
    expect(within(panel).queryByText("מספר הילדים הזכאים")).not.toBeInTheDocument();
  });

  it("not_eligible shows the backend's facts and no comparison", async () => {
    const { server, user } = await start();
    server.chat(
      response({
        ui_state: "not_eligible",
        reply: "אי אפשר לבחור ב-CARNIVAL כבחירה חדשה במסלול המשפחתי…",
        alternative: { kind: "company_car", car: CARNIVAL },
        eligibility: { missing: [], reason: "not_enough_children", facts: { min_children: 4, child_max_age: 23, eligible_children: 3 } },
      }),
    );

    await say(user, "יש לי 3 ילדים");

    const panel = decision();
    expect(await within(panel).findByText("אי אפשר לבחור כעת ב-CARNIVAL")).toBeInTheDocument();
    expect(within(panel).getByText("לפחות 4 ילדים עד גיל 23")).toBeInTheDocument();
    expect(within(panel).getByText("לפי מה שמסרת").nextElementSibling).toHaveTextContent("3");
    expect(within(panel).queryByText(/₪/)).not.toBeInTheDocument();
  });

  it("collecting lists what is given and what is missing", async () => {
    const { server, user } = await start();
    server.chat(
      response({
        ui_state: "collecting",
        reply: "כדי לחשב את השינוי בנטו חסרים לי עוד כמה פרטים:",
        alternative: { kind: "company_car", car: ATTO_3 },
        current: COROLLA_CROSS,
        details: { missing: ["base_salary", "credit_points"], provided: ["occupation", "current"] },
      }),
    );

    await say(user, "כמה ישתנה?");

    const panel = decision();
    expect(await within(panel).findByText("שכר בסיס חודשי (ברוטו)")).toBeInTheDocument();
    expect(within(panel).getByText("נקודות זיכוי")).toBeInTheDocument();
    expect(within(panel).getAllByText("חסר")).toHaveLength(2);
    expect(within(panel).getAllByText("נמסר")).toHaveLength(2);
    expect(within(panel).getByText("COROLLA CROSS")).toBeInTheDocument();
  });

  it("estimate renders the E-001 comparison from the API data", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "estimate", reply: E001_REPLY, current: COROLLA_CROSS, comparison: e001() }));

    await say(user, "כמה ישתנה?");

    const panel = decision();
    expect(await within(panel).findByText("הערכת שינוי בנטו")).toBeInTheDocument();
    expect(within(panel).getByText("+₪661")).toBeInTheDocument();
    expect(within(panel).getByText("הערכה · לא תוצאת שכר רשמית")).toBeInTheDocument();
    expect(within(panel).getByText("₪11,032")).toBeInTheDocument();
    expect(within(panel).getByText("₪11,693")).toBeInTheDocument();
    expect(within(panel).getByText("₪4,070")).toBeInTheDocument();
    expect(within(panel).getByText("₪3,130")).toBeInTheDocument();
    expect(within(panel).getByText("+₪250")).toBeInTheDocument();
    expect(within(panel).getByText("מגולם: המעסיק משלם את המס")).toBeInTheDocument();
    const table = within(panel).getByRole("table");
    expect(within(table).getByRole("columnheader", { name: "היום" })).toBeInTheDocument();
    expect(within(table).getByRole("columnheader", { name: "בחלופה" })).toBeInTheDocument();
  });

  it("discloses the catalogue-default participation of an estimate", async () => {
    const { server, user } = await start();
    server.chat(response({ ui_state: "estimate", reply: E001_REPLY, comparison: e001() }));

    await say(user, "כמה?");

    const note = await within(decision()).findByRole("note");
    expect(note).toHaveTextContent("ההשתתפות ברכב החלופי (₪300) היא ברירת מחדל מהקטלוג, כלומר הערכה ולא נתון מאושר.");
    expect(within(note).getByRole("button", { name: "לבקש מנציג התחבורה את הסכום המדויק" })).toBeInTheDocument();
    expect(within(decision()).getByText("ההשתתפות ברכב החלופי בחישוב הזה מבוססת על ברירת המחדל מהקטלוג.")).toBeInTheDocument();
  });

  it("takes every figure from the API, not from constants", async () => {
    const { server, user } = await start();
    const other = e001({
      net_difference: -86,
      current: { ...e001().current, net: 12345, use_value: 5000 },
      alternative: { ...e001().alternative, net: 12259, use_value: 5100, participation: 410 },
      changes: [{ item: "use_value", current: 5000, alternative: 5100 }],
      electric_benefit: undefined,
    });
    server.chat(response({ ui_state: "estimate", reply: "…", comparison: other }));

    await say(user, "כמה?");

    const panel = decision();
    expect(await within(panel).findByText("−₪86")).toBeInTheDocument();
    for (const figure of ["₪12,345", "₪12,259", "₪5,000", "₪5,100"]) {
      expect(within(panel).getByText(figure)).toBeInTheDocument();
    }
    expect(within(panel).getByRole("note")).toHaveTextContent("(₪410)");
    expect(panel.textContent).not.toMatch(/661|11,032|11,693|4,070|3,130|250/);
  });

  it("confirmed reuses the comparison without the estimate disclosure", async () => {
    const { server, user } = await start();
    const confirmed = e001({
      status: "confirmed",
      estimated_fields: [],
      alternative: { ...e001().alternative, participation_source: "stated" },
    });
    server.chat(response({ ui_state: "confirmed", reply: "…", comparison: confirmed }));

    await say(user, "ואם ההשתתפות 300?");

    const panel = decision();
    expect(await within(panel).findByText("השינוי בנטו")).toBeInTheDocument();
    expect(within(panel).getByText("+₪661")).toBeInTheDocument();
    expect(within(panel).queryByText("הערכה · לא תוצאת שכר רשמית")).not.toBeInTheDocument();
    expect(within(panel).queryByRole("note")).not.toBeInTheDocument();
    expect(within(panel).queryByText("נטו משוער")).not.toBeInTheDocument();
  });

  it("a confirmed status with estimated fields still discloses them", async () => {
    const { server, user } = await start();
    const mixed = e001({ status: "confirmed", alternative: { ...e001().alternative, participation_source: "stated" } });
    server.chat(response({ ui_state: "confirmed", reply: "…", comparison: mixed }));

    await say(user, "כמה?");

    expect(await within(decision()).findByRole("note")).toHaveTextContent("חלק מהנתונים בחישוב הזה הם הערכה");
  });

  it("representative shows the open question and where the answer will appear", async () => {
    const { server, user } = await start();
    server.chat(
      response({
        ui_state: "representative",
        reply: "שלחתי לנציג התחבורה שאלה על ההשתתפות ברכב החלופי.",
        comparison: e001(),
        representative: {
          questions: [{ question_id: "Q-1", topic: "alternative_participation", status: "open" }],
          latest: { question_id: "Q-1", topic: "alternative_participation", already_open: false },
        },
      }),
    );

    await say(user, "תשאל את הנציג");

    const card = await within(decision()).findByRole("region", { name: "השאלה הועברה לנציג התחבורה" });
    expect(card).toHaveTextContent("פתוחה אצל נציג התחבורה");
    expect(card).toHaveTextContent("כשהנציג יענה, התשובה תופיע כאן.");
    expect(card.textContent).not.toMatch(/אוטומטית|יעודכן|נעדכן/); // nothing is recalculated by itself
    expect(within(decision()).getByText("+₪661")).toBeInTheDocument(); // the comparison stays
    expect(within(decision()).queryByRole("button", { name: "לבקש מנציג התחבורה את הסכום המדויק" })).not.toBeInTheDocument();
  });

  it("car_choice sends the chosen car as the employee's words", async () => {
    const { server, user } = await start();
    server.chat(
      response({ ui_state: "car_choice", reply: "לאיזה מהם התכוונת?", car_choice: { target: "current", options: [COROLLA, COROLLA_CROSS] } }),
      response({ ui_state: "answered", turn: 2, reply: "נשמר" }),
    );
    await say(user, "יש לי קורולה, השתתפות 250");

    await user.click(await within(decision()).findByRole("button", { name: /COROLLA CROSS/ }));

    await waitFor(() =>
      expect(server.chats()[1]?.body).toEqual({ session_id: SESSION, message: "התכוונתי ל-טויוטה יפן COROLLA CROSS (2025)" }),
    );
    expect(await within(conversation()).findByText("התכוונתי ל-טויוטה יפן COROLLA CROSS (2025)")).toBeInTheDocument();
  });
});

describe("failures", () => {
  it("temporary_error offers Retry and resends the same message once", async () => {
    const { server, user } = await start();
    server.chat(
      response({ ui_state: "temporary_error", turn: 0, reply: "מצטער/ת, יש תקלה זמנית בחיבור ליועץ. אפשר לנסות שוב בעוד רגע.", retryable: true }),
      response({ ui_state: "answered", reply: "עכשיו זה עבד" }),
    );

    await say(user, "אפשר לקבל קרניבל?");
    const error = await screen.findByRole("alert");
    expect(error).toHaveTextContent("יש תקלה זמנית");

    await user.click(within(error).getByRole("button", { name: "לנסות שוב" }));

    expect(await screen.findByText("עכשיו זה עבד")).toBeInTheDocument();
    expect(server.chats().map((r) => r.body)).toEqual([
      { session_id: SESSION, message: "אפשר לקבל קרניבל?" },
      { session_id: SESSION, message: "אפשר לקבל קרניבל?" },
    ]);
    expect(within(conversation()).getAllByText("אפשר לקבל קרניבל?")).toHaveLength(1);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("409 keeps the session and keeps sending off until a retry succeeds", async () => {
    const { server, user } = await start();
    server.chatStatus(409, "turn_in_progress", "ההודעה הקודמת עדיין בטיפול. אפשר לשלוח שוב בעוד רגע.");
    server.chat(response({ ui_state: "answered", reply: "בסדר" }));

    await say(user, "שלום");

    const error = await screen.findByRole("alert");
    expect(screen.getByLabelText("הודעה ליועץ")).toBeDisabled();
    expect(window.sessionStorage.getItem(SESSION_KEY)).toBe(SESSION);
    await user.click(within(error).getByRole("button", { name: "לנסות שוב" }));
    await screen.findByText("בסדר");
    expect(screen.getByLabelText("הודעה ליועץ")).toBeEnabled();
  });

  it("422 shows a safe message and gives the text back", async () => {
    const { server, user } = await start();
    server.chatStatus(422, "invalid_request");

    await say(user, "הודעה");

    expect(await screen.findByText(TEXT.invalid)).toBeInTheDocument();
    expect(within(conversation()).queryByText("הודעה")).not.toBeInTheDocument();
    expect(screen.getByLabelText("הודעה ליועץ")).toHaveValue("הודעה");
  });

  it("500 and 503 show the server's safe message with Retry", async () => {
    const { server, user } = await start();
    server.chatStatus(500, "internal_error", "מצטער/ת, הייתה תקלה פנימית. אפשר לנסות שוב.");

    await say(user, "שלום");

    const error = await screen.findByRole("alert");
    expect(error).toHaveTextContent("הייתה תקלה פנימית");
    expect(within(error).getByRole("button", { name: "לנסות שוב" })).toBeInTheDocument();
  });

  it("a network failure says the server is unreachable", async () => {
    const { server, user } = await start();
    void server;
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError("Failed to fetch"));

    await say(user, "שלום");

    expect(await screen.findByRole("alert")).toHaveTextContent(TEXT.unreachable);
  });
});

describe("layout", () => {
  it("renders right to left in Hebrew", async () => {
    await start();
    const root = document.querySelector(".app");
    expect(root).toHaveAttribute("dir", "rtl");
    expect(root).toHaveAttribute("lang", "he");
  });
});
