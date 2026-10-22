import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, { REPRESENTATIVE_HASH } from "../App";
import type { QueueQuestion } from "../api/types";
import { CONTINUE_MESSAGE } from "../components/AnswerArrivedCard";
import { REPRESENTATIVE_POLL_MS } from "../hooks/useRepresentativeAnswers";
import { ATTO_3, COROLLA_CROSS, SESSION, e001, mockServer, response } from "../test/server";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  window.location.hash = "";
});

const OPEN = { question_id: "Q-1", topic: "alternative_participation" as const, status: "open" as const };
const ANSWERED = { ...OPEN, status: "answered" as const, answer: "אושר מול הספק: 450 ₪" };

function decision() {
  return screen.getByRole("region", { name: "ההחלטה שלך" });
}

async function employeeWithOpenQuestion() {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const server = mockServer();
  server.session();
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(<App />);
  const input = await screen.findByLabelText("הודעה ליועץ");
  await waitFor(() => expect(input).toBeEnabled());
  server.chat(response({
    ui_state: "representative", reply: "שלחתי לנציג התחבורה שאלה על ההשתתפות ברכב החלופי.", comparison: e001(),
    representative: { questions: [OPEN], latest: { question_id: "Q-1", topic: "alternative_participation", already_open: false } },
  }));
  await user.type(input, "תשאל את הנציג{Enter}");
  await within(decision()).findByRole("region", { name: "השאלה הועברה לנציג התחבורה" });
  return { server, user };
}

const polls = (server: ReturnType<typeof mockServer>) =>
  server.requests.filter((r) => r.method === "GET" && r.path === `/api/sessions/${SESSION}/representative`);

describe("employee: the representative's answer", () => {
  it("appears while the screen is open, and continuing sends it through the chat", async () => {
    const { server, user } = await employeeWithOpenQuestion();
    server.reply("GET /api/sessions/:id", 200, { questions: [OPEN] });
    server.reply("GET /api/sessions/:id", 200, { questions: [ANSWERED] });

    await act(() => vi.advanceTimersByTimeAsync(REPRESENTATIVE_POLL_MS));
    expect(screen.queryByText("התקבלה תשובה מנציג")).not.toBeInTheDocument(); // still open
    await act(() => vi.advanceTimersByTimeAsync(REPRESENTATIVE_POLL_MS));

    const card = await within(decision()).findByRole("region", { name: "התקבלה תשובה מנציג" });
    expect(card).toHaveTextContent("ההשתתפות ברכב החלופי אושרה");
    expect(card).toHaveTextContent("אושר מול הספק: 450 ₪");
    expect(polls(server)).toHaveLength(2);

    await act(() => vi.advanceTimersByTimeAsync(REPRESENTATIVE_POLL_MS * 3));
    expect(polls(server)).toHaveLength(2); // every watched question is answered: no more polling

    server.chat(response({
      ui_state: "confirmed", turn: 2, reply: "ההשוואה עודכנה.",
      comparison: e001({ status: "confirmed", estimated_fields: [] }),
      representative: { questions: [ANSWERED] },
    }));
    await user.click(within(card).getByRole("button", { name: "המשך לפי התשובה" }));

    await waitFor(() => expect(server.chats().at(-1)?.body).toEqual({ session_id: SESSION, message: CONTINUE_MESSAGE }));
    await waitFor(() => expect(screen.queryByText("התקבלה תשובה מנציג")).not.toBeInTheDocument());
  });

  it("when the employee writes on instead, the answer is no longer new and the question shows as answered", async () => {
    const { server, user } = await employeeWithOpenQuestion();
    server.reply("GET /api/sessions/:id", 200, { questions: [ANSWERED] });
    await act(() => vi.advanceTimersByTimeAsync(REPRESENTATIVE_POLL_MS));
    await within(decision()).findByRole("region", { name: "התקבלה תשובה מנציג" });

    server.chat(response({ ui_state: "answered", turn: 2, reply: "מה שכר הבסיס שלך?", representative: { questions: [ANSWERED] } }));
    await user.type(screen.getByLabelText("הודעה ליועץ"), "תודה, בוא נמשיך{Enter}");

    await waitFor(() => expect(screen.queryByText("התקבלה תשובה מנציג")).not.toBeInTheDocument());
    const card = within(decision()).getByRole("region", { name: "נציג התחבורה ענה" });
    expect(card).toHaveTextContent("אושר מול הספק: 450 ₪");
    expect(screen.queryByText("השאלה הועברה לנציג התחבורה")).not.toBeInTheDocument();
  });

  it("details given before any car show progress without claiming the eligibility step", async () => {
    const { server, user } = await employeeWithOpenQuestion();
    server.chat(response({
      ui_state: "collecting", turn: 2, reply: "כמה נקודות זיכוי?",
      details: { missing: ["occupation", "current", "credit_points"], provided: ["base_salary"] },
      representative: { questions: [ANSWERED] },
    }));
    await user.type(screen.getByLabelText("הודעה ליועץ"), "שכר בסיס 18000{Enter}");

    const panel = decision();
    await within(panel).findByText("כדי לחשב את השינוי בנטו, חסרים רק אלה");
    expect(within(panel).getByText("עוד לא נבחרה חלופה")).toBeInTheDocument();
    expect(within(panel).queryByText("עבר")).not.toBeInTheDocument();
    expect(panel).toHaveTextContent("עוד לא נבחר רכב חלופי");
    expect(panel).toHaveTextContent("שכר בסיס חודשי (ברוטו)נמסר");
  });

  it("does not poll without an open question", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const server = mockServer();
    server.session();
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<App />);
    const input = await screen.findByLabelText("הודעה ליועץ");
    await waitFor(() => expect(input).toBeEnabled());
    server.chat(response({ ui_state: "estimate", comparison: e001() }));
    await user.type(input, "כמה?{Enter}");
    await screen.findByText("הערכת שינוי בנטו");

    await act(() => vi.advanceTimersByTimeAsync(REPRESENTATIVE_POLL_MS * 3));
    expect(polls(server)).toEqual([]);
  });
});

// --- The representative's screen ---------------------------------------------------------------

const PARTICIPATION: QueueQuestion = {
  session_id: SESSION, question_id: "Q-1", topic: "alternative_participation", status: "open",
  question: "מה ההשתתפות ב-ATTO 3?", car: ATTO_3,
};
const CURRENT: QueueQuestion = {
  session_id: SESSION, question_id: "Q-2", topic: "current_car_not_in_catalogue", status: "open",
  question: "הרכב של העובדת: סקודה סופרב 2018",
};
const OTHER: QueueQuestion = {
  session_id: SESSION, question_id: "Q-3", topic: "other", status: "open", question: "האם אפשר להחליף רכב באמצע הליסינג?",
};

async function representativeScreen(questions: QueueQuestion[]) {
  const server = mockServer();
  server.reply("GET /api/representative/questions", 200, { questions });
  server.reply("GET /api/representative/cars", 200, { cars: [COROLLA_CROSS, ATTO_3] });
  window.location.hash = REPRESENTATIVE_HASH;
  const user = userEvent.setup();
  render(<App />);
  await screen.findByRole("heading", { name: `ממתינות לתשובה (${questions.filter((q) => q.status === "open").length})` });
  return { server, user };
}

function question(name: string) {
  return screen.getByRole("article", { name });
}

const answers = (server: ReturnType<typeof mockServer>) => server.requests.filter((r) => r.path === "/api/representative/answers");

describe("representative screen", () => {
  it("opens from the employee screen's header", async () => {
    const server = mockServer();
    server.session();
    render(<App />);
    expect(await screen.findByRole("link", { name: "למסך הנציג" })).toHaveAttribute("href", REPRESENTATIVE_HASH);
  });

  it("never starts an employee session", async () => {
    const { server } = await representativeScreen([]);
    expect(screen.getByText("אין שאלות פתוחות.")).toBeInTheDocument();
    expect(server.requests.some((r) => r.path === "/api/sessions")).toBe(false);
  });

  it("answers a participation question with the confirmed amount", async () => {
    const { server, user } = await representativeScreen([PARTICIPATION]);
    const card = question(PARTICIPATION.question);
    expect(card).toHaveTextContent("השתתפות ברכב חלופי");
    expect(card).toHaveTextContent("BYD ATTO 3 (2025)");
    const send = within(card).getByRole("button", { name: "שליחת תשובה" });
    expect(send).toBeDisabled(); // the amount is required

    await user.type(within(card).getByLabelText("השתתפות חודשית מאושרת (₪)"), "450");
    await user.type(within(card).getByLabelText("הערה לעובד/ת (לא חובה)"), "אושר מול הספק");
    server.reply("POST /api/representative/answers", 200, { ...PARTICIPATION, status: "answered", participation: 450, answer: "אושר מול הספק" });
    server.reply("GET /api/representative/questions", 200, {
      questions: [{ ...PARTICIPATION, status: "answered", participation: 450, answer: "אושר מול הספק" }],
    });
    await user.click(send);

    expect(answers(server)[0]?.body).toEqual({ session_id: SESSION, question_id: "Q-1", answer: "אושר מול הספק", participation: 450 });
    await screen.findByRole("heading", { name: "נענו (1)" });
    expect(question(PARTICIPATION.question)).toHaveTextContent("השתתפות: 450 ₪ בחודש");
  });

  it("an unknown current car takes a catalogue car with today's participation, or words only", async () => {
    const { server, user } = await representativeScreen([CURRENT]);
    const card = question(CURRENT.question);
    const send = within(card).getByRole("button", { name: "שליחת תשובה" });
    await user.type(within(card).getByLabelText("התשובה לעובד/ת"), "אין מקביל בקטלוג");
    expect(send).toBeEnabled();

    await user.selectOptions(within(card).getByLabelText("הרכב המקביל בקטלוג (אם יש)"), "CAR-09");
    expect(send).toBeDisabled(); // a car needs today's participation
    await user.type(within(card).getByLabelText("ההשתתפות של העובד/ת היום (₪ בחודש)"), "300");
    server.reply("POST /api/representative/answers", 200, { ...CURRENT, status: "answered" });
    server.reply("GET /api/representative/questions", 200, { questions: [] });
    await user.click(send);

    expect(answers(server)[0]?.body).toEqual({
      session_id: SESSION, question_id: "Q-2", answer: "אין מקביל בקטלוג", participation: 300, car_id: "CAR-09",
    });
  });

  it("shows the server's message when an answer is refused", async () => {
    const { server, user } = await representativeScreen([OTHER]);
    const card = question(OTHER.question);
    await user.type(within(card).getByLabelText("התשובה לעובד/ת"), "כן, בתיאום מראש");
    server.reply("POST /api/representative/answers", 409, { error: { code: "turn_in_progress", message: "ההודעה הקודמת עדיין בטיפול." } });
    await user.click(within(card).getByRole("button", { name: "שליחת תשובה" }));

    expect(await within(card).findByRole("alert")).toHaveTextContent("ההודעה הקודמת עדיין בטיפול.");
    expect(answers(server)[0]?.body).toEqual({ session_id: SESSION, question_id: "Q-3", answer: "כן, בתיאום מראש" });
  });
});
