import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type { DocumentDetails } from "../api/types";
import { MAX_FILE_BYTES } from "../hooks/useDocumentUpload";
import { SESSION, mockServer, response } from "../test/server";

afterEach(() => {
  vi.unstubAllGlobals();
});

const DETAILS: DocumentDetails = {
  found: [
    { field: "occupation", label: "תפקיד", value: "מנהל/ת מוצר" },
    { field: "base_salary", label: "שכר בסיס חודשי (ברוטו)", value: "18,500 ₪" },
    { field: "credit_points", label: "נקודות זיכוי", value: "2.75" },
  ],
  unclear: [{ field: "bonus", label: "בונוס חודשי" }],
  from_image: false,
};

async function start() {
  const server = mockServer();
  server.session();
  const user = userEvent.setup();
  render(<App />);
  await waitFor(() => expect(screen.getByLabelText("הודעה ליועץ")).toBeEnabled());
  return { server, user };
}

function decision() {
  return screen.getByRole("region", { name: "ההחלטה שלך" });
}

const uploads = (server: ReturnType<typeof mockServer>) =>
  server.requests.filter((r) => r.method === "POST" && r.path === `/api/sessions/${SESSION}/document`);

async function attach(user: ReturnType<typeof userEvent.setup>, file: File) {
  await user.upload(screen.getByLabelText("העלאת קובץ עם פרטים"), file);
}

describe("details from a file", () => {
  it("shows only what was certain, and sends the checked details as the employee's words", async () => {
    const { server, user } = await start();
    server.reply("POST /api/sessions/:id", 200, DETAILS);

    await attach(user, new File(["תלוש"], "payslip.txt", { type: "text/plain" }));

    const card = await within(decision()).findByRole("region", { name: "פרטים מהקובץ" });
    expect(uploads(server)).toHaveLength(1);
    expect(card).toHaveTextContent("payslip.txt");
    expect(card).toHaveTextContent("לא היו ברורים בקובץ, והיועץ ישאל עליהם: בונוס חודשי");
    const boxes = within(card).getAllByRole("checkbox");
    expect(boxes).toHaveLength(3);
    boxes.forEach((box) => expect(box).toBeChecked());
    expect(server.chats()).toEqual([]); // nothing is sent or saved before confirmation

    await user.click(within(card).getByRole("checkbox", { name: /נקודות זיכוי/ })); // the employee drops one
    server.chat(response({ ui_state: "collecting", reply: "מה הבונוס החודשי שלך?" }));
    await user.click(within(card).getByRole("button", { name: "אישור ושליחה ליועץ" }));

    await waitFor(() => expect(server.chats()).toHaveLength(1));
    expect(server.chats()[0]?.body).toEqual({
      session_id: SESSION,
      message: "אלה הפרטים מהקובץ שהעליתי, ובדקתי שהם נכונים:\n• תפקיד: מנהל/ת מוצר\n• שכר בסיס חודשי (ברוטו): 18,500 ₪\nלא היו ברורים בקובץ: בונוס חודשי.",
    });
    await waitFor(() => expect(within(decision()).queryByRole("region", { name: "פרטים מהקובץ" })).not.toBeInTheDocument());
  });

  it("warns to check closely when the file was a photo or a scan", async () => {
    const { server, user } = await start();
    server.reply("POST /api/sessions/:id", 200, { ...DETAILS, from_image: true });
    await attach(user, new File(["png"], "photo.png", { type: "image/png" }));
    expect(await within(decision()).findByText("הקובץ נקרא מתמונה או מסריקה, ולכן חשוב לבדוק כל ערך היטב.")).toBeInTheDocument();
  });

  it("with nothing certain, offers nothing to confirm", async () => {
    const { server, user } = await start();
    server.reply("POST /api/sessions/:id", 200, { found: [], unclear: [{ field: "base_salary", label: "שכר בסיס חודשי (ברוטו)" }], from_image: false });
    await attach(user, new File(["x"], "form.csv", { type: "text/csv" }));

    const card = await within(decision()).findByRole("region", { name: "פרטים מהקובץ" });
    expect(card).toHaveTextContent("לא מצאנו בקובץ פרטים שכתובים בבירור");
    expect(within(card).queryByRole("button", { name: "אישור ושליחה ליועץ" })).not.toBeInTheDocument();
    await user.click(within(card).getByRole("button", { name: "סגירה" }));
    expect(within(decision()).queryByRole("region", { name: "פרטים מהקובץ" })).not.toBeInTheDocument();
  });

  it("shows the server's message when a file is refused", async () => {
    const { server, user } = await start();
    server.reply("POST /api/sessions/:id", 415, { error: { code: "unsupported_file", message: "סוג הקובץ אינו נתמך." } });
    await attach(user, new File(["x"], "data.txt", { type: "text/plain" }));
    expect(await within(decision()).findByRole("alert")).toHaveTextContent("סוג הקובץ אינו נתמך.");
  });

  it("refuses a file over the limit without sending it", async () => {
    const { server, user } = await start();
    const big = new File(["x"], "big.pdf", { type: "application/pdf" });
    Object.defineProperty(big, "size", { value: MAX_FILE_BYTES + 1 });
    await attach(user, big);
    expect(await within(decision()).findByRole("alert")).toHaveTextContent("הקובץ גדול מדי");
    expect(uploads(server)).toEqual([]);
  });
});
