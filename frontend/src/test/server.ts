// A scripted stand-in for the backend: queued responses per route, every request recorded.
// No real network and no Gemini.

import { vi } from "vitest";
import type { Car, ChatResponse, Comparison } from "../api/types";

export const SESSION = "a".repeat(32);
export const SESSION_2 = "b".repeat(32);
export const TOKEN = "signed.resume-token";

type Reply = { status: number; body?: unknown } | { network: true } | Promise<{ status: number; body?: unknown }>;
export type Recorded = { method: string; path: string; body: unknown };

export function mockServer() {
  const queues = new Map<string, Reply[]>();
  const requests: Recorded[] = [];

  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input);
    const method = (init?.method ?? "GET").toUpperCase();
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    requests.push({ method, path, body });
    const collapse = path.startsWith("/api/sessions/") && path !== "/api/sessions/restore";
    const key = `${method} ${collapse ? "/api/sessions/:id" : path}`;
    const reply = await (queues.get(key)?.shift() ?? { status: 500, body: { error: { code: "unscripted", message: "x" } } });
    if ("network" in reply) throw new TypeError("Failed to fetch");
    return new Response(reply.body === undefined ? null : JSON.stringify(reply.body), {
      status: reply.status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);

  const queue = (key: string, ...replies: Reply[]) => {
    queues.set(key, [...(queues.get(key) ?? []), ...replies]);
  };

  return {
    requests,
    chats: () => requests.filter((r) => r.path === "/api/chat"),
    session: (id = SESSION) => queue("POST /api/sessions", { status: 201, body: { session_id: id, ui_state: "idle" } }),
    chat: (...responses: ChatResponse[]) => queue("POST /api/chat", ...responses.map((body) => ({ status: 200, body }))),
    chatStatus: (status: number, code: string, message?: string) =>
      queue("POST /api/chat", { status, body: { error: { code, message: message ?? "הודעה מהשרת" } } }),
    chatDeferred: () => {
      let resolve!: (body: ChatResponse) => void;
      queue("POST /api/chat", new Promise((r) => { resolve = (body) => r({ status: 200, body }); }));
      return (body: ChatResponse) => resolve(body);
    },
    deleted: () => queue("DELETE /api/sessions/:id", { status: 204 }),
    restores: () => requests.filter((r) => r.path === "/api/sessions/restore"),
    restore: (id = SESSION_2, turn = 1) =>
      queue("POST /api/sessions/restore", { status: 201, body: { session_id: id, turn, ui_state: "idle" } }),
    restoreGone: () =>
      queue("POST /api/sessions/restore", { status: 410, body: { error: { code: "resume_unavailable", message: "לא ניתן לשחזר" } } }),
  };
}

// --- Fixtures shaped like backend/app/api/schemas.py --------------------------------------------

export const COROLLA_CROSS: Car = {
  car_id: "CAR-09", manufacturer: "טויוטה יפן", commercial_name: "COROLLA CROSS", registration_year: 2025,
  group: "ג", electric: false, family_track: false,
};
export const COROLLA: Car = { ...COROLLA_CROSS, car_id: "CAR-07", manufacturer: "טויוטה טורקיה", commercial_name: "COROLLA", registration_year: 2026 };
export const ATTO_3: Car = {
  car_id: "CAR-10", manufacturer: "BYD", commercial_name: "ATTO 3", registration_year: 2025, group: "ג",
  electric: true, family_track: false,
};
export const CARNIVAL: Car = {
  car_id: "CAR-19", manufacturer: "קיה", commercial_name: "CARNIVAL", registration_year: 2025, group: "משפחתי",
  electric: false, family_track: true,
};

export function response(over: Partial<ChatResponse> & Pick<ChatResponse, "ui_state">): ChatResponse {
  return { session_id: SESSION, turn: 1, reply: "תשובת היועץ", retryable: false, ...over };
}

/** The verified E-001 engine output, as the API returns it. */
export function e001(over: Partial<Comparison> = {}): Comparison {
  return {
    status: "estimate",
    estimated_fields: ["alternative.participation"],
    current: { kind: "company_car", car: COROLLA_CROSS, net: 11032, use_value: 4070, participation: 300, participation_source: "stated", maintenance: 0 },
    alternative: { kind: "company_car", car: ATTO_3, net: 11693, use_value: 3130, participation: 300, participation_source: "catalogue_default", maintenance: 0 },
    net_difference: 661,
    changes: [{ item: "use_value", current: 4070, alternative: 3130 }],
    electric_benefit: { net: 250, grossed_up: true, applies_to: "alternative" },
    ...over,
  };
}

export const E001_REPLY =
  "נטו היום (COROLLA CROSS): 11,032 ₪\nנטו בחלופה (ATTO 3): 11,693 ₪\nהפרש: +661 ₪ בחודש\n" +
  "מה משתנה: שווי שימוש 4,070 → 3,130 ₪ · +250 ₪ נטו החזר חשמל (מגולם)";
