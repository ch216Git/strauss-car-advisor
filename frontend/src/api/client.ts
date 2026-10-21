// The only module that talks to the backend. It sends a session ID and the employee's message,
// never conversation state, and turns every failure into a safe, typed result.

import type { AnswerRequest, ApiErrorBody, Catalogue, ChatResponse, DocumentDetails, Queue, QueueQuestion, Representative, SessionCreated, SessionRestored } from "./types";

export type ApiFailure = {
  ok: false;
  status: number; // HTTP status; 0 when the server could not be reached
  code: string; // backend error code, or "network_error" / "unexpected_response"
  message?: string; // the backend's safe Hebrew message, when it sent one
};

export type ApiResult<T> = { ok: true; data: T } | ApiFailure;

const JSON_HEADERS = { "Content-Type": "application/json" };

async function request<T>(path: string, init: RequestInit): Promise<ApiResult<T>> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    return { ok: false, status: 0, code: "network_error" };
  }
  if (response.status === 204) {
    return { ok: true, data: undefined as T };
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return { ok: false, status: response.status, code: "unexpected_response" };
  }
  if (response.ok) {
    return { ok: true, data: body as T };
  }
  const error = (body as Partial<ApiErrorBody>)?.error;
  return {
    ok: false,
    status: response.status,
    code: typeof error?.code === "string" ? error.code : "unexpected_response",
    message: typeof error?.message === "string" ? error.message : undefined,
  };
}

export function createSession(): Promise<ApiResult<SessionCreated>> {
  return request<SessionCreated>("/api/sessions", { method: "POST" });
}

export function deleteSession(sessionId: string): Promise<ApiResult<void>> {
  return request<void>(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

export function restoreSession(resumeToken: string): Promise<ApiResult<SessionRestored>> {
  return request<SessionRestored>("/api/sessions/restore", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ resume_token: resumeToken }),
  });
}

export function sendChat(sessionId: string, message: string): Promise<ApiResult<ChatResponse>> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ session_id: sessionId, message }),
  });
}

// --- The transport representative ------------------------------------------------------------

/** The employee's questions and the representative's answers, between turns. */
export function getRepresentative(sessionId: string): Promise<ApiResult<Representative>> {
  return request<Representative>(`/api/sessions/${encodeURIComponent(sessionId)}/representative`, { method: "GET" });
}

export function getQueue(): Promise<ApiResult<Queue>> {
  return request<Queue>("/api/representative/questions", { method: "GET" });
}

export function getCatalogue(): Promise<ApiResult<Catalogue>> {
  return request<Catalogue>("/api/representative/cars", { method: "GET" });
}

export function sendAnswer(answer: AnswerRequest): Promise<ApiResult<QueueQuestion>> {
  return request<QueueQuestion>("/api/representative/answers", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(answer),
  });
}

/** Reads details from a file; nothing is saved until the employee confirms them in the chat. */
export function uploadDocument(sessionId: string, file: File): Promise<ApiResult<DocumentDetails>> {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentDetails>(`/api/sessions/${encodeURIComponent(sessionId)}/document`, { method: "POST", body: form });
}
