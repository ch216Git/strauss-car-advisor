// A file the employee uploads: the backend returns only the details it is certain of. Nothing is
// saved until the employee confirms them, as their own message in the chat.

import { useCallback, useEffect, useRef, useState } from "react";
import { uploadDocument } from "../api/client";
import type { DocumentDetails } from "../api/types";

export const ACCEPTED_FILES = ".pdf,.png,.jpg,.jpeg,.webp,.txt,.csv,.xlsx";
export const MAX_FILE_BYTES = 5 * 1024 * 1024; // the API's limit

export const DOCUMENT_TEXT = {
  tooLarge: "הקובץ גדול מדי. אפשר להעלות קובץ עד 5MB.",
  failed: "לא הצלחנו לקרוא את הקובץ כרגע. אפשר לנסות שוב או לכתוב את הפרטים בשיחה.",
};

export type DocumentState =
  | { status: "idle" }
  | { status: "reading"; name: string }
  | { status: "ready"; name: string; details: DocumentDetails }
  | { status: "failed"; name: string; message: string };

export function useDocumentUpload(sessionId: string | null) {
  const [state, setState] = useState<DocumentState>({ status: "idle" });
  const request = useRef(0); // a newer upload, a clear or a new conversation drops a late reply

  const clear = useCallback(() => {
    request.current += 1;
    setState({ status: "idle" });
  }, []);

  useEffect(() => clear(), [sessionId, clear]);

  const upload = useCallback(async (file: File) => {
    if (!sessionId) return;
    const mine = ++request.current;
    if (file.size > MAX_FILE_BYTES) {
      setState({ status: "failed", name: file.name, message: DOCUMENT_TEXT.tooLarge });
      return;
    }
    setState({ status: "reading", name: file.name });
    const result = await uploadDocument(sessionId, file);
    if (mine !== request.current) return;
    setState(result.ok
      ? { status: "ready", name: file.name, details: result.data }
      : { status: "failed", name: file.name, message: result.message ?? DOCUMENT_TEXT.failed });
  }, [sessionId]);

  return { state, upload, clear };
}

/** The confirmation, sent through the chat as the employee's own words. */
export function documentMessage(found: DocumentDetails["found"], unclear: DocumentDetails["unclear"]): string {
  const lines = found.length > 0
    ? ["אלה הפרטים מהקובץ שהעליתי, ובדקתי שהם נכונים:", ...found.map((f) => `• ${f.label}: ${f.value}`)]
    : ["העליתי קובץ, אבל לא נמצאו בו פרטים ברורים."];
  if (unclear.length > 0) lines.push(`לא היו ברורים בקובץ: ${unclear.map((f) => f.label).join(", ")}.`);
  return lines.join("\n");
}
