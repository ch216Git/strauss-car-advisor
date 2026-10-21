// Answers from the transport representative that arrived after the decision on screen. While the
// decision shows an open question, the server is asked every REPRESENTATIVE_POLL_MS; nothing is
// decided here: the employee continues through the chat, and the advisor uses the answer. Once the
// employee writes again (sent changes), the advisor already has the answer, so it is no longer "new".

import { useEffect, useRef, useState } from "react";
import { getRepresentative } from "../api/client";
import type { ChatResponse, RepresentativeQuestion } from "../api/types";

export const REPRESENTATIVE_POLL_MS = 5000;

export function useRepresentativeAnswers(
  sessionId: string | null,
  decision: ChatResponse | null,
  paused: boolean,
  sent: number, // the employee's messages so far
  onToken?: (token: string) => void, // a fresh resume token, already holding the answers, for the history
): RepresentativeQuestion[] {
  const [arrived, setArrived] = useState<RepresentativeQuestion[]>([]);
  const tokenSink = useRef(onToken);
  tokenSink.current = onToken;
  const openIds = (decision?.representative?.questions ?? []).filter((q) => q.status === "open").map((q) => q.question_id);
  const watching = openIds.join(",");

  useEffect(() => {
    setArrived([]); // a new decision, or a new message, already reflects whatever was answered before it
  }, [decision, sent]);

  useEffect(() => {
    if (!sessionId || !watching || paused) return;
    const open = new Set(watching.split(","));
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const check = async () => {
      const result = await getRepresentative(sessionId);
      if (stopped) return;
      if (result.ok) {
        if (result.data.resume_token) tokenSink.current?.(result.data.resume_token);
        const answered = result.data.questions.filter((q) => q.status === "answered" && open.has(q.question_id));
        if (answered.length > 0) {
          setArrived(answered);
          if (answered.length === open.size) return; // every watched question is answered: stop asking
        }
      }
      timer = setTimeout(() => void check(), REPRESENTATIVE_POLL_MS);
    };
    timer = setTimeout(() => void check(), REPRESENTATIVE_POLL_MS);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [sessionId, watching, paused]);

  return arrived;
}
