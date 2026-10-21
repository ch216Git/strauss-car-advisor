import { useEffect, useRef } from "react";
import type { Advisor } from "../hooks/useAdvisor";
import { ChatMessage } from "./ChatMessage";
import { ErrorCard } from "./ErrorCard";
import { MessageComposer } from "./MessageComposer";
import { PendingState } from "./PendingState";

// Example questions for an empty conversation (interface copy, not advisor replies).
const SUGGESTIONS = ["אפשר לקבל קרניבל?", "יש לי קורולה קרוס. כמה ישתנה הנטו אם אעבור לאטו 3?"];
const GREETING = "שלום! על איזה רכב חשבת? אפשר לשאול במילים שלך, ואבדוק מה זה אומר עבורך.";

export function AdvisorPanel({ advisor, onAttach }: { advisor: Advisor; onAttach?: (file: File) => void }) {
  const { entries, pending, status, notice, draft, canSend } = advisor;
  const log = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = log.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [entries, pending]);

  return (
    <section className="panel advisor" aria-labelledby="advisor-title">
      <header className="panel__head">
        <h2 id="advisor-title" className="panel__title">השיחה</h2>
        <span className="panel__hint">אפשר לשאול במילים שלך</span>
      </header>

      <div ref={log} className="advisor__log" role="log" aria-live="polite" aria-relevant="additions" aria-label="הודעות">
        <div className="bubble bubble--advisor">
          <p className="bubble__text">{GREETING}</p>
        </div>
        {entries.map((entry) => (
          <ChatMessage key={entry.id} entry={entry} onRetry={advisor.retry} />
        ))}
        {pending && <PendingState />}
      </div>

      <div className="advisor__foot">
        {status === "failed" && (
          <ErrorCard text="אין חיבור לשרת כרגע, ולכן אי אפשר להתחיל שיחה." onRetry={() => void advisor.retrySession()} />
        )}
        {notice && (
          <p className="notice" role="status">
            {notice}
          </p>
        )}
        {entries.length === 0 && status === "ready" && (
          <div className="suggestions" aria-label="שאלות לדוגמה">
            {SUGGESTIONS.map((text) => (
              <button key={text} type="button" className="chip" onClick={() => advisor.send(text)} disabled={!canSend}>
                {text}
              </button>
            ))}
          </div>
        )}
        <MessageComposer value={draft} onChange={advisor.setDraft} onSend={advisor.send} disabled={!canSend} onAttach={onAttach} />
      </div>
    </section>
  );
}
