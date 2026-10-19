import type { Entry } from "../hooks/useAdvisor";
import { BidiText } from "./BidiText";
import { ErrorCard } from "./ErrorCard";

export function ChatMessage({ entry, onRetry }: { entry: Entry; onRetry: (id: number) => void }) {
  if (entry.role === "error") {
    return (
      <ErrorCard
        text={entry.text}
        onRetry={entry.retryText ? () => onRetry(entry.id) : undefined}
        variant="bubble"
      />
    );
  }
  const advisor = entry.role === "advisor";
  return (
    <div className={`bubble ${advisor ? "bubble--advisor" : "bubble--employee"}`}>
      <span className="sr-only">{advisor ? "היועץ:" : "את/ה:"}</span>
      <p className="bubble__text">{advisor ? <BidiText text={entry.text} /> : entry.text}</p>
    </div>
  );
}
