import { Spinner } from "./Icon";

/** Shown while a chat request is in flight. Neutral: it does not say what the server is doing. */
export function PendingState({ variant = "bubble" }: { variant?: "bubble" | "chip" }) {
  return (
    <div className={`pending pending--${variant}`} role="status" aria-live="polite">
      <Spinner size={variant === "chip" ? 14 : 18} />
      <span>בודקים…</span>
    </div>
  );
}
