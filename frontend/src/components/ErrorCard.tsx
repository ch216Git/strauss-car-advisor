import { Icon } from "./Icon";

/** A safe error message (never exception text), with Retry when the same request can be sent again. */
export function ErrorCard({
  text,
  onRetry,
  variant = "card",
}: {
  text: string;
  onRetry?: () => void;
  variant?: "card" | "bubble";
}) {
  return (
    <div className={`error-card error-card--${variant}`} role="alert">
      <div className="error-card__body">
        <Icon name="info" size={18} />
        <p>{text}</p>
      </div>
      {onRetry && (
        <button type="button" className="button button--warning" onClick={onRetry}>
          לנסות שוב
        </button>
      )}
    </div>
  );
}
