import { Icon } from "./Icon";

export function Header({ onNewConversation, onOpenHistory, disabled }: {
  onNewConversation: () => void;
  onOpenHistory: () => void;
  disabled: boolean;
}) {
  return (
    <header className="header">
      <div className="header__brand">
        <span className="header__logo">
          <Icon name="car" />
        </span>
        <h1 className="header__title">יועץ הרכב</h1>
        <span className="header__subtitle">רכב חברה והנטו שלך</span>
      </div>
      <div className="header__actions">
        <button type="button" className="button button--ghost" onClick={onOpenHistory} disabled={disabled}>
          <Icon name="history" size={16} />
          היסטוריה
        </button>
        <button type="button" className="button button--ghost" onClick={onNewConversation} disabled={disabled}>
          <Icon name="plus" size={16} />
          שיחה חדשה
        </button>
        <a className="button button--ghost" href="#/representative">למסך הנציג</a>
      </div>
    </header>
  );
}
