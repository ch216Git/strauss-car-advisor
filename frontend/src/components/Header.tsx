import { Icon } from "./Icon";

export function Header({ onNewConversation, disabled }: { onNewConversation: () => void; disabled: boolean }) {
  return (
    <header className="header">
      <div className="header__brand">
        <span className="header__logo">
          <Icon name="car" />
        </span>
        <h1 className="header__title">יועץ הרכב</h1>
        <span className="header__subtitle">רכב חברה והנטו שלך</span>
      </div>
      <button type="button" className="button button--ghost" onClick={onNewConversation} disabled={disabled}>
        <Icon name="plus" size={16} />
        שיחה חדשה
      </button>
    </header>
  );
}
