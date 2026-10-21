// The local-history sheet: past conversations from this browser only. Opening a row loads it with
// no network call (the hook hydrates from localStorage). Accessible dialog: Esc closes, focus is
// trapped and restored, and it is full-screen on narrow viewports (styling in app.css).

import { useEffect, useRef, useState } from "react";
import type { ChatResponse } from "../api/types";
import type { Conversation } from "../lib/history";
import { signedShekels } from "../lib/format";
import { Amount } from "./BidiText";
import { Icon } from "./Icon";

const DATE = new Intl.DateTimeFormat("he-IL", { dateStyle: "medium", timeStyle: "short" });

export function HistoryDrawer({
  conversations, activeId, enabled, busy, onClose, onOpen, onDelete, onDeleteAll, onToggleEnabled,
}: {
  conversations: Conversation[];
  activeId: string | null;
  enabled: boolean;
  busy: boolean;
  onClose: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onDeleteAll: () => void;
  onToggleEnabled: (on: boolean) => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null); // a row pending delete confirmation
  const [confirmAll, setConfirmAll] = useState(false);

  // Esc to close; trap Tab within the panel; restore focus to the opener on unmount.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    const panel = panelRef.current;
    panel?.querySelector<HTMLElement>("[data-autofocus]")?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panel) return;
      const focusable = panel.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!first || !last) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      opener?.focus?.();
    };
  }, [onClose]);

  return (
    <div className="drawer" role="presentation" onMouseDown={onClose}>
      <div
        className="drawer__panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="history-title"
        ref={panelRef}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="drawer__head">
          <h2 id="history-title" className="drawer__title">שיחות קודמות</h2>
          <button type="button" className="icon-button" onClick={onClose} aria-label="סגירה" data-autofocus>
            <Icon name="close" size={18} />
          </button>
        </header>

        <p className="drawer__privacy">השיחות נשמרות רק בדפדפן הזה.</p>

        <label className="drawer__toggle">
          <input
            type="checkbox"
            checked={enabled}
            disabled={busy}
            onChange={(e) => {
              if (!e.target.checked && conversations.length > 0) {
                if (!window.confirm("לכבות את השמירה ולמחוק את ההיסטוריה השמורה במכשיר הזה?")) return;
              }
              onToggleEnabled(e.target.checked);
            }}
          />
          שמור שיחות במכשיר הזה
        </label>

        <div className="drawer__body">
          {conversations.length === 0 ? (
            <p className="drawer__empty">אין עדיין שיחות שמורות.</p>
          ) : (
            <ul className="history-list">
              {conversations.map((c) => (
                <li key={c.id} className={`history-row${c.id === activeId ? " history-row--active" : ""}`}>
                  <button
                    type="button"
                    className="history-row__open"
                    onClick={() => onOpen(c.id)}
                    disabled={busy}
                  >
                    <span className="history-row__title">{c.title || "שיחה"}</span>
                    <span className="history-row__meta">
                      <time dateTime={new Date(c.updatedAt).toISOString()}>{DATE.format(c.updatedAt)}</time>
                      <Badge decision={c.decision} resumable={c.resumable} />
                    </span>
                  </button>
                  {confirmId === c.id ? (
                    <span className="history-row__confirm">
                      <button type="button" className="button button--danger button--small" disabled={busy}
                        onClick={() => { onDelete(c.id); setConfirmId(null); }}>
                        למחוק
                      </button>
                      <button type="button" className="button button--ghost button--small"
                        onClick={() => setConfirmId(null)}>
                        ביטול
                      </button>
                    </span>
                  ) : (
                    <button
                      type="button"
                      className="icon-button history-row__delete"
                      aria-label={`מחיקת השיחה: ${c.title || "שיחה"}`}
                      disabled={busy}
                      onClick={() => setConfirmId(c.id)}
                    >
                      <Icon name="trash" size={16} />
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>

        {conversations.length > 0 && (
          <footer className="drawer__foot">
            {confirmAll ? (
              <span className="drawer__confirm-all">
                <span>למחוק את כל ההיסטוריה?</span>
                <button type="button" className="button button--danger button--small" disabled={busy}
                  onClick={() => { onDeleteAll(); setConfirmAll(false); }}>
                  למחוק הכול
                </button>
                <button type="button" className="button button--ghost button--small" onClick={() => setConfirmAll(false)}>
                  ביטול
                </button>
              </span>
            ) : (
              <button type="button" className="button button--ghost" disabled={busy} onClick={() => setConfirmAll(true)}>
                <Icon name="trash" size={16} />
                מחיקת כל ההיסטוריה
              </button>
            )}
          </footer>
        )}
      </div>
    </div>
  );
}

/** A compact badge from the saved decision: the net difference, or "הערכה" for an estimate. */
function Badge({ decision, resumable }: { decision: ChatResponse | null; resumable: boolean }) {
  if (!resumable) return <span className="history-badge history-badge--muted">לצפייה בלבד</span>;
  const comparison = decision?.comparison;
  if (!comparison) return null;
  const estimate = comparison.status === "estimate";
  return (
    <span className={`history-badge${estimate ? " history-badge--estimate" : ""}`}>
      <Amount>{signedShekels(comparison.net_difference)}</Amount>
      {estimate && <span className="history-badge__tag">הערכה</span>}
    </span>
  );
}
