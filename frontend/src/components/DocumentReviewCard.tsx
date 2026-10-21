import { useState } from "react";
import { type DocumentState, documentMessage } from "../hooks/useDocumentUpload";
import { Icon, Spinner } from "./Icon";

/** What was read from the file, for the employee to check before anything is saved. */
export function DocumentReviewCard({ state, onConfirm, onClose, disabled }: {
  state: Exclude<DocumentState, { status: "idle" }>;
  onConfirm: (message: string) => void;
  onClose: () => void;
  disabled: boolean;
}) {
  return (
    <section className="card card--document" aria-labelledby="document-title" aria-live="polite">
      <div className="card__title-row">
        <Icon name="attach" size={22} />
        <h2 id="document-title" className="card__title card__title--small">
          {state.status === "reading" ? "קוראים את הקובץ…" : "פרטים מהקובץ"}
        </h2>
      </div>
      <p className="card__note document__name">{state.name}</p>
      {state.status === "reading" && (
        <p className="card__note document__reading"><Spinner /> לוקחים רק פרטים שכתובים בקובץ בבירור.</p>
      )}
      {state.status === "failed" && (
        <>
          <p className="notice" role="alert">{state.message}</p>
          <button type="button" className="button button--ghost" onClick={onClose}>סגירה</button>
        </>
      )}
      {state.status === "ready" && (
        <Review key={state.name} details={state.details} onConfirm={onConfirm} onClose={onClose} disabled={disabled} />
      )}
    </section>
  );
}

function Review({ details, onConfirm, onClose, disabled }: {
  details: Extract<DocumentState, { status: "ready" }>["details"];
  onConfirm: (message: string) => void;
  onClose: () => void;
  disabled: boolean;
}) {
  const [kept, setKept] = useState<Set<string>>(() => new Set(details.found.map((f) => f.field)));
  const chosen = details.found.filter((f) => kept.has(f.field));
  const nothing = details.found.length === 0;

  const toggle = (field: string) =>
    setKept((current) => {
      const next = new Set(current);
      if (next.has(field)) next.delete(field);
      else next.add(field);
      return next;
    });

  return (
    <>
      {nothing ? (
        <p className="card__lead">לא מצאנו בקובץ פרטים שכתובים בבירור. אפשר לכתוב אותם בשיחה.</p>
      ) : (
        <>
          <p className="card__lead">אלה הפרטים שכתובים בקובץ בבירור. סמנו רק את מה שנכון, והיועץ ישמור אותם.</p>
          {details.from_image && (
            <p className="notice">הקובץ נקרא מתמונה או מסריקה, ולכן חשוב לבדוק כל ערך היטב.</p>
          )}
          <ul className="document__list">
            {details.found.map((f) => (
              <li key={f.field}>
                <label className="document__item">
                  <input type="checkbox" checked={kept.has(f.field)} onChange={() => toggle(f.field)} />
                  <span className="document__label">{f.label}</span>
                  <bdi className="document__value">{f.value}</bdi>
                </label>
              </li>
            ))}
          </ul>
        </>
      )}
      {details.unclear.length > 0 && (
        <p className="card__note">
          לא היו ברורים בקובץ, והיועץ ישאל עליהם: {details.unclear.map((f) => f.label).join(", ")}
        </p>
      )}
      <div className="document__actions">
        {!nothing && (
          <button
            type="button" className="button" disabled={disabled || chosen.length === 0}
            onClick={() => onConfirm(documentMessage(chosen, details.unclear))}
          >
            אישור ושליחה ליועץ
          </button>
        )}
        <button type="button" className="button button--ghost" onClick={onClose}>
          {nothing ? "סגירה" : "ביטול"}
        </button>
      </div>
    </>
  );
}
