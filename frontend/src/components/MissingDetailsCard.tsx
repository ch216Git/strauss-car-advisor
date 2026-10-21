import type { Details } from "../api/types";
import { fieldLabel } from "../lib/labels";

/** collecting: eligibility passed; which calculation details are given and which are still missing. */
export function MissingDetailsCard({ details, hasAlternative = true }: { details: Details; hasAlternative?: boolean }) {
  return (
    <section className="card" aria-labelledby="details-title">
      <h2 id="details-title" className="card__title card__title--small">כדי לחשב את השינוי בנטו, חסרים רק אלה</h2>
      <ul className="checklist">
        {details.provided.map((field) => (
          <li key={field} className="checklist__item checklist__item--done">
            <span className="checklist__dot" aria-hidden="true">✓</span>
            <span className="checklist__label">{fieldLabel(field)}</span>
            <span className="checklist__state">נמסר</span>
          </li>
        ))}
        {details.missing.map((field) => (
          <li key={field} className="checklist__item">
            <span className="checklist__dot" aria-hidden="true" />
            <span className="checklist__label">{fieldLabel(field)}</span>
            <span className="checklist__state">חסר</span>
          </li>
        ))}
      </ul>
      {!hasAlternative && <p className="card__note">עוד לא נבחר רכב חלופי. אפשר לכתוב בשיחה איזה רכב תרצה/י לבדוק.</p>}
      <p className="card__note">אפשר לענות על הכול בהודעה אחת בשיחה.</p>
    </section>
  );
}
