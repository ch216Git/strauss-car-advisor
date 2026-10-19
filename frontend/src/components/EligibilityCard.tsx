import { useState } from "react";
import type { ChatResponse, EligibilityFacts } from "../api/types";
import { CHOOSE_FROM, carName, fieldLabel } from "../lib/labels";
import { Icon } from "./Icon";

type Props = { response: ChatResponse; onAnswer: (text: string) => void; disabled: boolean };

function alternativeName(response: ChatResponse): string {
  if (response.alternative?.kind === "waiver") return "ויתור על הרכב";
  return carName(response.alternative?.car) || "החלופה";
}

/** eligibility_check: only the facts that decide eligibility, whichever fields the backend asks for. */
export function EligibilityCheck({ response, onAnswer, disabled }: Props) {
  const missing = response.eligibility?.missing ?? [];
  return (
    <section className="card card--check" aria-labelledby="eligibility-title">
      <div className="card__title-row">
        <Icon name="shield" size={26} />
        <h2 id="eligibility-title" className="card__title">בודקים קודם התאמה למסלול</h2>
      </div>
      <p className="card__lead">
        לפני שמחשבים נטו עבור {alternativeName(response)}, צריך לדעת רק {missing.length === 1 ? "דבר אחד" : "כמה דברים"}:
      </p>
      <ul className="facts-needed">
        {missing.map((field) => (
          <li key={field} className="fact-needed">
            <span className="fact-needed__mark" aria-hidden="true">?</span>
            <span className="fact-needed__label">{fieldLabel(field)}</span>
            <QuickAnswer field={field} onAnswer={onAnswer} disabled={disabled} />
          </li>
        ))}
      </ul>
      <p className="card__note">פרטי שכר יתבקשו רק אחרי בדיקת ההתאמה, ורק אם צריך.</p>
    </section>
  );
}

/** A shortcut for answering one decisive field; the answer is sent as the employee's own chat message. */
function QuickAnswer({ field, onAnswer, disabled }: { field: string; onAnswer: (text: string) => void; disabled: boolean }) {
  const [count, setCount] = useState(0);
  const [text, setText] = useState("");

  if (field === "eligible_children") {
    return (
      <div className="quick">
        <button type="button" className="stepper__button" aria-label="פחות" onClick={() => setCount((n) => Math.max(0, n - 1))}>
          −
        </button>
        <output className="stepper__value" aria-live="polite" aria-label="מספר הילדים">{count}</output>
        <button type="button" className="stepper__button" aria-label="יותר" onClick={() => setCount((n) => Math.min(20, n + 1))}>
          +
        </button>
        <button type="button" className="button" disabled={disabled} onClick={() => onAnswer(`יש לי ${count} ילדים`)}>
          שליחה
        </button>
      </div>
    );
  }
  if (field === "occupation") {
    return (
      <form
        className="quick"
        onSubmit={(event) => {
          event.preventDefault();
          if (text.trim()) onAnswer(`התפקיד שלי: ${text.trim()}`);
        }}
      >
        <label htmlFor="quick-occupation" className="sr-only">התפקיד שלך</label>
        <input id="quick-occupation" className="quick__input" value={text} onChange={(e) => setText(e.target.value)} placeholder="התפקיד שלך" />
        <button type="submit" className="button" disabled={disabled || !text.trim()}>
          שליחה
        </button>
      </form>
    );
  }
  return <span className="fact-needed__hint">אפשר לענות בשיחה</span>;
}

/** not_eligible: the backend's verdict and the facts it rests on, as the backend gave them. */
export function NotEligible({ response }: { response: ChatResponse }) {
  const { reason, facts } = response.eligibility ?? {};
  const name = response.alternative?.kind === "waiver" ? "בוויתור על הרכב" : `ב-${alternativeName(response)}`;
  return (
    <section className="card card--refused" aria-labelledby="refused-title">
      <div className="card__title-row">
        <Icon name="blocked" size={26} />
        <h2 id="refused-title" className="card__title">אי אפשר לבחור כעת {name}</h2>
      </div>
      {facts && <FactRows reason={reason} facts={facts} />}
      <p className="card__note">זה נוגע לבחירה חדשה בלבד. אפשר לבדוק חלופה אחרת בשיחה.</p>
    </section>
  );
}

function FactRows({ reason, facts }: { reason?: string; facts: EligibilityFacts }) {
  const rows: [string, string][] = [];
  if (reason === "not_enough_children" && facts.min_children != null) {
    rows.push(["המסלול המשפחתי מיועד למי שיש לו", `לפחות ${facts.min_children} ילדים עד גיל ${facts.child_max_age ?? ""}`.trim()]);
    if (facts.eligible_children != null) rows.push(["לפי מה שמסרת", String(facts.eligible_children)]);
  } else if (reason === "group_too_high") {
    if (facts.car_group) rows.push(["קבוצת הרכב", facts.car_group]);
    if (facts.employee_group) rows.push(["הקבוצה שלך", facts.employee_group]);
  } else if (reason === "waiver_not_allowed" && facts.occupation) {
    rows.push(["התפקיד", facts.occupation]);
  }
  const rule = reason === "group_too_high" && facts.choose_from ? CHOOSE_FROM[facts.choose_from] : undefined;
  if (rows.length === 0 && !rule) return null;
  return (
    <div className="fact-rows">
      {rows.length > 0 && (
        <dl>
          {rows.map(([term, value]) => (
            <div key={term} className="fact-row">
              <dt>{term}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {rule && <p className="fact-rows__rule">{rule}</p>}
    </div>
  );
}
