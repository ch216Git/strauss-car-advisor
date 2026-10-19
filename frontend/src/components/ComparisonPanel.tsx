import type { Comparison } from "../api/types";
import { shekels } from "../lib/format";
import { CHANGE_LABELS } from "../lib/labels";
import { Amount } from "./BidiText";
import { ComparisonSide } from "./ComparisonSide";
import { Icon } from "./Icon";
import { NetDifference } from "./NetDifference";

/**
 * The comparison exactly as the API calculated it, for both estimate and confirmed results.
 * The estimate disclosure appears only when the API marks the result (or any field) as an estimate.
 */
export function ComparisonPanel({ comparison, onAskRepresentative, questionOpen, disabled }: {
  comparison: Comparison;
  onAskRepresentative: (text: string) => void;
  questionOpen: boolean;
  disabled: boolean;
}) {
  const estimate = comparison.status === "estimate" || comparison.estimated_fields.length > 0;
  const { current, alternative, electric_benefit: electric } = comparison;
  const catalogueDefault = alternative.participation_source === "catalogue_default";
  const alternativeName = alternative.car?.commercial_name ?? "הרכב החלופי";

  return (
    <section className="comparison" aria-label="השוואת נטו">
      <NetDifference amount={comparison.net_difference} estimate={estimate} catalogueDefault={catalogueDefault} />

      <div className="sides">
        <ComparisonSide side={current} label="היום" estimate={estimate} />
        <ComparisonSide side={alternative} label="בחלופה" estimate={estimate} highlight />
      </div>

      {(comparison.changes.length > 0 || electric) && (
        <table className="changes">
          <caption className="changes__caption">מה משתנה</caption>
          <thead>
            <tr>
              <th scope="col"><span className="sr-only">פריט</span></th>
              <th scope="col">היום</th>
              <th scope="col">בחלופה</th>
            </tr>
          </thead>
          <tbody>
            {comparison.changes.map((change) => {
              const label = CHANGE_LABELS[change.item];
              return (
                <tr key={change.item}>
                  <th scope="row">
                    <span className="changes__label">{label?.label ?? change.item}</span>
                    {label?.hint && <span className="changes__hint">{label.hint}</span>}
                  </th>
                  <td><Amount>{shekels(change.current)}</Amount></td>
                  <td className="changes__after"><Amount>{shekels(change.alternative)}</Amount></td>
                </tr>
              );
            })}
            {electric && (
              <tr>
                <th scope="row">
                  <span className="changes__label changes__label--icon"><Icon name="bolt" size={14} /> החזר חשמל</span>
                  {electric.grossed_up && <span className="changes__hint">מגולם: המעסיק משלם את המס</span>}
                </th>
                <td>{electric.applies_to === "current" ? <Electric net={electric.net} /> : <span className="muted">—</span>}</td>
                <td className="changes__after">
                  {electric.applies_to === "alternative" ? <Electric net={electric.net} /> : <span className="muted">—</span>}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}

      {estimate && (
        <div className="disclosure" role="note">
          <Icon name="info" size={22} />
          <div className="disclosure__body">
            {catalogueDefault ? (
              <p>
                ההשתתפות ברכב החלופי (<Amount>{shekels(alternative.participation)}</Amount>) היא ברירת מחדל מהקטלוג,
                כלומר הערכה ולא נתון מאושר.
              </p>
            ) : (
              <p>חלק מהנתונים בחישוב הזה הם הערכה, ולא נתון מאושר.</p>
            )}
            {catalogueDefault && !questionOpen && (
              <button
                type="button"
                className="button button--outline-warning"
                disabled={disabled}
                onClick={() => onAskRepresentative(`אפשר לבקש מנציג התחבורה את ההשתתפות המדויקת ב-${alternativeName}`)}
              >
                לבקש מנציג התחבורה את הסכום המדויק
              </button>
            )}
          </div>
        </div>
      )}

      <p className="footnote">לפי קטלוג הרכבים ונתוני שווי השימוש של רשות המסים. אינו ייעוץ מס.</p>
    </section>
  );
}

function Electric({ net }: { net: number }) {
  return (
    <span className="gain-text">
      <Amount>{`+${shekels(net)}`}</Amount> נטו
    </span>
  );
}
