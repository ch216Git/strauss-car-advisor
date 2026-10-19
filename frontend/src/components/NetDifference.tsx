import { signedShekels } from "../lib/format";
import { Amount } from "./BidiText";
import { EstimateBadge } from "./EstimateBadge";

/** The result's headline: the API's net_difference, the strongest figure on the screen. */
export function NetDifference({ amount, estimate, catalogueDefault }: {
  amount: number;
  estimate: boolean;
  catalogueDefault: boolean;
}) {
  const tone = amount > 0 ? "gain" : amount < 0 ? "loss" : "even";
  return (
    <div className={`hero hero--${tone}`}>
      <div className="hero__top">
        <span className="hero__label">{estimate ? "הערכת שינוי בנטו" : "השינוי בנטו"}</span>
        {estimate && <EstimateBadge />}
      </div>
      <p className="hero__figure">
        <Amount className="hero__amount">{signedShekels(amount)}</Amount>
        <span className="hero__unit">בחודש</span>
      </p>
      {catalogueDefault && (
        <p className="hero__note">ההשתתפות ברכב החלופי בחישוב הזה מבוססת על ברירת המחדל מהקטלוג.</p>
      )}
    </div>
  );
}
