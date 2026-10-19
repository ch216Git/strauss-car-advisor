import type { ChatResponse } from "../api/types";
import { CarChoiceCard } from "./CarChoiceCard";
import { CaseHeader } from "./CaseHeader";
import { ComparisonPanel } from "./ComparisonPanel";
import { EligibilityCheck, NotEligible } from "./EligibilityCard";
import { Icon } from "./Icon";
import { MissingDetailsCard } from "./MissingDetailsCard";
import { PendingState } from "./PendingState";
import { RepresentativeCard } from "./RepresentativeCard";

/** Renders the latest decisive API response by its ui_state; nothing is inferred from reply text. */
export function DecisionPanel({ decision, pending, canSend, onSend }: {
  decision: ChatResponse | null;
  pending: boolean;
  canSend: boolean;
  onSend: (text: string) => void;
}) {
  return (
    <section className="panel decision" aria-labelledby="decision-title">
      <header className="panel__head">
        <h2 id="decision-title" className="panel__title">ההחלטה שלך</h2>
        {pending && <PendingState variant="chip" />}
      </header>
      <div className="decision__body">{decision ? <Decision response={decision} canSend={canSend} onSend={onSend} /> : <Empty />}</div>
    </section>
  );
}

function Decision({ response, canSend, onSend }: { response: ChatResponse; canSend: boolean; onSend: (text: string) => void }) {
  const disabled = !canSend;
  const comparison = response.comparison;
  const questionOpen = !!response.representative?.questions.some(
    (q) => q.status === "open" && q.topic === "alternative_participation",
  );
  const showComparison = (state: string) => comparison && (state === "estimate" || state === "confirmed" || state === "representative");

  switch (response.ui_state) {
    case "eligibility_check":
      return (
        <>
          <CaseHeader response={response} />
          <EligibilityCheck response={response} onAnswer={onSend} disabled={disabled} />
        </>
      );
    case "not_eligible":
      return (
        <>
          <CaseHeader response={response} />
          <NotEligible response={response} />
        </>
      );
    case "collecting":
      return (
        <>
          <CaseHeader response={response} />
          {response.details && <MissingDetailsCard details={response.details} />}
        </>
      );
    case "car_choice":
      return response.car_choice ? <CarChoiceCard choice={response.car_choice} onChoose={onSend} disabled={disabled} /> : <Empty />;
    case "estimate":
    case "confirmed":
    case "representative":
      return (
        <>
          {comparison && <CaseHeader response={response} />}
          {response.ui_state === "representative" && response.representative && (
            <RepresentativeCard representative={response.representative} />
          )}
          {showComparison(response.ui_state) && comparison && (
            <ComparisonPanel comparison={comparison} onAskRepresentative={onSend} questionOpen={questionOpen} disabled={disabled} />
          )}
        </>
      );
    default:
      return <Empty />;
  }
}

function Empty() {
  return (
    <div className="empty">
      <div className="empty__icon">
        <Icon name="compare" size={36} />
      </div>
      <h3 className="empty__title">ההחלטה שלך תופיע כאן</h3>
      <p className="empty__text">שואלים על רכב במילים שלך. נבדוק קודם אם הוא מתאים לך, נבקש רק את מה שחסר, ונציג כמה ישתנה הנטו.</p>
    </div>
  );
}
