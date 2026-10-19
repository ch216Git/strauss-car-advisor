import type { Car, ChatResponse } from "../api/types";
import { carName, carTag, stepsFor } from "../lib/labels";
import { Icon } from "./Icon";

/** Today's car, the alternative and the progress steps of the case on screen. */
export function CaseHeader({ response }: { response: ChatResponse }) {
  const comparison = response.comparison;
  const current: Car | undefined = response.current ?? comparison?.current.car;
  const alternativeKind = response.alternative?.kind ?? comparison?.alternative.kind;
  const alternative: Car | undefined = response.alternative?.car ?? comparison?.alternative.car;
  const steps = stepsFor(response.ui_state, comparison?.status);
  const tag = carTag(alternative);

  return (
    <div className="case">
      <div className="case__cars">
        <div className="case__pair">
          {current && (
            <>
              <div className="car-pill">
                <span className="car-pill__label">היום</span>
                <span className="car-pill__name">{carName(current)}</span>
                <span className="car-pill__meta">{current.manufacturer}</span>
              </div>
              <Icon name="arrowLeft" className="case__arrow" />
            </>
          )}
          {(alternative || alternativeKind === "waiver") && (
            <div className="car-pill car-pill--alternative">
              <span className="car-pill__label">בחלופה</span>
              <span className="car-pill__name">{alternative ? carName(alternative) : "ויתור על הרכב"}</span>
              {alternative && <span className="car-pill__meta">{alternative.manufacturer}</span>}
            </div>
          )}
        </div>
        {tag && <span className="tag">{tag}</span>}
      </div>

      {steps && (
        <ol className="steps" aria-label="התקדמות">
          {steps.map((step, index) => (
            <li key={step.label} className={`step step--${step.status}`} aria-current={step.status === "active" ? "step" : undefined}>
              <span className="step__dot" aria-hidden="true">
                {step.status === "done" ? "✓" : step.status === "blocked" ? "!" : index + 1}
              </span>
              <span className="step__text">
                <span className="step__label">{step.label}</span>
                {step.note && <span className="step__note">{step.note}</span>}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
