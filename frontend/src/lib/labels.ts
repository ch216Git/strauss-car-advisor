// Hebrew wording for the API's machine-readable values. Labels only: no rule is decided here.

import type { Car, UiState } from "../api/types";

const FIELD_LABELS: Record<string, string> = {
  occupation: "התפקיד",
  eligible_children: "מספר הילדים הזכאים",
  current: "הרכב היום",
  "current.participation": "השתתפות עצמית היום",
  base_salary: "שכר בסיס חודשי (ברוטו)",
  commissions: "עמלות",
  bonus: "בונוס",
  pension_rate: "הפרשה לפנסיה",
  training_fund_rate: "הפרשה לקרן השתלמות",
  credit_points: "נקודות זיכוי",
};

export function fieldLabel(field: string): string {
  return FIELD_LABELS[field] ?? "פרט נוסף";
}

export const CHANGE_LABELS: Record<string, { label: string; hint?: string }> = {
  use_value: { label: "שווי שימוש", hint: "נזקף לשכר לצורך חישוב המס, לא נכנס לחשבון" },
  maintenance: { label: "אחזקת רכב" },
  participation: { label: "השתתפות עצמית" },
  family_share: { label: "חלק במסלול המשפחתי" },
};

// The backend's rules.json chooseFrom values, in the backend's own wording.
export const CHOOSE_FROM: Record<string, string> = {
  own_group_or_lower: "אפשר לבחור רכב מהקבוצה שלך או מקבוצה נמוכה ממנה.",
};

export function carName(car: Car | undefined): string {
  return car ? car.commercial_name : "";
}

export function carTag(car: Car | undefined): string | null {
  if (!car) return null;
  if (car.family_track) return "מסלול משפחתי";
  if (car.electric) return "חשמלי";
  return null;
}

export type StepStatus = "done" | "active" | "todo" | "blocked";
export type Step = { label: string; status: StepStatus; note: string };

const STEP_LABELS = ["התאמה למסלול", "פרטים לחישוב", "השוואת נטו"] as const;

/** Progress shown for a screen state; the state itself comes from the API. */
export function stepsFor(
  state: UiState,
  comparisonStatus?: "estimate" | "confirmed",
  hasAlternative = true, // false: details are collected before any alternative was checked
): Step[] | null {
  const plan: Record<string, [StepStatus, string][]> = {
    eligibility_check: [["active", "בודקים עכשיו"], ["todo", "רק אחרי ההתאמה"], ["todo", ""]],
    not_eligible: [["blocked", "לא ניתן לבחור כעת"], ["todo", "לא נדרש"], ["todo", ""]],
    collecting: hasAlternative
      ? [["done", "עבר"], ["active", "חסרים פרטים"], ["todo", ""]]
      : [["todo", "עוד לא נבחרה חלופה"], ["active", "חסרים פרטים"], ["todo", ""]],
    comparison: [["done", "עבר"], ["done", "התקבלו"], ["done", comparisonStatus === "estimate" ? "הערכה" : "מוכנה"]],
  };
  const key = state === "estimate" || state === "confirmed" || (state === "representative" && comparisonStatus)
    ? "comparison"
    : state;
  const steps = plan[key];
  if (!steps) return null;
  return steps.map(([status, note], index) => ({ label: STEP_LABELS[index] ?? "", status, note }));
}
