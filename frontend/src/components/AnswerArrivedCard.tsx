import type { RepresentativeQuestion } from "../api/types";
import { BidiText } from "./BidiText";
import { Icon } from "./Icon";

/** Sent through the chat, as the employee's own words: the advisor then uses the answer. */
export const CONTINUE_MESSAGE = "קיבלתי תשובה מנציג. אפשר להמשיך לפיה?";

const TOPIC_TITLE: Record<string, string> = {
  alternative_participation: "ההשתתפות ברכב החלופי אושרה",
  current_car_not_in_catalogue: "נציג ענה על הרכב הנוכחי",
  other: "נציג התחבורה ענה על השאלה",
};

export function AnswerArrivedCard({ answers, onContinue, disabled }: {
  answers: RepresentativeQuestion[];
  onContinue: (text: string) => void;
  disabled: boolean;
}) {
  return (
    <section className="card card--answer" aria-labelledby="answer-title" aria-live="polite">
      <div className="card__title-row">
        <Icon name="message" size={24} />
        <h2 id="answer-title" className="card__title card__title--small">התקבלה תשובה מנציג</h2>
      </div>
      <ul className="answers">
        {answers.map((q) => (
          <li key={q.question_id} className="answers__item">
            <span className="answers__topic">{TOPIC_TITLE[q.topic] ?? TOPIC_TITLE.other}</span>
            {q.answer && <p className="answers__text"><BidiText text={q.answer} /></p>}
          </li>
        ))}
      </ul>
      <button type="button" className="button" disabled={disabled} onClick={() => onContinue(CONTINUE_MESSAGE)}>
        המשך לפי התשובה
      </button>
    </section>
  );
}
