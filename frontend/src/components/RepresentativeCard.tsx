import type { Representative } from "../api/types";
import { BidiText } from "./BidiText";
import { Icon } from "./Icon";

// The question is open with the transport representative. While this screen is open the answer
// appears here by itself (useRepresentativeAnswers); the advisor then continues from it.
const TOPIC_TEXT: Record<string, string> = {
  alternative_participation: "השאלה על ההשתתפות ברכב החלופי פתוחה אצל נציג התחבורה. בינתיים ההשוואה נשארת הערכה.",
  current_car_not_in_catalogue: "הרכב הנוכחי לא נמצא בקטלוג, ולכן אי אפשר לחשב השוואה כרגע. השאלה פתוחה אצל נציג התחבורה.",
  other: "השאלה פתוחה אצל נציג התחבורה.",
};

export function RepresentativeCard({ representative }: { representative: Representative }) {
  const open = representative.questions.filter((q) => q.status === "open");
  if (open.length === 0) return <Answered representative={representative} />;
  const latest = representative.latest;
  const topic = latest?.topic ?? open[open.length - 1]?.topic ?? "other";
  const title = latest?.already_open ? "כבר יש שאלה פתוחה בנושא הזה" : "השאלה הועברה לנציג התחבורה";

  return (
    <section className="card card--representative" aria-labelledby="representative-title">
      <div className="card__title-row">
        <Icon name="message" size={24} />
        <h2 id="representative-title" className="card__title card__title--small">{title}</h2>
      </div>
      <p className="card__lead">{TOPIC_TEXT[topic] ?? TOPIC_TEXT.other}</p>
      <p className="card__note">כשהנציג יענה, התשובה תופיע כאן.</p>
      {open.length > 1 && <p className="card__note">שאלות פתוחות אצל הנציג: {open.length}</p>}
    </section>
  );
}

/** Every question has been answered: the answers, not an open question that is no longer open. */
function Answered({ representative }: { representative: Representative }) {
  const answers = representative.questions.filter((q) => q.answer);
  return (
    <section className="card card--representative" aria-labelledby="representative-title">
      <div className="card__title-row">
        <Icon name="message" size={24} />
        <h2 id="representative-title" className="card__title card__title--small">נציג התחבורה ענה</h2>
      </div>
      {answers.map((q) => (
        <p key={q.question_id} className="card__lead answers__text">
          <BidiText text={q.answer ?? ""} />
        </p>
      ))}
      <p className="card__note">היועץ ממשיך בשיחה לפי התשובה.</p>
    </section>
  );
}
