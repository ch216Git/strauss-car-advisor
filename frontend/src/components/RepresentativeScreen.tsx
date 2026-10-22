// The transport representative's screen: open questions from every conversation, answered here.
// The answer goes to the server, which checks it against the question's topic; the employee's screen
// shows it, and the advisor continues from it. No authentication yet (hackathon scope).
// The queue is also kept in this browser (lib/repQueue), so it survives a server restart.

import { type FormEvent, useCallback, useEffect, useState } from "react";
import { getCatalogue, getQueue, sendAnswer } from "../api/client";
import type { Car, QueueQuestion } from "../api/types";
import { type ShownQuestion, cachedQueue, loadQueue, mergeQueue, questionKey } from "../lib/repQueue";
import { CompanyLogo } from "./CompanyLogo";
import { Icon } from "./Icon";

export const QUEUE_POLL_MS = 10000;

const TOPIC_LABEL: Record<string, string> = {
  alternative_participation: "השתתפות ברכב חלופי",
  current_car_not_in_catalogue: "רכב נוכחי שאינו בקטלוג",
  other: "שאלה כללית",
};

function carLabel(car: Car): string {
  return `${car.manufacturer} ${car.commercial_name} (${car.registration_year})`;
}

export function RepresentativeScreen() {
  const [questions, setQuestions] = useState<ShownQuestion[] | null>(() => {
    const cached = cachedQueue();
    return cached.length > 0 ? cached : null; // shown at once, offline until the server confirms them
  });
  const [cars, setCars] = useState<Car[]>([]);
  const [failed, setFailed] = useState(false);

  const load = useCallback(async () => {
    const result = await getQueue();
    if (result.ok) {
      setQuestions(mergeQueue(result.data.questions, loadQueue()));
      setFailed(false);
    } else {
      setFailed(true);
    }
  }, []);

  useEffect(() => {
    void load();
    void getCatalogue().then((result) => result.ok && setCars(result.data.cars));
    const timer = setInterval(() => void load(), QUEUE_POLL_MS);
    return () => clearInterval(timer);
  }, [load]);

  const open = questions?.filter((q) => q.status === "open") ?? [];
  const answered = questions?.filter((q) => q.status === "answered") ?? [];

  return (
    <div className="app" dir="rtl" lang="he">
      <header className="header">
        <div className="header__brand">
          <CompanyLogo />
          <span className="header__logo">
            <Icon name="message" />
          </span>
          <h1 className="header__title">מסך נציג</h1>
          <span className="header__subtitle">שאלות מהיועץ שממתינות לתשובה</span>
        </div>
        <div className="header__actions">
          <button type="button" className="button button--ghost" onClick={() => void load()}>
            רענון
          </button>
          <a className="button button--ghost" href="#/">למסך העובד</a>
        </div>
      </header>
      <main className="rep">
        {failed && <p className="notice" role="alert">אין חיבור לשרת כרגע. ננסה שוב בעוד רגע.</p>}
        {questions === null && !failed && <p className="muted">טוען…</p>}
        {questions !== null && (
          <>
            <section aria-labelledby="open-title" className="rep__section">
              <h2 id="open-title" className="panel__title">ממתינות לתשובה ({open.length})</h2>
              {open.length === 0 && <p className="muted">אין שאלות פתוחות.</p>}
              {open.map((q) => (
                <QuestionCard key={questionKey(q)} question={q} cars={cars} onAnswered={load} />
              ))}
            </section>
            {answered.length > 0 && (
              <section aria-labelledby="answered-title" className="rep__section">
                <h2 id="answered-title" className="panel__title">נענו ({answered.length})</h2>
                {answered.map((q) => (
                  <QuestionCard key={questionKey(q)} question={q} cars={cars} onAnswered={load} />
                ))}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function QuestionCard({ question, cars, onAnswered }: {
  question: ShownQuestion;
  cars: Car[];
  onAnswered: () => Promise<void>;
}) {
  const titleId = `q-${question.session_id}-${question.question_id}`;
  return (
    <article className="card rep__question" aria-labelledby={titleId}>
      <div className="rep__meta">
        <span className="tag">{TOPIC_LABEL[question.topic] ?? TOPIC_LABEL.other}</span>
        <span className="muted">שיחה {question.session_id.slice(0, 6)} · {question.question_id}</span>
      </div>
      <h3 id={titleId} className="card__title card__title--small">{question.question}</h3>
      {question.car && <p className="card__note">רכב: {carLabel(question.car)}</p>}
      {question.offline && (
        <p className="notice" role="note">
          {question.status === "open"
            ? "השיחה הזו לא מחוברת כרגע לשרת. היא תחזור כשהעובד/ת יפתח/תפתח אותה, ואז אפשר יהיה לענות."
            : "השיחה הזו לא מחוברת כרגע לשרת."}
        </p>
      )}
      {question.status === "answered" ? (
        <div className="rep__answer">
          {question.participation !== undefined && <p>השתתפות: {question.participation.toLocaleString("he-IL")} ₪ בחודש</p>}
          {question.answer && <p>{question.answer}</p>}
        </div>
      ) : question.offline ? null : (
        <AnswerForm question={question} cars={cars} onAnswered={onAnswered} />
      )}
    </article>
  );
}

function AnswerForm({ question, cars, onAnswered }: {
  question: QueueQuestion;
  cars: Car[];
  onAnswered: () => Promise<void>;
}) {
  const [answer, setAnswer] = useState("");
  const [participation, setParticipation] = useState("");
  const [carId, setCarId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const id = `${question.session_id}-${question.question_id}`;
  const topic = question.topic;
  const needsAmount = topic === "alternative_participation" || (topic === "current_car_not_in_catalogue" && carId !== "");
  const ready = topic === "alternative_participation"
    ? participation !== ""
    : topic === "current_car_not_in_catalogue"
      ? carId !== "" ? participation !== "" : answer.trim() !== ""
      : answer.trim() !== "";

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!ready || sending) return;
    setSending(true);
    setError(null);
    const result = await sendAnswer({
      session_id: question.session_id,
      question_id: question.question_id,
      answer: answer.trim(),
      ...(needsAmount ? { participation: Number(participation) } : {}),
      ...(topic === "current_car_not_in_catalogue" && carId ? { car_id: carId } : {}),
    });
    setSending(false);
    if (!result.ok) {
      setError(result.message ?? "התשובה לא נשלחה. אפשר לנסות שוב.");
      if (result.status === 409 && result.code === "already_answered") void onAnswered();
      return;
    }
    await onAnswered();
  }

  return (
    <form className="rep__form" onSubmit={submit}>
      {topic === "current_car_not_in_catalogue" && (
        <label className="rep__field">
          <span>הרכב המקביל בקטלוג (אם יש)</span>
          <select value={carId} onChange={(e) => setCarId(e.target.value)}>
            <option value="">— בלי רכב מקביל —</option>
            {cars.map((car) => (
              <option key={car.car_id} value={car.car_id}>{carLabel(car)}</option>
            ))}
          </select>
        </label>
      )}
      {needsAmount && (
        <label className="rep__field">
          <span>{topic === "alternative_participation" ? "השתתפות חודשית מאושרת (₪)" : "ההשתתפות של העובד/ת היום (₪ בחודש)"}</span>
          <input
            type="number" inputMode="numeric" min={0} max={100000} step={1} required
            value={participation} onChange={(e) => setParticipation(e.target.value)}
          />
        </label>
      )}
      <label className="rep__field">
        <span>{topic === "other" || (topic === "current_car_not_in_catalogue" && !carId) ? "התשובה לעובד/ת" : "הערה לעובד/ת (לא חובה)"}</span>
        <textarea id={`answer-${id}`} rows={3} maxLength={1000} value={answer} onChange={(e) => setAnswer(e.target.value)} />
      </label>
      {error && <p className="notice" role="alert">{error}</p>}
      <button type="submit" className="button" disabled={!ready || sending}>
        {sending ? "שולח…" : "שליחת תשובה"}
      </button>
    </form>
  );
}
