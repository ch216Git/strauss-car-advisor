import type { Car, CarChoice } from "../api/types";

/** The employee's choice is sent as their own words, through the chat, like any other answer. */
export function choiceMessage(car: Car): string {
  return `התכוונתי ל-${car.manufacturer} ${car.commercial_name} (${car.registration_year})`;
}

export function CarChoiceCard({ choice, onChoose, disabled }: {
  choice: CarChoice;
  onChoose: (text: string) => void;
  disabled: boolean;
}) {
  return (
    <section className="card" aria-labelledby="choice-title">
      <h2 id="choice-title" className="card__title card__title--small">
        {choice.target === "current" ? "לאיזה רכב התכוונת – הרכב שלך היום?" : "לאיזה רכב התכוונת?"}
      </h2>
      <ul className="choices">
        {choice.options.map((car) => (
          <li key={car.car_id}>
            <button type="button" className="choice" disabled={disabled} onClick={() => onChoose(choiceMessage(car))}>
              <span className="choice__name">{car.commercial_name}</span>
              <span className="choice__meta">
                {car.manufacturer} · {car.registration_year}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
