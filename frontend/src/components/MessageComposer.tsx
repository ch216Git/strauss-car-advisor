import { type FormEvent, useEffect, useRef } from "react";
import { Icon } from "./Icon";

const MAX_LENGTH = 2000; // the API's limit

export function MessageComposer({
  value,
  onChange,
  onSend,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: (text: string) => void;
  disabled: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!disabled) input.current?.focus();
  }, [disabled]);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!disabled && value.trim()) onSend(value);
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="advisor-message" className="sr-only">
        הודעה ליועץ
      </label>
      <input
        ref={input}
        id="advisor-message"
        className="composer__input"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="למשל: אפשר לקבל קרניבל?"
        maxLength={MAX_LENGTH}
        autoComplete="off"
        disabled={disabled}
      />
      <button type="submit" className="composer__send" aria-label="שליחה" disabled={disabled || !value.trim()}>
        <Icon name="send" />
      </button>
    </form>
  );
}
