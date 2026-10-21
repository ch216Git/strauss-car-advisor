import { type FormEvent, useEffect, useRef } from "react";
import { ACCEPTED_FILES } from "../hooks/useDocumentUpload";
import { Icon } from "./Icon";

const MAX_LENGTH = 2000; // the API's limit

export function MessageComposer({
  value,
  onChange,
  onSend,
  disabled,
  onAttach,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: (text: string) => void;
  disabled: boolean;
  onAttach?: (file: File) => void; // a file with the employee's details (reviewed before anything is saved)
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
      {onAttach && (
        <label className={`composer__attach${disabled ? " composer__attach--disabled" : ""}`} title="העלאת קובץ עם פרטים">
          <span className="sr-only">העלאת קובץ עם פרטים</span>
          <Icon name="attach" />
          <input
            type="file"
            className="sr-only"
            accept={ACCEPTED_FILES}
            disabled={disabled}
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = ""; // the same file can be chosen again
              if (file) onAttach(file);
            }}
          />
        </label>
      )}
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
