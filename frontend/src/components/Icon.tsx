// Inline stroke icons, decorative only (aria-hidden); the text next to them carries the meaning.

const PATHS = {
  car: "M5 16h14M6.5 16l1.4-5.2A2 2 0 0 1 9.8 9.3h4.4a2 2 0 0 1 1.9 1.5L17.5 16M4 16v2.5h2.5V16M17.5 16v2.5H20V16",
  plus: "M12 5v14M5 12h14",
  send: "M20 12H5M11 6l-6 6 6 6",
  arrowLeft: "M19 12H5M11 6l-6 6 6 6",
  shield: "M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6zM12 8v4M12 15h.01",
  blocked: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM8.5 8.5l7 7",
  check: "M5 12l4 4 10-10",
  info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 8v5M12 16h.01",
  bolt: "M13 3L5 13h6l-1 8 8-10h-6z",
  message: "M4 6h16v10H8l-4 4z",
  compare: "M4 7h11M4 7l3-3M4 7l3 3M20 17H9M20 17l-3-3M20 17l-3 3",
  history: "M3 12a9 9 0 1 0 3-6.7M3 4v3.5h3.5M12 7v5l3.5 2",
  trash: "M5 7h14M10 7V5h4v2M6 7l1 13h10l1-13",
  close: "M6 6l12 12M18 6L6 18",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 20, className }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.9"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

export function Spinner({ size = 18 }: { size?: number }) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" className="spinner">
      <path d="M12 3a9 9 0 1 0 9 9" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
    </svg>
  );
}
