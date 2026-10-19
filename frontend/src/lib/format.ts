// Display formatting only: amounts arrive from the API already calculated and rounded.

const NUMBER = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** 11032 -> "₪11,032" (shown inside an LTR run). */
export function shekels(amount: number): string {
  return `₪${NUMBER.format(Math.abs(amount))}`;
}

/** 661 -> "+₪661", -86 -> "−₪86", 0 -> "₪0". The sign is the API value's own sign. */
export function signedShekels(amount: number): string {
  if (amount > 0) return `+${shekels(amount)}`;
  if (amount < 0) return `−${shekels(amount)}`;
  return shekels(0);
}

export type TextPart = { text: string; ltr: boolean };

// A "before → after" number run, e.g. "4,070 → 3,130". In an RTL paragraph its arrow would point the
// wrong way, so it is shown as an isolated LTR run. The text itself is never changed.
const BEFORE_AFTER = /(\d[\d,.]*\s*→\s*\d[\d,.]*)/;

export function splitBidi(text: string): TextPart[] {
  return text
    .split(BEFORE_AFTER)
    .filter((part) => part !== "")
    .map((part) => ({ text: part, ltr: BEFORE_AFTER.test(part) }));
}
