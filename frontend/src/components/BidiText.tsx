import { splitBidi } from "../lib/format";

/** The backend's text exactly as written; "before → after" number runs are shown left to right. */
export function BidiText({ text }: { text: string }) {
  return (
    <>
      {splitBidi(text).map((part, index) =>
        part.ltr ? (
          <bdi key={index} dir="ltr" className="ltr-run">
            {part.text}
          </bdi>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </>
  );
}

/** An amount shown as an LTR island inside RTL text, so "+₪661" never flips. */
export function Amount({ children, className }: { children: string; className?: string }) {
  return (
    <bdi dir="ltr" className={className}>
      {children}
    </bdi>
  );
}
