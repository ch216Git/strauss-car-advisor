import type { Side } from "../api/types";
import { shekels } from "../lib/format";
import { Amount } from "./BidiText";

/** One side of the comparison ("היום" or "בחלופה"), labelled so before/after never depends on direction. */
export function ComparisonSide({ side, label, estimate, highlight = false }: {
  side: Side;
  label: "היום" | "בחלופה";
  estimate: boolean;
  highlight?: boolean;
}) {
  return (
    <div className={`side${highlight ? " side--alternative" : ""}`}>
      <span className="side__label">{label}</span>
      <span className="side__car">{side.kind === "waiver" ? "ויתור על הרכב" : side.car?.commercial_name}</span>
      {side.car && <span className="side__meta">{side.car.manufacturer}</span>}
      <span className="side__net-label">{estimate ? "נטו משוער" : "נטו"}</span>
      <Amount className="side__net">{shekels(side.net)}</Amount>
    </div>
  );
}
