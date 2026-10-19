// The frontend sources are ours: nothing injected by a network filter, no external runtime scripts.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import { splitBidi } from "../lib/format";

const ROOT = join(__dirname, "..", "..");
const SKIP = new Set(["node_modules", "dist", ".vite", "__tests__"]);

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    if (SKIP.has(name)) return [];
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}

const SOURCES = files(ROOT).filter((path) => /\.(tsx?|jsx?|css|html|json)$/.test(path) && !path.endsWith("package-lock.json"));

describe("frontend sources", () => {
  it("contain no NetFree or browser-injected content", () => {
    expect(SOURCES.length).toBeGreaterThan(20);
    const offenders = SOURCES.filter((path) => /netfree|injection by|injection-script/i.test(readFileSync(path, "utf8")));
    expect(offenders.map((path) => relative(ROOT, path))).toEqual([]);
  });

  it("load no external scripts", () => {
    const html = readFileSync(join(ROOT, "index.html"), "utf8");
    const scripts = [...html.matchAll(/<script\b[^>]*>/gi)].map((match) => match[0]);
    expect(scripts).toEqual(['<script type="module" src="/src/main.tsx">']);
    const external = SOURCES.filter((path) => /<script[^>]+src=["']?(https?:)?\/\//i.test(readFileSync(path, "utf8")));
    expect(external).toEqual([]);
  });

  it("index.html is Hebrew right to left", () => {
    expect(readFileSync(join(ROOT, "index.html"), "utf8")).toMatch(/<html lang="he" dir="rtl">/);
  });
});

describe("splitBidi", () => {
  it("isolates before → after runs and keeps the text", () => {
    const text = "מה משתנה: שווי שימוש 4,070 → 3,130 ₪ · +250 ₪ נטו";
    const parts = splitBidi(text);
    expect(parts.map((p) => p.text).join("")).toBe(text);
    expect(parts.filter((p) => p.ltr).map((p) => p.text)).toEqual(["4,070 → 3,130"]);
  });

  it("leaves text without such a run alone", () => {
    expect(splitBidi("שלום 300 ₪")).toEqual([{ text: "שלום 300 ₪", ltr: false }]);
  });
});
