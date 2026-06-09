"use client";

import { useState } from "react";

interface Props {
  /** Tooltip title in Albanian */
  title: string;
  /** Body text in Albanian — mathematically rigorous explanation */
  body: string;
  /** Optional formula / technical footnote line */
  formula?: string;
  /** Where the tooltip card opens relative to the dot */
  side?: "right" | "left" | "top" | "bottom";
}

/**
 * Absolute-positioned blue pulsing indicator used by "Albanian Explanation Mode".
 * Place inside a `relative` container; it pins to the top-right corner by default.
 * Hovering reveals a rigorous Albanian tooltip card.
 */
const CARD_POS: Record<NonNullable<Props["side"]>, string> = {
  left:   "right-6 top-1/2 -translate-y-1/2",
  top:    "bottom-6 left-1/2 -translate-x-1/2",
  bottom: "top-6 left-1/2 -translate-x-1/2",
  right:  "left-6 top-1/2 -translate-y-1/2",
};

export function ExplainDot({ title, body, formula, side = "right" }: Props) {
  const [open, setOpen] = useState(false);
  const cardPos = CARD_POS[side];

  return (
    <button
      type="button"
      aria-label={title}
      aria-expanded={open}
      className="absolute right-3 top-3 z-30 cursor-help bg-transparent p-0"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
    >
      <span className="relative flex h-3 w-3">
        <span className="explain-ring absolute inline-flex h-full w-full rounded-full" />
        <span className="relative inline-flex h-3 w-3 rounded-full bg-sky-400 ring-2 ring-sky-300/40" />
      </span>

      {open && (
        <span
          role="tooltip"
          className={`absolute ${cardPos} w-72 rounded-xl border border-sky-500/40 bg-[#0b1222]
                     p-4 text-left shadow-2xl shadow-sky-900/40 ring-1 ring-sky-500/10`}
        >
          <span className="mb-1 flex items-center gap-2">
            <span className="text-base leading-none">🇦🇱</span>
            <span className="text-sm font-bold text-sky-300">{title}</span>
          </span>
          <span className="block text-xs leading-relaxed text-slate-300">{body}</span>
          {formula && (
            <span className="mt-2 block rounded-md bg-slate-900/80 px-2 py-1 font-mono text-[11px] text-cyan-300">
              {formula}
            </span>
          )}
        </span>
      )}
    </button>
  );
}
