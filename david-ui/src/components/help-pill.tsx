"use client";

import { useState } from "react";

/**
 * Inline help tooltip — hover or click the ? badge to reveal the hint text.
 * Usage:  <HelpPill text="What this means in plain English." />
 */
export function HelpPill({ text }: { text: string }) {
  const [open, setOpen] = useState(false);

  return (
    <span className="relative inline-block">
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        className="ml-1.5 inline-flex h-4 w-4 items-center justify-center rounded-full
                   bg-slate-700 text-[10px] font-bold text-slate-400
                   hover:bg-slate-600 hover:text-slate-200 transition-colors cursor-help"
        aria-label="More info"
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className="pointer-events-none absolute bottom-6 left-1/2 z-50 w-60
                     -translate-x-1/2 rounded-lg border border-slate-600
                     bg-slate-800 px-3 py-2 text-xs leading-relaxed text-slate-200
                     shadow-xl"
        >
          {text}
          {/* caret */}
          <span className="absolute -bottom-1.5 left-1/2 h-0 w-0
                           -translate-x-1/2 border-x-4 border-x-transparent
                           border-t-[6px] border-t-slate-800" />
        </span>
      )}
    </span>
  );
}
