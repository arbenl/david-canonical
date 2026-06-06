"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/",         label: "Command Center", icon: "⬡" },
  { href: "/evidence", label: "Evidence",        icon: "◈" },
  { href: "/engine",   label: "AI Engine",       icon: "⬟" },
  { href: "/simulate", label: "Simulation",      icon: "⊛" },
  { href: "/predict",  label: "Forecast",        icon: "◎" },
  { href: "/validate", label: "Validation",      icon: "✦" },
];

export function Nav() {
  const path = usePathname();
  return (
    <aside className="fixed inset-y-0 left-0 w-56 border-r border-slate-800 bg-slate-950 flex flex-col z-20">
      <div className="px-5 py-6 border-b border-slate-800">
        <p className="text-xs font-bold tracking-[0.25em] uppercase text-sky-400">DAVID / M0.1</p>
        <p className="mt-1 text-[11px] text-slate-500">Prediction Engine</p>
      </div>
      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-0.5">
        {links.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-sky-950 text-sky-300 font-medium"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <span className="text-base leading-none">{l.icon}</span>
              {l.label}
            </Link>
          );
        })}
      </nav>
      <div className="px-5 py-4 border-t border-slate-800">
        <p className="text-[10px] text-slate-600">
          FastAPI → <span className="text-slate-500">:8080</span>
        </p>
      </div>
    </aside>
  );
}
