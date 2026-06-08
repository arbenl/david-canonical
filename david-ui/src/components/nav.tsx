"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type NavLink = { href: string; label: string; icon: string };

// Primary workspace tabs (per unified premium spec)
const primary: NavLink[] = [
  { href: "/roadmap",  label: "Arkitektura & Progresi", icon: "🗺️" },
  { href: "/router",   label: "Forecast Router",        icon: "📈" },
  { href: "/",         label: "Command Center",         icon: "⬡" },
  { href: "/evidence", label: "Evidence Pipeline",      icon: "◈" },
  { href: "/engine",   label: "AI Engine",              icon: "⬟" },
];

// Legacy / analyst tools — kept reachable, secondary group
const legacy: NavLink[] = [
  { href: "/predict",  label: "Forecast (classic)", icon: "◎" },
  { href: "/simulate", label: "Simulation",         icon: "⊛" },
  { href: "/validate", label: "Validation",         icon: "✦" },
];

function NavItem({ link, active }: { link: NavLink; active: boolean }) {
  return (
    <Link
      href={link.href}
      className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
        active
          ? "bg-sky-950 text-sky-300 font-medium ring-1 ring-sky-500/30"
          : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
      }`}
    >
      <span className="text-base leading-none">{link.icon}</span>
      {link.label}
    </Link>
  );
}

export function Nav() {
  const path = usePathname();
  const isActive = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));

  return (
    <aside className="fixed inset-y-0 left-0 w-56 border-r border-slate-800 bg-[#070b19] flex flex-col z-20">
      <div className="px-5 py-6 border-b border-slate-800">
        <p className="text-xs font-bold tracking-[0.25em] uppercase text-sky-400 neon-cyan">DAVID / M0.1</p>
        <p className="mt-1 text-[11px] text-slate-500">Prediction Engine</p>
      </div>

      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-0.5">
        {primary.map((link) => (
          <NavItem key={link.href} link={link} active={isActive(link.href)} />
        ))}

        <p className="px-3 pt-5 pb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-600">
          Analyst tools
        </p>
        {legacy.map((link) => (
          <NavItem key={link.href} link={link} active={isActive(link.href)} />
        ))}
      </nav>

      <div className="px-5 py-4 border-t border-slate-800">
        <p className="text-[10px] text-slate-600">
          FastAPI → <span className="text-slate-500">:8080</span>
        </p>
      </div>
    </aside>
  );
}
