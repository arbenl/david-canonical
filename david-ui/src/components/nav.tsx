"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type NavLink = { href: string; label: string; icon: string };

// Primary workspace tabs (per unified premium spec)
const primary: NavLink[] = [
  { href: "/roadmap",  label: "Arkitektura & Progresi", icon: "RM" },
  { href: "/router",   label: "Forecast Router",        icon: "RT" },
  { href: "/",         label: "Command Center",         icon: "CC" },
  { href: "/evidence", label: "Evidence Pipeline",      icon: "EV" },
  { href: "/engine",   label: "AI Engine",              icon: "EN" },
];

// Legacy / analyst tools — kept reachable, secondary group
const legacy: NavLink[] = [
  { href: "/predict",  label: "Forecast (classic)", icon: "FC" },
  { href: "/simulate", label: "Simulation",         icon: "SM" },
  { href: "/validate", label: "Validation",         icon: "VA" },
];

function NavItem({ link, active }: Readonly<{ link: NavLink; active: boolean }>) {
  return (
    <Link
      href={link.href}
      className={`flex shrink-0 items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors md:flex-shrink ${
        active
          ? "bg-sky-950 text-sky-300 font-medium ring-1 ring-sky-500/30"
          : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
      }`}
    >
      <span className="numeric flex h-6 w-7 items-center justify-center rounded border border-slate-700 bg-slate-950 text-[10px] leading-none text-slate-400">
        {link.icon}
      </span>
      <span className="whitespace-nowrap md:whitespace-normal">{link.label}</span>
    </Link>
  );
}

export function Nav() {
  const path = usePathname();
  const isActive = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));

  return (
    <aside className="z-20 flex w-full flex-col border-b border-slate-800 bg-[#070b19] md:fixed md:inset-y-0 md:left-0 md:w-56 md:border-b-0 md:border-r">
      <div className="border-b border-slate-800 px-5 py-4 md:py-6">
        <p className="numeric text-xs font-bold uppercase text-sky-400">DAVID / M0.1</p>
        <p className="mt-1 text-[11px] text-slate-500">Ledger Console</p>
      </div>

      <nav className="flex gap-2 overflow-x-auto px-3 py-3 md:block md:flex-1 md:space-y-0.5 md:overflow-y-auto md:py-4">
        {primary.map((link) => (
          <NavItem key={link.href} link={link} active={isActive(link.href)} />
        ))}

        <p className="hidden px-3 pb-2 pt-5 text-[10px] font-semibold uppercase tracking-wider text-slate-600 md:block">
          Analyst tools
        </p>
        {legacy.map((link) => (
          <span key={link.href} className="hidden md:block">
            <NavItem link={link} active={isActive(link.href)} />
          </span>
        ))}
      </nav>

      <div className="hidden border-t border-slate-800 px-5 py-4 md:block">
        <p className="text-[10px] text-slate-600">
          FastAPI → <span className="text-slate-500">:8080</span>
        </p>
      </div>
    </aside>
  );
}
