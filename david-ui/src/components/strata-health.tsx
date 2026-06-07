import type { Stratum } from "@/lib/api";
import Link from "next/link";

/** Adjudicated-item floor for Bayesian fit reliability. */
const MIN_FOR_FIT = 3;

/** Country code → flag emoji lookup. */
const FLAGS: Record<string, string> = {
  XK: "🇽🇰", AL: "🇦🇱", MK: "🇲🇰", RS: "🇷🇸", BA: "🇧🇦", ME: "🇲🇪",
  NL: "🇳🇱", IE: "🇮🇪", SE: "🇸🇪", TR: "🇹🇷", HR: "🇭🇷", SI: "🇸🇮",
  HU: "🇭🇺", PL: "🇵🇱", RO: "🇷🇴", BG: "🇧🇬", DE: "🇩🇪", FR: "🇫🇷",
  GB: "🇬🇧", EU: "🇪🇺",
};

type HealthLevel = "ready" | "partial" | "uncoded" | "empty";

function healthOf(s: Stratum): HealthLevel {
  if (s.n_adjudicated >= MIN_FOR_FIT) return "ready";
  if (s.n_adjudicated > 0)           return "partial";
  if (s.n_evidence > 0)              return "uncoded";
  return "empty";
}

const HEALTH_STYLES: Record<HealthLevel, { badge: string; bar: string; label: string }> = {
  ready:   { badge: "bg-emerald-950 text-emerald-400 border-emerald-800/50", bar: "bg-emerald-500", label: "Fit-ready" },
  partial: { badge: "bg-amber-950   text-amber-400   border-amber-800/50",   bar: "bg-amber-500",   label: "Partial" },
  uncoded: { badge: "bg-slate-800   text-slate-400   border-slate-700",       bar: "bg-slate-600",   label: "Uncoded" },
  empty:   { badge: "bg-red-950/40  text-red-500     border-red-900/50",      bar: "bg-red-800",     label: "Empty" },
};

interface Props {
  strata: Stratum[];
}

/** Server component: visual health grid for all strata. */
export function StrataHealth({ strata }: Props) {
  if (strata.length === 0) {
    return (
      <p className="text-sm text-slate-500 italic">
        No strata yet — run the pipeline to create evidence records.
      </p>
    );
  }

  const readyCount   = strata.filter((s) => healthOf(s) === "ready").length;
  const partialCount = strata.filter((s) => healthOf(s) === "partial").length;
  const emptyCount   = strata.filter((s) => healthOf(s) === "empty").length;
  const maxEvidence  = Math.max(...strata.map((s) => s.n_evidence), 1);

  return (
    <div>
      {/* Summary legend */}
      <div className="mb-3 flex flex-wrap gap-3 text-xs text-slate-500">
        {[
          { lvl: "ready"   as HealthLevel, count: readyCount,   note: `≥${MIN_FOR_FIT} adjudicated` },
          { lvl: "partial" as HealthLevel, count: partialCount, note: "some adjudicated" },
          { lvl: "empty"   as HealthLevel, count: emptyCount,   note: "no evidence" },
        ].map(({ lvl, count, note }) => (
          <span key={lvl} className="flex items-center gap-1.5">
            <span className={`inline-block h-2 w-2 rounded-full ${HEALTH_STYLES[lvl].bar}`} />
            <span>{HEALTH_STYLES[lvl].label}</span>
            <span className="font-semibold text-slate-400">{count}</span>
            <span className="text-slate-600">({note})</span>
          </span>
        ))}
      </div>

      {/* Strata cards grid */}
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {strata.map((s) => {
          const level = healthOf(s);
          const styles = HEALTH_STYLES[level];
          const barWidth = Math.max(4, Math.round((s.n_evidence / maxEvidence) * 100));
          const adjBarWidth = s.n_evidence > 0
            ? Math.round((s.n_adjudicated / s.n_evidence) * 100)
            : 0;
          const flag = FLAGS[s.country.toUpperCase()] ?? "🌐";
          const policy = s.policy.replace(/_/g, " ");

          return (
            <Link
              key={s.stratum_id}
              href={`/evidence?stratum_id=${s.stratum_id}`}
              className="group rounded-lg border border-slate-800 bg-slate-900/50
                         px-3 py-2.5 hover:border-slate-700 hover:bg-slate-900
                         transition-colors block"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span>{flag}</span>
                    <span className="text-xs font-semibold text-slate-300 truncate">
                      {s.country.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-slate-600">·</span>
                    <span className="text-[10px] text-slate-500 truncate capitalize">{policy}</span>
                  </div>
                  <p className="mt-0.5 text-[10px] font-mono text-slate-600">{s.stratum_id}</p>
                </div>
                <span
                  className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium ${styles.badge}`}
                >
                  {styles.label}
                </span>
              </div>

              {/* Evidence bar */}
              <div className="mt-2 space-y-1">
                <div className="flex items-center justify-between text-[10px] text-slate-600">
                  <span>Evidence</span>
                  <span className="text-slate-500">{s.n_evidence}</span>
                </div>
                <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full ${styles.bar} opacity-60`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>

                {/* Adjudicated bar */}
                <div className="flex items-center justify-between text-[10px] text-slate-600">
                  <span>Adjudicated</span>
                  <span className="text-slate-500">{s.n_adjudicated}</span>
                </div>
                <div className="h-1 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full rounded-full ${styles.bar}`}
                    style={{ width: `${adjBarWidth}%` }}
                  />
                </div>
              </div>

              {/* Fit hint */}
              {level !== "ready" && s.n_evidence > 0 && (
                <p className="mt-2 text-[10px] text-slate-600">
                  {level === "uncoded"
                    ? "Run pipeline to code these items"
                    : `Need ${MIN_FOR_FIT - s.n_adjudicated} more adjudicated item${
                        MIN_FOR_FIT - s.n_adjudicated === 1 ? "" : "s"
                      } for fit`}
                </p>
              )}
              {level === "empty" && (
                <p className="mt-2 text-[10px] text-slate-600">No articles found yet for this stratum</p>
              )}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
