import { api } from "@/lib/api";
import type { ForecastCell } from "@/lib/api";
import { CiChart, type CiRow } from "@/components/ci-chart";

export const dynamic = "force-dynamic";

const horizons = [3, 6, 9, 12];

interface Props {
  searchParams: Promise<{ horizon?: string; run_id?: string }>;
}

export default async function PredictPage({ searchParams }: Props) {
  const params  = await searchParams;
  const horizon = parseInt(params.horizon ?? "6", 10);
  const runId   = params.run_id;

  const cellsRes = await api.forecastCells(runId, horizon).catch(() => null);
  const cells    = cellsRes?.forecast_cells ?? [];
  const activeRunId = cellsRes?.run_id;

  // Build chart data
  const chartData: CiRow[] = cells.map((c: ForecastCell) => ({
    label: `S${c.series} / T${c.tactic}${c.stratum_id ? ` (${c.stratum_id})` : ""}`,
    p_active:  c.p_active  ?? 0,
    ci_80_lo:  c.ci_80_lo  ?? 0,
    ci_80_hi:  c.ci_80_hi  ?? 0,
    ci_95_lo:  c.ci_95_lo  ?? 0,
    ci_95_hi:  c.ci_95_hi  ?? 0,
    route:     c.forecast_route ?? "headline",
  }));

  const nConditional = cells.filter((c) => c.forecast_route === "headline").length;
  const nDominated   = cells.filter((c) => c.forecast_route === "horizon_prior_dominated").length;

  return (
    <div className="space-y-8 max-w-5xl">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Forecast Viewer</h1>
          {activeRunId && (
            <p className="mt-1 text-sm text-slate-400">
              Run <span className="font-mono">{activeRunId}</span>
            </p>
          )}
        </div>

        {/* Horizon selector */}
        <div className="flex gap-2">
          {horizons.map((h) => (
            <a
              key={h}
              href={`/predict?horizon=${h}${runId ? `&run_id=${runId}` : ""}`}
              className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                h === horizon
                  ? "bg-sky-800 text-sky-100"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              {h}m
            </a>
          ))}
        </div>
      </div>

      {/* Summary pills */}
      <div className="flex gap-3 flex-wrap">
        <span className="rounded-full border border-sky-700/40 bg-sky-950/30 px-3 py-1 text-xs text-sky-300">
          {nConditional} conditional forecast
        </span>
        <span className={`rounded-full border px-3 py-1 text-xs ${
          nDominated > 0
            ? "border-amber-700/40 bg-amber-950/20 text-amber-300"
            : "border-slate-800 bg-slate-900 text-slate-500"
        }`}>
          {nDominated} horizon prior-dominated
        </span>
        <span className="rounded-full border border-slate-800 bg-slate-900 px-3 py-1 text-xs text-slate-400">
          Horizon: {horizon} months
        </span>
      </div>

      {cells.length === 0 ? (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-10 text-center">
          <p className="text-slate-400">No forecast cells for horizon {horizon}m.</p>
          <p className="mt-2 text-sm text-slate-600">Run <code className="text-sky-500">david forecast --horizon {horizon}</code></p>
        </div>
      ) : (
        <>
          {/* CI chart */}
          <section>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
              p(active) with 80% / 95% credible intervals
            </h2>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="flex gap-4 mb-4 text-xs text-slate-500">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-1 w-6 rounded bg-sky-400" /> conditional forecast
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-1 w-6 rounded bg-amber-400" /> prior-dominated
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-3 w-1.5 rounded bg-sky-400 opacity-70" /> 80% CI
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-3 w-px rounded bg-sky-400 opacity-40" /> 95% CI
                </span>
              </div>
              <CiChart data={chartData} />
            </div>
          </section>

          {/* Cell table */}
          <section>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
              Cells ({cells.length})
            </h2>
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-slate-800 text-xs text-slate-500">
                  <tr>
                    {["Series", "Tactic", "Stratum", "p_active", "80% CI", "95% CI", "h* (mo)", "Route"].map((h) => (
                      <th key={h} className="px-4 py-2.5 text-left font-medium whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {cells.map((c, i) => (
                    <tr key={i} className="hover:bg-slate-800/30">
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-300">{c.series}</td>
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-300">{c.tactic}</td>
                      <td className="px-4 py-2.5 text-xs text-slate-400">{c.stratum_id || "—"}</td>
                      <td className="px-4 py-2.5 font-mono text-sm font-semibold text-white">
                        {c.p_active?.toFixed(3) ?? "—"}
                      </td>
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-400">
                        [{c.ci_80_lo?.toFixed(3)}, {c.ci_80_hi?.toFixed(3)}]
                      </td>
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-500">
                        [{c.ci_95_lo?.toFixed(3)}, {c.ci_95_hi?.toFixed(3)}]
                      </td>
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-400">{c.h_star_months ?? "—"}</td>
                      <td className="px-4 py-2.5">
                        <span className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                          c.forecast_route === "headline"
                            ? "bg-emerald-950/60 text-emerald-300"
                            : "bg-amber-950/60 text-amber-400"
                        }`}>
                          {c.forecast_route === "headline" ? "headline" : c.forecast_route.replace(/_/g, " ")}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
