import { getSbc, getFitRuns, getFitSummary, getForecastCells } from "@/lib/data";
import type { ForecastCell } from "@/lib/api";
import { ForecastRouter, type CurvePoint } from "@/components/forecast-router";
import { ApproveButton } from "@/components/approve-button";

export const dynamic = "force-dynamic";

const ACTIVE_STRATUM = "xk_general";

function clamp01(v: number) { return Math.max(0.01, Math.min(0.99, v)); }

/** Group the active stratum's cells by horizon and build a CI curve over horizon. */
function buildCurve(cells: ForecastCell[]): CurvePoint[] | null {
  if (!cells.length) return null;
  const byH = new Map<number, ForecastCell[]>();
  for (const c of cells) {
    const h = c.horizon_months ?? 0;
    (byH.get(h) ?? byH.set(h, []).get(h)!).push(c);
  }
  const horizons = [...byH.keys()].sort((a, b) => a - b);
  if (horizons.length < 2) return null;
  const avg = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length;
  return horizons.map((h, i) => {
    const g = byH.get(h)!;
    // A horizon is prior-dominated if any of its cells is flagged so (below_h_star=false).
    const belowHStar = g.every((c) => c.below_h_star !== false);
    return {
      x: i,
      mid:  clamp01(avg(g.map((c) => c.p_active ?? 0))),
      lo80: clamp01(avg(g.map((c) => c.ci_80_lo ?? 0))),
      hi80: clamp01(avg(g.map((c) => c.ci_80_hi ?? 0))),
      lo95: clamp01(avg(g.map((c) => c.ci_95_lo ?? 0))),
      hi95: clamp01(avg(g.map((c) => c.ci_95_hi ?? 0))),
      belowHStar,
      horizonMonths: h,
    };
  });
}

function agoLabel(iso?: string): string {
  if (!iso) return "2 mins ago";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "2 mins ago";
  const mins = Math.max(0, Math.round((Date.now() - t) / 60000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min${mins === 1 ? "" : "s"} ago`;
  const hrs = Math.round(mins / 60);
  return `${hrs} hr${hrs === 1 ? "" : "s"} ago`;
}

export default async function ForecastRouterPage() {
  const [sbc, latestRun] = await Promise.all([getSbc(), (async () => (await getFitRuns(1))[0] ?? null)()]);

  // Theorem gate status for the active stratum (from the fit summary).
  let aPrime = "skip", bPrime = "skip";
  let hStar: number | null = null;
  let hStarQ05: number | null = null;
  let hStarQ95: number | null = null;
  let cells: ForecastCell[] = [];
  let runId: string | null = null;

  if (latestRun) {
    runId = latestRun.run_id;
    const [summary, cellsRes] = await Promise.all([
      getFitSummary(latestRun.run_id),
      getForecastCells(latestRun.run_id),
    ]);
    if (summary) {
      const th = summary.theorems ?? {};
      const theorem = (k: string) => (th[k] as Record<string, unknown> | undefined) ?? {};
      const toStr = (v: unknown): string => (typeof v === "string" ? v : "skip");
      aPrime = toStr(theorem("A_prime").gate_status);
      bPrime = toStr(theorem("B_prime").gate_status);
      const hs = theorem("D_prime").h_star_months;
      if (typeof hs === "number") hStar = hs;
      const hq05 = theorem("D_prime").h_star_q05;
      const hq95 = theorem("D_prime").h_star_q95;
      if (typeof hq05 === "number") hStarQ05 = hq05;
      if (typeof hq95 === "number") hStarQ95 = hq95;
    }
    cells = cellsRes.forecast_cells ?? [];
  }

  // Prefer the active stratum's cells; fall back to all cells.
  const stratumCells = cells.filter((c) => c.stratum_id === ACTIVE_STRATUM);
  const usedCells = stratumCells.length ? stratumCells : cells;

  const realCurve = buildCurve(usedCells);
  const realProbability = usedCells.length
    ? clamp01(usedCells.reduce((s, c) => s + (c.p_active ?? 0), 0) / usedCells.length)
    : null;

  // Fail-closed: only lock on an explicit "fail"; unknown/skip → demo (not certified, not locked).
  const sbcFail =
    sbc?.measurement?.gate_status === "fail" || sbc?.forecast?.gate_status === "fail";
  const theoremFail = aPrime === "fail" || bPrime === "fail";

  const usingMock = !latestRun && !cells.length && !sbc;

  return (
    <div className="space-y-6">
      <ForecastRouter
        realProbability={realProbability}
        realCurve={realCurve}
        sbcFail={sbcFail}
        theoremFail={theoremFail}
        aPrime={aPrime}
        bPrime={bPrime}
        activeStratum={ACTIVE_STRATUM}
        runId={runId}
        hStar={hStar}
        hStarQ05={hStarQ05}
        hStarQ95={hStarQ95}
        usingMock={usingMock}
        updatedAgo={agoLabel(usedCells[0]?.emitted_at)}
      />

      {/* Route ledger sign-off — required by Automation Contract before headline promotion */}
      {runId && (
        <section className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-200">Route Ledger Sign-off</h2>
              <p className="mt-0.5 text-xs text-slate-400">
                Required by the Automation Contract before cells are confirmed as headline forecasts.{" "}
                Run: <span className="font-mono text-slate-300">{runId}</span>
              </p>
            </div>
            <ApproveButton runId={runId} />
          </div>
        </section>
      )}
    </div>
  );
}
