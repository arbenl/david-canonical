import { getSbc, getFitRuns, getFitSummary, getForecastCells, getRouteLedger } from "@/lib/data";
import type { ForecastCell } from "@/lib/api";
import { ForecastRouter, type CurvePoint } from "@/components/forecast-router";
import { ApproveButton } from "@/components/approve-button";

export const dynamic = "force-dynamic";

const ACTIVE_STRATUM = "xk_general";

function clamp01(v: number) { return Math.max(0.01, Math.min(0.99, v)); }

type RawLedgerCell = Partial<ForecastCell> & {
  cell?: { stratum_id?: string; series?: number; tactic?: number };
  credible_interval_80?: [number, number];
  credible_interval_95?: [number, number];
  horizon_validity?: {
    h_star_months?: number;
    below_h_star?: boolean;
    forecast_route?: ForecastCell["forecast_route"];
  };
};

function finite(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function text(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

function theoremNumber(
  summary: Awaited<ReturnType<typeof getFitSummary>>,
  theoremName: string,
  fieldName: string,
): number | null {
  const block = summary?.theorems?.[theoremName] as Record<string, unknown> | undefined;
  return finite(block?.[fieldName]);
}

function normalizeLedgerCell(raw: RawLedgerCell): ForecastCell | null {
  const series = finite(raw.series) ?? finite(raw.cell?.series);
  const tactic = finite(raw.tactic) ?? finite(raw.cell?.tactic);
  const horizon = finite(raw.horizon_months);
  const pActive = finite(raw.p_active);
  if (series == null || tactic == null || horizon == null || pActive == null) return null;

  const ci80 = raw.credible_interval_80;
  const ci95 = raw.credible_interval_95;
  const route = text(raw.forecast_route) ?? text(raw.horizon_validity?.forecast_route) ?? "withhold";
  const hStar = finite(raw.h_star_months) ?? finite(raw.horizon_validity?.h_star_months) ?? horizon;
  const belowHStar =
    typeof raw.below_h_star === "boolean"
      ? raw.below_h_star
      : typeof raw.horizon_validity?.below_h_star === "boolean"
        ? raw.horizon_validity.below_h_star
        : horizon <= hStar;

  return {
    series,
    tactic,
    horizon_months: horizon,
    p_active: pActive,
    ci_80_lo: finite(raw.ci_80_lo) ?? finite(ci80?.[0]) ?? pActive,
    ci_80_hi: finite(raw.ci_80_hi) ?? finite(ci80?.[1]) ?? pActive,
    ci_95_lo: finite(raw.ci_95_lo) ?? finite(ci95?.[0]) ?? pActive,
    ci_95_hi: finite(raw.ci_95_hi) ?? finite(ci95?.[1]) ?? pActive,
    h_star_months: hStar,
    below_h_star: belowHStar,
    forecast_route: route as ForecastCell["forecast_route"],
    stratum_id: text(raw.stratum_id) ?? text(raw.cell?.stratum_id) ?? ACTIVE_STRATUM,
    emitted_at: text(raw.emitted_at) ?? new Date(0).toISOString(),
    route_reasons: Array.isArray(raw.route_reasons) ? raw.route_reasons.filter((r): r is string => typeof r === "string") : [],
    identification_distance_posterior_median: finite(raw.identification_distance_posterior_median) ?? undefined,
    informativeness_I_O_lower_95: finite(raw.informativeness_I_O_lower_95) ?? undefined,
    informativeness_n_eff_i2: finite(raw.informativeness_n_eff_i2) ?? undefined,
    headline_flagged_by_posterior_fdp: raw.headline_flagged_by_posterior_fdp,
    headline_flagged_by_exceedance_gate: raw.headline_flagged_by_exceedance_gate,
    fdp_binding_reason: raw.fdp_binding_reason,
  };
}

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
  let dTheta: number | null = null;
  let dThetaFloor: number | null = null;
  let iLower95: number | null = null;
  let iLower95Floor: number | null = null;
  let nEffI2: number | null = null;
  let nEffI2Floor: number | null = null;
  let routeCounts: Record<string, number> | null = null;
  let routeTimestamp: string | null = null;

  if (latestRun) {
    runId = latestRun.run_id;
    const [summary, cellsRes, ledger] = await Promise.all([
      getFitSummary(latestRun.run_id),
      getForecastCells(latestRun.run_id),
      getRouteLedger(latestRun.run_id),
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
      dTheta = theoremNumber(summary, "A_prime", "median_d_theta");
      dThetaFloor = theoremNumber(summary, "A_prime", "floor");
      iLower95 = theoremNumber(summary, "B_prime", "lower_95_I_worst_source");
      iLower95Floor = theoremNumber(summary, "B_prime", "floor");
      nEffI2 = theoremNumber(summary, "B_prime", "n_eff_i2");
      nEffI2Floor = theoremNumber(summary, "B_prime", "n_eff_i2_floor");
    }
    routeCounts = ledger?.route_counts ?? null;
    routeTimestamp = ledger?.timestamp ?? null;
    const ledgerCells = Array.isArray(ledger?.cells)
      ? (ledger.cells as RawLedgerCell[]).map(normalizeLedgerCell).filter((c): c is ForecastCell => c != null)
      : [];
    cells = ledgerCells.length ? ledgerCells : (cellsRes.forecast_cells ?? []);
  }

  // Prefer the active stratum's cells; fall back to all cells.
  const stratumCells = cells.filter((c) => c.stratum_id === ACTIVE_STRATUM);
  const usedCells = stratumCells.length ? stratumCells : cells;

  const realCurve = buildCurve(usedCells);
  const realProbability = usedCells.length
    ? clamp01(usedCells.reduce((s, c) => s + (c.p_active ?? 0), 0) / usedCells.length)
    : null;
  const firstCell = usedCells[0];
  dTheta = finite(firstCell?.identification_distance_posterior_median) ?? dTheta;
  iLower95 = finite(firstCell?.informativeness_I_O_lower_95) ?? iLower95;
  nEffI2 = finite(firstCell?.informativeness_n_eff_i2) ?? nEffI2;

  // Fail-closed: only lock on an explicit "fail"; unknown/skip remains visible as demo/unknown state.
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
        activeRoute={firstCell?.forecast_route ?? null}
        routeReasons={firstCell?.route_reasons ?? []}
        routeCounts={routeCounts}
        routeTimestamp={routeTimestamp}
        dTheta={dTheta}
        dThetaFloor={dThetaFloor}
        iLower95={iLower95}
        iLower95Floor={iLower95Floor}
        nEffI2={nEffI2}
        nEffI2Floor={nEffI2Floor}
        hStar={hStar}
        hStarQ05={hStarQ05}
        hStarQ95={hStarQ95}
        usingMock={usingMock}
        updatedAgo={agoLabel(usedCells[0]?.emitted_at)}
      />

      {/* Route ledger sign-off — required by Automation Contract before headline promotion */}
      {runId && (
        <section className="console-panel p-5">
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
