"use client";

import { useMemo } from "react";
import { ExplainDot } from "@/components/explain-dot";
import { TrajectoryChart } from "@/components/premium-forecast-chart";
import type { RouteTag } from "@/lib/route-state";

export interface CurvePoint {
  x: number;
  mid: number;
  lo80: number;
  hi80: number;
  lo95: number;
  hi95: number;
  belowHStar?: boolean;
  horizonMonths?: number;
  piInfMid?: number;
  piInfLo80?: number;
  piInfHi80?: number;
}

export interface RouterDataProps {
  realProbability: number | null;
  realCurve: CurvePoint[] | null;
  sbcFail: boolean;
  theoremFail: boolean;
  aPrime: string;
  bPrime: string;
  activeStratum: string;
  runId: string | null;
  activeRoute: RouteTag | null;
  routeReasons: string[];
  routeCounts: Record<string, number> | null;
  routeTimestamp: string | null;
  dTheta: number | null;
  dThetaFloor: number | null;
  iLower95: number | null;
  iLower95Floor: number | null;
  nEffI2: number | null;
  nEffI2Floor: number | null;
  hStar: number | null;
  hStarQ05: number | null;
  hStarQ95: number | null;
  usingMock: boolean;
  updatedAgo: string;
}

type Tone = "pass" | "fail" | "skip" | "lock";

const ROUTE_ORDER: RouteTag[] = [
  "headline",
  "monitor_only",
  "aggregate_only",
  "evidence_gap",
  "withhold",
  "horizon_prior_dominated",
  "prior_dominated",
];

const ROUTE_LABEL: Record<RouteTag, string> = {
  headline: "HEADLINE",
  monitor_only: "MONITOR",
  aggregate_only: "AGG",
  evidence_gap: "EVIDENCE_GAP",
  withhold: "WITHHOLD",
  horizon_prior_dominated: "H_PRIOR",
  prior_dominated: "PRIOR",
};

const DEMO_CURVE: CurvePoint[] = [
  { x: 0, horizonMonths: 3, mid: 0.62, lo80: 0.53, hi80: 0.71, lo95: 0.47, hi95: 0.77, belowHStar: true },
  { x: 1, horizonMonths: 6, mid: 0.37, lo80: 0.29, hi80: 0.45, lo95: 0.22, hi95: 0.52, belowHStar: false },
  { x: 2, horizonMonths: 9, mid: 0.35, lo80: 0.27, hi80: 0.43, lo95: 0.21, hi95: 0.49, belowHStar: false },
  { x: 3, horizonMonths: 12, mid: 0.34, lo80: 0.26, hi80: 0.42, lo95: 0.2, hi95: 0.48, belowHStar: false },
];

const EXPLAIN_CI = {
  side: "left" as const,
  title: "Forecast intervals",
  body: "The chart renders stored posterior cell probabilities and credible intervals. Horizons beyond h* use the prior-dominated segment emitted by the engine.",
  formula: "route_ledger.cells[].p_active; ci_80; ci_95",
};

function fmt(value: number | null, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return value.toFixed(digits);
}

function fmtPct(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function toneClass(tone: Tone): string {
  switch (tone) {
    case "pass":
      return "border-emerald-800/70 bg-emerald-950/20 text-emerald-300";
    case "fail":
      return "border-red-800/70 bg-red-950/25 text-red-300";
    case "lock":
      return "border-amber-800/70 bg-amber-950/20 text-amber-300";
    case "skip":
      return "border-slate-700 bg-slate-900/70 text-slate-400";
  }
}

function routeTone(route: RouteTag | null): Tone {
  if (route === "headline") return "pass";
  if (route === "monitor_only" || route === "horizon_prior_dominated" || route === "aggregate_only") return "lock";
  if (route === "evidence_gap" || route === "withhold" || route === "prior_dominated") return "fail";
  return "skip";
}

function statusTone(status: string): Tone {
  if (status === "pass") return "pass";
  if (status === "fail" || status === "error") return "fail";
  return "skip";
}

function gateTone(value: number | null, floor: number | null, status?: string): Tone {
  if (status) return statusTone(status);
  if (value == null || floor == null) return "skip";
  return value >= floor ? "pass" : "fail";
}

function SourceLine({ field }: Readonly<{ field: string }>) {
  return <p className="ledger-field mt-1 truncate" title={field}>field: {field}</p>;
}

function RouteChip({ route, active, count }: Readonly<{ route: RouteTag; active: boolean; count: number | null }>) {
  return (
    <span
      className={`route-chip ${active ? toneClass(routeTone(route)) : "border-slate-800 bg-slate-950/50 text-slate-500"}`}
      title={`route_ledger.route_counts.${route}`}
    >
      {ROUTE_LABEL[route]}
      {count != null && <span className="numeric text-slate-300">{count}</span>}
    </span>
  );
}

function GateBadge({
  label,
  value,
  floor,
  tone,
  field,
  floorField,
  suffix = "",
}: Readonly<{
  label: string;
  value: string;
  floor: string;
  tone: Tone;
  field: string;
  floorField: string;
  suffix?: string;
}>) {
  return (
    <div className={`console-panel px-3 py-3 ${toneClass(tone)}`}>
      <div className="flex items-start justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">{label}</span>
        <span className="rounded border border-current/30 px-1.5 py-0.5 text-[10px] font-bold uppercase">
          {tone}
        </span>
      </div>
      <div className="mt-2 flex items-end justify-between gap-3">
        <span className="numeric text-xl font-semibold text-slate-100">{value}{suffix}</span>
        <span className="numeric text-xs text-slate-400">floor {floor}</span>
      </div>
      <SourceLine field={field} />
      <SourceLine field={floorField} />
    </div>
  );
}

function ConsoleMetric({
  label,
  value,
  field,
}: Readonly<{
  label: string;
  value: string;
  field: string;
}>) {
  return (
    <div className="console-panel p-3">
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</p>
      <p className="numeric mt-2 text-2xl font-semibold text-slate-100">{value}</p>
      <SourceLine field={field} />
    </div>
  );
}

function Header({ props }: Readonly<{ props: RouterDataProps }>) {
  const route = props.activeRoute ? ROUTE_LABEL[props.activeRoute] : "NO_ROUTE";
  return (
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`route-chip ${toneClass(routeTone(props.activeRoute))}`}>{route}</span>
          {props.usingMock && <span className="route-chip border-amber-800/70 bg-amber-950/20 text-amber-300">DEMO_SOURCE</span>}
          {props.sbcFail && <span className="route-chip border-red-800/70 bg-red-950/25 text-red-300">SBC_LOCK</span>}
          {props.theoremFail && <span className="route-chip border-red-800/70 bg-red-950/25 text-red-300">THEOREM_LOCK</span>}
        </div>
        <h1 className="mt-3 text-2xl font-semibold tracking-tight text-slate-100">
          Forecast Router Console
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Run <span className="numeric text-slate-200">{props.runId ?? "no-live-run"}</span> · stratum{" "}
          <span className="numeric text-slate-200">{props.activeStratum}</span> · updated{" "}
          <span className="numeric text-slate-200">{props.updatedAgo}</span>
        </p>
      </div>
      <div className="console-panel min-w-56 p-3">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">ledger timestamp</p>
        <p className="numeric mt-2 text-sm text-slate-200">{props.routeTimestamp ?? "--"}</p>
        <SourceLine field="route_ledger.timestamp" />
      </div>
    </header>
  );
}

function GateStatusRow({ props }: Readonly<{ props: RouterDataProps }>) {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
      <GateBadge
        label="d(theta)"
        value={fmt(props.dTheta)}
        floor={fmt(props.dThetaFloor)}
        tone={gateTone(props.dTheta, props.dThetaFloor, props.aPrime)}
        field="route_ledger.cells[].identification_distance_posterior_median"
        floorField="fit_summary.theorems.A_prime.floor"
      />
      <GateBadge
        label="I(O)"
        value={fmt(props.iLower95)}
        floor={fmt(props.iLower95Floor)}
        tone={gateTone(props.iLower95, props.iLower95Floor, props.bPrime)}
        field="route_ledger.cells[].informativeness_I_O_lower_95"
        floorField="fit_summary.theorems.B_prime.floor"
      />
      <GateBadge
        label="N_eff*I^2"
        value={fmt(props.nEffI2, 2)}
        floor={fmt(props.nEffI2Floor, 2)}
        tone={gateTone(props.nEffI2, props.nEffI2Floor, props.bPrime)}
        field="route_ledger.cells[].informativeness_n_eff_i2"
        floorField="fit_summary.theorems.B_prime.n_eff_i2_floor"
      />
      <GateBadge
        label="h*"
        value={fmt(props.hStar, 0)}
        floor="h <= h*"
        tone={props.hStar == null ? "skip" : props.activeRoute === "horizon_prior_dominated" ? "lock" : "pass"}
        field="fit_summary.theorems.D_prime.h_star_months"
        floorField="route_ledger.cells[].horizon_validity.below_h_star"
        suffix={props.hStar == null ? "" : " mo"}
      />
    </div>
  );
}

function RouteCounts({ routeCounts, activeRoute }: Readonly<{
  routeCounts: Record<string, number> | null;
  activeRoute: RouteTag | null;
}>) {
  return (
    <section className="console-panel p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-200">Route Codes</h2>
        <SourceLine field="route_ledger.route_counts" />
      </div>
      <div className="flex flex-wrap gap-2">
        {ROUTE_ORDER.map((route) => (
          <RouteChip
            key={route}
            route={route}
            active={route === activeRoute}
            count={typeof routeCounts?.[route] === "number" ? routeCounts[route] : null}
          />
        ))}
      </div>
    </section>
  );
}

function ReasonPanel({ reasons }: Readonly<{ reasons: string[] }>) {
  return (
    <section className="console-panel p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-200">Binding Reasons</h2>
        <SourceLine field="route_ledger.cells[].route_reasons" />
      </div>
      {reasons.length ? (
        <div className="flex flex-wrap gap-2">
          {reasons.map((reason) => (
            <span key={reason} className="route-chip border-slate-700 bg-slate-950/60 text-slate-300">
              {reason}
            </span>
          ))}
        </div>
      ) : (
        <p className="numeric text-sm text-slate-500">--</p>
      )}
    </section>
  );
}

function ForecastChartCard({ props, curve }: Readonly<{ props: RouterDataProps; curve: CurvePoint[] }>) {
  const locked = props.sbcFail || props.theoremFail;
  return (
    <section className="console-panel relative p-4 xl:col-span-2">
      <ExplainDot {...EXPLAIN_CI} />
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">Posterior Trajectory</h2>
          <SourceLine field="route_ledger.cells[].p_active + credible intervals + h_star_months" />
        </div>
        <div className="flex gap-3 text-[10px] uppercase tracking-wider text-slate-500">
          <span className="numeric">80 CI</span>
          <span className="numeric">95 CI</span>
          <span className="numeric">PI-INF</span>
        </div>
      </div>
      <div className="relative h-52">
        <TrajectoryChart
          curve={curve}
          accent="#38bdf8"
          hStarQ05={props.hStarQ05 ?? props.hStar ?? 0}
          hStarQ95={props.hStarQ95 ?? props.hStar ?? 0}
        />
        {locked && (
          <div className="absolute inset-0 flex items-center justify-center rounded-lg border border-red-900/60 bg-[#050813]/90">
            <div className="max-w-md px-4 text-center">
              <p className="route-chip mx-auto w-fit border-red-800/70 bg-red-950/25 text-red-300">
                {props.sbcFail ? "FG4_SBC_LOCK" : "THEOREM_GATE_LOCK"}
              </p>
              <p className="mt-3 text-xs leading-relaxed text-red-200">
                {props.sbcFail
                  ? "Forecast display is locked because SBC returned fail."
                  : "Forecast display is locked because A_prime or B_prime returned fail."}
              </p>
            </div>
          </div>
        )}
      </div>
      {props.hStar != null && (
        <p className="numeric mt-3 text-xs text-slate-400">
          h* {fmt(props.hStar, 0)} mo
          {props.hStarQ05 != null && props.hStarQ95 != null && (
            <span> [{fmt(props.hStarQ05, 0)}-{fmt(props.hStarQ95, 0)}]</span>
          )}
        </p>
      )}
    </section>
  );
}

export function ForecastRouter(props: Readonly<RouterDataProps>) {
  const curve = useMemo(
    () => (props.realCurve && props.realCurve.length > 1 ? props.realCurve : DEMO_CURVE),
    [props.realCurve],
  );

  return (
    <div className="space-y-5">
      <Header props={props} />

      <GateStatusRow props={props} />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <ForecastChartCard props={props} curve={curve} />
        <div className="space-y-4">
          <ConsoleMetric
            label="mean p(active)"
            value={fmtPct(props.realProbability)}
            field="route_ledger.cells[].p_active"
          />
          <ConsoleMetric
            label="A_prime"
            value={props.aPrime.toUpperCase()}
            field="fit_summary.theorems.A_prime.gate_status"
          />
          <ConsoleMetric
            label="B_prime"
            value={props.bPrime.toUpperCase()}
            field="fit_summary.theorems.B_prime.gate_status"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <RouteCounts routeCounts={props.routeCounts} activeRoute={props.activeRoute} />
        <ReasonPanel reasons={props.routeReasons} />
      </div>
    </div>
  );
}
