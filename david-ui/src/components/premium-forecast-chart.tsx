"use client";

/**
 * E-2: FG5 mathematical grey-out — standalone trajectory chart.
 *
 * Renders the conditional forecast in full accent color for h ≤ h*, then
 * switches to the desaturated stationary-prior π∞ band for h > h*.
 *
 * Rendering contract:
 *   - RouteState is required. No chart renders without a route.
 *   - "headline" → full trajectory with grey-out at h*.
 *   - Any non-headline route → gate-locked placeholder (no chart axes).
 *   - The h* vertical rule carries the A-5 posterior spread band.
 *   - The greyed segment plots what the engine actually emits (π∞ posterior
 *     interval from the stored CI values), never an extrapolated line.
 */

import type { CurvePoint } from "@/components/forecast-router";
import { type RouteState, routeAccent, assertNever } from "@/lib/route-state";
import { motion } from "framer-motion";

// ── Deterministic demo data (SSR-stable, no Math.random) ──────────────────────

const DEMO_HORIZONS = [3, 6, 9, 12];

// h* = 5 months → only h=3 is conditional; h=6,9,12 are prior-dominated.
const H_STAR = 5;
const H_STAR_Q05 = 4;
const H_STAR_Q95 = 6;

function buildDemoCurve(): CurvePoint[] {
  // Conditional segment: h=3 (below h*)
  // Prior-dominated segment: h=6,9,12 (above h*)
  // π∞ level ≈ 0.35, conditional level ≈ 0.62
  const PI_INF = 0.35;

  const data: Array<{ h: number; mid: number; lo80: number; hi80: number; lo95: number; hi95: number; belowHStar: boolean }> = [
    { h: 3,  mid: 0.62, lo80: 0.52, hi80: 0.72, lo95: 0.46, hi95: 0.78, belowHStar: true  },
    { h: 6,  mid: 0.37, lo80: 0.28, hi80: 0.46, lo95: 0.23, hi95: 0.51, belowHStar: false },
    { h: 9,  mid: 0.35, lo80: 0.27, hi80: 0.44, lo95: 0.21, hi95: 0.49, belowHStar: false },
    { h: 12, mid: 0.34, lo80: 0.26, hi80: 0.43, lo95: 0.20, hi95: 0.48, belowHStar: false },
  ];

  return data.map((d, i) => ({
    x: i,
    mid: d.mid,
    lo80: d.lo80, hi80: d.hi80,
    lo95: d.lo95, hi95: d.hi95,
    belowHStar: d.belowHStar,
    horizonMonths: d.h,
    piInfMid: PI_INF,
    piInfLo80: PI_INF - 0.08,
    piInfHi80: PI_INF + 0.08,
  }));
}

const DEMO_CURVE = buildDemoCurve();
const DEMO_ROUTE_STATE: RouteState = {
  tag: "headline",
  p_active: 0.62,
  ci_80_lo: 0.52, ci_80_hi: 0.72,
  ci_95_lo: 0.46, ci_95_hi: 0.78,
  h_star_months: H_STAR,
  horizon_months: 3,
  flagged_by_fdp: false,
  fdp_binding_reason: null,
};

// ── SVG chart constants ────────────────────────────────────────────────────────

const W = 480, H = 160, PAD = 12;

// ── SVG chart implementation ───────────────────────────────────────────────────

export function TrajectoryChart({
  curve,
  hStarQ05,
  hStarQ95,
  accent,
}: Readonly<{
  curve: CurvePoint[];
  hStarQ05: number;
  hStarQ95: number;
  accent: string;
}>) {
  const lo = Math.min(...curve.map((p) => p.lo95));
  const hi = Math.max(...curve.map((p) => p.hi95));
  const dom = [Math.max(0, lo - 0.06), Math.min(1, hi + 0.06)];

  const xmax = Math.max(1, curve.length - 1);
  const sx = (xi: number) => PAD + (xi / xmax) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - dom[0]) / (dom[1] - dom[0])) * (H - 2 * PAD);

  // FG5 split
  const domIdx = curve.findIndex((p) => p.belowHStar === false);
  const hasPriorDom = domIdx > 0;
  const condCurve  = hasPriorDom ? curve.slice(0, domIdx) : curve;
  const priorCurve = hasPriorDom ? curve.slice(domIdx)    : [];
  const ruleX = hasPriorDom ? sx(domIdx) : null;

  // Map hStarQ05/Q95 months to x-pixel using horizonMonths field.
  const toXPx = (months: number) => {
    const pts = curve.filter((p) => p.horizonMonths != null);
    if (!pts.length) return null;
    const i = pts.findIndex((p) => p.horizonMonths! >= months);
    if (i < 0) return sx(pts.at(-1)!.x);
    if (i === 0) return sx(pts[0].x);
    const prev = pts[i - 1], next = pts[i];
    const t = (months - prev.horizonMonths!) / (next.horizonMonths! - prev.horizonMonths!);
    return sx(prev.x + t * (next.x - prev.x));
  };

  const spreadLo = toXPx(hStarQ05);
  const spreadHi = toXPx(hStarQ95);

  const bandPath = (pts: CurvePoint[], loK: "lo80" | "lo95", hiK: "hi80" | "hi95") => {
    if (pts.length < 2) return "";
    const top = pts.map((p) => `${sx(p.x)},${sy(p[hiK])}`).join(" L ");
    const bot = [...pts].reverse().map((p) => `${sx(p.x)},${sy(p[loK])}`).join(" L ");
    return `M ${top} L ${bot} Z`;
  };

  const midPath = (pts: CurvePoint[]) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x)} ${sy(p.mid)}`).join(" ");

  const piInfY = priorCurve.length > 0
    ? sy(priorCurve.reduce((s, p) => s + (p.piInfMid ?? p.mid), 0) / priorCurve.length)
    : null;

  const gid = "fc-" + accent.replace("#", "");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" preserveAspectRatio="none"
      role="img" aria-label="Forecast trajectory with FG5 prior-dominated grey-out">
      <defs>
        <filter id={gid}>
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <linearGradient id={`${gid}-fill`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={accent} stopOpacity="0.3" />
          <stop offset="100%" stopColor={accent} stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {/* Horizontal gridlines */}
      {[0.25, 0.5, 0.75].map((g) => (
        <line key={g} x1={PAD} x2={W - PAD}
          y1={sy(dom[0] + (dom[1] - dom[0]) * g)} y2={sy(dom[0] + (dom[1] - dom[0]) * g)}
          stroke="#1e293b" strokeWidth="1" strokeDasharray="3 5" />
      ))}

      {/* Y-axis labels */}
      {[0.25, 0.5, 0.75].map((g) => (
        <text key={g} x={PAD - 2} y={sy(dom[0] + (dom[1] - dom[0]) * g) + 3}
          fill="#334155" fontSize="8" fontFamily="monospace" textAnchor="end">
          {((dom[0] + (dom[1] - dom[0]) * g) * 100).toFixed(0)}
        </text>
      ))}

      {/* X-axis horizon labels */}
      {curve.map((p) => p.horizonMonths != null ? (
        <text key={p.x} x={sx(p.x)} y={H - 1}
          fill="#334155" fontSize="8" fontFamily="monospace" textAnchor="middle">
          {p.horizonMonths}m
        </text>
      ) : null)}

      {/* ── Conditional forecast (h ≤ h*) ─────────────────────────────────── */}
      <motion.path
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 0.08 }}
        transition={{ duration: 0.8, ease: "easeOut", delay: 0.1 }}
        d={bandPath(condCurve, "lo95", "hi95")} fill={accent}
        stroke={accent} strokeOpacity="0.15" strokeWidth="1" />
      <motion.path
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
        d={bandPath(condCurve, "lo80", "hi80")} fill={`url(#${gid}-fill)`}
        stroke={accent} strokeOpacity="0.45" strokeWidth="1" />
      {condCurve.length > 1 && (
        <>
          <motion.path
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
            d={midPath(condCurve)} fill="none" stroke={accent} strokeWidth="2.2"
            filter={`url(#${gid})`} />
          <motion.circle
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", bounce: 0.5, delay: 1.1 }}
            cx={sx(condCurve.at(-1)!.x)} cy={sy(condCurve.at(-1)!.mid)}
            r="3" fill={accent} filter={`url(#${gid})`} />
        </>
      )}

      {/* ── FG5 boundary: h* rule + A-5 spread band ───────────────────────── */}
      {hasPriorDom && ruleX != null && (
        <g>
          {/* A-5 posterior spread */}
          {spreadLo != null && spreadHi != null && (
            <rect
              x={Math.min(spreadLo, spreadHi) - 1}
              y={PAD}
              width={Math.abs(spreadHi - spreadLo) + 2}
              height={H - PAD * 2}
              fill="#f59e0b" fillOpacity="0.09"
            />
          )}
          {/* π∞ mean guide */}
          {piInfY != null && (
            <line x1={ruleX} x2={W - PAD} y1={piInfY} y2={piInfY}
              stroke="#475569" strokeWidth="1" strokeDasharray="2 5" />
          )}
          {/* h* vertical rule */}
          <line x1={ruleX} y1={PAD} x2={ruleX} y2={H - PAD}
            stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="5 3" strokeOpacity="0.9" />
          <text x={ruleX + 4} y={PAD + 11} fill="#f59e0b" fontSize="9" fontFamily="monospace"
            fontWeight="600">h*</text>
          {spreadLo != null && spreadHi != null && (
            <text x={ruleX + 4} y={PAD + 21} fill="#b45309" fontSize="7" fontFamily="monospace">
              [{hStarQ05}–{hStarQ95}]mo
            </text>
          )}
        </g>
      )}

      {/* ── Prior-dominated π∞ band (h > h*) — desaturated grey ──────────── */}
      {priorCurve.length > 0 && (
        <motion.g
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.60 }}
          transition={{ duration: 0.6, delay: 1.2 }}
        >
          <path d={bandPath(priorCurve, "lo80", "hi80")}
            fill="#475569" fillOpacity="0.20" stroke="#475569" strokeOpacity="0.30" strokeWidth="1" />
          <path d={bandPath(priorCurve, "lo95", "hi95")}
            fill="none" stroke="#334155" strokeOpacity="0.35" strokeWidth="1" strokeDasharray="2 4" />
          <text x={sx(priorCurve[0].x) + 6} y={H - PAD - 4}
            fill="#64748b" fontSize="8" fontFamily="monospace">π∞ prior band</text>
        </motion.g>
      )}
    </svg>
  );
}

// ── Locked overlay for non-headline routes ────────────────────────────────────

function GateLockedOverlay({ routeState }: Readonly<{ routeState: Exclude<RouteState, { tag: "headline" }> }>) {
  const accent = routeAccent(routeState);
  const topReason = ("reasons" in routeState ? routeState.reasons[0] : undefined) ?? null;

  return (
    <div className={`flex flex-col items-center justify-center gap-3 rounded-xl border p-5 min-h-[160px] ${accent.bg} ${accent.border}`}>
      <span className={`text-[10px] font-bold uppercase tracking-widest font-mono ${accent.text}`}>
        {accent.label}
      </span>
      <p className="text-xs text-slate-400 text-center max-w-xs">
        Forecast trajectory unavailable — gate constraint prevents chart rendering.
      </p>
      {topReason && (
        <span className="text-[10px] font-mono text-slate-500">{topReason}</span>
      )}
    </div>
  );
}

// ── Public component ───────────────────────────────────────────────────────────

export interface PremiumForecastChartProps {
  /** E-1 contract: RouteState required; defaults to demo headline state. */
  routeState?: RouteState;
  curve?: CurvePoint[];
  hStarQ05?: number;
  hStarQ95?: number;
  accent?: string;
}

export function PremiumForecastChart({
  routeState = DEMO_ROUTE_STATE,
  curve = DEMO_CURVE,
  hStarQ05 = H_STAR_Q05,
  hStarQ95 = H_STAR_Q95,
  accent = "#22d3ee",
}: Readonly<PremiumForecastChartProps>) {
  switch (routeState.tag) {
    case "headline":
      return (
        <div className="bg-[#070b19]/90 border border-slate-800 rounded-xl p-4 flex flex-col min-h-[200px] relative w-full">
          {/* Header */}
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Forecast Trajectory
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                <span className="inline-flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm" style={{ background: accent, opacity: 0.6 }} />
                  Conditional h ≤ h*
                </span>
                {" · "}
                <span className="inline-flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm bg-slate-600 opacity-60" />
                  π∞ prior band h &gt; h*
                </span>
              </p>
            </div>
            <div className="text-right">
              <span className="text-[9px] font-mono text-amber-400/80">
                h* = {routeState.h_star_months}mo
              </span>
              <br />
              <span className="text-[8px] font-mono text-slate-600">
                [{hStarQ05}–{hStarQ95}] spread
              </span>
            </div>
          </div>

          {/* Chart */}
          <div className="flex-1 min-h-[130px]">
            <TrajectoryChart
              curve={curve}
              hStarQ05={hStarQ05}
              hStarQ95={hStarQ95}
              accent={accent}
            />
          </div>

          {/* Footer */}
          <div className="mt-2 flex items-center justify-between text-[9px] font-mono text-slate-600">
            <span>Theorem D-forecast · FG5 grey-out active</span>
            <span className="text-emerald-600">route: {routeState.tag}</span>
          </div>
        </div>
      );

    case "horizon_prior_dominated":
      return <GateLockedOverlay routeState={routeState} />;
    case "monitor_only":
      return <GateLockedOverlay routeState={routeState} />;
    case "aggregate_only":
      return <GateLockedOverlay routeState={routeState} />;
    case "evidence_gap":
      return <GateLockedOverlay routeState={routeState} />;
    case "withhold":
      return <GateLockedOverlay routeState={routeState} />;
    case "prior_dominated":
      return <GateLockedOverlay routeState={routeState} />;

    default:
      return assertNever(routeState);
  }
}
