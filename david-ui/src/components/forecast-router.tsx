"use client";

import { useMemo, useState } from "react";
import { ExplainDot } from "@/components/explain-dot";

/* ────────────────────────────────────────────────────────────────────────────
   Types
──────────────────────────────────────────────────────────────────────────── */

export interface CurvePoint {
  x: number;     // step index (e.g. horizon month)
  mid: number;   // p(active) point estimate, 0..1
  lo80: number;
  hi80: number;
  lo95: number;
  hi95: number;
  belowHStar?: boolean; // Theorem D-forecast: false → horizon_prior_dominated
  horizonMonths?: number;
}

interface Regime { bullish: number; bearish: number; volatile: number; stable: number }

interface ModelData {
  id: string;
  name: string;
  probability: number;   // 0..100, %
  direction: "up" | "down";
  expectedReturn: number; // %
  riskScore: number;      // 0..100 (lower = safer)
  curve: CurvePoint[];
  regimeSeries: Regime[]; // normalized per step, over x
  routing: { davidAlpha: number; deepNet: number }; // %
  indicators: { volume: number; vix: number };      // 0..100 health
  accent: string;         // hex
}

/** Server → client props */
export interface RouterDataProps {
  /** Real point estimate (mean p_active across cells), 0..1, if available */
  realProbability: number | null;
  /** Real forecast curve derived from forecast-cells, if available */
  realCurve: CurvePoint[] | null;
  /** SBC calibration failed locally → lock entire tab */
  sbcFail: boolean;
  /** Theorem A′ or B′ failed for the active stratum → lock its forecast curve */
  theoremFail: boolean;
  aPrime: string; // gate_status
  bPrime: string; // gate_status
  activeStratum: string;
  runId: string | null;
  /** Theorem D-forecast horizon-validity boundary h*(g) in months */
  hStar: number | null;
  /** True when no live data was returned and mock fallbacks are shown */
  usingMock: boolean;
  updatedAgo: string;
}

/* ────────────────────────────────────────────────────────────────────────────
   Deterministic mock generators (no Math.random → SSR-stable)
──────────────────────────────────────────────────────────────────────────── */

const N = 24; // horizon months

function genCurve(seed: number, base: number, amp: number, width: number): CurvePoint[] {
  const pts: CurvePoint[] = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const mid = clamp01(
      base + amp * Math.sin(t * Math.PI * 1.6 + seed) + 0.04 * Math.sin(t * Math.PI * 6 + seed * 2)
    );
    const spread = width * (0.45 + 0.55 * t); // uncertainty fans out with horizon
    pts.push({
      x: i,
      mid,
      lo80: clamp01(mid - spread * 0.6),
      hi80: clamp01(mid + spread * 0.6),
      lo95: clamp01(mid - spread),
      hi95: clamp01(mid + spread),
    });
  }
  return pts;
}

function genRegime(seed: number, weights: Regime): Regime[] {
  const out: Regime[] = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const raw = {
      bullish:  Math.max(0.05, weights.bullish  + 0.18 * Math.sin(t * Math.PI * 2 + seed)),
      bearish:  Math.max(0.05, weights.bearish  + 0.14 * Math.sin(t * Math.PI * 2.4 + seed + 1)),
      volatile: Math.max(0.05, weights.volatile + 0.12 * Math.sin(t * Math.PI * 3 + seed + 2)),
      stable:   Math.max(0.05, weights.stable   + 0.1 * Math.sin(t * Math.PI * 1.7 + seed + 3)),
    };
    const sum = raw.bullish + raw.bearish + raw.volatile + raw.stable;
    out.push({
      bullish: raw.bullish / sum,
      bearish: raw.bearish / sum,
      volatile: raw.volatile / sum,
      stable: raw.stable / sum,
    });
  }
  return out;
}

function clamp01(v: number) { return Math.max(0.01, Math.min(0.99, v)); }

const BASE_MODELS: ModelData[] = [
  {
    id: "router",
    name: "Forecast Router",
    probability: 71.4, direction: "up", expectedReturn: 12.8, riskScore: 34,
    curve: genCurve(0.4, 0.68, 0.12, 0.1),
    regimeSeries: genRegime(0.5, { bullish: 0.42, bearish: 0.18, volatile: 0.22, stable: 0.18 }),
    routing: { davidAlpha: 64, deepNet: 36 },
    indicators: { volume: 78, vix: 41 },
    accent: "#22d3ee",
  },
  {
    id: "macro",
    name: "Macro Trends AI",
    probability: 58.2, direction: "up", expectedReturn: 8.1, riskScore: 52,
    curve: genCurve(1.7, 0.55, 0.16, 0.13),
    regimeSeries: genRegime(2.1, { bullish: 0.3, bearish: 0.28, volatile: 0.24, stable: 0.18 }),
    routing: { davidAlpha: 48, deepNet: 52 },
    indicators: { volume: 63, vix: 58 },
    accent: "#a78bfa",
  },
  {
    id: "deepalpha",
    name: "DeepAlpha v4",
    probability: 81.9, direction: "up", expectedReturn: 19.4, riskScore: 47,
    curve: genCurve(3.2, 0.78, 0.1, 0.08),
    regimeSeries: genRegime(3.6, { bullish: 0.52, bearish: 0.12, volatile: 0.26, stable: 0.1 }),
    routing: { davidAlpha: 41, deepNet: 59 },
    indicators: { volume: 88, vix: 33 },
    accent: "#34d399",
  },
  {
    id: "arima",
    name: "ARIMA Hybrid",
    probability: 46.7, direction: "down", expectedReturn: -3.2, riskScore: 61,
    curve: genCurve(4.9, 0.46, 0.2, 0.15),
    regimeSeries: genRegime(5.3, { bullish: 0.22, bearish: 0.34, volatile: 0.28, stable: 0.16 }),
    routing: { davidAlpha: 55, deepNet: 45 },
    indicators: { volume: 51, vix: 69 },
    accent: "#fbbf24",
  },
];

/* ────────────────────────────────────────────────────────────────────────────
   SVG charts (custom, with neon glow filters)
──────────────────────────────────────────────────────────────────────────── */

const W = 480, H = 180, PAD = 8;

function ForecastAreaChart({ curve, accent }: Readonly<{ curve: CurvePoint[]; accent: string }>) {
  const lo = Math.min(...curve.map((p) => p.lo95));
  const hi = Math.max(...curve.map((p) => p.hi95));
  const dom = [Math.max(0, lo - 0.04), Math.min(1, hi + 0.04)];

  // Scale x to fill the full width regardless of point count (4 horizons or 24 mock steps).
  const xmax = Math.max(1, curve.length - 1);
  const sx = (x: number) => PAD + (x / xmax) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - ((v - dom[0]) / (dom[1] - dom[0])) * (H - 2 * PAD);

  // Theorem D-forecast: first index whose horizon exceeds h* → prior-dominated tail.
  const domIdx = curve.findIndex((p) => p.belowHStar === false);
  const hasPriorDom = domIdx > 0;

  const band = (loKey: "lo80" | "lo95", hiKey: "hi80" | "hi95") => {
    const top = curve.map((p) => `${sx(p.x)},${sy(p[hiKey])}`).join(" L ");
    const bot = [...curve].reverse().map((p) => `${sx(p.x)},${sy(p[loKey])}`).join(" L ");
    return `M ${top} L ${bot} Z`;
  };

  const midLine = curve.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x)} ${sy(p.mid)}`).join(" ");
  const gid = "fglow-" + accent.replace("#", "");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" preserveAspectRatio="none">
      <defs>
        <filter id={gid} x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3.2" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <linearGradient id={gid + "-fill"} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={accent} stopOpacity="0.35" />
          <stop offset="100%" stopColor={accent} stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {/* gridlines */}
      {[0.25, 0.5, 0.75].map((g) => (
        <line key={g} x1={PAD} x2={W - PAD} y1={sy(dom[0] + (dom[1] - dom[0]) * g)} y2={sy(dom[0] + (dom[1] - dom[0]) * g)}
          stroke="#1e293b" strokeWidth="1" strokeDasharray="3 4" />
      ))}

      {/* 95% CI */}
      <path d={band("lo95", "hi95")} fill={accent} fillOpacity="0.10" stroke={accent} strokeOpacity="0.18" strokeWidth="1" />
      {/* 80% CI */}
      <path d={band("lo80", "hi80")} fill={`url(#${gid}-fill)`} stroke={accent} strokeOpacity="0.4" strokeWidth="1" />
      {/* median line, glowing */}
      <path d={midLine} fill="none" stroke={accent} strokeWidth="2.5" filter={`url(#${gid})`} />
      {/* leading dot */}
      <circle cx={sx(curve.at(-1)!.x)} cy={sy(curve.at(-1)!.mid)} r="3.5" fill={accent} filter={`url(#${gid})`} />

      {/* Theorem D-forecast: prior-dominated horizon tail (h > h*) is greyed + hatched */}
      {hasPriorDom && (
        <g>
          <rect x={sx(domIdx)} y={PAD} width={W - PAD - sx(domIdx)} height={H - 2 * PAD}
            fill="#0b1222" fillOpacity="0.62" />
          <line x1={sx(domIdx)} y1={PAD} x2={sx(domIdx)} y2={H - PAD}
            stroke="#fbbf24" strokeWidth="1.5" strokeDasharray="4 3" strokeOpacity="0.8" />
          <text x={sx(domIdx) + 6} y={PAD + 14} fill="#fbbf24" fontSize="10" fontWeight="600">
            h&gt;h* · prior-dominated
          </text>
        </g>
      )}
    </svg>
  );
}

const REGIME_COLORS = {
  bullish: "#34d399", bearish: "#f87171", volatile: "#fbbf24", stable: "#38bdf8",
} as const;

function RegimeAreaChart({ series }: Readonly<{ series: Regime[] }>) {
  const sx = (x: number) => PAD + (x / N) * (W - 2 * PAD);
  const sy = (v: number) => H - PAD - v * (H - 2 * PAD);

  // normalized stacked area, order bottom→top: stable, volatile, bearish, bullish
  const order: (keyof Regime)[] = ["stable", "volatile", "bearish", "bullish"];
  const cum = series.map(() => 0);
  const layers = order.map((key) => {
    const top = series.map((s, i) => {
      cum[i] += s[key];
      return `${sx(i)},${sy(cum[i])}`;
    });
    // bottom edge = cumulative before this layer
    const bottomVals = series.map((s, i) => cum[i] - s[key]);
    const bot = [...series].map((_, i) => i).reverse().map((i) => `${sx(i)},${sy(bottomVals[i])}`);
    return { key, d: `M ${top.join(" L ")} L ${bot.join(" L ")} Z` };
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" preserveAspectRatio="none">
      {layers.map((l) => (
        <path key={l.key} d={l.d} fill={REGIME_COLORS[l.key]} fillOpacity="0.55"
          stroke={REGIME_COLORS[l.key]} strokeOpacity="0.7" strokeWidth="0.75" />
      ))}
    </svg>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
   Small UI helpers
──────────────────────────────────────────────────────────────────────────── */

function Bar({ label, value, color }: Readonly<{ label: string; value: number; color: string }>) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px]">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-200">{value}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-800">
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${value}%`, background: color, boxShadow: `0 0 10px ${color}` }} />
      </div>
    </div>
  );
}

function CardShell({ title, children, explain }: Readonly<{
  title: string; children: React.ReactNode;
  explain?: { title: string; body: string; formula?: string; side?: "right" | "left" | "top" | "bottom" };
}>) {
  return (
    <div className="card-premium relative p-4">
      {explain && <ExplainDot {...explain} />}
      <p className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{title}</p>
      {children}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────────────────────
   Main component
──────────────────────────────────────────────────────────────────────────── */

/* ── status / message constants (dedupe repeated literals) ────────────────── */
const ROUTER_ID = "router";
const MSG_SBC_LOCK = "Motori i bllokuar: Modeli nuk është i kalibruar matematikisht lokalisht.";
const MSG_THEOREM_BANNER = "Disa stratume janë të bllokuara nga teoremat. Shih kurbën më poshtë.";

const EXPLAIN_CURRENT = {
  title: "Probabiliteti Aktual",
  body: "Probabiliteti posterior i agreguar nga koduesit e pavarur LLM nëpërmjet modelit Dawid-Skene, i cili korrigjon për saktësinë (sensitivitet/specificitet) e secilit burim përpara se t'i bashkojë vlerësimet.",
  formula: "P(aktiv | të dhënat) = ∫ θ · p(θ|y) dθ",
};
const EXPLAIN_CI = {
  side: "left" as const,
  title: "Intervalet e Besueshmërisë",
  body: "Brezat tregojnë pasigurinë prior→posterior të vlerësuar nga zinxhirët MCMC. Brezi 80% dhe 95% janë intervale të besueshmërisë Bayesiane: me probabilitet 95% vlera e vërtetë ndodhet brenda brezit të jashtëm.",
  formula: "CI₉₅ = [Q₀.₀₂₅(θ|y), Q₀.₉₇₅(θ|y)]",
};
const EXPLAIN_REGIME = {
  side: "left" as const,
  title: "Shpërndarja e Regjimeve",
  body: "Klasat latente të modelit hartohen në taktikat e ndërhyrjes së industrisë së duhanit: Bullish↔ofensivë SIO, Bearish↔presion CSIO, Volatile↔MIO oportuniste, Stable↔periudha pa ndërhyrje.",
  formula: "z_t ~ Categorical(π),  Σπ = 1",
};
const EXPLAIN_ROUTING = {
  side: "left" as const,
  title: "Statusi i Rrugëzimit / Teoremat",
  body: "Rrugëzuesi kombinon dy motorë. Parashikimet zhbllokohen vetëm kur stratumi ka ≥ 3 burime strukturalisht të pavarura (Teorema A′) — kjo garanton identifikueshmëri dhe pengon kapjen nga një burim i vetëm.",
  formula: "A′:  #{burime të pavarura} ≥ 3",
};

function gateChipClass(status: string): string {
  return status === "pass" ? "bg-emerald-950/60 text-emerald-300" : "bg-red-950/60 text-red-300";
}

/** Inject live data into the "router" model; the others stay illustrative. */
function buildModels(realCurve: CurvePoint[] | null, realProbability: number | null): ModelData[] {
  return BASE_MODELS.map((m) => {
    if (m.id === ROUTER_ID) {
      const curve = realCurve && realCurve.length > 1 ? realCurve : m.curve;
      const probability = realProbability == null ? m.probability : realProbability * 100;
      const direction: "up" | "down" = probability >= 50 ? "up" : "down";
      return { ...m, curve, probability, direction };
    }
    return m;
  });
}

/* ── sub-components ───────────────────────────────────────────────────────── */

function RouterHeader({ updatedAgo, usingMock, explain, onToggle }: Readonly<{
  updatedAgo: string; usingMock: boolean; explain: boolean; onToggle: () => void;
}>) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white">
          {"FORECAST ROUTER"}{" "}<span className="text-slate-600">|</span>{" "}
          <span className="text-base font-medium text-slate-400">Performance Analytics &amp; Regime Allocation</span>
        </h1>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-700/40 bg-emerald-950/30 px-2.5 py-1 text-emerald-300">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 dot-live text-emerald-400 animate-pulse" />{" "}
            {"AI Engine Active"}
          </span>
          <span className="rounded-full border border-slate-700 bg-slate-900 px-2.5 py-1 text-slate-400">
            Data Updated: {updatedAgo}
          </span>
          <span className="rounded-full border border-slate-700 bg-slate-900 px-2.5 py-1 text-slate-400">
            Confidence: <span className="text-cyan-300">High</span>
          </span>
          {usingMock && (
            <span className="rounded-full border border-amber-700/40 bg-amber-950/20 px-2.5 py-1 text-amber-300">
              demo data (backend empty)
            </span>
          )}
        </div>
      </div>

      <button
        type="button"
        onClick={onToggle}
        aria-pressed={explain}
        className={`inline-flex items-center gap-2 rounded-lg border px-3.5 py-2 text-sm font-medium transition-all ${
          explain
            ? "border-sky-400/60 bg-sky-950/50 text-sky-200 glow-cyan"
            : "border-slate-700 bg-slate-900 text-slate-400 hover:text-slate-200"
        }`}
      >
        {"🇦🇱 Shpjegimi"}{" "}
        <span className={`text-[10px] uppercase ${explain ? "text-sky-300" : "text-slate-600"}`}>
          {explain ? "Aktiv" : "Joaktiv"}
        </span>
      </button>
    </div>
  );
}

function TrustBanner({ certified, sbcFail, usingMock }: Readonly<{
  certified: boolean; sbcFail: boolean; usingMock: boolean;
}>) {
  if (usingMock) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-amber-700/50 bg-amber-950/30 px-4 py-2.5 text-sm font-medium text-amber-300">
        🔍 Demo mode — live theorem and calibration checks are not active.
      </div>
    );
  }
  if (certified) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-emerald-700/50 bg-emerald-950/30 px-4 py-2.5 text-sm font-medium text-emerald-300 glow-emerald">
        🛡️ Parashikim i Certifikuar Matematikisht (SBC &amp; Teoremat A′ B′ Kaluan)
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 rounded-xl border border-red-700/50 bg-red-950/30 px-4 py-2.5 text-sm font-medium text-red-300">
      ⛔ {sbcFail ? MSG_SBC_LOCK : MSG_THEOREM_BANNER}
    </div>
  );
}

function ModelSelector({ models, activeId, onSelect, activeStratum, runId }: Readonly<{
  models: ModelData[]; activeId: string; onSelect: (id: string) => void;
  activeStratum: string; runId: string | null;
}>) {
  return (
    <div className="flex shrink-0 flex-row gap-2 lg:w-52 lg:flex-col">
      <p className="hidden text-[11px] font-semibold uppercase tracking-wider text-slate-600 lg:block">Models</p>
      {models.map((m) => {
        const active = m.id === activeId;
        return (
          <button
            key={m.id}
            onClick={() => onSelect(m.id)}
            className={`flex-1 rounded-lg border px-3 py-2.5 text-left text-sm transition-all lg:flex-none ${
              active
                ? "border-cyan-400/60 bg-cyan-950/30 text-cyan-200 glow-cyan"
                : "border-slate-800 bg-slate-900/50 text-slate-400 hover:border-slate-600 hover:text-slate-200"
            }`}
          >
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full" style={{ background: m.accent, boxShadow: active ? `0 0 8px ${m.accent}` : "none" }} />
              {m.name}
            </span>
          </button>
        );
      })}
      <div className="mt-2 hidden rounded-lg border border-slate-800 bg-slate-900/40 p-3 text-[10px] leading-relaxed text-slate-500 lg:block">
        Stratum aktiv: <span className="font-mono text-slate-300">{activeStratum}</span>
        {runId && (
          <>
            <br />Run: <span className="font-mono text-slate-400">{runId.slice(0, 14)}…</span>
          </>
        )}
      </div>
    </div>
  );
}

function CalibrationLock() {
  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center rounded-xl bg-[#020617]/85 backdrop-blur-sm">
      <div className="max-w-sm rounded-xl border border-red-700/50 bg-red-950/40 p-5 text-center">
        <p className="text-3xl">🔒</p>
        <p className="mt-2 text-sm font-semibold text-red-200">{MSG_SBC_LOCK}</p>
        <p className="mt-2 text-xs text-red-300/70">
          SBC (Simulation-Based Calibration) ka statusin <b>fail</b>. Rikalibroni modelin para se të parashikoni.
        </p>
      </div>
    </div>
  );
}

function CurrentProbabilitiesCard({ model, explain }: Readonly<{ model: ModelData; explain: boolean }>) {
  const up = model.direction === "up";
  return (
    <CardShell title="Current Probabilities" explain={explain ? EXPLAIN_CURRENT : undefined}>
      <div className="flex items-end gap-2">
        <span className={`text-4xl font-bold ${up ? "text-emerald-400 neon-green" : "text-red-400 neon-rose"}`}>
          {model.probability.toFixed(1)}%
        </span>
        <span className={`mb-1 text-xl ${up ? "text-emerald-400" : "text-red-400"}`}>{up ? "↑" : "↓"}</span>
      </div>
      <div className="mt-4 space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-slate-500">Expected Return</span>
          <span className={`font-mono ${model.expectedReturn >= 0 ? "text-emerald-300" : "text-red-300"}`}>
            {model.expectedReturn >= 0 ? "+" : ""}{model.expectedReturn.toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Risk Score</span>
          <span className="font-mono text-amber-300">{model.riskScore}/100</span>
        </div>
      </div>
    </CardShell>
  );
}

function theoremLockMsg(aPrime: string, bPrime: string): string {
  const aFail = aPrime === "fail";
  const bFail = bPrime === "fail";
  if (aFail && bFail) return "Teoremat A′ dhe B′ dështuan. Shto burime të pavarura dhe kontrollo informativitetin.";
  if (aFail) return "Teorema A′ dështoi (Mungesë burimesh të pavarura). Shto më shumë burime RSS për ta zhbllokuar.";
  if (bFail) return "Teorema B′ dështoi (Informativiteti i pamjaftueshëm i burimeve). Kontrollo cilësinë e burimeve RSS.";
  return "Teorema dështoi. Kontrollo statusin e portave.";
}

function ForecastChartCard({ model, explain, theoremFail, sbcFail, aPrime, bPrime, hStar }: Readonly<{
  model: ModelData; explain: boolean; theoremFail: boolean; sbcFail: boolean;
  aPrime: string; bPrime: string; hStar: number | null;
}>) {
  return (
    <div className="card-premium relative p-4 md:col-span-1 xl:col-span-2">
      {explain && <ExplainDot {...EXPLAIN_CI} />}
      <div className="mb-3 flex items-center justify-between">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Forecast Probabilities</p>
        <div className="flex gap-3 text-[10px] text-slate-500">
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded" style={{ background: model.accent, opacity: 0.5 }} /> 80% CI</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded" style={{ background: model.accent, opacity: 0.18 }} /> 95% CI</span>
        </div>
      </div>
      <div className="relative h-44">
        <ForecastAreaChart curve={model.curve} accent={model.accent} />
        {theoremFail && !sbcFail && (
          <div className="absolute inset-0 z-30 flex items-center justify-center rounded-lg bg-[#020617]/90 backdrop-blur-sm">
            <div className="max-w-md px-4 text-center">
              <p className="text-2xl">🔒</p>
              <p className="mt-1 text-xs font-semibold leading-relaxed text-red-200">
                Parashikimi i bllokuar: {theoremLockMsg(aPrime, bPrime)}
              </p>
            </div>
          </div>
        )}
      </div>
      {hStar != null && model.id === ROUTER_ID && (
        <p className="mt-2 text-[10px] text-slate-500">
          Theorem D-forecast: h* = <span className="font-mono text-slate-300">{hStar} muaj</span>
          {" "}· horizonet përtej h* shfaqen si <span className="text-amber-400">prior-dominated</span>.
        </p>
      )}
    </div>
  );
}

function RegimeCard({ model, explain }: Readonly<{ model: ModelData; explain: boolean }>) {
  return (
    <CardShell title="Regime Distribution" explain={explain ? EXPLAIN_REGIME : undefined}>
      <div className="h-28"><RegimeAreaChart series={model.regimeSeries} /></div>
      <div className="mt-3 grid grid-cols-2 gap-1.5 text-[10px]">
        {(["bullish", "bearish", "volatile", "stable"] as const).map((k) => (
          <span key={k} className="flex items-center gap-1.5 text-slate-400">
            <span className="h-2 w-2 rounded-sm" style={{ background: REGIME_COLORS[k] }} />
            {k[0].toUpperCase() + k.slice(1)}
          </span>
        ))}
      </div>
    </CardShell>
  );
}

function RoutingCard({ model, explain, aPrime, bPrime }: Readonly<{
  model: ModelData; explain: boolean; aPrime: string; bPrime: string;
}>) {
  return (
    <CardShell title="Active Routing Status" explain={explain ? EXPLAIN_ROUTING : undefined}>
      <div className="space-y-3">
        <Bar label="David Alpha" value={model.routing.davidAlpha} color="#22d3ee" />
        <Bar label="DeepNet" value={model.routing.deepNet} color="#a78bfa" />
      </div>
      <div className="mt-4 flex gap-2 text-[10px]">
        <span className={`rounded px-2 py-0.5 ${gateChipClass(aPrime)}`}>A′ {aPrime}</span>
        <span className={`rounded px-2 py-0.5 ${gateChipClass(bPrime)}`}>B′ {bPrime}</span>
      </div>
    </CardShell>
  );
}

function IndicatorsCard({ model }: Readonly<{ model: ModelData }>) {
  return (
    <CardShell title="Key Indicators">
      <div className="space-y-3">
        <Bar label="Volume" value={model.indicators.volume} color="#34d399" />
        <Bar label="VIX" value={model.indicators.vix} color="#fbbf24" />
      </div>
      <p className="mt-3 text-[10px] leading-relaxed text-slate-600">
        Treguesit e shëndetit përmbledhin vëllimin e provave dhe paqëndrueshmërinë e regjimit.
      </p>
    </CardShell>
  );
}

export function ForecastRouter(props: Readonly<RouterDataProps>) {
  const [activeId, setActiveId] = useState(ROUTER_ID);
  const [explain, setExplain] = useState(false);

  const models = useMemo(
    () => buildModels(props.realCurve, props.realProbability),
    [props.realCurve, props.realProbability],
  );
  const model = models.find((m) => m.id === activeId) ?? models[0];
  // Certified only when live data confirms explicit passes — skip/unknown gates are not certified.
  const certified = !props.usingMock && !props.sbcFail && props.aPrime === "pass" && props.bPrime === "pass";

  return (
    <div className="relative space-y-6">
      <RouterHeader
        updatedAgo={props.updatedAgo}
        usingMock={props.usingMock}
        explain={explain}
        onToggle={() => setExplain((v) => !v)}
      />
      <TrustBanner certified={certified} sbcFail={props.sbcFail} usingMock={props.usingMock} />

      <div className="flex flex-col gap-5 lg:flex-row">
        <ModelSelector
          models={models}
          activeId={activeId}
          onSelect={setActiveId}
          activeStratum={props.activeStratum}
          runId={props.runId}
        />

        <div className="relative grid flex-1 grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {props.sbcFail && <CalibrationLock />}
          <CurrentProbabilitiesCard model={model} explain={explain} />
          <ForecastChartCard
            model={model}
            explain={explain}
            theoremFail={props.theoremFail}
            sbcFail={props.sbcFail}
            aPrime={props.aPrime}
            bPrime={props.bPrime}
            hStar={props.hStar}
          />
          <RegimeCard model={model} explain={explain} />
          <RoutingCard model={model} explain={explain} aPrime={props.aPrime} bPrime={props.bPrime} />
          <IndicatorsCard model={model} />
        </div>
      </div>
    </div>
  );
}
