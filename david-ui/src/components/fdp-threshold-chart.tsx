"use client";

/**
 * E-5: FDP threshold visualization.
 *
 * Renders the posterior expected FDP threshold chart from the route_ledger
 * FDP block (C-7). The chart shows:
 *   - Sorted p_{(j)} bar chart (headline-eligible cells, descending)
 *   - Horizontal cutline at t*(q) = threshold_p
 *   - Running expected FDP curve e_m vs q = 0.10 reference line
 *   - Flagged prefix highlighted (top m* cells)
 *   - Metadata badges: worst-point θ index and FG6 exceedance fraction
 *
 * INVARIANT: m* is NEVER recomputed on the client. Only the t_star_q and
 * m_star values from the posterior_fdp block are used for classification.
 * The running e_m curve is computed for display only (pure visualization).
 */

import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  ReferenceLine,
  ComposedChart,
  Line,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type {
  FdpLedger,
  PosteriorFdpBlock,
  ExceedanceGateBlock,
  SensitivityEnvelopeBlock,
} from "@/lib/route-state";

// ── Types ──────────────────────────────────────────────────────────────────────

/** Minimal cell shape needed for the FDP chart (from route_ledger.cells). */
export interface FdpCell {
  p_active: number;
  /** C-7 field — present on all headline-eligible cells. */
  fdp_binding_reason?: string;
}

export interface FdpThresholdChartProps {
  /** posterior_fdp block from route_ledger.json */
  posteriorFdp: PosteriorFdpBlock;
  /** exceedance_gate block from route_ledger.json */
  exceedanceGate: ExceedanceGateBlock;
  /** mcse_floor block from route_ledger.json (used for badge only) */
  mcseFloor: FdpLedger["mcse_floor"];
  /** sensitivity_envelope block from route_ledger.json */
  sensitivityEnvelope: SensitivityEnvelopeBlock;
  /** Headline-eligible cells (those that survived FG1–FG5 + MCSE floor). */
  cells: FdpCell[];
}

// ── Chart data builders ────────────────────────────────────────────────────────

interface BarDatum {
  rank: number;           // 1-based sort rank
  p: number;              // p_{(j)}
  flagged: boolean;       // in the top m* prefix
}

interface CurveDatum {
  m: number;              // prefix size
  e_m: number;            // expected FDP at prefix m
  q_ref: number;          // q = 0.10 constant reference
}

function buildBarData(cells: FdpCell[], mStar: number): BarDatum[] {
  const sorted = [...cells].sort((a, b) => b.p_active - a.p_active);
  return sorted.map((c, i) => ({
    rank: i + 1,
    p: c.p_active,
    flagged: i < mStar,
  }));
}

function buildCurveData(barData: BarDatum[], q: number): CurveDatum[] {
  let cumFp = 0;
  return barData.map((d, i) => {
    cumFp += 1 - d.p;
    return {
      m: d.rank,
      e_m: cumFp / (i + 1),
      q_ref: q,
    };
  });
}

// ── Custom tooltip ─────────────────────────────────────────────────────────────

function BarTip({ active, payload }: Readonly<{ active?: boolean; payload?: Array<{ payload: BarDatum }> }>) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-[11px] font-mono shadow-xl">
      <div className="text-slate-400">rank <span className="text-slate-200">{d.rank}</span></div>
      <div className="text-slate-400">p<span className="align-sub text-[9px]">({d.rank})</span> = <span className="text-cyan-300">{d.p.toFixed(4)}</span></div>
      <div className={d.flagged ? "text-rose-300" : "text-slate-500"}>{d.flagged ? "flagged (m* prefix)" : "below threshold"}</div>
    </div>
  );
}

function CurveTip({ active, payload }: Readonly<{ active?: boolean; payload?: Array<{ payload: CurveDatum }> }>) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-[11px] font-mono shadow-xl">
      <div className="text-slate-400">m = <span className="text-slate-200">{d.m}</span></div>
      <div className="text-slate-400">ê[FDP] = <span className="text-amber-300">{d.e_m.toFixed(4)}</span></div>
      <div className="text-slate-500">q = {d.q_ref.toFixed(2)}</div>
    </div>
  );
}

// ── Metadata badges ────────────────────────────────────────────────────────────

function Badge({ label, value, accent }: Readonly<{ label: string; value: string; accent: string }>) {
  return (
    <div className="flex flex-col gap-0.5 min-w-[100px]">
      <span className="text-[9px] uppercase tracking-widest text-slate-500 font-medium">{label}</span>
      <span className={`text-sm font-bold tabular-nums font-mono ${accent}`}>{value}</span>
    </div>
  );
}

function exceedanceBadge(gate: ExceedanceGateBlock) {
  if (gate.skipped) return { value: "—", accent: "text-slate-500", title: gate.reason };
  const frac = gate.exceedance_fraction ?? gate.exceedance_fraction_at_accepted;
  const pass = frac <= gate.alpha;
  return {
    value: frac.toFixed(3),
    accent: pass ? "text-emerald-300" : "text-rose-300",
    title: `FG6 exceedance: P̂(FDP>${gate.gamma}) = ${frac.toFixed(3)} (α=${gate.alpha})`,
  };
}

function envelopeBadge(env: SensitivityEnvelopeBlock) {
  if (env.skipped || env.worst_theta_index == null) {
    return { value: "—", accent: "text-slate-500", title: "sensitivity envelope not available" };
  }
  return {
    value: `θ[${env.worst_theta_index}]`,
    accent: "text-fuchsia-300",
    title: `Conservative θ grid index = ${env.worst_theta_index}; envelope m* = ${env.n_flagged_envelope}`,
  };
}

// ── Main component ─────────────────────────────────────────────────────────────

export function FdpThresholdChart({
  posteriorFdp,
  exceedanceGate,
  mcseFloor,
  sensitivityEnvelope,
  cells,
}: Readonly<FdpThresholdChartProps>) {
  if (cells.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 rounded-xl border border-slate-800 bg-slate-900/40 text-slate-500 text-sm">
        No headline-eligible cells — FDP chart unavailable.
      </div>
    );
  }

  const { t_star_q, m_star, q_target, posterior_expected_fdp_at_threshold, n_cells } = posteriorFdp;
  const barData = buildBarData(cells, m_star);
  const curveData = buildCurveData(barData, q_target);

  const exc = exceedanceBadge(exceedanceGate);
  const env = envelopeBadge(sensitivityEnvelope);
  const mcseExcluded = !mcseFloor.skipped ? mcseFloor.n_cells_excluded : null;

  return (
    <div className="rounded-xl border border-slate-800 bg-[#070b19]/90 p-5 space-y-5">

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            Posterior Expected FDP — Theorem C
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">
            Sorted p<span className="align-sub text-[8px]">(j)</span> · t*(q = {q_target.toFixed(2)}) cutline · running ê[FDP]
          </p>
        </div>

        {/* Metadata badges */}
        <div className="flex flex-wrap gap-5 items-start">
          <Badge label="m*" value={String(m_star)} accent="text-cyan-300" />
          <Badge label="M eligible" value={String(n_cells)} accent="text-slate-300" />
          <Badge
            label="ê[FDP] at t*"
            value={posterior_expected_fdp_at_threshold.toFixed(4)}
            accent={posterior_expected_fdp_at_threshold <= q_target ? "text-emerald-300" : "text-rose-300"}
          />
          <div title={exc.title}>
            <Badge label="FG6 P̂(FDP>γ)" value={exc.value} accent={exc.accent} />
          </div>
          <div title={env.title}>
            <Badge label="Envelope θ" value={env.value} accent={env.accent} />
          </div>
          {mcseExcluded != null && mcseExcluded > 0 && (
            <Badge label="MCSE excl." value={String(mcseExcluded)} accent="text-amber-300" />
          )}
        </div>
      </div>

      {/* Sorted p_{(j)} bar chart */}
      <div>
        <p className="text-[9px] uppercase tracking-widest text-slate-600 mb-2">
          Sorted marginals p<span className="align-sub text-[7px]">(j)</span> — flagged prefix highlighted
        </p>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 4, right: 4, left: 0, bottom: 4 }} barCategoryGap="10%">
              <XAxis
                dataKey="rank"
                tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
                axisLine={{ stroke: "#1e293b" }}
                tickLine={false}
                label={{ value: "rank j", position: "insideBottomRight", fill: "#475569", fontSize: 9, dy: 4 }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
                axisLine={{ stroke: "#1e293b" }}
                tickLine={false}
                tickCount={5}
                tickFormatter={(v: number) => v.toFixed(1)}
                width={28}
              />
              <Tooltip content={<BarTip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              {/* t*(q) cutline */}
              <ReferenceLine
                y={t_star_q}
                stroke="#f59e0b"
                strokeDasharray="4 3"
                strokeWidth={1.5}
                label={{
                  value: `t*=${t_star_q.toFixed(3)}`,
                  position: "insideTopRight",
                  fill: "#f59e0b",
                  fontSize: 9,
                  fontFamily: "monospace",
                }}
              />
              <Bar dataKey="p" isAnimationActive={false} radius={[2, 2, 0, 0]}>
                {barData.map((d) => (
                  <Cell
                    key={d.rank}
                    fill={d.flagged ? "#f43f5e" : "#334155"}
                    fillOpacity={d.flagged ? 0.85 : 0.6}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="flex gap-4 mt-1.5 text-[9px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm" style={{ background: "#f43f5e", opacity: 0.85 }} />
            flagged (m* = {m_star} cells)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm" style={{ background: "#334155", opacity: 0.7 }} />
            below threshold
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-5 h-0 border-t border-dashed border-amber-500" />
            t*(q) = {t_star_q.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Running ê[FDP] curve vs q = 0.10 */}
      <div>
        <p className="text-[9px] uppercase tracking-widest text-slate-600 mb-2">
          Running ê[FDP<span className="align-sub text-[7px]">m</span>] = Σ(1−p<span className="align-sub text-[7px]">(j)</span>) / m
        </p>
        <div className="h-40">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={curveData} margin={{ top: 4, right: 4, left: 0, bottom: 4 }}>
              <XAxis
                dataKey="m"
                tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
                axisLine={{ stroke: "#1e293b" }}
                tickLine={false}
                label={{ value: "m", position: "insideBottomRight", fill: "#475569", fontSize: 9, dy: 4 }}
              />
              <YAxis
                domain={[0, Math.min(1, Math.max(q_target * 2.5, 0.25))]}
                tick={{ fill: "#475569", fontSize: 9, fontFamily: "monospace" }}
                axisLine={{ stroke: "#1e293b" }}
                tickLine={false}
                tickCount={5}
                tickFormatter={(v: number) => v.toFixed(2)}
                width={32}
              />
              <Tooltip content={<CurveTip />} cursor={{ stroke: "rgba(255,255,255,0.08)", strokeWidth: 1 }} />
              {/* q reference line */}
              <ReferenceLine
                y={q_target}
                stroke="#22d3ee"
                strokeDasharray="3 4"
                strokeWidth={1}
                label={{
                  value: `q=${q_target.toFixed(2)}`,
                  position: "insideTopRight",
                  fill: "#22d3ee",
                  fontSize: 9,
                  fontFamily: "monospace",
                }}
              />
              {/* m* vertical cutoff */}
              {m_star > 0 && (
                <ReferenceLine
                  x={m_star}
                  stroke="#f43f5e"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                  label={{
                    value: `m*=${m_star}`,
                    position: "insideTopLeft",
                    fill: "#f43f5e",
                    fontSize: 9,
                    fontFamily: "monospace",
                  }}
                />
              )}
              <Line
                dataKey="e_m"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                type="monotone"
              />
              <Line
                dataKey="q_ref"
                stroke="#22d3ee"
                strokeWidth={1}
                dot={false}
                strokeDasharray="3 4"
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="flex gap-4 mt-1.5 text-[9px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-5 h-0 border-t-2 border-amber-500" />
            ê[FDP<span className="align-sub text-[7px]">m</span>]
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-5 h-0 border-t border-dashed border-cyan-400" />
            q = {q_target.toFixed(2)} gate
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-5 h-0 border-t border-dashed border-rose-500" />
            m* cutoff
          </span>
        </div>
      </div>
    </div>
  );
}
