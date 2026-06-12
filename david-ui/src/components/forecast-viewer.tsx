"use client";

import { useEffect, useState } from "react";
import { api, ForecastCell, ConfigValues } from "@/lib/api";
import {
  RouteState,
  fromForecastCell,
  routeAccent,
  assertNever,
} from "@/lib/route-state";

type Taxonomy = { domain: string; tactics: Array<{ id: string; name: string; question: string; hint: string }> };

// ── Cell card renderer — exhaustive switch over RouteState ────────────────────

function CellCard({
  routeState,
  tacticName,
  tacticId,
}: Readonly<{ routeState: RouteState; tacticName: string; tacticId: string }>) {
  const accent = routeAccent(routeState);

  const header = (
    <div className="flex justify-between items-start mb-3">
      <div>
        <h3 className="text-sm font-bold text-slate-200">{tacticName}</h3>
        <p className="text-xs text-slate-500 font-mono mt-0.5">{tacticId}</p>
      </div>
      <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${accent.bg} ${accent.text} border ${accent.border}`}>
        {accent.label}
      </span>
    </div>
  );

  switch (routeState.tag) {
    case "headline": {
      const isHighRisk = routeState.p_active > 0.5;
      return (
        <div className={`p-4 rounded-xl border backdrop-blur-sm ${accent.bg} ${accent.border}`}>
          {header}
          <div className="flex items-end gap-2 mb-4">
            <span className={`text-3xl font-bold tabular-nums ${isHighRisk ? "text-rose-300" : "text-emerald-300"}`}>
              {(routeState.p_active * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-slate-500 mb-1">p(active)</span>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">80% CI</span>
              <span className="font-mono text-slate-300 tabular-nums">
                {(routeState.ci_80_lo * 100).toFixed(0)}–{(routeState.ci_80_hi * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">h*(g)</span>
              <span className="font-mono text-slate-300 tabular-nums">{routeState.h_star_months} mo</span>
            </div>
            {routeState.fdp_binding_reason && (
              <div className="pt-3 border-t border-slate-800/50 flex justify-between items-center">
                <span className="text-[10px] uppercase tracking-wider text-slate-500">FDP</span>
                <span className={`text-[10px] font-mono ${routeState.flagged_by_fdp ? "text-rose-400" : "text-slate-500"}`}>
                  {routeState.fdp_binding_reason.replace(/_/g, " ")}
                </span>
              </div>
            )}
          </div>
        </div>
      );
    }

    case "horizon_prior_dominated":
      return (
        <div className={`p-4 rounded-xl border backdrop-blur-sm ${accent.bg} ${accent.border}`}>
          {header}
          <div className="space-y-2 text-xs mt-2">
            <div className="flex justify-between">
              <span className="text-slate-500">Horizon</span>
              <span className="font-mono text-slate-300 tabular-nums">{routeState.horizon_months} mo</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">h*(g)</span>
              <span className="font-mono text-fuchsia-300 tabular-nums">{routeState.h_star_months} mo</span>
            </div>
          </div>
          <GateReasonList reasons={routeState.reasons} />
        </div>
      );

    case "monitor_only":
      return (
        <div className={`p-4 rounded-xl border backdrop-blur-sm ${accent.bg} ${accent.border}`}>
          {header}
          <p className="text-xs text-slate-400 mt-2">
            Endogenous observability λ-interval too wide — no probability claim authorised.
          </p>
          <GateReasonList reasons={routeState.reasons} />
        </div>
      );

    case "aggregate_only":
      return (
        <div className={`p-4 rounded-xl border backdrop-blur-sm ${accent.bg} ${accent.border}`}>
          {header}
          <p className="text-xs text-slate-400 mt-2">
            Per-tactic resolution insufficient — stratum aggregate only.
          </p>
          <GateReasonList reasons={routeState.reasons} />
        </div>
      );

    case "evidence_gap":
      return (
        <div className={`p-4 rounded-xl border backdrop-blur-sm ${accent.bg} ${accent.border}`}>
          {header}
          <p className="text-xs text-slate-400 mt-2">
            Identification distance below floor, or MCSE exceeds gate margin.
          </p>
          <GateReasonList reasons={routeState.reasons} />
        </div>
      );

    case "withhold":
      return (
        <div className={`p-4 rounded-xl border backdrop-blur-sm ${accent.bg} ${accent.border}`}>
          {header}
          <p className="text-xs text-red-300/70 mt-2">
            Hard gate failure — forecast withheld.
          </p>
          <GateReasonList reasons={routeState.reasons} />
        </div>
      );

    case "prior_dominated":
      return (
        <div className={`p-4 rounded-xl border backdrop-blur-sm ${accent.bg} ${accent.border}`}>
          {header}
          <p className="text-xs text-slate-400 mt-2">
            I(O) lower 95% CI below informativeness floor — prior dominates evidence.
          </p>
          <GateReasonList reasons={routeState.reasons} />
        </div>
      );

    default:
      return assertNever(routeState);
  }
}

function GateReasonList({ reasons }: Readonly<{ reasons: string[] }>) {
  if (reasons.length === 0) return null;
  return (
    <ul className="mt-3 pt-3 border-t border-slate-800/50 space-y-0.5">
      {reasons.map((r, i) => (
        <li key={i} className="text-[10px] font-mono text-slate-500">{r}</li>
      ))}
    </ul>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export function ForecastViewer() {
  const [cells, setCells] = useState<ForecastCell[]>([]);
  const [taxonomy, setTaxonomy] = useState<Taxonomy | null>(null);
  const [config, setConfig] = useState<ConfigValues | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedStratum, setSelectedStratum] = useState<string>("");
  const [selectedHorizon, setSelectedHorizon] = useState<number>(6);

  useEffect(() => {
    async function fetchData() {
      try {
        const [taxRes, confRes, forecastRes] = await Promise.all([
          api.taxonomy(),
          api.config(),
          api.forecastCells(),
        ]);

        setTaxonomy(taxRes);
        setConfig(confRes);

        setCells(forecastRes.forecast_cells || []);
        if (forecastRes.forecast_cells?.length > 0) {
          setSelectedStratum(forecastRes.forecast_cells[0].stratum_id);
        }

        if (confRes?.forecast_horizons_months?.length > 0) {
          if (!confRes.forecast_horizons_months.includes(selectedHorizon)) {
            setSelectedHorizon(confRes.forecast_horizons_months[0]);
          }
        }
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) return <div className="text-slate-400 animate-pulse">Initializing Observatory Link...</div>;
  if (error) return <div className="text-rose-400 p-4 border border-rose-900/50 bg-rose-950/20 rounded-md">Error: {error}</div>;
  if (cells.length === 0) return <div className="text-slate-400">No active forecasts found.</div>;

  // Stan assigns integer indices by sorting tactic_k values alphabetically
  // (via sorted(set(tactic_k)) in fit.py). Mirror that sort here before the
  // 1-based index lookup — NOT the insertion order from /config.
  const getTacticDetails = (tacticIndex: number) => {
    if (!config || !taxonomy) return { id: `Tactic ${tacticIndex}`, name: `Unknown Tactic ${tacticIndex}` };
    const sortedTactics = [...config.tactic_classes].sort();
    const tacticId = sortedTactics[tacticIndex - 1];
    const taxObj = taxonomy.tactics.find((t) => t.id === tacticId);
    return taxObj || { id: tacticId ?? `tactic_${tacticIndex}`, name: tacticId ?? `Tactic ${tacticIndex}` };
  };

  const strata = Array.from(new Set(cells.map((c) => c.stratum_id))).sort();
  const horizons = config?.forecast_horizons_months || [];

  const filteredCells = cells.filter(
    (c) => c.stratum_id === selectedStratum && c.horizon_months === selectedHorizon
  );

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center bg-slate-900/50 p-4 rounded-lg border border-slate-800/50">
        <div>
          <label className="block text-xs text-slate-500 font-medium mb-1 uppercase tracking-wider">Region / Stratum</label>
          <select
            className="bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            value={selectedStratum}
            onChange={(e) => setSelectedStratum(e.target.value)}
          >
            {strata.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-slate-500 font-medium mb-1 uppercase tracking-wider">Forecast Horizon</label>
          <select
            className="bg-slate-950 border border-slate-800 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            value={selectedHorizon}
            onChange={(e) => setSelectedHorizon(Number(e.target.value))}
          >
            {horizons.map((h) => (
              <option key={h} value={h}>{h} Months</option>
            ))}
          </select>
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredCells.length === 0 ? (
          <div className="col-span-full text-slate-500 text-sm py-8 text-center">
            No predictions available for this stratum and horizon.
          </div>
        ) : (
          filteredCells
            .sort((a, b) => b.p_active - a.p_active)
            .map((cell, idx) => {
              const tactic = getTacticDetails(cell.tactic);
              const routeState: RouteState = fromForecastCell(cell);
              return (
                <CellCard
                  key={idx}
                  routeState={routeState}
                  tacticName={tactic.name}
                  tacticId={tactic.id}
                />
              );
            })
        )}
      </div>
    </div>
  );
}
