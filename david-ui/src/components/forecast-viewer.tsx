"use client";

import { useEffect, useState } from "react";
import { api, ForecastCell, ConfigValues } from "@/lib/api";

type Taxonomy = { domain: string; tactics: Array<{ id: string; name: string; question: string; hint: string }> };

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
        
        // Use the default API response or error string
        if (forecastRes.detail) {
          setError(forecastRes.detail);
        } else {
          setCells(forecastRes.forecast_cells || []);
          if (forecastRes.forecast_cells?.length > 0) {
            setSelectedStratum(forecastRes.forecast_cells[0].stratum_id);
          }
        }
        
        // Sync default horizon to whatever the first config item is if 6 isn't available
        if (confRes?.forecast_horizons_months?.length > 0) {
          if (!confRes.forecast_horizons_months.includes(selectedHorizon)) {
            setSelectedHorizon(confRes.forecast_horizons_months[0]);
          }
        }
      } catch (err: any) {
        setError(err.message);
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
  // (via sorted(set(tactic_k)) in fit.py). We must mirror that sort here
  // before doing the 1-based index lookup — NOT the insertion order from /config.
  const getTacticDetails = (tacticIndex: number) => {
    if (!config || !taxonomy) return { id: `Tactic ${tacticIndex}`, name: `Unknown Tactic ${tacticIndex}` };
    const sortedTactics = [...config.tactic_classes].sort(); // mirror Stan's sorted(set(...))
    const tacticId = sortedTactics[tacticIndex - 1];         // Stan uses 1-based indexing
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
          <div className="col-span-full text-slate-500 text-sm py-8 text-center">No predictions available for this stratum and horizon.</div>
        ) : (
          filteredCells
            .sort((a, b) => b.p_active - a.p_active) // Sort by probability descending
            .map((cell, idx) => {
              const tactic = getTacticDetails(cell.tactic);
              const isHighRisk = cell.p_active > 0.5;
              
              let routeColor = "text-slate-400";
              let routeBg = "bg-slate-900/40 border-slate-800";
              if (cell.forecast_route === "warning_issued") {
                routeColor = "text-rose-400";
                routeBg = "bg-rose-950/20 border-rose-900/50";
              } else if (cell.forecast_route === "conditional_forecast" || cell.forecast_route === "horizon_prior_dominated") {
                routeColor = "text-indigo-400";
                routeBg = "bg-indigo-950/20 border-indigo-900/50";
              } else if (!cell.forecast_route) {
                routeColor = "text-amber-400";
                routeBg = "bg-amber-950/20 border-amber-900/50";
              }

              return (
                <div key={idx} className={`p-4 rounded-xl border backdrop-blur-sm transition-all duration-300 ${routeBg}`}>
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="text-sm font-bold text-slate-200">{tactic.name}</h3>
                      <p className="text-xs text-slate-500 font-mono mt-0.5">{tactic.id}</p>
                    </div>
                    <div className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider ${isHighRisk ? 'bg-rose-500/20 text-rose-300' : 'bg-slate-800 text-slate-300'}`}>
                      {(cell.p_active * 100).toFixed(1)}% Risk
                    </div>
                  </div>
                  
                  <div className="space-y-2 mt-4">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">Critical Threshold (H*)</span>
                      <span className="text-slate-300 font-mono">{cell.h_star_months} mos</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">80% Credible Interval</span>
                      <span className="text-slate-300 font-mono">
                        {(cell.ci_80_lo * 100).toFixed(0)}% – {(cell.ci_80_hi * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="mt-4 pt-3 border-t border-slate-800/50 flex justify-between items-center">
                      <span className="text-[10px] uppercase tracking-wider text-slate-500">Route Status</span>
                      <span className={`text-[10px] font-mono ${routeColor}`}>
                        {cell.forecast_route || "pending_routing"}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })
        )}
      </div>
    </div>
  );
}
