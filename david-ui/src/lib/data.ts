/**
 * E-6: Server-only data layer.
 *
 * All reads of run artifacts (route_ledger.json, forecast cells, fit summaries)
 * happen here. `server-only` prevents this module from being bundled into any
 * client component. `React.cache()` deduplicates identical calls within a
 * single request — pages that read the same run_id independently pay only one
 * fetch.
 *
 * Mutation surface: zero. This module is read-only.
 * The only permitted mutation (E-8) is the approve-button → /api/forecasts/approve route.
 */

import "server-only";
import { cache } from "react";
import { api } from "./api";
import type {
  FitRun,
  FitSummary,
  ForecastCell,
  Stratum,
  Queue,
  SbcResult,
  AutomationStatus,
  SourceIndependence,
} from "./api";
import type { FdpLedger } from "./route-state";

// ── Re-export types consumed by server components ──────────────────────────────
export type { FitRun, FitSummary, ForecastCell, Stratum, Queue, SbcResult };

// ── Cached readers ─────────────────────────────────────────────────────────────

export const getFitRuns = cache(async (limit = 10) => {
  try {
    const res = await api.fitRuns(limit);
    return res.fit_runs ?? [];
  } catch {
    return [];
  }
});

export const getLatestFitRun = cache(async (): Promise<FitRun | null> => {
  const runs = await getFitRuns(1);
  return runs[0] ?? null;
});

export const getFitSummary = cache(async (run_id: string): Promise<FitSummary | null> => {
  try {
    return await api.fitRun(run_id);
  } catch {
    return null;
  }
});

export const getForecastCells = cache(async (run_id?: string, horizon?: number) => {
  try {
    return await api.forecastCells(run_id, horizon);
  } catch {
    return { forecast_cells: [] as ForecastCell[], n: 0, run_id: "" };
  }
});

export const getRouteLedger = cache(async (run_id: string) => {
  try {
    return await api.routeLedger(run_id) as FdpLedger & {
      route_counts: Record<string, number>;
      m_claim_eligible: number;
      n_cells_total: number;
      cells: ForecastCell[];
      timestamp: string;
    };
  } catch {
    return null;
  }
});

export const getStrata = cache(async (): Promise<Stratum[]> => {
  try {
    const res = await api.strata();
    return res.strata ?? [];
  } catch {
    return [];
  }
});

export const getQueue = cache(async (): Promise<Queue | null> => {
  try {
    return await api.queue();
  } catch {
    return null;
  }
});

export const getSbc = cache(async (): Promise<{ measurement?: SbcResult; forecast?: SbcResult } | null> => {
  try {
    return await api.sbc();
  } catch {
    return null;
  }
});

export const getAutomationStatus = cache(async (): Promise<AutomationStatus | null> => {
  try {
    return await api.automationStatus();
  } catch {
    return null;
  }
});

export const getSourceIndependence = cache(async (): Promise<SourceIndependence | null> => {
  try {
    return await api.sourceIndependence();
  } catch {
    return null;
  }
});

export const getConfig = cache(async () => {
  try {
    return await api.config();
  } catch {
    return null;
  }
});

/** Pipeline live status — not cached (intentionally no-store). */
export async function getPipelineStatus(): Promise<{ running: boolean; step?: string; started_at?: string | null }> {
  try {
    const res = await fetch(
      `${process.env.DAVID_API_URL ?? "http://localhost:8080"}/internal/pipeline/status`,
      {
        headers: process.env.PIPELINE_SECRET
          ? { Authorization: `Bearer ${process.env.PIPELINE_SECRET}` }
          : {},
        cache: "no-store",
      }
    );
    return res.json();
  } catch {
    return { running: false };
  }
}
