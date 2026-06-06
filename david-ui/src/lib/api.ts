/** Typed API client — fetches from FastAPI via Next.js rewrite /api/* */

// Server components: call Railway directly (rewrites are browser-only).
// Browser / client components: use the /api rewrite to stay same-origin.
const BASE =
  typeof window === "undefined"
    ? (process.env.DAVID_API_URL ?? "http://localhost:8080")
    : "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate: 30 } });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ── types ──────────────────────────────────────────────────────────────────

export type GateStatus = "pass" | "fail" | "skip" | "error";

export interface FitRun {
  run_id: string;
  model_version: string;
  gate_status: GateStatus;
  rhat_max: number;
  ess_bulk_min: number;
  divergences: number;
  n_strata: number;
  n_labels: number;
  started_at: string;
}

export interface FitSummary extends FitRun {
  theorems?: Record<string, Record<string, unknown>>;
  gates?: Record<string, unknown>;
  reason?: string;
}

export interface ForecastCell {
  series: number;
  tactic: number;
  horizon_months: number;
  p_active: number;
  ci_80_lo: number;
  ci_80_hi: number;
  ci_95_lo: number;
  ci_95_hi: number;
  h_star_months: number;
  below_h_star: boolean;
  forecast_route: string;
  stratum_id: string;
  emitted_at: string;
}

export interface EvidenceItem {
  evidence_id: string;
  stratum_id: string;
  source_id: string;
  evidence_date: string;
  title: string | null;
  url: string | null;
  adjudicated: boolean;
}

export interface Stratum {
  stratum_id: string;
  country: string;
  policy: string;
  observability: number;
  n_evidence: number;
  n_adjudicated: number;
}

export interface QueueItem {
  evidence_id: string;
  reason: string;
  eig_score: number;
  estimated_minutes: number;
  worst_tactic: string;
  minority_share: number;
}

export interface Queue {
  generated_at: string | null;
  n_in_queue: number;
  disagreement_threshold: number;
  items: QueueItem[];
}

export interface ConfigValues {
  model_version: string;
  tactic_classes: string[];
  forecast_horizons_months: number[];
  theorem_floors: Record<string, number>;
  fit_thresholds: Record<string, number>;
  sbc: { ks_alpha: number; bonferroni: boolean };
  human_loop: { disagreement_threshold: number; gold_sample_rate: number };
}

export interface SbcResult {
  gate_status: GateStatus;
  n_worlds?: number;
  ks_p_min?: number;
  reason?: string;
}

// ── endpoints ──────────────────────────────────────────────────────────────

export const api = {
  health:        () => get<{ status: string }>("/healthz"),
  fitRuns:       (limit = 10) => get<{ fit_runs: FitRun[]; n: number }>(`/fit-runs?limit=${limit}`),
  fitRun:        (id: string) => get<FitSummary>(`/fit-runs/${id}`),
  strata:        () => get<{ strata: Stratum[]; n: number }>("/strata"),
  evidence:      (p: { stratum_id?: string; adjudicated?: boolean; limit?: number; offset?: number }) => {
    const q = new URLSearchParams();
    if (p.stratum_id)              q.set("stratum_id", p.stratum_id);
    if (p.adjudicated !== undefined) q.set("adjudicated", String(p.adjudicated));
    if (p.limit)                   q.set("limit", String(p.limit));
    if (p.offset)                  q.set("offset", String(p.offset));
    return get<{ evidence: EvidenceItem[]; n: number; total: number }>(`/evidence?${q}`);
  },
  forecastCells: (run_id?: string, horizon?: number) => {
    const q = new URLSearchParams();
    if (run_id)  q.set("run_id", run_id);
    if (horizon) q.set("horizon_months", String(horizon));
    return get<{ forecast_cells: ForecastCell[]; n: number; run_id: string }>(`/forecast-cells?${q}`);
  },
  queue:         () => get<Queue>("/queue"),
  sbc:           () => get<{ measurement?: SbcResult; forecast?: SbcResult }>("/sbc"),
  config:        () => get<ConfigValues>("/config"),
};

// ── SWR fetcher ────────────────────────────────────────────────────────────
export const fetcher = (url: string) =>
  fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${r.status}`);
    return r.json();
  });
