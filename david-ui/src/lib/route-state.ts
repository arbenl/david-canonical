/**
 * E-1: Route-typed rendering contract.
 *
 * The RouteState discriminated union is the single claim-firewall for all
 * chart rendering decisions. No component may accept raw numbers without a
 * route wrapper. TypeScript enforces exhaustiveness at compile time — add a
 * variant here and every consumer gets a tsc error until it handles the case.
 *
 * Routes mirror router.py ROUTES exactly; any divergence is a compile-time
 * error on the fromForecastCell() switch default.
 */

import type { ForecastCell } from "@/lib/api";

// ── Route tag ──────────────────────────────────────────────────────────────────

export type RouteTag =
  | "headline"
  | "monitor_only"
  | "aggregate_only"
  | "evidence_gap"
  | "withhold"
  | "horizon_prior_dominated"
  | "prior_dominated";

// ── Discriminated union ────────────────────────────────────────────────────────

/**
 * headline              — passed FG1–FG6 + FDP threshold; full claims authorised
 * monitor_only          — FG6 λ-width too wide; observe but no probability claim
 * aggregate_only        — insufficient per-tactic resolution; stratum aggregate only
 * evidence_gap          — FG2 d_θ below floor, or C-6 MCSE excluded
 * withhold              — FG1/FG4 hard failure; no forecast output at all
 * horizon_prior_dominated — FG5 h > h*(g); D-forecast tail suppressed
 * prior_dominated       — FG3 I(O) below floor; prior overwhelms evidence
 */
export type RouteState =
  | {
      tag: "headline";
      p_active: number;
      ci_80_lo: number;
      ci_80_hi: number;
      ci_95_lo: number;
      ci_95_hi: number;
      h_star_months: number;
      horizon_months: number;
      flagged_by_fdp: boolean;
      fdp_binding_reason: string | null;
    }
  | { tag: "monitor_only"; reasons: string[] }
  | { tag: "aggregate_only"; reasons: string[] }
  | { tag: "evidence_gap"; reasons: string[] }
  | { tag: "withhold"; reasons: string[] }
  | {
      tag: "horizon_prior_dominated";
      reasons: string[];
      h_star_months: number;
      horizon_months: number;
    }
  | { tag: "prior_dominated"; reasons: string[] };

// ── Exhaustiveness guard ───────────────────────────────────────────────────────

/** Ensures every switch over RouteState.tag is exhaustive. */
export function assertNever(x: never): never {
  throw new Error(`Unhandled RouteState tag: ${JSON.stringify(x)}`);
}

// ── FDP ledger types ───────────────────────────────────────────────────────────

/** posterior_fdp block from route_ledger.json (C-7). */
export interface PosteriorFdpBlock {
  q_target: number;
  n_cells: number;
  n_flagged: number;
  threshold_p: number;
  posterior_expected_fdp_at_threshold: number;
  flagged_indices: number[];
  /** Thesis field alias for threshold_p. */
  t_star_q: number;
  /** Thesis field alias for n_flagged. */
  m_star: number;
}

/** exceedance_gate block from route_ledger.json (C-7 / C-4). */
export type ExceedanceGateBlock =
  | { skipped: true; reason: string }
  | {
      skipped?: false;
      m_star_expectation: number;
      m_accepted: number;
      exceedance_fraction_at_accepted: number;
      /** C-7 alias for exceedance_fraction_at_accepted. */
      exceedance_fraction: number;
      gamma: number;
      alpha: number;
      markov_fallback_used: boolean;
    };

/** mcse_floor block from route_ledger.json (C-6). */
export type McseFloorBlock =
  | { skipped: true; reason: string }
  | {
      skipped?: false;
      n_cells_checked: number;
      n_cells_excluded: number;
      excluded_indices: number[];
      bulk_ess: number[];
      mcse: number[];
      mcse_threshold: number;
      binding_reason: string;
    };

/** sensitivity_envelope block from route_ledger.json (C-3). */
export type SensitivityEnvelopeBlock =
  | { skipped: true; reason: string; worst_theta_index?: null }
  | {
      skipped?: false;
      worst_theta_index: number | null;
      n_flagged_envelope: number;
      threshold_p_envelope: number;
      posterior_expected_fdp_envelope: number;
    };

/** The full FDP section from route_ledger.json. */
export interface FdpLedger {
  posterior_fdp: PosteriorFdpBlock;
  exceedance_gate: ExceedanceGateBlock;
  mcse_floor: McseFloorBlock;
  sensitivity_envelope: SensitivityEnvelopeBlock;
}

// ── Factory ────────────────────────────────────────────────────────────────────

/** Convert a raw API ForecastCell into a typed RouteState. Fail-closed. */
export function fromForecastCell(cell: ForecastCell): RouteState {
  const reasons: string[] = cell.route_reasons ?? [];

  switch (cell.forecast_route) {
    case "headline":
      return {
        tag: "headline",
        p_active: cell.p_active,
        ci_80_lo: cell.ci_80_lo,
        ci_80_hi: cell.ci_80_hi,
        ci_95_lo: cell.ci_95_lo,
        ci_95_hi: cell.ci_95_hi,
        h_star_months: cell.h_star_months,
        horizon_months: cell.horizon_months,
        flagged_by_fdp: cell.headline_flagged_by_posterior_fdp ?? false,
        fdp_binding_reason: cell.fdp_binding_reason ?? null,
      };
    case "monitor_only":
      return { tag: "monitor_only", reasons };
    case "aggregate_only":
      return { tag: "aggregate_only", reasons };
    case "evidence_gap":
      return { tag: "evidence_gap", reasons };
    case "withhold":
      return { tag: "withhold", reasons };
    case "horizon_prior_dominated":
      return {
        tag: "horizon_prior_dominated",
        reasons,
        h_star_months: cell.h_star_months,
        horizon_months: cell.horizon_months,
      };
    case "prior_dominated":
      return { tag: "prior_dominated", reasons };
    default:
      // Unknown route from backend — fail-closed: withhold rather than render.
      return { tag: "withhold", reasons: [`unknown_route:${cell.forecast_route}`] };
  }
}

// ── Visual encoding ────────────────────────────────────────────────────────────

export interface RouteAccent {
  text: string;
  bg: string;
  border: string;
  label: string;
}

/** Dark-console Tailwind accent for each route state. */
export function routeAccent(rs: RouteState): RouteAccent {
  switch (rs.tag) {
    case "headline":
      return {
        text: "text-emerald-300",
        bg: "bg-emerald-950/40",
        border: "border-emerald-800/50",
        label: "HEADLINE",
      };
    case "monitor_only":
      return {
        text: "text-amber-300",
        bg: "bg-amber-950/30",
        border: "border-amber-800/40",
        label: "MONITOR ONLY",
      };
    case "aggregate_only":
      return {
        text: "text-sky-300",
        bg: "bg-sky-950/30",
        border: "border-sky-800/40",
        label: "AGGREGATE ONLY",
      };
    case "evidence_gap":
      return {
        text: "text-rose-300",
        bg: "bg-rose-950/30",
        border: "border-rose-800/40",
        label: "EVIDENCE GAP",
      };
    case "withhold":
      return {
        text: "text-red-300",
        bg: "bg-red-950/40",
        border: "border-red-800/60",
        label: "WITHHELD",
      };
    case "horizon_prior_dominated":
      return {
        text: "text-fuchsia-300",
        bg: "bg-fuchsia-950/30",
        border: "border-fuchsia-800/40",
        label: "H-PRIOR DOM",
      };
    case "prior_dominated":
      return {
        text: "text-violet-300",
        bg: "bg-violet-950/30",
        border: "border-violet-800/40",
        label: "PRIOR DOM",
      };
    default:
      return assertNever(rs);
  }
}
