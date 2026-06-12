/**
 * E-3: EVIDENCE_GAP warning block.
 *
 * Renders a full-width, high-contrast hazard block when:
 *   - route = evidence_gap (FG2 identification distance below floor, or
 *     C-6 MCSE floor excluded), OR
 *   - Opportunity-Frame Abstention triggers (Q_g unknown / unbounded).
 *
 * INVARIANT: This component NEVER renders chart axes or zero-filled charts.
 * It is a replacement for the chart, not a chart wrapper.
 *
 * Rendering is driven entirely by the RouteState discriminated union from E-1.
 * The binding reasons (Λ_g) are the verbatim strings from the ledger's
 * route_reasons field — no client-side reformatting.
 */

import { assertNever, routeAccent } from "@/lib/route-state";
import type { RouteState } from "@/lib/route-state";

// ── Gate chain mapping ─────────────────────────────────────────────────────────

const GATE_CHAIN: Record<string, { gate: string; theorem: string; detail: string }> = {
  FG2_d_theta_below_floor: {
    gate: "FG2",
    theorem: "Theorem A′ (Identification Distance)",
    detail: "Posterior median d(θ) < 0.05 floor — model parameters not practically identified.",
  },
  fdp_mcse_exceeds_gate_margin: {
    gate: "C-6",
    theorem: "Theorem C (MCSE/ESS Floor)",
    detail: "MCSE(p_i) ≥ 0.10 × q — Monte Carlo error exceeds gate margin; cell excluded from claim family.",
  },
};

function parseGateCode(reason: string): { gate: string; theorem: string; detail: string } | null {
  for (const [key, val] of Object.entries(GATE_CHAIN)) {
    if (reason.includes(key)) return val;
  }
  if (reason.startsWith("FG")) {
    const gate = reason.split("_")[0];
    return { gate, theorem: "Gate constraint", detail: reason.replace(/_/g, " ") };
  }
  return null;
}

// ── Opportunity-Frame Abstention ───────────────────────────────────────────────

const OFA_THEOREM_CITATION =
  "Theorem 1 — Opportunity-Frame Abstention: when Q_g is unknown or unbounded, " +
  "the engine cannot form a bounded probability claim. " +
  "Routing to EVIDENCE_GAP is the correct fail-closed response.";

// ── Sub-components ─────────────────────────────────────────────────────────────

function HazardStripe() {
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full">
      {Array.from({ length: 12 }).map((_, i) => (
        <div
          key={i}
          className="flex-1"
          style={{ background: i % 2 === 0 ? "#f59e0b" : "#1e293b" }}
        />
      ))}
    </div>
  );
}

function ReasonRow({ reason }: Readonly<{ reason: string }>) {
  const parsed = parseGateCode(reason);
  return (
    <div className="rounded border border-amber-900/40 bg-amber-950/20 px-3 py-2">
      <div className="flex items-start gap-3">
        {parsed && (
          <span className="shrink-0 rounded bg-amber-900/50 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-amber-300 font-mono">
            {parsed.gate}
          </span>
        )}
        <div className="min-w-0 space-y-0.5">
          <p className="text-[11px] font-mono text-amber-200 break-all">{reason}</p>
          {parsed && (
            <p className="text-[10px] text-slate-500">{parsed.detail}</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export interface EvidenceGapBlockProps {
  /** Must be evidence_gap — call site enforced by RouteState discriminant. */
  routeState: Extract<RouteState, { tag: "evidence_gap" }>;
  /** Optional: signals Opportunity-Frame Abstention specifically. */
  ofaTriggered?: boolean;
  /** Optional: cell / tactic label for context. */
  cellLabel?: string;
}

export function EvidenceGapBlock({
  routeState,
  ofaTriggered = false,
  cellLabel,
}: Readonly<EvidenceGapBlockProps>) {
  const accent = routeAccent(routeState);

  return (
    <div className={`rounded-xl border ${accent.border} ${accent.bg} p-5 space-y-4 w-full`}>
      <HazardStripe />

      {/* Route code + cell label */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <span className={`text-[10px] font-bold uppercase tracking-[0.2em] font-mono ${accent.text}`}>
            {accent.label}
          </span>
          {cellLabel && (
            <p className="mt-0.5 text-[10px] text-slate-500 font-mono">{cellLabel}</p>
          )}
        </div>
        <span className="rounded border border-rose-900/50 bg-rose-950/30 px-2 py-0.5 text-[9px] font-mono text-rose-300">
          chart suppressed
        </span>
      </div>

      {/* Binding reasons (Λ_g) verbatim from ledger */}
      {routeState.reasons.length > 0 ? (
        <div className="space-y-2">
          <p className="text-[9px] uppercase tracking-widest text-slate-600 font-medium">
            Binding constraint Λ_g
          </p>
          {routeState.reasons.map((r, i) => (
            <ReasonRow key={i} reason={r} />
          ))}
        </div>
      ) : (
        <div className="rounded border border-slate-800 bg-slate-900/40 px-3 py-2">
          <p className="text-[10px] font-mono text-slate-500">
            No binding reasons recorded in ledger — consult route_ledger.json.
          </p>
        </div>
      )}

      {/* Theorem citation */}
      <div className="rounded border border-amber-900/30 bg-[#070b19]/80 px-4 py-3">
        <p className="text-[9px] uppercase tracking-widest text-amber-600 font-medium mb-1.5">
          Theorem citation
        </p>
        <p className="text-[11px] text-slate-300 leading-relaxed">
          {ofaTriggered
            ? OFA_THEOREM_CITATION
            : "FDP claim family — cells excluded by FG2 (identification distance) or " +
              "C-6 (MCSE floor) cannot form posterior probability claims. " +
              "Routing to EVIDENCE_GAP enforces the fail-closed discipline."}
        </p>
      </div>

      {/* Failed gate chain summary */}
      <div className="flex flex-wrap gap-2 text-[9px] font-mono">
        {routeState.reasons.map((r, i) => {
          const parsed = parseGateCode(r);
          return parsed ? (
            <span key={i} className="rounded border border-red-900/40 bg-red-950/20 px-2 py-0.5 text-red-400">
              {parsed.gate} fail
            </span>
          ) : null;
        })}
        <span className="rounded border border-slate-800 bg-slate-900/40 px-2 py-0.5 text-slate-500">
          route: {routeState.tag}
        </span>
      </div>

      <HazardStripe />
    </div>
  );
}

// ── Convenience wrapper that dispatches on RouteState ─────────────────────────

/**
 * Drop-in replacement for any chart component that receives a RouteState.
 * When the route is evidence_gap → renders EvidenceGapBlock.
 * Otherwise → renders children (the real chart).
 */
export function EvidenceGapGuard({
  routeState,
  cellLabel,
  children,
}: Readonly<{
  routeState: RouteState;
  cellLabel?: string;
  children: React.ReactNode;
}>) {
  if (routeState.tag === "evidence_gap") {
    return (
      <EvidenceGapBlock
        routeState={routeState}
        cellLabel={cellLabel}
      />
    );
  }

  // Exhaustive check — any route that is not evidence_gap passes through.
  switch (routeState.tag) {
    case "headline":
    case "monitor_only":
    case "aggregate_only":
    case "withhold":
    case "horizon_prior_dominated":
    case "prior_dominated":
      return <>{children}</>;
    default:
      return assertNever(routeState);
  }
}
