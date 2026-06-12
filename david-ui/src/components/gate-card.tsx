/**
 * E-4: Gate-stack panel (FG1–FG6).
 *
 * Exports:
 *   GateCard   — single gate card (existing; kept for backward compat).
 *   GateStack  — vertical ladder of all 6 FG gates from a routed cell.
 *
 * Data contract:
 *   GateStackProps carries the pre-computed gate statistics from the
 *   route_ledger.json. The component is pure presentation — no data
 *   fetching, no recomputation.
 *
 * F13 special treatment (KNOWN_GATES.md):
 *   F13 (forecast horizon respect) is an expected fail when h > h*.
 *   When f13KnownFail=true, FG5 renders a muted amber "expected under
 *   current regime" callout instead of a red error. This is correct
 *   fail-closed behaviour per the Automation Contract.
 */

import type { GateStatus } from "@/lib/api";

// ── GateCard (unchanged public API) ───────────────────────────────────────────

interface GateCardProps {
  label: string;
  theorem: string;
  status: GateStatus | undefined;
  value?: number | string;
  threshold?: number | string;
  unit?: string;
  detail?: string;
}

const CARD_COLORS: Record<string, string> = {
  pass:  "border-emerald-700/50 bg-emerald-950/40",
  fail:  "border-red-700/50 bg-red-950/40",
  skip:  "border-amber-700/50 bg-amber-950/40",
  error: "border-red-700/50 bg-red-950/40",
};

const CARD_DOTS: Record<string, string> = {
  pass:  "bg-emerald-400",
  fail:  "bg-red-400",
  skip:  "bg-amber-400",
  error: "bg-red-400",
};

export function GateCard({ label, theorem, status, value, threshold, unit, detail }: GateCardProps) {
  const s = status ?? "skip";
  return (
    <div className={`rounded-xl border p-4 ${CARD_COLORS[s] ?? CARD_COLORS.skip}`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[11px] font-mono text-slate-500">{theorem}</p>
          <p className="mt-0.5 text-sm font-semibold text-slate-200">{label}</p>
        </div>
        <span className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${CARD_DOTS[s] ?? CARD_DOTS.skip}`} />
      </div>
      {value !== undefined && (
        <div className="mt-3 flex items-end gap-2">
          <span className="text-2xl font-mono font-bold text-white tabular-nums">
            {typeof value === "number" ? value.toFixed(3) : value}
          </span>
          {unit && <span className="mb-0.5 text-xs text-slate-500">{unit}</span>}
        </div>
      )}
      {threshold !== undefined && (
        <p className="mt-1 text-[11px] text-slate-500">
          floor: {typeof threshold === "number" ? threshold.toFixed(3) : threshold}
        </p>
      )}
      {detail && (
        <p className="mt-2 text-[11px] text-slate-400 leading-relaxed">{detail}</p>
      )}
      <div className="mt-3 text-[11px] font-semibold uppercase tracking-wider" style={{
        color: s === "pass" ? "#34d399" : s === "fail" ? "#f87171" : "#fbbf24"
      }}>
        {s}
      </div>
    </div>
  );
}

// ── GateStack types ────────────────────────────────────────────────────────────

/** Pass/fail/skip status for a single gate rung. */
export interface GateRunStatus {
  status: GateStatus;
  /** Measured statistic (shown as the primary number). */
  measured?: number | null;
  /** Floor / threshold to beat. */
  floor?: number | null;
  /** Unit label (e.g. "months"). */
  unit?: string;
  /** Free-text detail line shown below the number. */
  detail?: string | null;
}

/** All 6 FG gate inputs for a single cell. */
export interface GateStackProps {
  /**
   * FG1: measurement battery F1, F3, F4, F5 on the fit run.
   * Pass when all required F-tests pass or skip.
   */
  fg1: GateRunStatus & {
    /** F-tests that caused a fail (e.g. ["F1", "F3"]). */
    failedTests?: string[];
  };

  /**
   * FG2: Theorem A′ identification distance d(θ) ≥ 0.05.
   */
  fg2: GateRunStatus;

  /**
   * FG3: Theorem B′ informativeness.
   * B′.1 — I(O) lower 95% CI ≥ 0.10
   * B′.2 — N_eff · I² ≥ 3.0
   */
  fg3: {
    iLower95: GateRunStatus;   // I(O) lower 95% CI
    nEffI2:   GateRunStatus;   // N_eff · I² (weak-channel guard)
  };

  /**
   * FG4: Forecast SBC (F14) pass.
   */
  fg4: GateRunStatus & { sbcReason?: string | null };

  /**
   * FG5: Horizon validity h ≤ h*(g).
   * f13KnownFail = true → F13 expected fail under current regime (KNOWN_GATES.md).
   * When true, render "expected under current regime" callout, not a red error.
   */
  fg5: GateRunStatus & {
    horizonMonths?: number | null;
    hStarMonths?: number | null;
    f13KnownFail?: boolean;
  };

  /**
   * FG6: endogenous observability + FDP family.
   * lambda: λ-interval width vs max.
   * fdp:    posterior expected FDP vs q.
   * exceedance: P̂(FDP > γ) vs α.
   */
  fg6: {
    lambda:     GateRunStatus;
    fdp:        GateRunStatus;
    exceedance: GateRunStatus;
  };
}

// ── Rung rendering ─────────────────────────────────────────────────────────────

const STATUS_DOT: Record<GateStatus | "unknown", string> = {
  pass:    "bg-emerald-400",
  fail:    "bg-red-400",
  skip:    "bg-amber-400",
  error:   "bg-red-400",
  unknown: "bg-slate-600",
};

const STATUS_TEXT: Record<GateStatus | "unknown", string> = {
  pass:    "text-emerald-300",
  fail:    "text-red-300",
  skip:    "text-amber-300",
  error:   "text-red-300",
  unknown: "text-slate-500",
};

const STATUS_BG: Record<GateStatus | "unknown", string> = {
  pass:    "border-emerald-800/40 bg-emerald-950/25",
  fail:    "border-red-800/50 bg-red-950/30",
  skip:    "border-amber-800/40 bg-amber-950/20",
  error:   "border-red-800/50 bg-red-950/30",
  unknown: "border-slate-800 bg-slate-900/30",
};

function statusKey(s: GateStatus | undefined): GateStatus | "unknown" {
  return s ?? "unknown";
}

function fmt(v: number | null | undefined, decimals = 3): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

interface RungProps {
  id: string;
  theorem: string;
  label: string;
  run: GateRunStatus;
  extra?: React.ReactNode;
  isKnownFail?: boolean;
}

function Rung({ id, theorem, label, run, extra, isKnownFail }: Readonly<RungProps>) {
  const sk = statusKey(run.status);
  const isEffectiveFail = sk === "fail" && !isKnownFail;

  return (
    <div className={`rounded-lg border px-3 py-2.5 ${STATUS_BG[sk]}`}>
      {/* Gate ID + label row */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="shrink-0 rounded bg-slate-800 px-1.5 py-0.5 text-[9px] font-bold font-mono text-slate-400 uppercase tracking-wider">
            {id}
          </span>
          <div className="min-w-0">
            <p className="text-[9px] font-mono text-slate-600 truncate">{theorem}</p>
            <p className="text-[11px] font-semibold text-slate-300 leading-tight">{label}</p>
          </div>
        </div>
        <span className={`shrink-0 h-2 w-2 rounded-full ${STATUS_DOT[sk]}`} />
      </div>

      {/* Statistic row */}
      {(run.measured != null || run.floor != null) && (
        <div className="mt-2 flex items-end gap-2 text-[11px] font-mono">
          {run.measured != null && (
            <span className={`font-bold tabular-nums ${isEffectiveFail ? "text-red-300" : "text-slate-200"}`}>
              {fmt(run.measured)}
            </span>
          )}
          {run.unit && <span className="text-slate-600 text-[10px]">{run.unit}</span>}
          {run.floor != null && (
            <span className="text-slate-600 text-[10px]">≥ {fmt(run.floor)}</span>
          )}
        </div>
      )}

      {/* Detail */}
      {run.detail && (
        <p className="mt-1 text-[10px] text-slate-500 leading-relaxed">{run.detail}</p>
      )}

      {/* F13 known-fail callout */}
      {isKnownFail && run.status === "fail" && (
        <div className="mt-2 rounded border border-amber-900/40 bg-amber-950/20 px-2 py-1.5">
          <p className="text-[9px] text-amber-400 font-medium uppercase tracking-wider mb-0.5">
            Expected under current regime
          </p>
          <p className="text-[10px] text-slate-400 leading-snug">
            F13 correctly detects prior domination beyond h*. This is the designed
            fail-closed response — see KNOWN_GATES.md for full rationale.
          </p>
        </div>
      )}

      {/* Status chip */}
      <div className={`mt-1.5 text-[9px] font-bold uppercase tracking-wider font-mono ${STATUS_TEXT[sk]}`}>
        {sk}
      </div>

      {extra}
    </div>
  );
}

// ── FG6 compound rung ──────────────────────────────────────────────────────────

function Fg6Rung({ fg6 }: Readonly<{ fg6: GateStackProps["fg6"] }>) {
  // Overall status: worst of the three sub-gates.
  const statuses = [fg6.lambda.status, fg6.fdp.status, fg6.exceedance.status];
  const overall: GateStatus =
    statuses.includes("fail") ? "fail" :
    statuses.includes("error") ? "error" :
    statuses.includes("skip") ? "skip" : "pass";

  const sk = statusKey(overall);

  return (
    <div className={`rounded-lg border px-3 py-2.5 ${STATUS_BG[sk]}`}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span className="shrink-0 rounded bg-slate-800 px-1.5 py-0.5 text-[9px] font-bold font-mono text-slate-400 uppercase tracking-wider">
            FG6
          </span>
          <div>
            <p className="text-[9px] font-mono text-slate-600">Observability + FDP</p>
            <p className="text-[11px] font-semibold text-slate-300">Endogenous-Obs · Theorem C</p>
          </div>
        </div>
        <span className={`h-2 w-2 rounded-full ${STATUS_DOT[sk]}`} />
      </div>

      <div className="space-y-2">
        {/* λ sub-gate */}
        <div className="flex items-center justify-between text-[10px] font-mono">
          <span className="text-slate-500">λ width</span>
          <div className="flex items-center gap-1.5">
            <span className={`tabular-nums ${fg6.lambda.status === "pass" ? "text-emerald-300" : "text-red-300"}`}>
              {fmt(fg6.lambda.measured, 3)}
            </span>
            <span className="text-slate-600">≤ {fmt(fg6.lambda.floor, 2)}</span>
            <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[statusKey(fg6.lambda.status)]}`} />
          </div>
        </div>

        {/* FDP sub-gate */}
        <div className="flex items-center justify-between text-[10px] font-mono">
          <span className="text-slate-500">ê[FDP] at m*</span>
          <div className="flex items-center gap-1.5">
            <span className={`tabular-nums ${fg6.fdp.status === "pass" ? "text-emerald-300" : "text-amber-300"}`}>
              {fmt(fg6.fdp.measured, 4)}
            </span>
            <span className="text-slate-600">≤ q={fmt(fg6.fdp.floor, 2)}</span>
            <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[statusKey(fg6.fdp.status)]}`} />
          </div>
        </div>

        {/* Exceedance sub-gate */}
        <div className="flex items-center justify-between text-[10px] font-mono">
          <span className="text-slate-500">P̂(FDP &gt; γ)</span>
          <div className="flex items-center gap-1.5">
            <span className={`tabular-nums ${fg6.exceedance.status === "pass" ? "text-emerald-300" : fg6.exceedance.status === "skip" ? "text-amber-300" : "text-red-300"}`}>
              {fmt(fg6.exceedance.measured, 3)}
            </span>
            <span className="text-slate-600">≤ α={fmt(fg6.exceedance.floor, 2)}</span>
            <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[statusKey(fg6.exceedance.status)]}`} />
          </div>
        </div>
      </div>

      <div className={`mt-1.5 text-[9px] font-bold uppercase tracking-wider font-mono ${STATUS_TEXT[sk]}`}>
        {sk}
      </div>
    </div>
  );
}

// ── Main GateStack ─────────────────────────────────────────────────────────────

export function GateStack({ fg1, fg2, fg3, fg4, fg5, fg6 }: Readonly<GateStackProps>) {
  // Connector between rungs — amber if any gate above has failed.
  const failed = [
    fg1.status, fg2.status,
    fg3.iLower95.status, fg3.nEffI2.status,
    fg4.status, fg5.status,
  ].some((s) => s === "fail");

  const connColor = failed ? "bg-red-900/40" : "bg-emerald-900/30";

  const connector = <div className={`mx-auto w-0.5 h-3 ${connColor}`} />;

  const fg3Combined: GateRunStatus = {
    status:
      fg3.iLower95.status === "fail" || fg3.nEffI2.status === "fail" ? "fail" :
      fg3.iLower95.status === "error" || fg3.nEffI2.status === "error" ? "error" :
      fg3.iLower95.status === "pass" && fg3.nEffI2.status === "pass" ? "pass" : "skip",
  };

  return (
    <div className="space-y-0">
      {/* FG1 */}
      <Rung
        id="FG1"
        theorem="Measurement battery F1, F3, F4, F5"
        label="Fit quality gates"
        run={fg1}
        extra={fg1.failedTests && fg1.failedTests.length > 0 ? (
          <p className="mt-1 text-[10px] font-mono text-red-400">
            failed: {fg1.failedTests.join(", ")}
          </p>
        ) : undefined}
      />
      {connector}

      {/* FG2 */}
      <Rung
        id="FG2"
        theorem="Theorem A′ — Identification distance"
        label="d(θ) posterior median"
        run={{ ...fg2, floor: fg2.floor ?? 0.05 }}
      />
      {connector}

      {/* FG3: two sub-gates shown as a single rung with split display */}
      <div className={`rounded-lg border px-3 py-2.5 ${STATUS_BG[statusKey(fg3Combined.status)]}`}>
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <span className="shrink-0 rounded bg-slate-800 px-1.5 py-0.5 text-[9px] font-bold font-mono text-slate-400 uppercase tracking-wider">FG3</span>
            <div>
              <p className="text-[9px] font-mono text-slate-600">Theorem B′ — Informativeness</p>
              <p className="text-[11px] font-semibold text-slate-300">I(O) lower 95% CI · N_eff·I²</p>
            </div>
          </div>
          <span className={`h-2 w-2 rounded-full ${STATUS_DOT[statusKey(fg3Combined.status)]}`} />
        </div>
        <div className="space-y-1.5 text-[10px] font-mono">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">I_lower_95</span>
            <div className="flex items-center gap-1.5">
              <span className={`tabular-nums ${fg3.iLower95.status === "pass" ? "text-emerald-300" : "text-red-300"}`}>
                {fmt(fg3.iLower95.measured, 3)}
              </span>
              <span className="text-slate-600">≥ {fmt(fg3.iLower95.floor ?? 0.10, 2)}</span>
              <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[statusKey(fg3.iLower95.status)]}`} />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">N_eff·I²</span>
            <div className="flex items-center gap-1.5">
              <span className={`tabular-nums ${fg3.nEffI2.status === "pass" ? "text-emerald-300" : "text-red-300"}`}>
                {fmt(fg3.nEffI2.measured, 2)}
              </span>
              <span className="text-slate-600">≥ {fmt(fg3.nEffI2.floor ?? 3.0, 1)}</span>
              <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[statusKey(fg3.nEffI2.status)]}`} />
            </div>
          </div>
        </div>
        <div className={`mt-1.5 text-[9px] font-bold uppercase tracking-wider font-mono ${STATUS_TEXT[statusKey(fg3Combined.status)]}`}>
          {statusKey(fg3Combined.status)}
        </div>
      </div>
      {connector}

      {/* FG4 */}
      <Rung
        id="FG4"
        theorem="F14 — Forecast SBC coverage"
        label="Forecast calibration (SBC)"
        run={fg4}
        extra={fg4.sbcReason ? (
          <p className="mt-1 text-[10px] font-mono text-slate-500">{fg4.sbcReason}</p>
        ) : undefined}
      />
      {connector}

      {/* FG5 — F13 known-fail support */}
      <Rung
        id="FG5"
        theorem="Theorem D-forecast — Horizon validity"
        label="h ≤ h*(g)"
        run={{
          ...fg5,
          measured: fg5.horizonMonths ?? fg5.measured,
          floor:    fg5.hStarMonths  ?? fg5.floor,
          unit:     "months",
          detail:   fg5.detail ?? (
            fg5.horizonMonths != null && fg5.hStarMonths != null
              ? `h = ${fg5.horizonMonths}mo, h* = ${fg5.hStarMonths}mo`
              : null
          ),
        }}
        isKnownFail={fg5.f13KnownFail}
      />
      {connector}

      {/* FG6 compound */}
      <Fg6Rung fg6={fg6} />
    </div>
  );
}

// ── Factory helpers ────────────────────────────────────────────────────────────

/** Floor constants mirroring config.py */
export const GATE_FLOORS = {
  ID_DISTANCE: 0.05,
  INFORMATIVENESS_LOWER_95: 0.10,
  N_EFF_I2: 3.0,
  LAMBDA_ENDOG_MAX_WIDTH: 0.20,
  FDP_Q: 0.10,
  FDP_EXCEEDANCE_ALPHA: 0.05,
} as const;

/**
 * Build a GateStackProps from a route_ledger cell + ledger root data.
 * Call site is responsible for selecting the correct cell from the ledger.
 */
export function buildGateStack(
  cell: {
    forecast_route: string;
    route_reasons?: string[];
    identification_distance_posterior_median?: number | null;
    informativeness_I_O_lower_95?: number | null;
    informativeness_n_eff_i2?: number | null;
    lambda_endogenous_bounds?: [number, number] | null;
    horizon_validity?: { h_star_months?: number; below_h_star?: boolean };
    horizon_months?: number;
  },
  ledger: {
    fg1Pass: boolean;
    fg1FailedTests?: string[];
    fg4Pass: boolean;
    fg4Reason?: string | null;
    fdpValue?: number | null;
    exceedanceFraction?: number | null;
    f13KnownFail?: boolean;
  },
): GateStackProps {
  const dTheta = cell.identification_distance_posterior_median ?? null;
  const iLo95  = cell.informativeness_I_O_lower_95 ?? null;
  const nEffI2 = cell.informativeness_n_eff_i2 ?? null;
  const lambdaBounds = cell.lambda_endogenous_bounds ?? null;
  const lambdaWidth  = lambdaBounds ? lambdaBounds[1] - lambdaBounds[0] : null;
  const hv = cell.horizon_validity ?? {};
  const hStar    = hv.h_star_months ?? null;
  const belowH   = hv.below_h_star ?? null;
  const hMonths  = cell.horizon_months ?? null;

  const fg2Status: GateStatus =
    dTheta == null ? "skip" : dTheta >= GATE_FLOORS.ID_DISTANCE ? "pass" : "fail";

  const iStatus: GateStatus =
    iLo95 == null ? "skip" : iLo95 >= GATE_FLOORS.INFORMATIVENESS_LOWER_95 ? "pass" : "fail";

  const nStatus: GateStatus =
    nEffI2 == null ? "skip" : nEffI2 >= GATE_FLOORS.N_EFF_I2 ? "pass" : "fail";

  const lambdaStatus: GateStatus =
    lambdaWidth == null ? "skip" :
    lambdaWidth <= GATE_FLOORS.LAMBDA_ENDOG_MAX_WIDTH ? "pass" : "fail";

  const fg5Status: GateStatus =
    belowH == null ? "skip" : belowH ? "pass" : "fail";

  const fdpStatus: GateStatus =
    ledger.fdpValue == null ? "skip" :
    ledger.fdpValue <= GATE_FLOORS.FDP_Q ? "pass" : "fail";

  const excStatus: GateStatus =
    ledger.exceedanceFraction == null ? "skip" :
    ledger.exceedanceFraction <= GATE_FLOORS.FDP_EXCEEDANCE_ALPHA ? "pass" : "fail";

  return {
    fg1: {
      status: ledger.fg1Pass ? "pass" : "fail",
      failedTests: ledger.fg1FailedTests,
    },
    fg2: {
      status: fg2Status,
      measured: dTheta,
      floor: GATE_FLOORS.ID_DISTANCE,
    },
    fg3: {
      iLower95: {
        status: iStatus,
        measured: iLo95,
        floor: GATE_FLOORS.INFORMATIVENESS_LOWER_95,
      },
      nEffI2: {
        status: nStatus,
        measured: nEffI2,
        floor: GATE_FLOORS.N_EFF_I2,
      },
    },
    fg4: {
      status: ledger.fg4Pass ? "pass" : "fail",
      sbcReason: ledger.fg4Reason,
    },
    fg5: {
      status: fg5Status,
      horizonMonths: hMonths,
      hStarMonths: hStar,
      f13KnownFail: ledger.f13KnownFail,
    },
    fg6: {
      lambda: {
        status: lambdaStatus,
        measured: lambdaWidth,
        floor: GATE_FLOORS.LAMBDA_ENDOG_MAX_WIDTH,
      },
      fdp: {
        status: fdpStatus,
        measured: ledger.fdpValue,
        floor: GATE_FLOORS.FDP_Q,
      },
      exceedance: {
        status: excStatus,
        measured: ledger.exceedanceFraction,
        floor: GATE_FLOORS.FDP_EXCEEDANCE_ALPHA,
      },
    },
  };
}
