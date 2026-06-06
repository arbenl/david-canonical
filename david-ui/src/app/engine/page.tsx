import { api } from "@/lib/api";
import { GateCard } from "@/components/gate-card";
import { StatCard } from "@/components/stat-card";
import type { GateStatus } from "@/lib/api";

export const dynamic = "force-dynamic";

function gv(summary: any, path: string[]): any {
  return path.reduce((o, k) => o?.[k], summary);
}

export default async function EnginePage() {
  const runsRes = await api.fitRuns(1).catch(() => null);
  const latestRun = runsRes?.fit_runs?.[0];
  const summary = latestRun
    ? await api.fitRun(latestRun.run_id).catch(() => null)
    : null;

  const theorems = summary?.theorems ?? {};
  const gates    = summary?.gates    ?? {};

  const AP = (theorems["A_prime"] ?? {}) as Record<string, unknown>;
  const BP = (theorems["B_prime"] ?? {}) as Record<string, unknown>;
  const CP = (theorems["C_prime"] ?? {}) as Record<string, unknown>;
  const DP = (theorems["D_prime"] ?? {}) as Record<string, unknown>;

  const f1 = (gates["F1"] ?? {}) as Record<string, unknown>;

  return (
    <div className="space-y-8 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-white">AI Engine Verification</h1>
        {latestRun ? (
          <p className="mt-1 text-sm text-slate-400">
            Run <span className="font-mono">{latestRun.run_id}</span> ·{" "}
            <span className={latestRun.gate_status === "pass" ? "text-emerald-400" : "text-red-400"}>
              {latestRun.gate_status}
            </span>
          </p>
        ) : (
          <p className="mt-1 text-sm text-amber-400">No fit run found. Run `david fit` first.</p>
        )}
      </div>

      {/* MCMC diagnostics */}
      {latestRun && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">MCMC diagnostics</h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard
              label="R̂ max"
              value={latestRun.rhat_max?.toFixed(4) ?? "—"}
              sub="threshold: ≤ 1.05"
              warn={(latestRun.rhat_max ?? 0) > 1.05}
            />
            <StatCard
              label="ESS bulk min"
              value={latestRun.ess_bulk_min?.toFixed(0) ?? "—"}
              sub="threshold: ≥ 400"
              warn={(latestRun.ess_bulk_min ?? 999) < 400}
            />
            <StatCard
              label="Divergences"
              value={latestRun.divergences ?? "—"}
              sub="threshold: 0"
              warn={(latestRun.divergences ?? 0) > 0}
            />
            <StatCard
              label="Strata"
              value={latestRun.n_strata ?? "—"}
              sub={`${latestRun.n_labels ?? "—"} labels`}
            />
          </div>
        </section>
      )}

      {/* Theorem gates */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Theorem gates
        </h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <GateCard
            label="Practical identifiability"
            theorem="Theorem A′"
            status={AP.gate_status as GateStatus}
            value={AP.median_d_theta as number | undefined}
            threshold={0.05}
            unit="d(θ)"
            detail="Posterior identification distance from prior. Must exceed floor to confirm identifiability."
          />
          <GateCard
            label="Source informativeness"
            theorem="Theorem B′"
            status={BP.gate_status as GateStatus}
            value={BP.lower_95_I_worst_source as number | undefined}
            threshold={0.10}
            unit="I(O) lower 95%"
            detail="Lower 95% CI of worst-case source informativeness I(O). Guards against empty sources."
          />
          <GateCard
            label="Posterior FDP routing"
            theorem="Theorem C′"
            status={CP.gate_status as GateStatus}
            value={CP.posterior_fdp as number | undefined}
            threshold={0.10}
            unit="FDP"
            detail="Expected false-discovery proportion in posterior tactic activations. Routes to strict tier if exceeded."
          />
          <GateCard
            label="Forecast horizon validity"
            theorem="Theorem D′"
            status={DP.gate_status as GateStatus}
            value={DP.h_star_months as number | undefined}
            unit="h* months"
            detail="Horizon beyond which prior dominates. Cells with h > h* flagged horizon_prior_dominated."
          />
        </div>
      </section>

      {/* F1 — prior predictive gate */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          F1 — prior predictive gate
        </h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <GateCard
            label="Prior predictive coverage"
            theorem="Gate F1"
            status={f1.gate_status as GateStatus}
            value={f1.coverage_rate as number | undefined}
            threshold={0.80}
            unit="coverage"
            detail="Fraction of historical observations within the 200-world prior predictive band."
          />
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <p className="text-xs text-slate-500 mb-3">F1 band parameters</p>
            <div className="space-y-2 text-sm">
              {([
                ["N worlds",      String(f1.n_worlds ?? "—")],
                ["Width floor",   "≥ 0.10"],
                ["Coverage req.", "≥ 80% historical points"],
              ] as [string, string][]).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-slate-400">{k}</span>
                  <span className="font-mono text-slate-200">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Raw theorem JSON */}
      {summary && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Raw fit summary
          </h2>
          <pre className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-xs text-slate-400 overflow-x-auto leading-relaxed">
            {JSON.stringify({ theorems, gates }, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
