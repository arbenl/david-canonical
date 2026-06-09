import { api } from "@/lib/api";
import { GateCard } from "@/components/gate-card";
import { StatCard } from "@/components/stat-card";
import type { GateStatus } from "@/lib/api";

export const dynamic = "force-dynamic";

function gv(summary: any, path: string[]): any {
  return path.reduce((o, k) => o?.[k], summary);
}

export default async function EnginePage() {
  const [runsRes, automationRes, independenceRes] = await Promise.allSettled([
    api.fitRuns(1),
    api.automationStatus(),
    api.sourceIndependence(),
  ]);
  const runsData   = runsRes.status   === "fulfilled" ? runsRes.value   : null;
  const automation = automationRes.status === "fulfilled" ? automationRes.value : null;
  const independence = independenceRes.status === "fulfilled" ? independenceRes.value : null;

  const latestRun = runsData?.fit_runs?.[0];
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

      {/* Automation status */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Automation status
        </h2>
        {automation ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {/* Nightly ingest */}
            <div className={`rounded-xl border p-4 ${
              automation.nightly_ingest.stale
                ? "border-amber-700/50 bg-amber-900/10"
                : "border-emerald-700/50 bg-emerald-900/10"
            }`}>
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Nightly Ingest</p>
                  <p className={`mt-1 text-sm font-medium ${
                    automation.nightly_ingest.stale ? "text-amber-300" : "text-emerald-300"
                  }`}>
                    {automation.nightly_ingest.stale ? "⚠ Stale" : "✓ Current"}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {automation.nightly_ingest.last_run
                      ? `Last: ${new Date(automation.nightly_ingest.last_run).toLocaleString()}`
                      : "No log found"}
                  </p>
                </div>
                <span className={`rounded px-2 py-0.5 text-xs font-mono ${
                  automation.nightly_ingest.launchd === "loaded"
                    ? "bg-emerald-900/40 text-emerald-400"
                    : "bg-red-900/40 text-red-400"
                }`}>
                  {automation.nightly_ingest.launchd}
                </span>
              </div>
            </div>
            {/* Weekly fit */}
            <div className={`rounded-xl border p-4 ${
              automation.weekly_fit.stale
                ? "border-amber-700/50 bg-amber-900/10"
                : "border-emerald-700/50 bg-emerald-900/10"
            }`}>
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Weekly Fit</p>
                  <p className={`mt-1 text-sm font-medium ${
                    automation.weekly_fit.stale ? "text-amber-300" : "text-emerald-300"
                  }`}>
                    {automation.weekly_fit.stale ? "⚠ Stale" : "✓ Current"}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {automation.weekly_fit.last_run
                      ? `Last pass: ${new Date(automation.weekly_fit.last_run).toLocaleString()}`
                      : "No passing fit yet"}
                  </p>
                </div>
                <span className={`rounded px-2 py-0.5 text-xs font-mono ${
                  automation.weekly_fit.launchd === "loaded"
                    ? "bg-emerald-900/40 text-emerald-400"
                    : "bg-red-900/40 text-red-400"
                }`}>
                  {automation.weekly_fit.launchd}
                </span>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-500">Automation status unavailable.</p>
        )}
      </section>

      {/* Source independence */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Source structural independence (Theorem A′)
        </h2>
        {independence ? (
          <div className="rounded-xl border border-slate-700/60 bg-slate-900/40 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className={`rounded-full px-3 py-0.5 text-xs font-semibold ${
                  independence.effective_independent_sources.status === "pass"
                    ? "bg-emerald-900/50 text-emerald-300"
                    : "bg-red-900/50 text-red-300"
                }`}>
                  S_eff = {independence.effective_independent_sources.s_eff}
                </span>
                <span className="text-xs text-slate-400">floor = 3</span>
              </div>
              <div className="text-right">
                <span className={`text-xs ${
                  independence.stale ? "text-amber-400" : "text-slate-400"
                }`}>
                  {independence.stale ? "⚠ Review overdue" : "✓ Current"}
                </span>
                <p className="text-xs text-slate-500">
                  Reviewed {independence.reviewed_at} · next {independence.next_review_due}
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-1 sm:grid-cols-3">
              {independence.sources.map((src) => (
                <div key={src.source_id} className="rounded-lg bg-slate-800/50 px-3 py-2">
                  <p className="text-xs font-mono text-slate-300 truncate">{src.source_id}</p>
                  <p className="text-xs text-slate-500">{src.family}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-500">Source independence ledger unavailable.</p>
        )}
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
