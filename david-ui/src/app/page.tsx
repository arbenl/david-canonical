import { getFitRuns, getStrata, getQueue, getPipelineStatus } from "@/lib/data";
import { StatCard } from "@/components/stat-card";
import { PipelineButton } from "@/components/pipeline-button";
import { FitButton } from "@/components/fit-button";
import { NextStepCard } from "@/components/next-step-card";
import { StrataHealth } from "@/components/strata-health";
import { PipelineStatusLive } from "@/components/pipeline-status-live";
import { PremiumForecastChart } from "@/components/premium-forecast-chart";
import { PremiumRegimeDistribution } from "@/components/premium-regime-distribution";
import Link from "next/link";

export const dynamic = "force-dynamic";

const STATUS_COLOR: Record<string, string> = {
  pass:  "text-emerald-400",
  fail:  "text-red-400",
  skip:  "text-amber-400",
  error: "text-red-400",
};

export default async function CommandCenter() {
  const [runs, strata, queue, pipeline] = await Promise.all([
    getFitRuns(5),
    getStrata(),
    getQueue(),
    getPipelineStatus(),
  ]);

  const latestRun        = runs[0];
  const totalEvidence    = strata.reduce((s, x) => s + x.n_evidence,    0);
  const totalAdjudicated = strata.reduce((s, x) => s + x.n_adjudicated, 0);
  const strataReadyForFit = strata.filter((s) => s.n_adjudicated >= 3).length;
  const adjPct = totalEvidence > 0
    ? Math.round((totalAdjudicated / totalEvidence) * 100) : 0;
  const apiOk = runs.length > 0 || strata.length > 0;

  return (
    <div className="space-y-8 max-w-6xl">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Command Center</h1>
          <p className="mt-1 text-sm text-slate-400">
            DAVID/M0.1 · {new Date().toLocaleDateString("en-GB", { dateStyle: "long" })}
          </p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          {/* Primary action: full pipeline */}
          <PipelineButton />
          {/* Secondary action: fit only */}
          <FitButton />
          {/* API health badge */}
          <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium ${
            apiOk ? "bg-emerald-950 text-emerald-300" : "bg-red-950 text-red-300"
          }`}>
            <span className={`h-1.5 w-1.5 rounded-full ${apiOk ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
            {apiOk ? "API connected" : "API unreachable"}
          </div>
        </div>
      </div>

      {/* ── Live pipeline banner (auto-polls when running) ──────────────── */}
      <PipelineStatusLive initial={pipeline} />

      {/* ── "What to do next" card ─────────────────────────────────────── */}
      <NextStepCard
        totalEvidence={totalEvidence}
        totalAdjudicated={totalAdjudicated}
        nStrata={strata.length}
        strataReadyForFit={strataReadyForFit}
        hasActiveFit={!!latestRun}
        fitPassed={latestRun?.gate_status === "pass"}
        queueCount={queue?.n_in_queue ?? 0}
        pipelineRunning={pipeline.running}
      />

      {/* ── Stats row ───────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          label="Evidence items"
          value={totalEvidence}
          sub={`${totalAdjudicated} adjudicated`}
        />
        <StatCard
          label="Adjudicated %"
          value={`${adjPct}%`}
          sub="LLM + human labels"
          accent={adjPct > 50}
        />
        <StatCard
          label="Strata"
          value={strata.length}
          sub={`${strataReadyForFit} fit-ready`}
        />
        <StatCard
          label="Human queue"
          value={queue?.n_in_queue ?? "—"}
          sub="items awaiting review"
          warn={(queue?.n_in_queue ?? 0) > 0}
        />
      </div>

      {/* ── Human queue alert ───────────────────────────────────────────── */}
      {(queue?.n_in_queue ?? 0) > 0 && (
        <div className="rounded-xl border border-amber-700/40 bg-amber-950/20 p-4">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <p className="text-sm font-semibold text-amber-300">
                {queue!.n_in_queue} item{queue!.n_in_queue !== 1 ? "s" : ""} need human review
              </p>
              <p className="mt-0.5 text-xs text-amber-500/70">
                LLM coder agreement below {Math.round((1 - (queue!.disagreement_threshold ?? 0.3)) * 100)}%
                · click to adjudicate and improve fit quality
              </p>
            </div>
            <Link
              href="/evidence"
              className="rounded-lg bg-amber-800/40 px-3 py-1.5 text-xs font-semibold
                         text-amber-200 hover:bg-amber-800/60 transition-colors"
            >
              Open review queue →
            </Link>
          </div>
        </div>
      )}

      {/* ── Latest fit ──────────────────────────────────────────────────── */}
      {latestRun ? (
        <section>
          <div className="mb-3 flex items-center gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500">
              Latest Bayesian fit
            </h2>
            <span className={`text-xs font-bold ${STATUS_COLOR[latestRun.gate_status] ?? "text-slate-400"}`}>
              {latestRun.gate_status.toUpperCase()}
            </span>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div>
                <p className="font-mono text-xs text-slate-500">{latestRun.run_id}</p>
                <p className="mt-1 font-semibold text-white">{latestRun.model_version}</p>
              </div>
              <div className="flex gap-3 flex-wrap">
                <Link href="/engine"   className="btn-sm">Theorem gates</Link>
                <Link href="/predict"  className="btn-sm">Forecasts</Link>
                <Link href="/validate" className="btn-sm">Validation</Link>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { k: "R̂ max",       v: latestRun.rhat_max?.toFixed(3),     hint: "Gelman-Rubin convergence. < 1.01 required to pass Theorem A'." },
                { k: "ESS bulk",    v: latestRun.ess_bulk_min?.toFixed(0),  hint: "Effective sample size. > 400 required. Low ESS means chains didn't mix." },
                { k: "Divergences", v: String(latestRun.divergences ?? "—"), hint: "MCMC divergences. Must be 0 for reliable posteriors." },
                { k: "Strata fit",  v: String(latestRun.n_strata ?? "—"),    hint: "Number of country×policy strata included in this fit run." },
              ].map(({ k, v, hint }) => (
                <div key={k} className="group relative">
                  <p className="text-[11px] text-slate-500 flex items-center gap-1">
                    {k}
                    <span className="inline-block cursor-help" title={hint}>
                      <span className="text-[10px] text-slate-700 hover:text-slate-400">?</span>
                    </span>
                  </p>
                  <p className="mt-0.5 font-mono text-sm text-white">{v ?? "—"}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      ) : (
        /* No fit yet — explain what's needed */
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Bayesian fit
          </h2>
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
            <p className="text-sm text-slate-400">No fit runs yet.</p>
            <p className="mt-1 text-xs text-slate-600 leading-relaxed">
              Once you have ≥3 adjudicated evidence items per stratum, click{" "}
              <span className="text-slate-400 font-medium">⚙ Run Fit</span> or{" "}
              <span className="text-slate-400 font-medium">▶ Run Pipeline</span> to start the
              Dawid-Skene Bayesian model. Results appear here with MCMC diagnostics and theorem
              gate status.
            </p>
            <div className="mt-3 flex gap-2 flex-wrap text-[11px] text-slate-600">
              {[
                ["Theorem A'",  "Structural independence of coder sources"],
                ["Theorem B'",  "Informativeness lower-95 % ≥ floor"],
                ["Theorem C'",  "Posterior false-discovery rate below q"],
                ["Theorem D'",  "Horizon prior drift τ within tolerance"],
              ].map(([gate, desc]) => (
                <span key={gate}
                  className="rounded-full border border-slate-800 bg-slate-900 px-2 py-0.5"
                  title={desc}
                >
                  {gate}
                </span>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* ── Premium Charts (New) ─────────────────────────────────────────── */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8">
          <PremiumForecastChart />
        </div>
        <div className="lg:col-span-4">
          <PremiumRegimeDistribution />
        </div>
      </section>

      {/* ── Fit history table ────────────────────────────────────────────── */}
      {runs.length > 1 && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Fit history
          </h2>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-800 text-xs text-slate-500">
                <tr>
                  {["Run ID", "Version", "Status", "R̂", "ESS", "Div.", "Started"].map((h) => (
                    <th key={h} className="px-4 py-2.5 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {runs.map((r) => (
                  <tr key={r.run_id} className="hover:bg-slate-800/30">
                    <td className="px-4 py-2.5 font-mono text-xs text-slate-400">{r.run_id.slice(0, 16)}…</td>
                    <td className="px-4 py-2.5 text-slate-300">{r.model_version}</td>
                    <td className={`px-4 py-2.5 font-semibold ${STATUS_COLOR[r.gate_status] ?? ""}`}>
                      {r.gate_status}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs text-slate-300">{r.rhat_max?.toFixed(3) ?? "—"}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-slate-300">{r.ess_bulk_min?.toFixed(0) ?? "—"}</td>
                    <td className="px-4 py-2.5 font-mono text-xs text-slate-300">{r.divergences ?? "—"}</td>
                    <td className="px-4 py-2.5 text-xs text-slate-500">{String(r.started_at).slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ── Strata health grid ───────────────────────────────────────────── */}
      <section>
        <div className="mb-3 flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500">
              Strata health
            </h2>
            <span className="text-xs text-slate-600">country × policy coverage</span>
          </div>
          <Link href="/evidence" className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
            View all evidence →
          </Link>
        </div>
        <StrataHealth strata={strata} />
      </section>

      {/* ── Pipeline flow diagram ─────────────────────────────────────────── */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Pipeline flow
        </h2>
        <div className="flex flex-wrap items-center gap-1">
          {[
            {
              label: "1. Scrape",
              href: "/evidence",
              done: totalEvidence > 0,
              hint: "RSS sources polled → articles filtered by tobacco keywords → stored as evidence items",
            },
            {
              label: "2. Code",
              href: "/evidence",
              done: totalAdjudicated > 0,
              hint: "Two independent LLM coders assign SIO/MIO/CSIO tactic labels per article",
            },
            {
              label: "3. Fit",
              href: "/engine",
              done: !!latestRun,
              hint: "Dawid-Skene Bayesian model aggregates coder labels into interference probability posteriors",
            },
            {
              label: "4. Forecast",
              href: "/predict",
              done: !!latestRun && latestRun.gate_status === "pass",
              hint: "Forecast cells: P(tactic active) at 1/3/6/12/24-month horizons per stratum",
            },
            {
              label: "5. Validate",
              href: "/validate",
              done: false,
              hint: "SBC: simulation-based calibration checks model uncertainty is correct",
            },
          ].map((stage, i, arr) => (
            <div key={stage.label} className="flex items-center">
              <Link
                href={stage.href}
                title={stage.hint}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium
                            transition-colors border ${
                  stage.done
                    ? "bg-emerald-950/50 text-emerald-300 border-emerald-800/40 hover:bg-emerald-950"
                    : "bg-slate-900 text-slate-500 border-slate-800 hover:border-slate-700 hover:text-slate-400"
                }`}
              >
                <span>{stage.done ? "✓" : "○"}</span>
                {stage.label}
              </Link>
              {i < arr.length - 1 && (
                <span className="mx-1 text-slate-700 text-sm">→</span>
              )}
            </div>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-slate-600">
          Hover each stage for details · Pipeline auto-runs daily at 06:00 UTC · Manual trigger: ▶ Run Pipeline
        </p>
      </section>

    </div>
  );
}
