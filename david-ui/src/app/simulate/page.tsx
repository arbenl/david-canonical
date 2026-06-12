import { getSbc } from "@/lib/data";
import { GateCard } from "@/components/gate-card";
import { StatCard } from "@/components/stat-card";
import type { GateStatus } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function SimulatePage() {
  const sbcRes = await getSbc();
  const measurement = sbcRes?.measurement;
  const forecast    = sbcRes?.forecast;

  return (
    <div className="space-y-8 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Simulation Workbench</h1>
        <p className="mt-1 text-sm text-slate-400">
          Simulation-Based Calibration (Talts et al. 2018) — rank statistics must be uniform
        </p>
      </div>

      {/* Run commands */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <p className="text-sm font-semibold text-slate-300 mb-3">Run from terminal</p>
        <div className="space-y-2">
          {[
            { cmd: "david sbc",              label: "Measurement SBC (200 worlds)"   },
            { cmd: "david sbc --forecast",   label: "Forecast SBC (coverage check)"  },
            { cmd: "david sbc --n-worlds 500", label: "High-precision SBC"           },
          ].map(({ cmd, label }) => (
            <div key={cmd} className="flex items-center gap-3">
              <code className="rounded bg-slate-800 px-3 py-1.5 text-xs font-mono text-sky-300 flex-1">{cmd}</code>
              <span className="text-xs text-slate-500">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {!sbcRes && (
        <p className="text-sm text-amber-400">No SBC results yet. Run `david sbc` first.</p>
      )}

      {/* Measurement SBC */}
      {measurement && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Measurement SBC
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <GateCard
              label="Rank uniformity (KS)"
              theorem="SBC gate"
              status={measurement.gate_status as GateStatus}
              detail="KS test on rank statistics across all parameters. Uniform rank = calibrated posteriors."
            />
            <StatCard label="Worlds simulated" value={measurement.n_worlds ?? "—"} />
            <StatCard
              label="KS p-value min"
              value={measurement.ks_p_min?.toFixed(4) ?? "—"}
              sub="Bonferroni-corrected"
              warn={typeof measurement.ks_p_min === "number" && measurement.ks_p_min < 0.05}
            />
          </div>
          {measurement.reason && (
            <div className="mt-3 rounded-lg border border-red-800/40 bg-red-950/20 p-3 text-xs text-red-300">
              {measurement.reason}
            </div>
          )}
        </section>
      )}

      {/* Forecast SBC */}
      {forecast && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Forecast SBC — coverage
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <GateCard
              label="Forecast coverage"
              theorem="Forecast SBC"
              status={forecast.gate_status as GateStatus}
              detail="Empirical 80% and 95% coverage must fall within nominal bands [0.75–0.85] and [0.90–0.98]."
            />
          </div>
          <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <p className="text-xs text-slate-500 mb-3">Nominal coverage bands (pre-registered)</p>
            <div className="flex gap-8 text-sm">
              <div>
                <p className="text-slate-400 text-xs">80% band</p>
                <p className="font-mono text-white">[0.75, 0.85]</p>
              </div>
              <div>
                <p className="text-slate-400 text-xs">95% band</p>
                <p className="font-mono text-white">[0.90, 0.98]</p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Theory */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          SBC procedure
        </h2>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            {
              step: "1. Simulate",
              desc: "Draw θ* from prior. Generate synthetic observations Y* from the HSMM likelihood under θ*.",
            },
            {
              step: "2. Fit",
              desc: "Run m01_forward.stan on Y*. Collect posterior draws of θ.",
            },
            {
              step: "3. Rank",
              desc: "Count how many posterior draws fall below θ*. Ranks must be uniform U[0, L] for calibrated posteriors.",
            },
          ].map(({ step, desc }) => (
            <div key={step} className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
              <p className="text-xs font-bold text-sky-400 mb-1">{step}</p>
              <p className="text-xs text-slate-400 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
