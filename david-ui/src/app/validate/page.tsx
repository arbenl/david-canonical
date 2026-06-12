import { getFitRuns, getConfig } from "@/lib/data";

export const dynamic = "force-dynamic";

const FALSIFICATION_TESTS = [
  { id: "F1",  name: "Prior predictive coverage",        layer: "measurement" },
  { id: "F2",  name: "Posterior predictive check",       layer: "measurement" },
  { id: "F3",  name: "Leave-one-out cross-validation",   layer: "measurement" },
  { id: "F4",  name: "Measurement SBC uniformity",       layer: "measurement" },
  { id: "F5",  name: "Forecast SBC coverage",            layer: "forecast"    },
  { id: "F6",  name: "Endogenous observability sensitivity", layer: "design"  },
  { id: "F7",  name: "Identification distance (A′)",     layer: "theorem"     },
  { id: "F8",  name: "Source informativeness (B′)",      layer: "theorem"     },
  { id: "F9",  name: "Posterior FDP routing (C′)",       layer: "theorem"     },
  { id: "F10", name: "Horizon validity (D′)",            layer: "theorem"     },
  { id: "F11", name: "Capture-resistance adversarial",   layer: "adversarial" },
  { id: "F12", name: "Structural independence audit",    layer: "adversarial" },
  { id: "F13", name: "Horizon prior-dominance check",    layer: "forecast"    },
  { id: "F14", name: "Forecast SBC nominal coverage",    layer: "forecast"    },
  { id: "F15", name: "Endogenous-observability bounds",  layer: "design"      },
];

const layerColor: Record<string, string> = {
  measurement: "bg-sky-950/40 text-sky-300",
  forecast:    "bg-violet-950/40 text-violet-300",
  theorem:     "bg-emerald-950/40 text-emerald-300",
  adversarial: "bg-red-950/40 text-red-300",
  design:      "bg-slate-800 text-slate-400",
};

export default async function ValidatePage() {
  const [cfg, runs] = await Promise.all([getConfig(), getFitRuns(1)]);
  const latestRun = runs[0] ?? null;

  // Try to load falsification ledger from file-based endpoint
  let ledger: Record<string, any> | null = null;
  if (latestRun) {
    try {
      const r = await fetch(
        `${process.env.DAVID_API_URL ?? "http://localhost:8080"}/forecasts/${latestRun.run_id}/falsification_ledger.json`,
        { next: { revalidate: 60 } }
      );
      if (r.ok) ledger = await r.json();
    } catch {}
  }

  return (
    <div className="space-y-8 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Thesis Validation</h1>
        <p className="mt-1 text-sm text-slate-400">
          Pre-registration audit · F1–F15 falsification battery · Coder calibration
        </p>
      </div>

      {/* Pre-registration audit */}
      {cfg && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Pre-registered configuration
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {/* Theorem floors */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs font-bold text-emerald-400 mb-3">Theorem floors</p>
              <div className="space-y-2 text-sm">
                {Object.entries(cfg.theorem_floors).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-slate-400 text-xs font-mono">{k}</span>
                    <span className="font-mono text-white">{v.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Fit thresholds */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs font-bold text-sky-400 mb-3">Fit thresholds</p>
              <div className="space-y-2 text-sm">
                {Object.entries(cfg.fit_thresholds).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-slate-400 text-xs font-mono">{k}</span>
                    <span className="font-mono text-white">{v}</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Model + tactics */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs font-bold text-violet-400 mb-3">Model</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400 text-xs">version</span>
                  <span className="font-mono text-white text-xs">{cfg.model_version}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 text-xs">tactic classes</span>
                  <span className="font-mono text-white text-xs">{cfg.tactic_classes.join(", ")}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 text-xs">horizons</span>
                  <span className="font-mono text-white text-xs">{cfg.forecast_horizons_months.join(", ")} months</span>
                </div>
              </div>
            </div>
            {/* Human loop */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <p className="text-xs font-bold text-amber-400 mb-3">Human-loop controls</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400 text-xs">disagreement threshold</span>
                  <span className="font-mono text-white">{cfg.human_loop.disagreement_threshold}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 text-xs">gold sample rate</span>
                  <span className="font-mono text-white">{cfg.human_loop.gold_sample_rate}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400 text-xs">SBC α (Bonferroni)</span>
                  <span className="font-mono text-white">{cfg.sbc.ks_alpha}</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* F1–F15 battery */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Falsification battery — F1–F15
        </h2>
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="border-b border-slate-800 text-xs text-slate-500">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium w-12">ID</th>
                <th className="px-4 py-2.5 text-left font-medium">Test</th>
                <th className="px-4 py-2.5 text-left font-medium">Layer</th>
                <th className="px-4 py-2.5 text-left font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {FALSIFICATION_TESTS.map((t) => {
                const ledgerEntry = ledger?.tests?.[t.id] ?? ledger?.[t.id];
                const status: string = ledgerEntry?.gate_status ?? "pending";
                return (
                  <tr key={t.id} className="hover:bg-slate-800/30">
                    <td className="px-4 py-2.5 font-mono text-xs font-bold text-slate-500">{t.id}</td>
                    <td className="px-4 py-2.5 text-slate-200">{t.name}</td>
                    <td className="px-4 py-2.5">
                      <span className={`rounded px-2 py-0.5 text-[11px] font-medium ${layerColor[t.layer] ?? "text-slate-400"}`}>
                        {t.layer}
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <span className={`text-xs font-semibold ${
                        status === "pass"    ? "text-emerald-400" :
                        status === "fail"    ? "text-red-400"     :
                        status === "skip"    ? "text-amber-400"   :
                                              "text-slate-600"
                      }`}>
                        {status === "pending" ? "not run" : status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!ledger && (
          <p className="mt-2 text-xs text-slate-600">
            Run <code className="text-sky-500">david falsify</code> to populate the battery results.
          </p>
        )}
      </section>

      {/* Automation guarantee */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Automation guarantee
        </h2>
        <div className="rounded-xl border border-emerald-800/30 bg-emerald-950/10 p-5">
          <div className="grid gap-4 sm:grid-cols-3 text-sm">
            {[
              { label: "Human input",        value: "Queue only",    note: "minority_share > threshold" },
              { label: "Auto-adjudication",  value: "Unanimous",     note: "all M coders agree" },
              { label: "Gate enforcement",   value: "Fail-closed",   note: "never relaxed at runtime" },
            ].map(({ label, value, note }) => (
              <div key={label} className="space-y-1">
                <p className="text-xs text-slate-500">{label}</p>
                <p className="font-semibold text-emerald-300">{value}</p>
                <p className="text-xs text-slate-500">{note}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
