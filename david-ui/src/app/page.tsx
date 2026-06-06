import { api } from "@/lib/api";
import { StatCard } from "@/components/stat-card";
import Link from "next/link";

const statusColor: Record<string, string> = {
  pass:  "text-emerald-400",
  fail:  "text-red-400",
  skip:  "text-amber-400",
  error: "text-red-400",
};

export const dynamic = "force-dynamic";

export default async function CommandCenter() {
  const [runsRes, strataRes, queueRes] = await Promise.allSettled([
    api.fitRuns(5),
    api.strata(),
    api.queue(),
  ]);

  const runs   = runsRes.status   === "fulfilled" ? runsRes.value.fit_runs     : [];
  const strata = strataRes.status === "fulfilled" ? strataRes.value.strata      : [];
  const queue  = queueRes.status  === "fulfilled" ? queueRes.value             : null;

  const latestRun = runs[0];
  const totalEvidence    = strata.reduce((s, x) => s + x.n_evidence, 0);
  const totalAdjudicated = strata.reduce((s, x) => s + x.n_adjudicated, 0);
  const adjPct = totalEvidence > 0
    ? Math.round((totalAdjudicated / totalEvidence) * 100)
    : 0;

  const apiOk = runsRes.status === "fulfilled" || strataRes.status === "fulfilled";

  return (
    <div className="space-y-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Command Center</h1>
          <p className="mt-1 text-sm text-slate-400">
            Pipeline status · {new Date().toLocaleDateString("en-GB", { dateStyle: "long" })}
          </p>
        </div>
        <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium ${
          apiOk ? "bg-emerald-950 text-emerald-300" : "bg-red-950 text-red-300"
        }`}>
          <span className={`h-1.5 w-1.5 rounded-full ${apiOk ? "bg-emerald-400" : "bg-red-400"}`} />
          {apiOk ? "FastAPI connected" : "FastAPI unreachable"}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Evidence items"    value={totalEvidence}    sub={`${totalAdjudicated} adjudicated`} />
        <StatCard label="Adjudicated %"     value={`${adjPct}%`}     sub="auto + human" accent={adjPct > 50} />
        <StatCard label="Queue (human)"     value={queue?.n_in_queue ?? "—"}
                  sub="items awaiting review" warn={(queue?.n_in_queue ?? 0) > 0} />
        <StatCard label="Strata"            value={strata.length}    sub="country × policy" />
      </div>

      {/* Queue alert */}
      {(queue?.n_in_queue ?? 0) > 0 && (
        <div className="rounded-xl border border-amber-700/40 bg-amber-950/20 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-amber-300">
                {queue!.n_in_queue} item{queue!.n_in_queue !== 1 ? "s" : ""} need human review
              </p>
              <p className="mt-0.5 text-xs text-amber-500/70">
                Inter-coder disagreement above {queue!.disagreement_threshold ?? 0.30} threshold
              </p>
            </div>
            <Link
              href="/evidence"
              className="rounded-lg bg-amber-800/40 px-3 py-1.5 text-xs font-semibold text-amber-200 hover:bg-amber-800/60"
            >
              Review queue →
            </Link>
          </div>
        </div>
      )}

      {/* Latest fit */}
      {latestRun && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">Latest fit</h2>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="font-mono text-xs text-slate-500">{latestRun.run_id}</p>
                <p className="mt-1 font-semibold text-white">{latestRun.model_version}</p>
              </div>
              <span className={`text-sm font-bold ${statusColor[latestRun.gate_status] ?? "text-slate-400"}`}>
                {latestRun.gate_status.toUpperCase()}
              </span>
            </div>
            <div className="mt-4 grid grid-cols-4 gap-3">
              {[
                { k: "R̂ max",         v: latestRun.rhat_max?.toFixed(3)   },
                { k: "ESS bulk",      v: latestRun.ess_bulk_min?.toFixed(0) },
                { k: "Divergences",   v: latestRun.divergences             },
                { k: "Started",       v: String(latestRun.started_at).slice(0, 10) },
              ].map(({ k, v }) => (
                <div key={k}>
                  <p className="text-[11px] text-slate-500">{k}</p>
                  <p className="mt-0.5 font-mono text-sm text-white">{v ?? "—"}</p>
                </div>
              ))}
            </div>
            <div className="mt-4 flex gap-3">
              <Link href="/engine"   className="btn-sm">View theorem gates</Link>
              <Link href="/predict"  className="btn-sm">View forecasts</Link>
              <Link href="/validate" className="btn-sm">Validation battery</Link>
            </div>
          </div>
        </section>
      )}

      {/* Fit run history */}
      {runs.length > 1 && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">Fit history</h2>
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
                    <td className={`px-4 py-2.5 font-semibold ${statusColor[r.gate_status] ?? ""}`}>
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

      {/* Pipeline stages */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">Pipeline</h2>
        <div className="flex items-center gap-0">
          {[
            { label: "Scrape", href: "/evidence",  done: totalEvidence > 0 },
            { label: "Code",   href: "/evidence",  done: totalAdjudicated > 0 },
            { label: "Fit",    href: "/engine",    done: !!latestRun },
            { label: "Forecast", href: "/predict", done: !!latestRun && latestRun.gate_status === "pass" },
            { label: "Validate", href: "/validate",done: false },
          ].map((stage, i, arr) => (
            <div key={stage.label} className="flex items-center">
              <Link
                href={stage.href}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium transition-colors ${
                  stage.done
                    ? "bg-emerald-950/50 text-emerald-300 border border-emerald-800/40"
                    : "bg-slate-900 text-slate-500 border border-slate-800"
                }`}
              >
                <span>{stage.done ? "✓" : "○"}</span>
                {stage.label}
              </Link>
              {i < arr.length - 1 && (
                <span className="mx-1 text-slate-700">→</span>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
