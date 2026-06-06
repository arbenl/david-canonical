import { api } from "@/lib/api";
import { StatCard } from "@/components/stat-card";

export const dynamic = "force-dynamic";

export default async function EvidencePage() {
  const [strataRes, queueRes, evRes] = await Promise.allSettled([
    api.strata(),
    api.queue(),
    api.evidence({ limit: 50 }),
  ]);

  const strata = strataRes.status === "fulfilled" ? strataRes.value.strata : [];
  const queue  = queueRes.status  === "fulfilled" ? queueRes.value         : null;
  const ev     = evRes.status     === "fulfilled" ? evRes.value.evidence   : [];
  const total  = evRes.status     === "fulfilled" ? evRes.value.total      : 0;

  const totalEvidence    = strata.reduce((s, x) => s + x.n_evidence, 0);
  const totalAdjudicated = strata.reduce((s, x) => s + x.n_adjudicated, 0);
  const totalPending     = totalEvidence - totalAdjudicated;

  return (
    <div className="space-y-8 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Evidence Pipeline</h1>
        <p className="mt-1 text-sm text-slate-400">
          Sources → scrape → normalize → LLM code → auto-adjudicate
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Total evidence"   value={totalEvidence}    sub="all items" />
        <StatCard label="Adjudicated"      value={totalAdjudicated} sub="ready for fit" accent={totalAdjudicated > 0} />
        <StatCard label="Pending coding"   value={totalPending}     sub="not yet adjudicated" warn={totalPending > 0} />
        <StatCard label="Queue (human)"    value={queue?.n_in_queue ?? 0} sub="need review" warn={(queue?.n_in_queue ?? 0) > 0} />
      </div>

      {/* Adjudicator queue */}
      {(queue?.n_in_queue ?? 0) > 0 && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
            Adjudicator queue — {queue!.n_in_queue} item{queue!.n_in_queue !== 1 ? "s" : ""}
          </h2>
          <div className="rounded-xl border border-amber-700/30 bg-amber-950/10 overflow-hidden">
            <div className="border-b border-amber-800/30 px-4 py-2.5 text-xs text-amber-500/80">
              Disagreement threshold: {queue!.disagreement_threshold ?? 0.30} ·
              Ranked by Expected Information Gain (EIG)
            </div>
            <table className="w-full text-sm">
              <thead className="text-xs text-slate-500">
                <tr>
                  {["Evidence ID", "Reason", "Worst tactic", "Minority share", "EIG", "Est. min"].map((h) => (
                    <th key={h} className="px-4 py-2.5 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40">
                {queue!.items.slice(0, 20).map((item) => (
                  <tr key={item.evidence_id} className="hover:bg-amber-950/20">
                    <td className="px-4 py-2 font-mono text-xs text-slate-400">{item.evidence_id}</td>
                    <td className="px-4 py-2">
                      <span className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
                        item.reason === "disagree" ? "bg-red-950/50 text-red-300" : "bg-slate-800 text-slate-400"
                      }`}>{item.reason}</span>
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-slate-300">{item.worst_tactic || "—"}</td>
                    <td className="px-4 py-2 font-mono text-xs">
                      <span className={item.minority_share > 0.4 ? "text-red-400" : "text-amber-400"}>
                        {item.minority_share?.toFixed(2) ?? "—"}
                      </span>
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-slate-300">{item.eig_score.toFixed(2)}</td>
                    <td className="px-4 py-2 text-xs text-slate-500">{item.estimated_minutes} min</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-600">
            To adjudicate: <span className="font-mono">UPDATE evidence_items SET adjudicated=TRUE WHERE evidence_id=&apos;…&apos;</span>
          </p>
        </section>
      )}

      {/* Strata table */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Strata ({strata.length})
        </h2>
        {strata.length === 0 ? (
          <p className="text-sm text-slate-500">No strata yet. Run `david ingest` to populate.</p>
        ) : (
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-800 text-xs text-slate-500">
                <tr>
                  {["Stratum", "Country", "Policy", "I(O)", "Evidence", "Adjudicated", "Adj %"].map((h) => (
                    <th key={h} className="px-4 py-2.5 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {strata.map((s) => {
                  const pct = s.n_evidence > 0 ? Math.round((s.n_adjudicated / s.n_evidence) * 100) : 0;
                  return (
                    <tr key={s.stratum_id} className="hover:bg-slate-800/30">
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-400">{s.stratum_id}</td>
                      <td className="px-4 py-2.5 text-slate-200">{s.country}</td>
                      <td className="px-4 py-2.5 text-slate-300">{s.policy}</td>
                      <td className="px-4 py-2.5 font-mono text-xs text-sky-300">{s.observability.toFixed(2)}</td>
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-300">{s.n_evidence}</td>
                      <td className="px-4 py-2.5 font-mono text-xs text-emerald-300">{s.n_adjudicated}</td>
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 rounded-full bg-slate-800">
                            <div className="h-full rounded-full bg-emerald-500" style={{ width: `${pct}%` }} />
                          </div>
                          <span className="text-xs text-slate-400">{pct}%</span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Evidence items */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Recent evidence ({total} total)
        </h2>
        {ev.length === 0 ? (
          <p className="text-sm text-slate-500">No evidence yet. Run `david ingest`.</p>
        ) : (
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-800 text-xs text-slate-500">
                <tr>
                  {["ID", "Stratum", "Source", "Date", "Title", "Adj."].map((h) => (
                    <th key={h} className="px-4 py-2.5 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {ev.map((e) => (
                  <tr key={e.evidence_id} className="hover:bg-slate-800/30">
                    <td className="px-4 py-2 font-mono text-[11px] text-slate-500">{e.evidence_id}</td>
                    <td className="px-4 py-2 text-xs text-slate-400">{e.stratum_id}</td>
                    <td className="px-4 py-2 text-xs text-slate-400">{e.source_id}</td>
                    <td className="px-4 py-2 text-xs text-slate-500">{e.evidence_date}</td>
                    <td className="px-4 py-2 text-xs text-slate-300 max-w-xs truncate">{e.title ?? "—"}</td>
                    <td className="px-4 py-2">
                      <span className={`text-[11px] font-medium ${e.adjudicated ? "text-emerald-400" : "text-slate-600"}`}>
                        {e.adjudicated ? "✓" : "○"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
