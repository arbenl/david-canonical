import { api } from "@/lib/api";
import { StatCard } from "@/components/stat-card";
import { AdjudicatorQueue } from "@/components/adjudicator-queue";

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

  const totalEvidence    = strata.reduce((s, x) => s + x.n_evidence,    0);
  const totalAdjudicated = strata.reduce((s, x) => s + x.n_adjudicated, 0);
  const totalPending     = totalEvidence - totalAdjudicated;

  return (
    <div className="space-y-8 max-w-5xl">
      {/* ── header ─────────────────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold text-white">Evidence Pipeline</h1>
        <p className="mt-1 text-sm text-slate-400">
          Sources → scrape → normalize → LLM code → adjudicate → Bayesian fit
        </p>
      </div>

      {/* ── stats ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard label="Total evidence"    value={totalEvidence}    sub="all scraped items" />
        <StatCard label="Adjudicated"       value={totalAdjudicated} sub="ready for fit"
                  accent={totalAdjudicated > 0} />
        <StatCard label="Pending"           value={totalPending}     sub="awaiting coding"
                  warn={totalPending > 0} />
        <StatCard label="Human queue"       value={queue?.n_in_queue ?? 0}
                  sub="need your review" warn={(queue?.n_in_queue ?? 0) > 0} />
      </div>

      {/* ── adjudicator queue (interactive) ────────────────────────────── */}
      {(queue?.n_in_queue ?? 0) > 0 && (
        <section>
          <div className="mb-3 flex items-center justify-between gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-500">
              Adjudicator queue — {queue!.n_in_queue} item{queue!.n_in_queue !== 1 ? "s" : ""}
            </h2>
            <span className="text-[11px] text-slate-600">
              Each review takes ~{queue!.items[0]?.estimated_minutes ?? 4} min · improves Bayesian fit accuracy
            </span>
          </div>
          <AdjudicatorQueue
            items={queue!.items.slice(0, 20)}
            threshold={queue!.disagreement_threshold ?? 0.3}
          />
          <p className="mt-2 text-[11px] text-slate-600">
            After adjudicating, click{" "}
            <span className="text-slate-400 font-medium">⚙ Run Fit</span> on the Command Center
            to update the Bayesian model.
          </p>
        </section>
      )}

      {/* ── no queue state ─────────────────────────────────────────────── */}
      {(queue?.n_in_queue ?? 0) === 0 && totalAdjudicated > 0 && (
        <div className="rounded-xl border border-emerald-700/30 bg-emerald-950/10 px-5 py-4">
          <p className="text-sm font-semibold text-emerald-400">
            ✓ No items in the human queue
          </p>
          <p className="mt-0.5 text-xs text-emerald-700">
            All coded items have been auto-adjudicated (coders agreed) or have been reviewed.
          </p>
        </div>
      )}

      {/* ── strata table ───────────────────────────────────────────────── */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Strata ({strata.length})
        </h2>
        {strata.length === 0 ? (
          <p className="text-sm text-slate-500 italic">
            No strata yet — run the pipeline to populate.
          </p>
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
                  const pct = s.n_evidence > 0
                    ? Math.round((s.n_adjudicated / s.n_evidence) * 100) : 0;
                  return (
                    <tr key={s.stratum_id} className="hover:bg-slate-800/30">
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-400">{s.stratum_id}</td>
                      <td className="px-4 py-2.5 text-slate-200">{s.country}</td>
                      <td className="px-4 py-2.5 text-slate-300 capitalize">
                        {s.policy.replace(/_/g, " ")}
                      </td>
                      <td className="px-4 py-2.5 font-mono text-xs text-sky-300">
                        {s.observability.toFixed(2)}
                      </td>
                      <td className="px-4 py-2.5 font-mono text-xs text-slate-300">{s.n_evidence}</td>
                      <td className="px-4 py-2.5 font-mono text-xs text-emerald-300">{s.n_adjudicated}</td>
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 rounded-full bg-slate-800">
                            <div
                              className="h-full rounded-full bg-emerald-500"
                              style={{ width: `${pct}%` }}
                            />
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

      {/* ── recent evidence items ───────────────────────────────────────── */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">
          Recent evidence ({total} total)
        </h2>
        {ev.length === 0 ? (
          <p className="text-sm text-slate-500 italic">
            No evidence yet — run the pipeline to start scraping.
          </p>
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
                    <td className="px-4 py-2 font-mono text-[11px] text-slate-500">
                      {e.evidence_id}
                    </td>
                    <td className="px-4 py-2 text-xs text-slate-400">{e.stratum_id}</td>
                    <td className="px-4 py-2 text-xs text-slate-400">{e.source_id}</td>
                    <td className="px-4 py-2 text-xs text-slate-500">{e.evidence_date}</td>
                    <td className="px-4 py-2 text-xs text-slate-300 max-w-xs truncate">
                      {e.title ?? "—"}
                    </td>
                    <td className="px-4 py-2">
                      <span className={`text-[11px] font-medium ${
                        e.adjudicated ? "text-emerald-400" : "text-slate-600"
                      }`}>
                        {e.adjudicated ? "✓" : "○"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {total > 50 && (
              <p className="border-t border-slate-800 px-4 py-2.5 text-xs text-slate-600">
                Showing 50 of {total} items
              </p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
