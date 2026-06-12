/**
 * E-9: Run-provenance footer.
 *
 * Every page shows the artifact hashes and versions from the latest run:
 * model_version, code_version, data_cutoff, pre_registration_version,
 * theorem_packet_sha. If model_version from the run differs from the current
 * backend model_version (config endpoint), a mismatch warning chip is shown.
 *
 * This is a Server Component — no "use client" needed.
 */

import { getLatestFitRun, getFitSummary, getConfig, getRouteLedger } from "@/lib/data";

function Field({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <span className="flex flex-col gap-px">
      <span className="text-[9px] font-semibold uppercase tracking-widest text-slate-600">{label}</span>
      <span className={`text-[11px] text-slate-500 ${mono ? "font-mono" : ""}`}>{value}</span>
    </span>
  );
}

export async function ProvenanceFooter() {
  const [latestRun, config] = await Promise.all([getLatestFitRun(), getConfig()]);

  const summary = latestRun ? await getFitSummary(latestRun.run_id) : null;
  const ledger  = latestRun ? await getRouteLedger(latestRun.run_id) : null;

  const runModelVersion    = latestRun?.model_version      ?? null;
  const currentModelVersion = config?.model_version        ?? null;
  const codeVersion        = summary?.code_version         ?? (process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 8) ?? null);
  const dataCutoff         = summary?.data_cutoff          ?? null;
  const preRegVersion      = summary?.pre_registration_version ?? null;
  const theoremPacketSha   = summary?.theorem_packet_sha   ?? null;
  const ledgerTimestamp    = ledger?.timestamp              ?? null;

  // Mismatch: run was produced with a different model version than the current backend.
  const modelMismatch =
    runModelVersion !== null &&
    currentModelVersion !== null &&
    runModelVersion !== currentModelVersion;

  if (!latestRun) {
    return (
      <footer className="mt-auto border-t border-slate-800/50 px-8 py-3">
        <p className="text-[10px] text-slate-700 font-mono">DAVID/M0.1 — no run artifact</p>
      </footer>
    );
  }

  return (
    <footer className="mt-auto border-t border-slate-800/50 bg-slate-950/60 px-8 py-3">
      <div className="flex flex-wrap items-start gap-x-6 gap-y-2">

        {/* Run ID */}
        <Field label="run" value={latestRun.run_id} />

        {/* Model version — warning if stale */}
        <span className="flex flex-col gap-px">
          <span className="text-[9px] font-semibold uppercase tracking-widest text-slate-600">model</span>
          <span className="flex items-center gap-1.5">
            <span className="font-mono text-[11px] text-slate-500">{runModelVersion ?? "—"}</span>
            {modelMismatch && (
              <span
                className="rounded px-1.5 py-px text-[9px] font-bold uppercase tracking-wide bg-amber-900/40 text-amber-400 border border-amber-700/40"
                title={`Run used ${runModelVersion}; current backend is ${currentModelVersion}`}
              >
                stale
              </span>
            )}
          </span>
        </span>

        {/* Code version */}
        <Field label="code" value={codeVersion ?? "—"} />

        {/* Data cutoff */}
        <Field label="data cutoff" value={dataCutoff ?? "—"} mono={false} />

        {/* Pre-registration version */}
        <Field label="pre-reg" value={preRegVersion ?? "—"} />

        {/* Theorem packet SHA */}
        <Field label="theorem pkg" value={theoremPacketSha ? theoremPacketSha.slice(0, 12) : "—"} />

        {/* Ledger timestamp */}
        {ledgerTimestamp && (
          <Field label="ledger at" value={String(ledgerTimestamp).slice(0, 19).replace("T", " ")} mono={false} />
        )}

        {/* Gate status pill */}
        <span className="flex flex-col gap-px">
          <span className="text-[9px] font-semibold uppercase tracking-widest text-slate-600">gate</span>
          <span className={`text-[11px] font-semibold ${
            latestRun.gate_status === "pass" ? "text-emerald-500" :
            latestRun.gate_status === "fail" ? "text-red-500"     :
            "text-slate-500"
          }`}>
            {latestRun.gate_status}
          </span>
        </span>

      </div>
    </footer>
  );
}
