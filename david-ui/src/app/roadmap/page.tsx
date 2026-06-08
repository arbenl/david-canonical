import { api } from "@/lib/api";
import { RoadmapDiagram, type NodeStatus, type RoadmapStatuses } from "@/components/roadmap-diagram";

export const dynamic = "force-dynamic";

function gate(v: unknown): NodeStatus {
  return v === "pass" ? "pass" : v === "fail" ? "fail" : "pending";
}

export default async function RoadmapPage() {
  const [strataRes, runsRes, sbcRes] = await Promise.allSettled([
    api.strata(),
    api.fitRuns(1),
    api.sbc(),
  ]);

  const strata    = strataRes.status === "fulfilled" ? strataRes.value.strata : [];
  const latestRun = runsRes.status   === "fulfilled" ? runsRes.value.fit_runs?.[0] : null;
  const sbc       = sbcRes.status    === "fulfilled" ? sbcRes.value : null;

  const totalEvidence    = strata.reduce((s, x) => s + x.n_evidence, 0);
  const totalAdjudicated = strata.reduce((s, x) => s + x.n_adjudicated, 0);

  let aPrime: unknown = "pending", bPrime: unknown = "pending";
  if (latestRun) {
    const summary = await api.fitRun(latestRun.run_id).catch(() => null);
    const th = summary?.theorems ?? {};
    aPrime = (th["A_prime"] as Record<string, unknown>)?.gate_status;
    bPrime = (th["B_prime"] as Record<string, unknown>)?.gate_status;
  }

  const theoremStatus: NodeStatus =
    aPrime === "fail" || bPrime === "fail" ? "fail"
    : aPrime === "pass" && bPrime === "pass" ? "pass"
    : latestRun ? "active" : "pending";

  const sbcStatus: NodeStatus =
    sbc?.measurement?.gate_status === "fail" || sbc?.forecast?.gate_status === "fail" ? "fail"
    : sbc?.measurement?.gate_status === "pass" || sbc?.forecast?.gate_status === "pass" ? "pass"
    : latestRun ? "active" : "pending";

  const routerStatus: NodeStatus =
    theoremStatus === "pass" && sbcStatus === "pass" ? "pass"
    : theoremStatus === "fail" || sbcStatus === "fail" ? "fail"
    : "pending";

  const statuses: RoadmapStatuses = {
    sources:    strata.length > 0 ? "pass" : "pending",
    coding:     totalEvidence > 0 ? "pass" : "pending",
    adjudicate: totalAdjudicated > 0 ? "pass" : totalEvidence > 0 ? "active" : "pending",
    fit:        latestRun ? gate(latestRun.gate_status) : "pending",
    theorems:   theoremStatus,
    sbc:        sbcStatus,
    router:     routerStatus,
  };

  return (
    <div className="max-w-5xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Arkitektura &amp; Progresi</h1>
        <p className="mt-1 text-sm text-slate-400">
          Architecture &amp; Theory Roadmap — nga burimet RSS te parashikimi i certifikuar matematikisht.
          Ngjyrat pasqyrojnë gjendjen reale të motorit.
        </p>
      </div>

      <RoadmapDiagram statuses={statuses} />
    </div>
  );
}
