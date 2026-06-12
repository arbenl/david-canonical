import { getStrata, getFitRuns, getFitSummary, getSbc } from "@/lib/data";
import { RoadmapDiagram, type NodeStatus, type RoadmapStatuses } from "@/components/roadmap-diagram";

export const dynamic = "force-dynamic";

const PASS = "pass";
const FAIL = "fail";

/** Map a raw gate_status string onto a node status (pass / fail / pending). */
function gate(v: unknown): NodeStatus {
  if (v === PASS) return "pass";
  if (v === FAIL) return "fail";
  return "pending";
}

/** Pipeline step: done → pass, else optional active, else pending. */
function step(done: boolean, active = false): NodeStatus {
  if (done) return "pass";
  return active ? "active" : "pending";
}

/** Combine two gate statuses; `both` requires both to pass, else any-pass suffices. */
function combine(a: unknown, b: unknown, hasRun: boolean, requireBoth: boolean): NodeStatus {
  if (a === FAIL || b === FAIL) return "fail";
  const passed = requireBoth ? a === PASS && b === PASS : a === PASS || b === PASS;
  if (passed) return "pass";
  return hasRun ? "active" : "pending";
}

function routerStatusOf(theorem: NodeStatus, sbc: NodeStatus): NodeStatus {
  if (theorem === "pass" && sbc === "pass") return "pass";
  if (theorem === "fail" || sbc === "fail") return "fail";
  return "pending";
}

export default async function RoadmapPage() {
  const [strata, runs, sbc] = await Promise.all([
    getStrata(),
    getFitRuns(1),
    getSbc(),
  ]);

  const latestRun = runs[0] ?? null;
  const hasRun    = !!latestRun;

  const totalEvidence    = strata.reduce((s, x) => s + x.n_evidence, 0);
  const totalAdjudicated = strata.reduce((s, x) => s + x.n_adjudicated, 0);

  let aPrime: unknown;
  let bPrime: unknown;
  if (latestRun) {
    const summary = await getFitSummary(latestRun.run_id);
    const th = summary?.theorems ?? {};
    aPrime = th["A_prime"]?.gate_status;
    bPrime = th["B_prime"]?.gate_status;
  }

  const theoremStatus = combine(aPrime, bPrime, hasRun, true);
  const sbcStatus = combine(sbc?.measurement?.gate_status, sbc?.forecast?.gate_status, hasRun, false);

  const statuses: RoadmapStatuses = {
    sources:    step(strata.length > 0),
    coding:     step(totalEvidence > 0),
    adjudicate: step(totalAdjudicated > 0, totalEvidence > 0),
    fit:        latestRun ? gate(latestRun.gate_status) : "pending",
    theorems:   theoremStatus,
    sbc:        sbcStatus,
    router:     routerStatusOf(theoremStatus, sbcStatus),
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
