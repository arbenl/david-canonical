"use client";

import { useState } from "react";

export type NodeStatus = "pass" | "fail" | "pending" | "active";

export interface RoadmapStatuses {
  sources: NodeStatus;
  coding: NodeStatus;
  adjudicate: NodeStatus;
  fit: NodeStatus;
  theorems: NodeStatus;
  sbc: NodeStatus;
  router: NodeStatus;
}

interface NodeDef {
  id: keyof RoadmapStatuses;
  title: string;     // English label
  sub: string;       // short Albanian sublabel
  x: number;
  y: number;
  detailTitle: string;
  detailBody: string;
  formula?: string;
}

const NW = 178, NH = 74;

const NODES: NodeDef[] = [
  {
    id: "sources", title: "1 · RSS Sources", sub: "Burimet e provave", x: 30, y: 30,
    detailTitle: "Mbledhja e burimeve (Scrape)",
    detailBody:
      "Burimet RSS monitorohen vazhdimisht; artikujt filtrohen me fjalë-kyçe të industrisë së duhanit dhe ruhen si njësi provash për çdo stratum (vend × politikë).",
  },
  {
    id: "coding", title: "2 · LLM Coders ×2", sub: "Kodim i pavarur", x: 238, y: 30,
    detailTitle: "Kodimi me dy LLM të pavarur",
    detailBody:
      "Dy kodues LLM strukturalisht të pavarur etiketojnë secilin artikull me taktikat SIO / MIO / CSIO. Pavarësia është parakusht për Teoremën A′.",
  },
  {
    id: "adjudicate", title: "3 · Adjudicate", sub: "Njeriu në qark", x: 446, y: 30,
    detailTitle: "Arbitrimi (Human-in-the-loop)",
    detailBody:
      "Kur koduesit pajtohen njëzëri → auto-arbitrim. Kur mospajtimi e kalon pragun, rasti shkon në radhën njerëzore. Asnjë prag nuk relaksohet në kohë ekzekutimi (fail-closed).",
  },
  {
    id: "fit", title: "4 · Bayesian Fit", sub: "Dawid-Skene", x: 654, y: 30,
    detailTitle: "Përshtatja Bayesiane Dawid-Skene",
    detailBody:
      "Modeli Dawid-Skene agregon etiketat e koduesve duke korrigjuar për saktësinë e secilit, dhe prodhon posteriorë të probabilitetit të ndërhyrjes përmes zinxhirëve MCMC (R̂, ESS, divergjencat).",
    formula: "p(θ | y) ∝ p(y | θ) · p(θ)",
  },
  {
    id: "theorems", title: "5 · Theorem Gates", sub: "A′ B′ C′ D′", x: 654, y: 196,
    detailTitle: "Portat e Teoremave A′–D′",
    detailBody:
      "A′: identifikueshmëri praktike (≥3 burime të pavarura). B′: informativiteti i burimit më të dobët. C′: kontrolli i FDP posterior. D′: vlefshmëria e horizontit h*. Çdo portë duhet të kalojë për të zhbllokuar parashikimin.",
    formula: "A′: #{burime të pavarura} ≥ 3",
  },
  {
    id: "sbc", title: "6 · SBC Calibration", sub: "F1–F15", x: 446, y: 196,
    detailTitle: "Kalibrimi i bazuar në simulim (SBC)",
    detailBody:
      "SBC (Talts et al. 2018) verifikon që statistikat e renditjes janë uniforme — domethënë pasiguria e modelit është e saktë. Bashkë me bateritë F1–F15 formon shtresën e falsifikimit. Nëse SBC dështon, i gjithë motori bllokohet.",
    formula: "rank(θ₀) ~ Uniform{0,…,L}",
  },
  {
    id: "router", title: "7 · Forecast Router", sub: "Parashikimi i certifikuar", x: 238, y: 196,
    detailTitle: "Rrugëzuesi i parashikimit",
    detailBody:
      "Vetëm kur SBC dhe teoremat kalojnë, qelizat e parashikimit (p_active me intervale 80%/95%) zhbllokohen për horizonte 1–24 muaj dhe shfaqet vula: «Parashikim i Certifikuar Matematikisht».",
  },
];

const EDGES: [keyof RoadmapStatuses, keyof RoadmapStatuses][] = [
  ["sources", "coding"],
  ["coding", "adjudicate"],
  ["adjudicate", "fit"],
  ["fit", "theorems"],
  ["theorems", "sbc"],
  ["sbc", "router"],
];

const STATUS_STROKE: Record<NodeStatus, string> = {
  pass: "#34d399", fail: "#f87171", active: "#22d3ee", pending: "#475569",
};
const STATUS_FILL: Record<NodeStatus, string> = {
  pass: "rgba(52,211,153,0.10)", fail: "rgba(248,113,113,0.10)",
  active: "rgba(34,211,238,0.10)", pending: "rgba(15,23,42,0.6)",
};
const STATUS_LABEL: Record<NodeStatus, string> = {
  pass: "kaluar", fail: "dështoi", active: "aktiv", pending: "në pritje",
};

function nodeCenter(n: NodeDef) { return { cx: n.x + NW / 2, cy: n.y + NH / 2 }; }

export function RoadmapDiagram({ statuses }: Readonly<{ statuses: RoadmapStatuses }>) {
  const [selected, setSelected] = useState<keyof RoadmapStatuses>("router");
  const sel = NODES.find((n) => n.id === selected)!;

  const VBW = 860, VBH = 300;

  return (
    <div className="space-y-5">
      <div className="card-premium overflow-hidden p-2">
        <svg viewBox={`0 0 ${VBW} ${VBH}`} className="w-full" style={{ minHeight: 280 }}>
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L8,3 L0,6 Z" fill="#475569" />
            </marker>
            <filter id="node-glow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {/* edges */}
          {EDGES.map(([from, to]) => {
            const a = nodeCenter(NODES.find((n) => n.id === from)!);
            const b = nodeCenter(NODES.find((n) => n.id === to)!);
            // route around: horizontal then vertical for the fit→theorems drop
            const sameRow = Math.abs(a.cy - b.cy) < 4;
            let d: string;
            if (sameRow) {
              const dir = b.cx > a.cx ? 1 : -1;
              d = `M ${a.cx + dir * (NW / 2)} ${a.cy} L ${b.cx - dir * (NW / 2)} ${b.cy}`;
            } else {
              d = `M ${a.cx} ${a.cy + NH / 2} L ${b.cx} ${b.cy - NH / 2}`;
            }
            return <path key={`${from}-${to}`} d={d} stroke="#334155" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />;
          })}

          {/* nodes */}
          {NODES.map((n) => {
            const st = statuses[n.id];
            const active = selected === n.id;
            return (
              <g
                key={n.id}
                role="button"
                tabIndex={0}
                aria-label={`${n.title} — ${STATUS_LABEL[st]}`}
                onClick={() => setSelected(n.id)}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSelected(n.id); } }}
                style={{ cursor: "pointer" }}
              >
                <rect
                  x={n.x} y={n.y} width={NW} height={NH} rx={12}
                  fill={STATUS_FILL[st]} stroke={STATUS_STROKE[st]}
                  strokeWidth={active ? 2.5 : 1.5}
                  filter={active ? "url(#node-glow)" : undefined}
                  opacity={active ? 1 : 0.92}
                />
                <circle cx={n.x + 16} cy={n.y + 16} r="4" fill={STATUS_STROKE[st]} filter="url(#node-glow)" />
                <text x={n.x + 30} y={n.y + 21} fill="#e2e8f0" fontSize="13" fontWeight="600">{n.title}</text>
                <text x={n.x + 16} y={n.y + 44} fill="#94a3b8" fontSize="11">{n.sub}</text>
                <text x={n.x + 16} y={n.y + 62} fill={STATUS_STROKE[st]} fontSize="10" fontWeight="700" style={{ textTransform: "uppercase" }}>
                  {st}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* legend */}
      <div className="flex flex-wrap gap-4 text-[11px] text-slate-400">
        {(["pass", "active", "pending", "fail"] as NodeStatus[]).map((s) => (
          <span key={s} className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: STATUS_STROKE[s] }} />
            {STATUS_LABEL[s]}
          </span>
        ))}
        <span className="text-slate-600">· Kliko çdo nyje për detaje</span>
      </div>

      {/* detail panel for the selected node */}
      <div className="card-premium p-5">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: STATUS_STROKE[statuses[sel.id]], boxShadow: `0 0 8px ${STATUS_STROKE[statuses[sel.id]]}` }} />
          <h3 className="text-base font-bold text-white">{sel.detailTitle}</h3>
          <span className="ml-auto rounded px-2 py-0.5 text-[10px] font-bold uppercase"
            style={{ color: STATUS_STROKE[statuses[sel.id]], background: STATUS_FILL[statuses[sel.id]] }}>
            {statuses[sel.id]}
          </span>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">{sel.detailBody}</p>
        {sel.formula && (
          <p className="mt-3 inline-block rounded-md bg-slate-900/80 px-3 py-1.5 font-mono text-xs text-cyan-300">
            {sel.formula}
          </p>
        )}
      </div>
    </div>
  );
}
