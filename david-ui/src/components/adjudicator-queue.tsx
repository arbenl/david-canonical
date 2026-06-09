"use client";

import { useState } from "react";
import type { QueueItem } from "@/lib/api";

// ── types ─────────────────────────────────────────────────────────────────────

interface CoderLabel {
  coder_id: string;
  tactic_k: string;
  label: number;   // 0 | 1
  coded_at: string;
}

interface EvidenceDetail {
  evidence_id: string;
  stratum_id: string;
  source_id: string;
  evidence_date: string;
  title: string | null;
  url: string | null;
  text_content: string | null;
  adjudicated: boolean;
}

interface EvidenceResponse {
  evidence: EvidenceDetail;
  labels: CoderLabel[];
  n_labels: number;
}

// ── constants ─────────────────────────────────────────────────────────────────

const TACTICS = ["SIO", "MIO", "CSIO"] as const;
type Tactic = (typeof TACTICS)[number];

const TACTIC_QUESTION: Record<Tactic, string> = {
  SIO:  "Did state institutions (ministries, regulators, legislators) act in ways that obstruct tobacco control policy?",
  MIO:  "Did media or information channels promote industry-favourable framing, spread disinformation, or suppress evidence?",
  CSIO: "Did civil-society bodies (NGOs, academic groups, professional associations) obstruct or water down this tobacco policy?",
};

const TACTIC_HINT: Record<Tactic, string> = {
  SIO:  "State Institution Obstruction — look for: delayed regulations, weakened amendments, captured agencies",
  MIO:  "Media / Information Obstruction — look for: industry-funded studies cited, policy-discrediting narratives",
  CSIO: "Civil-Society Institution Obstruction — look for: front groups posing as civil society, captured NGOs",
};

// ── helpers ───────────────────────────────────────────────────────────────────

/** Shorten coder_id to a readable label. */
function coderShortName(id: string): string {
  if (id === "human_adjudicator_001") return "Human";
  // groq_llama31_8b_seed_001 → Coder A  (seed 1→A, 2→B, etc.)
  const m = id.match(/_seed_0*(\d+)$/);
  if (m) return `Coder ${String.fromCharCode(64 + Number(m[1]))}`;
  return id.slice(0, 12);
}

/** Given tactic labels from all coders, compute majority vote (0 or 1). */
function majorityVote(labels: CoderLabel[], tactic: Tactic): number {
  const relevant = labels.filter((l) => l.tactic_k === tactic);
  const yes = relevant.filter((l) => l.label === 1).length;
  return yes > relevant.length / 2 ? 1 : 0;
}

// ── component ─────────────────────────────────────────────────────────────────

interface Props {
  items: QueueItem[];
  threshold: number;
}

export function AdjudicatorQueue({ items, threshold }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [detail,   setDetail]   = useState<Record<string, EvidenceResponse>>({});
  const [loading,  setLoading]  = useState<Record<string, boolean>>({});
  // decisions[evidenceId][tactic] = 0 | 1
  const [decisions, setDecisions] = useState<Record<string, Record<string, number>>>({});
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [done,       setDone]       = useState<Set<string>>(new Set());
  const [errors,     setErrors]     = useState<Record<string, string>>({});

  // ── expand / collapse ──────────────────────────────────────────────────────

  async function toggleRow(evidenceId: string) {
    if (expanded === evidenceId) { setExpanded(null); return; }
    setExpanded(evidenceId);
    if (detail[evidenceId]) return; // already fetched

    setLoading((l) => ({ ...l, [evidenceId]: true }));
    try {
      const res = await fetch(`/api/evidence/${evidenceId}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const data: EvidenceResponse = await res.json();
      setDetail((d) => ({ ...d, [evidenceId]: data }));
      // Pre-fill decisions from majority vote
      const init: Record<string, number> = {};
      for (const t of TACTICS) init[t] = majorityVote(data.labels, t);
      setDecisions((d) => ({ ...d, [evidenceId]: init }));
    } catch (e) {
      setErrors((err) => ({ ...err, [evidenceId]: `Load failed: ${e}` }));
    } finally {
      setLoading((l) => ({ ...l, [evidenceId]: false }));
    }
  }

  // ── tactic decision ────────────────────────────────────────────────────────

  function decide(evidenceId: string, tactic: string, val: number) {
    setDecisions((d) => ({
      ...d,
      [evidenceId]: { ...(d[evidenceId] ?? {}), [tactic]: val },
    }));
  }

  // ── submit ─────────────────────────────────────────────────────────────────

  async function submit(evidenceId: string) {
    setSubmitting(evidenceId);
    setErrors((e) => ({ ...e, [evidenceId]: "" }));
    try {
      const labels = decisions[evidenceId] ?? {};
      const res = await fetch("/api/adjudicate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ evidence_id: evidenceId, labels }),
      });
      if (res.ok) {
        setDone((s) => new Set([...s, evidenceId]));
        setExpanded(null);
      } else {
        const data = await res.json();
        setErrors((e) => ({ ...e, [evidenceId]: data.error ?? "Submit failed" }));
      }
    } catch (e) {
      setErrors((err) => ({ ...err, [evidenceId]: String(e) }));
    } finally {
      setSubmitting(null);
    }
  }

  // ── render ─────────────────────────────────────────────────────────────────

  const visible = items.filter((item) => !done.has(item.evidence_id));

  if (visible.length === 0 && done.size > 0) {
    return (
      <div className="rounded-xl border border-emerald-700/30 bg-emerald-950/10 px-6 py-8 text-center">
        <p className="text-lg text-emerald-400 font-semibold">✓ All items adjudicated!</p>
        <p className="mt-1 text-sm text-emerald-700">
          Click <span className="font-medium text-emerald-500">⚙ Run Fit</span> to update the Bayesian model with your labels.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* ── Udhëzimet e Adjudikimit (Premium Glassmorphism Card) ── */}
      <div className="rounded-xl border border-indigo-700/20 bg-indigo-950/10 px-5 py-4 text-slate-300">
        <h4 className="text-sm font-semibold text-indigo-400 flex items-center gap-2 mb-1.5">
          <span>ℹ️</span> Udhëzimet e Adjudikimit (Human-in-the-Loop)
        </h4>
        <p className="text-xs leading-relaxed text-slate-400 mb-3">
          Si ekspert (adjudicator), detyra juaj është të zgjidhni mospajtimet midis koduesve inteligjentë (AI). Përzgjedhja juaj përditëson automatikisht bazën e të dhënave dhe skedarin e kalibrimit <code className="text-indigo-300 font-mono bg-indigo-900/30 px-1 py-0.5 rounded text-[10px]">gold_b_calibration.csv</code>.
        </p>
        <div className="grid gap-3 sm:grid-cols-3 text-xs">
          <div className="rounded-lg bg-slate-900/50 border border-slate-800/60 p-2.5">
            <span className="font-bold text-slate-200 block mb-0.5">1. Lexo & Krahaso</span>
            <span className="text-slate-500 leading-normal">Kliko mbi rresht për të parë detajet dhe krahaso votat (YES/NO) të koduesve AI.</span>
          </div>
          <div className="rounded-lg bg-slate-900/50 border border-slate-800/60 p-2.5">
            <span className="font-bold text-slate-200 block mb-0.5">2. Zgjidh Statusin</span>
            <span className="text-slate-500 leading-normal">Për secilën taktikë, përcakto YES ose NO bazuar në pyetjen orientuese.</span>
          </div>
          <div className="rounded-lg bg-slate-900/50 border border-slate-800/60 p-2.5">
            <span className="font-bold text-slate-200 block mb-0.5">3. Mirato Adjudikimin</span>
            <span className="text-slate-500 leading-normal">Kliko <strong>Submit adjudication</strong> për të regjistruar "Standardin e Artë".</span>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-amber-700/30 bg-amber-950/10 overflow-hidden">
        {/* ── section header ──────────────────────────────────────────────── */}
        <div className="border-b border-amber-800/30 px-4 py-2.5 flex items-center justify-between flex-wrap gap-2">
          <span className="text-xs text-amber-500/80">
            Disagreement threshold: {threshold} · Ranked by Expected Information Gain (EIG)
            <span className="ml-2 text-amber-700/60">— click a row to adjudicate</span>
          </span>
          {done.size > 0 && (
            <span className="text-xs text-emerald-500 font-medium">
              {done.size} adjudicated this session ✓
            </span>
          )}
        </div>

        {/* ── table ───────────────────────────────────────────────────────── */}
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-500 border-b border-slate-800/30">
            <tr>
              <th className="px-4 py-2.5 text-left font-medium w-px" />
              <th className="px-4 py-2.5 text-left font-medium">Evidence ID</th>
              <th className="px-4 py-2.5 text-left font-medium">Reason</th>
              <th className="px-4 py-2.5 text-left font-medium">Worst tactic</th>
              <th className="px-4 py-2.5 text-left font-medium">Minority share</th>
              <th className="px-4 py-2.5 text-left font-medium">EIG</th>
              <th className="px-4 py-2.5 text-left font-medium">Est. min</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((item) => {
              const isExpanded  = expanded === item.evidence_id;
              const isLoading   = loading[item.evidence_id] ?? false;
              const isSubmitting = submitting === item.evidence_id;
              const itemDetail  = detail[item.evidence_id];
              const itemDecisions = decisions[item.evidence_id] ?? {};
              const itemError   = errors[item.evidence_id];

              return [
                // ── summary row ──────────────────────────────────────────
                <tr
                  key={item.evidence_id}
                  onClick={() => toggleRow(item.evidence_id)}
                  className={`cursor-pointer transition-colors ${
                    isExpanded
                      ? "bg-amber-950/30"
                      : "hover:bg-amber-950/20"
                  }`}
                >
                  <td className="px-3 py-2.5 text-amber-500/50 text-xs select-none">
                    {isExpanded ? "▼" : "▶"}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-400">
                    {item.evidence_id}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className="rounded px-1.5 py-0.5 text-[11px] font-medium bg-red-950/50 text-red-300">
                      {item.reason}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-300">
                    {item.worst_tactic || "—"}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs">
                    <span className={item.minority_share > 0.4 ? "text-red-400" : "text-amber-400"}>
                      {item.minority_share?.toFixed(2) ?? "—"}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-300">
                    {item.eig_score.toFixed(2)}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-slate-500">
                    {item.estimated_minutes} min
                  </td>
                </tr>,

                // ── expanded adjudication panel ──────────────────────────
                isExpanded && (
                  <tr key={`${item.evidence_id}-panel`}>
                    <td colSpan={7} className="px-0 py-0 border-t border-amber-800/20">
                      <div className="bg-slate-950/80 px-5 py-5 space-y-5">

                        {isLoading && (
                          <p className="text-xs text-slate-500 animate-pulse">
                            Loading article details…
                          </p>
                        )}

                        {itemError && !isLoading && (
                          <p className="text-xs text-red-400">{itemError}</p>
                        )}

                        {itemDetail && !isLoading && (
                          <>
                            {/* ── article info ──────────────────────────── */}
                            <div>
                              <p className="font-semibold text-slate-100 leading-snug">
                                {itemDetail.evidence.title ?? "(no title)"}
                              </p>
                              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                <span>{itemDetail.evidence.source_id}</span>
                                <span className="text-slate-700">·</span>
                                <span>{itemDetail.evidence.evidence_date}</span>
                                <span className="text-slate-700">·</span>
                                <span className="font-mono text-slate-600">
                                  {itemDetail.evidence.stratum_id}
                                </span>
                                {itemDetail.evidence.url && (
                                  <>
                                    <span className="text-slate-700">·</span>
                                    <a
                                      href={itemDetail.evidence.url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="text-sky-500 hover:text-sky-400 transition-colors"
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      Open article ↗
                                    </a>
                                  </>
                                )}
                              </div>
                              {itemDetail.evidence.text_content && (
                                <p className="mt-2 max-h-28 overflow-y-auto rounded border
                                              border-slate-800 bg-slate-900/60 px-3 py-2
                                              text-xs leading-relaxed text-slate-400">
                                  {itemDetail.evidence.text_content.slice(0, 700)}
                                  {itemDetail.evidence.text_content.length > 700 && "…"}
                                </p>
                              )}
                            </div>

                            {/* ── tactic label panel ────────────────────── */}
                            <div>
                              <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                                Coder labels · your adjudication
                              </p>
                              <div className="space-y-2">
                                {TACTICS.map((tactic) => {
                                  const tacticLabels = itemDetail.labels.filter(
                                    (l) => l.tactic_k === tactic
                                  );
                                  const uniqueCoders = [
                                    ...new Set(tacticLabels.map((l) => l.coder_id)),
                                  ].filter((id) => id !== "human_adjudicator_001");

                                  const allAgree =
                                    uniqueCoders.length > 1 &&
                                    tacticLabels
                                      .filter((l) => uniqueCoders.includes(l.coder_id))
                                      .every((l) => l.label === tacticLabels[0].label);

                                  const isWorst = tactic === item.worst_tactic;
                                  const myVal   = itemDecisions[tactic];

                                  return (
                                    <div
                                      key={tactic}
                                      className={`grid gap-3 rounded-lg px-4 py-3 ${
                                        isWorst
                                          ? "bg-amber-950/40 border border-amber-800/30"
                                          : "bg-slate-900/60 border border-slate-800/40"
                                      }`}
                                      style={{ gridTemplateColumns: "180px 1fr auto" }}
                                    >
                                      {/* Tactic name + hint */}
                                      <div>
                                        <p className="text-xs font-bold text-slate-200">
                                          {tactic}
                                          {isWorst && (
                                            <span className="ml-2 text-[10px] text-amber-500">
                                              ⚠ most disagreement
                                            </span>
                                          )}
                                        </p>
                                        <p
                                          className="mt-0.5 text-[10px] text-slate-600 leading-snug"
                                          title={TACTIC_HINT[tactic]}
                                        >
                                          {TACTIC_QUESTION[tactic]}
                                        </p>
                                      </div>

                                      {/* Coder signals */}
                                      <div className="flex flex-wrap items-center gap-2 self-center">
                                        {uniqueCoders.map((coderId) => {
                                          const lbl = tacticLabels.find(
                                            (l) => l.coder_id === coderId
                                          );
                                          return (
                                            <span
                                              key={coderId}
                                              title={coderId}
                                              className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                                                lbl?.label === 1
                                                  ? "bg-rose-900/50 text-rose-300 border border-rose-800/50"
                                                  : "bg-slate-800 text-slate-400 border border-slate-700"
                                              }`}
                                            >
                                              {coderShortName(coderId)}: {lbl?.label === 1 ? "YES" : "NO"}
                                            </span>
                                          );
                                        })}
                                        {uniqueCoders.length === 0 && (
                                          <span className="text-[11px] text-slate-600">
                                            no labels yet
                                          </span>
                                        )}
                                        {uniqueCoders.length > 1 && (
                                          <span
                                            className={`text-[11px] font-medium ${
                                              allAgree ? "text-emerald-500" : "text-amber-400"
                                            }`}
                                          >
                                            {allAgree ? "✓ agree" : "⚠ disagree"}
                                          </span>
                                        )}
                                      </div>

                                      {/* Human YES/NO buttons */}
                                      <div className="flex items-center gap-1.5 self-center shrink-0">
                                        <span className="text-[10px] text-slate-600 mr-1">You:</span>
                                        <button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            decide(item.evidence_id, tactic, 1);
                                          }}
                                          className={`rounded px-3 py-1.5 text-xs font-semibold transition-colors ${
                                            myVal === 1
                                              ? "bg-rose-700 text-rose-100 ring-1 ring-rose-500"
                                              : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
                                          }`}
                                        >
                                          YES
                                        </button>
                                        <button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            decide(item.evidence_id, tactic, 0);
                                          }}
                                          className={`rounded px-3 py-1.5 text-xs font-semibold transition-colors ${
                                            myVal === 0
                                              ? "bg-slate-500 text-slate-100 ring-1 ring-slate-400"
                                              : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
                                          }`}
                                        >
                                          NO
                                        </button>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>

                            {/* ── submit / cancel ───────────────────────── */}
                            <div className="flex items-center gap-3 pt-1">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  submit(item.evidence_id);
                                }}
                                disabled={isSubmitting}
                                className={`rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
                                  isSubmitting
                                    ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                                    : "bg-indigo-600 hover:bg-indigo-500 text-white"
                                }`}
                              >
                                {isSubmitting ? "Saving…" : "✓ Submit adjudication"}
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setExpanded(null);
                                }}
                                className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                              >
                                Cancel
                              </button>
                              {errors[item.evidence_id] && (
                                <span className="text-xs text-red-400">
                                  {errors[item.evidence_id]}
                                </span>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                ),
              ];
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

