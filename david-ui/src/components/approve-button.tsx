"use client";

import { useState } from "react";

interface Props {
  runId: string;
}

export function ApproveButton({ runId }: Props) {
  const [status, setStatus] = useState<"idle" | "loading" | "approved" | "error">("idle");
  const [approvedAt, setApprovedAt] = useState<string | null>(null);

  async function handleApprove() {
    setStatus("loading");
    try {
      const res = await fetch(`/api/forecasts/${runId}/approve`, {
        method: "POST",
        cache: "no-store",
      });
      const data = await res.json();
      if (res.ok && data.status === "approved") {
        setStatus("approved");
        setApprovedAt(data.approved_at ?? null);
      } else {
        console.error("Approve failed:", data);
        setStatus("error");
      }
    } catch (err) {
      console.error("Approve error:", err);
      setStatus("error");
    }
  }

  if (status === "approved") {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-emerald-700/50 bg-emerald-900/30 px-4 py-2">
        <span className="h-2 w-2 rounded-full bg-emerald-400" />
        <span className="text-sm font-medium text-emerald-300">Approved</span>
        {approvedAt && (
          <span className="text-xs text-emerald-500">
            {new Date(approvedAt).toLocaleTimeString()}
          </span>
        )}
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-red-700/50 bg-red-900/20 px-4 py-2">
        <span className="text-sm text-red-300">Approval failed — check PIPELINE_SECRET</span>
        <button
          onClick={() => setStatus("idle")}
          className="ml-2 text-xs text-slate-400 underline hover:text-slate-200"
        >
          retry
        </button>
      </div>
    );
  }

  return (
    <button
      id="approve-headline-btn"
      onClick={handleApprove}
      disabled={status === "loading"}
      className="flex items-center gap-2 rounded-lg border border-amber-600/60 bg-amber-900/20 px-4 py-2 text-sm font-semibold text-amber-300 transition hover:border-amber-500 hover:bg-amber-900/40 disabled:opacity-50"
    >
      {status === "loading" ? (
        <>
          <span className="h-3 w-3 animate-spin rounded-full border-2 border-amber-400 border-t-transparent" />
          Approving…
        </>
      ) : (
        <>
          <span className="text-base">✓</span>
          Approve for Headline
        </>
      )}
    </button>
  );
}
