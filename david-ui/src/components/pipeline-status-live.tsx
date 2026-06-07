"use client";

import { useEffect, useState } from "react";

export interface PipelineStatus {
  running: boolean;
  step?: string;
  started_at?: string | null;
  polled_at?: string;
  last_fit?: { started_at: string; gate_status: string } | null;
  n_evidence?: number;
  n_adjudicated?: number;
  error?: string;
}

const STEP_LABELS: Record<string, string> = {
  ingesting: "Scraping RSS sources…",
  fitting:   "Running Bayesian fit…",
  done:      "Pipeline complete",
};

/**
 * Client component — shows a live "Pipeline is running" banner while running.
 * Polls GET /api/pipeline every 5 s when `running === true`.
 * Reloads the page once the pipeline stops so server data refreshes.
 */
export function PipelineStatusLive({ initial }: { initial: PipelineStatus }) {
  const [status, setStatus] = useState<PipelineStatus>(initial);
  const [pollCount, setPollCount] = useState(0);

  useEffect(() => {
    if (!status.running) return;
    const id = setInterval(async () => {
      try {
        const res = await fetch("/api/pipeline");
        if (res.ok) {
          const data: PipelineStatus = await res.json();
          setStatus(data);
          setPollCount((n) => n + 1);
          if (!data.running) {
            // Give the DB a second to settle, then reload server components
            setTimeout(() => window.location.reload(), 1200);
          }
        }
      } catch {
        // swallow transient errors
      }
    }, 5000);
    return () => clearInterval(id);
  }, [status.running]);

  if (!status.running) return null;

  const stepLabel = STEP_LABELS[status.step ?? ""] ?? (status.step ?? "Running…");

  return (
    <div className="rounded-xl border border-blue-700/40 bg-blue-950/20 px-4 py-3
                    flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        {/* pulsing dot */}
        <span className="relative flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
          <span className="relative inline-flex h-3 w-3 rounded-full bg-blue-500" />
        </span>
        <div>
          <p className="text-sm font-semibold text-blue-300">Pipeline is running</p>
          <p className="text-xs text-blue-500/80">{stepLabel}</p>
        </div>
      </div>
      <p className="text-xs text-blue-600/60 tabular-nums">
        {pollCount > 0 ? `polled ${pollCount}×` : "polling every 5 s"}
      </p>
    </div>
  );
}
