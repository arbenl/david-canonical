"use client";

import { useState } from "react";

type Status = "idle" | "running" | "done" | "error";

/** Triggers only the Bayesian fit step (skips ingest). */
export function FitButton() {
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");

  async function run() {
    setStatus("running");
    setMessage("");
    try {
      const res = await fetch("/api/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fit_only: true }),
      });
      const data = await res.json();
      if (res.ok) {
        setStatus("done");
        setMessage(data.message ?? "Fit started.");
      } else {
        setStatus("error");
        setMessage(data.error ?? "Error triggering fit.");
      }
    } catch (err) {
      setStatus("error");
      setMessage(String(err));
    }
    setTimeout(() => {
      setStatus("idle");
      setMessage("");
    }, 6000);
  }

  const labels: Record<Status, string> = {
    idle:    "⚙ Run Fit",
    running: "⏳ Running…",
    done:    "✓ Queued",
    error:   "✗ Error",
  };

  const colors: Record<Status, string> = {
    idle:    "bg-slate-700 hover:bg-slate-600 text-slate-200",
    running: "bg-slate-600 text-slate-300 cursor-not-allowed",
    done:    "bg-emerald-700 text-emerald-100",
    error:   "bg-red-700 text-red-100",
  };

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={run}
        disabled={status === "running"}
        title="Re-run Bayesian fit only — no new scraping"
        className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${colors[status]}`}
      >
        {labels[status]}
      </button>
      {message && (
        <span className={`text-xs ${status === "error" ? "text-red-400" : "text-slate-400"}`}>
          {message}
        </span>
      )}
    </div>
  );
}
