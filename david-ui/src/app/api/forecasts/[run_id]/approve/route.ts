/**
 * E-8: POST /api/forecasts/[run_id]/approve — route ledger sign-off.
 *
 * This is the ONLY permitted mutation from the UI, per ARCHITECTURE §7 and the
 * Automation Contract. It signs the route_ledger.json for M01 human review.
 *
 * No other mutation endpoint exists or should be created from the UI layer.
 * Routes, FDP thresholds, and gate outcomes are immutable per-run artifacts —
 * they are set by the backend engine and never touched by the UI.
 */
import { NextRequest, NextResponse } from "next/server";

const RAILWAY = process.env.DAVID_API_URL ?? "http://localhost:8080";
const SECRET  = process.env.PIPELINE_SECRET ?? "";

export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ run_id: string }> }
) {
  const { run_id } = await params;

  if (!run_id) {
    return NextResponse.json({ error: "run_id required" }, { status: 400 });
  }

  try {
    const res = await fetch(`${RAILWAY}/forecasts/${run_id}/approve`, {
      method: "POST",
      headers: {
        Authorization: SECRET,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(10_000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.ok ? 200 : res.status });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to reach backend", detail: String(err) },
      { status: 502 }
    );
  }
}

/** Block all non-POST methods explicitly. */
export async function GET() {
  return NextResponse.json({ error: "Method not allowed" }, { status: 405 });
}
