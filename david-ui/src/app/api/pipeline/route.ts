/**
 * POST /api/pipeline — trigger ingest + fit on Railway from any client.
 * GET  /api/pipeline — poll pipeline running status (read-only).
 *
 * E-8 audit: POST is a PERMITTED operational trigger. It fires the backend
 * pipeline job but does NOT mutate routes, FDP thresholds, or gate outcomes.
 * Those are determined exclusively by the backend engine during fitting.
 */
import { NextRequest, NextResponse } from "next/server";

const RAILWAY = process.env.DAVID_API_URL ?? "http://localhost:8080";
const SECRET  = process.env.PIPELINE_SECRET ?? "";

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const sinceDays: number = body.since_days ?? 7;
  const fitOnly: boolean  = body.fit_only   ?? false;

  const params = new URLSearchParams({ since_days: String(sinceDays) });
  if (fitOnly) params.set("fit_only", "true");

  try {
    const res = await fetch(
      `${RAILWAY}/internal/pipeline/run?${params}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${SECRET}`,
          "Content-Type": "application/json",
        },
        // Don't wait for pipeline to finish — just fire-and-forget
        signal: AbortSignal.timeout(10_000),
      }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.ok ? 200 : 502 });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to reach Railway", detail: String(err) },
      { status: 502 }
    );
  }
}

export async function GET() {
  try {
    const res = await fetch(`${RAILWAY}/internal/pipeline/status`, {
      headers: { Authorization: `Bearer ${SECRET}` },
      signal: AbortSignal.timeout(5_000),
    });
    return NextResponse.json(await res.json());
  } catch (err) {
    return NextResponse.json({ running: false, error: String(err) });
  }
}
