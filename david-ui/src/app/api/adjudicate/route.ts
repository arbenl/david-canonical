/** POST /api/adjudicate — submit human tactic labels for one evidence item. */
import { NextRequest, NextResponse } from "next/server";

const RAILWAY = process.env.DAVID_API_URL ?? "http://localhost:8080";
const SECRET  = process.env.PIPELINE_SECRET ?? "";

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => ({}));
  const { evidence_id, labels } = body as { evidence_id?: string; labels?: Record<string, number> };

  if (!evidence_id || !labels) {
    return NextResponse.json(
      { error: "evidence_id and labels are required" },
      { status: 400 }
    );
  }

  try {
    const res = await fetch(
      `${RAILWAY}/evidence/${evidence_id}/adjudicate`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${SECRET}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ labels }),
        signal: AbortSignal.timeout(10_000),
      }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.ok ? 200 : res.status });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to reach Railway", detail: String(err) },
      { status: 502 }
    );
  }
}
