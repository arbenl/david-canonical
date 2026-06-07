/** GET /api/cron — called by Vercel Cron daily at 06:00 UTC. */
import { NextRequest, NextResponse } from "next/server";

const RAILWAY = process.env.DAVID_API_URL ?? "http://localhost:8080";
const SECRET  = process.env.PIPELINE_SECRET ?? "";

export async function GET(req: NextRequest) {
  // Vercel passes CRON_SECRET in the Authorization header automatically.
  const cronSecret = process.env.CRON_SECRET;
  if (cronSecret) {
    const auth = req.headers.get("authorization") ?? "";
    if (auth !== `Bearer ${cronSecret}`) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  try {
    const res = await fetch(
      `${RAILWAY}/internal/pipeline/run?since_days=2`,
      {
        method: "POST",
        headers: { Authorization: `Bearer ${SECRET}` },
        signal: AbortSignal.timeout(10_000),
      }
    );
    const data = await res.json();
    return NextResponse.json({ cron: "triggered", railway: data });
  } catch (err) {
    return NextResponse.json({ cron: "error", detail: String(err) }, { status: 500 });
  }
}
