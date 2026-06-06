import type { GateStatus } from "@/lib/api";

interface Props {
  label: string;
  theorem: string;
  status: GateStatus | undefined;
  value?: number | string;
  threshold?: number | string;
  unit?: string;
  detail?: string;
}

const colors: Record<string, string> = {
  pass:  "border-emerald-700/50 bg-emerald-950/40",
  fail:  "border-red-700/50 bg-red-950/40",
  skip:  "border-amber-700/50 bg-amber-950/40",
  error: "border-red-700/50 bg-red-950/40",
};

const dots: Record<string, string> = {
  pass:  "bg-emerald-400",
  fail:  "bg-red-400",
  skip:  "bg-amber-400",
  error: "bg-red-400",
};

export function GateCard({ label, theorem, status, value, threshold, unit, detail }: Props) {
  const s = status ?? "skip";
  return (
    <div className={`rounded-xl border p-4 ${colors[s] ?? colors.skip}`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[11px] font-mono text-slate-500">{theorem}</p>
          <p className="mt-0.5 text-sm font-semibold text-slate-200">{label}</p>
        </div>
        <span className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${dots[s] ?? dots.skip}`} />
      </div>
      {value !== undefined && (
        <div className="mt-3 flex items-end gap-2">
          <span className="text-2xl font-mono font-bold text-white">
            {typeof value === "number" ? value.toFixed(3) : value}
          </span>
          {unit && <span className="mb-0.5 text-xs text-slate-500">{unit}</span>}
        </div>
      )}
      {threshold !== undefined && (
        <p className="mt-1 text-[11px] text-slate-500">
          threshold: {typeof threshold === "number" ? threshold.toFixed(3) : threshold}
        </p>
      )}
      {detail && (
        <p className="mt-2 text-[11px] text-slate-400 leading-relaxed">{detail}</p>
      )}
      <div className="mt-3 text-[11px] font-semibold uppercase tracking-wider" style={{
        color: s === "pass" ? "#34d399" : s === "fail" ? "#f87171" : "#fbbf24"
      }}>
        {s}
      </div>
    </div>
  );
}
