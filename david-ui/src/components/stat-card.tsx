interface Props {
  label: string;
  value: string | number;
  sub?: string;
  accent?: boolean;
  warn?: boolean;
}

export function StatCard({ label, value, sub, accent, warn }: Props) {
  return (
    <div className={`rounded-xl border p-4 ${
      warn   ? "border-amber-700/40 bg-amber-950/20" :
      accent ? "border-sky-700/40  bg-sky-950/20"   :
               "border-slate-800   bg-slate-900/60"
    }`}>
      <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
      <p className={`mt-2 text-3xl font-mono font-bold ${
        warn ? "text-amber-300" : accent ? "text-sky-300" : "text-white"
      }`}>
        {value}
      </p>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}
