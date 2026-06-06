"use client";

import {
  ComposedChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine, ErrorBar,
} from "recharts";

export interface CiRow {
  label: string;
  p_active: number;
  ci_80_lo: number;
  ci_80_hi: number;
  ci_95_lo: number;
  ci_95_hi: number;
  route: string;
}

function routeColor(route: string) {
  return route === "conditional_forecast" ? "#38bdf8" : "#fbbf24";
}

// Custom error bar: draws the CI as a vertical line with caps
const CiBar = (props: any) => {
  const { x, y, width, value, ci_95_lo, ci_95_hi, ci_80_lo, ci_80_hi, fill } = props;
  if (!value) return null;
  const cx = x + width / 2;
  const scale = (v: number) => props.yAxis?.scale?.(v) ?? y;
  const y95lo = scale(ci_95_lo);
  const y95hi = scale(ci_95_hi);
  const y80lo = scale(ci_80_lo);
  const y80hi = scale(ci_80_hi);
  return (
    <g>
      {/* 95% CI thin line */}
      <line x1={cx} y1={y95lo} x2={cx} y2={y95hi} stroke={fill} strokeWidth={2} strokeOpacity={0.4} />
      {/* 80% CI thick line */}
      <line x1={cx} y1={y80lo} x2={cx} y2={y80hi} stroke={fill} strokeWidth={6} strokeOpacity={0.7} />
      {/* median dot */}
      <circle cx={cx} cy={y} r={4} fill={fill} />
      {/* caps */}
      <line x1={cx - 5} y1={y95lo} x2={cx + 5} y2={y95lo} stroke={fill} strokeWidth={1.5} strokeOpacity={0.5} />
      <line x1={cx - 5} y1={y95hi} x2={cx + 5} y2={y95hi} stroke={fill} strokeWidth={1.5} strokeOpacity={0.5} />
    </g>
  );
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as CiRow;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-3 text-xs shadow-xl">
      <p className="font-semibold text-white mb-1">{d.label}</p>
      <p className="text-sky-300">p_active: <b>{d.p_active.toFixed(3)}</b></p>
      <p className="text-slate-400">80% CI: [{d.ci_80_lo.toFixed(3)}, {d.ci_80_hi.toFixed(3)}]</p>
      <p className="text-slate-400">95% CI: [{d.ci_95_lo.toFixed(3)}, {d.ci_95_hi.toFixed(3)}]</p>
      <p className={`mt-1 font-mono ${d.route === "conditional_forecast" ? "text-sky-400" : "text-amber-400"}`}>
        {d.route}
      </p>
    </div>
  );
};

export function CiChart({ data }: { data: CiRow[] }) {
  if (!data.length) {
    return (
      <div className="flex h-48 items-center justify-center text-slate-500 text-sm">
        No forecast cells available
      </div>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 44)}>
      <ComposedChart
        data={data}
        layout="vertical"
        margin={{ top: 8, right: 24, bottom: 8, left: 8 }}
      >
        <XAxis
          type="number" domain={[0, 1]}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: "#64748b", fontSize: 11 }}
          axisLine={{ stroke: "#1e293b" }} tickLine={false}
        />
        <YAxis
          type="category" dataKey="label" width={120}
          tick={{ fill: "#94a3b8", fontSize: 11 }}
          axisLine={false} tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine x={0.5} stroke="#334155" strokeDasharray="3 3" />
        <Bar dataKey="p_active" barSize={8} radius={4}>
          {data.map((row, i) => (
            <Cell key={i} fill={routeColor(row.route)} />
          ))}
        </Bar>
      </ComposedChart>
    </ResponsiveContainer>
  );
}
