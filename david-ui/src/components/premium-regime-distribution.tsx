export function PremiumRegimeDistribution() {
  return (
    <div className="bg-[#070b19]/90 border border-slate-800 rounded-xl p-5 relative group mt-4">
      {/* Tooltip Dot */}
      <div className="absolute top-3 right-3 tooltip-dot z-20">
        <span className="w-5 h-5 rounded-full bg-cyan-500 text-xs text-white font-bold flex items-center justify-center cursor-help explain-ring pulse-dot font-mono">?</span>
        <div className="absolute right-0 top-7 w-64 p-3 bg-slate-900 border border-slate-700 rounded-lg shadow-xl text-[11px] leading-relaxed text-slate-200 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity z-50">
          <p className="font-bold text-cyan-400 mb-1">REGIME DISTRIBUTION</p>
          Tregon shpërndarjen e regjimit të ndërhyrjes sipas kohës. Klasat latente të DAVID ndahen në 4 regjime kryesore: Bullish (aktivitet i lartë), Bearish (aktivitet i ulët), Volatile (ndryshime të shpejta), dhe Stable (ndërhyrje e qëndrueshme).
        </div>
      </div>

      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-4">Regime Distribution</p>
      
      {/* Legend labels */}
      <div className="grid grid-cols-4 gap-4 mb-4">
        <div className="border-l-2 border-emerald-500 pl-2">
          <p className="text-[9px] text-slate-500 uppercase">BULLISH</p>
          <p className="numeric text-sm font-bold text-emerald-400">45%</p>
        </div>
        <div className="border-l-2 border-orange-500 pl-2">
          <p className="text-[9px] text-slate-500 uppercase">BEARISH</p>
          <p className="numeric text-sm font-bold text-orange-400">15%</p>
        </div>
        <div className="border-l-2 border-pink-500 pl-2">
          <p className="text-[9px] text-slate-500 uppercase">VOLATILE</p>
          <p className="numeric text-sm font-bold text-pink-400">25%</p>
        </div>
        <div className="border-l-2 border-cyan-500 pl-2">
          <p className="text-[9px] text-slate-500 uppercase">STABLE</p>
          <p className="numeric text-sm font-bold text-cyan-400">15%</p>
        </div>
      </div>

      {/* Stacked area visual */}
      <div className="h-28 flex items-end gap-1.5 w-full bg-[#05070e] p-2 rounded border border-slate-800/40 relative">
        <svg className="absolute inset-0 w-full h-full px-2" viewBox="0 0 600 100" preserveAspectRatio="none">
          {/* Bullish Bars (Green) */}
          <path d="M 10 100 L 10 60 L 50 40 L 90 70 L 130 50 L 170 30 L 210 80 L 250 90 L 250 100 Z" fill="#059669" fillOpacity="0.3" stroke="#10b981" strokeWidth="1.5" />
          {/* Bearish Bars (Orange) */}
          <path d="M 250 100 L 250 90 L 290 70 L 330 65 L 370 75 L 390 100 Z" fill="#d97706" fillOpacity="0.3" stroke="#f59e0b" strokeWidth="1.5" />
          {/* Volatile Bars (Pink) */}
          <path d="M 390 100 L 390 75 L 430 40 L 470 30 L 510 50 L 530 100 Z" fill="#db2777" fillOpacity="0.3" stroke="#ec4899" strokeWidth="1.5" />
          {/* Stable Bars (Blue) */}
          <path d="M 530 100 L 530 50 L 560 30 L 590 40 L 600 100 Z" fill="#2563eb" fillOpacity="0.3" stroke="#3b82f6" strokeWidth="1.5" />
        </svg>
      </div>
      
      {/* Timeline X-Axis */}
      <div className="flex justify-between mt-2 text-[9px] text-slate-500 font-mono">
        <span>09:00</span>
        <span>11:00</span>
        <span>19:00</span>
        <span>12:00</span>
        <span>15:00</span>
        <span>19:00</span>
        <span>21:00</span>
        <span>13:00</span>
      </div>
    </div>
  );
}
