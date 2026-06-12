export function PremiumForecastChart() {
  return (
    <div className="bg-[#070b19]/90 border border-slate-800 rounded-xl p-5 flex flex-col min-h-[240px] relative w-full group">
      {/* Tooltip Dot */}
      <div className="absolute top-3 right-3 tooltip-dot z-20">
        <span className="w-5 h-5 rounded-full bg-cyan-500 text-xs text-white font-bold flex items-center justify-center cursor-help explain-ring pulse-dot font-mono">?</span>
        <div className="absolute right-0 top-7 w-64 p-3 bg-slate-900 border border-slate-700 rounded-lg shadow-xl text-[11px] leading-relaxed text-slate-200 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity z-50">
          <p className="font-bold text-cyan-400 mb-1">FORECAST PROBABILITIES & CI</p>
          Këtu shfaqet grafiku i intervaleve kredibile (Credible Intervals 80% dhe 95%) të parashikimit. Linja e mesme përfaqëson mesataren e parashikimit (Mean Forecast) ndërsa zona e ndriçuar përfaqëson pasigurinë statistikore të parashikimit sipas modelit MCMC.
        </div>
      </div>

      <div className="flex items-center justify-between mb-2">
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Forecast Probabilities</p>
          <p className="text-[10px] text-slate-500 uppercase mt-0.5">Credible Intervals: <span className="text-cyan-400">80% CI</span> · <span className="text-indigo-400">95% CI</span></p>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-slate-400 font-mono">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-cyan-500"></span>Actual</span>
          <span className="flex items-center gap-1"><span className="w-2 h-0.5 bg-red-400"></span>Predicted</span>
        </div>
      </div>

      {/* Main Glowing Chart (SVG Representation) */}
      <div className="flex-1 relative mt-2 min-h-[140px] flex items-end">
        {/* Grid Lines Background */}
        <div className="absolute inset-0 flex flex-col justify-between pointer-events-none opacity-20">
          <div className="border-b border-slate-700 w-full h-0"></div>
          <div className="border-b border-slate-700 w-full h-0"></div>
          <div className="border-b border-slate-700 w-full h-0"></div>
          <div className="border-b border-slate-700 w-full h-0"></div>
        </div>

        {/* Glowing SVG Chart Path */}
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 500 130" preserveAspectRatio="none">
          <defs>
            {/* Credible Interval Gradient */}
            <linearGradient id="chart-glow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.25"/>
              <stop offset="100%" stopColor="#4f46e5" stopOpacity="0.0"/>
            </linearGradient>
            {/* Glowing Filter */}
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>
          {/* Credible Interval Area (95% CI & 80% CI combined) */}
          <path d="M 0 90 Q 100 110 200 70 T 400 40 T 500 45 L 500 130 L 0 130 Z" fill="url(#chart-glow)"/>
          
          {/* Credible Interval Borders */}
          <path d="M 0 80 Q 100 95 200 55 T 400 30 T 500 35" fill="none" stroke="#22d3ee" strokeWidth="1.5" strokeDasharray="2,2"/>
          <path d="M 0 100 Q 100 120 200 85 T 400 50 T 500 55" fill="none" stroke="#4f46e5" strokeWidth="1.5" strokeDasharray="2,2"/>
          
          {/* Actual Line (Blue) */}
          <path d="M 0 95 Q 100 115 200 75 T 400 45 T 500 50" fill="none" stroke="#0ea5e9" strokeWidth="2.5" filter="url(#glow)"/>
          
          {/* Predicted Line (Red/Orange) */}
          <path d="M 300 62 Q 350 48 400 45 T 500 48" fill="none" stroke="#ef4444" strokeWidth="2" strokeDasharray="3,3"/>

          {/* Dotted Time Indicator Line */}
          <line x1="280" y1="0" x2="280" y2="130" stroke="#475569" strokeWidth="1" strokeDasharray="3,3" />
        </svg>

        {/* Chart labels */}
        <div className="absolute bottom-1 left-2 text-[8px] text-slate-500 font-mono">0</div>
        <div className="absolute bottom-1 right-2 text-[8px] text-slate-500 font-mono">100%</div>
        
        {/* Tooltip Overlay */}
        <div className="absolute top-2 left-1/2 -translate-x-1/2 bg-[#0b1329] border border-cyan-500/30 rounded px-2.5 py-1 text-[9px] font-mono text-slate-300 shadow-lg pointer-events-none">
          Mean Forecast: <span className="text-emerald-400 font-bold">+3.8%</span> <span className="text-slate-500">[+1.2%, +6.4%]</span> 80% CI
        </div>
      </div>
    </div>
  );
}
