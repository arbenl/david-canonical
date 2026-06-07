import Link from "next/link";

/** Minimum adjudicated items per stratum for Theorem A' to hold. */
const MIN_ADJ_FOR_FIT = 3;

interface Props {
  totalEvidence: number;
  totalAdjudicated: number;
  nStrata: number;
  strataReadyForFit: number;   // strata with ≥ MIN_ADJ_FOR_FIT adjudicated items
  hasActiveFit: boolean;
  fitPassed: boolean;
  queueCount: number;
  pipelineRunning: boolean;
}

interface Step {
  emoji: string;
  title: string;
  desc: string;
  action?: { label: string; href: string };
  pills: { label: string; hint: string }[];
  border: string;
  bg: string;
  titleColor: string;
}

function computeStep(p: Props): Step {
  // 1. Pipeline actively running
  if (p.pipelineRunning) {
    return {
      emoji: "⏳",
      title: "Pipeline is running…",
      desc: "Scraping sources, LLM-coding tobacco items, and fitting the Bayesian model. This takes 2–4 minutes. The page auto-refreshes when done.",
      pills: [{ label: "Ingest → Code → Fit", hint: "Three sequential steps run automatically. No manual action needed right now." }],
      border: "border-blue-700/40", bg: "bg-blue-950/20", titleColor: "text-blue-200",
    };
  }

  // 2. No evidence at all → first-run state
  if (p.totalEvidence === 0) {
    return {
      emoji: "🚀",
      title: "Start here — click ▶ Run Pipeline",
      desc: "No evidence scraped yet. The pipeline will fetch RSS feeds, filter tobacco-relevant articles, and code them with two independent LLM coders.",
      action: { label: "What happens? →", href: "/evidence" },
      pills: [
        { label: "RSS → LLM → Bayesian", hint: "Pipeline: scrape 8 RSS sources → filter tobacco keywords → code each article with 2 LLM coders → Dawid-Skene Bayesian fit." },
        { label: "Auto-runs daily at 06:00 UTC", hint: "Vercel Cron triggers the pipeline every morning. You can also click Run Pipeline any time." },
      ],
      border: "border-indigo-700/40", bg: "bg-indigo-950/20", titleColor: "text-indigo-200",
    };
  }

  // 3. Evidence exists but nothing adjudicated → coding pending or needed
  if (p.totalAdjudicated === 0) {
    return {
      emoji: "🤖",
      title: "Waiting for LLM coding to complete",
      desc: `${p.totalEvidence} articles scraped. LLM coders assign SIO/MIO/CSIO tactic labels. If coding stalled, click ▶ Run Pipeline again.`,
      pills: [
        { label: "SIO / MIO / CSIO", hint: "Tactic classes: State Institution Obstruction, Media/Information Obstruction, Civil-Society Institution Obstruction." },
        { label: "2 coders per item", hint: "Two independent LLM coders code each item so the Dawid-Skene model can estimate their accuracy." },
      ],
      border: "border-amber-700/40", bg: "bg-amber-950/20", titleColor: "text-amber-200",
    };
  }

  // 4. Human queue needs review
  if (p.queueCount > 0) {
    return {
      emoji: "👁️",
      title: `Review ${p.queueCount} item${p.queueCount === 1 ? "" : "s"} in the human queue`,
      desc: "The LLM coders disagreed on these items. Human adjudication improves Bayesian fit accuracy and is required for Theorem A'.",
      action: { label: "Open review queue →", href: "/evidence" },
      pills: [
        { label: "Coder disagreement > 30%", hint: "When coder agreement on a tactic label falls below 70 %, the item enters the human-review queue." },
        { label: "Affects Theorem A'", hint: "Theorem A' requires structural independence between coders. Human labels count as a separate coder signal." },
      ],
      border: "border-amber-700/40", bg: "bg-amber-950/20", titleColor: "text-amber-200",
    };
  }

  // 5. Has adjudicated items but no fit yet → needs fit
  if (!p.hasActiveFit) {
    return {
      emoji: "📊",
      title: "Run Bayesian fit — ⚙ Run Fit button below",
      desc: `${p.totalAdjudicated} adjudicated items ready across ${p.nStrata} strata. ${p.strataReadyForFit} strata have ≥${MIN_ADJ_FOR_FIT} adjudicated items. Click ⚙ Run Fit to estimate interference probabilities.`,
      pills: [
        { label: "Dawid-Skene model", hint: "Bayesian model estimating the true interference probability from multiple noisy LLM coder signals." },
        { label: `Need ≥${MIN_ADJ_FOR_FIT} per stratum`, hint: `Each country×policy stratum needs at least ${MIN_ADJ_FOR_FIT} adjudicated items for the Bayesian posterior to be reliable.` },
      ],
      border: "border-violet-700/40", bg: "bg-violet-950/20", titleColor: "text-violet-200",
    };
  }

  // 6. Fit ran but theorem gates failed
  if (!p.fitPassed) {
    return {
      emoji: "⚠️",
      title: "Fit ran but theorem gates didn't pass yet",
      desc: `${p.strataReadyForFit} of ${p.nStrata} strata have enough data. To pass Theorem A', add more independent RSS sources covering Kosovo, Ireland, or Netherlands and re-run the pipeline.`,
      action: { label: "View theorem gates →", href: "/engine" },
      pills: [
        { label: "Theorem A'", hint: "Requires structurally independent evidence sources per stratum. Check /engine for which gate failed." },
        { label: "R̂ < 1.01, ESS > 400", hint: "MCMC convergence thresholds. If these fail first, you need more adjudicated items before gates can pass." },
      ],
      border: "border-amber-700/40", bg: "bg-amber-950/20", titleColor: "text-amber-200",
    };
  }

  // 7. Everything green
  return {
    emoji: "✅",
    title: "System is live — all gates passed",
    desc: "Forecasts are publishing. The pipeline auto-runs daily. Review forecasts or run the validation battery.",
    action: { label: "View forecasts →", href: "/predict" },
    pills: [
      { label: "Auto-daily at 06:00 UTC", hint: "Vercel Cron triggers ingest + fit every morning automatically." },
      { label: "SBC validation available", hint: "Simulation-based calibration checks that the model's uncertainty is correctly calibrated." },
    ],
    border: "border-emerald-700/40", bg: "bg-emerald-950/20", titleColor: "text-emerald-200",
  };
}

/** Server component — reads pre-fetched state and renders a contextual action card. */
export function NextStepCard(props: Props) {
  const step = computeStep(props);

  return (
    <div className={`rounded-xl border ${step.border} ${step.bg} px-5 py-4`}>
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-start gap-3 min-w-0">
          <span className="text-2xl mt-0.5 shrink-0">{step.emoji}</span>
          <div className="min-w-0">
            <p className={`font-semibold text-sm ${step.titleColor}`}>{step.title}</p>
            <p className="mt-1 text-xs text-slate-400 leading-relaxed">{step.desc}</p>
            {/* Help pills */}
            {step.pills.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {step.pills.map((pill) => (
                  <PillWithTooltip key={pill.label} label={pill.label} hint={pill.hint} />
                ))}
              </div>
            )}
          </div>
        </div>
        {step.action && (
          <Link
            href={step.action.href}
            className="shrink-0 rounded-lg bg-white/5 px-3 py-1.5 text-xs
                       font-semibold text-slate-200 hover:bg-white/10 transition-colors"
          >
            {step.action.label}
          </Link>
        )}
      </div>
    </div>
  );
}

/** Inline pill with tooltip — rendered server-side (hover via CSS only). */
function PillWithTooltip({ label, hint }: { label: string; hint: string }) {
  return (
    <span className="group relative inline-flex">
      <span className="inline-flex items-center gap-1 rounded-full bg-slate-800/80
                       border border-slate-700 px-2 py-0.5 text-[11px] text-slate-400
                       cursor-default select-none">
        {label}
        <span className="text-[10px] text-slate-600">?</span>
      </span>
      {/* CSS-only tooltip (no JS needed here since parent is server-rendered) */}
      <span className="pointer-events-none absolute bottom-7 left-0 z-50 hidden
                       w-56 rounded-lg border border-slate-600 bg-slate-800
                       px-3 py-2 text-xs leading-relaxed text-slate-200 shadow-xl
                       group-hover:block">
        {hint}
        <span className="absolute -bottom-1.5 left-4 h-0 w-0
                         border-x-4 border-x-transparent border-t-[6px] border-t-slate-800" />
      </span>
    </span>
  );
}
