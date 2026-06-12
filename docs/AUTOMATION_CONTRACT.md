# DAVID/M0.1 Automation Contract

What is automated and what requires a human.

## Goals

1. Daily refresh of evidence with zero human action.
2. Weekly fit + SBC + falsification with zero human action.
3. Adjudicator interaction capped at ~4 hours per 90-day cycle.
4. No silent failure: every fail-closed exit produces a typed reason and a log.

## Automated (no human required)

| Stage | Trigger | Owner | Failure mode |
|-------|---------|-------|--------------|
| Source ingestion | cron nightly | `david ingest` | Failed scraper is logged; pipeline continues with remaining sources. |
| Normalization | post-ingest | `ingest/normalize.py` | Items with malformed schema are dropped to `data/raw/quarantine/`. |
| LLM coding | post-normalize | `ingest/llm_coder.py` | Provider error logs to `data/coded/errors.jsonl`; item retried next cycle. |
| Coder calibration | nightly | `coder_calibration.stan` | Gate-fail emits typed reason; no main fit runs. |
| Adjudicator queue rebuild | post-coding | `ingest/adjudicator_queue.py` | Queue capped by `ADJUDICATOR_HOURS_PER_CYCLE`. |
| Fit | weekly | `model/fit.py` | R-hat / ESS / divergence fails block all downstream. |
| Measurement SBC | weekly | `simulator/sbc.py` | KS uniformity fail blocks forecast emission. |
| Forecast SBC | weekly | `simulator/forecast_sbc.py` | Coverage outside bands fails F14. |
| Falsification F1..F15 | weekly | `validation/falsification.py` | Any failed F-test routes affected cells to withhold. |
| Forecast emission | weekly | `engine/forecast.py` | Cells beyond h* are reported as horizon_prior_dominated. |
| Routing | weekly | `engine/router.py` | All cells get a route; none are silently dropped. |

## Human-required (capped)

| Action | Cap | Frequency | Owner | Failure mode |
|--------|-----|-----------|-------|--------------|
| Adjudicator queue review | 4 hours | per cycle | adjudicator | Items not reviewed within cycle remain llm_only_acceptable. |
| Source-independence ledger review | 1 hour | quarterly | source steward | Stale ledger flagged after 120 days. |
| Route ledger review | 1 hour | weekly | reviewer | Unreviewed runs do not promote to API headline. |
| Theorem packet review | as needed | per packet update | math reviewer | Stan models locked to last reviewed packet hash. |

## What is NEVER automated

- Promoting a forecast to "headline" without route ledger sign-off.
- Modifying source rho / delta priors without quarterly review.
- Changing the falsification thresholds (F1..F15) at runtime.
- Skipping a fail-closed gate.
- Authoring primary mathematical, theoretical, or thesis-chapter documentation as plain markdown (all formal documents must be written in LaTeX with TikZ/pgfplots vector diagrams).

## How to know automation is working

Three observable signals:

1. `data/logs/nightly_ingest_*.log` exists for every weekday morning.
2. `data/forecasts/{latest}/route_ledger.json` has `timestamp` within the
   last 7 days and `route_counts` sums to a non-zero total.
3. `data/forecasts/{latest}/falsification_ledger.json` `battery_result.gate_status`
   is recorded (pass OR fail; what matters is that it is recorded).

If any of these is missing, the automation is silently broken; investigate
before consuming forecast outputs.
