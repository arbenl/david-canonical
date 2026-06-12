# DAVID/M0.1 Canonical Architecture — Forward Prediction Engine

Status: BLUEPRINT v1.0
Date: 2026-06-06
Supersedes for forecasting layer: nothing (additive to council_m01)
Does NOT supersede: council_m01 measurement layer, theorem packets, gate runners

---

## 0. Purpose

This document defines the canonical architecture for the operational DAVID/M0.1
forward-prediction engine. It extends the measurement-first model proved in
council_m01 with two new layers:

1. A 6 to 12 month forward-prediction layer with its own identifiability proof,
   simulator-based calibration, and fail-closed routing.
2. An automated ingestion and coding layer that reduces human-in-the-loop
   bottlenecks for upload, source assignment, multi-LLM coding, and adjudicator
   escalation, while preserving the Dawid-Skene coder-reliability discipline.

The mathematical AI engine remains at the heart: every output is gated by the
M0.1 theorem stack (A', B', C, D-forecast) and the F1 to F15 falsification
battery. Nothing prints to a reviewer unless the gates pass.

---

## 1. What this architecture preserves and what it adds

### Preserved from council_m01

- Unit of analysis g = (c, t, p, k): country, time, policy area, tactic class
- Latent activity A_g with measurement chain A -> B -> Y
- Source-channel latent class (Theorem A')
- Coder-reliability Dawid-Skene layer
- HSMM with pi_ii = 0 and right-censored final segment
- Opportunity-frame ascertainment gate
- Observability-tiered detection rho(O), delta(O)
- Causal firewall and claim firewall
- Fail-closed routing to {aggregate_only, monitor_only, evidence_gap, withhold}
- Stan models in council_m01/simulation/*.stan
- Theorem packets in council_m01/theorem_packets/
- Pre-registration v2 grid

### New in canonical layer

- Forward-time joint forecast block on top of the HSMM posterior
- Theorem D-forecast: forecast horizon bound under prior domination
- Forecast-SBC simulator (proves the forecast block the way SBC proves the fit)
- Adversarial battery extension F13, F14, F15 for forward predictions
- Automated multi-source ingestion with structural-independence ledger
- Multi-LLM coder pool with Dawid-Skene reliability calibration against gold
- Adjudicator queue with disagreement-triggered escalation only
- Single `david` CLI as the orchestration entry point
- Forecast routing layer with new gate FG1 to FG6
- API layer for read-only forecast consumption (no UI/Stan coupling)

---

## 2. End-to-end pipeline

```
[external sources]
    | (cron / webhook)
    v
[ingest.scrapers] --> raw/                              (JSON-L per source)
    |
    v
[ingest.normalize] --> raw_normalized/                  (canonical evidence schema)
    |
    v
[ingest.llm_coder] --> coded/                           (per-coder labels Y_{e,r,k})
    |
    v
[ingest.adjudicator_queue] -- IF disagree --> [human Gold_B] --> adjudicated/
    |
    v
[ingest.source_independence_ledger] --> source_indep.json
    |
    v
[model.fit] (cmdstanpy NUTS, m01_forward.stan) --> fits/{run_id}/
    |
    v
[engine.identification_distance] --> d_theta_per_stratum.json
[engine.observability_sensitivity] --> lambda_bounds_per_cell.json
[validation.falsification F1..F15] --> falsification_ledger.json
    |
    v   IF any gate FAILS -> route = withhold or evidence_gap; STOP
    |
    v   ELSE
    |
[engine.forecast] (6, 9, 12 month horizons) --> forecasts/{run_id}/
    |
    v
[validation.forecast_calibration] --> forecast_calibration.json
[routing.forecast_router] --> route_ledger.json
    |
    v
[api / dashboard] (read-only)                          (operator review)
```

Every arrow above is a contract: the downstream module rejects inputs that do
not match the schema and emits a typed evidence_gap with binding reason.

---

## 3. The mathematical AI engine at the heart

The engine has four mathematical responsibilities, each backed by an executable
diagnostic:

### 3.1 Identification distance (Theorem A' practical check)

For each stratum g, compute per MCMC draw

```
d(theta) = min( J_(3), min(phi_g, 1 - phi_g) )
```

where J_(3) is the third-largest per-source Youden index J_s = |rho_s - delta_s|
(sorted descending). J_(3) matches the identifiability theorem: Kruskal's
condition requires three independent informative views, not all S, so using
min over all s would disqualify an identified stratum merely because one weak
nuisance source is included. When S < 3, the conservative fallback min_s J_s is
used (the stratum fails FG2 regardless, since the Kruskal rank condition is
unmet).

Strata with posterior median d below floor (default 0.05,
`ID_DISTANCE_FLOOR`) are flagged practically_non_identified and excluded from
headline forecasts.

AUDIT CORRECTION (June 2026 math audit): an earlier draft of this section
included a `|rho_s - (1 - delta_s)|` term. That term is WITHDRAWN. The
anti-diagonal rho = 1 - delta is NOT a non-identification boundary — the
symmetric channel (rho, delta) = (0.9, 0.1) lies on it with J = 0.8 and is
among the most informative channels. The only singular boundary is the
diagonal rho = delta. See `thesis_mathematical_core.tex`, Theorem A'
operational diagnostic.

Implementation: `david/engine/identification_distance.py` (must implement the
corrected formula above; note that `tests/test_identification_distance.py`
historically asserted against the withdrawn label-flip term and requires
realignment — tracked as an implementation-conformance item)

### 3.2 Channel informativeness (Theorem B'.2)

For each cell, compute observability informativeness

```
I(O) = |rho(O) - delta(O)|
```

with posterior credible interval. Gate FG3 imposes TWO co-registered
conditions, both of which must hold (constants in `david/config.py`):

1. Informativeness floor: the LOWER endpoint of the central 95% credible
   interval of I(O) — i.e. the 2.5% posterior quantile — must satisfy
   `I_lower_95 >= INFORMATIVENESS_FLOOR_LOWER_95 = 0.10`. (An earlier draft of
   this section said "upper-CI", which contradicted both §7.2 and the thesis
   core; the lower endpoint is the operative quantity — the gate must hold
   even under the pessimistic end of posterior uncertainty.)
2. Information floor: `N_eff * I^2 >= N_EFF_I2_FLOOR = 3.0`, where I is the
   posterior median informativeness and N_eff is the replicate count,
   dependence-adjusted via the score-autocovariance (Godambe) correction
   N_eff = N * gamma_0 / (gamma_0 + 2 * sum_{h>=1} gamma_h), capped at
   min(N, N_eff). Rationale: below this floor the Cramér–Rao variance scale
   exceeds 1/12 (the variance of a Uniform(0,1) prior) at the worst operating
   point — the cell cannot beat knowing nothing.

Cells failing either condition are flagged prior_dominated and reported with
prior interval, not point forecast.

OPEN LIMITATION (carried from June 2026 math audit, item #6): whether the
replicate count supplied to Gate FG3 in the PRODUCTION pipeline is actually
dependence-adjusted (N_eff per the formula above, computed from the fitted
dwell/transition posteriors) — rather than the raw N — could not be verified
from the audited files. Until verified, FG3 results on serially dependent
units must be treated as potentially anti-conservative. Verification is a
blocking item for headline routing on time-indexed strata.

Implementation: `david/theorems/B_prime.py`

### 3.3 Posterior expected FDP control (Theorem C renamed)

Forecast flag threshold t*(q) chosen by

```
t*(q) = inf { t : sum_{flagged} (1 - p_hat) / max(1, |flagged|) <= q }
```

Implementation: `david/engine/router.py`

### 3.4 Forecast horizon bound (NEW Theorem D-forecast)

For horizon h, the posterior forecast variance decomposition is

```
Var(A_{g, t+h}) = signal_h + prior_drift_h
```

where prior_drift_h is the expected drift toward the marginal regime
distribution under repeated HSMM transitions. Define horizon-validity h* as
the FIRST CROSSING of the drift threshold:

```
h* = min { h >= 1 : drift(h) >= tau } - 1
```

with tau = 0.5 by default (`HORIZON_PRIOR_DRIFT_TAU`), and h* set to the
maximal emitted horizon if no crossing occurs.

AUDIT CORRECTION (June 2026 math audit): the earlier max-form definition
`h* = max { h : drift(h) < tau }` is WITHDRAWN. drift(h) is not proven
monotone in h, so the max-form could admit horizons BEYOND a first crossing
whenever the drift dips back below tau; the first-crossing form is the
conservative reading and is the operative one. h* is an estimated diagnostic
(Monte Carlo over posterior draws), not a certified validity boundary, and
should be reported with its posterior spread (e.g. h* induced by the 5% and
95% drift envelopes).

Forecasts at h > h* are returned as the marginal
regime prediction, not the conditional one, and the route is labeled
horizon_prior_dominated.

This is the formal counterpart to the Gap-2 information-decay argument: the
forecast is only as good as the regime-distribution evidence the data provide,
and at long horizons the conditional posterior collapses to the marginal.

Implementation: `david/theorems/D_forecast_horizon.py`

---

## 4. Stan model layout

Three Stan models are canonical. They live in `canonical/stan/`.

### 4.1 `m01_forward.stan`

Extends `council_m01/simulation/full_integrated_m01.stan` with:

- Same source-channel + coder layer + HSMM with pi_ii = 0 (no change)
- Same selection model (no change)
- NEW `generated quantities` block that draws forward H-step regime and
  activity trajectories for each chain draw:

```
generated quantities {
  array[R, H] int<lower=1, upper=L> z_future;        // forecast regime path
  array[R, H, K] real<lower=0, upper=1> a_future;     // forecast activity prob
  array[R, H, K] int<lower=0, upper=1> a_future_draw; // forecast activity draw

  for (r in 1:R) {
    // Recover terminal state via forward filter
    int z_last = sample_terminal_regime_categorical(...);
    int dwell_remaining = sample_dwell_remaining(z_last, ...);

    // Step forward
    for (h in 1:H) {
      if (dwell_remaining > 1) {
        z_future[r, h] = z_last;
        dwell_remaining -= 1;
      } else {
        z_future[r, h] = categorical_logit_rng(log_jump[z_last, ]);
        dwell_remaining = shifted_poisson_rng(dwell_lambda[z_future[r, h]]);
        z_last = z_future[r, h];
      }
      for (k in 1:K) {
        a_future[r, h, k] = inv_logit(alpha_activity[z_future[r, h], k]);
        a_future_draw[r, h, k] = bernoulli_rng(a_future[r, h, k]);
      }
    }
  }
}
```

Output: posterior draws of (z_future, a_future) per (series, horizon, tactic).

### 4.2 `coder_calibration.stan`

Stand-alone Dawid-Skene with gold-standard anchoring. Inputs:
- coder_pool: M coders (R real + M_llm LLM-instances)
- gold_items: subset with Gold_B
- ungold_items: rest

Outputs:
- kappa_plus[m], kappa_minus[m] posteriors per coder
- per-item latent B_e posterior

Used by ingest layer before items flow into the main fit.

### 4.3 `synthetic_generator.stan`

Same likelihood as m01_forward.stan but in `generated quantities` only.
Generates synthetic worlds for SBC and the falsification battery. Takes a
hyper-prior draw and produces the full (B, Y, Q, n) observation tuple.

Used by `david/simulator/sbc.py` and `david/simulator/forecast_sbc.py`.

---

## 5. Simulator: the proof-by-recovery engine

A model is only trustworthy if its parameters can be recovered from data the
model itself generated. Three simulators are canonical.

### 5.1 SBC for measurement layer

`david/simulator/sbc.py`

For each of N synthetic worlds:
  1. Draw theta_true from the prior.
  2. Generate (B, Y, Q, n) from synthetic_generator.stan.
  3. Fit m01_forward.stan to the synthetic data.
  4. Compute rank statistic of theta_true within posterior draws.

Pass criterion: rank uniform within Simulation-Based Calibration tolerance
(Talts et al. 2018). Implemented as Kolmogorov-Smirnov against uniform with
pre-registered alpha.

### 5.2 SBC for forecast layer (NEW)

`david/simulator/forecast_sbc.py`

For each of N synthetic worlds:
  1. Draw theta_true from prior.
  2. Generate history (B_{1:T}, Y_{1:T}, Q_{1:T}, n_{1:T}) and continuation
     (B_{T+1:T+H}, A_{T+1:T+H}) from synthetic_generator.stan.
  3. Fit m01_forward.stan to history only.
  4. Read out forecast draws (a_future, z_future).
  5. Compute coverage of true a_future and z_future by posterior credible
     intervals.

Pass criterion: nominal 80%/95% credible intervals achieve coverage in
[75%, 85%] / [90%, 98%] over N synthetic worlds. Pre-registered.

### 5.3 Adversarial battery (extends F-battery)

`david/simulator/adversarial_battery.py`

Implements F1, F3, F4, F5, F6 (corrected per third-round), F7, F8, F9, F10,
F11, F12 plus new forecast-specific:

- F13: Forecast horizon respect. Forecasts at h > h* must equal marginal
  regime prediction. Failure: forecast block produced non-marginal output
  at horizon claimed prior-dominated.
- F14: Forecast-SBC coverage. From 5.2 above.
- F15: Forecast no-improvement under permuted history. Permute the history of
  the regime path; refit; forecast quality must drop to chance.

---

## 6. Automated ingestion: the human-loop budget

Total human time per refresh cycle is the budget. Target: under 4 hours of
adjudicator time per 90-day refresh, scaling sub-linearly with evidence volume.

### 6.1 Source layer

`david/ingest/sources.py` declares a registry of source families with
structural-independence pairwise scores. Each source has:

```
{
  "source_id": "stop_global_index_2024",
  "family": "civil_society_monitor",
  "ingest_kind": "api" | "rss" | "scrape" | "manual",
  "endpoint": "...",
  "refresh_cadence_days": 30,
  "country_coverage": [...],
  "policy_coverage": [...],
  "structural_independence_scores": {
    "<other_source_id>": 0.0..1.0,   // 1.0 = fully independent
    ...
  },
  "rho_prior": {"mean": 0.7, "scale": 0.15},
  "delta_prior": {"mean": 0.05, "scale": 0.05}
}
```

The structural-independence ledger (`config/source_independence_ledger.json`)
is reviewed quarterly. F11 (conditional independence) tests it.

### 6.2 Scrapers

Each scraper is a self-contained module in `david/ingest/scrapers/` with
contract:

```
def fetch(since: date, until: date) -> Iterator[RawEvidenceItem]:
    """Yield typed raw evidence items, normalized to canonical schema."""
```

No human interaction. Cron at `scripts/nightly_ingest.sh`.

### 6.3 LLM coder pool

`david/ingest/llm_coder.py` runs each evidence item through M_llm independent
LLM coders (different models, different prompt seeds). Each emits Y_{e, r, k}
for each of K tactic classes.

Critical: LLM coders are first calibrated on the gold set
(`data/gold/gold_b_calibration.csv`) via `coder_calibration.stan`. Their
kappa_plus / kappa_minus posteriors enter the main fit exactly like human
coders.

This is the key automation move: the engine treats LLM coders as imperfect
coders with measurable reliability, not as oracles.

### 6.4 Adjudicator queue

`david/ingest/adjudicator_queue.py` escalates an item to a human adjudicator
only if:

1. Inter-coder disagreement exceeds threshold (default: any tactic class with
   |Y_majority - Y_minority| crosses 0.3 posterior); or
2. Item is sampled into the rolling gold-standard expansion (default: 5% of
   items at random); or
3. Item is in a stratum currently below the Theorem A' replication floor
   (active sampling).

Otherwise the item enters the main pipeline with LLM-only labels and the
coder-reliability layer absorbs the LLM error.

Output: `data/adjudicator_queue.json` ranked by marginal information gain.

### 6.5 Human-loop budget enforcement

`david/ingest/human_loop_budget.py` tracks adjudicator time spent per cycle.
If projected queue exceeds budget, the adjudicator queue is truncated to the
top-information items and the rest are flagged llm_only_acceptable in the
ledger.

---

## 7. Forecast endpoints

### 7.1 Primary forecast object

For each (c, t_now, p, k, horizon h in {3, 6, 9, 12} months):

```
{
  "cell": {"c": "...", "t_now": "2026-06-06", "p": "...", "k": "..."},
  "horizon_months": 6,
  "p_active": 0.42,                            // posterior P(A_{t+h, k} = 1)
  "credible_interval_80": [0.31, 0.55],
  "credible_interval_95": [0.22, 0.66],
  "regime_distribution": {                     // p(Z_{t+h} = r | data)
    "Z_1": 0.10, "Z_2": 0.18, "Z_3": 0.05,
    "Z_4": 0.40, "Z_5": 0.22, "Z_6": 0.05
  },
  "horizon_validity": {
    "h_star_months": 7,
    "prior_drift_share": 0.31,
    "below_h_star": true                       // if false, marginal returned
  },
  "identification_distance_posterior_median": 0.18,
  "informativeness_I_O_posterior_median": 0.42,
  "informativeness_I_O_lower_95": 0.21,
  "lambda_endogenous_bounds": [0.0, 0.25],
  "forecast_route": "headline" | "monitor_only" | "aggregate_only"
                  | "evidence_gap" | "withhold" | "horizon_prior_dominated",
  "route_reasons": [...],
  "model_version": "m01_forward_v1.0",
  "fit_run_id": "...",
  "evidence_cutoff": "2026-05-31"
}
```

### 7.2 Forecast routing gates (NEW)

`david/routing/forecast_router.py` applies, in order:

- FG1: All measurement gates F1, F3, F4, F5 pass on the fit run.
- FG2: Theorem A' identification_distance posterior median >= 0.05 for the cell
  stratum.
- FG3: Theorem B' — BOTH conditions for the cell: (a) I(O) lower 95% CI
  >= 0.10, and (b) N_eff * I^2 >= 3.0 with N_eff dependence-adjusted per
  §3.2 (production adjustment status: open audit item #6).
- FG4: Forecast SBC F14 pass on the current model version.
- FG5: h <= h* (Theorem D-forecast horizon validity).
- FG6: Endogenous observability sensitivity interval width <= 0.20.

All six must pass for `forecast_route = headline`. Otherwise the route
degrades: missing FG5 -> horizon_prior_dominated; missing FG3 or FG6 ->
prior_dominated; missing FG2 -> evidence_gap; missing FG1 or FG4 -> withhold.

---

## 8. CLI and automation contracts

Single entry point: `david` (installed by pyproject.toml).

Subcommands:

```
david ingest                          # run scrapers and LLM coders
david ingest --since 2026-05-01
david calibrate-coders                # fit coder_calibration.stan against gold
david fit                             # fit m01_forward.stan on adjudicated data
david sbc                             # run measurement SBC
david sbc forecast                    # run forecast SBC
david falsify                         # run F1..F15
david forecast --horizon 6
david forecast --horizon 12 --cell c=AL,p=tax,k=t6
david route                           # apply FG1..FG6 and emit route ledger
david serve                           # read-only API on port 8080
```

Each subcommand writes a typed JSON result and an exit code. CI / cron uses
the exit code; humans use the dashboard.

Cron schedule (in `scripts/`):
- `nightly_ingest.sh` runs `david ingest` and `david calibrate-coders`
- `weekly_fit.sh` runs `david fit` and `david sbc forecast` and `david falsify`
- `weekly_forecast.sh` runs `david forecast --horizon {6,9,12}` and `david route`

Human interaction is restricted to:
1. Reviewing the adjudicator queue (capped at 4h / cycle).
2. Reviewing the route ledger (no edits, only acceptance).
3. Approving quarterly source-independence ledger updates.

Everything else is automated.

---

## 9. Data layout

```
canonical/data/
  raw/{source_id}/{YYYY-MM-DD}.jsonl       # scraper output
  coded/{coder_id}/{YYYY-MM-DD}.jsonl      # LLM/human labels
  adjudicated/gold_b_v{N}.csv              # adjudicated truth
  gold/gold_b_calibration.csv              # calibration set for coder kappa
  fits/{run_id}/                           # cmdstanpy output
    fit_summary.json
    draws.parquet
    diagnostics.json
  forecasts/{run_id}/
    cells.parquet                          # per-cell forecast object
    route_ledger.json
    falsification_ledger.json
```

`raw/`, `coded/`, `adjudicated/`, `gold/` are append-only with content-hash
addressing. `fits/` and `forecasts/` are immutable per run.

---

## 10. Versioning and reproducibility

Every artifact carries:
- `model_version` (Stan model SHA)
- `code_version` (git SHA of canonical/)
- `data_cutoff` (latest evidence_date included)
- `pre_registration_version` (m01_preregistration_v3.json reference)
- `theorem_packet_versions` (A', B', C, D-forecast SHAs)

Full reproduction: `david replay --run-id <id>` reruns end-to-end from raw
data using the recorded versions.

---

## 11. What the user must implement vs. what is scaffolded

Scaffolded in this drop (skeleton present, fill in):
- `david/cli.py` — argument parsing and dispatcher
- `david/engine/orchestrator.py` — pipeline runner
- `david/engine/forecast.py` — horizon h* enforcement
- `david/engine/router.py` — FG1..FG6 routing
- `david/engine/identification_distance.py` — Theorem A' diagnostic
- `david/engine/observability_sensitivity.py` — lambda bounds
- `david/theorems/{A_prime,B_prime,C_renamed,D_forecast_horizon}.py` — math kernels
- `david/simulator/{sbc,forecast_sbc,synthetic_world,adversarial_battery}.py`
- `david/validation/{falsification,scoring,reliability_diagrams,murphy_decomposition}.py`
- `david/ingest/sources.py`, `adjudicator_queue.py`, `human_loop_budget.py`
- `stan/m01_forward.stan` — extends council_m01 fit with forward block
- `stan/coder_calibration.stan` — Dawid-Skene gold-calibrated
- `stan/synthetic_generator.stan` — generative twin for SBC
- `tests/` — pytest skeletons for each module

You must implement:
- Scraper adapters per source (4 to 8 sources, one module each)
- LLM coder backends (OpenAI / Anthropic / local) behind common interface
- Source-independence ledger initial population (quarterly review)
- Gold-standard expansion procedure with adjudicator UI integration

You must reuse from council_m01:
- The existing fit_contract gates F1..F12 (import directly)
- The existing real_readiness preflights
- The theorem packets as ground-truth proof references

---

## 12. Migration path from current state

The repository today is at:
- M01-011: FAIL_CLOSED_DIAGNOSTICS_REQUIRED (data acquisition blocker)
- M01-012: PACKET_READY_DATA_REQUIRED
- M02 cross-domain governance: in progress

This canonical layer does NOT unblock M01-011 by itself. It provides the
forward-prediction layer on top of M01-011 once that closes. Specifically:

Step 1: Land canonical/ in tree; CI green on synthetic SBC and forecast SBC.
Step 2: Wire `david ingest` for one source family; LLM coder against gold;
        confirm Dawid-Skene recovers a known coder reliability profile.
Step 3: Connect to council_m01 fit artifacts as soon as M01-011 lands.
Step 4: Run forecast SBC end-to-end; pass F14 before any real forecast.
Step 5: Emit first real 6-month forecast for one (c, p, k) stratum that
        satisfies all six FG gates. Route everything else.

Until Step 5 lands, all forecast outputs are explicitly labeled
`pre_validation_demo` and disabled from any external interface.

---

## 13. Where the math sits

The mathematical AI engine is not metaphor. Every gate and every routing
decision in this architecture is a check of a stated mathematical claim:

- A' practical identifiability -> identification_distance.py
- B' channel informativeness -> B_prime.py informativeness_I_O()
- C posterior FDP control -> router.py compute_posterior_fdp_threshold()
- D-forecast horizon bound -> D_forecast_horizon.py compute_h_star()
- HSMM with pi_ii = 0 -> stan model log_jump masks diagonal
- Right-censored final segment -> hsmm_right_censored_lpdf in Stan
- Dawid-Skene coder reliability -> coder_calibration.stan + Stan kappa
- Endogenous observability sensitivity -> observability_sensitivity.py
- Multi-source structural independence -> ingest.source_independence_ledger
- Forecast calibration -> validation.scoring proper-scoring rules

If any of these checks fail, the output is gated. The thesis is not "we
predicted X with probability Y"; the thesis is "under the stated theorems
and the stated falsification battery, the model has either earned the right
to predict or has fail-closed honestly."

That distinction is what makes the system defensible.

---

## 14. Documentation Standards & Scientific Publication Rule

To preserve the academic and mathematical rigor of the DAVID project, all written documentation, thesis chapters, and design reports must adhere to the following permanent standards:

1. **LaTeX Requirement:** Any formal document describing the mathematics, implementation, or results of the predictive engine must be authored in LaTeX (`.tex` files) and compiled into scientific-publication-ready PDFs. Plain-text markdown files should only serve as quick summaries or indices.
2. **Premium Vector Graphics:** Diagrams, workflows, and charts must not use hand-drawn shapes or rasterized screenshots. They must be defined programmatically as vector graphics using **TikZ** (for flowcharts/system maps) and **pgfplots** (for probability distributions, Bayes error bounds, and trajectory curves) directly inside the LaTeX source.
3. **Reproducibility:** The LaTeX source files must be checked into the `docs/` folder alongside their compiled PDFs so they can be re-compiled and verified by any reviewer using standard tools (such as `tectonic` or `pdflatex`).
