# DAVID/M0.1 Canonical Theorem Index

Single-page index of the mathematical claims this engine enforces. For full
proof packets see `../council_m01/theorem_packets/` (relative to repo root).

## Theorem A' — Source-channel latent-class identifiability

**Claim.** Within a stratum g, parameters phi_g, {rho_s, delta_s} are
generically identifiable from the replicated joint law P(B_1, ..., B_S) up to
label permutation, under (a) S_eff >= 3 conditionally independent informative
sources, (b) rho_s != delta_s for each used source, (c) phi_g not at boundary.

**Proof basis.** Two-class product-Bernoulli latent-class model. Three-way
tensor decomposition is generically unique up to label swap by
Allman-Matias-Rhodes (2009) under Kruskal rank-2 per source.

**Operational diagnostic.** `david/theorems/A_prime.py::identification_distance_draws`
computes d(theta) per draw. The gate is `engine/router.py` FG2 with
ID_DISTANCE_FLOOR.

## Theorem B'.1 — Bayes classification error at informativeness I

**Claim.** Under equal prior odds, the minimum classification error for
recovering A from a single Y at informativeness I = |rho(O) - delta(O)| is
(1 - I) / 2.

**Proof.** Bayes error for two equiprobable hypotheses is (1 - TV)/2, where
TV is the total variation distance between the two conditional distributions,
which equals I for binary Y.

**Operational use.** `david/theorems/B_prime.py::bayes_classification_error`
provides the cell-level information floor for cell-level activity claims.

## Theorem B'.2 — Information scaling for population probability

**Claim.** Information about phi from N replicates with detection channel
Y ~ Bernoulli(delta + I * phi) scales as N * I^2. Effective sample size
collapses quadratically as I -> 0.

**Operational diagnostic.** `engine/router.py` FG3 demands I lower 95% CI to
exceed INFORMATIVENESS_FLOOR_LOWER_95 (default 0.10).

## Theorem C — Bayesian posterior expected FDP control

**Claim.** Sort cells by posterior probability p_i descending. Take the
largest set whose mean (1 - p_i) is at most q. Posterior expected false
discovery proportion among flagged cells is at most q.

**Proof.** Linearity of expectation over the marginal posteriors. Holds
without independence; HSMM correlation is already in the posteriors.

**Operational gate.** `david/theorems/C_renamed.py::compute_posterior_fdp_threshold`
applied by `engine/router.py` after FG1..FG6 pass.

## Theorem D-forecast — Horizon validity bound

**Claim.** Under HSMM with pi_ii = 0 and dwell mean mu, the conditional
forecast distribution P(Z_{t+h} | Z_t, history) converges to the
time-weighted stationary marginal pi_inf as h grows. Define horizon-validity
h*(g) = max{h : prior_drift_share(h, g) < tau}. Forecasts at h > h*(g) collapse
to pi_inf and are reported as `horizon_prior_dominated`, not as conditional
predictions.

**Proof.** Standard ergodic theorem for the embedded segment chain plus
time-weighting. The drift share is the share of total variation movement
toward the stationary marginal.

**Operational gate.** `david/theorems/D_forecast_horizon.py::horizon_validity`
applied by `engine/forecast.py` FG5.

## Supporting design rules (not theorems but enforced)

- HSMM transition matrix Pi has pi_ii = 0 (m01_forward.stan log_jump masks diagonal).
- Final segment dwell is right-censored under the shifted-Poisson dwell.
- Coder reliability layer is Dawid-Skene with item ambiguity.
- Selection model is opportunity-frame-aware with logistic link.
- Endogenous-observability sensitivity is reported as a lambda interval, not point.

## Falsification battery

See `simulator/adversarial_battery.py` for F1..F15 implementations. Each test
is a check of one stated claim above or a baseline-dominance / no-leakage
sanity check.
