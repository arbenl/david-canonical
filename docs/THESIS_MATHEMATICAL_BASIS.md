# DAVID / M0.1 — Thesis Mathematical Basis

**Scope.** This document is the scientific foundation for the thesis chapters on
the capture-resistant prediction engine. It states each enforced theorem in its
formal form, gives the proof base, and links the claim to (i) the exact Python
kernel that implements it, (ii) the test that exercises it, and (iii) the
front-end node where it becomes visible to a reader of the dashboard.

All paths are relative to the `canonical/` repository root. Pre-registered
floors live in [`david/config.py`](../david/config.py) and are reproduced inline.

> **Naming note.** The repository uses primed names (A′, B′, C′, D′) in code and
> gate dictionaries. The thesis names below (A′, B′.1, B′.2, C, D-forecast) map
> onto the same kernels; the mapping is given per section. "C′" in the fit
> summary is the same object as "Theorem C" here.

---

## 0. Summary table

| Theorem | Question it answers | Kernel | Gate site | Floor (config) | Test | Front-end node |
|---|---|---|---|---|---|---|
| **A′** | Is the latent class *practically* identifiable in stratum *g*? | [`theorems/A_prime.py`](../david/theorems/A_prime.py) | `model/fit.py::_run_theorem_gates` → `A_prime` | `ID_DISTANCE_FLOOR = 0.05` | [`tests/test_identification_distance.py`](../tests/test_identification_distance.py) | Router curve-lock + A′ badge; Engine "Practical identifiability"; Roadmap node `theorems` |
| **B′.1** | Single-observation Bayes error at informativeness *I* | [`theorems/B_prime.py`](../david/theorems/B_prime.py)`::bayes_classification_error` | informational floor (cell-level) | — | indirect ([`tests/test_forecast_engine.py`](../tests/test_forecast_engine.py)) — *see §2.4* | Engine "Source informativeness"; Router B′ badge |
| **B′.2** | Effective sample size for estimating φ under *N·I²* | [`theorems/B_prime.py`](../david/theorems/B_prime.py)`::effective_sample_size` | `model/fit.py` → `B_prime` (lower-95 % I) | `INFORMATIVENESS_FLOOR_LOWER_95 = 0.10` | indirect — *see §2.4* | Engine "Source informativeness"; Router B′ badge |
| **C** | Which cells can be flagged active at controlled FDP? | [`theorems/C_renamed.py`](../david/theorems/C_renamed.py) | `model/fit.py` → `C_prime`; `engine/router.py` | `POSTERIOR_FDP_DEFAULT_Q = 0.10` | [`tests/test_posterior_fdp.py`](../tests/test_posterior_fdp.py) | Engine "Posterior FDP routing"; Validate row F9 |
| **D-forecast** | Up to what horizon is the forecast informative, not prior? | [`theorems/D_forecast_horizon.py`](../david/theorems/D_forecast_horizon.py) | `model/fit.py` → `D_prime`; `engine/forecast.py` | `HORIZON_PRIOR_DRIFT_TAU = 0.50` | [`tests/test_horizon_validity.py`](../tests/test_horizon_validity.py) | Router prior-dominated tail + `h*` caption; Predict route pills; Engine "Forecast horizon validity" |

The four gates are **fail-closed**: an exception in any kernel surfaces as
`gate_status = "error"`, which is treated as a failure (never as a pass) when
the overall fit gate is computed in
[`david/model/fit.py`](../david/model/fit.py) (`run_fit`).

---

## 1. Theorem A′ — Practical identifiability & Kruskal rank

### 1.1 Setting

Within a stratum *g* (country × policy), each evidence item is coded by
*S* conditionally independent binary sources. Source *s* has detection
parameters $\rho_s = P(B_s = 1 \mid A = 1)$ (sensitivity) and
$\delta_s = P(B_s = 1 \mid A = 0)$ (false-positive rate), and $\phi_g = P(A=1)$
is the marginal interference-activity probability in the stratum. The observed
object is the replicated joint law $P(B_1, \dots, B_S)$.

### 1.2 Claim (generic identifiability)

The parameters $(\phi_g, \{\rho_s, \delta_s\}_{s=1}^{S})$ are **generically
identifiable up to label permutation** provided

1. $S_{\text{eff}} \ge 3$ conditionally independent *informative* sources;
2. $\rho_s \neq \delta_s$ for each used source (non-degenerate channel);
3. $\phi_g$ is not on the boundary $\{0, 1\}$.

### 1.3 Proof base

The two-class product-Bernoulli latent-class model is a three-way tensor

$$
T_{ijk} = \sum_{a \in \{0,1\}} \pi_a \, M^{(1)}_{ia} M^{(2)}_{ja} M^{(3)}_{ka},
$$

whose CP (Kruskal) decomposition is **generically unique up to the label swap of
the two classes** when each factor matrix has **Kruskal rank ≥ 2**, by
**Kruskal (1977)** and its latent-class specialization
**Allman–Matias–Rhodes (2009)**. Three conditionally independent sources supply
the three tensor modes; Kruskal rank 2 per source is exactly condition (2). The
"≥ 3 independent sources" requirement is therefore the operational expression of
the tensor-uniqueness condition, and is what the front-end "Theorem A′" message
refers to.

Full proof packet:
[`council_m01/theorem_packets/A_prime_latent_class_identifiability.md`](../../council_m01/theorem_packets/A_prime_latent_class_identifiability.md).

### 1.4 Practical-identifiability distance (the operational gate)

Generic identifiability is a measure-zero statement; finite data can still sit
arbitrarily close to the singular set. The engine therefore measures a
**posterior identification distance** per draw θ:

$$
d(\theta) = \min\!\Big(
\min_s \min\big(|\rho_s-\delta_s|,\;|\rho_s-(1-\delta_s)|\big),\;
\min(\phi_g,\,1-\phi_g)
\Big).
$$

The first term guards against a non-informative or label-flipped source; the
second against a boundary $\phi_g$. The gate passes iff the posterior **median**
of $d(\theta)$ is at least `ID_DISTANCE_FLOOR = 0.05`; otherwise the stratum is
flagged `practically_non_identified` and its forecast is withheld.

- Kernel: [`A_prime.py::identification_distance_draws`](../david/theorems/A_prime.py) and `check_stratum`.
- Gate assembly: [`model/fit.py::_run_theorem_gates`](../david/model/fit.py) (`gates["A_prime"]`).
- Test: [`tests/test_identification_distance.py`](../tests/test_identification_distance.py).
- Front-end: a failed A′ greys the Forecast-Probabilities curve and renders
  *"Parashikimi i bllokuar: Teorema A′ dështoi (Mungesë burimesh të pavarura për
  Kosovën)…"* in [`david-ui/src/components/forecast-router.tsx`](../david-ui/src/components/forecast-router.tsx);
  the A′ status chip in "Active Routing Status"; the GateCard "Practical
  identifiability" in [`david-ui/src/app/engine/page.tsx`](../david-ui/src/app/engine/page.tsx);
  and the `theorems` node of [`roadmap-diagram.tsx`](../david-ui/src/components/roadmap-diagram.tsx).

---

## 2. Theorem B′ — Bayes error & information scaling under *N·I²*

The informativeness of a detection channel at observability *O* is

$$
I(O) = |\rho(O) - \delta(O)| \in [0, 1].
$$

Kernel: [`B_prime.py::informativeness_draws`](../david/theorems/B_prime.py).

### 2.1 Theorem B′.1 — single-observation Bayes error

**Claim.** Under equal prior odds, the minimum classification error for
recovering $A$ from a single detection $Y$ is

$$
\varepsilon^\star = \tfrac{1}{2}\big(1 - I(O)\big).
$$

**Proof.** For two equiprobable hypotheses the Bayes error is
$\tfrac12(1 - \mathrm{TV})$, where $\mathrm{TV}$ is the total-variation distance
between the two class-conditional laws of $Y$. For a binary channel
$\mathrm{TV} = |\rho - \delta| = I$. ∎

Kernel: `B_prime.py::bayes_classification_error(I)` returns $0.5(1-I)$.

### 2.2 Theorem B′.2 — information scaling

**Claim.** With $N$ replicate detections from a channel
$Y \sim \text{Bernoulli}(\delta + I\,\phi)$, the information about $\phi$ scales as

$$
N_{\text{eff}}(\phi) \;\propto\; N \cdot I^2,
$$

so the effective sample size collapses **quadratically** as $I \to 0$. Cells with
small $I$ are prior-dominated regardless of $N$.

**Proof base.** The Fisher information of a Bernoulli mean w.r.t. $\phi$ carries
a factor $(\partial p/\partial\phi)^2 = I^2$; summing $N$ i.i.d. replicates gives
$N\,I^2$ up to the Bernoulli-variance factor. This is the estimation-theoretic
counterpart of B′.1's classification bound.

Kernel: `B_prime.py::effective_sample_size(N, I)` returns $N \cdot I^2$.

### 2.3 The operational gate

The cell gate is on the **lower 95 % credible bound** of $I(O)$:
it passes iff $I_{\text{lower 95\%}} \ge$ `INFORMATIVENESS_FLOOR_LOWER_95 = 0.10`.
Assembled in [`model/fit.py`](../david/model/fit.py) as `gates["B_prime"]`
(`lower_95_I_worst_source`). A failing B′ marks the cell `prior_dominated`.

### 2.4 Test-coverage note (honest gap)

The B′ kernels are currently exercised **indirectly** through
[`tests/test_forecast_engine.py`](../tests/test_forecast_engine.py) (the fit-summary
B′ fields are read by the forecast emitter). There is **no dedicated unit test**
asserting $\varepsilon^\star = \tfrac12(1-I)$ or the $N\cdot I^2$ scaling. For the
thesis's reproducibility claim, adding `tests/test_informativeness.py` with these
two algebraic identities is recommended.

- Front-end: GateCard "Source informativeness" in `engine/page.tsx`; B′ status
  chip in the Router "Active Routing Status" card.

---

## 3. Theorem C — Posterior expected-FDP thresholding

### 3.1 Claim

Let $p_i = P(A_i = 1 \mid \text{data})$ be the posterior activity probability of
cell $i$, $i = 1,\dots,M$. Sort descending $p_{(1)} \ge \dots \ge p_{(M)}$ and
take the largest prefix

$$
m^\star = \max\Big\{ m : \tfrac{1}{m}\sum_{j=1}^{m}\big(1 - p_{(j)}\big) \le q \Big\}.
$$

Flagging the top $m^\star$ cells guarantees that the **posterior expected false
discovery proportion** among flagged cells is at most $q$.

### 3.2 Proof

The posterior expected number of false discoveries in the flagged set is
$\sum_{j\le m}(1-p_{(j)})$ by linearity of expectation over the marginal
posteriors; dividing by $m$ gives the posterior expected FDP. The construction
chooses the largest $m$ keeping this average $\le q$. The argument uses **only
linearity of expectation** and so holds **without an independence assumption** —
HSMM correlation is already absorbed into the joint posterior. This is a
Bayesian FDR control (cf. Newton et al. 2004; Müller et al. 2004), **not** a
frequentist Benjamini–Hochberg procedure.

- Kernel: [`C_renamed.py::compute_posterior_fdp_threshold`](../david/theorems/C_renamed.py).
- Gate assembly: `model/fit.py` → `gates["C_prime"]`. **C routes; it does not
  block the fit** (`gate_status` is set to `"pass"` and the threshold/flag set is
  reported), so it tightens the active-cell set rather than failing the run.
- Applied operationally in [`engine/router.py`](../david/engine/router.py) after the FG1–FG6 checks.
- Test: [`tests/test_posterior_fdp.py`](../tests/test_posterior_fdp.py).
- Front-end: GateCard "Posterior FDP routing" in `engine/page.tsx`; row **F9**
  ("Posterior FDP routing (C′)") in [`david-ui/src/app/validate/page.tsx`](../david-ui/src/app/validate/page.tsx).

---

## 4. Theorem D-forecast — Horizon validity under ergodic transition drift

### 4.1 Setting

Regimes follow a hidden semi-Markov model (HSMM) with embedded transition matrix
$\Pi$ ($\pi_{ii} = 0$, no self-transitions) and dwell-mean vector $\mu$. The
time-weighted stationary marginal is

$$
\pi_\infty[r] = \frac{\nu[r]\,\mu[r]}{\sum_j \nu[j]\,\mu[j]}, \qquad
\nu = \text{left eigenvector of } \Pi \text{ with eigenvalue } 1,
$$

i.e. the embedded-chain stationary distribution $\nu$ re-weighted by expected
dwell time (`stationary_marginal_embedded` → `stationary_marginal_time`).

### 4.2 Claim

The conditional forecast $P(Z_{t+h} \mid Z_t, \text{history})$ converges to
$\pi_\infty$ as $h \to \infty$ (ergodicity). Define the **prior-drift share**

$$
\text{drift}(h) = 1 - \frac{\mathrm{TV}(\text{forecast}_h,\ \pi_\infty)}
{\mathrm{TV}(\text{forecast}_h,\ \pi_\infty) + \mathrm{TV}(\text{forecast}_h,\ z_t)},
$$

the fraction of the forecast's total-variation movement that is **toward the
stationary marginal**. The **horizon-validity bound** is

$$
h^\star(g) = \max\{\,h : \text{drift}(h) < \tau\,\}, \qquad \tau = 0.50.
$$

For $h > h^\star(g)$ the conditional forecast has collapsed toward $\pi_\infty$;
the cell is routed `horizon_prior_dominated` and the engine returns the marginal
regime prediction instead of a conditional one.

### 4.3 Proof base

Standard **ergodic theorem** for the embedded segment chain (irreducible,
aperiodic given $\pi_{ii}=0$ and a connected $\Pi$) guarantees convergence of the
$h$-step law to $\nu$; time-weighting by $\mu$ gives convergence of the regime
marginal to $\pi_\infty$. The drift share is computed by a Monte-Carlo
decomposition over posterior draws of $(\Pi, \mu, Z_T)$ using the embedded chain
(`forecast_regime_distribution`, `horizon_validity`).

### 4.4 Integration into `/forecast-cells` (verified end-to-end)

`h^\star` is computed **once** in `run_fit` (stationary-weighted over per-regime
$h^\star$ values) and stored in the fit summary under `theorems.D_prime.
h_star_months`. The forecast emitter then gates each cell:

```
below_h_star  = (horizon_months <= h_star)
forecast_route = "conditional_forecast" if below_h_star
                 else "horizon_prior_dominated"
```

- Kernel: [`D_forecast_horizon.py::horizon_validity`](../david/theorems/D_forecast_horizon.py).
- `h^\star` computed: [`model/fit.py::_run_theorem_gates`](../david/model/fit.py) (`gates["D_prime"]`).
- Applied per cell: [`engine/forecast.py::emit_forecasts`](../david/engine/forecast.py).
- Persisted: `db/repositories.py::write_forecast_cells` → columns `h_star_months,
  below_h_star, forecast_route`.
- Served: `db/repositories.py::get_forecast_cells` →
  `GET /forecast-cells` in [`david/api/server.py`](../david/api/server.py).
- Test: [`tests/test_horizon_validity.py`](../tests/test_horizon_validity.py).
- Front-end: the prior-dominated tail ($h > h^\star$) is greyed behind a dashed
  `h>h*` boundary with the caption *"Theorem D-forecast: h* = … muaj"* in
  [`forecast-router.tsx`](../david-ui/src/components/forecast-router.tsx); the
  conditional/prior-dominated route pills in
  [`predict/page.tsx`](../david-ui/src/app/predict/page.tsx); GateCard "Forecast
  horizon validity" in `engine/page.tsx`.

**Live confirmation.** With $h^\star = 6$, `GET /forecast-cells` returns
`below_h_star=true, route=conditional_forecast` for the 3 m and 6 m horizons, and
`below_h_star=false, route=horizon_prior_dominated` for 9 m and 12 m — and the
front-end greys the 9 m–12 m segment accordingly.

---

## 5. Fail-closed calibration layer (SBC) and the trust boundary

The theorem gates above bound *what the model can claim once it is trusted*. A
separate **simulation-based calibration** (SBC, Talts et al. 2018) layer decides
*whether the model is trusted at all*:

- Measurement SBC (`simulator/sbc.py`) checks rank-statistic uniformity per
  parameter (KS test, Bonferroni-corrected, $\alpha = 0.05$).
- Forecast SBC (`simulator/forecast_sbc.py`) checks nominal 80 %/95 % coverage
  bands.
- Served by `GET /sbc`; consumed by the front-end as a **hard global lock**: if
  either layer reports `gate_status = "fail"`, the entire Forecast Router is
  overlaid with *"Motori i bllokuar: Modeli nuk është i kalibruar matematikisht
  lokalisht."*

The trust boundary is therefore two-tier and strictly fail-closed:

```
SBC fail            → whole engine locked   (calibration not established)
Theorem A′/B′ fail  → that stratum's curve locked   (claim not licensed)
both pass           → certified: "🛡️ Parashikim i Certifikuar Matematikisht"
```

The front-end fetch layer uses `cache: "no-store"`
([`david-ui/src/lib/api.ts`](../david-ui/src/lib/api.ts)) so a gate flip on the
backend is reflected immediately — a stale "unlocked" view can never outlive a
backend `fail`.

---

## 6. Reproduction

```bash
# 1. Postgres (any free host port; 5432 may be taken)
docker run -d --name david_pg -e POSTGRES_USER=david -e POSTGRES_PASSWORD=david \
  -e POSTGRES_DB=david -p 5544:5432 postgres:16
export DATABASE_URL=postgresql://david:david@localhost:5544/david

# 2. Schema + theorem tests
david db init
pytest tests/test_identification_distance.py tests/test_horizon_validity.py \
       tests/test_posterior_fdp.py tests/test_forecast_engine.py -q   # 21 passed

# 3. API + UI
python -m uvicorn david.api.server:api --port 8080      # backend
cd david-ui && npm run dev                              # http://localhost:3001/router
```

A full MCMC fit (`david fit`) requires a compiled CmdStan model
(`stan/m01_forward.stan`); the gate logic and front-end lock states can be
exercised without it by seeding a fit fixture — see
[`scripts/seed_verify.py`](../scripts/seed_verify.py).

---

## 7. References

- Kruskal, J. B. (1977). *Three-way arrays: rank and uniqueness of trilinear
  decompositions.* Linear Algebra Appl. 18, 95–138.
- Allman, E. S., Matias, C., Rhodes, J. A. (2009). *Identifiability of parameters
  in latent structure models with many observed variables.* Ann. Statist. 37(6A).
- Newton, M. A., et al. (2004). *Detecting differential gene expression with a
  semiparametric hierarchical mixture method.* Biostatistics 5(2).
- Müller, P., Parmigiani, G., et al. (2004). *Optimal sample size for multiple
  testing: the case of gene expression microarrays.* JASA 99(468).
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., Gelman, A. (2018).
  *Validating Bayesian inference algorithms with simulation-based calibration.*
  arXiv:1804.06788.
- Dawid, A. P., Skene, A. M. (1979). *Maximum likelihood estimation of observer
  error-rates using the EM algorithm.* Appl. Statist. 28(1).
