"""Generative twin of m01_forward.stan in pure Python.

Used for SBC and the adversarial battery. Keeps numpy-only so it can run
without cmdstanpy and act as a parallel correctness check on the Stan model.

The generative process strictly mirrors m01_forward.stan:
  1) HSMM with pi_ii = 0 and shifted-Poisson dwell h_i(d)
  2) Per-tactic activity A_{ig,k} ~ Bernoulli(inv_logit(alpha_activity[z, k]))
  3) Source detection:
       delta_s = delta_max * inv_logit(delta_raw[s] + delta_obs[s] * O)
       j_s     = inv_logit(j_raw[s] + j_obs[s] * O)
       rho_s   = delta_s + (1-delta_s) * j_s
       B_{ig,s} ~ Bernoulli(rho_s if A=1 else delta_s)
  4) Coder labels Y_{e,r} ~ Dawid-Skene given B with item ambiguity
  5) Selection P(selected[k] | O, A_k) per unit — matches Stan's per-unit likelihood
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HyperPrior:
    R: int          # number of series (country x policy)
    T: int          # observed history length in months
    L: int          # number of regimes
    K: int          # tactic classes
    S: int          # source families
    M: int          # coders
    H: int          # forecast horizon


@dataclass
class WorldDraw:
    # Truth
    theta: dict           # parameter dict
    z: np.ndarray         # (R, T+H) true regimes
    a: np.ndarray         # (R, T+H, K) true activity 0/1
    # Observed during history
    b: np.ndarray         # (R, T, K, S) source detection 0/1
    y: np.ndarray         # (R, T, K, S, M) coder labels 0/1
    selected: np.ndarray  # (R, T, K) per-unit selection indicator
    observability: np.ndarray  # (R, T)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _sample_pi_stan_prior(rng: np.random.Generator, L: int) -> np.ndarray:
    """Sample Pi using the same construction as Stan's jump_raw prior.

    Stan: jump_raw[i, j] ~ normal(0, 1); diagonal masked to -inf before
    log-sum-exp normalisation per row. This matches the Stan transformed-
    parameters block exactly so SBC rank statistics are valid.
    """
    Pi = np.zeros((L, L))
    for i in range(L):
        raw = rng.normal(0.0, 1.0, size=L)
        raw[i] = -np.inf
        finite = np.isfinite(raw)
        lse = np.log(np.sum(np.exp(raw[finite] - raw[finite].max()))) + raw[finite].max()
        for j in range(L):
            if j == i:
                Pi[i, j] = 0.0
            else:
                Pi[i, j] = np.exp(raw[j] - lse)
    return Pi


def _logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def sample_world(prior: HyperPrior, seed: int | None = None) -> WorldDraw:
    rng = np.random.default_rng(seed)
    R, T, L, K, S, M, H = prior.R, prior.T, prior.L, prior.K, prior.S, prior.M, prior.H

    # Parameters (priors must match Stan model exactly for SBC to be valid)
    Pi = _sample_pi_stan_prior(rng, L)
    dwell_lambda = np.exp(rng.normal(np.log(6.0), 0.5, size=L))  # synced to Stan: lognormal(log(6),0.5)
    alpha_activity = rng.normal(-1.5, 1.5, size=(L, K))  # synced to Stan: N(-1.5,1.5); mean phi≈0.18
    alpha_activity[:, 0] = np.sort(alpha_activity[:, 0])  # order statistics — matches Stan ordered[L]
    # Detection: Stan parameterization — delta_raw/j_raw with observability slopes
    delta_raw = rng.normal(0.0, 1.0, size=S)
    j_raw = rng.normal(0.0, 1.0, size=S)
    delta_observability = rng.normal(0.0, 0.5, size=S)
    j_observability = rng.normal(0.0, 0.5, size=S)
    kappa_plus = 0.5 + 0.5 * _logistic(rng.normal(1.0, 0.5, size=M))   # synced to Stan: normal(1,0.5) mean kappa≈0.87
    kappa_minus = 0.5 + 0.5 * _logistic(rng.normal(1.0, 0.5, size=M))  # reverted: lower kappa↑Y-rate when P(b0)>P(b1)
    # init: softmax of normal(0,1) — same as Stan's log_softmax(init_raw)
    init_raw = rng.normal(0.0, 1.0, size=L)
    init = _softmax(init_raw)
    selection_alpha = rng.normal(0.0, 1.0)
    selection_observability = rng.normal(0.0, 0.5)
    selection_activity = rng.normal(0.0, 0.5)

    theta = {
        "Pi": Pi, "dwell_lambda": dwell_lambda, "alpha_activity": alpha_activity,
        "delta_raw": delta_raw, "j_raw": j_raw,
        "delta_observability": delta_observability, "j_observability": j_observability,
        "kappa_plus": kappa_plus, "kappa_minus": kappa_minus,
        "init": init,
        "selection_alpha": selection_alpha,
        "selection_observability": selection_observability,
        "selection_activity": selection_activity,
    }

    z = np.zeros((R, T + H), dtype=int)
    a = np.zeros((R, T + H, K), dtype=int)
    b = np.zeros((R, T, K, S), dtype=int)
    y = np.zeros((R, T, K, S, M), dtype=int)
    selected = np.zeros((R, T, K), dtype=int)
    observability = np.zeros((R, T))

    for r in range(R):
        # series-level observability covariate (slow drift)
        observability[r] = np.clip(rng.normal(0.0, 1.0, size=T).cumsum() / np.sqrt(T), -2.0, 2.0)
        observability[r] = (observability[r] - observability[r].min()) / (
            observability[r].max() - observability[r].min() + 1e-9
        )

        # Sample HSMM segment path
        t = 0
        cur = int(rng.choice(L, p=init))
        while t < T + H:
            dwell = 1 + int(rng.poisson(dwell_lambda[cur]))
            end = min(t + dwell, T + H)
            z[r, t:end] = cur
            t = end
            if t >= T + H:
                break
            row = Pi[cur].copy()
            row = row / row.sum()
            cur = int(rng.choice(L, p=row))

        # Activity per timestep
        for tt in range(T + H):
            probs = _logistic(alpha_activity[z[r, tt]])
            a[r, tt] = (rng.uniform(size=K) < probs).astype(int)

        # delta_max mirrors the Stan data constant (0.30 pre-registered)
        _delta_max = 0.30

        # Observations during T history
        for tt in range(T):
            O = observability[r, tt]
            for k in range(K):
                a_val = a[r, tt, k]
                # Matches Stan: delta_us = delta_max * inv_logit(delta_raw[s] + delta_obs[s]*O)
                #               j_us     = inv_logit(j_raw[s] + j_obs[s]*O)
                #               rho_us   = delta_us + (1-delta_us)*j_us
                delta_t = _delta_max * _logistic(delta_raw + delta_observability * O)
                j_t = _logistic(j_raw + j_observability * O)
                rho_t = delta_t + (1.0 - delta_t) * j_t
                p_detect = np.where(a_val == 1, rho_t, delta_t)
                b[r, tt, k] = (rng.uniform(size=S) < p_detect).astype(int)
                for s in range(S):
                    if b[r, tt, k, s] == 1:
                        y[r, tt, k, s] = (rng.uniform(size=M) < kappa_plus).astype(int)
                    else:
                        y[r, tt, k, s] = (rng.uniform(size=M) < (1 - kappa_minus)).astype(int)
                # Per-unit selection: each (r, t, k) is independently selected.
                # Stan's likelihood counts selected[u] once per unit u=(r,t,k),
                # with the logit depending on whether THIS tactic is active.
                logit_sel = (
                    selection_alpha
                    + selection_observability * O
                    + selection_activity * float(a_val)
                )
                selected[r, tt, k] = int(rng.uniform() < _logistic(logit_sel))

    return WorldDraw(
        theta=theta, z=z, a=a, b=b, y=y,
        selected=selected, observability=observability,
    )
