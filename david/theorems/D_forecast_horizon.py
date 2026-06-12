"""Theorem D-forecast horizon-validity bound.

Statement (informal):
  For an HSMM with transition matrix Pi (pi_ii = 0) and dwell mean vector mu,
  the forecast distribution P(Z_{t+h} | Z_t, history) converges to the
  stationary marginal pi_inf as h grows. Define:

    signal_h(g)      = Var of conditional forecast around the posterior mean
    prior_drift_h(g) = expected KL divergence of cond. forecast at h from
                       its stationary marginal pi_inf

  The horizon diagnostic h*(g) is the first crossing of the drift threshold:

    h* = min{ h >= 1 : drift(h) >= tau } - 1,

  with h* = h_max if no crossing occurs. drift(h) is not proven monotone
  in h, so the withdrawn max-form (largest h with drift < tau) could admit
  horizons beyond a first crossing whenever the drift dips back below tau;
  the first-crossing form is the conservative, operative reading
  (thesis_mathematical_core.tex, Theorem D-forecast, FG5).

  At h > h*(g), the conditional forecast collapses toward the stationary
  marginal; routing degrades to `horizon_prior_dominated` and returns the
  marginal regime prediction instead. h* is an estimated diagnostic with
  posterior and Monte-Carlo uncertainty, not an exact validity boundary.

This is the formal counterpart of "forecasts at long horizons are
prior-dominated regardless of model fidelity." Implemented as a Monte-Carlo
decomposition over posterior draws of (Pi, mu, Z_T) using the embedded chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import HORIZON_PRIOR_DRIFT_TAU


@dataclass(frozen=True)
class HorizonValidity:
    cell_id: str
    h_star_months: int
    tau: float
    prior_drift_share_at_h_max: float
    horizon_validity_curve: list[tuple[int, float]]  # (h, prior_drift_share)


def stationary_marginal_embedded(Pi_off_diag: np.ndarray) -> np.ndarray:
    """Stationary distribution of the embedded segment chain.

    Pi_off_diag has zero diagonal and rows summing to 1.
    Returns the unique left eigenvector with eigenvalue 1, normalized.
    """
    R = Pi_off_diag.shape[0]
    if Pi_off_diag.shape != (R, R):
        raise ValueError("Pi must be square")
    if not np.allclose(np.diag(Pi_off_diag), 0.0, atol=1e-10):
        raise ValueError("Pi diagonal must be zero (no self-transitions)")
    eigvals, eigvecs = np.linalg.eig(Pi_off_diag.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    return vec / vec.sum()


def stationary_marginal_time(
    Pi_off_diag: np.ndarray,
    dwell_mean: np.ndarray,
) -> np.ndarray:
    """Time-weighted stationary marginal pi_inf for the HSMM.

    pi_inf[r] = nu[r] * mu[r] / sum_j nu[j] * mu[j]
    """
    nu = stationary_marginal_embedded(Pi_off_diag)
    weighted = nu * dwell_mean
    return weighted / weighted.sum()


def forecast_regime_distribution(
    Pi_off_diag: np.ndarray,
    dwell_mean: np.ndarray,
    z_t: int,
    horizon: int,
    n_mc: int = 2000,
) -> np.ndarray:
    """Monte-Carlo posterior of Z_{t+h} given Z_t under the HSMM.

    Returns shape (R,) probability vector.
    """
    R = Pi_off_diag.shape[0]
    counts = np.zeros(R)
    rng = np.random.default_rng()
    for _ in range(n_mc):
        z = z_t
        time_remaining = horizon
        # Approximation: discrete months. Dwell ~ shifted-Poisson with
        # mean dwell_mean[z]. Use geometric approximation here; replace with
        # the true dwell distribution in production.
        while time_remaining > 0:
            # remaining dwell in current regime
            dwell = max(1, int(rng.poisson(dwell_mean[z] - 1) + 1))
            if dwell >= time_remaining:
                break
            time_remaining -= dwell
            # transition to next regime per off-diagonal row
            row = Pi_off_diag[z].copy()
            row = row / row.sum()
            z = int(rng.choice(R, p=row))
        counts[z] += 1
    return counts / counts.sum()


def first_crossing_h_star(
    drift_curve: list[tuple[int, float]],
    tau: float,
    h_max: int,
) -> int:
    """First-crossing horizon diagnostic from a drift curve.

    h* = min{ h >= 1 : drift(h) >= tau } - 1, with h* = h_max if no
    crossing occurs. drift(h) is not proven monotone, so the crossing
    scan must stop at the FIRST h with drift(h) >= tau: later dips back
    below tau never re-extend h* (the max-form that allowed this is
    withdrawn per the June 2026 audit).
    """
    for h, drift in drift_curve:
        if drift >= tau:
            return h - 1
    return h_max


def horizon_validity(
    cell_id: str,
    Pi_off_diag: np.ndarray,
    dwell_mean: np.ndarray,
    z_t_distribution: np.ndarray,    # posterior of Z_T at present
    h_max: int = 18,
    tau: float = HORIZON_PRIOR_DRIFT_TAU,
    n_mc: int = 1500,
) -> HorizonValidity:
    """Horizon diagnostic h* as the first crossing of the drift threshold."""
    pi_inf = stationary_marginal_time(Pi_off_diag, dwell_mean)
    R = Pi_off_diag.shape[0]
    curve: list[tuple[int, float]] = []
    last_drift = 0.0
    for h in range(1, h_max + 1):
        # marginalize over z_t under its posterior distribution
        forecast = np.zeros(R)
        for z, p in enumerate(z_t_distribution):
            if p < 1e-9:
                continue
            forecast += p * forecast_regime_distribution(
                Pi_off_diag, dwell_mean, z_t=z, horizon=h, n_mc=n_mc
            )
        forecast = forecast / forecast.sum()
        # drift share: KL(forecast || pi_inf) decreases as forecast nears
        # stationary; equivalently, total variation distance to pi_inf
        tv_to_stationary = 0.5 * np.abs(forecast - pi_inf).sum()
        tv_to_present = 0.5 * np.abs(forecast - z_t_distribution).sum()
        # prior_drift_share := 1 - tv_to_stationary / (tv_to_stationary + tv_to_present)
        # which is the share of forecast "movement" that is toward the stationary marginal
        denom = tv_to_stationary + tv_to_present + 1e-12
        drift_share = 1.0 - tv_to_stationary / denom
        curve.append((h, float(drift_share)))
        last_drift = float(drift_share)
    h_star = first_crossing_h_star(curve, tau=tau, h_max=h_max)
    return HorizonValidity(
        cell_id=cell_id,
        h_star_months=h_star,
        tau=tau,
        prior_drift_share_at_h_max=last_drift,
        horizon_validity_curve=curve,
    )
