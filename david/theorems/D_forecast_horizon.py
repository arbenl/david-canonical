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
prior-dominated regardless of model fidelity." The production gate consumes the
Stan generated-quantities forecast path (`z_future`) so the diagnostic is tied
to the same posterior-predictive law that emits forecasts. The legacy
`horizon_validity` simulator remains for focused kernel tests and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import HORIZON_PRIOR_DRIFT_TAU


@dataclass(frozen=True)
class HorizonValidity:
    cell_id: str
    h_star_months: int
    h_star_q05: int
    h_star_q95: int
    tau: float
    prior_drift_share_at_h_max: float
    horizon_validity_curve: list[tuple[int, float]]  # (h, prior_drift_share)
    horizon_validity_curve_q05: list[tuple[int, float]]
    horizon_validity_curve_q95: list[tuple[int, float]]


def stationary_marginal_embedded(pi_off_diag: np.ndarray) -> np.ndarray:
    """Stationary distribution of the embedded segment chain.

    pi_off_diag has zero diagonal and rows summing to 1.
    Returns the unique left eigenvector with eigenvalue 1, normalized.
    """
    R = pi_off_diag.shape[0]
    if pi_off_diag.shape != (R, R):
        raise ValueError("Pi must be square")
    if not np.allclose(np.diag(pi_off_diag), 0.0, atol=1e-10):
        raise ValueError("Pi diagonal must be zero (no self-transitions)")
    eigvals, eigvecs = np.linalg.eig(pi_off_diag.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    vec = np.real(eigvecs[:, idx])
    vec = np.abs(vec)
    return vec / vec.sum()


def stationary_marginal_time(
    pi_off_diag: np.ndarray,
    dwell_mean: np.ndarray,
) -> np.ndarray:
    """Time-weighted stationary marginal pi_inf for the HSMM.

    pi_inf[r] = nu[r] * mu[r] / sum_j nu[j] * mu[j]
    """
    nu = stationary_marginal_embedded(pi_off_diag)
    weighted = nu * dwell_mean
    return weighted / weighted.sum()


def forecast_regime_distribution(
    pi_off_diag: np.ndarray,
    dwell_mean: np.ndarray,
    z_t: int,
    horizon: int,
    n_mc: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Monte-Carlo posterior of Z_{t+h} given Z_t under the HSMM.

    Returns shape (R,) probability vector.
    """
    R = pi_off_diag.shape[0]
    counts = np.zeros(R)
    rng = rng or np.random.default_rng(0)
    for _ in range(n_mc):
        z = z_t
        dwell_remaining = _stationary_residual_dwell(rng, float(dwell_mean[z]))
        for _h in range(horizon):
            if dwell_remaining > 1:
                dwell_remaining -= 1
                continue
            row = pi_off_diag[z].copy()
            row = row / row.sum()
            z = int(rng.choice(R, p=row))
            dwell_remaining = _shifted_poisson_dwell_from_mean(rng, float(dwell_mean[z]))
        counts[z] += 1
    return counts / counts.sum()


def _shifted_poisson_dwell_from_mean(rng: np.random.Generator, dwell_mean: float) -> int:
    """Sample D = 1 + Poisson(lambda) from its mean mu = lambda + 1."""
    if dwell_mean < 1.0:
        raise ValueError("shifted-Poisson dwell mean must be >= 1")
    return int(rng.poisson(dwell_mean - 1.0) + 1)


def _stationary_residual_dwell(rng: np.random.Generator, dwell_mean: float) -> int:
    """Stationary renewal residual for shifted-Poisson dwell.

    Matches m01_forward.stan generated quantities:
    D* ~ 1/mu · SP(lambda) + lambda/mu · (SP(lambda)+1),
    residual ~ Uniform{1, ..., D*}, where mu = lambda + 1.
    """
    if dwell_mean < 1.0:
        raise ValueError("shifted-Poisson dwell mean must be >= 1")
    p_base = 1.0 / dwell_mean
    d_star = _shifted_poisson_dwell_from_mean(rng, dwell_mean)
    if rng.random() >= p_base:
        d_star += 1
    return int(rng.integers(1, d_star + 1))


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


def _coerce_terminal_distribution(
    z_t_distribution: np.ndarray,
    n_draws: int,
    n_regimes: int,
) -> np.ndarray:
    z_t_distribution = np.asarray(z_t_distribution, dtype=float)
    if z_t_distribution.ndim == 1:
        if z_t_distribution.shape[0] != n_regimes:
            raise ValueError("z_t_distribution length must match regime count")
        z_t_draws = np.broadcast_to(z_t_distribution, (n_draws, n_regimes))
    elif z_t_distribution.ndim == 2:
        if z_t_distribution.shape[1] != n_regimes:
            raise ValueError("z_t_distribution second dimension must match regime count")
        if z_t_distribution.shape[0] == 1 and n_draws > 1:
            z_t_draws = np.broadcast_to(z_t_distribution, (n_draws, n_regimes))
        elif z_t_distribution.shape[0] != n_draws:
            raise ValueError("z_t_distribution draw dimension must match posterior draws")
        else:
            z_t_draws = z_t_distribution
    else:
        raise ValueError("z_t_distribution must have shape (R,) or (D, R)")
    return z_t_draws / (z_t_draws.sum(axis=1, keepdims=True) + 1e-12)


def _coerce_z_future_draws(
    z_future_draws: np.ndarray,
    n_draws: int,
    n_regimes: int,
    h_max: int,
) -> np.ndarray:
    z_future = np.asarray(z_future_draws, dtype=int)
    if z_future.ndim != 2:
        raise ValueError("z_future_draws must have shape (draws, horizons)")
    if z_future.shape[0] != n_draws:
        raise ValueError("z_future_draws draw dimension must match posterior draws")
    if z_future.shape[1] < h_max:
        raise ValueError("z_future_draws does not cover h_max")
    z_future = z_future[:, :h_max].copy()
    if z_future.min(initial=1) >= 1:
        z_future -= 1
    if z_future.min(initial=0) < 0 or z_future.max(initial=0) >= n_regimes:
        raise ValueError("z_future_draws regime index out of bounds")
    return z_future


def _drift_share(forecast: np.ndarray, pi_inf: np.ndarray, z_present: np.ndarray) -> float:
    forecast = forecast / (forecast.sum() + 1e-12)
    pi_inf = pi_inf / (pi_inf.sum() + 1e-12)
    z_present = z_present / (z_present.sum() + 1e-12)
    tv_to_stationary = 0.5 * np.abs(forecast - pi_inf).sum()
    tv_to_present = 0.5 * np.abs(forecast - z_present).sum()
    denom = tv_to_stationary + tv_to_present + 1e-12
    return float(1.0 - tv_to_stationary / denom)


def horizon_validity_from_z_future_draws(
    cell_id: str,
    pi_off_diag_draws: np.ndarray,
    dwell_mean_draws: np.ndarray,
    z_t_distribution: np.ndarray,
    z_future_draws: np.ndarray,
    h_max: int = 18,
    tau: float = HORIZON_PRIOR_DRIFT_TAU,
    n_bootstrap: int = 200,
    seed: int | None = 0,
) -> HorizonValidity:
    """Horizon diagnostic computed from Stan posterior-predictive z_future draws.

    `z_future_draws` is the generated-quantities regime path for one series,
    shape (posterior_draws, H). Values may be Stan's 1-indexed regimes or
    Python's 0-indexed regimes.
    """
    if pi_off_diag_draws.ndim == 2:
        pi_off_diag_draws = pi_off_diag_draws[np.newaxis, :, :]
        dwell_mean_draws = dwell_mean_draws[np.newaxis, :]

    n_draws, n_regimes, _ = pi_off_diag_draws.shape
    if dwell_mean_draws.shape != (n_draws, n_regimes):
        raise ValueError("dwell_mean_draws must have shape (draws, regimes)")
    z_t_draws = _coerce_terminal_distribution(z_t_distribution, n_draws, n_regimes)
    z_future = _coerce_z_future_draws(z_future_draws, n_draws, n_regimes, h_max)

    pi_inf_draws = np.array([
        stationary_marginal_time(pi_off_diag_draws[i], dwell_mean_draws[i])
        for i in range(n_draws)
    ])

    def curve_for_indices(indices: np.ndarray) -> list[float]:
        pi_inf = pi_inf_draws[indices].mean(axis=0)
        z_present = z_t_draws[indices].mean(axis=0)
        curve = []
        for h_idx in range(h_max):
            counts = np.bincount(z_future[indices, h_idx], minlength=n_regimes).astype(float)
            forecast = counts / max(1, len(indices))
            curve.append(_drift_share(forecast, pi_inf, z_present))
        return curve

    base_indices = np.arange(n_draws)
    curves = [curve_for_indices(base_indices)]
    if n_draws > 1 and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_bootstrap):
            curves.append(curve_for_indices(rng.choice(base_indices, size=n_draws, replace=True)))

    drift_array = np.asarray(curves, dtype=float)
    median_drift = np.median(drift_array, axis=0)
    q05_drift = np.percentile(drift_array, 5, axis=0)
    q95_drift = np.percentile(drift_array, 95, axis=0)

    curve_median = [(h + 1, float(v)) for h, v in enumerate(median_drift)]
    curve_q05 = [(h + 1, float(v)) for h, v in enumerate(q05_drift)]
    curve_q95 = [(h + 1, float(v)) for h, v in enumerate(q95_drift)]

    h_star = first_crossing_h_star(curve_median, tau=tau, h_max=h_max)
    h_star_q05 = first_crossing_h_star(curve_q95, tau=tau, h_max=h_max)
    h_star_q95 = first_crossing_h_star(curve_q05, tau=tau, h_max=h_max)

    return HorizonValidity(
        cell_id=cell_id,
        h_star_months=h_star,
        h_star_q05=h_star_q05,
        h_star_q95=h_star_q95,
        tau=tau,
        prior_drift_share_at_h_max=float(median_drift[-1]),
        horizon_validity_curve=curve_median,
        horizon_validity_curve_q05=curve_q05,
        horizon_validity_curve_q95=curve_q95,
    )


def horizon_validity(
    cell_id: str,
    pi_off_diag_draws: np.ndarray,
    dwell_mean_draws: np.ndarray,
    z_t_distribution: np.ndarray,    # posterior of Z_T at present: (R,) or (D, R)
    h_max: int = 18,
    tau: float = HORIZON_PRIOR_DRIFT_TAU,
    n_mc: int = 1500,
    seed: int | None = 0,
) -> HorizonValidity:
    """Horizon diagnostic h* as the first crossing of the drift threshold."""
    # Support single point-estimates for testing
    if pi_off_diag_draws.ndim == 2:
        pi_off_diag_draws = pi_off_diag_draws[np.newaxis, :, :]
        dwell_mean_draws = dwell_mean_draws[np.newaxis, :]

    N_draws, R, _ = pi_off_diag_draws.shape
    z_t_draws = _coerce_terminal_distribution(z_t_distribution, N_draws, R)

    rng = np.random.default_rng(seed)
    curves_per_draw = []

    for i in range(N_draws):
        pi_off_diag = pi_off_diag_draws[i]
        dwell_mean = dwell_mean_draws[i]
        z_dist = z_t_draws[i]
        pi_inf = stationary_marginal_time(pi_off_diag, dwell_mean)

        curve = []
        for h in range(1, h_max + 1):
            # marginalize over z_t under its posterior distribution
            forecast = np.zeros(R)
            for z, p in enumerate(z_dist):
                if p < 1e-9:
                    continue
                forecast += p * forecast_regime_distribution(
                    pi_off_diag, dwell_mean, z_t=z, horizon=h, n_mc=n_mc, rng=rng
                )
            forecast = forecast / forecast.sum()
            # drift share: KL(forecast || pi_inf) decreases as forecast nears
            # stationary; equivalently, total variation distance to pi_inf
            tv_to_stationary = 0.5 * np.abs(forecast - pi_inf).sum()
            tv_to_present = 0.5 * np.abs(forecast - z_dist).sum()
            denom = tv_to_stationary + tv_to_present + 1e-12
            drift_share = 1.0 - tv_to_stationary / denom
            curve.append(float(drift_share))
        curves_per_draw.append(curve)

    drift_array = np.array(curves_per_draw) # shape (N_draws, h_max)
    
    median_drift = np.median(drift_array, axis=0)
    q05_drift = np.percentile(drift_array, 5, axis=0)
    q95_drift = np.percentile(drift_array, 95, axis=0)
    
    curve_median = [(h+1, float(v)) for h, v in enumerate(median_drift)]
    curve_q05 = [(h+1, float(v)) for h, v in enumerate(q05_drift)]
    curve_q95 = [(h+1, float(v)) for h, v in enumerate(q95_drift)]
    
    h_star = first_crossing_h_star(curve_median, tau=tau, h_max=h_max)
    # Higher drift means earlier crossing of tau -> smaller h_star
    h_star_q05 = first_crossing_h_star(curve_q95, tau=tau, h_max=h_max) 
    h_star_q95 = first_crossing_h_star(curve_q05, tau=tau, h_max=h_max)
    
    return HorizonValidity(
        cell_id=cell_id,
        h_star_months=h_star,
        h_star_q05=h_star_q05,
        h_star_q95=h_star_q95,
        tau=tau,
        prior_drift_share_at_h_max=float(median_drift[-1]),
        horizon_validity_curve=curve_median,
        horizon_validity_curve_q05=curve_q05,
        horizon_validity_curve_q95=curve_q95,
    )
