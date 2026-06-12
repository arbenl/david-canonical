"""Theorem B' channel informativeness I(O) and information scaling N * I^2.

B'.1 (classification):  Bayes error for recovering A from a single Y is
                        (1 - I) / 2 where I = |rho(O) - delta(O)|.

B'.2 (estimation):      Effective sample size for estimating phi from N
                        replicates scales as N * I^2. Cells with low I(O)
                        are prior-dominated.

The cell-level gate is on the lower 95% credible bound of I(O). If it falls
below the floor, the cell is flagged prior_dominated and routed accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import INFORMATIVENESS_FLOOR_LOWER_95, N_EFF_I2_FLOOR


@dataclass(frozen=True)
class InformativenessResult:
    cell_id: str
    posterior_median_I: float
    lower_95_I: float
    upper_95_I: float
    upper_95_I: float
    n_eff_adjusted: float        # N_adjusted * I^2 (per B'.2 with Godambe correction)
    gate_status: str             # "pass" | "fail"
    reason: str


def informativeness_draws(
    rho_draws: np.ndarray,         # (D,)
    delta_draws: np.ndarray,       # (D,)
) -> np.ndarray:
    """Per-draw I(O) = |rho - delta|. Returns shape (D,)."""
    if rho_draws.shape != delta_draws.shape:
        raise ValueError("rho and delta draws must share shape")
    return np.abs(rho_draws - delta_draws)


def bayes_classification_error(I: float) -> float:
    """B'.1 Bayes error rate under equal prior odds."""
    if not 0.0 <= I <= 1.0:
        raise ValueError("I must be in [0, 1]")
    return 0.5 * (1.0 - I)


def effective_sample_size(n_replicates: int, I: float) -> float:
    """B'.2 effective N for estimating phi from N detections at informativeness I."""
    return float(n_replicates) * (I ** 2)


def dependence_adjusted_n_eff(
    N: int,
    I: float,
    phi: float,
    p: float,
    corr_A_h: np.ndarray
) -> float:
    """Godambe dependence adjustment for sample size under serial correlation.
    
    N_eff = N * gamma_0 / (gamma_0 + 2 * sum(gamma_h))
    gamma_h / gamma_0 = c * Corr(A_0, A_h)
    c = I**2 * phi * (1 - phi) / (p * (1 - p))
    """
    if N <= 1 or len(corr_A_h) == 0:
        return float(N)
        
    c = (I**2) * phi * (1.0 - phi) / (p * (1.0 - p) + 1e-12)
    sum_gamma_ratio = c * np.sum(corr_A_h)
    
    denom = 1.0 + 2.0 * sum_gamma_ratio
    if denom <= 0.0:
        return float(N)
        
    n_eff = float(N) / denom
    # Cap at N if negative correlation inflates N_eff
    return float(min(N, n_eff))


def compute_activity_autocorrelation(
    Pi_off_diag: np.ndarray,
    dwell_mean: np.ndarray,
    phi_k: np.ndarray,
    max_lag: int,
    n_mc: int = 500,
) -> np.ndarray:
    """Compute Corr(A_0, A_h) from renewal functional."""
    from .D_forecast_horizon import stationary_marginal_time, forecast_regime_distribution
    pi_inf = stationary_marginal_time(Pi_off_diag, dwell_mean)
    phi = np.dot(pi_inf, phi_k)
    var_A = phi * (1.0 - phi)
    
    if var_A < 1e-9:
        return np.zeros(max_lag)
        
    corr = np.zeros(max_lag)
    for h in range(1, max_lag + 1):
        cov_A = 0.0
        for z0 in range(len(pi_inf)):
            if pi_inf[z0] < 1e-9:
                continue
            p_zh = forecast_regime_distribution(
                Pi_off_diag, dwell_mean, z_t=z0, horizon=h, n_mc=n_mc
            )
            p_zh = p_zh / (p_zh.sum() + 1e-12)
            E_Ah_given_z0 = np.dot(p_zh, phi_k)
            cov_A += pi_inf[z0] * phi_k[z0] * E_Ah_given_z0
        cov_A -= phi**2
        corr[h-1] = cov_A / var_A
        
    return corr


def check_cell(
    cell_id: str,
    rho_draws: np.ndarray,
    delta_draws: np.ndarray,
    n_replicates: int,
    phi: float = 0.5,
    p: float = 0.5,
    corr_A_h: np.ndarray | None = None,
    floor_lower_95: float = INFORMATIVENESS_FLOOR_LOWER_95,
    n_eff_i2_floor: float = N_EFF_I2_FLOOR,
) -> InformativenessResult:
    I = informativeness_draws(rho_draws, delta_draws)
    median = float(np.median(I))
    lower, upper = (float(x) for x in np.quantile(I, [0.025, 0.975]))
    
    if corr_A_h is not None:
        n_adj = dependence_adjusted_n_eff(n_replicates, median, phi, p, corr_A_h)
    else:
        n_adj = float(n_replicates)
        
    n_eff_i2 = effective_sample_size(int(round(n_adj)), median)
    
    # Gate 1: I lower-95 floor (weak-signal guard)
    if lower < floor_lower_95:
        return InformativenessResult(
            cell_id=cell_id,
            posterior_median_I=median,
            lower_95_I=lower,
            upper_95_I=upper,
            n_eff_adjusted=n_eff_i2,
            gate_status="fail",
            reason=f"I_lower_95_{lower:.4f}_below_floor_{floor_lower_95:.4f}",
        )
    # Gate 2: N_eff × I² floor — rules out I=0.11 with N=5 (reviewer §5)
    if n_eff_i2 < n_eff_i2_floor:
        return InformativenessResult(
            cell_id=cell_id,
            posterior_median_I=median,
            lower_95_I=lower,
            upper_95_I=upper,
            n_eff_adjusted=n_eff_i2,
            gate_status="fail",
            reason=f"N_eff_I2_{n_eff_i2:.2f}_below_floor_{n_eff_i2_floor:.2f}",
        )
    return InformativenessResult(
        cell_id=cell_id,
        posterior_median_I=median,
        lower_95_I=lower,
        upper_95_I=upper,
        n_eff_adjusted=n_eff_i2,
        gate_status="pass",
        reason="I_lower_95_and_N_eff_I2_above_floor",
    )
