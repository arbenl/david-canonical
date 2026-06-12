import numpy as np

def dependence_adjusted_n_eff(
    n_replicates: int,
    I: float,
    phi: float,
    p: float,
    corr_A_h: np.ndarray
) -> float:
    """
    Godambe dependence adjustment for sample size.
    N_eff = N * gamma_0 / (gamma_0 + 2 * sum(gamma_h))
    gamma_h / gamma_0 = c * Corr(A_0, A_h)
    c = I**2 * phi * (1 - phi) / (p * (1 - p))
    """
    if n_replicates <= 1 or len(corr_A_h) == 0:
        return float(n_replicates)
        
    c = (I**2) * phi * (1.0 - phi) / (p * (1.0 - p))
    
    # Sum over h from 1 to T-1. 
    # corr_A_h[0] corresponds to h=1.
    # In a finite sample of size N, the number of lag-h pairs is (N - h).
    # Wait, does the sum use (1 - h/N) weights?
    # Standard Godambe / Newey-West variance sums (1 - h/N) * gamma_h.
    # Let's just sum gamma_h as specified, or with (1 - h/N) if we want finite sample.
    # The prompt says: "sum(gamma_h)" verbatim.
    
    # We will use exactly what is written:
    sum_gamma_ratio = c * np.sum(corr_A_h)
    
    denominator = 1.0 + 2.0 * sum_gamma_ratio
    
    if denominator <= 0.0:
        # If negative correlation is extremely strong, denominator could be <= 0.
        # But we cap at min(N, N_eff) anyway. If denom <= 0, the variance is theoretically 0,
        # so N_eff = infinity. We return n_replicates.
        return float(n_replicates)
        
    n_eff = n_replicates / denominator
    return float(min(n_replicates, n_eff))

print(dependence_adjusted_n_eff(100, 0.5, 0.5, 0.5, np.array([0.0, 0.0])))
print(dependence_adjusted_n_eff(100, 0.5, 0.5, 0.5, np.array([0.5, 0.25])))
print(dependence_adjusted_n_eff(100, 0.5, 0.5, 0.5, np.array([-0.5, -0.25])))
