"""Proper scoring rules and calibration metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


def brier_score(p_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p_hat - y) ** 2))


def class_balanced_log_loss(p_hat: np.ndarray, y: np.ndarray) -> float:
    pos = float(y.sum())
    neg = float(y.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    weights = np.where(y == 1, 1.0 / pos, 1.0 / neg)
    weights = weights * (y.size / 2)
    return float(log_loss(y, np.clip(p_hat, 1e-9, 1 - 1e-9), sample_weight=weights))


def auroc(p_hat: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p_hat))


def auprc(p_hat: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p_hat))


def expected_calibration_error(p_hat: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    n = p_hat.shape[0]
    ece = 0.0
    for b in range(bins):
        mask = (p_hat >= edges[b]) & (p_hat < edges[b + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(p_hat[mask].mean() - y[mask].mean())
    return float(ece)


def murphy_decomposition(p_hat: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Reliability + Resolution + Uncertainty decomposition of Brier score."""
    n = p_hat.shape[0]
    o_bar = y.mean()
    uncertainty = o_bar * (1 - o_bar)
    reliability = 0.0
    resolution = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for b in range(bins):
        mask = (p_hat >= edges[b]) & (p_hat < edges[b + 1])
        if not mask.any():
            continue
        n_b = mask.sum()
        p_b = p_hat[mask].mean()
        y_b = y[mask].mean()
        reliability += (n_b / n) * (p_b - y_b) ** 2
        resolution += (n_b / n) * (y_b - o_bar) ** 2
    return {
        "brier": brier_score(p_hat, y),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_check": float(reliability - resolution + uncertainty),
    }
