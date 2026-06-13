"""Forecast routing: FG1..FG6 + posterior expected FDP threshold.

Applies the gate stack to each forecast cell and writes route_ledger.json.

Order of operations:
  FG1  F1 passed and falsification battery ledger passed for this run
  FG2  Theorem A' identification distance >= floor
  FG3  Theorem B' I(O) lower 95% CI >= floor
  FG4  Forecast SBC F14 passed for current model version
  FG5  Horizon h <= h*(g)
  FG6  Endogenous-observability lambda interval width <= max
Then apply posterior expected FDP threshold across cells that passed FG1..FG6.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    FORECASTS_DIR, FITS_DIR, POSTERIOR_FDP_DEFAULT_Q,
    FDP_EXCEEDANCE_GAMMA, FDP_EXCEEDANCE_ALPHA, FDP_MCSE_MARGIN_RATIO,
    ID_DISTANCE_FLOOR, INFORMATIVENESS_FLOOR_LOWER_95,
    N_EFF_I2_FLOOR, LAMBDA_ENDOG_INTERVAL_MAX_WIDTH,
)
from ..theorems.C_renamed import (
    compute_posterior_fdp_threshold,
    compute_fdp_exceedance_gate,
    compute_posterior_fdp_envelope,
    check_mcse_floor,
)


ROUTES = (
    "headline",
    "monitor_only",
    "aggregate_only",
    "evidence_gap",
    "withhold",
    "horizon_prior_dominated",
    "prior_dominated",
)


def _finite_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def _source_independence_pass(cell: dict[str, Any]) -> tuple[bool, str | None]:
    summary = cell.get("source_independence")
    if summary is None:
        return False, "FG2_source_independence_missing"
    if not isinstance(summary, dict):
        return False, "FG2_source_independence_malformed"
    if summary.get("gate_status") == "pass":
        return True, None
    reason = str(summary.get("reason") or "fail")
    return False, f"FG2_structural_source_independence:{reason}"


def latest_forecast_dir() -> Path:
    """Return the forecast dir matching the most recent passing fit."""
    if not FORECASTS_DIR.exists():
        raise FileNotFoundError("no forecasts to route; run `david forecast` first")
    candidates = [p for p in FORECASTS_DIR.iterdir() if p.is_dir()
                  and list(p.glob("cells_h*.json"))]
    if not candidates:
        raise FileNotFoundError("no forecasts to route; run `david forecast` first")

    def _ts(p: Path) -> str:
        # Prefer timestamp from the matching fit_summary.json
        fit_summary = FITS_DIR / p.name / "fit_summary.json"
        if fit_summary.exists():
            try:
                return json.loads(fit_summary.read_text()).get("timestamp", "")
            except Exception:
                pass
        return p.name

    return max(candidates, key=_ts)


def latest_fit_dir() -> Path:
    """Return the most recent passing fit directory, sorted by fit_summary timestamp."""
    if not FITS_DIR.exists():
        raise FileNotFoundError("no fit found")

    def _ts(p: Path) -> str:
        try:
            return json.loads((p / "fit_summary.json").read_text()).get("timestamp", "")
        except Exception:
            return ""

    candidates = [
        p for p in FITS_DIR.iterdir()
        if p.is_dir() and (p / "fit_summary.json").exists()
        and json.loads((p / "fit_summary.json").read_text()).get("gate_status") == "pass"
    ]
    if not candidates:
        raise FileNotFoundError("no passing fit found; run `david fit` first")
    return max(candidates, key=_ts)


def _falsification_ledger_paths(
    forecast_dir: Path | None,
    fit_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    if forecast_dir is not None:
        paths.append(forecast_dir / "falsification_ledger.json")
    paths.append(fit_dir / "falsification_ledger.json")
    return list(dict.fromkeys(paths))


def _falsification_battery_pass(
    forecast_dir: Path | None,
    fit_dir: Path,
) -> tuple[bool, list[str]]:
    """Return whether the current run has a passing falsification ledger."""
    for ledger_path in _falsification_ledger_paths(forecast_dir, fit_dir):
        if not ledger_path.exists():
            continue
        try:
            ledger = json.loads(ledger_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False, [f"falsification_ledger_unreadable:{ledger_path.name}"]
        battery = ledger.get("battery_result")
        if not isinstance(battery, dict):
            return False, [f"falsification_battery_missing:{ledger_path.name}"]
        if battery.get("gate_status") == "pass":
            return True, []
        failed = battery.get("failed_tests") or []
        if failed:
            return False, [f"falsification_failed:{','.join(map(str, failed))}"]
        return False, [f"falsification_failed:{battery.get('reason', 'unknown')}"]
    return False, ["falsification_ledger_missing"]


def _measurement_gates_pass(forecast_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Confirm F1 and the falsification battery passed for the current run."""
    fit_dir = latest_fit_dir()
    summary_path = fit_dir / "fit_summary.json"
    if not summary_path.exists():
        return False, ["fit_summary_missing"]
    data = json.loads(summary_path.read_text())
    failed = []
    if data.get("gates", {}).get("F1", {}).get("gate_status") != "pass":
        failed.append("F1")
    falsification_ok, falsification_failed = _falsification_battery_pass(
        forecast_dir,
        fit_dir,
    )
    if not falsification_ok:
        failed.extend(falsification_failed)
    return (not failed), failed


def _forecast_sbc_pass() -> tuple[bool, str]:
    summary = FITS_DIR / "forecast_sbc" / "forecast_sbc_summary.json"
    if not summary.exists():
        return False, "forecast_sbc_summary_missing"
    data = json.loads(summary.read_text())
    return data.get("gate_status") == "pass", data.get("reason", "")


def apply_forecast_routing(
    q: float = POSTERIOR_FDP_DEFAULT_Q,
    gamma: float = FDP_EXCEEDANCE_GAMMA,
    alpha: float = FDP_EXCEEDANCE_ALPHA,
    horizon_files_glob: str = "cells_h*.json",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    forecast_dir = latest_forecast_dir()
    out_dir = out_dir or forecast_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meas_ok, meas_failed = _measurement_gates_pass(forecast_dir)
    sbc_ok, sbc_reason = _forecast_sbc_pass()

    route_counts: dict[str, int] = {r: 0 for r in ROUTES}
    routed_cells = []

    for cells_path in sorted(forecast_dir.glob(horizon_files_glob)):
        records = json.loads(cells_path.read_text())
        for cell in records:
            reasons: list[str] = []
            route = "headline"

            # FG1
            if not meas_ok:
                route = "withhold"; reasons.append(f"FG1_fail:{','.join(meas_failed)}")

            # FG2 identification distance
            if route == "headline":
                source_ok, source_reason = _source_independence_pass(cell)
                if not source_ok:
                    route = "evidence_gap"
                    reasons.append(source_reason or "FG2_source_independence_fail")
                elif cell["identification_distance_posterior_median"] < ID_DISTANCE_FLOOR:
                    route = "evidence_gap"; reasons.append("FG2_d_theta_below_floor")

            # FG3 informativeness
            if route == "headline":
                I_lower95 = _finite_float(cell.get("informativeness_I_O_lower_95"))
                n_eff_i2 = _finite_float(cell.get("informativeness_n_eff_i2"))
                if I_lower95 is None:
                    route = "prior_dominated"
                    reasons.append("FG3_I_O_lower95_missing")
                elif I_lower95 < INFORMATIVENESS_FLOOR_LOWER_95:
                    route = "prior_dominated"
                    reasons.append("FG3_I_O_lower95_below_floor")
                elif n_eff_i2 is None:
                    route = "prior_dominated"
                    reasons.append("FG3_N_eff_I2_missing")
                elif n_eff_i2 < N_EFF_I2_FLOOR:
                    route = "prior_dominated"
                    reasons.append("FG3_N_eff_I2_below_floor")

            # FG4 forecast SBC
            if not sbc_ok and route == "headline":
                route = "withhold"; reasons.append(f"FG4_forecast_sbc:{sbc_reason}")

            # FG5 horizon validity
            if not cell["horizon_validity"]["below_h_star"] and route == "headline":
                route = "horizon_prior_dominated"
                reasons.append("FG5_above_h_star")

            # P6: h* prior-sensitivity. If ±1 SD dwell-prior perturbation
            # changes this horizon's route, the cell is not eligible for a
            # headline claim even when nominal h* passes.
            sensitivity = (
                cell.get("horizon_prior_sensitivity")
                or cell.get("horizon_validity", {}).get("prior_sensitivity")
            )
            if route == "headline":
                if not isinstance(sensitivity, dict) or "route_changes_at_horizon" not in sensitivity:
                    route = "monitor_only"
                    reasons.append("FG5_h_star_prior_sensitivity_missing")
                elif sensitivity.get("route_changes_at_horizon") is True:
                    route = "monitor_only"
                    reasons.append("FG5_h_star_prior_sensitive")

            # FG6 endogenous observability
            lo, hi = cell["lambda_endogenous_bounds"]
            if (hi - lo) > LAMBDA_ENDOG_INTERVAL_MAX_WIDTH and route == "headline":
                route = "monitor_only"
                reasons.append(f"FG6_lambda_width_{hi-lo:.3f}_above_max")

            cell["forecast_route"] = route
            cell["route_reasons"] = reasons
            route_counts[route] = route_counts.get(route, 0) + 1
            routed_cells.append(cell)

    # Apply posterior-FDP threshold across "headline" cells only (expectation gate).
    # σ(D)-measurability: this path consumes ONLY marginals p_i — never joint draws.
    headline_cells = [c for c in routed_cells if c["forecast_route"] == "headline"]
    exceedance_summary: dict[str, Any]
    mcse_summary: dict[str, Any]
    envelope_summary: dict[str, Any]

    if headline_cells:
        p_hat = np.array([c["p_active"] for c in headline_cells])
        n_headline = len(headline_cells)
        draws_path = forecast_dir / "headline_draws.npy"

        # ------------------------------------------------------------------
        # C-6: MCSE/ESS floor — run before FDP threshold to keep the
        # expectation gate σ(D)-measurable (marginals only).  Cells failing
        # the MCSE floor are demoted to evidence_gap with a typed reason and
        # removed from the claim-eligible family.
        # ------------------------------------------------------------------
        mcse_excluded_set: set[int] = set()
        if draws_path.exists():
            joint_draws = np.load(str(draws_path))
            if joint_draws.ndim == 2 and joint_draws.shape[1] == n_headline:
                mcse_result = check_mcse_floor(joint_draws, q=q)
                mcse_excluded_set = set(mcse_result.excluded_indices)
                for idx in mcse_result.excluded_indices:
                    headline_cells[idx]["forecast_route"] = "evidence_gap"
                    headline_cells[idx]["route_reasons"] = [
                        mcse_result.binding_reason
                    ]
                    route_counts["headline"] = route_counts.get("headline", 0) - 1
                    route_counts["evidence_gap"] = route_counts.get("evidence_gap", 0) + 1
                mcse_summary = {
                    "n_cells_checked": mcse_result.n_cells_checked,
                    "n_cells_excluded": mcse_result.n_cells_excluded,
                    "excluded_indices": mcse_result.excluded_indices,
                    "mcse_threshold": mcse_result.mcse_threshold,
                    "bulk_ess": mcse_result.bulk_ess,
                    "mcse": mcse_result.mcse,
                    "binding_reason": mcse_result.binding_reason,
                }
            else:
                mcse_summary = {
                    "skipped": True,
                    "reason": "draws_shape_mismatch",
                    "expected_cols": n_headline,
                    "got_shape": list(joint_draws.shape),
                }
        else:
            mcse_summary = {"skipped": True, "reason": "no_joint_draws_file"}

        # Rebuild claim-eligible family after MCSE exclusions.
        eligible_cells = [
            (i, c) for i, c in enumerate(headline_cells)
            if i not in mcse_excluded_set
        ]
        eligible_indices = [i for i, _ in eligible_cells]
        p_hat_eligible = np.array([c["p_active"] for _, c in eligible_cells])

        if eligible_cells:
            fdp_result = compute_posterior_fdp_threshold(p_hat_eligible, q=q)
            flagged_by_exp = set(fdp_result.flagged_indices)
            # Map back to original headline_cells indices.
            for rank, (orig_i, c) in enumerate(eligible_cells):
                c["headline_flagged_by_posterior_fdp"] = rank in flagged_by_exp
            fdp_summary: dict[str, Any] = {
                **fdp_result.__dict__,
                # C-7: explicit aliases for the thesis field names.
                "t_star_q": fdp_result.threshold_p,
                "m_star": fdp_result.n_flagged,
            }

            # ------------------------------------------------------------------
            # C-3 sensitivity envelope (optional) — activated when the forecast
            # pipeline provides per-θ marginals in headline_theta_grid.npy.
            # Shape must be (n_theta, n_eligible).
            # ------------------------------------------------------------------
            theta_grid_path = forecast_dir / "headline_theta_grid.npy"
            if theta_grid_path.exists():
                p_theta_grid = np.load(str(theta_grid_path))
                if (
                    p_theta_grid.ndim == 2
                    and p_theta_grid.shape[1] == len(eligible_cells)
                ):
                    env_result = compute_posterior_fdp_envelope(p_theta_grid, q=q)
                    envelope_summary = {
                        "worst_theta_index": env_result.worst_theta_index,
                        "n_flagged_envelope": env_result.n_flagged,
                        "threshold_p_envelope": env_result.threshold_p,
                        "posterior_expected_fdp_envelope": env_result.posterior_expected_fdp_at_threshold,
                    }
                else:
                    envelope_summary = {
                        "skipped": True,
                        "reason": "theta_grid_shape_mismatch",
                        "got_shape": list(p_theta_grid.shape),
                    }
            else:
                envelope_summary = {
                    "skipped": True,
                    "reason": "no_theta_grid_file",
                    "worst_theta_index": None,
                }

            # ------------------------------------------------------------------
            # C-4: FG6 exceedance gate — joint posterior draws, σ(D)-separate.
            # ------------------------------------------------------------------
            if draws_path.exists() and fdp_result.n_flagged > 0:
                joint_draws_eligible = np.load(str(draws_path))
                if (
                    joint_draws_eligible.ndim == 2
                    and joint_draws_eligible.shape[1] == n_headline
                ):
                    # Restrict to eligible columns, then sort by descending p_i.
                    draws_elig = joint_draws_eligible[:, eligible_indices]
                    sort_order = np.argsort(-p_hat_eligible)
                    exc_result = compute_fdp_exceedance_gate(
                        draws_elig[:, sort_order],
                        m_star=fdp_result.n_flagged,
                        gamma=gamma,
                        alpha=alpha,
                        q=q,
                    )
                    # Map accepted prefix back to original eligible indices.
                    flagged_by_exc_ranks = set(sort_order[: exc_result.m_accepted].tolist())
                    for rank, (orig_i, c) in enumerate(eligible_cells):
                        c["headline_flagged_by_exceedance_gate"] = rank in flagged_by_exc_ranks
                    exceedance_summary = {
                        **exc_result.__dict__,
                        # C-7: explicit alias for the key statistic.
                        "exceedance_fraction": exc_result.exceedance_fraction_at_accepted,
                        "exceedance_ucl": exc_result.exceedance_ucl_at_accepted,
                    }
                else:
                    exceedance_summary = {
                        "skipped": True,
                        "reason": "draws_shape_mismatch",
                        "expected_cols": n_headline,
                        "got_shape": list(joint_draws_eligible.shape),
                    }
            else:
                reason = (
                    "no_joint_draws_file"
                    if not draws_path.exists()
                    else "m_star_is_zero"
                )
                exceedance_summary = {"skipped": True, "reason": reason}

            # ------------------------------------------------------------------
            # C-7: per-cell fdp_binding_reason — no silent cells.
            # ------------------------------------------------------------------
            exc_available = not exceedance_summary.get("skipped", False)
            for rank, (orig_i, c) in enumerate(eligible_cells):
                by_exp = c.get("headline_flagged_by_posterior_fdp", False)
                by_exc = c.get("headline_flagged_by_exceedance_gate")
                if not by_exp:
                    c["fdp_binding_reason"] = "unflagged_below_fdp_threshold"
                elif exc_available and by_exc is False:
                    c["fdp_binding_reason"] = "unflagged_exceedance_shrinkage"
                elif exc_available and by_exc:
                    c["fdp_binding_reason"] = "flagged_fdp_expectation_and_exceedance"
                else:
                    c["fdp_binding_reason"] = "flagged_fdp_expectation_only"

        else:
            # All headline cells excluded by MCSE floor.
            fdp_summary = {
                "q_target": q, "n_cells": 0, "n_flagged": 0,
                "t_star_q": 1.0, "m_star": 0,
            }
            exceedance_summary = {"skipped": True, "reason": "no_eligible_cells_after_mcse"}
            envelope_summary = {"skipped": True, "reason": "no_eligible_cells_after_mcse"}

    else:
        fdp_summary = {
            "q_target": q, "n_cells": 0, "n_flagged": 0,
            "t_star_q": 1.0, "m_star": 0,
        }
        exceedance_summary = {"skipped": True, "reason": "no_headline_cells"}
        mcse_summary = {"skipped": True, "reason": "no_headline_cells"}
        envelope_summary = {"skipped": True, "reason": "no_headline_cells"}

    # m_claim_eligible: family AFTER MCSE exclusions (Theorem C pre-registered family M).
    m_claim_eligible = sum(
        1 for c in routed_cells if c.get("forecast_route") == "headline"
    )
    h_star_prior_sensitive_cells = [
        c.get("cell_id") or c.get("cell") or f"cell_{i}"
        for i, c in enumerate(routed_cells)
        if "FG5_h_star_prior_sensitive" in c.get("route_reasons", [])
    ]
    h_star_prior_sensitivity_missing_cells = [
        c.get("cell_id") or c.get("cell") or f"cell_{i}"
        for i, c in enumerate(routed_cells)
        if "FG5_h_star_prior_sensitivity_missing" in c.get("route_reasons", [])
    ]
    h_star_prior_sensitivity_summary = {
        "n_cells_prior_sensitive": len(h_star_prior_sensitive_cells),
        "n_cells_missing_sensitivity": len(h_star_prior_sensitivity_missing_cells),
        "prior_sensitive_cells": h_star_prior_sensitive_cells,
        "missing_sensitivity_cells": h_star_prior_sensitivity_missing_cells,
        "binding_reason": "FG5_h_star_prior_sensitive",
        "missing_binding_reason": "FG5_h_star_prior_sensitivity_missing",
    }
    ledger = {
        "route_counts": route_counts,
        # M: the claim-eligible family — cells that survived FG1–FG5 AND the
        # MCSE floor (C-6), eligible for the Theorem C posterior expected FDP
        # selection.
        "m_claim_eligible": m_claim_eligible,
        # C-7 FDP block: t*(q), m*, per-gate summaries.
        "posterior_fdp": fdp_summary,
        "exceedance_gate": exceedance_summary,
        "mcse_floor": mcse_summary,
        "sensitivity_envelope": envelope_summary,
        "h_star_prior_sensitivity": h_star_prior_sensitivity_summary,
        "n_cells_total": len(routed_cells),
        "cells": routed_cells,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    ledger_path = out_dir / "route_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2))
    return {
        "gate_status": "pass",
        "ledger_path": str(ledger_path),
        "route_counts": route_counts,
    }
