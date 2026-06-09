"""Seed a faithful fit fixture for live frontend lock-state verification.

This does NOT run MCMC (Stan compilation is out of scope for a UI-lock check).
It produces a fit_summary.json structurally identical to what david.model.fit.run_fit
emits, plus DB rows via the REAL repository writers, so the live API endpoints
(/fit-runs, /fit-runs/{id}, /forecast-cells, /strata) return data exactly as in
production.

Usage:
    DATABASE_URL=postgresql://david:david@localhost:5544/david \
    python scripts/seed_verify.py <scenario>

scenarios:
    a_fail     Theorem A' gate_status=fail  (curve lock on xk_general)
    certified  All theorem gates pass       (certified badge + live curve)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from david.config import FITS_DIR, MODEL_VERSION, ID_DISTANCE_FLOOR, FORECAST_HORIZONS_MONTHS, INFORMATIVENESS_FLOOR_LOWER_95
from david.db.repositories import write_fit_run, write_forecast_cells, upsert_stratum

scenario = sys.argv[1] if len(sys.argv) > 1 else "a_fail"
a_pass = scenario == "certified"

run_id = f"fit_verify_{scenario}"
ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# Strata (so /strata returns rows; xk_general is the frontend ACTIVE_STRATUM)
for sid, country, policy in [
    ("xk_general", "Kosovo", "general"),
    ("al_general", "Albania", "general"),
    ("mk_general", "North Macedonia", "general"),
]:
    upsert_stratum(stratum_id=sid, country=country, policy=policy, observability=0.55)

# A' fail scenario: median d(theta) below the 0.05 floor (Kosovo has < 3 independent
# sources → practically non-identified). certified: comfortably above floor.
med_d = 0.082 if a_pass else 0.012
a_status = "pass" if med_d >= ID_DISTANCE_FLOOR else "fail"

h_star = 6  # horizons 9, 12 exceed h* → horizon_prior_dominated (Theorem D-forecast)

theorems = {
    "A_prime": {
        "theorem": "A_prime",
        "median_d_theta": med_d,
        "q05_d_theta": round(med_d * 0.6, 4),
        "floor": ID_DISTANCE_FLOOR,
        "gate_status": a_status,
        "reason": ("d_theta_above_floor" if a_status == "pass"
                   else f"d_theta_median_{med_d:.4f}_below_floor_{ID_DISTANCE_FLOOR}"),
    },
    "B_prime": {
        "theorem": "B_prime",
        "median_I_worst_source": 0.187,
        "lower_95_I_worst_source": 0.121,
        "floor": INFORMATIVENESS_FLOOR_LOWER_95,
        "gate_status": "pass",
        "reason": "informativeness_above_floor",
    },
    "C_prime": {
        "theorem": "C_prime",
        "n_cells": 24, "n_flagged": 5, "threshold_p": 0.62,
        "posterior_expected_fdp": 0.083, "q_target": 0.10,
        "gate_status": "pass", "reason": "flagged_5_of_24_cells_at_q=0.1",
    },
    "D_prime": {
        "theorem": "D_prime",
        "h_star_months": h_star,
        "h_stars_per_regime": [5, 6, 7],
        "tau": 0.50,
        "prior_drift_share_at_h_max": 0.61,
        "forecast_horizons": list(FORECAST_HORIZONS_MONTHS),
        "gate_status": "pass",
        "reason": f"h_star={h_star}_months",
    },
}

all_failed = [k for k, v in theorems.items() if v["gate_status"] in ("fail", "error")]

fit_summary = {
    "model_version": MODEL_VERSION,
    "run_id": run_id,
    "rhat_max": 1.004,
    "ess_bulk_min": 2310.0,
    "ess_tail_min": 1980.0,
    "divergences": 0,
    "theorems": theorems,
    "gates": {
        "F1": {"gate_status": "pass", "coverage_rate": 0.86, "n_worlds": 200,
               "reason": "prior_predictive_in_band"},
        "F3": {"gate_status": "skip", "reason": "no_held_out_labels"},
        "F4": {"gate_status": "skip", "reason": "activity_truth_unknown_in_real_data_mode"},
        "F5": {"gate_status": "skip", "reason": "no_held_out_set"},
    },
    "gate_status": "fail" if all_failed else "pass",
    "reason": "fit_pass" if not all_failed else "; ".join(f"theorem_{k}" for k in all_failed),
    "timestamp": ts,
}

# Write fit_summary.json (file-based endpoint /fit-runs/{id} reads this)
fit_dir = FITS_DIR / run_id
fit_dir.mkdir(parents=True, exist_ok=True)
(fit_dir / "fit_summary.json").write_text(json.dumps(fit_summary, indent=2))

# DB row (real writer) — /fit-runs reads this
write_fit_run({**fit_summary, "n_strata": 3, "n_labels": 142})

# Forecast cells for xk_general across all horizons; 3 tactics (SIO/MIO/CSIO).
# Horizons > h_star are flagged horizon_prior_dominated (Theorem D-forecast).
cells = []
base_p = {1: 0.714, 2: 0.523, 3: 0.392}
for tactic in (1, 2, 3):
    for h in FORECAST_HORIZONS_MONTHS:
        below = h <= h_star
        # uncertainty widens with horizon; prior-dominated cells collapse toward marginal
        spread = 0.05 + 0.012 * h
        p = base_p[tactic] if below else 0.5 + (base_p[tactic] - 0.5) * 0.25
        cells.append({
            "cell": {"stratum_id": "xk_general", "series": 1, "tactic": tactic,
                     "country": "Kosovo", "policy": "general", "tactic_label": f"tactic_{tactic}"},
            "horizon_months": h,
            "p_active": round(p, 4),
            "credible_interval_80": [round(max(0.01, p - spread), 4), round(min(0.99, p + spread), 4)],
            "credible_interval_95": [round(max(0.01, p - 1.7 * spread), 4), round(min(0.99, p + 1.7 * spread), 4)],
            "horizon_validity": {
                "h_star_months": h_star,
                "below_h_star": below,
                "forecast_route": "conditional_forecast" if below else "horizon_prior_dominated",
            },
            "model_version": MODEL_VERSION,
        })

n = write_forecast_cells(cells, run_id)
print(json.dumps({
    "scenario": scenario, "run_id": run_id,
    "A_prime_gate": a_status, "fit_gate_status": fit_summary["gate_status"],
    "h_star_months": h_star, "n_forecast_cells_written": n,
    "fit_summary_path": str(fit_dir / "fit_summary.json"),
}, indent=2))
