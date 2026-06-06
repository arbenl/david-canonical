"""cmdstanpy wrapper for m01_forward.stan with fit-contract gates.

The fit-contract gates mirror council_m01/simulation/full_integrated_fit_contract.py
(R-hat, ESS, divergences, posterior draws). Pass-through here; do not relax.

Output: data/fits/{run_id}/{fit_summary.json, draws.parquet, diagnostics.json}.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from cmdstanpy import CmdStanModel

from ..config import (
    ADJUDICATED_DIR, BULK_ESS_MIN, DIVERGENCES_ALLOWED, FITS_DIR,
    FORECAST_HORIZONS_MONTHS, MIN_CHAINS, MIN_POSTERIOR_DRAWS,
    M01_FORWARD_STAN, MODEL_VERSION, R_HAT_MAX, TAIL_ESS_MIN,
)

_COMPILED_MODEL: CmdStanModel | None = None


def _get_compiled_model() -> CmdStanModel:
    global _COMPILED_MODEL
    if _COMPILED_MODEL is None:
        _COMPILED_MODEL = CmdStanModel(stan_file=str(M01_FORWARD_STAN))
    return _COMPILED_MODEL


def assemble_fit_data_from_synthetic(world: Any, horizon: int) -> dict[str, Any]:
    """Convert a WorldDraw to the Stan data dict expected by m01_forward.stan.

    Shapes:
      U = R * T * K  (unit = (series, time, tactic) triple, 1-indexed)
      N_label = R * T * K * S * M
    H_forecast is clamped to >= 1 because the Stan data block requires lower=1.
    """
    R: int = world.selected.shape[0]
    T: int = world.selected.shape[1]
    K: int = world.y.shape[2]   # (R, T, K, S, M)
    S: int = world.y.shape[3]
    M: int = world.y.shape[4]
    L: int = world.theta["Pi"].shape[0]
    H_forecast: int = max(1, horizon)

    U = R * T * K
    unit_series: list[int] = []
    unit_time: list[int] = []
    unit_tactic: list[int] = []
    selected_flat: list[int] = []
    observability_flat: list[float] = []

    for r in range(R):
        for t in range(T):
            for k in range(K):
                unit_series.append(r + 1)
                unit_time.append(t + 1)
                unit_tactic.append(k + 1)
                selected_flat.append(int(world.selected[r, t, k]))
                observability_flat.append(float(world.observability[r, t]))

    label_unit: list[int] = []
    label_source: list[int] = []
    label_coder: list[int] = []
    y_labels: list[int] = []

    for r in range(R):
        for t in range(T):
            for k in range(K):
                u = r * T * K + t * K + k + 1  # 1-indexed
                for s in range(S):
                    for m_i in range(M):
                        label_unit.append(u)
                        label_source.append(s + 1)
                        label_coder.append(m_i + 1)
                        y_labels.append(int(world.y[r, t, k, s, m_i]))

    # Sort label arrays by (unit, source) then build the slice index.
    # The current construction loop already inserts in (unit, source) order
    # because it iterates r,t,k in order and s in order for each unit.
    label_start, label_len = _build_label_index(label_unit, label_source, U, S)

    return {
        "R": R, "T": T, "L": L, "K": K, "S": S, "M": M, "U": U,
        "unit_series": unit_series,
        "unit_time": unit_time,
        "unit_tactic": unit_tactic,
        "selected": selected_flat,
        "observability": observability_flat,
        "N_label": len(y_labels),
        "label_unit": label_unit,
        "label_source": label_source,
        "label_coder": label_coder,
        "y": y_labels,
        "label_start": label_start,
        "label_len": label_len,
        "delta_max": 0.30,
        "H_forecast": H_forecast,
    }


def _build_label_index(
    label_unit: list[int],
    label_source: list[int],
    U: int,
    S: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build sorted label slice arrays for the Stan sorted-index optimization.

    Returns (label_start, label_len) as U×S nested lists (1-based start index;
    0 means no labels for that (unit, source) pair).

    The label arrays passed in must already be ordered by (unit, source) —
    i.e. the same order that was used when building label_unit / label_source.
    This function computes the run-length encoding of that ordering.
    """
    # Initialize with 0 (no labels)
    start = [[0] * S for _ in range(U)]
    length = [[0] * S for _ in range(U)]
    n = len(label_unit)
    i = 0
    while i < n:
        u = label_unit[i] - 1   # 0-based
        s = label_source[i] - 1  # 0-based
        run_start = i + 1        # 1-based Stan index
        while i < n and label_unit[i] - 1 == u and label_source[i] - 1 == s:
            i += 1
        start[u][s] = run_start
        length[u][s] = i - (run_start - 1)
    return start, length


def _flatten_draws(name: str, arr: np.ndarray) -> dict[str, np.ndarray]:
    """Map (n_draws, *dims) posterior draws to flat {name[i]: (n_draws,)} dict."""
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    if flat.shape[1] == 1:
        return {name: flat[:, 0]}
    return {f"{name}[{i}]": flat[:, i] for i in range(flat.shape[1])}


def extract_theta_space_draws(fit: Any) -> dict[str, np.ndarray]:
    """Extract posterior draws mapped to the theta-space names used by sample_world.

    Parameters that don't have a direct Stan counterpart (rho, delta, rho_o,
    delta_o) are omitted; the SBC loop skips them via dict.get returning None.
    """
    out: dict[str, np.ndarray] = {}

    # Direct matches in parameters block
    for name in ("alpha_activity", "dwell_lambda",
                 "selection_alpha", "selection_observability", "selection_activity",
                 "delta_raw", "j_raw", "delta_observability", "j_observability"):
        arr = fit.stan_variable(name)
        out.update(_flatten_draws(name, arr))

    # Transformed parameters: kappa_plus / kappa_minus already constrained
    for name in ("kappa_plus", "kappa_minus"):
        arr = fit.stan_variable(name)
        out.update(_flatten_draws(name, arr))

    # init: softmax of init_raw  →  initial_probability in generated quantities
    init_raw = fit.stan_variable("init_raw")                      # (D, L)
    init_raw_shifted = init_raw - init_raw.max(axis=1, keepdims=True)
    exp_raw = np.exp(init_raw_shifted)
    init_draws = exp_raw / exp_raw.sum(axis=1, keepdims=True)     # (D, L)
    out.update(_flatten_draws("init", init_draws))

    # Pi: off-diagonal only — diagonal is identically 0 (pi_ii = 0 constraint).
    # Diagonal elements would give degenerate ranks (all-zero) in SBC; skip them.
    log_jump = fit.stan_variable("log_jump")                      # (D, L, L)
    D_draws, L_val, _ = log_jump.shape
    for i in range(L_val):
        for j in range(L_val):
            if i != j:
                flat_idx = i * L_val + j
                out[f"Pi[{flat_idx}]"] = np.exp(log_jump[:, i, j])

    return out


def assemble_fit_data(
    input_dir: Path | None = None,
    L: int | None = None,
    H_forecast: int | None = None,
) -> dict[str, Any]:
    """Construct the Stan data dict from adjudicated CSV files.

    Expected files in input_dir (default: data/adjudicated/):
      evidence_items.csv  — columns: evidence_id, stratum_g, source_id, evidence_date
      sources.csv         — columns: source_id (unique per source family / block)
      coder_labels.csv    — columns: evidence_id, coder_id, label, tactic_k
      strata.csv          — columns: stratum_g, I_O (observability score 0-1)

    L defaults to 3 (pre-registered candidate); override once F21 gate selects L.
    H_forecast defaults to FORECAST_HORIZONS_MONTHS[0] (first horizon).

    Raises FileNotFoundError if any required CSV is missing.
    """
    import csv as csv_mod

    input_dir = input_dir or ADJUDICATED_DIR
    L = L or 3
    H_forecast = H_forecast or FORECAST_HORIZONS_MONTHS[0]

    required = ["evidence_items.csv", "sources.csv", "coder_labels.csv", "strata.csv"]
    for fname in required:
        p = input_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Real-data fit requires {fname} in {input_dir}. "
                "Populate data/adjudicated/ before running `david fit`."
            )

    def _read_csv(fname: str) -> list[dict[str, str]]:
        with (input_dir / fname).open(newline="") as fh:
            return [{k: (v or "").strip() for k, v in row.items()}
                    for row in csv_mod.DictReader(fh)]

    evidence_rows = _read_csv("evidence_items.csv")
    source_rows = _read_csv("sources.csv")
    label_rows = _read_csv("coder_labels.csv")
    strata_rows = _read_csv("strata.csv")

    # Build index maps (1-based)
    def _index_map(values: list[str]) -> dict[str, int]:
        return {v: i for i, v in enumerate(sorted(set(values)), start=1)}

    stratum_to_series = _index_map([r["stratum_g"] for r in strata_rows])
    tactic_to_index = _index_map([r["tactic_k"] for r in label_rows if r.get("tactic_k")])
    source_to_index = _index_map([r["source_id"] for r in source_rows])
    coder_to_index = _index_map([r["coder_id"] for r in label_rows if r.get("coder_id")])
    stratum_obs = {r["stratum_g"]: float(r.get("I_O") or 0.0) for r in strata_rows}

    # Assign time index: month-offset within each stratum, sorted by evidence_date
    from collections import defaultdict
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in evidence_rows:
        by_stratum[row["stratum_g"]].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda r: r.get("evidence_date", ""))
    # Build time index within stratum (month: 1-based consecutive)
    evidence_time: dict[str, int] = {}
    for rows in by_stratum.values():
        # Group by date, assign consecutive month index
        seen_dates: dict[str, int] = {}
        t = 0
        for row in rows:
            d = row.get("evidence_date", "")
            if d not in seen_dates:
                t += 1
                seen_dates[d] = t
            evidence_time[row["evidence_id"]] = seen_dates[d]

    # Tactic per evidence (from coder_labels)
    evidence_tactic: dict[str, str] = {}
    for row in label_rows:
        eid = row["evidence_id"]
        tk = row.get("tactic_k", "")
        if tk and eid not in evidence_tactic:
            evidence_tactic[eid] = tk

    # Units: one per (stratum, time, tactic) combination — 1-indexed
    unit_key_to_idx: dict[tuple, int] = {}
    unit_series: list[int] = []
    unit_time_list: list[int] = []
    unit_tactic: list[int] = []
    selected_flat: list[int] = []
    observability_flat: list[float] = []

    sorted_evidence = sorted(
        evidence_rows,
        key=lambda r: (stratum_to_series[r["stratum_g"]], evidence_time.get(r["evidence_id"], 1), r["evidence_id"]),
    )
    evidence_to_unit: dict[str, int] = {}
    for row in sorted_evidence:
        sg = row["stratum_g"]
        eid = row["evidence_id"]
        tk = evidence_tactic.get(eid, "unknown")
        key = (stratum_to_series[sg], evidence_time.get(eid, 1), tactic_to_index.get(tk, 1))
        if key not in unit_key_to_idx:
            unit_key_to_idx[key] = len(unit_key_to_idx) + 1
            unit_series.append(key[0])
            unit_time_list.append(key[1])
            unit_tactic.append(key[2])
            selected_flat.append(1)   # observed-selected only (pilot assumption)
            observability_flat.append(stratum_obs.get(sg, 0.0))
        evidence_to_unit[eid] = unit_key_to_idx[key]

    U = len(unit_series)
    S = len(source_to_index)
    M = len(coder_to_index)
    K = len(tactic_to_index)
    R = len(stratum_to_series)
    T = max(unit_time_list) if unit_time_list else 1

    # Labels
    label_unit: list[int] = []
    label_source: list[int] = []
    label_coder: list[int] = []
    y_vals: list[int] = []
    for row in sorted(label_rows, key=lambda r: (evidence_to_unit.get(r["evidence_id"], 0), r.get("coder_id", ""))):
        eid = row.get("evidence_id", "")
        if eid not in evidence_to_unit:
            continue
        sid = source_to_index.get(row.get("source_id", ""), 1)
        cid = coder_to_index.get(row.get("coder_id", ""), 1)
        lbl = int(row.get("label", 0))
        label_unit.append(evidence_to_unit[eid])
        label_source.append(sid)
        label_coder.append(cid)
        y_vals.append(lbl)

    # Pre-sorted label slices for the Stan sorted-index optimization
    label_start, label_len = _build_label_index(label_unit, label_source, U, S)

    return {
        "R": R, "T": T, "L": L, "K": K, "S": S, "M": M, "U": U,
        "unit_series": unit_series,
        "unit_time": unit_time_list,
        "unit_tactic": unit_tactic,
        "selected": selected_flat,
        "observability": observability_flat,
        "N_label": len(y_vals),
        "label_unit": label_unit,
        "label_source": label_source,
        "label_coder": label_coder,
        "y": y_vals,
        "label_start": label_start,
        "label_len": label_len,
        "delta_max": 0.30,
        "H_forecast": max(1, H_forecast),
    }


def run_fit(run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or f"fit_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
    fit_dir = FITS_DIR / run_id
    fit_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = assemble_fit_data()
    except FileNotFoundError as e:
        return {
            "gate_status": "fail",
            "reason": "fit_data_missing",
            "detail": str(e),
            "fit_dir": str(fit_dir),
        }

    model = CmdStanModel(stan_file=str(M01_FORWARD_STAN))
    fit = model.sample(
        data=data,
        chains=MIN_CHAINS,
        iter_warmup=1000,
        iter_sampling=MIN_POSTERIOR_DRAWS // MIN_CHAINS,
        seed=42,
        adapt_delta=0.95,
        max_treedepth=12,
        output_dir=str(fit_dir / "cmdstan"),
    )

    summary = fit.summary()
    diagnostics = fit.diagnose()

    rhat_max = float(summary["R_hat"].dropna().max())
    bulk_ess_min = float(summary["ESS_bulk"].dropna().min())
    tail_ess_min = float(summary["ESS_tail"].dropna().min())
    divergences = int(sum(1 for s in diagnostics.split("\n") if "divergent" in s))

    failed: list[str] = []
    if rhat_max > R_HAT_MAX: failed.append(f"R_hat={rhat_max:.4f}>{R_HAT_MAX}")
    if bulk_ess_min < BULK_ESS_MIN: failed.append(f"ESS_bulk={bulk_ess_min:.0f}<{BULK_ESS_MIN}")
    if tail_ess_min < TAIL_ESS_MIN: failed.append(f"ESS_tail={tail_ess_min:.0f}<{TAIL_ESS_MIN}")
    if divergences > DIVERGENCES_ALLOWED: failed.append(f"divergences={divergences}>{DIVERGENCES_ALLOWED}")

    fit_summary = {
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "rhat_max": rhat_max,
        "ess_bulk_min": bulk_ess_min,
        "ess_tail_min": tail_ess_min,
        "divergences": divergences,
        "gates": {
            "F1": {"gate_status": "pending", "reason": "wire_prior_predictive"},
            "F3": {"gate_status": "pending", "reason": "wire_ece"},
            "F4": {"gate_status": "pending", "reason": "wire_auroc_auprc"},
            "F5": {"gate_status": "pending", "reason": "wire_baselines"},
        },
        "gate_status": "fail" if failed else "pass",
        "reason": "fit_pass" if not failed else "; ".join(failed),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    (fit_dir / "fit_summary.json").write_text(json.dumps(fit_summary, indent=2))
    fit.draws_pd().to_parquet(fit_dir / "draws.parquet")

    return {**fit_summary, "fit_dir": str(fit_dir)}
