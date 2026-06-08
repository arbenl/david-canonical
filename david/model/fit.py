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
    FORECAST_HORIZONS_MONTHS, ID_DISTANCE_FLOOR, INFORMATIVENESS_FLOOR_LOWER_95,
    MIN_CHAINS, MIN_POSTERIOR_DRAWS, M01_FORWARD_STAN, MODEL_VERSION,
    POSTERIOR_FDP_DEFAULT_Q, R_HAT_MAX, TAIL_ESS_MIN,
)

_COMPILED_MODEL: CmdStanModel | None = None


_CMDSTAN_VERSION = "2.39.0"
# Direct tarball URL — bypasses the GitHub API (60 req/hr anon rate limit).
_CMDSTAN_URL = (
    f"https://github.com/stan-dev/cmdstan/releases/download/"
    f"v{_CMDSTAN_VERSION}/cmdstan-{_CMDSTAN_VERSION}.tar.gz"
)


def _ensure_cmdstan() -> None:
    """Install CmdStan (and build tools if missing) on first use.

    Fast path : ~/.cmdstan/cmdstan-{version} already present — returns immediately.
    Slow path : apt-get update → install make/g++ → download tarball directly
                (no GitHub API) → extract → make build → set_cmdstan_path.
    """
    import os
    import subprocess
    import tarfile
    import urllib.request
    from pathlib import Path

    try:
        from cmdstanpy.utils.cmdstan import cmdstan_path
        cmdstan_path()   # raises ValueError if not installed
        return           # already present
    except ValueError:
        pass

    make_bin = _find_executable("make")
    if make_bin is None:
        print("[david] make not found — running apt-get update + install…", flush=True)
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(
            ["apt-get", "install", "-y", "make", "g++", "libstdc++-12-dev"],
            check=False,
        )
        make_bin = _find_executable("make")

    if make_bin is None:
        raise RuntimeError(
            "make not found after apt-get install. "
            "Ensure nixpacks.toml has aptPkgs=[\"make\",\"g++\"] "
            "or pre-install CmdStan in build.sh."
        )

    print(f"[david] Downloading CmdStan {_CMDSTAN_VERSION}…", flush=True)
    tarball = Path(f"/tmp/cmdstan-{_CMDSTAN_VERSION}.tar.gz")
    urllib.request.urlretrieve(_CMDSTAN_URL, tarball)

    install_parent = Path.home() / ".cmdstan"
    install_parent.mkdir(parents=True, exist_ok=True)
    install_dir = install_parent / f"cmdstan-{_CMDSTAN_VERSION}"

    print("[david] Extracting…", flush=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(install_parent)

    print(f"[david] Building CmdStan with {make_bin} (make build — ~8 min)…", flush=True)
    # Pass a broad PATH so g++ is also locatable by make sub-processes
    env = os.environ.copy()
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + env.get("PATH", "")
    subprocess.run([make_bin, "build"], cwd=str(install_dir), env=env, check=True)

    import cmdstanpy as _csp
    _csp.set_cmdstan_path(str(install_dir))
    print(f"[david] CmdStan ready at {install_dir}", flush=True)
    tarball.unlink(missing_ok=True)


def _find_executable(name: str) -> str | None:
    """Return the full path to *name* if found on PATH or common system locations."""
    import shutil
    found = shutil.which(name)
    if found:
        return found
    # Fallback: check standard locations that may be missing from PATH
    from pathlib import Path
    for d in ("/usr/bin", "/usr/local/bin", "/bin", "/usr/sbin",
              "/nix/var/nix/profiles/default/bin", "/root/.nix-profile/bin"):
        p = Path(d) / name
        if p.exists():
            return str(p)
    return None


def _get_compiled_model() -> CmdStanModel:
    global _COMPILED_MODEL
    if _COMPILED_MODEL is None:
        _ensure_cmdstan()
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


def _build_stan_data_from_rows(
    evidence_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    strata_rows: list[dict[str, str]],
    L: int,
    H_forecast: int,
) -> dict[str, Any]:
    """Build the Stan data dict from pre-loaded row dicts.

    Shared by both the Postgres and CSV paths of assemble_fit_data().
    All row dicts use string values; numeric conversion is done here.
    """
    from collections import defaultdict

    def _index_map(values: list[str]) -> dict[str, int]:
        return {v: i for i, v in enumerate(sorted(set(values)), start=1)}

    stratum_to_series = _index_map([r["stratum_g"] for r in strata_rows])
    tactic_to_index   = _index_map([r["tactic_k"]  for r in label_rows   if r.get("tactic_k")])
    source_to_index   = _index_map([r["source_id"]  for r in source_rows])
    coder_to_index    = _index_map([r["coder_id"]   for r in label_rows   if r.get("coder_id")])
    stratum_obs       = {r["stratum_g"]: float(r.get("I_O") or 0.0) for r in strata_rows}

    # Time index: 1-based consecutive month offset within each stratum
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in evidence_rows:
        by_stratum[row["stratum_g"]].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda r: r.get("evidence_date", ""))
    evidence_time: dict[str, int] = {}
    for rows in by_stratum.values():
        seen_dates: dict[str, int] = {}
        t = 0
        for row in rows:
            d = row.get("evidence_date", "")
            if d not in seen_dates:
                t += 1
                seen_dates[d] = t
            evidence_time[row["evidence_id"]] = seen_dates[d]

    # Primary tactic per evidence item (from coder_labels)
    evidence_tactic: dict[str, str] = {}
    for row in label_rows:
        eid = row["evidence_id"]
        if row.get("tactic_k") and eid not in evidence_tactic:
            evidence_tactic[eid] = row["tactic_k"]

    # Units: one per (stratum, time, tactic) combination — 1-indexed
    unit_key_to_idx: dict[tuple, int] = {}
    unit_series:      list[int]   = []
    unit_time_list:   list[int]   = []
    unit_tactic:      list[int]   = []
    selected_flat:    list[int]   = []
    observability_flat: list[float] = []

    sorted_evidence = sorted(
        evidence_rows,
        key=lambda r: (
            stratum_to_series[r["stratum_g"]],
            evidence_time.get(r["evidence_id"], 1),
            r["evidence_id"],
        ),
    )
    evidence_to_unit: dict[str, int] = {}
    for row in sorted_evidence:
        sg  = row["stratum_g"]
        eid = row["evidence_id"]
        tk  = evidence_tactic.get(eid, "unknown")
        key = (
            stratum_to_series[sg],
            evidence_time.get(eid, 1),
            tactic_to_index.get(tk, 1),
        )
        if key not in unit_key_to_idx:
            unit_key_to_idx[key] = len(unit_key_to_idx) + 1
            unit_series.append(key[0])
            unit_time_list.append(key[1])
            unit_tactic.append(key[2])
            selected_flat.append(1)   # all adjudicated evidence is selected
            observability_flat.append(stratum_obs.get(sg, 0.0))
        evidence_to_unit[eid] = unit_key_to_idx[key]

    U = len(unit_series)
    S = len(source_to_index)
    M = len(coder_to_index)
    K = len(tactic_to_index)
    R = len(stratum_to_series)
    T = max(unit_time_list) if unit_time_list else 1

    label_unit:   list[int] = []
    label_source: list[int] = []
    label_coder:  list[int] = []
    y_vals:       list[int] = []
    for row in sorted(
        label_rows,
        key=lambda r: (evidence_to_unit.get(r["evidence_id"], 0), r.get("coder_id", "")),
    ):
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

    label_start, label_len = _build_label_index(label_unit, label_source, U, S)

    return {
        "R": R, "T": T, "L": L, "K": K, "S": S, "M": M, "U": U,
        "unit_series":     unit_series,
        "unit_time":       unit_time_list,
        "unit_tactic":     unit_tactic,
        "selected":        selected_flat,
        "observability":   observability_flat,
        "N_label":         len(y_vals),
        "label_unit":      label_unit,
        "label_source":    label_source,
        "label_coder":     label_coder,
        "y":               y_vals,
        "label_start":     label_start,
        "label_len":       label_len,
        "delta_max":       0.30,
        "H_forecast":      max(1, H_forecast),
    }


def assemble_fit_data(
    input_dir: Path | None = None,
    L: int | None = None,
    H_forecast: int | None = None,
) -> dict[str, Any]:
    """Construct the Stan data dict from adjudicated data.

    Source priority:
      1. Postgres (DATABASE_URL) — used when the database is reachable and
         contains adjudicated evidence (evidence_items.adjudicated = TRUE).
      2. CSV fallback — reads four files from input_dir (data/adjudicated/):
           evidence_items.csv, sources.csv, coder_labels.csv, strata.csv

    Raises FileNotFoundError when both sources are unavailable.

    L defaults to 3 (pre-registered candidate).
    H_forecast defaults to FORECAST_HORIZONS_MONTHS[0].
    """
    import csv as csv_mod

    L = L or 3
    H_forecast = H_forecast or FORECAST_HORIZONS_MONTHS[0]

    # ── 1. Try Postgres ───────────────────────────────────────────────────────
    _db_error: str = ""
    try:
        from ..db.repositories import get_adjudicated_data
        rows = get_adjudicated_data()
        return _build_stan_data_from_rows(
            rows["evidence_rows"], rows["source_rows"],
            rows["label_rows"],   rows["strata_rows"],
            L, H_forecast,
        )
    except Exception as exc:
        _db_error = str(exc)

    # ── 2. CSV fallback ───────────────────────────────────────────────────────
    input_dir = input_dir or ADJUDICATED_DIR
    required  = ["evidence_items.csv", "sources.csv", "coder_labels.csv", "strata.csv"]
    missing   = [f for f in required if not (input_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Postgres unavailable ({_db_error}) and CSV fallback incomplete — "
            f"missing: {missing} in {input_dir}.\n"
            "To use Postgres: docker compose up -d && david db init\n"
            "To use CSVs: populate data/adjudicated/ with the four required files."
        )

    def _read_csv(fname: str) -> list[dict[str, str]]:
        with (input_dir / fname).open(newline="") as fh:
            return [{k: (v or "").strip() for k, v in row.items()}
                    for row in csv_mod.DictReader(fh)]

    return _build_stan_data_from_rows(
        _read_csv("evidence_items.csv"),
        _read_csv("sources.csv"),
        _read_csv("coder_labels.csv"),
        _read_csv("strata.csv"),
        L, H_forecast,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _run_theorem_gates(fit: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Extract posterior draws and run theorem A'/B'/C'/D' gates.

    Each theorem returns a typed dict with a 'gate_status' key.
    Exceptions are caught per-theorem and surface as gate_status='error'
    so the fit summary is always fully populated (fail-closed: error ≠ pass).
    """
    from ..theorems.A_prime import identification_distance_draws
    from ..theorems.B_prime import informativeness_draws
    from ..theorems.C_renamed import compute_posterior_fdp_threshold
    from ..theorems.D_forecast_horizon import horizon_validity, stationary_marginal_time

    gates: dict[str, Any] = {}

    # Extract raw draws (shared across theorems)
    delta_raw_d = fit.stan_variable("delta_raw")             # (D, S)
    j_raw_d     = fit.stan_variable("j_raw")                 # (D, S)
    delta_obs_d = fit.stan_variable("delta_observability")   # (D, S)
    j_obs_d     = fit.stan_variable("j_observability")       # (D, S)
    alpha_act_d = fit.stan_variable("alpha_activity")        # (D, L, K)
    dwell_d     = fit.stan_variable("dwell_lambda")          # (D, L)
    log_jump_d  = fit.stan_variable("log_jump")              # (D, L, L)

    D_draws = delta_raw_d.shape[0]
    # Detection probabilities at representative observability O = 0.5
    O_rep   = 0.5
    delta_d = 0.30 * _sigmoid(delta_raw_d + delta_obs_d * O_rep)  # (D, S)
    j_d     = _sigmoid(j_raw_d + j_obs_d * O_rep)                 # (D, S)
    rho_d   = delta_d + (1.0 - delta_d) * j_d                     # (D, S)
    # phi: marginal activity probability per draw.
    # Use min over (L, K) — worst-case regime × tactic — so the identification
    # check is conservative: the hardest-to-identify cell sets the bound.
    phi_d = _sigmoid(alpha_act_d.reshape(D_draws, -1)).min(axis=1)  # (D,)

    # ── Theorem A' — practical identification distance ────────────────────────
    try:
        d_theta = identification_distance_draws(phi_d, rho_d, delta_d)  # (D,)
        med_d = float(np.median(d_theta))
        gates["A_prime"] = {
            "theorem": "A_prime",
            "median_d_theta": med_d,
            "q05_d_theta": float(np.quantile(d_theta, 0.05)),
            "floor": ID_DISTANCE_FLOOR,
            "gate_status": "pass" if med_d >= ID_DISTANCE_FLOOR else "fail",
            "reason": (
                "d_theta_above_floor" if med_d >= ID_DISTANCE_FLOOR
                else f"d_theta_median_{med_d:.4f}_below_floor_{ID_DISTANCE_FLOOR}"
            ),
        }
    except Exception as exc:
        gates["A_prime"] = {"theorem": "A_prime", "gate_status": "error", "reason": str(exc)}

    # ── Theorem B' — source informativeness ───────────────────────────────────
    try:
        I_d = informativeness_draws(rho_d, delta_d)       # (D, S)
        # Conservative: worst source per draw, then lower 95% credible bound
        I_worst    = I_d.min(axis=1)                      # (D,)
        I_lower95  = float(np.quantile(I_worst, 0.025))
        I_med      = float(np.median(I_worst))
        gates["B_prime"] = {
            "theorem": "B_prime",
            "median_I_worst_source": I_med,
            "lower_95_I_worst_source": I_lower95,
            "floor": INFORMATIVENESS_FLOOR_LOWER_95,
            "gate_status": "pass" if I_lower95 >= INFORMATIVENESS_FLOOR_LOWER_95 else "fail",
            "reason": (
                "informativeness_above_floor"
                if I_lower95 >= INFORMATIVENESS_FLOOR_LOWER_95
                else f"I_lower95_{I_lower95:.4f}_below_floor_{INFORMATIVENESS_FLOOR_LOWER_95}"
            ),
        }
    except Exception as exc:
        gates["B_prime"] = {"theorem": "B_prime", "gate_status": "error", "reason": str(exc)}

    # ── Theorem C' — posterior FDP routing ────────────────────────────────────
    try:
        # Mean posterior activity probability for each (L, K) cell
        p_hat_cells = _sigmoid(alpha_act_d.reshape(D_draws, -1)).mean(axis=0)  # (L*K,)
        C_result = compute_posterior_fdp_threshold(p_hat_cells)
        gates["C_prime"] = {
            "theorem": "C_prime",
            "n_cells": C_result.n_cells,
            "n_flagged": C_result.n_flagged,
            "threshold_p": C_result.threshold_p,
            "posterior_expected_fdp": C_result.posterior_expected_fdp_at_threshold,
            "q_target": POSTERIOR_FDP_DEFAULT_Q,
            "gate_status": "pass",   # C' routes; it does not block the fit
            "reason": (
                f"flagged_{C_result.n_flagged}_of_{C_result.n_cells}_"
                f"cells_at_q={POSTERIOR_FDP_DEFAULT_Q}"
            ),
        }
    except Exception as exc:
        gates["C_prime"] = {"theorem": "C_prime", "gate_status": "error", "reason": str(exc)}

    # ── Theorem D' — forecast horizon validity ────────────────────────────────
    try:
        # Use posterior median Pi and dwell_lambda (representative point for MC)
        Pi_log_med = np.median(log_jump_d, axis=0)   # (L, L) log-scale
        Pi_med     = np.exp(Pi_log_med)
        np.fill_diagonal(Pi_med, 0.0)
        row_sums   = Pi_med.sum(axis=1, keepdims=True)
        row_sums   = np.where(row_sums == 0, 1.0, row_sums)  # guard zero rows
        Pi_med     = Pi_med / row_sums
        dwell_med  = np.median(dwell_d, axis=0)       # (L,)
        # z_T prior: use stationary marginal (conservative; no stratum data here)
        pi_inf = stationary_marginal_time(Pi_med, dwell_med)
        D_hv = horizon_validity(
            cell_id="overall",
            Pi_off_diag=Pi_med,
            dwell_mean=dwell_med,
            z_t_distribution=pi_inf,
            h_max=max(FORECAST_HORIZONS_MONTHS),
            n_mc=500,   # fast approximation; increase in production
        )
        gates["D_prime"] = {
            "theorem": "D_prime",
            "h_star_months": D_hv.h_star_months,
            "tau": D_hv.tau,
            "prior_drift_share_at_h_max": D_hv.prior_drift_share_at_h_max,
            "forecast_horizons": list(FORECAST_HORIZONS_MONTHS),
            "gate_status": (
                "pass" if D_hv.h_star_months >= min(FORECAST_HORIZONS_MONTHS)
                else "fail"
            ),
            "reason": f"h_star={D_hv.h_star_months}_months",
        }
    except Exception as exc:
        gates["D_prime"] = {"theorem": "D_prime", "gate_status": "error", "reason": str(exc)}

    return gates


def _run_f1_gate(data: dict[str, Any], n_prior_worlds: int = 200) -> dict[str, Any]:
    """F1 prior predictive realism check.

    Draws n_prior_worlds synthetic worlds from the prior with the same
    (R, T, L, K, S, M) as the fit data, computes per-world Y-rate
    (fraction of labels = 1), and checks whether the prior predictive
    median Y-rate falls within the historical band derived from the
    observed label data.

    Historical band: [5th, 95th] percentile of per-time-period Y-rates.
    Band width clamped to ≥ 0.10 to prevent false negatives on small T.
    """
    from ..simulator.synthetic_world import sample_world, HyperPrior
    from ..simulator.adversarial_battery import F1_prior_predictive_realism

    prior = HyperPrior(
        R=data["R"], T=data["T"], L=data["L"],
        K=data["K"], S=data["S"], M=data["M"], H=1,
    )
    prior_Y_rates = np.array([
        float(sample_world(prior, seed=i).y.ravel().mean())
        for i in range(n_prior_worlds)
    ])

    # Historical band from per-time-period label positive rates
    unit_time_arr  = np.array(data["unit_time"])      # (U,)
    y_arr          = np.array(data["y"], dtype=float) # (N_label,)
    label_unit_arr = np.array(data["label_unit"])     # (N_label,)
    label_time     = unit_time_arr[label_unit_arr - 1]  # (N_label,) — 0-based unit → time
    unique_times   = np.unique(label_time)
    per_time_rates = np.array([
        y_arr[label_time == t].mean() for t in unique_times
    ])
    hist_5th  = float(np.quantile(per_time_rates, 0.05))
    hist_95th = float(np.quantile(per_time_rates, 0.95))
    # Clamp band width to avoid trivially-failing gates on sparse data
    if hist_95th - hist_5th < 0.10:
        mid       = (hist_5th + hist_95th) / 2.0
        hist_5th  = max(0.0, mid - 0.05)
        hist_95th = min(1.0, mid + 0.05)

    result = F1_prior_predictive_realism(prior_Y_rates, hist_5th, hist_95th)
    return {
        "gate_status": result.gate_status,
        "statistic": result.statistic,
        "prior_predictive_Y_rate_median": float(np.median(prior_Y_rates)),
        "prior_predictive_Y_rate_5th": float(np.quantile(prior_Y_rates, 0.05)),
        "prior_predictive_Y_rate_95th": float(np.quantile(prior_Y_rates, 0.95)),
        "historical_band_5th": hist_5th,
        "historical_band_95th": hist_95th,
        "n_prior_worlds": n_prior_worlds,
        "reason": result.reason,
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

    model = _get_compiled_model()
    fit = model.sample(
        data=data,
        chains=MIN_CHAINS,
        parallel_chains=1,          # run chains sequentially to cap peak RAM
        iter_warmup=1000,
        iter_sampling=MIN_POSTERIOR_DRAWS // MIN_CHAINS,
        seed=42,
        adapt_delta=0.95,
        max_treedepth=10,           # was 12 (4096 steps); 10 = 1024 steps, 4× less RAM
        output_dir=str(fit_dir / "cmdstan"),
    )

    summary = fit.summary()
    diagnostics = fit.diagnose()

    rhat_max = float(summary["R_hat"].dropna().max())
    bulk_ess_min = float(summary["ESS_bulk"].dropna().min())
    tail_ess_min = float(summary["ESS_tail"].dropna().min())
    divergences = int(sum(1 for s in diagnostics.split("\n") if "divergent" in s))

    mcmc_failed: list[str] = []
    if rhat_max > R_HAT_MAX:
        mcmc_failed.append(f"R_hat={rhat_max:.4f}>{R_HAT_MAX}")
    if bulk_ess_min < BULK_ESS_MIN:
        mcmc_failed.append(f"ESS_bulk={bulk_ess_min:.0f}<{BULK_ESS_MIN}")
    if tail_ess_min < TAIL_ESS_MIN:
        mcmc_failed.append(f"ESS_tail={tail_ess_min:.0f}<{TAIL_ESS_MIN}")
    if divergences > DIVERGENCES_ALLOWED:
        mcmc_failed.append(f"divergences={divergences}>{DIVERGENCES_ALLOWED}")

    # Theorem gates (A'/B'/C'/D') — run against posterior draws
    theorem_gates = _run_theorem_gates(fit, data)
    theorem_failed = [
        k for k, v in theorem_gates.items()
        if v.get("gate_status") in ("fail", "error")
    ]

    # F1: prior predictive realism
    f1_gate = _run_f1_gate(data)

    all_failed = (
        mcmc_failed
        + [f"theorem_{k}" for k in theorem_failed]
        + (["F1"] if f1_gate["gate_status"] == "fail" else [])
    )

    fit_summary = {
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "rhat_max": rhat_max,
        "ess_bulk_min": bulk_ess_min,
        "ess_tail_min": tail_ess_min,
        "divergences": divergences,
        "theorems": theorem_gates,
        "gates": {
            "F1": f1_gate,
            # F3/F4/F5 require held-out labels or true activity ground-truth;
            # skip gracefully until cross-validation infrastructure is wired.
            "F3": {"gate_status": "skip", "reason": "no_held_out_labels"},
            "F4": {"gate_status": "skip", "reason": "activity_truth_unknown_in_real_data_mode"},
            "F5": {"gate_status": "skip", "reason": "no_held_out_set"},
        },
        "gate_status": "fail" if all_failed else "pass",
        "reason": "fit_pass" if not all_failed else "; ".join(all_failed),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    (fit_dir / "fit_summary.json").write_text(json.dumps(fit_summary, indent=2))

    # Write to DB first — before the potentially large parquet write — so a
    # memory spike during draws_pd() doesn't silently lose the gate result.
    try:
        from ..db.repositories import write_fit_run
        write_fit_run({
            **fit_summary,
            "n_strata": data.get("R"),
            "n_labels": data.get("N_label"),
        })
    except Exception as db_exc:
        print(f"[fit] DB write failed (non-fatal): {db_exc}", flush=True)

    # Parquet is best-effort — a memory spike here must not lose the fit result.
    try:
        fit.draws_pd().to_parquet(fit_dir / "draws.parquet")
    except Exception as parquet_exc:
        print(f"[fit] draws parquet write failed (non-fatal): {parquet_exc}", flush=True)

    return {**fit_summary, "fit_dir": str(fit_dir)}
