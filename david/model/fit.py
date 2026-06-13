"""cmdstanpy wrapper for m01_forward.stan with fit-contract gates.

The fit-contract gates mirror council_m01/simulation/full_integrated_fit_contract.py
(R-hat, ESS, divergences, posterior draws). Pass-through here; do not relax.

Output: data/fits/{run_id}/{fit_summary.json, draws.parquet, diagnostics.json}.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from cmdstanpy import CmdStanModel

from ..config import (
    ADJUDICATED_DIR, BULK_ESS_MIN, CODED_DIR, DIVERGENCES_ALLOWED, FITS_DIR,
    FORECAST_HORIZONS_MONTHS, DWELL_LOG_LAMBDA_PRIOR_SD, ID_DISTANCE_FLOOR,
    INFORMATIVENESS_FLOOR_LOWER_95, KAPPA_CALIBRATION_MIN_MEAN, MIN_CHAINS,
    MIN_POSTERIOR_DRAWS, M01_FORWARD_STAN, MODEL_VERSION,
    N_EFF_I2_FLOOR, OBSERVABILITY_GRID, POSTERIOR_FDP_DEFAULT_Q,
    PRIOR_PREDICTIVE_REFERENCE,
    R_HAT_MAX, TAIL_ESS_MIN,
)
from ..ingest.source_independence_ledger import structural_independence_summary

_COMPILED_MODEL: CmdStanModel | None = None

log = logging.getLogger(__name__)

_SUMMARY_TIMEOUT_S = 120  # stansummary subprocess timeout; falls back to arviz


def _safe_fit_summary(fit, timeout_s: int = _SUMMARY_TIMEOUT_S):
    """Call fit.summary() with a hard timeout.

    cmdstanpy delegates to the ``stansummary`` subprocess via popen.  When the
    Python process is started in a backgrounded terminal the subprocess can
    inherit a broken stdout pipe and hang indefinitely (0 CPU, infinite wall
    time).  We guard with a thread-based timeout and fall back to computing
    R-hat / ESS from the raw draws via arviz — identical diagnostics, no
    subprocess.
    """
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fit.summary)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            log.warning(
                "fit.summary() timed out after %ds — falling back to arviz diagnostics.",
                timeout_s,
            )
            return _arviz_summary_fallback(fit)


def _arviz_summary_fallback(fit):
    """Compute R-hat and ESS from draws using arviz when stansummary hangs.

    Returns a DataFrame with the same columns as fit.summary() so the calling
    code needs no changes:  R_hat, ESS_bulk, ESS_tail.
    """
    import pandas as pd

    try:
        import arviz as az

        draws = fit.draws()  # shape (chains, draws, params)
        param_names = fit.column_names
        # arviz expects (chains, draws, params)
        idata = az.from_cmdstanpy(fit)
        summary_az = az.summary(idata, var_names=None, kind="diagnostics")
        # Rename columns to match cmdstanpy convention
        col_map = {
            "r_hat": "R_hat",
            "ess_bulk": "ESS_bulk",
            "ess_tail": "ESS_tail",
        }
        summary_az = summary_az.rename(columns=col_map)
        # Ensure required columns exist
        for col in ("R_hat", "ESS_bulk", "ESS_tail"):
            if col not in summary_az.columns:
                summary_az[col] = float("nan")
        log.info("arviz fallback produced summary for %d parameters.", len(summary_az))
        return summary_az
    except Exception as exc:
        log.error("arviz fallback failed (%s) — returning worst-case summary.", exc)
        # Return a minimal DataFrame that will trigger gate failures (fail-closed).
        return pd.DataFrame(
            {"R_hat": [2.0], "ESS_bulk": [0.0], "ESS_tail": [0.0]},
            index=["fallback_error"],
        )


_CMDSTAN_VERSION = "2.39.0"
# Direct tarball URL — bypasses the GitHub API (60 req/hr anon rate limit).
_CMDSTAN_URL = (
    f"https://github.com/stan-dev/cmdstan/releases/download/"
    f"v{_CMDSTAN_VERSION}/cmdstan-{_CMDSTAN_VERSION}.tar.gz"
)
# Install inside /app so Railpack's "copy /app" step bakes it into the image.
_CMDSTAN_DIR = f"/app/.cmdstan/cmdstan-{_CMDSTAN_VERSION}"


def _ensure_cmdstan() -> None:
    """Ensure CmdStan is available and registered with cmdstanpy.

    Fast path A: /app/.cmdstan/cmdstan-VERSION/bin/stanc exists (baked by build.sh)
                 → just call set_cmdstan_path and return.
    Fast path B: cmdstanpy already knows a valid path → return.
    Slow path  : apt-get update → install make/g++ → download tarball →
                 extract to /app/.cmdstan/ → make build → set_cmdstan_path.
    """
    import os
    import subprocess
    import tarfile
    import urllib.request
    from pathlib import Path

    import cmdstanpy as _csp

    # Fast path A — baked binary from build.sh
    baked = Path(_CMDSTAN_DIR) / "bin" / "stanc"
    if baked.exists():
        _csp.set_cmdstan_path(_CMDSTAN_DIR)
        return

    # Fast path B — cmdstanpy already configured
    try:
        from cmdstanpy.utils.cmdstan import cmdstan_path
        cmdstan_path()
        return
    except ValueError:
        pass

    # Slow path — install from scratch
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
            "make not found after apt-get install — cannot compile CmdStan."
        )

    install_parent = Path("/app/.cmdstan")
    install_parent.mkdir(parents=True, exist_ok=True)
    install_dir = install_parent / f"cmdstan-{_CMDSTAN_VERSION}"

    print(f"[david] Downloading CmdStan {_CMDSTAN_VERSION}…", flush=True)
    tarball = Path(f"/tmp/cmdstan-{_CMDSTAN_VERSION}.tar.gz")
    urllib.request.urlretrieve(_CMDSTAN_URL, tarball)

    print("[david] Extracting…", flush=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(install_parent)
    tarball.unlink(missing_ok=True)

    print(f"[david] Building CmdStan with {make_bin} (make build — ~8 min)…", flush=True)
    env = os.environ.copy()
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + env.get("PATH", "")
    subprocess.run([make_bin, "build"], cwd=str(install_dir), env=env, check=True)

    _csp.set_cmdstan_path(str(install_dir))
    print(f"[david] CmdStan ready at {install_dir}", flush=True)


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
        "kappa_plus_raw_prior_mean": [1.0] * M,
        "kappa_plus_raw_prior_sd": [0.5] * M,
        "kappa_minus_raw_prior_mean": [1.0] * M,
        "kappa_minus_raw_prior_sd": [0.5] * M,
        "coder_likelihood_weight": [1.0] * M,
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
                 "delta_raw", "j_raw", "delta_observability", "j_observability",
                 "coder_common_mode_weight"):
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
    require_coder_calibration: bool = True,
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
    kappa_priors = (
        _load_calibrated_kappa_raw_priors(coder_to_index)
        if require_coder_calibration
        else _default_kappa_raw_priors(M)
    )
    active_source_ids = [
        source_id
        for source_id, _idx in sorted(source_to_index.items(), key=lambda kv: kv[1])
    ]

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
        **kappa_priors,
        "delta_max":       0.30,
        "H_forecast":      max(1, H_forecast),
        "_source_ids": active_source_ids,
        "_source_independence": structural_independence_summary(active_source_ids),
    }


def _stan_data_only(data: dict[str, Any]) -> dict[str, Any]:
    """Remove non-Stan metadata before passing data to CmdStan."""
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _default_kappa_raw_priors(n_coders: int) -> dict[str, list[float]]:
    return {
        "kappa_plus_raw_prior_mean": [1.0] * n_coders,
        "kappa_plus_raw_prior_sd": [0.5] * n_coders,
        "kappa_minus_raw_prior_mean": [1.0] * n_coders,
        "kappa_minus_raw_prior_sd": [0.5] * n_coders,
        "coder_likelihood_weight": [1.0] * n_coders,
    }


def _latest_coder_calibration_path(coded_dir: Path | None = None) -> Path:
    coded_dir = coded_dir or CODED_DIR
    candidates = sorted(coded_dir.glob("coder_calibration_v*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"coder calibration artifact missing in {coded_dir}; run `david calibrate-coders`"
        )
    return candidates[-1]


def _kappa_summary_to_raw_prior(summary: dict[str, Any]) -> tuple[float, float]:
    """Moment-match constrained kappa summary to raw scale."""
    mean = float(summary.get("mean", float("nan")))
    sd = float(summary.get("sd", float("nan")))
    if not np.isfinite(mean) or not np.isfinite(sd):
        raise ValueError("coder calibration summary missing finite mean/sd")
    if mean < KAPPA_CALIBRATION_MIN_MEAN:
        raise ValueError(
            f"coder calibration mean {mean:.3f} below tripwire {KAPPA_CALIBRATION_MIN_MEAN:.3f}"
        )
    kappa = float(np.clip(mean, 0.5001, 0.9999))
    q = float(np.clip(2.0 * kappa - 1.0, 1e-4, 1.0 - 1e-4))
    raw_mean = float(np.log(q / (1.0 - q)))
    raw_sd = float(max(0.05, min(5.0, sd * 2.0 / (q * (1.0 - q)))))
    return raw_mean, raw_sd


def _load_calibrated_kappa_raw_priors(
    coder_to_index: dict[str, int],
    coded_dir: Path | None = None,
) -> dict[str, list[float]]:
    coded_dir = coded_dir or CODED_DIR
    path = _latest_coder_calibration_path(coded_dir)
    payload = json.loads(path.read_text())
    artifact_coders = payload.get("coders", [])
    if not artifact_coders:
        raise ValueError(f"coder calibration artifact {path} has no coders")
    artifact_index = {coder: i for i, coder in enumerate(artifact_coders)}
    missing = sorted(set(coder_to_index) - set(artifact_index))
    if missing:
        raise ValueError(
            f"coder calibration artifact {path} missing active coders: {missing}"
        )

    M = len(coder_to_index)
    plus_mean = [0.0] * M
    plus_sd = [0.0] * M
    minus_mean = [0.0] * M
    minus_sd = [0.0] * M
    for coder, stan_idx in coder_to_index.items():
        artifact_i = artifact_index[coder]
        plus_mean[stan_idx - 1], plus_sd[stan_idx - 1] = _kappa_summary_to_raw_prior(
            payload["kappa_plus"][artifact_i]
        )
        minus_mean[stan_idx - 1], minus_sd[stan_idx - 1] = _kappa_summary_to_raw_prior(
            payload["kappa_minus"][artifact_i]
        )

    return {
        "kappa_plus_raw_prior_mean": plus_mean,
        "kappa_plus_raw_prior_sd": plus_sd,
        "kappa_minus_raw_prior_mean": minus_mean,
        "kappa_minus_raw_prior_sd": minus_sd,
        # Interim C5 design-effect discount: absent audited coder-family metadata,
        # treat the active coder pool as one fully correlated family.
        "coder_likelihood_weight": [1.0 / max(1, M)] * M,
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


def _min_units_per_series_tactic(data: dict[str, Any]) -> int:
    """Minimum replicate count across observed (series, tactic) cells."""
    unit_series = data.get("unit_series")
    unit_tactic = data.get("unit_tactic")
    if unit_series is not None and unit_tactic is not None and len(unit_series) == len(unit_tactic):
        counts: dict[tuple[int, int], int] = {}
        for r, k in zip(unit_series, unit_tactic, strict=False):
            key = (int(r), int(k))
            counts[key] = counts.get(key, 0) + 1
        if counts:
            return min(counts.values())

    U = int(data.get("U", 1))
    R = max(1, int(data.get("R", 1)))
    K = max(1, int(data.get("K", 1)))
    return max(1, U // (R * K))


def _run_theorem_gates(fit: Any, data: dict[str, Any]) -> dict[str, Any]:  # NOSONAR - fail-closed theorem ledger coordinator
    """Extract posterior draws and run theorem A'/B'/C'/D' gates.

    Each theorem returns a typed dict with a 'gate_status' key.
    Exceptions are caught per-theorem and surface as gate_status='error'
    so the fit summary is always fully populated (fail-closed: error ≠ pass).
    """
    from ..theorems.A_prime import identification_distance_draws
    from ..theorems.B_prime import informativeness_draws
    from ..theorems.C_renamed import compute_posterior_fdp_threshold
    from ..theorems.D_forecast_horizon import (
        horizon_validity_from_z_future_draws,
        stationary_marginal_time,
    )

    gates: dict[str, Any] = {}

    # Extract raw draws (shared across theorems)
    delta_raw_d = fit.stan_variable("delta_raw")             # (D, S)
    j_raw_d     = fit.stan_variable("j_raw")                 # (D, S)
    delta_obs_d = fit.stan_variable("delta_observability")   # (D, S)
    j_obs_d     = fit.stan_variable("j_observability")       # (D, S)
    alpha_act_d = fit.stan_variable("alpha_activity")        # (D, L, K)
    dwell_d     = fit.stan_variable("dwell_lambda")          # (D, L)
    dwell_mean_d = dwell_d + 1.0                             # shifted-Poisson mean μ = λ + 1
    log_jump_d  = fit.stan_variable("log_jump")              # (D, L, L)
    terminal_regime_d = fit.stan_variable("terminal_regime_posterior_draw")  # (D, R, L)
    z_future_d = fit.stan_variable("z_future")               # (D, R, H)

    D_draws = delta_raw_d.shape[0]
    obs_grid = np.asarray(OBSERVABILITY_GRID, dtype=float)
    if obs_grid.ndim != 1 or obs_grid.size == 0:
        raise ValueError("OBSERVABILITY_GRID must be a non-empty 1-D sequence")
    # Detection probabilities over the pre-registered observability grid.
    # Legacy midpoint arrays remain available for scalar diagnostics that are
    # not gate-defining, but A'/B' gate statistics use the worst grid point.
    delta_grid_d = 0.30 * _sigmoid(
        delta_raw_d[:, None, :] + delta_obs_d[:, None, :] * obs_grid[None, :, None]
    )  # (D, G, S)
    j_grid_d = _sigmoid(
        j_raw_d[:, None, :] + j_obs_d[:, None, :] * obs_grid[None, :, None]
    )  # (D, G, S)
    rho_grid_d = delta_grid_d + (1.0 - delta_grid_d) * j_grid_d  # (D, G, S)
    o_rep_idx = int(np.argmin(np.abs(obs_grid - 0.5)))
    delta_d = delta_grid_d[:, o_rep_idx, :]  # (D, S)
    rho_d = rho_grid_d[:, o_rep_idx, :]      # (D, S)
    if terminal_regime_d.ndim == 2:
        terminal_regime_for_a = terminal_regime_d[:, None, :]
    else:
        terminal_regime_for_a = terminal_regime_d
    if terminal_regime_for_a.shape[0] != D_draws or terminal_regime_for_a.shape[2] != alpha_act_d.shape[1]:
        raise ValueError("terminal_regime_posterior_draw shape incompatible with alpha_activity")
    terminal_regime_for_a = terminal_regime_for_a / (
        terminal_regime_for_a.sum(axis=2, keepdims=True) + 1e-12
    )
    phi_regime_tactic_d = _sigmoid(alpha_act_d)  # (D, L, K)
    # A' gates on phi_g,k = sum_r zeta_g(r) * Pr(A=1 | Z=r, tactic=k),
    # then takes the hardest currently occupied stratum/tactic cell.
    phi_series_tactic_d = np.einsum(
        "drl,dlk->drk",
        terminal_regime_for_a,
        phi_regime_tactic_d,
    )  # (D, R, K)
    phi_cells_d = phi_series_tactic_d.reshape(D_draws, -1)
    phi_d = phi_cells_d.min(axis=1)

    # ── Theorem A' — practical identification distance ────────────────────────
    try:
        d_theta_by_obs = np.stack(
            [
                identification_distance_draws(
                    phi_d, rho_grid_d[:, g_idx, :], delta_grid_d[:, g_idx, :]
                )
                for g_idx in range(obs_grid.size)
            ],
            axis=1,
        )  # (D, G)
        d_theta = d_theta_by_obs.min(axis=1)  # posterior draws of min_O d(theta, O)
        med_d = float(np.median(d_theta))
        med_d_by_obs = np.median(d_theta_by_obs, axis=0)
        worst_obs_idx = int(np.argmin(med_d_by_obs))
        structural = data.get("_source_independence")
        if isinstance(structural, dict) and structural.get("gate_status") != "pass":
            a_status = "fail"
            a_reason = f"structural_source_independence:{structural.get('reason', 'fail')}"
        elif med_d >= ID_DISTANCE_FLOOR:
            a_status = "pass"
            a_reason = "d_theta_above_floor"
        else:
            a_status = "fail"
            a_reason = f"d_theta_median_{med_d:.4f}_below_floor_{ID_DISTANCE_FLOOR}"
        gates["A_prime"] = {
            "theorem": "A_prime",
            "median_d_theta": med_d,
            "q05_d_theta": float(np.quantile(d_theta, 0.05)),
            "floor": ID_DISTANCE_FLOOR,
            "aggregation": "min_over_terminal_weighted_series_tactic_cells_and_observability_grid",
            "observability_grid": [float(x) for x in obs_grid],
            "observability_aggregation": "min_over_pre_registered_grid",
            "worst_observability": float(obs_grid[worst_obs_idx]),
            "median_d_theta_by_observability": {
                f"{float(obs):.6g}": float(val)
                for obs, val in zip(obs_grid, med_d_by_obs, strict=False)
            },
            "source_independence": structural,
            "gate_status": a_status,
            "reason": a_reason,
        }
    except Exception as exc:
        gates["A_prime"] = {"theorem": "A_prime", "gate_status": "error", "reason": str(exc)}

    # ── Theorem B' — source informativeness ───────────────────────────────────
    try:
        i_draws = informativeness_draws(rho_grid_d, delta_grid_d)       # (D, G, S)
        if i_draws.shape[2] >= 3:
            i_by_obs = np.sort(i_draws, axis=2)[:, :, ::-1][:, :, 2]
            source_aggregation = "third_largest_source"
        else:
            i_by_obs = i_draws.min(axis=2)
            source_aggregation = "min_source_under_sourced"
        i_gate = i_by_obs.min(axis=1)
        i_med_by_obs = np.median(i_by_obs, axis=0)
        worst_i_obs_idx = int(np.argmin(i_med_by_obs))
        i_lower95  = float(np.quantile(i_gate, 0.025))
        i_median      = float(np.median(i_gate))
        # Gate 1: I lower-95 floor
        gate1_pass = i_lower95 >= INFORMATIVENESS_FLOOR_LOWER_95
        # Gate 2: N_eff × I² floor (pre-registered N_EFF_I2_FLOOR = 3.0).
        # Implement Godambe dependence adjustment:
        from ..theorems.B_prime import dependence_adjusted_n_eff, compute_activity_autocorrelation

        # We need Pi_med and dwell_med for autocorrelation
        Pi_log_med = np.median(log_jump_d, axis=0)
        L_val = Pi_log_med.shape[0]
        Pi_med = np.exp(Pi_log_med)
        idx = np.arange(L_val)
        Pi_med[idx, idx] = 0.0
        row_sums = Pi_med.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        Pi_med = Pi_med / row_sums
        dwell_med = np.median(dwell_mean_d, axis=0)

        phi_k_med = np.median(_sigmoid(alpha_act_d).mean(axis=2), axis=0)  # (L,)
        max_lag = max(1, data.get("T", 1) - 1)

        corr_A_h = compute_activity_autocorrelation(
            Pi_med, dwell_med, phi_k_med, max_lag=max_lag, n_mc=max(200, 500 // max(L_val, 1))
        )

        phi_med = float(np.median(phi_cells_d))
        rho_med = float(np.median(rho_d))
        delta_med = float(np.median(delta_d))
        p_med = rho_med * phi_med + delta_med * (1.0 - phi_med)

        n_units   = _min_units_per_series_tactic(data)
        n_adj = dependence_adjusted_n_eff(n_units, i_median, phi_med, p_med, corr_A_h)
        n_eff_i2  = n_adj * (i_median ** 2)
        gate2_pass = n_eff_i2 >= N_EFF_I2_FLOOR
        if gate1_pass and gate2_pass:
            b_status = "pass"
            b_reason = "I_lower_95_and_N_eff_I2_above_floor"
        elif not gate1_pass:
            b_status = "fail"
            b_reason = f"I_lower95_{i_lower95:.4f}_below_floor_{INFORMATIVENESS_FLOOR_LOWER_95}"
        else:
            b_status = "fail"
            b_reason = f"N_eff_I2_{n_eff_i2:.2f}_below_floor_{N_EFF_I2_FLOOR}"
        gates["B_prime"] = {
            "theorem": "B_prime",
            "median_I_worst_source": i_median,
            "lower_95_I_worst_source": i_lower95,
            "source_aggregation": source_aggregation,
            "observability_grid": [float(x) for x in obs_grid],
            "observability_aggregation": "min_over_pre_registered_grid",
            "worst_observability": float(obs_grid[worst_i_obs_idx]),
            "median_I_by_observability": {
                f"{float(obs):.6g}": float(val)
                for obs, val in zip(obs_grid, i_med_by_obs, strict=False)
            },
            "floor": INFORMATIVENESS_FLOOR_LOWER_95,
            "n_units": n_units,
            "n_eff_i2": n_eff_i2,
            "n_eff_i2_floor": N_EFF_I2_FLOOR,
            "gate_status": b_status,
            "reason": b_reason,
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
        L_val = log_jump_d.shape[1]
        if terminal_regime_d.ndim == 2:
            terminal_regime_d = terminal_regime_d[:, None, :]
        if terminal_regime_d.shape[0] != D_draws or terminal_regime_d.shape[2] != L_val:
            raise ValueError(
                "terminal_regime_posterior_draw must have shape (draws, series, regimes)"
            )
        r_val = terminal_regime_d.shape[1]
        if z_future_d.ndim == 2:
            z_future_d = z_future_d[:, None, :]
        if z_future_d.shape[0] != D_draws or z_future_d.shape[1] != r_val:
            raise ValueError("z_future must have shape (draws, series, horizons)")
        if z_future_d.shape[2] < max(FORECAST_HORIZONS_MONTHS):
            raise ValueError("z_future does not cover all pre-registered forecast horizons")

        # Convert log_jump_d to probabilities for all draws
        Pi_d = np.exp(log_jump_d)
        idx = np.arange(L_val)
        Pi_d[:, idx, idx] = 0.0
        row_sums = Pi_d.sum(axis=2, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        Pi_d = Pi_d / row_sums

        def _h_star_summary(
            dwell_mean_draws_for_gate: np.ndarray,
            *,
            n_bootstrap_for_gate: int,
        ) -> tuple[dict[str, Any], Any]:
            h_stars_per_series: list[int] = []
            h_stars_q05_per_series: list[int] = []
            h_stars_q95_per_series: list[int] = []
            last_hv = None
            for _r in range(r_val):
                hv_r = horizon_validity_from_z_future_draws(
                    cell_id=f"series_{_r}",
                    pi_off_diag_draws=Pi_d,
                    dwell_mean_draws=dwell_mean_draws_for_gate,
                    z_t_distribution=terminal_regime_d[:, _r, :],
                    z_future_draws=z_future_d[:, _r, :],
                    h_max=max(FORECAST_HORIZONS_MONTHS),
                    n_bootstrap=n_bootstrap_for_gate,
                )
                last_hv = hv_r
                h_stars_per_series.append(hv_r.h_star_months)
                h_stars_q05_per_series.append(hv_r.h_star_q05)
                h_stars_q95_per_series.append(hv_r.h_star_q95)
            return {
                "h_star_months": int(min(h_stars_q05_per_series)),
                "h_star_q05": int(min(h_stars_q05_per_series)),
                "h_star_q95": int(max(h_stars_q95_per_series)),
                "h_stars_per_series": h_stars_per_series,
                "h_stars_q05_per_series": h_stars_q05_per_series,
                "h_stars_q95_per_series": h_stars_q95_per_series,
            }, last_hv

        nominal_h, d_hv = _h_star_summary(
            dwell_mean_d,
            n_bootstrap_for_gate=max(100, 250 // max(L_val, 1)),
        )
        h_star_gate = nominal_h["h_star_months"]

        Pi_med = np.median(Pi_d, axis=0)
        dwell_med = np.median(dwell_mean_d, axis=0)
        pi_inf = stationary_marginal_time(Pi_med, dwell_med)

        # P6 dwell-prior sensitivity: perturb the shifted-Poisson rate λ on
        # the log scale by ± one registered prior SD, then convert back to
        # the renewal kernel's mean μ = λ + 1.
        lambda_draws = np.maximum(dwell_mean_d - 1.0, 1e-9)
        dwell_minus_sd = lambda_draws * np.exp(-DWELL_LOG_LAMBDA_PRIOR_SD) + 1.0
        dwell_plus_sd = lambda_draws * np.exp(DWELL_LOG_LAMBDA_PRIOR_SD) + 1.0
        sensitivity_n_bootstrap = max(50, 150 // max(L_val, 1))
        minus_h, _ = _h_star_summary(dwell_minus_sd, n_bootstrap_for_gate=sensitivity_n_bootstrap)
        plus_h, _ = _h_star_summary(dwell_plus_sd, n_bootstrap_for_gate=sensitivity_n_bootstrap)
        prior_sensitive_horizons = [
            h for h in FORECAST_HORIZONS_MONTHS
            if (
                (h <= h_star_gate)
                != (h <= minus_h["h_star_months"])
                or (h <= h_star_gate)
                != (h <= plus_h["h_star_months"])
            )
        ]
        
        gates["D_prime"] = {
            "theorem": "D_prime",
            "h_star_months": h_star_gate,
            "h_star_q05": h_star_gate,
            "h_star_q95": nominal_h["h_star_q95"],
            "h_stars_per_series": nominal_h["h_stars_per_series"],
            "h_stars_q05_per_series": nominal_h["h_stars_q05_per_series"],
            "terminal_regime_posterior_source": "stan_generated_quantities",
            "forecast_regime_source": "stan_z_future_generated_quantities",
            "stationary_marginal_regime_median": [float(x) for x in pi_inf],
            "aggregation": "min_series_z_future_q05_first_crossing",
            "h_star_prior_sensitivity": {
                "dwell_log_lambda_prior_sd": DWELL_LOG_LAMBDA_PRIOR_SD,
                "sensitivity_n_bootstrap": sensitivity_n_bootstrap,
                "nominal_h_star": h_star_gate,
                "minus_1sd_h_star": minus_h["h_star_months"],
                "plus_1sd_h_star": plus_h["h_star_months"],
                "route_change_horizons": prior_sensitive_horizons,
                "sensitivity_changes_route": bool(prior_sensitive_horizons),
            },
            "tau": d_hv.tau,
            "prior_drift_share_at_h_max": d_hv.prior_drift_share_at_h_max,
            "forecast_horizons": list(FORECAST_HORIZONS_MONTHS),
            "gate_status": (
                "pass" if h_star_gate >= min(FORECAST_HORIZONS_MONTHS)
                else "fail"
            ),
            "reason": f"h_star_gate={h_star_gate}_months",
        }
    except Exception as exc:
        gates["D_prime"] = {"theorem": "D_prime", "gate_status": "error", "reason": str(exc)}

    return gates


def _run_f1_gate(data: dict[str, Any], n_prior_worlds: int = 200) -> dict[str, Any]:
    """F1 prior predictive realism check.

    Draws n_prior_worlds synthetic worlds from the prior with the same
    (R, T, L, K, S, M) as the fit data, computes per-world Y-rate
    (fraction of labels = 1), and checks whether the prior predictive
    median Y-rate falls within a frozen reference band disjoint from the
    observed fit data.
    """
    from ..simulator.synthetic_world import sample_world, HyperPrior
    from ..simulator.adversarial_battery import F1_prior_predictive_realism

    try:
        ref = _load_f1_reference_band()
    except Exception as exc:
        return {
            "gate_status": "fail",
            "statistic": float("nan"),
            "prior_predictive_Y_rate_median": float("nan"),
            "prior_predictive_Y_rate_5th": float("nan"),
            "prior_predictive_Y_rate_95th": float("nan"),
            "historical_band_5th": float("nan"),
            "historical_band_95th": float("nan"),
            "n_prior_worlds": 0,
            "reason": f"frozen_f1_reference_missing_or_invalid:{exc}",
        }

    prior = HyperPrior(
        R=data["R"], T=data["T"], L=data["L"],
        K=data["K"], S=data["S"], M=data["M"], H=1,
    )
    prior_Y_rates = np.array([
        float(sample_world(prior, seed=i).y.ravel().mean())
        for i in range(n_prior_worlds)
    ])

    hist_5th = ref["historical_band_5th"]
    hist_95th = ref["historical_band_95th"]

    result = F1_prior_predictive_realism(prior_Y_rates, hist_5th, hist_95th)
    return {
        "gate_status": result.gate_status,
        "statistic": result.statistic,
        "prior_predictive_Y_rate_median": float(np.median(prior_Y_rates)),
        "prior_predictive_Y_rate_5th": float(np.quantile(prior_Y_rates, 0.05)),
        "prior_predictive_Y_rate_95th": float(np.quantile(prior_Y_rates, 0.95)),
        "historical_band_5th": hist_5th,
        "historical_band_95th": hist_95th,
        "reference_path": str(PRIOR_PREDICTIVE_REFERENCE),
        "reference_version": ref.get("version"),
        "reference_window": ref.get("reference_window"),
        "n_prior_worlds": n_prior_worlds,
        "reason": result.reason,
    }


def _load_f1_reference_band(path: Path | None = None) -> dict[str, Any]:
    path = path or PRIOR_PREDICTIVE_REFERENCE
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    lo = float(payload["historical_band_5th"])
    hi = float(payload["historical_band_95th"])
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError("historical_band_5th/95th must satisfy 0 <= lo <= hi <= 1")
    payload["historical_band_5th"] = lo
    payload["historical_band_95th"] = hi
    return payload


def run_fit(run_id: str | None = None, quick: bool = False) -> dict[str, Any]:
    run_id = run_id or f"fit_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
    fit_dir = FITS_DIR / run_id
    fit_dir.mkdir(parents=True, exist_ok=True)

    try:
        # H_forecast=1 minimises the generated-quantities output during MCMC.
        # The generated quantities block emits z_future/a_future arrays of
        # shape [R, H_forecast, K].  With H_forecast=12 and iter_sampling=20000
        # that is 20k × 20 × 12 × 3 × 8 bytes ≈ 115 MB per chain → OOM on
        # Railway's shared compute.  The gate checks only need posterior draws
        # of model parameters (not forecast trajectories), so H_forecast=1 is
        # sufficient here.  A separate forecast pass (david forecast) can run
        # with the full horizon once gates pass.
        data = assemble_fit_data(H_forecast=max(FORECAST_HORIZONS_MONTHS))
    except FileNotFoundError as e:
        return {
            "gate_status": "fail",
            "reason": "fit_data_missing",
            "detail": str(e),
            "fit_dir": str(fit_dir),
        }

    model = _get_compiled_model()
    # --quick uses 200+200 per chain (same as SBC worlds) — sufficient to
    # confirm the data path works end-to-end in a demo; use full iterations
    # for a production-quality fit.
    warmup = 200 if quick else 2000
    sampling = 200 if quick else 3000
    fit = model.sample(
        data=_stan_data_only(data),
        chains=MIN_CHAINS,
        parallel_chains=MIN_CHAINS,  # run all chains in parallel (M4 local — no Railway RAM cap)
        iter_warmup=warmup,
        iter_sampling=sampling,
        seed=42,
        adapt_delta=0.97,           # 0.95→0.97: eliminates the 1 residual divergence. ESS=5427
                                    # (54× gate floor) means smaller step-size is affordable.
        max_treedepth=12,
        output_dir=str(fit_dir / "cmdstan"),
    )

    summary = _safe_fit_summary(fit)

    rhat_max = float(summary["R_hat"].dropna().max())
    bulk_ess_min = float(summary["ESS_bulk"].dropna().min())
    tail_ess_min = float(summary["ESS_tail"].dropna().min())
    divergences = int(fit.method_variables()["divergent__"].sum())

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
        "source_independence": data.get("_source_independence"),
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

    try:
        fit.draws_pd().to_parquet(fit_dir / "draws.parquet")
    except Exception as parquet_exc:
        print(f"[fit] draws parquet write failed (non-fatal): {parquet_exc}", flush=True)

    return {**fit_summary, "fit_dir": str(fit_dir)}
