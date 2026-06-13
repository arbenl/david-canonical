"""B-4 twin-parity battery: Python generative twin vs Stan prior generator.

SBC validity rests on the Python generator (david/simulator/synthetic_world.py)
and the Stan model priors (stan/m01_forward.stan, mirrored by
stan/synthetic_generator.stan) drawing from the same prior. This battery
fails closed on statistical drift between the two generators.

Design:
  1. Draw N_PY_WORLDS prior samples through the real Python code path
     (sample_world), including the t=0 forward mechanism (initial regime,
     first dwell segment, first activity draw).
  2. Draw N_STAN_DRAWS prior samples from synthetic_generator.stan's
     generated quantities block under fixed_param sampling.
  3. Per parameter: two-sample KS test plus mean (Welch) and variance
     (Brown-Forsythe) tests, Bonferroni-corrected at SBC_KS_ALPHA across
     the whole battery — same convention as the SBC gate.
  4. Internal-consistency checks pin the detection-link formula used here
     to the actual emission code path inside sample_world, so link drift in
     synthetic_world.py cannot hide behind a stale formula in this test.

Seeds are fixed: the battery is deterministic. Per the Automation Contract,
never widen ALPHA or the seed set to make a failing comparison pass — a
failure is evidence of twin drift (or a genuinely unlucky fixed draw, which
must be investigated, not suppressed).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats
from scipy.special import expit

from david.config import SBC_KS_ALPHA, SYNTHETIC_GENERATOR_STAN
from david.simulator.synthetic_world import HyperPrior, sample_world

# Dimensions mirror the SBC default world (david/simulator/sbc.py).
L, K, S, M = 3, 3, 2, 2
T_HIST = 6
H_FUTURE = 94  # T + H = 100 months: first dwell segment effectively never
               # right-censored under dwell_lambda ~ lognormal(log 6, 0.5)
DELTA_MAX = 0.30  # pre-registered delta_max (mirrors synthetic_world / Stan data)
OBS_GRID = [0.0, 0.5, 1.0]

N_PY_WORLDS = 1500
N_STAN_DRAWS = 4000
PY_SEED_BASE = 20260612
STAN_SEED = 20260612

# Hypotheses outside the parity table: 1 binary proportion test (a_draw)
# + 2 emission-consistency tests (delta path, rho path).
N_EXTRA_TESTS = 3


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def py_draws() -> dict:
    """Prior samples through the real Python generator code path."""
    prior = HyperPrior(R=1, T=T_HIST, L=L, K=K, S=S, M=M, H=H_FUTURE)
    acc: dict[str, list] = {
        "Pi": [], "alpha_activity": [], "dwell_lambda": [],
        "delta_raw": [], "j_raw": [],
        "delta_observability": [], "j_observability": [],
        "kappa_plus": [], "kappa_minus": [], "init": [],
        "coder_common_mode_weight": [],
        "selection_alpha": [], "selection_observability": [],
        "selection_activity": [],
    }
    dwell_first = []
    n_censored = 0
    a_first = []
    # Emission-consistency accumulators, split by latent activity.
    emis = {
        "delta_path": {"obs": 0.0, "mean": 0.0, "var": 0.0},
        "rho_path": {"obs": 0.0, "mean": 0.0, "var": 0.0},
    }

    for i in range(N_PY_WORLDS):
        w = sample_world(prior, seed=PY_SEED_BASE + i)
        for name in acc:
            acc[name].append(np.asarray(w.theta[name]))

        # First dwell segment of the HSMM path = 1 + Poisson(lambda[z0]).
        z_path = w.z[0]
        change = np.nonzero(z_path != z_path[0])[0]
        if change.size == 0:
            n_censored += 1
        else:
            dwell_first.append(int(change[0]))

        # First activity draw for tactic 1 (the ordered column).
        a_first.append(int(w.a[0, 0, 0]))

        # Emission consistency: expected detection probability per unit,
        # recomputed here with the documented link formula, against the b
        # actually emitted by sample_world.
        th = w.theta
        O = w.observability[0]                                    # (T,)
        delta_t = DELTA_MAX * expit(
            th["delta_raw"][None, :] + th["delta_observability"][None, :] * O[:, None]
        )                                                         # (T, S)
        j_t = expit(th["j_raw"][None, :] + th["j_observability"][None, :] * O[:, None])
        rho_t = delta_t + (1.0 - delta_t) * j_t
        a = w.a[0, :T_HIST, :]                                    # (T, K)
        b = w.b[0]                                                # (T, K, S)
        p = np.where(a[:, :, None] == 1, rho_t[:, None, :], delta_t[:, None, :])
        for key, mask in (("delta_path", a == 0), ("rho_path", a == 1)):
            b_sel, p_sel = b[mask], p[mask]  # (n_units, S)
            emis[key]["obs"] += float(b_sel.sum())
            emis[key]["mean"] += float(p_sel.sum())
            emis[key]["var"] += float((p_sel * (1.0 - p_sel)).sum())

    out = {name: np.stack(vals) for name, vals in acc.items()}
    out["dwell_first"] = np.asarray(dwell_first, dtype=float)
    out["n_censored"] = n_censored
    out["a_first"] = np.asarray(a_first)
    out["emission_consistency"] = emis
    return out


@pytest.fixture(scope="module")
def stan_draws() -> dict:
    """Prior samples from synthetic_generator.stan generated quantities."""
    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=str(SYNTHETIC_GENERATOR_STAN))
    fit = model.sample(
        data={
            "L": L, "K": K, "S": S, "M": M,
            "delta_max": DELTA_MAX,
            "G": len(OBS_GRID), "obs_grid": OBS_GRID,
        },
        fixed_param=True,
        chains=1,
        iter_sampling=N_STAN_DRAWS,
        seed=STAN_SEED,
        show_progress=False,
        show_console=False,
    )
    names = [
        "Pi", "alpha_activity", "dwell_lambda",
        "delta_raw", "j_raw", "delta_observability", "j_observability",
        "kappa_plus", "kappa_minus", "init",
        "coder_common_mode_weight",
        "selection_alpha", "selection_observability", "selection_activity",
        "delta_link", "j_link", "rho_link",
        "dwell_draw", "a_draw",
    ]
    return {name: np.asarray(fit.stan_variable(name)) for name in names}


@pytest.fixture(scope="module")
def parity_table(py_draws, stan_draws) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (python_sample, stan_sample) for every compared scalar."""
    py, st = py_draws, stan_draws
    table: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for i in range(L):
        for j in range(L):
            if i != j:
                table[f"Pi[{i},{j}]"] = (py["Pi"][:, i, j], st["Pi"][:, i, j])
    for l in range(L):
        table[f"alpha_act_col1[{l}]"] = (
            py["alpha_activity"][:, l, 0], st["alpha_activity"][:, l, 0]
        )
        for k in range(1, K):
            table[f"alpha_act_rest[{l},{k}]"] = (
                py["alpha_activity"][:, l, k], st["alpha_activity"][:, l, k]
            )
        table[f"dwell_lambda[{l}]"] = (py["dwell_lambda"][:, l], st["dwell_lambda"][:, l])
        table[f"init[{l}]"] = (py["init"][:, l], st["init"][:, l])
    for s in range(S):
        for name in ("delta_raw", "j_raw", "delta_observability", "j_observability"):
            table[f"{name}[{s}]"] = (py[name][:, s], st[name][:, s])
    for m in range(M):
        table[f"kappa_plus[{m}]"] = (py["kappa_plus"][:, m], st["kappa_plus"][:, m])
        table[f"kappa_minus[{m}]"] = (py["kappa_minus"][:, m], st["kappa_minus"][:, m])
    table["coder_common_mode_weight"] = (
        py["coder_common_mode_weight"],
        st["coder_common_mode_weight"],
    )
    for name in ("selection_alpha", "selection_observability", "selection_activity"):
        table[name] = (py[name], st[name])

    # Detection links: Python side computed from Python raw prior draws with
    # the formula pinned to the real code path by test_emission_consistency.
    for s in range(S):
        for g, obs in enumerate(OBS_GRID):
            d_py = DELTA_MAX * expit(
                py["delta_raw"][:, s] + py["delta_observability"][:, s] * obs
            )
            j_py = expit(py["j_raw"][:, s] + py["j_observability"][:, s] * obs)
            r_py = d_py + (1.0 - d_py) * j_py
            table[f"delta_link[{s},O={obs}]"] = (d_py, st["delta_link"][:, s, g])
            table[f"j_link[{s},O={obs}]"] = (j_py, st["j_link"][:, s, g])
            table[f"rho_link[{s},O={obs}]"] = (r_py, st["rho_link"][:, s, g])

    # Shifted-Poisson dwell: first HSMM segment vs Stan's 1 + poisson_rng.
    table["dwell_first_segment"] = (py["dwell_first"], st["dwell_draw"].astype(float))

    return table


@pytest.fixture(scope="module")
def alpha(parity_table) -> float:
    """Bonferroni-corrected level across the whole battery (SBC convention)."""
    n_hypotheses = 3 * len(parity_table) + N_EXTRA_TESTS
    return SBC_KS_ALPHA / n_hypotheses


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare(name: str, py: np.ndarray, st: np.ndarray, alpha: float) -> list[str]:
    """KS + mean + variance comparison; returns failure descriptions."""
    failures = []
    ks = sp_stats.ks_2samp(py, st)
    if ks.pvalue < alpha:
        failures.append(f"{name}: KS p={ks.pvalue:.3e} < {alpha:.3e} (D={ks.statistic:.4f})")
    tt = sp_stats.ttest_ind(py, st, equal_var=False)
    if tt.pvalue < alpha:
        failures.append(
            f"{name}: mean p={tt.pvalue:.3e} < {alpha:.3e} "
            f"(py={py.mean():.4f}, stan={st.mean():.4f})"
        )
    lv = sp_stats.levene(py, st, center="median")
    if lv.pvalue < alpha:
        failures.append(
            f"{name}: variance p={lv.pvalue:.3e} < {alpha:.3e} "
            f"(py={py.var():.4f}, stan={st.var():.4f})"
        )
    return failures


def _run_family(parity_table, alpha, prefixes: tuple[str, ...]) -> None:
    names = [n for n in parity_table if n.startswith(prefixes)]
    assert names, f"no parity-table entries for prefixes {prefixes}"
    failures: list[str] = []
    for name in names:
        py, st = parity_table[name]
        failures.extend(_compare(name, py, st, alpha))
    assert not failures, (
        "Twin drift detected (Python generator vs Stan generator):\n  "
        + "\n  ".join(failures)
    )


def _two_proportion_pvalue(x1: int, n1: int, x2: int, n2: int) -> float:
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return float(2.0 * sp_stats.norm.sf(abs(z)))


# ---------------------------------------------------------------------------
# Battery — one family per tracker-named target, plus the full remainder
# ---------------------------------------------------------------------------

def test_pi_softmax_rows_parity(parity_table, alpha):
    """Pi off-diagonals: row-softmax of N(0,1) jump_raw, masked diagonal."""
    _run_family(parity_table, alpha, ("Pi[",))


def test_pi_rows_are_simplex(py_draws, stan_draws):
    """Structural: zero diagonal, rows sum to 1, on both sides."""
    for side, pi in (("python", py_draws["Pi"]), ("stan", stan_draws["Pi"])):
        diag = pi[:, np.arange(L), np.arange(L)]
        np.testing.assert_allclose(diag, 0.0, atol=1e-12, err_msg=f"{side} Pi diagonal")
        np.testing.assert_allclose(
            pi.sum(axis=2), 1.0, atol=1e-9, err_msg=f"{side} Pi row sums"
        )


def test_alpha_act_col1_order_statistics_parity(parity_table, alpha):
    """alpha_activity column 1: order statistics of L iid N(-2.0, 1.5)."""
    _run_family(parity_table, alpha, ("alpha_act_col1[",))


def test_alpha_act_rest_parity(parity_table, alpha):
    _run_family(parity_table, alpha, ("alpha_act_rest[",))


def test_dwell_lambda_prior_parity(parity_table, alpha):
    """dwell_lambda ~ lognormal(log 6, 0.5) on both sides."""
    _run_family(parity_table, alpha, ("dwell_lambda[",))


def test_dwell_shifted_poisson_parity(parity_table, alpha, py_draws):
    """First HSMM dwell segment vs Stan 1 + Poisson(lambda[z0])."""
    # Right-censoring at T + H = 100 months must be negligible; growth here
    # signals dwell-prior drift, not a tolerance to widen.
    assert py_draws["n_censored"] <= max(1, N_PY_WORLDS // 1000), (
        f"{py_draws['n_censored']} of {N_PY_WORLDS} first dwell segments were "
        f"right-censored at {T_HIST + H_FUTURE} months — investigate dwell prior"
    )
    _run_family(parity_table, alpha, ("dwell_first_segment",))


def test_delta_j_link_parity(parity_table, alpha):
    """delta/j/rho links on the observability grid, plus the raw vectors."""
    _run_family(
        parity_table, alpha,
        ("delta_raw[", "j_raw[", "delta_observability[", "j_observability[",
         "delta_link[", "j_link[", "rho_link["),
    )


def test_remaining_parameters_parity(parity_table, alpha):
    """kappa, init, selection — EVERY parameter is compared (tracker B-4)."""
    _run_family(
        parity_table, alpha,
        ("kappa_plus[", "kappa_minus[", "init[", "selection_"),
    )


def test_parity_table_is_exhaustive(parity_table):
    """Guard: every theta entry of the Python twin appears in the battery."""
    covered_prefixes = {
        "Pi": "Pi[", "alpha_activity": "alpha_act_col1[",
        "dwell_lambda": "dwell_lambda[",
        "delta_raw": "delta_raw[", "j_raw": "j_raw[",
        "delta_observability": "delta_observability[",
        "j_observability": "j_observability[",
            "kappa_plus": "kappa_plus[", "kappa_minus": "kappa_minus[",
            "coder_common_mode_weight": "coder_common_mode_weight",
            "init": "init[",
        "selection_alpha": "selection_alpha",
        "selection_observability": "selection_observability",
        "selection_activity": "selection_activity",
    }
    prior = HyperPrior(R=1, T=2, L=L, K=K, S=S, M=M, H=0)
    theta = sample_world(prior, seed=0).theta
    missing = [
        name for name in theta
        if name not in covered_prefixes
        or not any(t.startswith(covered_prefixes[name]) for t in parity_table)
    ]
    assert not missing, (
        f"theta parameters missing from the twin-parity battery: {missing} — "
        "extend the parity table; do not ship an uncompared parameter"
    )


def test_initial_activity_draw_parity(py_draws, stan_draws, alpha):
    """P(a=1) for tactic 1 at the initial regime: init -> z0 -> Bernoulli."""
    x_py, n_py = int(py_draws["a_first"].sum()), py_draws["a_first"].size
    a_st = stan_draws["a_draw"]
    x_st, n_st = int(a_st.sum()), a_st.size
    pvalue = _two_proportion_pvalue(x_py, n_py, x_st, n_st)
    assert pvalue >= alpha, (
        f"initial activity rate drift: python {x_py}/{n_py}={x_py / n_py:.4f} vs "
        f"stan {x_st}/{n_st}={x_st / n_st:.4f} (p={pvalue:.3e} < {alpha:.3e})"
    )


def test_emission_consistency(py_draws, alpha):
    """The link formula used in this battery matches sample_world's actual
    emission path: empirical b rates vs expected delta (a=0) and rho (a=1)."""
    for key in ("delta_path", "rho_path"):
        e = py_draws["emission_consistency"][key]
        se = np.sqrt(e["var"])
        z = 0.0 if se == 0.0 else (e["obs"] - e["mean"]) / se
        pvalue = float(2.0 * sp_stats.norm.sf(abs(z)))
        assert pvalue >= alpha, (
            f"emission {key}: observed detections {e['obs']:.0f} vs expected "
            f"{e['mean']:.1f} (z={z:.2f}, p={pvalue:.3e} < {alpha:.3e}) — the "
            "link formula in synthetic_world.py has drifted from this battery"
        )
