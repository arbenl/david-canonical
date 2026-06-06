"""D6 test: round-trip a synthetic world through assemble_fit_data_from_synthetic
and verify shapes match the Stan data block contract."""

from __future__ import annotations

import pytest
from david.simulator.synthetic_world import HyperPrior, sample_world
from david.model.fit import assemble_fit_data_from_synthetic


@pytest.fixture(scope="module")
def tiny_world():
    prior = HyperPrior(R=2, T=8, L=3, K=4, S=2, M=2, H=4)
    return sample_world(prior, seed=0), prior


def test_scalar_dimensions(tiny_world):
    world, prior = tiny_world
    d = assemble_fit_data_from_synthetic(world, horizon=prior.H)
    assert d["R"] == prior.R
    assert d["T"] == prior.T
    assert d["L"] == prior.L
    assert d["K"] == prior.K
    assert d["S"] == prior.S
    assert d["M"] == prior.M
    assert d["H_forecast"] == prior.H


def test_unit_count(tiny_world):
    world, prior = tiny_world
    d = assemble_fit_data_from_synthetic(world, horizon=prior.H)
    U_expected = prior.R * prior.T * prior.K
    assert d["U"] == U_expected
    assert len(d["unit_series"]) == U_expected
    assert len(d["unit_time"]) == U_expected
    assert len(d["unit_tactic"]) == U_expected
    assert len(d["selected"]) == U_expected
    assert len(d["observability"]) == U_expected


def test_label_count(tiny_world):
    world, prior = tiny_world
    d = assemble_fit_data_from_synthetic(world, horizon=prior.H)
    N_expected = prior.R * prior.T * prior.K * prior.S * prior.M
    assert d["N_label"] == N_expected
    assert len(d["label_unit"]) == N_expected
    assert len(d["label_source"]) == N_expected
    assert len(d["label_coder"]) == N_expected
    assert len(d["y"]) == N_expected


def test_index_bounds(tiny_world):
    world, prior = tiny_world
    d = assemble_fit_data_from_synthetic(world, horizon=prior.H)
    U = d["U"]
    assert min(d["unit_series"]) >= 1 and max(d["unit_series"]) <= prior.R
    assert min(d["unit_time"]) >= 1 and max(d["unit_time"]) <= prior.T
    assert min(d["unit_tactic"]) >= 1 and max(d["unit_tactic"]) <= prior.K
    assert min(d["label_unit"]) >= 1 and max(d["label_unit"]) <= U
    assert min(d["label_source"]) >= 1 and max(d["label_source"]) <= prior.S
    assert min(d["label_coder"]) >= 1 and max(d["label_coder"]) <= prior.M
    assert all(v in (0, 1) for v in d["y"])
    assert all(v in (0, 1) for v in d["selected"])


def test_horizon_zero_clamped_to_one(tiny_world):
    world, _ = tiny_world
    d = assemble_fit_data_from_synthetic(world, horizon=0)
    assert d["H_forecast"] == 1
