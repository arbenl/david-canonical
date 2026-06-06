"""D6 test: 3-world SBC runs end-to-end and gate_status field is present.

The gate may legitimately fail at N=3 due to undersampling; the test only
checks that the machinery runs to completion and the result is typed correctly.
"""

from __future__ import annotations

import pytest
from david.simulator.synthetic_world import HyperPrior
from david.simulator.sbc import run_sbc


@pytest.fixture(scope="module")
def sbc_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("sbc_smoke")
    prior = HyperPrior(R=2, T=8, L=3, K=3, S=2, M=2, H=0)
    return run_sbc(n_worlds=3, prior=prior, out_dir=out)


def test_gate_status_present(sbc_result):
    assert "gate_status" in sbc_result
    assert sbc_result["gate_status"] in ("pass", "fail")


def test_summary_path_exists(sbc_result):
    from pathlib import Path
    assert Path(sbc_result["summary_path"]).exists()


def test_per_parameter_ks_nonempty(sbc_result):
    ks = sbc_result.get("per_parameter_ks", {})
    assert len(ks) > 0, "No KS results — draws dict was empty"


def test_n_worlds_recorded(sbc_result):
    assert sbc_result["n_worlds"] == 3
