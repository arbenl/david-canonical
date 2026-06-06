"""D6 test: m01_forward.stan compiles without errors via cmdstanpy."""

from __future__ import annotations

from cmdstanpy import CmdStanModel

from david.config import M01_FORWARD_STAN


def test_m01_forward_compiles():
    """Stan model compiles and exe file is produced."""
    model = CmdStanModel(stan_file=str(M01_FORWARD_STAN))
    assert model.exe_file is not None
    assert str(M01_FORWARD_STAN).replace(".stan", "") in model.exe_file
