"""Mathematical kernels for DAVID/M0.1. One module per theorem.

A_prime.py        Theorem A' practical identification distance
B_prime.py        Theorem B' channel informativeness I(O)
C_renamed.py      Theorem C posterior expected FDP control
D_forecast_horizon.py  Theorem D-forecast horizon-validity bound h*

Each module exposes (a) the math kernel as a pure function of posterior draws
and (b) a gate function that returns {gate_status, reason, diagnostic} usable
by the routing layer.
"""
