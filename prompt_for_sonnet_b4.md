# Context
You are working on the DAVID M0.1 project. Please read `AGENTS.md` and `.gemini/skills/david-slice-lifecycle/SKILL.md` before doing anything. 
We are working slice-by-slice, and your task is to implement item **B-4** from `docs/M0.1_TRACKER.md`.

# Task: B-4 (Twin-parity test battery)
Our Simulation-Based Calibration (SBC) depends entirely on twin parity. If the Python generator in `david/simulator/synthetic_world.py` and the Stan generator in `stan/synthetic_generator.stan` drift, SBC becomes invalid.

Your task is to implement a strict moment-matching and distribution-matching test suite:
1. Create `tests/test_twin_parity.py`.
2. The test must sample priors from the Python `synthetic_world.py` generator.
3. The test must compile and sample from the Stan `synthetic_generator.stan` `generated quantities` block.
4. Perform per-parameter Kolmogorov-Smirnov (KS) tests and moment matching (mean/variance) across EVERY parameter:
   - `Pi` (softmax rows of log_jump)
   - `alpha_activity` (specifically column 1 order statistics)
   - `dwell_lambda` (shifted-Poisson: 1 + Pois(lambda))
   - `delta_raw` and `j_raw` links
5. Ensure that `pytest` fails if there is any statistical drift between the two generators.

# Constraints
- Obey all mathematical invariants in `AGENTS.md`. No thresholds may be relaxed to make the test pass.
- Execute this task on a new branch following the `david-slice-lifecycle` SOP (e.g. `feat/b4-twin-parity-tests`).
- Update the `docs/M0.1_TRACKER.md` row for B-4 to DONE when complete.
- Verify your tests pass by running `./.venv/bin/pytest tests/test_twin_parity.py`.
