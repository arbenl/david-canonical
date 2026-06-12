# Context
You are working on the DAVID M0.1 project. Before taking any action or writing any code, you MUST review:
1. `AGENTS.md` (The DAVID Agent Constitution)
2. `.gemini/skills/david-slice-lifecycle/SKILL.md` (The execution SOP)

We are working in strict, mathematically isolated slices. Your task is to implement item **C-3** from `docs/M0.1_TRACKER.md`.

# Task: C-3 (Sensitivity-envelope FDP)
Currently, Theorem C uses a naive (single-theta) rule. We need to implement the Sensitivity-envelope FDP logic to guarantee that our False Discovery Proportion (FDP) bounds the risk at the worst grid point.

Your task is to:
1. Update `david/theorems/C_renamed.py` to compute `p_i^- = min_θ p_i^θ` over the pre-registered grid `Θ^meas`.
2. The grid definition should be imported from or defined in `david/engine/observability_sensitivity.py`.
3. Sort and perform the prefix scan on this conservative `p^-` array instead of the single-point estimates. This restores monotonicity and guarantees `FDP_env(R) ≤ q_d` at the worst grid point.
4. Add a test in `tests/test_posterior_fdp.py` with a mocked 3-point sensitivity grid where the naive (single-θ) rule flags a set that violates the envelope, but your new conservative rule correctly shrinks the set.

# Constraints & Workflow
- Obey all mathematical invariants in `AGENTS.md` (especially I-4: do not relax any math).
- Create a new branch `feat/c3-sensitivity-envelope` and adhere to the `david-slice-lifecycle` SOP for PR handling.
- Verify everything works by running `./.venv/bin/pytest tests/test_posterior_fdp.py`.
- Update the `docs/M0.1_TRACKER.md` row for C-3 to **DONE** when tests pass.
