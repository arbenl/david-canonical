# Context
You are working on the DAVID M0.1 project. Before taking any action or writing any code, you MUST review:
1. `AGENTS.md` (The DAVID Agent Constitution)
2. `.gemini/skills/david-slice-lifecycle/SKILL.md` (The execution SOP)

We are working in strict, mathematically isolated slices. Your task is to implement item **C-2** from `docs/M0.1_TRACKER.md`.

# Task: C-2 (Claim-eligible family definition)
Currently, in `david/engine/router.py`, the Posterior Expected FDP calculation (Theorem C) might be inappropriately consuming the $p\_active$ vector from cells that have failed prior measurement and horizon gates. FDP selection must ONLY run over cells that survive FG1–FG5 (i.e., those assigned the `headline` route).

Your task is to:
1. Examine `apply_forecast_routing` in `david/engine/router.py`.
2. Ensure that the `p_hat` array passed to `compute_posterior_fdp_threshold` strictly excludes any cell that was routed to `evidence_gap`, `withhold`, `prior_dominated`, or `horizon_prior_dominated`.
3. Add a unit test to `tests/test_router.py` (create it if it doesn't exist) that explicitly verifies this. For example, mock a set of cells where one is `evidence_gap`, and ensure its $p\_active$ value does NOT enter the vector passed to the Theorem C kernel.
4. Ensure the `route_ledger` counts the size of the claim-eligible family (`M`) accurately.

# Constraints & Workflow
- Do NOT alter the math kernels in `C_renamed.py`. This is strictly a router ordering/filtering fix.
- Create a new branch `feat/c2-claim-eligible-family` and adhere to the `david-slice-lifecycle` SOP.
- When the tests are green, update the `docs/M0.1_TRACKER.md` row for C-2 to **DONE**.
- Verify your changes by running `./.venv/bin/pytest tests/`.
