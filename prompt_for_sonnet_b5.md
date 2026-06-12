# Context
You are working on the DAVID M0.1 project. Before taking any action or writing any code, you MUST review:
1. `AGENTS.md` (The DAVID Agent Constitution)
2. `.gemini/skills/david-slice-lifecycle/SKILL.md` (The execution SOP)

We are working slice-by-slice. Your task is to implement item **B-5** from `docs/M0.1_TRACKER.md`. 

# Task: B-5 (Shared Stan functions via `#include`)
Now that B-4 has established the twin-parity test battery between the Python generator and `stan/synthetic_generator.stan`, there is one remaining structural vulnerability for Simulation-Based Calibration (SBC): if `m01_forward.stan` and `synthetic_generator.stan` share identical copy-pasted implementations of the core HSMM logic, an error in that logic will be invisible to SBC.

Your task is to establish a single source of truth for the HSMM math:
1. Extract the `hsmm_alpha` and `hsmm_right_censored_lpdf` function definitions from `stan/m01_forward.stan`.
2. Move them into a new file: `stan/functions/hsmm.stanfunctions`.
3. Use Stan's `#include` directive (`#include functions/hsmm.stanfunctions`) to include these functions in both `stan/m01_forward.stan` and `stan/synthetic_generator.stan`.
4. Remove the redundant function implementations from both `.stan` files so they strictly rely on the shared include.

# Constraints & Workflow
- Do not alter the mathematical logic inside `hsmm_alpha` or `hsmm_right_censored_lpdf`. This is purely an architectural refactor.
- Ensure that both models still compile perfectly.
- Create a new branch `chore/b5-shared-stan-functions` and adhere to the `david-slice-lifecycle` SOP for branch management.
- Update the `docs/M0.1_TRACKER.md` row for B-5 to **DONE** when complete.
- Verify everything works by running `./.venv/bin/pytest tests/` and ensuring `test_m01_forward_compiles.py` and `test_twin_parity.py` pass.
