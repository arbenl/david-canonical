# Implementation Prompt for Claude Sonnet 4.6 — DAVID/M0.1 Canonical Layer, Milestone 1

> Paste everything below the rule into Sonnet 4.6 (Claude Code, Cowork, or a fresh
> chat with workspace access to `/Users/arbenlila/development/david/`). The prompt
> is self-contained: it does not depend on prior conversation memory.

---

You are the implementing engineer for DAVID/M0.1, a Bayesian Hidden Semi-Markov
measurement and 6-12 month forecast engine for tobacco-industry interference
tactic activity under partial observability. Arben Lila is the chief architect.
You are the implementer. The mathematics is settled; your job is to make the
canonical scaffold actually run end-to-end on synthetic data, in fail-closed
discipline, without expanding scope.

## What already exists

Repository root: `/Users/arbenlila/development/david/`

Two trees matter for this task:

1. `council_m01/` — the existing measurement-layer infrastructure. Contains:
   - `simulation/full_integrated_m01.stan` — full Stan HSMM with pi_ii = 0,
     right-censored final segment, Dawid-Skene coder layer, selection model.
     DO NOT modify this file.
   - `simulation/full_integrated_fit_runner.py`,
     `simulation/full_integrated_fit_contract.py` — fit runner and gates
     (R-hat, ESS, divergences). Import these; do not duplicate.
   - `theorem_packets/{A_prime,B_prime,C,...}.md` — the proof packets.
     Read but do not edit.

2. `canonical/` — the new layer you will implement against. Contains:
   - `docs/ARCHITECTURE.md` — read this first; it is the single source of truth
     for what this layer does.
   - `docs/THEOREMS.md` — index of mathematical claims this layer enforces.
   - `docs/AUTOMATION_CONTRACT.md` — what is automated vs. human-required.
   - `david/` — Python package (CLI, engine, ingest, model, simulator,
     validation, theorems, routing). Most files are skeletons with explicit
     TODOs or `NotImplementedError`. The math kernels in
     `david/theorems/` are complete and tested.
   - `stan/m01_forward.stan` — extends council_m01's Stan model with a
     `generated quantities` block for H-step forecasts. Contains TWO clearly
     marked TODOs (the likelihood loop body and the per-series emit
     reconstruction in generated quantities), which must be copied verbatim
     from `council_m01/simulation/full_integrated_m01.stan`.
   - `stan/coder_calibration.stan` — complete, no edits needed.
   - `tests/` — pytest scaffolds; some pass already, more must follow.

## Your milestone (Milestone 1): synthetic end-to-end

The goal of this session is a single thing:

> Make `david sbc` and `david sbc --forecast` run to completion on synthetic
> data, with the measurement SBC passing KS uniformity at alpha = 0.05 and the
> forecast SBC achieving nominal-80 and nominal-95 coverage inside the
> pre-registered bands.

No real evidence. No scrapers. No LLM coders. No adjudicator. Synthetic only.
The point is to prove the entire mathematical chain — generative model, Stan
fit, posterior extraction, forecast block, gate computation, coverage check —
end-to-end on data the model itself generated.

If this milestone passes, every downstream task (real ingestion, real fit,
real forecast) builds on a validated foundation. If it fails, everything else
is built on sand.

## Concrete deliverables for Milestone 1

You will edit or create exactly these files. Do not touch anything else.

### D1. Complete `stan/m01_forward.stan`

Open `canonical/stan/m01_forward.stan` and replace the two TODO regions:

- In the `model` block: copy the per-series likelihood construction from
  `council_m01/simulation/full_integrated_m01.stan` (the loop that builds
  `emit` and adds `hsmm_right_censored_lpdf(emit | log_init, log_jump,
  dwell_lambda)` to target). Use the same selection model, the same coder
  layer, the same observability functions.
- In `generated quantities`: rebuild `emit` per series identically so the
  forward filter has the right inputs. Use the helper `terminal_regime_posterior`
  already declared. Sample `z_now` from that posterior and step forward
  `H_forecast` months under the HSMM, emitting `z_future`, `a_future`,
  `a_future_draw`.

Acceptance:
- `cmdstan` compiles the file without errors.
- `cmdstanpy` can sample on a synthetic data dict for at least 2 chains x 200
  warmup x 200 sampling. Divergences allowed during development; tighten
  later.

### D2. Wire `david/model/fit.py::assemble_fit_data` for synthetic input

Add a function `assemble_fit_data_from_synthetic(world: WorldDraw, horizon: int)`
that converts a `WorldDraw` from `david/simulator/synthetic_world.py` into the
Stan data dict expected by `m01_forward.stan`. The mapping is direct:
- R, T, L, K, S, M = prior dimensions
- U = R * T * K (unit = (series, time, tactic) triple)
- `unit_series`, `unit_time`, `unit_tactic` are the flattened indices
- `selected` is the (R, T) selected indicator broadcast to U
- `observability` is the (R, T) observability broadcast to U
- For `N_label`, `label_unit`, `label_source`, `label_coder`, `y`:
  flatten the (R, T, K, S, M) `world.y` tensor into one long list of
  observed coder labels with the right unit/source/coder indices.
- `delta_max` = 0.30
- `H_forecast` = horizon

Acceptance:
- Returns a dict whose shapes match what Stan expects.
- `assemble_fit_data` (the existing function) is left as a `NotImplementedError`
  for now; the synthetic adapter is separate.

### D3. Implement `simulator.sbc.fit_measurement_layer`

Replace the `NotImplementedError` in `david/simulator/sbc.py::fit_measurement_layer`
so that it:
1. Calls `assemble_fit_data_from_synthetic(world, horizon=0)`.
2. Compiles `stan/m01_forward.stan` via cmdstanpy (cache the compiled
   model across calls for speed).
3. Samples with 2 chains x 200 warmup x 200 sampling, seed = world index.
4. Returns `{"n_draws": int, "draws": {param_name: np.ndarray, ...}}`
   keyed by the same names that `flatten_params(world.theta)` produces.

Acceptance:
- `david sbc --n-worlds 5` (you may need to add this option) runs to
  completion on 5 worlds and emits `data/fits/sbc/sbc_summary.json` with
  a `per_parameter_ks` block and `gate_status` (may legitimately fail at
  N=5 due to undersampling; that is fine).

### D4. Implement `simulator.forecast_sbc.fit_and_forecast`

Replace the `NotImplementedError` in `david/simulator/forecast_sbc.py::fit_and_forecast`
so that it:
1. Calls `assemble_fit_data_from_synthetic(world, horizon=world.theta_prior.H)`.
2. Samples as above.
3. Pulls the `a_future` posterior draws and reshapes to (R, H, K, D).
4. Returns `{"a_future": np.ndarray}`.

Acceptance:
- `david sbc --forecast --n-worlds 5` runs to completion and writes
  `data/fits/forecast_sbc/forecast_sbc_summary.json` with `coverage_80`,
  `coverage_95`, `pass_80`, `pass_95`.

### D5. Add `--n-worlds` option to the CLI

In `david/cli.py`, the `sbc` command already accepts `n_worlds`. Confirm it
is passed through to both `run_sbc` and `run_forecast_sbc`. No other CLI
changes for this milestone.

### D6. Tests

Add three tests:

1. `tests/test_assemble_fit_data_synthetic.py` — round-trip a synthetic world
   through the assembler and check shapes.
2. `tests/test_m01_forward_compiles.py` — calls `cmdstanpy.CmdStanModel(...)`
   on `stan/m01_forward.stan`; passes if compile succeeds.
3. `tests/test_smoke_sbc.py` — runs 3-world SBC end-to-end and checks the
   `gate_status` field is set (pass or fail, just present).

Acceptance:
- `pytest canonical/tests/` exits 0 (existing 4 test files plus your 3 = 7).

## How to verify Milestone 1 is done

Run, in order:

```bash
cd /Users/arbenlila/development/david/canonical
pip install -e .
pytest tests/
david sbc --n-worlds 50         # 5-15 min on a laptop
david sbc --forecast --n-worlds 50
cat data/fits/sbc/sbc_summary.json | jq '.gate_status'
cat data/fits/forecast_sbc/forecast_sbc_summary.json | jq '.gate_status'
```

Milestone passes when both `gate_status` are `"pass"` at N=50 worlds, OR you
report a precise mathematical reason for any KS / coverage failure (e.g., a
prior is too wide, a parameter is degenerate, label switching is present).

If a parameter consistently fails KS, your job is NOT to relax the gate; it is
to investigate (label switching is the most likely cause — fix it via order
constraints or post-hoc relabeling, then re-run).

## Discipline rules — non-negotiable

These rules carry over from council_m01 and apply to every line you write:

1. **Fail closed.** Every function returns a typed result with `gate_status`.
   Never silently return junk. Never swallow exceptions. Never relax a gate
   threshold to make a test pass.

2. **No scope creep.** Real ingestion, scrapers, LLM coders, adjudicator UI,
   API endpoints, route promotion — all of these are NOT this milestone.
   If you find yourself opening files outside `canonical/stan/`,
   `canonical/david/model/`, `canonical/david/simulator/`, or
   `canonical/tests/`, stop and ask why.

3. **Math kernels are settled.** Do not edit
   `canonical/david/theorems/{A_prime, B_prime, C_renamed, D_forecast_horizon}.py`.
   They have been reviewed and tested. If you find a bug, report it; do not
   patch silently.

4. **Council_m01 is read-only.** You may IMPORT from it. You may not edit it.
   If you need a function that does not exist there, add it to `canonical/`,
   not to council_m01.

5. **Pre-registration discipline.** All thresholds live in
   `canonical/david/config.py`. Do not hard-code thresholds elsewhere. Do not
   change values in `config.py` without an explicit, logged reason.

6. **One commit per deliverable.** D1..D6 are six commits, each with the
   acceptance criterion verifiable from the message. Squash later if needed,
   but during the session keep them separate.

## What to report when you stop

At the end of the session, write `canonical/docs/MILESTONE_1_REPORT.md`
containing:

- What you completed (link D1..D6 to commits).
- The verification commands you ran and their outputs.
- The contents of both `gate_status` fields.
- If any deliverable was not finished, the specific blocker.
- If you discovered a math or design issue, a short note pointing to the
  relevant ARCHITECTURE.md section and proposing the next milestone's work.

Do not promote any forecast to "headline" route. Do not run real ingestion.
Do not serve the API. This milestone is a synthetic correctness proof and
nothing more.

## Reference reading order (do this first, 30 minutes)

1. `canonical/README.md` — orientation.
2. `canonical/docs/ARCHITECTURE.md` — full design.
3. `canonical/docs/THEOREMS.md` — what the math kernels enforce.
4. `council_m01/simulation/full_integrated_m01.stan` — the Stan model you
   will partially copy from.
5. `canonical/david/simulator/synthetic_world.py` — the generative twin.
6. `canonical/david/config.py` — thresholds.

Then start with D1.

You have everything you need. Go.
