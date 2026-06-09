# DAVID/M0.1 — Known Gate Behaviours

This document records gate outcomes that are **expected and correct** under the current
data regime. It exists so reviewers do not misinterpret a recorded failure as a system
defect.

---

## F13 — Forecast Horizon Respect (EXPECTED FAIL at h > h*)

### What F13 tests
F13 checks that the median forecast probability does not drift by more than 2 pp (0.02)
beyond the theoretical horizon `h*` computed by Theorem D′. If cells beyond h* show
drift > 0.02, the gate fails.

### Why F13 fails in the current regime
- Theorem D′ sets `h* = 5 months` given the current posterior.
- The pipeline emits forecasts at h = 3, 6, 9, 12 months.
- Cells at h = 6 have `forecast_route = horizon_prior_dominated` because h > h*.
- F13 computes drift **across all emitted cells including those beyond h***.
- The measured max_diff is ~0.07 (well above the 0.02 threshold), which reflects
  genuine prior domination — not a model bug.

### Why this is correct behaviour (FAIL_CLOSED discipline)
The Automation Contract states: *"Never relax a gate threshold to make a test pass."*
F13 **correctly detects** that the model is prior-dominated beyond 5 months and
**correctly routes** 63 cells to `horizon_prior_dominated`. The routing layer is working
exactly as designed. The failure is meaningful information, not noise.

### What to expect
```
F13: gate_status = "fail"
     statistic   = 0.0739
     threshold   = 0.02
     reason      = "max_diff_beyond_h_star=0.0739"
```
The `falsification_ledger.json` records this as `gate_status: "fail"` at the
`battery_result` level. The `route_ledger.json` records 63 cells as
`horizon_prior_dominated`. Both are correct.

### When would F13 pass?
F13 will pass once either:
1. Enough real evidence accumulates that h* increases beyond 12 months (posterior
   sharpens and prior no longer dominates at h = 6).
2. The forecast horizon is restricted to h ≤ h* (emit only h = 3 until h* ≥ 6).

---

## F3, F4, F5, F6, F11, F15 — SKIP (no held-out data yet)

These tests require held-out labels or known ground truth for activity:

| Test | Requirement | Status |
|------|-------------|--------|
| F3   | Held-out test-set labels | Skip — no held-out set |
| F4   | Ground-truth activity indicator | Skip — unknown in real-data mode |
| F5   | Held-out set | Skip — no held-out set |
| F6   | Held-out set | Skip — no held-out set |
| F11  | Held-out labels | Skip — no held-out set |
| F15  | Held-out labels | Skip — no held-out set |

These will automatically activate once a prospective hold-out period is defined
(e.g. use data before 2025-01-01 for fitting, 2025 onward for validation).

---

## Automation Contract Reminder

> "If any F-test fails, the affected cells route to `withhold` — unless the failure
> is F13 (horizon respect), in which case routing to `horizon_prior_dominated` is
> the correct non-withholding response."

The routing layer implements this correctly. Do not override gate failures at the
routing level without updating this document and re-running the full falsification
battery.
