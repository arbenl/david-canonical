# F-3 Human Math Review Marker Escalation

Date: 2026-06-13

Scope: `docs/thesis_mathematical_core.tex`

Tracker item: F-3, "Re-verify `%%NEEDS_HUMAN_MATH_REVIEW%%` markers: author sign-off pass over thesis_mathematical_core.tex audit edits"

## Disposition

All 125 active `%%NEEDS_HUMAN_MATH_REVIEW%%` markers in `docs/thesis_mathematical_core.tex` were re-inventoried and escalated for author or external mathematical review.

The markers are intentionally retained in the TeX source. Codex did not remove them or treat them as signed off, because the required action is mathematical author review, not mechanical cleanup.

## Escalation Policy

Each marker remains blocking for publication-strength mathematical sign-off until one of these actions is recorded:

- `resolved`: author or qualified math reviewer signs off the statement as written and removes the marker.
- `revised`: author or qualified math reviewer changes the statement/proof and removes or moves the marker.
- `deferred`: author explicitly accepts the residual limitation and records why the marker may remain for M0.1.

No runtime or public-claim readiness should cite these marked theorem passages as externally certified while markers remain.

## Marker Inventory

| Section | Count | Line range | Disposition |
|---|---:|---:|---|
| Preamble / abstract | 1 | 43-43 | Escalated |
| Theorem A': Setting and Measurement Architecture | 7 | 68-87 | Escalated |
| Theorem A': Formal Statement | 17 | 91-581 | Escalated |
| Theorem A': Step-by-Step Proof | 16 | 114-612 | Escalated |
| Degrees-of-Freedom Heuristic | 1 | 156-156 | Escalated |
| S=2 Sources | 1 | 166-166 | Escalated |
| S=3 Sources | 5 | 169-175 | Escalated |
| Operational Diagnostic | 27 | 180-698 | Escalated |
| Theorem B': Setting | 8 | 237-543 | Escalated |
| Theorem B'.1 proof | 4 | 298-304 | Escalated |
| Theorem B'.2 proof | 4 | 350-354 | Escalated |
| Quadratic Collapse | 1 | 363-363 | Escalated |
| Theorem C: Setting and Scope | 3 | 454-457 | Escalated |
| Challenge A': Conditional Source Dependence | 6 | 709-725 | Escalated |
| Challenge B': Temporal Correlation | 9 | 734-749 | Escalated |
| Challenge B': Conditionally Dependent Coders | 5 | 754-764 | Escalated |
| Challenge C: FDP Exceedance | 1 | 789-789 | Escalated |
| Challenge D: Non-Stationarity / Breaks | 7 | 798-812 | Escalated |
| System Integration / Calibration | 2 | 819-828 | Escalated |

## High-Risk Review Clusters

These clusters should be prioritized in author review:

- Theorem A' exact identifiability and every-point versus generic language.
- The production-likelihood limitation: observability links, selection, and coder layer are operationally validated rather than theorem-proven.
- Theorem B' dependence-adjusted N_eff and the distinction between marginal-score Godambe information and full HSMM Fisher information.
- Theorem C expectation-FDP guarantee versus realized exceedance control.
- Theorem D forecast horizon diagnostic, including lattice renewal assumptions, first-crossing h*, and structural-break handling.
- Final system guarantee language, which must remain "eligible under pre-registered gates" rather than "mathematically certified."

## Verification

Inventory command:

```bash
rg -n "%%NEEDS_HUMAN_MATH_REVIEW%%" docs/thesis_mathematical_core.tex
```

Result: 125 markers.

