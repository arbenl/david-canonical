# DAVID/M0.1 Copilot Instructions

DAVID/M0.1 is a measurement-first Bayesian engine. It is not a generative
assistant. Every emitted result must be a deterministic route plus typed payload
and binding reasons from the gate ledger.

Repository invariants:
- Do not introduce free-form public conclusions, causal claims, policy warnings,
  or legal accusations.
- Preserve the causal firewall: `policy_warning`, `legal_accusation`, and
  `causal_claim` remain `withhold` unless an audited causal module exists.
- Never relax thresholds, priors, grids, abstention logic, Stan semantics, or
  routing rules to make a test pass.
- `Gobs` and evidence-derived network features must not feed core `A` or `Z`
  inference, feature construction, eligibility filtering, ranking, or routing.
- The UI is a renderer only. Do not compute gates, thresholds, routes, or model
  decisions client-side.

Expected validation for nontrivial changes:
- Python deterministic suite: `uv run pytest tests -q`
- UI package: `cd david-ui && npm run build`
- JSON configuration: parse every `config/*.json`
- Gate/runtime changes: add focused tests and keep fail-closed behavior.
- Stan/model/simulator changes invalidate prior SBC evidence; run or record the
  blocked SBC/twin-parity validation before claiming readiness.

Review discipline:
- Treat SonarQube and Copilot PR review findings as pre-merge blockers unless a
  human owner records a non-blocking exception.
- Findings should cite exact files and lines and distinguish deterministic bugs
  from expected fail-closed gate outcomes.

