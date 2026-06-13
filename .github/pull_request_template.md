## Summary

<!-- What changed and why. Keep claim language bounded by the gate evidence. -->

## David Classification

- Class:
- Tier:
- Touched surfaces:
- Model-version decision required: yes/no

## Validation

- [ ] `uv run pytest tests -q`
- [ ] `cd david-ui && npm run build` when UI/package files changed
- [ ] JSON config validated when `config/*.json` changed
- [ ] Stan syntax/compile or blocked Stan gate recorded when `stan/*.stan` changed
- [ ] SBC/twin-parity/falsification rerun or explicit blocked note when model,
      simulator, theorem, or Stan kernels changed

## Review Gates

- [ ] SonarQube quality gate passed, or a recorded exception exists
- [ ] CodeQL completed without unresolved actionable findings
- [ ] GitHub Copilot PR review requested for substantive changes
- [ ] Copilot/Sonar actionable findings resolved before merge
- [ ] Council review disposition recorded when Tier 2/3 gate-runtime,
      simulation, theorem, causal, legal, or public-claim surfaces changed

## Claim Boundary

- [ ] No causal, legal, or policy-warning claim was added
- [ ] No route emits more than the pre-registered gates license
- [ ] UI changes render ledger fields only and do not recompute gates client-side

