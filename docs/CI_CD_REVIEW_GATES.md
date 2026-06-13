# CI/CD and Review Gates

This repository uses GitHub Actions for deterministic checks and GitHub/Sonar
review gates for merge readiness.

## Required Workflows

- `CI / Python deterministic gates`: installs the locked Python environment,
  ensures CmdStan is available, validates JSON config, and runs
  `uv run pytest tests -q`.
- `CI / Next.js build`: installs `david-ui` with `npm ci` and runs
  `npm run build`.
- `CodeQL`: scans Python and JavaScript/TypeScript with
  `security-extended` and `security-and-quality` queries when GitHub code
  scanning is enabled and repository variable `CODEQL_ENABLED=true` is set.
  Until then, the workflow reports a successful skip and SonarQube Cloud is the
  required SAST/security gate.
- `SonarQube`: runs only after these repository settings are configured:
  - secret `SONAR_TOKEN`
  - variable `SONAR_PROJECT_KEY`
  - variable `SONAR_ORGANIZATION`

Make the Python, UI, and Sonar jobs required branch-protection checks once
Sonar is configured. Add CodeQL as required only after GitHub code scanning is
enabled for the private repository.

## Advisory Checks

`ruff` and `mypy` currently run as advisory checks because the existing tree has
known lint/type debt. New PRs should not add new findings. Promote these checks
to required after the baseline is clean.

## Copilot Review Gate

For substantive PRs, request GitHub Copilot PR review after deterministic CI is
green. Resolve actionable Copilot findings before merge. If Copilot cannot run,
record a blocked disposition in the PR and in the relevant council review record
for Tier 2/3 work.

## Sonar Gate

Sonar must fail the workflow when the quality gate fails. The workflow passes
`sonar.qualitygate.wait=true` and should be configured as a required status
check. Do not merge with unresolved Sonar vulnerabilities, bugs, or security
hotspots unless the owner records an explicit non-blocking exception.

## David-Specific Merge Rule

No CI or review tool may broaden the engine's claim boundary. A green pipeline
only means the checked artifacts passed the configured gates; it does not prove
the model, authorize causal claims, or authorize legal/policy accusations.
