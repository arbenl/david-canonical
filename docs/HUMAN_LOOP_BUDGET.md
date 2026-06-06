# Human-Loop Budget

Target: ≤ 4 hours adjudicator + ≤ 1 hour reviewer per 90-day cycle.

## What competes for adjudicator time

1. Items where ≥ 2 LLM coders disagree (priority).
2. Items sampled for gold-standard expansion (5% baseline rate, deterministic).
3. Items in strata below A' replication floor (active sampling).

Items 1, 2, 3 above are ranked by Expected Information Gain (EIG). The queue
is truncated to fit the budget. Truncated-out items are flagged
`llm_only_acceptable` and flow into the main fit using LLM labels only; the
coder-reliability layer absorbs the LLM error.

## Why this matters

Adjudicator hours scale as O(N_evidence) without budget enforcement. The
DAVID/M0.1 spec requires at least 800 evidence items for Theorem A' to be
operationally identified at the prototype scale. Manual review of all 800 is
not feasible. EIG ranking + budget cap ensures the highest-value items get
attention while the rest are still useful via Dawid-Skene.

## How to set the budget

The budget is set in `david/config.py::ADJUDICATOR_HOURS_PER_CYCLE`. Default
4.0. If adjudicator capacity changes seasonally, adjust per cycle. The change
is logged in `data/adjudicator_queue.json` so reviewers can see the budget
that produced any queue.

## Failure mode

If LLM-only items dominate the dataset (e.g., > 80%), the coder-reliability
posterior for human coders narrows and the kappa estimates for LLM coders
widen. The fit will still converge but with broader uncertainty. The
falsification battery F4 (discrimination) is the first signal that quality
has degraded; investigate by re-expanding the gold set.

## What humans should NOT do

- Edit LLM labels directly. Add an adjudicator label and let Dawid-Skene
  reconcile.
- Override fail-closed gates. The route is the truth; if the route is wrong,
  fix the gate logic, not the output.
- Skip the source-independence ledger review. Conditional dependence is the
  silent killer of the A' identification claim.
