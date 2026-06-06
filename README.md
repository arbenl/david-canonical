# DAVID/M0.1 Canonical Layer

Forward-prediction engine and automated ingestion stack on top of the
council_m01 measurement model.

Start with `docs/ARCHITECTURE.md`.

## Quick start

```bash
pip install -e .
david ingest --since 2026-05-01
david calibrate-coders
david fit
david sbc forecast
david falsify
david forecast --horizon 6
david route
david serve  # read-only API at localhost:8080
```

## Layout

- `david/`            — Python package (CLI, engine, ingest, model, simulator, validation, theorems, routing)
- `stan/`             — Stan models (m01_forward, coder_calibration, synthetic_generator)
- `scripts/`          — cron-style automation scripts
- `tests/`            — pytest scaffolds
- `docs/`             — architecture + theorem documentation
- `config/`           — source registry, independence ledger, pre-registration
- `data/`             — append-only data store (gitignored)

## Relationship to existing repo

This `canonical/` tree is **additive**. It does NOT replace `council_m01/`.
It imports the existing fit_contract, real_readiness preflights, and theorem
packets as ground-truth proof references.

The migration path is documented in `docs/ARCHITECTURE.md` § 12.

## What is NOT in this drop

- Real scraper credentials, API keys, or live data
- Production database (Supabase migration is in council_m01)
- Final tuned priors (placeholders + cross-references to council_m01 priors)
- Domain-specific tactic ontology beyond the 13-class M0.1 baseline

## License

Internal thesis work. Author: Arben Lila.
