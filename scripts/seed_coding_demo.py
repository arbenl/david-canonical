"""Seed synthetic LLM coder labels + gold set for local pipeline demo.

This script lets you run the full coding → calibration → adjudication pipeline
locally without a live LLM API key.  It reads the real evidence items from the
latest normalized JSONL, assigns two synthetic coders with realistic label
noise, and creates a small gold CSV for calibration.

Usage:
    DATABASE_URL=postgresql://david:david@localhost:5544/david \
    python scripts/seed_coding_demo.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import timezone, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from david.config import (
    CODED_DIR,
    DATA_ROOT,
    GOLD_DIR,
    MODEL_VERSION,
    TACTIC_CLASSES,
)
from david.ingest.normalize import is_tobacco_relevant

DEMO_NOISE_RNG = np.random.default_rng(42)

# ── find latest normalized file ───────────────────────────────────────────────
norm_dir = DATA_ROOT / "raw_normalized"
jsonl_files = [f for f in sorted(norm_dir.glob("*.jsonl")) if "errors" not in f.name]
if not jsonl_files:
    print("No normalized data found. Run: david ingest --skip-coding")
    sys.exit(1)
latest = jsonl_files[-1]

# ── load tobacco-relevant non-gl items ───────────────────────────────────────
items = []
for line in latest.read_text().splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    if not d.get("stratum_id", "").startswith("gl_") and is_tobacco_relevant(
        d.get("title", ""), d.get("text", "")
    ):
        items.append(d)

if not items:
    print("No tobacco-relevant items found.")
    sys.exit(1)

print(f"Seeding synthetic labels for {len(items)} evidence items...")

# ── coder config ──────────────────────────────────────────────────────────────
# coder_1 is high-accuracy (kappa ~0.85), coder_2 is moderate (kappa ~0.70)
CODERS = [
    {"coder_id": "groq_llama31_8b_seed_001", "accuracy": 0.90},
    {"coder_id": "groq_llama31_8b_seed_002", "accuracy": 0.75},
]

# "True" label: 1 if title/text strongly implies tobacco industry interference
def true_label(item: dict, tactic: str) -> int:
    text = (item.get("title", "") + " " + item.get("text", "")).lower()
    if tactic == "SIO":   # State institution obstruction
        return int(any(kw in text for kw in ["fctc", "industry interfere", "lobby", "front group"]))
    if tactic == "MIO":   # Media/information obstruction
        return int(any(kw in text for kw in ["marketing", "advertis", "promot", "nicotine pouch"]))
    if tactic == "CSIO":  # Civil-society institution obstruction
        return int(any(kw in text for kw in ["civil society", "public health", "ban", "tax", "excise"]))
    return 0


def noisy_label(true_y: int, accuracy: float) -> int:
    """Flip with probability (1 - accuracy) for deterministic demo data only."""
    if DEMO_NOISE_RNG.random() < accuracy:
        return true_y
    return 1 - true_y


# ── write coded JSONL ─────────────────────────────────────────────────────────
CODED_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now(timezone.utc).date().isoformat()
out_path = CODED_DIR / f"coded_demo_{today}.jsonl"
coded_records = []

with out_path.open("w") as f:
    for coder in CODERS:
        for item in items:
            for k in TACTIC_CLASSES:
                y_true = true_label(item, k)
                y = noisy_label(y_true, coder["accuracy"])
                rec = {
                    "evidence_id": item["evidence_id"],
                    "coder_id": coder["coder_id"],
                    "tactic_class": k,
                    "Y": y,
                    "model_version": MODEL_VERSION,
                    "coded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                coded_records.append(rec)
                f.write(json.dumps(rec) + "\n")

                # Upsert to Postgres (non-fatal)
                try:
                    from david.db.repositories import upsert_coder_label
                    upsert_coder_label(
                        evidence_id=item["evidence_id"],
                        coder_id=coder["coder_id"],
                        tactic_k=k,
                        label=y,
                    )
                except Exception:
                    pass

print(f"  {len(coded_records)} labels written → {out_path}")

# ── write gold CSV (50% of items as gold standard) ────────────────────────────
GOLD_DIR.mkdir(parents=True, exist_ok=True)
gold_path = GOLD_DIR / "gold_b_calibration.csv"

# Gold label: 1 if any tactic fires (item is clearly about tobacco interference)
gold_items = items[: max(2, len(items) // 2)]   # first half as gold
with gold_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["evidence_id", "gold_label"])
    writer.writeheader()
    for item in gold_items:
        # Gold_B label: 1 if at least one tactic is active
        any_active = int(any(true_label(item, k) for k in TACTIC_CLASSES))
        writer.writerow({"evidence_id": item["evidence_id"], "gold_label": any_active})

print(f"  {len(gold_items)} gold items → {gold_path}")
print()
print("Next steps:")
print("  david calibrate-coders   # fit Dawid-Skene kappa posteriors")
print("  david ingest --skip-coding  # rebuild adjudicator queue")
