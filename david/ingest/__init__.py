"""Automated ingestion layer.

sources.py                    Source registry + scraper dispatcher
scrapers/                     Per-source adapters (news, legislative, civil-society, transparency)
normalize.py                  Raw -> canonical evidence schema
llm_coder.py                  Multi-LLM coder pool + Dawid-Skene gold calibration
adjudicator_queue.py          Human escalation triggered only by disagreement/active sampling
source_independence_ledger.py Pairwise structural-independence registry
human_loop_budget.py          Enforce adjudicator hours/cycle budget
"""
