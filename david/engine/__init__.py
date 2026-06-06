"""Engine: end-to-end orchestration and operational gates.

orchestrator.py             Pipeline runner (fit -> SBC -> falsify -> forecast -> route)
forecast.py                 H-step forward prediction with horizon-validity gating
router.py                   Forecast routing FG1..FG6 + posterior-FDP threshold
identification_distance.py  Theorem A' practical-id diagnostic across strata
observability_sensitivity.py Endogenous-observability lambda sensitivity bounds
"""
