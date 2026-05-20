# Production Readiness Assessment — C-MAPSS FD001 RUL Prediction

**Prepared for:** Deployment Engineering / Operations
**Model:** Honest-tuned XGBoost (`fd001_xgb_tuned_honest`)
**Project:** [predictive-maintenance-cmapss](https://github.com/beebzy-droid/predictive-maintenance-cmapss)
**Date:** 2026-05-XX (fill in actual date)

---

## Summary

A classical gradient-boosted model trained on C-MAPSS FD001 turbofan data achieves **100% recall and 96.2% precision (F1 = 0.98)** as a maintenance alarm classifier at alarm threshold T = 35. Regression RMSE is 11.62, but the alarm-task performance is the operationally relevant metric — and it is operationally excellent.

**Recommendation:** suitable for deployment as a maintenance alarm system on equipment of similar design and operating profile, subject to the caveats below.

---

## Model behavior by operational regime

The model's accuracy is not uniform across RUL ranges. Predictions in operationally-critical regimes are dramatically more accurate than the overall RMSE suggests:

| Regime | Actual RUL | Test engines | RMSE | Score / engine | Behavior | Operational handling |
|---|---|---|---|---|---|---|
| Critical | 0–30 | 25 | **5.08** | 0.53 | Tight predictions, slight late bias | **Trust and act.** Alarm signal is reliable. |
| Watch | 30–80 | 20 | 16.61 | 4.62 | Systematic late bias (~7 cycles) | **Soft signal.** Schedule inspection, not maintenance. |
| Healthy | 80–125 | 55 | 11.55 | 1.84 | Conservative bias (predictions slightly early) | **Trust to confirm safety.** No action needed when model predicts > 100. |

The Critical regime has effective RMSE half the headline value (5 cycles vs 11.62). The model is **most accurate exactly where operational decisions get made.**

---

## Recommended alarm threshold

**T = 35** (alarm when the model predicts RUL ≤ 35 cycles).

| Outcome | At T = 35 |
|---|---|
| Failures caught | 25 of 25 (100%) |
| False alarms | 1 of 75 healthy engines (1.3%) |
| Precision | 96.2% |
| F1 | 0.980 |

Alternative operating points if costs are unusual:
- **T = 30** if false alarms cost as much as missed failures (zero false alarms, but misses 4 of 25 failures)
- **T = 50** for safety-critical equipment where missed failures are catastrophic (100% recall, 24% false-alarm rate)

---

## Caveats — what would change in real deployment

### 1. C-MAPSS is a simulator, not real equipment.

The model was trained and evaluated on the NASA C-MAPSS simulated turbofan dataset (Saxena & Goebel 2008). The simulator produces clean, deterministic degradation trajectories. Real equipment data has:

- Higher sensor noise and intermittent missing data
- Operating-condition variation (FD001 is single-mode; real engines see varying load, weather, ambient conditions)
- Multi-mode failure patterns (FD001 has 1 fault mode; real engines have many)

**Deployment data must be validated against C-MAPSS statistical distributions** before trusting predictions. A simple goodness-of-fit test on sensor mean/variance distributions would identify domain shift.

### 2. The Watch-regime late bias is partly a benchmark artifact.

C-MAPSS test trajectories are truncated mid-life by design — every test engine has *less* remaining life than an equivalent-looking train engine. The 75% late-prediction rate in the Watch regime (Task 3) reflects this. In real deployment on continuously-monitored engines (no truncation), this bias would not exist. The model's Watch-regime predictions may be more accurate in production than test metrics suggest.

### 3. The training set has 100 engines and one fault mode.

This is small by industrial standards. A production model would benefit from:

- 1000+ engines covering multiple fault modes
- Multi-year operational history per engine
- Validated maintenance-event labels from real work orders

The current model should be re-trained with site-specific data once 50+ failures have been observed on the deployed fleet.

---

## Deployment readiness checklist

Items completed in this project (✅) vs items required before going live (☐):

- ✅ Model achieves benchmark-competitive accuracy (RMSE 11.62, Score 207)
- ✅ Alarm-task performance characterized (F1 = 0.98 at T = 35)
- ✅ Calibration verified by regime (Task 2)
- ✅ Error patterns characterized by RUL regime (Task 3)
- ✅ Alarm-threshold tradeoff curve produced (Task 4)
- ✅ Unit tests covering preprocessing, modeling, evaluation (35 tests pass)
- ✅ Model artifacts saved and version-controlled
- ☐ Inference service (FastAPI or equivalent) — *Week 7 deliverable*
- ☐ Monitoring dashboard for prediction drift, sensor anomalies — *future work*
- ☐ Site-specific validation on first 50 deployed engines — *post-deployment*
- ☐ Retraining pipeline (model + data versioning) — *MLOps phase*
- ☐ A/B comparison vs. existing reliability-engineering rules — *operational validation*
- ☐ Failure mode coverage analysis (1 fault mode in FD001 vs many in real equipment) — *blocker for production*

**Status: ready for prototype deployment on equipment matching the FD001 operating profile. Not ready for general industrial production without site-specific revalidation.**

---

## Key references

- Saxena, A. and Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set." NASA Prognostics Data Repository, NASA Ames Research Center.
- See `docs/decisions.md` for engineering decisions and tradeoffs.
- See `notebooks/05_operational.ipynb` for full operational analysis.
- See `notebooks/03_baseline_models.ipynb` for model selection and SHAP interpretation.