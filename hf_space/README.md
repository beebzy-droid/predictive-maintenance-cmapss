---
title: Predictive Maintenance C-MAPSS
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: RUL prediction demo on NASA C-MAPSS turbofan data
---

# Predictive Maintenance Demo — C-MAPSS FD001

Live demo of a Remaining Useful Life (RUL) prediction model trained on NASA's C-MAPSS turbofan dataset.

**Headline result:** Test RMSE 11.62 · F1 = 0.98 as a maintenance alarm at threshold T = 35.

This Space runs the **XGBoost model only** for hosting feasibility. The full version with both XGBoost and LSTM live inference + FastAPI backend is in the [GitHub repo](https://github.com/beebzy-droid/predictive-maintenance-cmapss).

## What this demo shows

- **Live RUL predictions** on 6 demo engines spanning Critical, Watch, and Healthy regimes
- **Sensor traces** for the 30-cycle input window with the four most-informative sensors highlighted
- **Alarm decisions** at the operationally-recommended threshold (T = 35)
- **Classical vs Deep comparison** explaining why XGBoost was chosen over LSTM

## How to use

Pick one of the 6 demo engines from the sidebar dropdown. The dashboard displays:
- The sensor readings the model sees (z-scored, 30-cycle window)
- The XGBoost prediction with alarm decision and error vs ground truth
- A summary of the classical-vs-deep comparison from the full project

## Background

NASA's C-MAPSS dataset provides simulated turbofan run-to-failure trajectories. It is the standard benchmark in the Prognostics and Health Management community for evaluating RUL prediction methods. This project uses C-MAPSS as a proxy for industrial rotating equipment (pumps, compressors) — degradation physics maps conceptually.

See the [GitHub repo](https://github.com/beebzy-droid/predictive-maintenance-cmapss) for the complete project, including:
- 5 Jupyter notebooks (EDA, features, classical modeling, LSTM, operational analysis)
- 46 passing tests
- FastAPI inference service + Streamlit dashboard with both models live
- ~30 dated engineering decisions in `docs/decisions.md`
- 1-page production-readiness summary in `docs/production_readiness.md`