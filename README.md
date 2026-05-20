# Predictive Maintenance — NASA C-MAPSS

**Status:** Work in progress.

## Overview
This project builds a predictive maintenance model for turbofan engine degradation using the NASA C-MAPSS dataset, as a proxy for rotating industrial equipment such as pumps and compressors. The goal is to predict whether an engine will fail within the next [N] operational cycles, with high recall on failure events and a tolerable false-alarm rate.

## Current results

**Classical baselines (Week 4) outperformed deep learning (Week 5) on FD001 with matched preprocessing.**

| Model | Test RMSE | Test Score |
|---|---|---|
| **Honest-tuned XGBoost** | **11.62** | **207** |
| Best LSTM (LSTM-64×2) | 12.66 | 291 |
| Vanilla LSTM-64 (untuned) | 13.69 | 425 |

Reference benchmarks from the literature:
- Asif et al. (2022) deep LSTM with heavy preprocessing: RMSE 7.78, Score ~100 (near state-of-the-art)
- Sayah et al. clustering LSTM: RMSE 14.08, Score 308
- CNN approaches: RMSE 12–13

**Reframed as an alarm classifier (Week 6):** the same XGBoost model achieves **F1 = 0.98** at alarm threshold T = 35 — catching all 25 failure-imminent engines in test with only 1 false alarm out of 75 healthy engines. The regression-RMSE framing under-sells the model's operational utility. See [`docs/production_readiness.md`](docs/production_readiness.md) for the deployment-engineering summary.

XGBoost lands in the CNN-tier benchmark range. The LSTM closed half the gap to XGBoost through architecture tuning but did not overtake it. The honest interpretation: for this dataset with engineered tabular features, gradient-boosted trees outperform deep learning. Matching state-of-the-art results requires more preprocessing than this project's scope allows.

## Live Demo

A working dashboard runs locally with two commands:

```bash
# Terminal 1 — start the FastAPI inference service (port 8000)
uvicorn app.api:app --reload --port 8000

# Terminal 2 — start the Streamlit dashboard (port 8501)
streamlit run app/dashboard.py
```

Then open http://localhost:8501 in a browser.

**What you can do:**
- Pick one of 6 demo engines from a dropdown (spanning Critical, Watch, and Healthy regimes)
- See its 30-cycle sensor traces with the four most-informative sensors highlighted
- Get live RUL predictions from both XGBoost (recommended) and LSTM (comparison)
- See alarm decisions at the operationally-recommended threshold (T=35)
- Compare classical vs deep predictions on the same engine in real time

The FastAPI service has auto-generated Swagger docs at http://localhost:8000/docs and 11 integration tests in `tests/test_api.py`.

**Demo screenshots:** see `docs/screenshots/` (Task 7c will add these).

### Screenshots

**Live Prediction tab — Engine #34 (Critical, actual RUL=7):**
![Live prediction view](docs/screenshots/live_prediction_engine_34.png)

**Classical vs Deep comparison tab:**
![Classical vs Deep comparison](docs/screenshots/classical_vs_deep_tab.png)

## Why this dataset
Real plant data is not publicly available. C-MAPSS is the standard benchmark in the Prognostics and Health Management (PHM) community and contains run-to-failure trajectories for rotating equipment, which maps conceptually to pumps and compressors (shared degradation physics: bearings, seals, performance loss).

## Project structure
- `notebooks/` — exploratory and modeling notebooks
- `src/` — reusable code (data loading, features, models, evaluation)
- `app/` — deployment demo (FastAPI + Streamlit)
- `docs/` — decision log and design notes
- `tests/` — unit tests

## Setup
Conda environment:

    conda env create -f environment.yml
    conda activate predmaint

Or with pip:

    pip install -r requirements.txt

## Roadmap
- [x] Week 1: Scoping, data, setup
- [x] Week 2: EDA
- [x] Week 3: Feature engineering
- [x] Week 4: Classical ML (XGBoost, Random Forest)
- [x] Week 5: LSTM
- [x] Week 6: Evaluation & threshold optimization
- [x] Week 7: Deployment demo
- [ ] Week 8: Polish & writeup