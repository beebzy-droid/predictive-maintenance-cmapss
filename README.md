# Predictive Maintenance — NASA C-MAPSS

**Status:** Work in progress.

## Overview
This project builds a predictive maintenance model for turbofan engine degradation using the NASA C-MAPSS dataset, as a proxy for rotating industrial equipment such as pumps and compressors. The goal is to predict whether an engine will fail within the next [N] operational cycles, with high recall on failure events and a tolerable false-alarm rate.

The project demonstrates the full ML lifecycle — EDA, feature engineering, classical and deep learning models, cost-aware evaluation, and a deployment-ready inference service — as applied to industrial sensor data.

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
- [/] Week 1: Scoping, data, setup
- [ ] Week 2: EDA
- [ ] Week 3: Feature engineering
- [ ] Week 4: Classical ML (XGBoost, Random Forest)
- [ ] Week 5: LSTM
- [ ] Week 6: Evaluation & threshold optimization
- [ ] Week 7: Deployment demo
- [ ] Week 8: Polish & writeup