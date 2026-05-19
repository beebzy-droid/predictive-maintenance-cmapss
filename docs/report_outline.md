# Academic Report — Predictive Maintenance Model on NASA C-MAPSS

## Working title (TBD)

---

## 1. Introduction
- Problem context: predictive maintenance for rotating industrial equipment
- Why C-MAPSS as a proxy for pumps / compressors
- Project scope and goals

## 2. Background & Related Work
- Prognostics and Health Management (PHM)
- Remaining Useful Life — definitions, framings
- Existing C-MAPSS benchmarks (Asif et al., Sayah et al., Peringal et al.)

## 3. Dataset
- C-MAPSS overview
- FD001 vs FD002 / FD003 / FD004
- Data structure and sensor inventory

## 4. Exploratory Data Analysis
- Structural overview
- Sensor analysis: identifying signal vs noise
- The flat-then-degrade pattern and the RUL cap

## 5. Feature Engineering
- Sensor selection
- Sliding-window construction
- Normalization strategy
- Train/validation splits at engine level

## 6. Modeling
- Classical baseline: XGBoost
- Classical baseline: Random Forest
- Deep learning: LSTM
- Hyperparameter tuning

## 7. Evaluation
- RMSE and NASA Score on FD001
- Threshold optimization for classification framing
- Comparison to published benchmarks

## 8. Deployment Demo
- FastAPI inference service
- Streamlit dashboard

## 9. Discussion
- Limitations of the C-MAPSS proxy
- What would change in a real plant deployment (sensor noise, missing data, operating regimes)
- Future work

## 10. Conclusion

## References