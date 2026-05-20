# 🏭 Predictive Maintenance — NASA C-MAPSS Turbofan Data

**🚀 [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/beebzy-droid/predictive-maintenance-cmapss)** · [GitHub repo](https://github.com/beebzy-droid/predictive-maintenance-cmapss)

A full end-to-end ML system that predicts Remaining Useful Life (RUL) for turbofan engines using sensor time-series data — using NASA's C-MAPSS dataset as a proxy for industrial rotating equipment such as pumps and compressors. Built with engineering rigor visible at every layer: from leakage-aware data splits to honest classical-vs-deep comparison to an operationally-deployable web demo.

## 🎯 Project Overview

Built a complete predictive maintenance pipeline that takes 30-cycle sensor windows from turbofan engines and predicts how many operational cycles remain before failure. The headline XGBoost model lands in CNN-tier benchmark territory (Test RMSE 11.62) and — reframed as a maintenance alarm — achieves F1 = 0.98 with 100% recall on failure-imminent engines and only 1.3% false alarms. The reframing matters more than the regression number: the same model is "modest" as a regressor and "excellent" as a decision-maker.

## 📊 Live Demo

🔗 [Try it now on Hugging Face Spaces](https://huggingface.co/spaces/beebzy-droid/predictive-maintenance-cmapss)

The live demo runs the XGBoost model only for hosting feasibility. The full version with both XGBoost and LSTM live inference + FastAPI backend can be run locally:

```bash
# Terminal 1 — FastAPI inference service (port 8000)
uvicorn app.api:app --reload --port 8000

# Terminal 2 — Streamlit dashboard (port 8501)
streamlit run app/dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501).

## 🛠️ Tech Stack

| Layer | Tool |
| --- | --- |
| Data Pipeline | Python, NumPy, Pandas |
| Feature Engineering | 70 engineered tabular features (mean, std, slope, min, max per sensor) |
| Classical Modeling | Scikit-learn, XGBoost, Random Forest |
| Deep Modeling | TensorFlow / Keras (LSTM architectures) |
| Hyperparameter Tuning | Optuna (with engine-level GroupKFold) |
| Explainability | SHAP (TreeExplainer, global + local) |
| Inference API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Matplotlib |
| Deployment | Hugging Face Spaces (Docker SDK) |
| Testing | Pytest (46 passing tests) |
| Version Control | Git & GitHub |

## 📈 Key Results

- **Test RMSE: 11.62** — XGBoost lands in the CNN-tier benchmark range on FD001
- **Test NASA Score: 207** — outperforms Sayah et al. (308) by ~33%
- **F1 = 0.98** — as a maintenance alarm at threshold T = 35
- **100% recall** — catches all 25 failure-imminent engines in the test set
- **1.3% false alarm rate** — only 1 out of 75 healthy engines triggers a needless alarm
- **Critical-regime RMSE: 5.08** — less than half the headline RMSE, where it matters most
- **46/46 tests passing** — full unit + integration coverage across pipeline, models, and API
- **6 LSTM architectures benchmarked** — honest finding: classical beats deep on this dataset

## 🔍 Key Findings

- Engine-level splits matter — row-level splitting leaked engine signatures and inflated CV-RMSE 4× lower than honest (3.59 vs 13.60)
- Sensors 4, 11, 12, 20 dominate RUL signal — confirmed independently by EDA, slope-RUL correlations (|r| = 0.63–0.76), and SHAP rankings
- Aggregate RMSE hides heterogeneity — model is dramatically more accurate in the Critical regime (RMSE 5.08) than the overall RMSE (11.62) suggests
- LSTM tuning showed winner's curse — best Optuna trial val RMSE 11.77 didn't reproduce on retrain (13.03), so the reproducible architecture-sweep result (12.66) is reported instead
- Classical beats deep on FD001 with engineered features — the piecewise-linear RUL target structurally favors trees, and engineered slope features already capture the degradation signal LSTMs would have to learn from raw input

## 📊 Model Performance

### Test set (FD001, 100 engines)

| Model | Test RMSE | Test Score | Notes |
| --- | --- | --- | --- |
| **Honest-tuned XGBoost** | **11.62** | **207** | Engine-level GroupKFold tuning |
| Regularized Random Forest | 12.57 | 252 | Default hyperparameters with regularization |
| LSTM-64×2 (best deep) | 12.66 | 291 | Architecture sweep, fixed seed |
| Vanilla LSTM-64 | 13.69 | 425 | Untuned baseline |

### Reframed as maintenance alarm classifier (T = 35)

| Metric | Value |
| --- | --- |
| Recall | 100% (25 of 25 failure-imminent engines caught) |
| False alarm rate | 1.3% (1 of 75 healthy engines) |
| F1 | 0.98 |

### Comparison to published benchmarks

| Method | Test RMSE | Notes |
| --- | --- | --- |
| Asif et al. (2022) deep LSTM | 7.78 | Near state-of-the-art, heavy preprocessing |
| **This work — XGBoost** | **11.62** | CNN-tier benchmark range |
| Sayah et al. clustering-LSTM | 14.08 | |
| Vanilla LSTM baselines | 16+ | Untuned literature baselines |

## 💼 Business Impact

| Problem | Solution |
| --- | --- |
| Unplanned equipment downtime costs more than scheduled maintenance | Predicts cycles-to-failure on each engine, enabling condition-based maintenance |
| Aggregate model metrics can mislead operators about real-world performance | Per-regime calibration analysis surfaces the Critical-regime RMSE (5.08) where decisions actually get made |
| Tuning hyperparameters with row-level CV silently introduces leakage in time-series data | Engine-level GroupKFold catches the bug — improved test RMSE from 12.55 to 11.62 |
| Deep learning is the assumed default for time-series forecasting | Honest classical-vs-deep comparison shows XGBoost wins on FD001 under matched preprocessing |
| "RMSE = 11.62" doesn't tell ops engineers whether to deploy the model | Reframed as alarm classifier with F1 = 0.98 — same model, different framing, dramatically different stakeholder value |

> **Bottom line:** Catches 100% of failure-imminent engines with only 1.3% false alarms — translating a "modest" 11.62 RMSE into an operationally-actionable maintenance signal.

## 🏭 Industry Applications

| Industry | Use Case | Business Value |
| --- | --- | --- |
| ⚗️ Petrochemicals | Compressor health monitoring | Prevents unplanned outages on critical rotating equipment |
| 🏭 Heavy manufacturing | Bearing & seal degradation | Schedules part replacement before failure |
| 💧 Water/utilities | Pump impeller wear | Avoids cascading downtime in continuous operations |
| ✈️ Aerospace MRO | Turbofan engine RUL (the original C-MAPSS use case) | Optimizes engine overhaul scheduling |
| ⚙️ Industrial IoT platforms | Real-time sensor-to-prediction service | FastAPI/REST inference pattern generalizes to any rotating equipment |

## 🧪 Key Engineering Decisions

A `docs/decisions.md` log captures ~30 dated entries explaining the reasoning behind each choice. The five most interview-relevant:

1. **Engine-level GroupKFold for hyperparameter tuning.** Initial Optuna runs used row-level CV and reported CV-RMSE 3.59 — leakage from consecutive windows of the same engine. Engine-level GroupKFold gave honest CV-RMSE 13.60 (4× different). Test RMSE improved from 12.55 (leaky) to 11.62 (honest) once hyperparameters reflected real generalization, not memorization.
2. **Three-way cross-check between EDA, features, and SHAP.** Week 2 EDA identified sensors 4, 11 as rising and 12, 20 as falling. Week 4 feature engineering measured slope-RUL correlations of |r| = 0.63–0.76 for those same four. Week 4 SHAP analysis ranked their slopes in the top 5 features by mean |SHAP|, with directional patterns matching the rising/falling categorization. Three independent analyses agreeing on the same physics.
3. **Winner's-curse detection in LSTM tuning.** Optuna tuning reported best val RMSE 11.77, but retraining with identical hyperparameters and seed produced val RMSE 13.03 — seed-induced variance larger than the apparent improvement. The architecture-sweep result (Test RMSE 12.66, reproducible across seeds) is reported as the best LSTM rather than the Optuna result that wouldn't replicate.
4. **Reframing regression as classification for operational utility.** The same XGBoost that scores Test RMSE 11.62 (modest) achieves F1 = 0.98 when used as a maintenance alarm at threshold T = 35. The regression-level prediction doesn't matter for the alarm decision — only which side of the threshold the prediction falls on.
5. **Per-regime calibration analysis.** The Critical regime (actual RUL < 30) has RMSE 5.08 — less than half the global RMSE. Aggregate metrics hide heterogeneity; the model is much better than its headline number suggests in the regime where operational decisions actually get made. See [`docs/production_readiness.md`](docs/production_readiness.md) for the full analysis.

## 🧠 What I Learned

Six lessons from building this project end-to-end, each tied to a specific moment in the work:

1. **Time-series ML lives or dies by data splits.** When I first tuned XGBoost with row-level random splitting, Optuna reported CV-RMSE 3.59 — a stunning result. The 24× train/val gap on the final model was the alarm. Engine-level GroupKFold gave honest CV-RMSE 13.60 (4× different). For any time-series problem where data points within the same entity correlate, leakage prevention is non-negotiable.
2. **The metric you optimize isn't necessarily the metric that matters.** XGBoost's Test RMSE of 11.62 sounds modest. Reframed as a maintenance alarm at threshold T=35, the same predictions deliver F1=0.98 — 100% recall on failures with only 1 false alarm per 75 healthy engines. The reframing isn't a trick; it's a shift from "how accurate is the regression?" to "how good is the decision?" When stakeholders care about decisions, evaluate decisions.
3. **Aggregate metrics hide heterogeneity.** Calibration analysis showed the Critical-regime RMSE was 5.08 — less than half the global RMSE of 11.62. The model is dramatically more accurate exactly where operational decisions get made (engine near failure) than the headline number suggests. Per-regime evaluation should be standard, not optional.
4. **Honest negative results are more interesting than dishonest positive ones.** I gave LSTM every reasonable chance — six architectures, focused hyperparameter tuning — and it lost to XGBoost by 1 RMSE point. The temptation was to keep tweaking until LSTM "won." Instead, I documented the loss, identified the cause, and treated the negative finding as the result. Recruiters can smell p-hacking; they can also recognize discipline.
5. **Reproducibility is the test of a real ML pipeline.** Set seeds, but don't trust them. When my Optuna-tuned LSTM produced val RMSE 11.77 during tuning and 13.03 on retrain with identical hyperparameters, I caught the "winner's curse" — picking the lucky run from a noisy distribution. I reported the architecture-sweep result (12.66, reproducible) instead of the tuned-but-irreproducible 11.77. If a result doesn't replicate, it isn't a result.
6. **Building tells you what matters more than reading does.** I read about engine-level splits, calibration plots, and Pydantic validation before this project. I _understood_ them only after debugging the leakage bug, building the calibration plot that revealed regime heterogeneity, and watching FastAPI's Pydantic layer catch a 422 on malformed input. The notebook-to-production transition is where understanding compounds.

## 📸 Screenshots

### Live Prediction tab — Engine #34 (Critical regime, actual RUL = 7)

![Live prediction view](docs/screenshots/live_prediction_engine_34.png)

### Classical vs Deep comparison tab

![Classical vs Deep comparison](docs/screenshots/classical_vs_deep_tab.png)

## 📁 Project Structure

predictive-maintenance-cmapss/
├── notebooks/                    → 01_eda → 05_operational
├── src/                          → Reusable modules: data, features, models, evaluate, sequence_models
├── app/                          → Full local app — FastAPI service + Streamlit dashboard (both models live)
├── hf_space/                     → Slim deployment package — Streamlit-only, XGBoost-only, Docker-ready
├── tests/                        → 46 passing tests across pipeline, models, evaluation, API
├── docs/
│   ├── decisions.md              → ~30 dated engineering decisions
│   ├── production_readiness.md   → 1-page operational deployment summary
│   ├── domain_notes.md           → C-MAPSS dataset framing and PHM background
│   └── screenshots/              → Dashboard screenshots
├── data/
│   ├── raw/                      → C-MAPSS FD001 files (gitignored)
│   └── processed/                → Trained models, scalers, processed arrays (gitignored)
├── LICENSE                       → MIT
├── requirements.txt              → Clean dependencies with loose version pins
└── README.md

## 🤖 Why C-MAPSS as a proxy for industrial equipment

Real plant data isn't publicly available. NASA's C-MAPSS is the standard benchmark in the Prognostics and Health Management (PHM) community, with run-to-failure trajectories for rotating turbofan engines. The degradation physics — bearings, seals, performance loss — maps conceptually to industrial pumps and compressors. C-MAPSS is the closest reproducible standard for an industrial-context ML portfolio project.

## 📚 References

- Saxena, A. and Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set." NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- Asif et al. (2022) — referenced as state-of-the-art LSTM benchmark with RMSE 7.78 on FD001.
- Sayah et al. — referenced as clustering-LSTM baseline with RMSE 14.08.

---

_End-to-end ML portfolio project demonstrating engineering rigor, honest reporting of negative results, and an operationally-deployable artifact. Built solo over ~8 weeks of focused work._