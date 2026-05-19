# Decisions Log

A record of meaningful project decisions and the reasoning behind them.

---

**2026-05-19** — **Dataset: NASA C-MAPSS chosen over PHM 2008 and CWRU Bearing data.**
Reasoning: C-MAPSS is the most widely benchmarked dataset in the PHM community, giving published results to compare against. It contains run-to-failure trajectories (matches the prediction-horizon framing in the project outline) rather than fault classification on short signals. Turbofan physics map conceptually to industrial pumps/compressors (shared rotating equipment, bearing/seal degradation modes).

**2026-05-19** — **Sub-dataset: Starting with FD001 before expanding.**
Reasoning: FD001 has a single operating condition and a single fault mode. Validates the pipeline on the simplest problem before introducing the complexity of FD002–FD004 (multiple conditions, multiple faults).

**2026-05-19** — **Framework: TensorFlow/Keras chosen over PyTorch for LSTM.**
Reasoning: Faster path to a working baseline. PyTorch is the modern standard in production codebases, so a follow-up implementation may be added.

**2026-05-19** — **Environment: Python 3.11 + conda for scientific stack, pip for application layer.**
Reasoning: 3.11 is stable across all required libraries (TF, XGBoost, sklearn). Conda handles BLAS/numpy correctly; pip handles app-layer packages with fewer conflicts.

**2026-05-19** — **Project location: `OneDrive\Desktop` with sync disabled and Windows long-path support enabled.**
Reasoning: Default location preference; sync conflicts mitigated by disabling OneDrive sync for Desktop; path length risk mitigated by enabling `LongPathsEnabled` in registry and `core.longpaths` in git.

**2026-05-19** — **Library versions: using latest available (numpy 2.4, pandas 3.0, scikit-learn 1.8, tensorflow 2.21, xgboost 3.2).**
Reasoning: Installed fresh from current channels in May 2026. Some online examples will reference older APIs — defer to current docs when in doubt.