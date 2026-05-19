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

**2026-05-19** — **Window size for sequence modeling: target 30–50 cycles.**
Reasoning: Empirically constrained by FD001's shortest training trajectory (~128 cycles). A 30-cycle window leaves >95 cycles of context per engine even at the minimum lifetime, which is enough for the model to observe degradation onset and trajectory.

**2026-05-19** — **Window size for sequence modeling: 30 cycles (revised).**
Reasoning: Originally considered 30–50 cycles based on shortest training trajectory (~128). On reviewing test trajectories, the shortest is ~30 cycles, which is the actual binding constraint for inference. A 30-cycle window is the largest size that makes a prediction possible for every test engine without padding workarounds.

**2026-05-19** — **Dropping op_setting_1, op_setting_2, op_setting_3 from modeling features on FD001.**
Reasoning: Empirically confirmed in EDA Q5. op_setting_3 is exactly constant (std = 0, single value = 100.0); op_setting_1 and op_setting_2 vary only within noise bands (std < 0.003, ranges ±0.009 and ±0.0006 respectively). No information for FD001. (Note: same columns are highly informative on FD002/FD004 — would be retained there.)

**2026-05-19** — **Dropping 7 of 21 sensors on FD001: sensor_1, 5, 6, 10, 16, 18, 19.**
Reasoning: Six are numerically constant (std = 0) across the entire training set. Sensor_6 has std = 0.00139 but inspection shows only 2 unique values (21.60 vs 21.61), separated by a single quantization step — variance is a numerical artifact, not a physical signal. Retained 14 informative sensors. Drop list matches the convention used in Asif et al. (2022) and other published C-MAPSS benchmarks; arrived at here independently from data inspection.

**2026-05-19** — **RUL target will be clipped at 125 cycles (piecewise-linear).**
Reasoning: EDA Task 4b shows informative sensors are flat for ~100–125 cycles before degradation onset becomes visible. Above 125 cycles remaining, no degradation signal exists in the data — modeling that range would force the network to predict from noise. Cap value of 125 is supported empirically and matches the convention in published C-MAPSS benchmarks (Asif et al., 2022 and others).

**2026-05-19** — **Flagged sensor_9 and sensor_14 for behavioral inconsistency across engines.**
Reasoning: Both show upward drift on the long-lived Unit 56 but flat or downward drift on shorter-lived engines. Possible bidirectional degradation or secondary fault interaction. Keeping in the model for now, but worth revisiting if any sensor shows up as harmful in feature importance analysis in Week 4.

