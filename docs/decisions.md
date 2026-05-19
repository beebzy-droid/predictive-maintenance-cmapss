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

**2026-05-19** — **RUL cap impact: 8,031 of 20,631 training rows (38.9%) have their RUL clipped at 125.**
Reasoning: This is the "healthy" phase of each engine's trajectory — the early cycles where sensors show no degradation signal. The remaining ~61% of rows (the "degrading" phase) carry the learnable signal the model trains on. Verification: matches the per-engine arithmetic where Unit 1 (lifetime 192) has 67/192 ≈ 35% capped rows, and the dataset average lifetime of ~206 cycles gives ~(206−125)/206 ≈ 39% capped.

**2026-05-19** — **Piecewise-linear RUL target implemented: clip(upper=125).**
Reasoning: Empirically justified in Week 2 Task 4c (sensors flat for RUL > 125). Roughly 60–65% of training rows have their RUL capped at 125 — these are "healthy" cycles where no degradation signal exists. The model trains to output 125 for all healthy rows and to predict declining values only in the final ~125 cycles of each engine's life.

**2026-05-19** — **RUL label computed via `groupby("unit")["cycle"].transform("max")` — uses future information.**
Reasoning: The label encodes the engine's eventual lifetime, which is information from the future of each trajectory. This is appropriate because RUL is only used as a training target, never as an input feature. The model is trained to predict this future-derived value from past sensor readings alone. Documented to avoid confusion: future information in labels is fine; future information in features would be leakage.

**2026-05-19** — **Engine-level train/validation split: 80/20 (RANDOM_SEED=42).**
Reasoning: Splitting at the row level would put early and late cycles of the same engine on opposite sides of the split, allowing the model to "validate" by memorizing individual engine signatures instead of learning generalizable degradation patterns. Engine-level splitting ensures every validation engine is unseen during training, simulating the real deployment condition. Verified the two splits have similar lifetime distributions (means within ~10 cycles).

**2026-05-19** — **Fixed RANDOM_SEED = 42 for reproducibility.**
Reasoning: Portfolio work requires that anyone running this notebook gets the same train/val split. The seed is set once at the top of the notebook and propagated to every random operation.

**2026-05-19** — **Noted: validation split (seed=42) contains no engines with lifetime > ~265 cycles.**
Reasoning: Random chance with a 20-engine validation set produced a split where all long-lived engines (280–360 cycle lifetimes) landed in training. In principle this could bias validation metrics; in practice it does not, because the RUL cap at 125 makes the model's behavior identical across all "healthy" cycles regardless of total lifetime. The model is evaluated on its ability to predict RUL during the degradation phase (last ~125 cycles), where the validation set has full representation. Accepted limitation rather than changing the seed (which would be p-hacking the validation split).

**2026-05-19** — **Validation split (seed=42) check: train mean lifetime 207.0, val mean 203.5 (difference 3.5 cycles).**
Reasoning: Means are nearly identical, confirming the split is representative on average. The validation set happens to contain no engines with lifetime > 265 (random-chance artifact of small val size, n=20). Impact is limited because (a) bulk of both distributions sits in the 150–230 range, where 80%+ of engines live, and (b) the RUL cap at 125 makes model behavior identical across "healthy" cycles regardless of total lifetime — the model is evaluated on its ability to predict RUL during the degradation phase, where val has full representation. Split accepted as-is.

**2026-05-19** — **Z-score normalization: `StandardScaler` fitted on train_df only, then transformed train/val/test with the same fitted scaler.**
Reasoning: Fitting separately on each split would cause information leakage (each split's statistics influence its own normalization, hiding real distributional shift). Fitting on all-data combined would leak val/test information into train normalization. The fit-on-train-only approach matches the deployment condition: at inference, one sample at a time is normalized using stored training-time statistics. Scaler's `mean_` and `scale_` arrays will be persisted alongside the model in Week 7 (deployment) so they can be applied to inference data.

**2026-05-19** — **`StandardScaler` over `MinMaxScaler` for this project.**
Reasoning: Z-score normalization (mean=0, std=1) is the convention in C-MAPSS literature (Asif et al. 2022 and most LSTM-based approaches use z-score) and is more robust to outliers than min-max scaling. The piecewise-linear RUL target makes z-score appropriate — sensor values during degradation can occasionally spike past the training range, and z-score handles this more gracefully than min-max (which clips at 0 and 1).

**2026-05-19** — **Distributional shift between train and test confirmed via post-normalization check.**
Reasoning: After fitting StandardScaler on train and applying to all three splits, test means drift to ±0.4 and stds to ~0.8 across the 14 sensors. This is not noise — it reflects C-MAPSS's design: train trajectories run to failure (average lifetime ~206 cycles, ~63% degrading-phase rows), while test trajectories are truncated mid-life (average lifetime ~76 cycles, mostly healthy-phase). Test data is structurally biased toward healthy sensor readings, which have values closer to engine baseline and lower variance. The piecewise-linear RUL cap is the design response — the model only needs to be accurate at low RUL, and outputs 125 for healthy cycles regardless of total lifetime.

**2026-05-19** — **Sliding-window construction: overlapping windows on train/val (one per cycle position), single final window on test.**
Reasoning: Train/val use overlapping windows to maximize training data — each engine produces (lifetime − window_size + 1) windows, all sharing structure but with different RUL targets. Test uses only the engine's final `window_size` cycles because (a) test trajectories are truncated mid-life and we don't have per-cycle ground truth, and (b) the deployment scenario is "given an engine's recent history, predict its current RUL once." Using one window per test engine matches both the data structure and the operational use case.

**2026-05-19** — **Test labels: y_test taken from rul_truth and clipped at RUL_CAP=125.**
Reasoning: rul_truth is the NASA-provided ground-truth file (one RUL value per test engine, taken at the engine's final observed cycle). Clipping at 125 matches the training target distribution — without it, the model would be penalized for predicting 125 on a test engine whose true RUL is 140 (the model literally cannot learn to distinguish these). This is the standard C-MAPSS evaluation protocol from the original Saxena & Goebel 2008 paper.

**2026-05-19** — **Note on cap percentage discrepancy after windowing: 38.9% (row-level) → 30.3% (window-target-level).**
Reasoning: When windows are constructed, each engine's first 29 cycles never become the "last cycle of a window," and therefore never become RUL targets. Those early cycles (which had the highest concentration of capped RUL = 125 values) disappear from the label distribution but remain as features inside the windows. The cap percentage in y_train (30.3%) is lower than the row-level percentage (38.9%), and this is expected behavior, not a bug.

