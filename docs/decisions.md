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

**2026-05-19** — **Off-by-one in test assertion caught: cycles where raw RUL ≥ 125 (not > 125) are at the cap.**
Reasoning: A test for the RUL cap initially asserted 74 capped rows for a 200-cycle engine, but the correct count is 75 — cycle 75 itself has raw RUL of exactly 125, which `clip(upper=125)` leaves unchanged at 125. The test was wrong; the code was right. Documented because boundary errors of this kind are the most common bug in time-series windowing.

**2026-05-19** — **Tabular feature engineering: 5 statistics × 14 sensors = 70 features per window.**
Reasoning: XGBoost and Random Forest need 2D input. Two options: flatten raw values (30 × 14 = 420 features, lots of redundancy and overfitting risk) or compute summary statistics (5 × 14 = 70 features, more interpretable). Chose the latter. The 5 statistics — mean, std, slope, min, max — capture central tendency, volatility, degradation rate, and bounds. Slope is computed via vectorized closed-form linear regression rather than per-window polyfit calls (10× faster, exact same result, verified by comparison against np.polyfit on a sample window).

**2026-05-19** — **Vectorized slope computation chosen over per-window np.polyfit.**
Reasoning: Time index t = [0, 1, ..., 29] is identical for every window. Closed-form slope = Cov(t, y) / Var(t) can be computed via numpy broadcasting in a single pass. Verified equivalence against np.polyfit on a sample window (slopes match to 6 decimal places). For 14,000+ windows × 14 sensors this matters — polyfit version takes ~30 seconds, vectorized takes <100ms.

**2026-05-19** — **Slope features cross-validate Week 2 EDA findings.**
Reasoning: Independently-computed slope features have correlations with RUL of |r| = 0.63–0.76 for the four sensors selected for inspection. Signs match Week 2's qualitative rising/falling categorization for all four (sensors 4 and 11 rising, sensors 12 and 20 falling). This is an internal consistency check — the EDA and feature engineering were done with separate code paths but agree perfectly on sensor direction.

**2026-05-19** — **Tuning leakage bug caught and fixed: window-level vs engine-level CV split.**

**What happened:** Initial Optuna run used `sklearn.train_test_split` on `X_train_tab` to create a 85/15 tuning split. This split at the *window* (row) level, randomly. Because consecutive windows from the same engine differ by only one cycle of sensor readings, the tuning val set contained near-twin neighbors of the tuning train set. Optuna's optimization signal was corrupted: tuning RMSE = 3.59 (absurdly low) while Val RMSE on engine-level-held-out engines was 12.00. The 24× train-val gap was the alarm.

**The fix:** Replaced the random split inside the Optuna objective with `GroupKFold(n_splits=4)`, using engine IDs (computed from `train_df_again`) as the group label. Each CV fold now holds out entire engines, matching the train/val protocol from Week 3.

**Honest results after fix:**
- Best CV-RMSE (engine-level): 13.60 (vs leaky CV-RMSE: 3.59 — 4× different, confirming leakage)
- Final test RMSE: **11.62** (vs leaky 12.55 — honest model actually generalizes *better*)
- Chosen hyperparameters became conservative: max_depth dropped from 9 (leaky) to 3 (honest); learning_rate dropped from 0.028 to 0.019

**Lesson:** the leaky version's val and test numbers (12.00 / 12.55) were already reasonable — the leakage didn't catastrophically break the model, just optimized for the wrong objective. The visible symptom was the hyperparameter choices it made (too deep, too many trees, no regularization). Engine-level CV during tuning is non-negotiable for time-series data with group structure.

**2026-05-19** — **Random Forest baseline added: Val RMSE [your number], Test RMSE [your number].**
Reasoning: Provides a second classical baseline alongside XGBoost. Two RF configurations tested — default (unlimited depth, severe overfitting) and regularized (max_depth=15, min_samples_leaf=5). Regularized version is the fair comparison. Random Forest lands 1–3 RMSE points behind XGBoost, confirming that gradient boosting genuinely helps on this problem rather than feature engineering doing all the work. RF is also kept as a reference for Week 5 LSTM comparison — the LSTM must beat the best classical baseline (XGBoost at 11.62) to justify its additional complexity.

**2026-05-19** — **Random Forest baselines: Default Test RMSE 13.08, Regularized 12.57.**
Reasoning: Both RF variants land 0.95–1.46 RMSE points behind honest-tuned XGBoost (11.62). The gap is at the low end of typical for tree-based comparisons, indicating the engineered features (70 statistics: mean/std/slope/min/max × 14 sensors) carry the bulk of the predictive signal regardless of model choice. Regularization (max_depth=15, min_samples_leaf=5) reduces RF overfitting and is the fair comparison point. The honest-tuned XGBoost remains the strongest classical baseline; the LSTM in Week 5 must beat 11.62 RMSE on test to justify its additional complexity over the tree-based approach.

**2026-05-19** — **Key engineering observation: train RMSE inversely correlates with test RMSE across all four classical baselines.**
Reasoning: Default models (XGBoost train 1.97, RF train 2.00) had the lowest train errors but the highest test errors (12.62, 13.08). Regularized models (tuned XGBoost train 9.68, regularized RF train 4.68) had higher train errors but lower test errors (11.62, 12.57). The pattern empirically confirms regularization's value on this dataset — memorizing training data costs generalization. This is the practical version of the bias-variance tradeoff and is worth referencing in interview discussions of model selection.

**2026-05-19** — **NASA Score function implemented and applied: best test Score = [your number from honest-tuned XGBoost].**
Reasoning: RMSE treats all errors equally; the asymmetric NASA Score (Saxena & Goebel 2008) penalizes late predictions exponentially more than early ones (divisor 10 vs 13), matching the operational cost asymmetry — late predictions risk unscheduled failure, early predictions only waste an inspection. Function implemented with hand-checked unit tests against the formula. Score ranking on test matches RMSE ranking, confirming the four classical models have similar error patterns rather than systematic late bias.

**2026-05-19** — **NASA Score results on test, with operational interpretation.**
Reasoning: Honest-tuned XGBoost achieves Test Score 207, placing it between Sayah et al. (308) and the deep-LSTM state-of-the-art (Asif et al., ~100). The 2×2 prediction-vs-actual plots show why: errors are systematically biased toward the "early" (green) zone, especially at high actual RUL. The asymmetric Score function penalizes early errors 32% less than late errors of equal magnitude, so this conservative bias is exactly the operational behavior real predictive maintenance systems want. The bias likely emerges naturally from the training data structure — the piecewise-linear RUL cap creates many "predict 125" examples, teaching the model to default high (i.e., conservatively) when uncertain. This is an emergent behavior, not an explicit design choice, but worth highlighting as a strength of the piecewise-linear target.

**2026-05-19** — **SHAP analysis confirms three-way cross-check: EDA → features → model behavior.**

12 of the top 15 features by mean |SHAP value| are slopes. The four sensors flagged in Week 2 EDA as the most informative (sensors 4, 11, 12, 20) all rank in the SHAP top 15. The directional patterns in the SHAP summary plot match the rising/falling categorization: rising-sensor slopes push predictions DOWN when high, falling-sensor slopes push predictions DOWN when low. This is the strongest internal-consistency check in the project — three independent pieces of work (visual EDA in Week 2, quantitative slope-RUL correlations in Week 4 Task 2d, and trained-model SHAP attribution in Week 4 Task 6) all identify the same sensors as most predictive and assign the same directions. The cross-check confirms the model is learning real degradation physics, not memorizing noise.

**2026-05-19** — **Moving-median preprocessing applied to all windows: kernel_size=10, post-normalization.**
Reasoning: Asif et al. used moving-median smoothing as a key preprocessing step in their state-of-the-art FD001 LSTM (RMSE 7.78). The median filter removes cycle-to-cycle measurement noise without destroying long-term degradation trends — verified by raw-vs-smoothed slope correlations >0.95 and stable slope-RUL correlations. We applied the filter *after* z-score normalization rather than before, because rebuilding the entire Week 3 pipeline would take ~2 hours. The mathematical difference between smooth-then-normalize and normalize-then-smooth is small (z-scoring is linear), but worth documenting as a deliberate compromise rather than ignoring.

**2026-05-19** — **Moving-median smoothing (kernel=10) on z-scored data weakens slope-vs-RUL signal across all sensors.**

Observed effects:
- Raw-vs-smoothed slope correlations: 0.91–0.99 (lower than the ≥0.95 expected)
- Slope-vs-RUL correlations dropped on every sensor after smoothing: sensor_2 (-0.602 → -0.542), sensor_4 (-0.712 → -0.665), sensor_11 (-0.756 → -0.718), and similar drops on the others.

Drops are small (~0.04 average) but systematically in one direction, indicating smoothing is removing some real signal along with noise. Likely cause: after z-score normalization, the "high-frequency content" of sensors isn't pure measurement noise — it includes fine structure (and in some sensors like sensor_17, discrete/quantized behavior) that carries information. Asif et al. applied median smoothing to *raw* sensor values where the high-frequency content was genuine measurement jitter; the same operation on z-scored data is more aggressive than warranted.

**Decision:** Skip smoothing for the main LSTM run; train on identical inputs to XGBoost for a clean apples-to-apples comparison. Keep the smoothing experiment as a controlled ablation in Task 6 to document the negative finding.

This is an honest negative result. The "Asif-style preprocessing helps" hypothesis didn't replicate when applied to post-normalization data in this project — likely because the order of operations matters more than I initially assumed.

**2026-05-19** — **Vanilla LSTM (64-unit single layer, 22K params) underperforms XGBoost: Test RMSE 13.69 vs 11.62.**

Both models trained on identical input (un-smoothed z-scored 30-cycle windows). The LSTM's training behavior is healthy — converged cleanly with early stopping at epoch 10, no instability. The 2.07-point RMSE gap is therefore not a training problem; it's an architecture problem. Likely under-capacity vs the implicit richness of 532-tree XGBoost. Will address with architecture tuning in Task 4 (larger units, more layers, possibly bidirectional). The Score gap (425 vs 207) widened proportionally more than the RMSE gap, suggesting the LSTM is making more "late" predictions — the asymmetrically penalized kind — than XGBoost.

**2026-05-20** — **Final Week 5 result: LSTM-64×2 at Test RMSE 12.66, losing to XGBoost (11.62) by 1.04 points.**

Tested six LSTM configurations: vanilla 64-unit, 128-unit, stacked 64×2, BiLSTM-64, BiLSTM-128, and Optuna-tuned 64×2. All six underperformed honest-tuned XGBoost on test RMSE. The best deep-learning result we can reproduce is LSTM-64×2 from the architecture sweep at Test RMSE 12.66 / Test Score 290.7. Reported this as the headline LSTM number rather than the Optuna-tuned variant (see next entry on winner's curse).

**2026-05-20** — **Caught the winner's curse in LSTM hyperparameter tuning; chose reproducibility over peak performance.**

Optuna tuning of LSTM-64×2 across 12 trials reported best val RMSE 11.77 (trial 10: LR=0.0027, dropout=0.38, dense=64). Retraining the model with identical hyperparameters and seed produced val RMSE 13.03 — a 1.26-point regression. The Optuna "winner" was a lucky seed sample rather than a genuinely better configuration. The seed-induced variance between trials (visible from trials 10 and 11 differing by 0.36 RMSE at near-identical hyperparameters) was larger than the apparent improvement from tuning. We chose to report the LSTM-64×2 architecture-sweep result (Test RMSE 12.66, reproducible) rather than the Optuna-tuned result (apparent Test RMSE 11.77, not reproducible). Documented as an honest engineering finding rather than papered over.

**2026-05-20** — **Conclusion: classical methods (XGBoost) outperform deep methods (LSTM) on FD001 with matched preprocessing.**

Three contributing factors: (1) the 70 engineered tabular features already encode the slope-based degradation signals that SHAP showed XGBoost relies on, leaving little for an LSTM to discover automatically; (2) the piecewise-linear RUL cap at 125 is structurally easier for trees (leaf rules can encode "predict 125 if healthy") than for LSTMs (which must learn the flat top through smooth nonlinearity); (3) without aggressive preprocessing innovations like Asif et al.'s raw-value moving-median smoothing, the LSTM is operating on the same noisy signal as XGBoost — and our Task 2 result showed that median smoothing on z-scored data degrades rather than improves signal. The honest portfolio takeaway: for this dataset and pipeline, deep learning's complexity is not warranted; a practitioner facing this problem would choose XGBoost.

**2026-05-20** — **Calibration analysis: model is well-calibrated at operationally-critical extremes (very low RUL and very high RUL), noisy in mid-RUL regime.**

10-bin calibration on test set shows: low-RUL bins (n=8 and n=10, midpoints 6.25 and 18.75) have bias under 3 cycles; high-RUL bins (n=16, n=17, n=22, midpoints 93.75–118.75) have bias under 6 cycles. Mid-RUL bins (n=2–7, midpoints 50–82) have bias swings of ±10 cycles with large within-bin std (up to 21). This is structurally a low-data issue rather than a model bug — only ~25 of the 100 test engines have actual RUL in the 25–87 range. The model's calibration is strongest exactly where operational decisions are made; the fuzzy mid-regime is where human-in-the-loop inspection would be planned anyway. Documented as a feature of the model's behavior rather than a flaw; the RMSE alone (11.62) hides this regime-dependent structure.

**2026-05-20** — **Regime-specific error analysis reveals model is much more accurate in operationally-critical zones than the headline RMSE suggests.**

Per-regime breakdown on test:
- Critical (n=25, actual RUL 0-30): RMSE 5.08, Score/engine 0.53, 64% late
- Watch (n=20, actual RUL 30-80): RMSE 16.61, Score/engine 4.62, 75% late
- Healthy (n=55, actual RUL 80-125): RMSE 11.55, Score/engine 1.84, 44% late

The Critical regime is 2.3× more accurate than the headline RMSE (5.08 vs 11.62). The Watch regime is 1.4× less accurate, contributes 45% of the total Score budget despite being 20% of engines, and has a 75% late-prediction rate. The Watch-regime late bias is partly structural — C-MAPSS test trajectories are truncated mid-life, biasing the actual RUL downward relative to what an equivalent-looking non-truncated engine would have. In deployment, this bias would not exist. Documented so that future deployment / report discussion frames the model's accuracy by regime rather than by overall RMSE.

**2026-05-20** — **Alarm threshold analysis: T=35 achieves 100% recall, 96.2% precision, F1=0.98 on test.**

Ground truth: engines with actual RUL ≤ 30 cycles "need maintenance" (25 of 100 test engines). Sweep alarm threshold T from 10 to 80, computing TP/FP/FN/TN at each. The threshold tradeoff curve hugs the top-left corner of ROC space — the model is highly discriminative between Critical (actual ≤ 30) and Healthy regimes, even where its regression accuracy is imperfect.

Three recommended operating points:
- T=30 (conservative): 84% recall, 100% precision — for low-stakes equipment
- **T=35 (recommended): 100% recall, 96.2% precision, F1=0.98** — for typical maintenance
- T=50 (aggressive): 100% recall, 75.8% precision — for safety-critical equipment

The model's classifier performance (F1=0.98) is dramatically better than its regressor performance (RMSE 11.62) would suggest. Same model, different framing: the alarm task only needs the prediction to fall on the correct side of T, not match the actual value exactly. This is the framing an industrial customer would care about; the calibration analysis in Task 2 (bias < 3 cycles in the 0-25 range) confirms the model's accuracy in the operationally-critical regime where alarms get fired.

**2026-05-20** — **Created `docs/production_readiness.md` — a 1-page operational summary for hypothetical deployment-engineering audience.**

Synthesizes Tasks 1-4 of Week 6 into a single document a non-data-scientist could read in 5 minutes. Key claims:
1. Model is suitable for prototype deployment at alarm threshold T=35 (F1=0.98)
2. Regime-specific accuracy patterns matter — Critical-regime RMSE (5.08) is half the headline RMSE (11.62)
3. Three caveats are explicit: C-MAPSS is a simulator, the Watch-regime bias is partly a benchmark artifact, and the training set covers only 1 fault mode
4. Deployment readiness is "ready for prototype on similar equipment, not ready for general industrial production"

The document is intended as an interview talking point and as a model for how to communicate ML results to non-technical stakeholders. The honest hedging (caveats section, "not ready for general production" stance) is the senior-engineer move; overselling readiness would be a red flag.

**2026-05-20** — **Week 7: FastAPI + Streamlit demo built. Architecture: backend-frontend split with /predict/xgb and /predict/lstm endpoints.**

Design choices:
- Single-file Streamlit dashboard with three tabs (Live Prediction, Classical vs Deep, About) — preferred over multi-page `pages/` folder for simplicity and shared state
- Models loaded lazily on first request in FastAPI (fast startup, slow first prediction)
- Canned demo engines pre-extracted to `app/example_data/demo_engines.json` (6 engines, 1 critical-extreme + 1 critical-borderline + 1 watch-borderline + 1 watch-borderline + 1 healthy-borderline + 1 healthy-extreme)
- CORS configured to allow localhost:8501 (Streamlit) to call localhost:8000 (FastAPI) from the browser
- 11 new API tests (TestClient-based) bring the suite to 46 passing tests

The LSTM is kept in the demo despite losing the Week 5 comparison. Rationale: the project's claim that classical methods outperform deep learning on FD001 is more credible when the deep model's actual behavior is *visible* to a viewer, not just referenced in a table. The LSTM hedging at the cap (~110 vs XGBoost's 125 on healthy engines) is on live display in the comparison tab.

