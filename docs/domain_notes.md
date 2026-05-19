# Domain Notes: Predictive Maintenance & C-MAPSS

## 1. Core concepts

- **Prognostics:** Predicting when a system or component will fail or no longer perform its intended function.
- **Remaining Useful Life (RUL):** The estimated time left before a component reaches its failure threshold. RUL is the primary output of a prognostics model.
- **Condition-based maintenance vs. time-based maintenance:** Time-based maintenance replaces parts on a fixed schedule regardless of actual condition. Condition-based maintenance triggers work only when sensor data indicates degradation, reducing unnecessary downtime and cost.
- **Run-to-failure data:** Sensor readings collected from a component operated continuously until it actually fails. Essential for training prognostics models, but expensive to obtain in practice — most operators replace parts before failure.
- **Censored data:** Data from units that were removed or stopped before failure occurred, so the true failure time is unknown. Right-censored data is very common in industrial settings and complicates label construction.
- **Health index / degradation trajectory:** A scalar value, often engineered from multiple sensors, that represents the overall health of a component over time. Typically declines from 1 (healthy) toward 0 (failure).
- **Prognostic horizon:** How far in advance an algorithm can reliably predict failure with acceptable accuracy. A longer horizon gives operators more time to act.

## 2. Industrial sensor types (general vocabulary)

- **Vibration analysis:** Measuring the oscillatory motion of machinery to detect faults. As components degrade they vibrate differently, making vibration a rich signal for health monitoring.
  - **1×, 2×, 3× running speed:** Multiples of the shaft's rotational frequency (1× = one vibration per revolution). 1× typically indicates imbalance or misalignment; 2× suggests mechanical looseness or coupling misalignment; 3× and higher harmonics point to more complex mechanical faults.
  - **Bearing defect frequencies:**
    - **BPFO (Ball Pass Frequency, Outer race):** Frequency at which rolling elements pass a defect on the *stationary* outer race.
    - **BPFI (Ball Pass Frequency, Inner race):** Frequency at which rolling elements pass a defect on the *rotating* inner race.
    - **BSF (Ball Spin Frequency):** Rotational frequency of an individual rolling element about its own axis — defects on the ball or roller surface show up here.
    - **FTF (Fundamental Train Frequency):** Rotational frequency of the cage assembly that holds the rolling elements — cage damage appears at this frequency.
- **Motor current signature analysis (MCSA):** Analyzing the electrical current drawn by a motor to detect mechanical and electrical faults (broken rotor bars, bearing wear, eccentricity) without needing physical sensors mounted on the machine.
- **Temperature monitoring:** Tracking heat as a proxy for friction, electrical resistance, or lubrication failure. A rising temperature trend often precedes mechanical breakdown.
- **Lubricant analysis:** Examining oil or grease samples for metal particles, viscosity changes, or contamination. Reveals internal wear before it becomes detectable through vibration.

## 3. C-MAPSS dataset specifics

- **What is C-MAPSS:** A NASA simulation tool (Commercial Modular Aero-Propulsion System Simulation) that models a realistic large commercial turbofan engine. It simulates engine degradation by varying health parameters (flow and efficiency) of five rotating components: Fan, LPC (Low Pressure Compressor), HPC (High Pressure Compressor), HPT (High Pressure Turbine), and LPT (Low Pressure Turbine).
- **Data structure:**
  - **Train set:** full run-to-failure trajectories for each engine unit.
  - **Test set:** trajectories truncated at some point before failure.
  - **RUL file:** ground-truth remaining cycles for each test engine unit, used only for scoring.
- **Columns:** unit number, time cycle, 3 operational settings, 21 sensor channels.
- **Sensor list:** the 21 sensors include temperatures (T2, T24, T30, T50), pressures (P2, P15, P30, Ps30), speeds (Nf, Nc, NRf, NRc), and derived quantities (EPR, BPR, fuel-air ratio, bleed enthalpy, demanded fan speeds, coolant bleeds W31 and W32).
- **FD001 vs. FD002 vs. FD003 vs. FD004 — what differs:** vary by number of fault modes and operating conditions:
  - **FD001:** 1 fault mode, 1 operating condition (simplest)
  - **FD002:** 1 fault mode, 6 operating conditions
  - **FD003:** 2 fault modes, 1 operating condition
  - **FD004:** 2 fault modes, 6 operating conditions (most challenging)
- **Why training data goes to failure but test data is truncated:** training data runs to failure so the model can learn the full degradation trajectory and what "end of life" looks like in the sensor signals. Test data is truncated to mimic the real-world scenario: you never know when a live engine will actually fail. You only have data up to the present moment, and you must predict the remaining life. This is the core RUL prediction challenge.

## 4. Framing options for the model

- **Regression on RUL**
  - **Target:** predict the exact number of remaining cycles until failure (continuous value).
  - **Pros:** rich signal — the model learns the full degradation trajectory. Standard RMSE applies. Output is directly actionable for maintenance scheduling.
  - **Cons:** difficult to get right early in engine life (high uncertainty when failure is far away). Requires careful RUL label construction — common approaches use a piecewise-linear target that caps RUL during the early "healthy" phase, since no useful degradation signal exists there.
- **Binary classification (failure within N cycles)**
  - **Target:** will this engine fail within the next N cycles? (yes/no)
  - **Pros:** simpler target. Easier to threshold for alarms. Useful when only early warning is needed.
  - **Cons:** loses granularity — you only know whether failure is near, not how much life remains. The choice of N is somewhat arbitrary and must be justified.
- **Current lean (subject to revision after EDA in Week 2):** start with **regression on RUL with a piecewise-linear target** (clipping RUL above some threshold like 125 cycles, which is standard in the C-MAPSS literature). It's the harder problem and the one published benchmarks address, so it gives the strongest comparison. A classification head can be derived from the regression output if needed.

## 5. Standard metrics

- **RMSE on RUL:** typical FD001 benchmark range — basic LSTM ~16, CNN approaches 12–13, clustering-based LSTM ~14 (Sayah et al.), deep LSTM with preprocessing ~7.78 (Asif et al., near state-of-the-art for that approach).
- **NASA scoring function:** asymmetric error metric defined per engine, then summed across all test engines. Let `d = predicted RUL − actual RUL`.
  - If `d < 0` (early prediction): `s = exp(-d / 13) − 1`
  - If `d ≥ 0` (late prediction): `s = exp(d / 10) − 1`
  - **Why asymmetric:** late predictions are operationally dangerous — if you underestimate remaining life, the engine may fail without warning. Early predictions waste some maintenance budget but are safe. The divisor of 10 (vs. 13) penalizes late predictions more heavily, and the exponential form makes large errors disproportionately costly.
- **For classification (if used):** precision, recall, F1, F2, PR-AUC. F2 weights recall higher than precision, prioritizing catching failures over avoiding false alarms — appropriate when missed failures are more costly than unnecessary inspections.

## 6. Published benchmarks (from papers I read)

- **Paper 1: Asif et al. (2022). "A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset."**
  - **Architecture:** 4-layer deep LSTM with dropout, followed by two fully connected layers and a regression output.
  - **Features:** correlation-based sensor selection (14 of 21 sensors retained for FD001); moving-median filtering for noise reduction; z-score normalization; automated piecewise-linear RUL labeling to identify the degradation start point; grid search for hyperparameters.
  - **FD001 RMSE / Score:** 7.78 / ~100 (near state-of-the-art at time of publication).

- **Paper 2: Peringal et al. (2024). "RUL Prediction for Aircraft Engines Using LSTM."**
  - **Architecture:** Single LSTM network vs. MLP baseline (PyTorch), 20-timestep input sequences, Adam optimizer
  - **Features:** Dropped 7 constant sensors, exponentially weighted moving average for smoothing, min-max normalization, 80/20 train/validation split
  - **FD001 RMSE / Score:** LSTM MSE = 796.42 (no NASA score reported) MLP MSE = 1745 - paper focused on LSTM vs MLP comparison, not benchmarch competition

- **Paper 3: Sayah et al. "Clustering-Based Deep LSTM for RUL Prediction."**
  - **Architecture:** Distribution-based clustering determines the number of LSTM layers and cells automatically - fully connected NN output. Best model: L(12.10.7.2)N(2) - 4 LSTM layers with 12,10,7,2 cells
  - **Features:** Sensor normalization, transaction based clustering via Weka, RUL cap at 130 cycles
  - **FD001 RMSE / Score:** 14.08/308 competetive across all fous sub-datasets, especially strong on FD002 and FD004

## 7. Vocabulary cheat-sheet (for interviews)

- **Prognostics:** Predicting when a component will fail or stop functioning as intended.
- **Remaining Useful Life (RUL):** Estimated cycles, hours, or operations left before failure.
- **Prognostics and Health Management (PHM):** The engineering discipline covering condition monitoring, diagnostics, and prognostics for industrial assets.
- **Run-to-failure data:** Sensor data collected continuously from healthy operation through actual failure.
- **Censored data:** Data from units that didn't reach failure during observation — failure time is unknown.
- **Health index:** A single scalar (often engineered) that summarizes overall component condition over time.
- **Degradation trajectory:** The path a component's health index takes from healthy to failed.
- **Prognostic horizon:** How far before failure a model can produce reliable predictions.
- **Operating regime:** A distinct mode of operation (e.g., cruise vs. takeoff for an aircraft engine) — sensor baselines differ across regimes.
- **Fault mode:** A specific way a component can fail (e.g., HPC degradation vs. fan degradation in C-MAPSS).
- **Condition-based maintenance (CBM):** Maintenance triggered by sensor evidence of degradation, not by a fixed schedule.
- **Time-based maintenance (TBM):** Maintenance on a fixed schedule regardless of actual condition.
- **NASA Score function:** The asymmetric scoring metric used for C-MAPSS RUL evaluation; penalizes late predictions exponentially more than early ones.
- **Walk-forward validation:** Time-series cross-validation that respects temporal order — the model is always trained on past data and evaluated on future data, never the reverse. Prevents leakage from future to past.