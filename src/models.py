"""
Classical baseline models for C-MAPSS RUL prediction.

Provides factory functions for the four model configurations evaluated in
Week 4 of the project:

- XGBoost (default hyperparameters)
- XGBoost (honest-tuned via engine-level GroupKFold + Optuna in the notebook)
- Random Forest (default)
- Random Forest (regularized: max_depth=15, min_samples_leaf=5)

Each factory returns an untrained estimator. Callers are responsible for
fitting on the tabular feature matrix produced by tabular_features_from_windows.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Hyperparameters from Week 4 Task 3b — found via Optuna with engine-level
# GroupKFold cross-validation on FD001. See notebooks/03_baseline_models.ipynb
# and docs/decisions.md for the derivation and the leakage bug that motivated
# the engine-level approach.
XGB_HONEST_TUNED_PARAMS = {
    "n_estimators": 532,
    "max_depth": 3,
    "learning_rate": 0.01865366775523775,
    "subsample": 0.7730982444721965,
    "colsample_bytree": 0.7557510462462189,
    "reg_alpha": 0.0005396862681589784,
    "reg_lambda": 4.4926651852775794e-05,
    "min_child_weight": 4,
}


def build_xgb_default(random_state: int = 42) -> xgb.XGBRegressor:
    """XGBoost with sensible default hyperparameters (no tuning)."""
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgb_tuned(random_state: int = 42) -> xgb.XGBRegressor:
    """XGBoost with hyperparameters from honest engine-level GroupKFold tuning.

    These hyperparameters were found by Optuna in Week 4 Task 3b, using a
    4-fold GroupKFold CV that splits at the engine level. See docs/decisions.md
    for the discussion of why window-level splits caused leakage.
    """
    return xgb.XGBRegressor(
        **XGB_HONEST_TUNED_PARAMS,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )


def build_rf_default(random_state: int = 42) -> RandomForestRegressor:
    """Random Forest with library defaults (unlimited depth, single-sample leaves)."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )


def build_rf_regularized(random_state: int = 42) -> RandomForestRegressor:
    """Random Forest with sensible regularization for tabular regression."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


def tabular_features_from_windows(X_3d: np.ndarray, sensor_names: list) -> pd.DataFrame:
    """Convert 3D windowed sensor data into a 2D tabular feature matrix.

    For each window and each sensor, compute 5 summary statistics:
    mean, std, slope, min, max. The slope is the linear-regression slope
    of sensor values against the timestep index, computed in a single
    vectorized pass.

    Parameters
    ----------
    X_3d : np.ndarray, shape (n_windows, timesteps, n_sensors)
    sensor_names : list of str
        Names of the sensors. Length must equal n_sensors.

    Returns
    -------
    pd.DataFrame, shape (n_windows, 5 * n_sensors)
        Column names follow the convention '{sensor}_{stat}' for stat
        in [mean, std, slope, min, max].
    """
    n_windows, n_timesteps, n_sensors = X_3d.shape
    if n_sensors != len(sensor_names):
        raise ValueError(
            f"sensor_names length {len(sensor_names)} doesn't match "
            f"n_sensors {n_sensors}"
        )

    # Easy stats: collapse along the timestep axis
    means = X_3d.mean(axis=1)
    stds = X_3d.std(axis=1, ddof=0)
    mins = X_3d.min(axis=1)
    maxs = X_3d.max(axis=1)

    # Vectorized slope: closed-form linear regression
    # slope = Cov(t, y) / Var(t), where t = [0, 1, ..., timesteps-1]
    t = np.arange(n_timesteps)
    t_centered = t - t.mean()
    t_var = (t_centered**2).mean()
    y_centered = X_3d - X_3d.mean(axis=1, keepdims=True)
    slopes = (t_centered.reshape(1, -1, 1) * y_centered).mean(axis=1) / t_var

    # Assemble: per sensor, [mean, std, slope, min, max]
    feature_blocks = []
    column_names = []
    for s_idx, s_name in enumerate(sensor_names):
        feature_blocks.append(
            np.column_stack(
                [
                    means[:, s_idx],
                    stds[:, s_idx],
                    slopes[:, s_idx],
                    mins[:, s_idx],
                    maxs[:, s_idx],
                ]
            )
        )
        column_names.extend(
            [
                f"{s_name}_mean",
                f"{s_name}_std",
                f"{s_name}_slope",
                f"{s_name}_min",
                f"{s_name}_max",
            ]
        )

    feature_matrix = np.concatenate(feature_blocks, axis=1)
    return pd.DataFrame(feature_matrix, columns=column_names)
