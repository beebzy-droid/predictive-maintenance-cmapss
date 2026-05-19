"""
Feature engineering pipeline for C-MAPSS RUL prediction.

Provides functions to:
- compute the piecewise-linear RUL target
- split engines at the engine-id level (no row-level leakage)
- fit z-score normalization on train only and apply to all splits
- construct sliding windows for both sequence models (LSTM) and tabular
  models (XGBoost, via window summaries downstream).

The main entry point is `build_pipeline()` which returns the six arrays
ready for modeling: X_train, y_train, X_val, y_val, X_test, y_test.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import load_train, load_test, load_rul

# Decisions from Week 2 EDA — see docs/decisions.md
KEEP_SENSORS = (
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
)
DEFAULT_RUL_CAP = 125
DEFAULT_WINDOW_SIZE = 30
DEFAULT_VAL_SIZE = 0.20
DEFAULT_RANDOM_SEED = 42


def compute_rul_target(df: pd.DataFrame, cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Compute the piecewise-linear RUL target.

    For each engine, the lifetime is the maximum cycle observed. RUL at each
    row is (lifetime − current cycle), then clipped at `cap` from above.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit' and 'cycle' columns.
    cap : int
        Upper limit on RUL. Values above are clipped down.

    Returns
    -------
    pd.DataFrame
        Copy of input with added 'rul' column (the capped target).
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    df = df.copy()
    lifetime = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (lifetime - df["cycle"]).clip(upper=cap)
    return df


def engine_level_split(
    df: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple:
    """Split engines into train and validation sets.

    All cycles of a given engine go to the same side — no engine appears
    in both splits. Prevents row-level leakage.

    Returns
    -------
    train_df, val_df : pd.DataFrame
        Disjoint subsets of the input.
    """
    if not 0 < val_size < 1:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    all_units = df["unit"].unique()
    train_units, val_units = train_test_split(
        all_units, test_size=val_size, random_state=random_seed
    )
    train_df = df[df["unit"].isin(train_units)].copy()
    val_df = df[df["unit"].isin(val_units)].copy()
    return train_df, val_df


def fit_normalizer(
    train_df: pd.DataFrame, feature_cols: tuple = KEEP_SENSORS
) -> StandardScaler:
    """Fit a z-score normalizer on training data only.

    The returned scaler can be applied to val and test data via
    `scaler.transform(df[feature_cols])`. Never refit on other splits —
    that would leak information.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[list(feature_cols)])
    return scaler


def apply_normalizer(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: tuple = KEEP_SENSORS,
) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame, returning a normalized copy."""
    df = df.copy()
    df[list(feature_cols)] = scaler.transform(df[list(feature_cols)])
    return df


def make_windows_for_engine(
    engine_df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    feature_cols: tuple = KEEP_SENSORS,
) -> tuple:
    """Build sliding windows for a single engine.

    Each window covers `window_size` consecutive cycles. The target (if a
    'rul' column is present) is the RUL value at the last cycle of the window.
    Returns empty arrays if the engine has fewer than `window_size` cycles.

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size, n_features)
    y : np.ndarray, shape (n_windows,) or None
    """
    n_cycles = len(engine_df)
    n_features = len(feature_cols)

    if n_cycles < window_size:
        return np.empty((0, window_size, n_features)), (
            np.empty(0) if "rul" in engine_df.columns else None
        )

    feature_data = engine_df[list(feature_cols)].values
    has_rul = "rul" in engine_df.columns
    rul_data = engine_df["rul"].values if has_rul else None

    n_windows = n_cycles - window_size + 1
    X = np.empty((n_windows, window_size, n_features))
    y = np.empty(n_windows) if has_rul else None

    for i in range(n_windows):
        X[i] = feature_data[i : i + window_size]
        if has_rul:
            y[i] = rul_data[i + window_size - 1]

    return X, y


def build_train_val_arrays(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    feature_cols: tuple = KEEP_SENSORS,
) -> tuple:
    """Build full windowed dataset (all engines stacked) for training or validation."""
    all_X, all_y = [], []
    for unit_id in sorted(df["unit"].unique()):
        engine_df = df[df["unit"] == unit_id]
        X_eng, y_eng = make_windows_for_engine(engine_df, window_size, feature_cols)
        if len(X_eng) > 0:
            all_X.append(X_eng)
            all_y.append(y_eng)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    return X, y


def build_test_arrays(
    df: pd.DataFrame,
    rul_truth: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    feature_cols: tuple = KEEP_SENSORS,
    cap: int = DEFAULT_RUL_CAP,
) -> tuple:
    """Build test arrays: one window per engine (the engine's last `window_size` cycles).

    Test labels are taken from `rul_truth` (NASA-provided ground truth, one RUL value
    per engine) and capped at `cap` to match the training target distribution.

    Returns
    -------
    X_test : np.ndarray, shape (n_engines, window_size, n_features)
    y_test : np.ndarray, shape (n_engines,) — capped at `cap`
    unit_ids : np.ndarray, shape (n_engines,) — engine IDs in the order of X_test rows
    """
    all_X, all_units = [], []
    for unit_id in sorted(df["unit"].unique()):
        engine_df = df[df["unit"] == unit_id]
        if len(engine_df) < window_size:
            continue
        last_window = engine_df[list(feature_cols)].values[-window_size:]
        all_X.append(last_window)
        all_units.append(unit_id)
    X_test = np.stack(all_X, axis=0)
    unit_ids = np.array(all_units)
    y_test = np.clip(rul_truth.loc[unit_ids, "rul"].values, a_min=None, a_max=cap)
    return X_test, y_test, unit_ids


def build_pipeline(
    subset: str = "FD001",
    rul_cap: int = DEFAULT_RUL_CAP,
    window_size: int = DEFAULT_WINDOW_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    feature_cols: tuple = KEEP_SENSORS,
) -> dict:
    """End-to-end pipeline: from raw C-MAPSS files to model-ready arrays.

    Returns a dict with the six arrays plus the fitted scaler and test unit IDs.
    """
    # Load raw data
    train_raw = load_train(subset)
    test_raw = load_test(subset)
    rul_truth = load_rul(subset)

    # Compute RUL target on train (test labels come from rul_truth)
    train_with_rul = compute_rul_target(train_raw, cap=rul_cap)

    # Engine-level split
    train_df, val_df = engine_level_split(train_with_rul, val_size, random_seed)

    # Fit normalizer on train ONLY, apply to all
    scaler = fit_normalizer(train_df, feature_cols)
    train_df = apply_normalizer(train_df, scaler, feature_cols)
    val_df = apply_normalizer(val_df, scaler, feature_cols)
    test_df = apply_normalizer(test_raw, scaler, feature_cols)

    # Build windowed arrays
    X_train, y_train = build_train_val_arrays(train_df, window_size, feature_cols)
    X_val, y_val = build_train_val_arrays(val_df, window_size, feature_cols)
    X_test, y_test, test_unit_ids = build_test_arrays(
        test_df, rul_truth, window_size, feature_cols, cap=rul_cap
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "test_unit_ids": test_unit_ids,
        "scaler": scaler,
    }
