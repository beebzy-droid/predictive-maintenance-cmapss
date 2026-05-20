"""
Slim inference module for the deployed HF Space (XGBoost only).

Self-contained — does not import from src/. Designed to be the only model
code that needs to ship to the deployment environment.
"""

from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).resolve().parent / "models"
ALARM_THRESHOLD = 35


# Sensors retained from FD001, in the order used by the trained scaler/model.
# Matches src.features.KEEP_SENSORS in the main project.
KEEP_SENSORS = [
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
]


def _regime_from_rul(predicted_rul: float) -> Literal["Critical", "Watch", "Healthy"]:
    if predicted_rul < 30:
        return "Critical"
    elif predicted_rul < 80:
        return "Watch"
    else:
        return "Healthy"


def _build_tabular_features(window_3d: np.ndarray) -> pd.DataFrame:
    """5 statistics × 14 sensors = 70 features per window.

    Copy of src.models.tabular_features_from_windows, inlined here so the
    deployment doesn't need to import from src/.
    """
    n_windows, n_timesteps, n_sensors = window_3d.shape
    assert n_sensors == len(KEEP_SENSORS)

    # Easy stats
    means = window_3d.mean(axis=1)
    stds = window_3d.std(axis=1, ddof=0)
    mins = window_3d.min(axis=1)
    maxs = window_3d.max(axis=1)

    # Vectorized linear-regression slope per (window, sensor)
    t = np.arange(n_timesteps)
    t_centered = t - t.mean()
    t_var = (t_centered**2).mean()
    y_centered = window_3d - window_3d.mean(axis=1, keepdims=True)
    slopes = (t_centered.reshape(1, -1, 1) * y_centered).mean(axis=1) / t_var

    # Assemble into 2D matrix
    feature_blocks = []
    column_names = []
    for s_idx, s_name in enumerate(KEEP_SENSORS):
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

    return pd.DataFrame(np.concatenate(feature_blocks, axis=1), columns=column_names)


class InferenceEngine:
    """Loads and serves XGBoost predictions. Model loads lazily on first request."""

    def __init__(self):
        self._xgb_model = None
        self._scaler = None

    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = joblib.load(MODELS_DIR / "fd001_scaler.joblib")
        return self._scaler

    @property
    def xgb_model(self):
        if self._xgb_model is None:
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(MODELS_DIR / "fd001_xgb_tuned_honest.json")
            self._xgb_model = model
        return self._xgb_model

    def predict_xgb(self, window: np.ndarray, is_normalized: bool = True) -> dict:
        """Predict RUL for one 30x14 window using XGBoost.

        Returns a dict with prediction, alarm decision, and regime label.
        """
        window = np.asarray(window, dtype=np.float32)
        if not is_normalized:
            window = self.scaler.transform(window)
        window_3d = window[np.newaxis, :, :]
        features = _build_tabular_features(window_3d)
        raw_pred = float(self.xgb_model.predict(features)[0])
        pred = float(np.clip(raw_pred, 0, 125))
        return {
            "predicted_rul": pred,
            "alarm": pred <= ALARM_THRESHOLD,
            "alarm_threshold": ALARM_THRESHOLD,
            "regime": _regime_from_rul(pred),
        }
