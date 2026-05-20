"""
Model loading and prediction logic.

Decoupled from the FastAPI HTTP layer so it can be tested in isolation and
reused in batch-processing contexts.
"""

from pathlib import Path
from typing import Literal
import numpy as np
import joblib

# Try to import tensorflow lazily — only needed if LSTM endpoint is used.
# This keeps the API able to start even if TF has issues.
_tf_available = True
try:
    import tensorflow as tf
except ImportError:
    _tf_available = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ALARM_THRESHOLD = 35


def _regime_from_rul(predicted_rul: float) -> Literal["Critical", "Watch", "Healthy"]:
    if predicted_rul < 30:
        return "Critical"
    elif predicted_rul < 80:
        return "Watch"
    else:
        return "Healthy"


class InferenceEngine:
    """Loads and serves predictions from the trained models.

    Models are loaded lazily on first use to keep startup fast.
    """

    def __init__(self):
        self._xgb_model = None
        self._lstm_model = None
        self._scaler = None

    @property
    def scaler(self):
        """The StandardScaler fit on FD001 training data (Week 3)."""
        if self._scaler is None:
            self._scaler = joblib.load(PROCESSED_DIR / "fd001_scaler.joblib")
        return self._scaler

    @property
    def xgb_model(self):
        """The honest-tuned XGBoost model from Week 4."""
        if self._xgb_model is None:
            self._xgb_model = joblib.load(
                PROCESSED_DIR / "fd001_xgb_tuned_honest.joblib"
            )
        return self._xgb_model

    @property
    def lstm_model(self):
        """The LSTM-64x2 model from Week 5."""
        if self._lstm_model is None:
            if not _tf_available:
                raise RuntimeError("TensorFlow not installed — LSTM model unavailable.")
            self._lstm_model = tf.keras.models.load_model(
                PROCESSED_DIR / "fd001_lstm_64x2.keras"
            )
        return self._lstm_model

    def loaded_models(self) -> list[str]:
        """Return a list of model names that have been loaded so far."""
        loaded = []
        if self._xgb_model is not None:
            loaded.append("xgb")
        if self._lstm_model is not None:
            loaded.append("lstm")
        return loaded

    def _normalize_if_needed(
        self, window: np.ndarray, is_normalized: bool
    ) -> np.ndarray:
        """Apply scaler to a (30, 14) raw window if it's not already z-scored."""
        if is_normalized:
            return window
        # The scaler was fit on flattened sensor columns from the train df.
        # We need to apply it per-timestep, per-sensor, matching the fit shape.
        return self.scaler.transform(window)

    def _build_tabular_features(self, window_3d: np.ndarray) -> np.ndarray:
        """Reuse the same feature engineering as the XGBoost training pipeline.

        Input: (1, 30, 14) — one window
        Output: (1, 70) — flat tabular feature vector
        """
        from src.models import tabular_features_from_windows
        from src.features import KEEP_SENSORS

        return tabular_features_from_windows(window_3d, list(KEEP_SENSORS)).values

    def predict_xgb(self, window: np.ndarray, is_normalized: bool = True) -> float:
        """Predict RUL using the honest-tuned XGBoost model.

        Parameters
        ----------
        window : np.ndarray, shape (30, 14)
            Sensor window. If is_normalized=False, raw values are accepted and
            scaled server-side.
        is_normalized : bool
            Whether the window is already z-scored.

        Returns
        -------
        float
            Predicted RUL, clipped to [0, 125].
        """
        window = np.asarray(window, dtype=np.float32)
        window = self._normalize_if_needed(window, is_normalized)
        window_3d = window[np.newaxis, :, :]  # (1, 30, 14)
        features = self._build_tabular_features(window_3d)
        pred = float(self.xgb_model.predict(features)[0])
        return float(np.clip(pred, 0, 125))

    def predict_lstm(self, window: np.ndarray, is_normalized: bool = True) -> float:
        """Predict RUL using the LSTM-64x2 model.

        Same interface as predict_xgb.
        """
        window = np.asarray(window, dtype=np.float32)
        window = self._normalize_if_needed(window, is_normalized)
        window_3d = window[np.newaxis, :, :]  # (1, 30, 14)
        pred = float(self.lstm_model.predict(window_3d, verbose=0).flatten()[0])
        return float(np.clip(pred, 0, 125))

    def predict(
        self,
        window: np.ndarray,
        model_name: Literal["xgb", "lstm"],
        is_normalized: bool = True,
        engine_id: int | None = None,
    ) -> dict:
        """Predict and return a fully-formed response dict."""
        if model_name == "xgb":
            pred = self.predict_xgb(window, is_normalized)
        elif model_name == "lstm":
            pred = self.predict_lstm(window, is_normalized)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        return {
            "predicted_rul": pred,
            "alarm": pred <= ALARM_THRESHOLD,
            "alarm_threshold": ALARM_THRESHOLD,
            "regime": _regime_from_rul(pred),
            "model_name": model_name,
            "engine_id": engine_id,
        }
