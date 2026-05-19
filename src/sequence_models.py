"""
Sequence models for C-MAPSS RUL prediction.

Provides factory functions for the LSTM architectures evaluated in Week 5:
- Vanilla single-layer LSTM (64 units)
- Single-layer LSTM with 128 units
- Stacked 2-layer LSTM (64+64 units) — the reported best
- Bidirectional LSTM (64 and 128 unit variants)

Each factory returns a compiled but untrained Keras model. Callers are
responsible for fitting via model.fit() on the windowed input arrays.
"""

from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam


def build_vanilla_lstm(
    window_size: int = 30,
    n_features: int = 14,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """Single-layer LSTM for RUL regression.

    Architecture:
        LSTM(units, return_sequences=False) → Dropout → Dense(ReLU) → Dense(1, linear)
    """
    model = Sequential(
        [
            Input(shape=(window_size, n_features)),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )
    return model


def build_stacked_lstm(
    window_size: int = 30,
    n_features: int = 14,
    lstm_units_1: int = 64,
    lstm_units_2: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """Two stacked LSTM layers — the reported best deep architecture for FD001.

    The Week 5 evaluation found this architecture (64+64 units) gave the most
    reproducible Test RMSE (12.66 / Score 290.7) across the four variants tested.
    """
    model = Sequential(
        [
            Input(shape=(window_size, n_features)),
            LSTM(lstm_units_1, return_sequences=True),
            Dropout(dropout),
            LSTM(lstm_units_2, return_sequences=False),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )
    return model


def build_bidirectional_lstm(
    window_size: int = 30,
    n_features: int = 14,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """Single bidirectional LSTM layer.

    The bidirectional wrapper doubles the effective unit count (forward + backward
    pass). Tested in Week 5 with 64 and 128 unit variants; neither outperformed
    the stacked LSTM-64×2.
    """
    model = Sequential(
        [
            Input(shape=(window_size, n_features)),
            Bidirectional(LSTM(lstm_units, return_sequences=False)),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )
    return model


def set_keras_seeds(seed: int = 42) -> None:
    """Set seeds for TensorFlow and numpy. Does NOT guarantee full determinism
    on CPU (some TF ops are inherently non-deterministic across parallel threads).
    Use as a reasonable-effort reproducibility helper, not a strict guarantee.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
