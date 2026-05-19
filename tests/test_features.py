"""Tests for src/features.py."""

import numpy as np
import pandas as pd
import pytest

from src import features


def test_compute_rul_target_basic():
    """RUL is correctly computed and capped."""
    df = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
        }
    )
    result = features.compute_rul_target(df, cap=125)
    assert list(result["rul"]) == [2, 1, 0, 2, 1, 0]


def test_compute_rul_target_caps_at_125():
    """RUL above the cap is clipped."""
    df = pd.DataFrame(
        {
            "unit": [1] * 200,
            "cycle": list(range(1, 201)),
        }
    )
    result = features.compute_rul_target(df, cap=125)
    assert result["rul"].max() == 125
    assert result["rul"].min() == 0
    assert (result["rul"] == 125).sum() == 75  # cycles where (199 - cycle) > 125


def test_compute_rul_target_rejects_invalid_cap():
    """Cap must be positive."""
    df = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2]})
    with pytest.raises(ValueError, match="must be >= 1"):
        features.compute_rul_target(df, cap=0)


def test_engine_level_split_disjoint():
    """No engine appears in both train and val splits."""
    df = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(1, 11), 50),  # 10 engines, 50 cycles each
            "cycle": np.tile(np.arange(1, 51), 10),
        }
    )
    train_df, val_df = features.engine_level_split(df, val_size=0.3, random_seed=42)
    train_units = set(train_df["unit"].unique())
    val_units = set(val_df["unit"].unique())
    assert train_units.isdisjoint(val_units)
    assert train_units | val_units == set(range(1, 11))


def test_pipeline_end_to_end_shapes():
    """The full pipeline produces arrays with the expected shapes for FD001."""
    pipeline = features.build_pipeline(subset="FD001")
    assert pipeline["X_train"].ndim == 3
    assert pipeline["X_train"].shape[1] == 30  # window_size
    assert pipeline["X_train"].shape[2] == 14  # n_sensors
    assert pipeline["X_train"].shape[0] == pipeline["y_train"].shape[0]
    assert pipeline["X_test"].shape[0] == 100  # 100 test engines
    assert pipeline["y_test"].max() <= 125  # capped
