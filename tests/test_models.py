"""Tests for src/models.py."""

import numpy as np
import pandas as pd
import pytest

from src import models


def test_tabular_features_shape():
    """Window array (N, T, S) becomes feature matrix (N, 5*S)."""
    X_3d = np.random.RandomState(0).randn(50, 30, 14)
    sensor_names = [f"sensor_{i}" for i in range(14)]
    df = models.tabular_features_from_windows(X_3d, sensor_names)
    assert df.shape == (50, 70)


def test_tabular_features_column_names():
    """Column names follow '{sensor}_{stat}' convention in order."""
    X_3d = np.random.RandomState(0).randn(3, 30, 2)
    df = models.tabular_features_from_windows(X_3d, ["sensor_a", "sensor_b"])
    expected_cols = [
        "sensor_a_mean",
        "sensor_a_std",
        "sensor_a_slope",
        "sensor_a_min",
        "sensor_a_max",
        "sensor_b_mean",
        "sensor_b_std",
        "sensor_b_slope",
        "sensor_b_min",
        "sensor_b_max",
    ]
    assert list(df.columns) == expected_cols


def test_tabular_features_mean_correctness():
    """The mean feature equals the actual mean along the timestep axis."""
    rng = np.random.RandomState(42)
    X_3d = rng.randn(5, 30, 3)
    df = models.tabular_features_from_windows(X_3d, ["s0", "s1", "s2"])
    expected_mean_s0 = X_3d[:, :, 0].mean(axis=1)
    assert np.allclose(df["s0_mean"].values, expected_mean_s0)


def test_tabular_features_slope_for_constant_signal():
    """A constant signal has slope = 0."""
    X_3d = np.ones((10, 30, 4))  # all values identical
    df = models.tabular_features_from_windows(X_3d, ["s0", "s1", "s2", "s3"])
    assert np.allclose(df["s0_slope"].values, 0.0)
    assert np.allclose(df["s3_slope"].values, 0.0)


def test_tabular_features_slope_for_linear_signal():
    """A signal y = t has slope = 1."""
    # Each window is the same linear ramp [0, 1, 2, ..., 29]
    ramp = np.arange(30).reshape(1, 30, 1).astype(float)
    X_3d = np.tile(ramp, (5, 1, 2))  # 5 windows, 2 sensors, all identical ramps
    df = models.tabular_features_from_windows(X_3d, ["s0", "s1"])
    assert np.allclose(df["s0_slope"].values, 1.0)
    assert np.allclose(df["s1_slope"].values, 1.0)


def test_tabular_features_sensor_name_mismatch_raises():
    """Wrong number of sensor names raises ValueError."""
    X_3d = np.random.RandomState(0).randn(3, 30, 5)
    with pytest.raises(ValueError, match="doesn't match"):
        models.tabular_features_from_windows(X_3d, ["only", "two"])


def test_build_xgb_default_is_untrained():
    """Factory returns an untrained estimator."""
    model = models.build_xgb_default()
    assert not hasattr(model, "n_features_in_") or model.n_features_in_ is None


def test_build_xgb_tuned_uses_honest_params():
    """Tuned XGBoost has the documented hyperparameters."""
    model = models.build_xgb_tuned()
    params = model.get_params()
    assert params["n_estimators"] == 532
    assert params["max_depth"] == 3


def test_build_rf_regularized_has_depth_cap():
    """Regularized RF has documented regularization settings."""
    model = models.build_rf_regularized()
    params = model.get_params()
    assert params["max_depth"] == 15
    assert params["min_samples_leaf"] == 5
