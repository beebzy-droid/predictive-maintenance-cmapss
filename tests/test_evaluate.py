"""Tests for src/evaluate.py."""

import numpy as np
import pytest

from src import evaluate


def test_rmse_basic():
    """RMSE returns sqrt(MSE) for a simple case."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # MSE = (4 + 4 + 9) / 3 = 17/3 ≈ 5.667
    expected = np.sqrt(17.0 / 3.0)
    assert np.isclose(evaluate.rmse(y_true, y_pred), expected)


def test_rmse_perfect_prediction():
    """Perfect predictions give zero RMSE."""
    y = np.array([1.0, 2.0, 3.0])
    assert evaluate.rmse(y, y) == 0.0


def test_nasa_score_perfect():
    """Perfect predictions give zero score."""
    y = np.array([10.0, 20.0, 30.0])
    assert evaluate.nasa_score(y, y) == 0.0


def test_nasa_score_early_branch():
    """Early prediction (d < 0) uses divisor 13."""
    y_true = np.array([30.0])
    y_pred = np.array([20.0])  # d = -10
    expected = np.exp(10.0 / 13.0) - 1
    assert np.isclose(evaluate.nasa_score(y_true, y_pred), expected)


def test_nasa_score_late_branch():
    """Late prediction (d > 0) uses divisor 10."""
    y_true = np.array([20.0])
    y_pred = np.array([30.0])  # d = +10
    expected = np.exp(10.0 / 10.0) - 1
    assert np.isclose(evaluate.nasa_score(y_true, y_pred), expected)


def test_nasa_score_asymmetry():
    """Late errors are penalized more heavily than early errors of equal magnitude."""
    y_true = np.array([20.0])
    score_early = evaluate.nasa_score(y_true, np.array([10.0]))
    score_late = evaluate.nasa_score(y_true, np.array([30.0]))
    # Late should be roughly 1.48x more expensive than early for |error|=10
    assert score_late > score_early
    assert np.isclose(score_late / score_early, 1.484, atol=0.01)


def test_nasa_score_shape_mismatch_raises():
    """Mismatched input shapes raise ValueError."""
    with pytest.raises(ValueError, match="same shape"):
        evaluate.nasa_score(np.array([1.0, 2.0]), np.array([1.0]))


def test_clip_predictions_default_range():
    """Predictions are clipped to [0, 125]."""
    preds = np.array([-10.0, 50.0, 200.0])
    clipped = evaluate.clip_predictions(preds)
    assert (clipped == np.array([0.0, 50.0, 125.0])).all()


def test_clip_predictions_custom_range():
    """Custom clip bounds are respected."""
    preds = np.array([-10.0, 50.0, 200.0])
    clipped = evaluate.clip_predictions(preds, lower=10, upper=100)
    assert (clipped == np.array([10.0, 50.0, 100.0])).all()
