"""
Evaluation metrics for C-MAPSS RUL prediction.

Provides:
- rmse: standard regression metric (square root of mean squared error)
- nasa_score: the asymmetric scoring function from Saxena & Goebel (2008),
  which penalizes late predictions exponentially more than early ones.

Both functions accept arrays/lists of any compatible shape, but expect 1D
sequences of true and predicted RUL values.
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted RUL values. Must have the same length.

    Returns
    -------
    float
        RMSE in cycles (same units as input).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred) -> float:
    """NASA C-MAPSS asymmetric score function (Saxena & Goebel 2008).

    For each prediction, compute d = predicted - actual. The score component is:
      - exp(-d/13) - 1     if d < 0  (early — model predicted failure too soon)
      - exp( d/10) - 1     if d >= 0 (late  — model missed the warning window)

    The asymmetry (divisor 10 vs 13) penalizes late predictions ~48% more than
    early predictions of equal magnitude. Total score is the sum across samples.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted RUL values. Must have the same length.

    Returns
    -------
    float
        Total score across all samples. Lower is better. Perfect predictions
        give a score of 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    d = y_pred - y_true
    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,
        np.exp(d / 10.0) - 1,
    )
    return float(score.sum())


def clip_predictions(y_pred, lower: float = 0.0, upper: float = 125.0) -> np.ndarray:
    """Clip predictions to the valid RUL range.

    Models can predict negative or super-cap values; this enforces the
    physical/operational range [0, 125]. Same clip is applied during evaluation
    in all classical baselines so the metric comparison is fair.
    """
    return np.clip(np.asarray(y_pred, dtype=float), lower, upper)
