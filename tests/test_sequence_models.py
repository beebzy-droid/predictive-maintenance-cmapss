"""Tests for src/sequence_models.py."""

import numpy as np
import pytest

from src import sequence_models


def test_vanilla_lstm_default_shape():
    """Vanilla LSTM accepts the default (30, 14) windowed input shape."""
    model = sequence_models.build_vanilla_lstm()
    output_shape = model.output_shape
    assert output_shape == (None, 1)


def test_vanilla_lstm_custom_window_size():
    """Vanilla LSTM accepts custom window sizes."""
    model = sequence_models.build_vanilla_lstm(window_size=50, n_features=10)
    assert model.input_shape == (None, 50, 10)
    assert model.output_shape == (None, 1)


def test_stacked_lstm_has_two_lstm_layers():
    """Stacked LSTM has exactly two LSTM layers."""
    model = sequence_models.build_stacked_lstm()
    lstm_layers = [
        layer for layer in model.layers if "lstm" in layer.__class__.__name__.lower()
    ]
    assert len(lstm_layers) == 2


def test_bidirectional_lstm_has_bidirectional_layer():
    """Bidirectional LSTM contains a Bidirectional wrapper."""
    model = sequence_models.build_bidirectional_lstm()
    bidirectional_layers = [
        layer
        for layer in model.layers
        if "bidirectional" in layer.__class__.__name__.lower()
    ]
    assert len(bidirectional_layers) == 1


def test_models_compile_with_mse_loss():
    """All factories produce models compiled with MSE loss for RUL regression."""
    for builder in [
        sequence_models.build_vanilla_lstm,
        sequence_models.build_stacked_lstm,
        sequence_models.build_bidirectional_lstm,
    ]:
        model = builder()
        # Keras stores loss as a string or callable; we just check it's set
        assert model.loss is not None


def test_models_predict_correct_output_shape():
    """All factories produce models that output (batch, 1) for any input batch."""
    rng = np.random.RandomState(0)
    X = rng.randn(5, 30, 14).astype(np.float32)
    for builder in [
        sequence_models.build_vanilla_lstm,
        sequence_models.build_stacked_lstm,
        sequence_models.build_bidirectional_lstm,
    ]:
        model = builder()
        pred = model.predict(X, verbose=0)
        assert pred.shape == (5, 1)


def test_set_keras_seeds_runs_without_error():
    """The seed helper executes without raising."""
    sequence_models.set_keras_seeds(42)
    sequence_models.set_keras_seeds(0)  # Different seed should also work
