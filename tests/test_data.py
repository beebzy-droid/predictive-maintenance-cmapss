"""Tests for src/data.py."""

import pytest
import pandas as pd

from src import data


def test_load_train_returns_dataframe():
    """load_train should return a DataFrame with the expected shape and columns."""
    df = data.load_train("FD001")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20631, 26)
    assert list(df.columns) == data.COLUMN_NAMES


def test_load_train_case_insensitive():
    """load_train should accept lowercase subset names."""
    df_upper = data.load_train("FD001")
    df_lower = data.load_train("fd001")
    pd.testing.assert_frame_equal(df_upper, df_lower)


def test_load_train_invalid_subset_raises():
    """Invalid subset names should raise ValueError with a helpful message."""
    with pytest.raises(ValueError, match="must be one of"):
        data.load_train("FD005")


def test_load_train_non_string_subset_raises():
    """Non-string subset values should raise TypeError."""
    with pytest.raises(TypeError, match="must be a string"):
        data.load_train(42)


def test_load_rul_index_starts_at_one():
    """load_rul should return a DataFrame indexed by unit, starting from 1."""
    df = data.load_rul("FD001")
    assert df.index[0] == 1
    assert df.index.name == "unit"
    assert df.shape == (100, 1)
