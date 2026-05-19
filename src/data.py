"""
Data loading for the C-MAPSS dataset.

Provides functions to load train, test, and RUL files for each of the four
sub-datasets (FD001–FD004), with proper column naming and basic validation.
"""

from pathlib import Path
import pandas as pd

# Project root is two levels up from this file: src/data.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Column names — derived from the C-MAPSS readme.
# 3 operational settings + 21 sensor measurements.
COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_train(subset: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the training set for a given C-MAPSS sub-dataset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    data_dir : Path or str
        Directory containing the raw C-MAPSS files.

    Returns
    -------
    pd.DataFrame
        Columns: unit, cycle, op_setting_1..3, sensor_1..21.
    """
    path = Path(data_dir) / f"train_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_test(subset: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the test set for a given C-MAPSS sub-dataset."""
    path = Path(data_dir) / f"test_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_rul(subset: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the ground-truth RUL file for a given C-MAPSS sub-dataset.

    Returns
    -------
    pd.DataFrame
        Single column 'rul', indexed by unit number (1-based).
    """
    path = Path(data_dir) / f"RUL_{subset}.txt"
    df = pd.read_csv(path, header=None, names=["rul"])
    df.index = df.index + 1  # match unit numbering in train/test files
    df.index.name = "unit"
    return df
