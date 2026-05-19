"""
Data loading for the C-MAPSS dataset.

Provides functions to load train, test, and RUL files for each of the four
sub-datasets (FD001–FD004), with proper column naming and input validation.
"""

from pathlib import Path
import pandas as pd

# Project root is two levels up from this file: src/data.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# The four valid C-MAPSS sub-datasets.
VALID_SUBSETS = frozenset({"FD001", "FD002", "FD003", "FD004"})

# Column names — derived from the C-MAPSS readme.
# 3 operational settings + 21 sensor measurements.
COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _validate_subset(subset: str) -> str:
    """Validate and normalize a C-MAPSS subset identifier.

    Parameters
    ----------
    subset : str
        Subset name (case-insensitive). One of 'FD001', 'FD002', 'FD003', 'FD004'.

    Returns
    -------
    str
        The normalized (uppercase) subset name.

    Raises
    ------
    TypeError
        If subset is not a string.
    ValueError
        If subset is not one of the four valid C-MAPSS sub-datasets.
    """
    if not isinstance(subset, str):
        raise TypeError(f"subset must be a string, got {type(subset).__name__}")
    normalized = subset.upper()
    if normalized not in VALID_SUBSETS:
        raise ValueError(
            f"subset must be one of {sorted(VALID_SUBSETS)}, got {subset!r}"
        )
    return normalized


def load_train(subset: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the training set for a given C-MAPSS sub-dataset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004' (case-insensitive).
    data_dir : Path or str
        Directory containing the raw C-MAPSS files.

    Returns
    -------
    pd.DataFrame
        Columns: unit, cycle, op_setting_1..3, sensor_1..21.

    Raises
    ------
    ValueError
        If subset is not one of the four valid C-MAPSS sub-datasets.
    FileNotFoundError
        If the data file for the given subset cannot be found in data_dir.
    """
    subset = _validate_subset(subset)
    path = Path(data_dir) / f"train_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_test(subset: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the test set for a given C-MAPSS sub-dataset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004' (case-insensitive).
    data_dir : Path or str
        Directory containing the raw C-MAPSS files.

    Returns
    -------
    pd.DataFrame
        Columns: unit, cycle, op_setting_1..3, sensor_1..21.

    Raises
    ------
    ValueError
        If subset is not one of the four valid C-MAPSS sub-datasets.
    FileNotFoundError
        If the data file for the given subset cannot be found in data_dir.
    """
    subset = _validate_subset(subset)
    path = Path(data_dir) / f"test_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_rul(subset: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the ground-truth RUL file for a given C-MAPSS sub-dataset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004' (case-insensitive).
    data_dir : Path or str
        Directory containing the raw C-MAPSS files.

    Returns
    -------
    pd.DataFrame
        Single column 'rul', indexed by unit number (1-based).

    Raises
    ------
    ValueError
        If subset is not one of the four valid C-MAPSS sub-datasets.
    FileNotFoundError
        If the data file for the given subset cannot be found in data_dir.
    """
    subset = _validate_subset(subset)
    path = Path(data_dir) / f"RUL_{subset}.txt"
    df = pd.read_csv(path, header=None, names=["rul"])
    df.index = df.index + 1  # match unit numbering in train/test files
    df.index.name = "unit"
    return df
