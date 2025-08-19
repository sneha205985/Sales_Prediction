from __future__ import annotations

from typing import Tuple, List

import pandas as pd


def load_advertising_csv(csv_path: str) -> pd.DataFrame:
    """Load the advertising dataset.

    Expects columns: TV, Radio, Newspaper, Sales
    """
    dataframe = pd.read_csv(csv_path)
    expected_columns: List[str] = ["TV", "Radio", "Newspaper", "Sales"]
    missing_columns = [col for col in expected_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}. Found: {list(dataframe.columns)}")
    return dataframe


def split_features_and_target(
    dataframe: pd.DataFrame, target_column: str = "Sales"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split into features X and target y.

    Returns X with feature columns in the order: TV, Radio, Newspaper to
    ensure consistent training and prediction.
    """
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not in dataframe columns: {list(dataframe.columns)}")
    feature_columns: List[str] = [col for col in ["TV", "Radio", "Newspaper"] if col in dataframe.columns]
    X = dataframe[feature_columns].copy()
    y = dataframe[target_column].copy()
    return X, y

