"""
data_processing.py
==================
Functions for loading and cleaning the Big Mart Sales dataset.

Pipeline mirrors the original TypeScript preprocessing:
- Standardise Item_Fat_Content categories.
- Impute missing Item_Weight with the column mean.
- Impute missing Outlet_Size with the mode per Outlet_Type.
- Drop rows without a sales target (train only).
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAT_CONTENT_MAP: dict[str, str] = {
    "lf": "Low Fat",
    "low fat": "Low Fat",
    "reg": "Regular",
    "regular": "Regular",
}

NUMERIC_FEATURES: list[str] = [
    "Item_Weight",
    "Item_Visibility",
    "Item_MRP",
    "Outlet_Establishment_Year",
]

CATEGORICAL_FEATURES: list[str] = [
    "Item_Fat_Content",
    "Item_Type",
    "Outlet_Identifier",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]

TARGET_COLUMN: str = "Item_Outlet_Sales"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV dataset from *filepath* and return as a DataFrame.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with original column names preserved.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *filepath*.
    """
    df = pd.read_csv(filepath)
    return df


def _normalize_fat_content(series: pd.Series) -> pd.Series:
    """Standardise the Item_Fat_Content labels to 'Low Fat' or 'Regular'."""
    return series.str.strip().str.lower().map(
        lambda v: FAT_CONTENT_MAP.get(v, v.title() if v else "Unknown")
    )


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean and impute the raw Big Mart dataset.

    Steps
    -----
    1. Standardise ``Item_Fat_Content`` variants (e.g. 'lf', 'LF', 'reg').
    2. Impute ``Item_Weight`` NaNs with the global column mean.
    3. Impute ``Item_Visibility`` zeros with the per-item-type mean visibility.
    4. Impute ``Outlet_Size`` NaNs with the mode for each ``Outlet_Type``.
    5. (Train only) Drop rows where ``Item_Outlet_Sales`` is NaN.

    Parameters
    ----------
    df:
        Raw DataFrame as returned by :func:`load_data`.
    is_train:
        Set to ``True`` for the training set (drops rows missing the target).
        Set to ``False`` for the test / inference set.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input DataFrame.
    """
    df = df.copy()

    # 1. Normalise fat content
    df["Item_Fat_Content"] = _normalize_fat_content(df["Item_Fat_Content"])

    # 2. Impute missing Item_Weight with mean
    mean_weight = df["Item_Weight"].mean()
    df["Item_Weight"] = df["Item_Weight"].fillna(mean_weight)

    # 3. Replace zero visibility with per-item-type mean
    visibility_means = (
        df[df["Item_Visibility"] > 0]
        .groupby("Item_Type")["Item_Visibility"]
        .mean()
    )
    zero_mask = df["Item_Visibility"] == 0
    df.loc[zero_mask, "Item_Visibility"] = df.loc[zero_mask, "Item_Type"].map(
        visibility_means
    )
    # Fallback if still NaN
    df["Item_Visibility"] = df["Item_Visibility"].fillna(
        df["Item_Visibility"].mean()
    )

    # 4. Impute missing Outlet_Size using mode per Outlet_Type
    outlet_size_mode = (
        df.groupby("Outlet_Type")["Outlet_Size"]
        .apply(lambda s: s.mode()[0] if not s.mode().empty else "Small")
    )
    missing_size = df["Outlet_Size"].isna()
    df.loc[missing_size, "Outlet_Size"] = df.loc[
        missing_size, "Outlet_Type"
    ].map(outlet_size_mode)
    df["Outlet_Size"] = df["Outlet_Size"].fillna("Small")

    # 5. Drop rows with missing target (training set only)
    if is_train and TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])

    return df
