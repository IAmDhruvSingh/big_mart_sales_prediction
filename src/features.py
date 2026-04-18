"""
features.py
===========
Feature engineering and sklearn preprocessing pipeline construction.

The preprocessing pipeline applies:
- StandardScaler  to numeric features.
- OneHotEncoder   to categorical features (unknown categories handled gracefully).

A ``ColumnTransformer`` combines both into a single sklearn-compatible object
that can be embedded inside a ``Pipeline`` for clean train/inference workflows.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Create and return the feature preprocessing ColumnTransformer.

    Numeric pipeline
    ----------------
    - **StandardScaler** – zero mean, unit variance (mirrors the z-score
      normalisation used in the TypeScript implementation).

    Categorical pipeline
    --------------------
    - **OneHotEncoder** – produces a binary indicator column per category.
      ``handle_unknown='ignore'`` allows unseen categories at inference time.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted transformer; call ``.fit_transform(X)`` or embed in a
        :class:`sklearn.pipeline.Pipeline`.
    """
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())],
        verbose=False,
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ],
        verbose=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    validation_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a cleaned training DataFrame into train and validation sets.

    Parameters
    ----------
    df:
        Cleaned training DataFrame containing :data:`TARGET_COLUMN`.
    validation_ratio:
        Fraction of data reserved for validation (default 0.2).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    X_train, X_val, y_train, y_val : tuple of pd.DataFrame / pd.Series
        Feature matrices and target vectors ready for modelling.

    Raises
    ------
    ValueError
        If ``Target_COLUMN`` is absent from *df*.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{TARGET_COLUMN}' not found in DataFrame. "
            "Make sure to pass a training DataFrame."
        )

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_ratio, random_state=random_state
    )

    return X_train, X_val, y_train, y_val


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable feature names after fitting the *preprocessor*.

    Parameters
    ----------
    preprocessor:
        A **fitted** :class:`~sklearn.compose.ColumnTransformer`.

    Returns
    -------
    list[str]
        Feature names in the same order as the transformer output columns.
    """
    return list(preprocessor.get_feature_names_out())
