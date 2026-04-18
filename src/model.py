"""
model.py
========
Model training, evaluation, and persistence for Big Mart Sales Prediction.

Supports Ridge Regression (default, alpha=0.8 as in the TypeScript original)
and can be easily extended with additional estimators.

Typical usage
-------------
>>> from src.data_processing import load_data, clean_data
>>> from src.features import build_preprocessor, split_data
>>> from src.model import build_pipeline, train_model, evaluate_model, save_model
>>>
>>> df = clean_data(load_data("data/Train.csv"))
>>> X_train, X_val, y_train, y_val = split_data(df)
>>> pipeline = build_pipeline()
>>> pipeline = train_model(pipeline, X_train, y_train)
>>> metrics = evaluate_model(pipeline, X_val, y_val)
>>> save_model(pipeline, "models/ridge_pipeline.joblib")
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, get_feature_names


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ModelMetrics = dict[str, float | int | list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pipeline(alpha: float = 0.8) -> Pipeline:
    """Create an end-to-end sklearn Pipeline: preprocessor + Ridge Regressor.

    The ``alpha`` regularisation parameter matches the lambda value (0.8) used
    in the TypeScript Ridge Regression implementation.

    Parameters
    ----------
    alpha:
        Ridge regularisation strength. Higher values increase regularisation.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for training.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )
    return pipeline


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the *pipeline* on the training data and return the fitted pipeline.

    Parameters
    ----------
    pipeline:
        Unfitted sklearn Pipeline as returned by :func:`build_pipeline`.
    X_train:
        Feature DataFrame (numeric + categorical columns).
    y_train:
        Target series (``Item_Outlet_Sales``).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_top_features: int = 6,
) -> ModelMetrics:
    """Evaluate a fitted *pipeline* on validation data.

    Computes RMSE, R² score, and the top-N most influential features by
    absolute coefficient magnitude (mirrors the TypeScript metrics panel).

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    X_val:
        Validation feature DataFrame.
    y_val:
        Validation target series.
    n_top_features:
        Number of top features to report.

    Returns
    -------
    dict
        ``rmse``, ``r2``, ``val_count``, ``important_features`` keys.
    """
    y_pred = np.maximum(0, pipeline.predict(X_val))

    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))

    # Extract feature importances from the Ridge coefficients
    preprocessor = pipeline.named_steps["preprocessor"]
    coefficients = pipeline.named_steps["regressor"].coef_
    feature_names = get_feature_names(preprocessor)

    feature_importance = sorted(
        [
            {"feature": name, "coefficient": float(coef)}
            for name, coef in zip(feature_names, coefficients)
        ],
        key=lambda x: abs(x["coefficient"]),
        reverse=True,
    )[:n_top_features]

    return {
        "rmse": rmse,
        "r2": r2,
        "val_count": int(len(y_val)),
        "important_features": feature_importance,
    }


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Persist the fitted *pipeline* to disk using joblib.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    filepath:
        Destination path (e.g. ``"models/ridge_pipeline.joblib"``).
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Pipeline:
    """Load a previously saved pipeline from *filepath*.

    Parameters
    ----------
    filepath:
        Path to a ``.joblib`` file created by :func:`save_model`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for inference.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at: {filepath}")
    return joblib.load(filepath)
