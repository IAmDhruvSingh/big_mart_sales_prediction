"""
predict.py
==========
Inference script for Big Mart Sales Prediction.

Can be used as a module or run directly from the command line to generate
predictions on the held-out test set (``data/test.csv``).

CLI Usage
---------
Train the model first, then run:

    python -m src.predict

The script will:
1. Load the saved model from ``models/ridge_pipeline.joblib``.
2. Load and clean ``data/test.csv``.
3. Generate predictions and save them to ``reports/predictions.csv``.

Module Usage
------------
>>> from src.predict import load_and_predict
>>> preds = load_and_predict("data/test.csv", "models/ridge_pipeline.joblib")
>>> print(preds.head())
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_data,
    clean_data,
)
from src.model import load_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_sales(
    input_data: pd.DataFrame,
    model_path: str = "models/ridge_pipeline.joblib",
) -> np.ndarray:
    """Generate sales predictions for a DataFrame of product-outlet records.

    The function loads the saved pipeline and runs inference. Predictions are
    clipped to a minimum of 0 (no negative sales).

    Parameters
    ----------
    input_data:
        DataFrame containing the same feature columns used during training
        (``Item_Weight``, ``Item_MRP``, ``Outlet_Type``, etc.).
    model_path:
        Path to the saved ``.joblib`` pipeline file.

    Returns
    -------
    np.ndarray
        Array of predicted Item_Outlet_Sales values, same length as *input_data*.

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist. Train the model first via the notebook
        or ``src.model.save_model``.
    """
    pipeline = load_model(model_path)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = input_data[feature_cols]
    predictions = np.maximum(0, pipeline.predict(X))
    return predictions


def load_and_predict(
    test_filepath: str,
    model_path: str = "models/ridge_pipeline.joblib",
) -> pd.DataFrame:
    """Load the test CSV, clean it, and return a DataFrame with predictions.

    Parameters
    ----------
    test_filepath:
        Path to the test CSV file (e.g. ``"data/test.csv"``).
    model_path:
        Path to the saved pipeline file.

    Returns
    -------
    pd.DataFrame
        Original test DataFrame with an additional ``Predicted_Sales`` column.
    """
    df_test = load_data(test_filepath)
    df_test_clean = clean_data(df_test, is_train=False)
    preds = predict_sales(df_test_clean, model_path)
    df_test_clean = df_test_clean.copy()
    df_test_clean["Predicted_Sales"] = preds
    return df_test_clean


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_CSV = "data/test.csv"
    MODEL_PATH = "models/ridge_pipeline.joblib"
    OUTPUT_PATH = "reports/predictions.csv"

    print("=" * 50)
    print("  Big Mart Sales Prediction - Inference")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(
            f"\n[ERROR] No saved model found at '{MODEL_PATH}'.\n"
            "Please train the model first by running the Jupyter notebook\n"
            "or training via the src.model module.\n"
        )
        raise SystemExit(1)

    print(f"\nLoading test data from: {TEST_CSV}")
    results = load_and_predict(TEST_CSV, MODEL_PATH)

    os.makedirs("reports", exist_ok=True)
    output_df = results[["Item_Identifier", "Outlet_Identifier", "Predicted_Sales"]].copy()
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH}")
    print(f"\nSample predictions (first 5 rows):")
    print(output_df.head().to_string(index=False))
    print(f"\nTotal predictions: {len(output_df)}")
    print(f"Min predicted sales: {output_df['Predicted_Sales'].min():.2f}")
    print(f"Max predicted sales: {output_df['Predicted_Sales'].max():.2f}")
    print(f"Mean predicted sales: {output_df['Predicted_Sales'].mean():.2f}")
