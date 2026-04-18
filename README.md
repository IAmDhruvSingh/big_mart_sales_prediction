# Big Mart Sales Prediction

A Python machine-learning project that predicts item-level outlet sales for the **Big Mart** retail chain using the [Kaggle Big Mart Sales dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data).

---

## 📌 Project Overview

Big Mart collects sales data across multiple stores and product types. The goal is to build a regression model that estimates `Item_Outlet_Sales` for a given product-outlet combination, enabling better inventory planning and revenue forecasting.

**Approach:** Data cleaning → Feature engineering (StandardScaler + OneHotEncoder) → Ridge Regression (α = 0.8)

---

## 🗂️ Repository Structure

```
big_mart_sales_prediction/
│
├── data/                        # Raw and processed data (excluded from git)
│   └── .gitkeep
│
├── notebooks/                   # Jupyter notebooks
│   └── exploration_and_training.ipynb
│
├── src/                         # Python source modules
│   ├── __init__.py
│   ├── data_processing.py       # Data loading & cleaning
│   ├── features.py              # Feature engineering & train/val split
│   ├── model.py                 # Model building, training & evaluation
│   └── predict.py               # Inference script & CLI entry-point
│
├── reports/                     # Generated plots and prediction outputs
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/IAmDhruvSingh/big_mart_sales_prediction.git
cd big_mart_sales_prediction
```

### 2. Place the data

Download `Train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) and put them in the `data/` directory:

```
data/
├── Train.csv
├── test.csv
└── .gitkeep
```

### 3. Set up the Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Run the Jupyter notebook

```bash
jupyter notebook notebooks/exploration_and_training.ipynb
```

The notebook walks through:
- Exploratory Data Analysis (EDA)
- Preprocessing and feature engineering
- Model training and validation metrics
- Sample predictions

### 5. Run inference via CLI

After training the model in the notebook (it saves to `models/ridge_pipeline.joblib`):

```bash
python -m src.predict
```

Predictions are written to `reports/predictions.csv`.

---

## 🔧 Module Reference

| Module | Key Functions | Description |
|--------|---------------|-------------|
| `src.data_processing` | `load_data()`, `clean_data()` | Load CSV and apply imputation/normalisation |
| `src.features` | `build_preprocessor()`, `split_data()` | Build sklearn ColumnTransformer; train/val split |
| `src.model` | `build_pipeline()`, `train_model()`, `evaluate_model()`, `save_model()`, `load_model()` | End-to-end Ridge Regression pipeline |
| `src.predict` | `predict_sales()`, `load_and_predict()` | Inference on new data |

---

## 📊 Features Used

| Feature | Type | Notes |
|---------|------|-------|
| `Item_Weight` | Numeric | Missing values imputed with mean |
| `Item_Visibility` | Numeric | Zero values replaced with per-type mean |
| `Item_MRP` | Numeric | Maximum Retail Price |
| `Outlet_Establishment_Year` | Numeric | Year the outlet was established |
| `Item_Fat_Content` | Categorical | Standardised ('lf'/'LF' → 'Low Fat') |
| `Item_Type` | Categorical | Product category |
| `Outlet_Identifier` | Categorical | Unique outlet ID |
| `Outlet_Size` | Categorical | Outlet size (Small/Medium/High) |
| `Outlet_Location_Type` | Categorical | Tier 1/2/3 city classification |
| `Outlet_Type` | Categorical | Supermarket type or Grocery Store |

---

## 📈 Model Details

- **Algorithm:** Ridge Regression (L2 regularisation, α = 0.8)
- **Preprocessing:** StandardScaler for numerics, OneHotEncoder for categoricals
- **Evaluation Metrics:** RMSE and R² on a 20% held-out validation split

---

## 🔮 Extending the Project

The modular structure makes it easy to swap or add models:

```python
from sklearn.ensemble import GradientBoostingRegressor
from src.features import build_preprocessor

preprocessor = build_preprocessor()
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=500))
])
```

---

## 📄 License

This project is open-source. Dataset credit: [Kaggle – Big Mart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data).
