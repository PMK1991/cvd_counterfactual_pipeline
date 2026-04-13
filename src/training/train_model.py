# conda-env: mtech-env
"""
Train and export XGBoost pipeline model for the counterfactual pipeline.

Replicates the training logic from nb_cvd_pipeline.ipynb (cells 1-4).
Saves the fitted sklearn Pipeline to model/xgb_pipeline.pkl.

Usage:
    python train_model.py
"""

import sys
import pickle
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils.dataLoader import DataLoader

DATA_PATH = "data/heart_statlog_cleveland_hungary_final.csv"
MODEL_OUTPUT = Path("model/xgb_pipeline.pkl")


def train_and_export():
    # Load and clean data (matches notebook cell-1)
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    if df is not None:
        df = loader.remove_outliers_iqr(df)

    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical = X_train.columns.difference(numerical)

    transformations = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numerical),
            (
                "cat",
                Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical,
            ),
        ]
    )

    # Hyperparameters match notebook cell-3
    pipeline = Pipeline(
        steps=[
            ("preprocessor", transformations),
            (
                "classifier",
                XGBClassifier(
                    max_depth=3,
                    learning_rate=0.01,
                    n_estimators=300,
                    subsample=0.8,
                    colsample_bytree=1.0,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"Test Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Test Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Test Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")

    # Export
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    train_and_export()
