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

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils.dataLoader import DataLoader, MODEL_RANDOM_STATE, MODEL_TEST_SIZE

DATA_PATH = "data/heart_statlog_cleveland_hungary_final.csv"
MODEL_OUTPUT = Path("model/xgb_pipeline.pkl")
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BASELINE_XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.01,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
}


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_train_test_data():
    """Load the cleaned dataset and return the reproducible train/test split."""
    loader = DataLoader(DATA_PATH)
    df = loader.load_clean_data()

    X = df.drop(columns="target")
    y = df["target"]

    return train_test_split(
        X, y, test_size=MODEL_TEST_SIZE, random_state=MODEL_RANDOM_STATE
    )


def build_preprocessor(X_train):
    categorical = X_train.columns.difference(NUMERICAL_FEATURES)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERICAL_FEATURES),
            (
                "cat",
                Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical,
            ),
        ]
    )


def build_xgb_pipeline(X_train, classifier_params=None):
    params = dict(BASELINE_XGB_PARAMS)
    if classifier_params:
        params.update(classifier_params)

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", XGBClassifier(**params)),
        ]
    )


def compute_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    f1 = _safe_rate(2 * precision * recall, precision + recall)
    sensitivity = _safe_rate(tp, tp + fn)
    specificity = _safe_rate(tn, tn + fp)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def print_metrics(split_name, metrics):
    print(f"\n{split_name} metrics")
    print(f"Accuracy:      {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:     {metrics['precision'] * 100:.2f}%")
    print(f"Recall:        {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:      {metrics['f1'] * 100:.2f}%")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:       {metrics['roc_auc'] * 100:.2f}%")
    print(
        f"Sensitivity:   {metrics['sensitivity'] * 100:.2f}%  "
        "(TPR: actual CVD/high-risk correctly identified)"
    )
    print(
        f"Specificity:   {metrics['specificity'] * 100:.2f}%  "
        "(TNR: actual non-CVD/low-risk correctly identified)"
    )
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")


def train_and_export():
    X_train, X_test, y_train, y_test = load_train_test_data()

    # Hyperparameters match notebook cell-3
    pipeline = build_xgb_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    print_metrics(
        "Train",
        compute_metrics(
            y_train, pipeline.predict(X_train), pipeline.predict_proba(X_train)[:, 1]
        ),
    )
    print_metrics(
        "Test",
        compute_metrics(
            y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1]
        ),
    )

    # Export
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    train_and_export()
