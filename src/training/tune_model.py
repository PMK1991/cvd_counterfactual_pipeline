"""Tune and compare model hyperparameters for the CVD risk classifier."""

import inspect
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.training.train_model import (
    BASELINE_XGB_PARAMS,
    NUMERICAL_FEATURES,
    build_xgb_pipeline,
    compute_metrics,
    load_train_test_data,
    print_metrics,
)

RESULTS_OUTPUT = Path("reports/model_tuning_results.json")

OLD_HYPERPARAMETER_SPACE = {
    f"classifier__{key}": [value] for key, value in BASELINE_XGB_PARAMS.items()
}

NEW_XGB_HYPERPARAMETER_SPACE = {
    "classifier__max_depth": [2, 3, 4, 5, 6],
    "classifier__learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
    "classifier__n_estimators": [100, 200, 300, 500, 800],
    "classifier__subsample": [0.6, 0.8, 1.0],
    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5, 7],
    "classifier__gamma": [0, 0.1, 0.25, 0.5, 1.0],
    "classifier__reg_alpha": [0, 0.001, 0.01, 0.1],
    "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "classifier__scale_pos_weight": [0.75, 1.0, 1.25, 1.5],
}


def _fit_and_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_metrics = compute_metrics(
        y_train, model.predict(X_train), model.predict_proba(X_train)[:, 1]
    )
    test_metrics = compute_metrics(
        y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1]
    )
    return train_metrics, test_metrics


def _metric_for_json(value):
    if isinstance(value, dict):
        return {key: _metric_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metric_for_json(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _print_space(title, space):
    print(f"\n{title}")
    for key, values in space.items():
        print(f"- {key}: {values}")


def _build_tabpfn_pipeline(X_train):
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return None

    signature = inspect.signature(TabPFNClassifier)
    kwargs = {}
    if "random_state" in signature.parameters:
        kwargs["random_state"] = 42
    if "device" in signature.parameters:
        kwargs["device"] = "cpu"
    if "n_estimators" in signature.parameters:
        kwargs["n_estimators"] = 16
    if "N_ensemble_configurations" in signature.parameters:
        kwargs["N_ensemble_configurations"] = 16
    if "ignore_pretraining_limits" in signature.parameters:
        kwargs["ignore_pretraining_limits"] = True

    categorical = X_train.columns.difference(NUMERICAL_FEATURES)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUMERICAL_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        )
                    ]
                ),
                categorical,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", TabPFNClassifier(**kwargs)),
        ]
    )


def run_tuning(n_iter=120, include_tabpfn=True):
    X_train, X_test, y_train, y_test = load_train_test_data()

    _print_space("Old hyperparameter space", OLD_HYPERPARAMETER_SPACE)
    _print_space("New XGBoost hyperparameter space", NEW_XGB_HYPERPARAMETER_SPACE)

    baseline = build_xgb_pipeline(X_train)
    baseline_train, baseline_test = _fit_and_score(
        baseline, X_train, X_test, y_train, y_test
    )
    print_metrics("Baseline XGBoost train", baseline_train)
    print_metrics("Baseline XGBoost test", baseline_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=build_xgb_pipeline(X_train),
        param_distributions=NEW_XGB_HYPERPARAMETER_SPACE,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        refit=True,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)

    tuned_train = compute_metrics(
        y_train,
        search.best_estimator_.predict(X_train),
        search.best_estimator_.predict_proba(X_train)[:, 1],
    )
    tuned_test = compute_metrics(
        y_test,
        search.best_estimator_.predict(X_test),
        search.best_estimator_.predict_proba(X_test)[:, 1],
    )

    print(f"\nBest XGBoost CV balanced accuracy: {search.best_score_ * 100:.2f}%")
    print(f"Best XGBoost params: {search.best_params_}")
    print_metrics("Tuned XGBoost train", tuned_train)
    print_metrics("Tuned XGBoost test", tuned_test)

    results = {
        "old_hyperparameter_space": OLD_HYPERPARAMETER_SPACE,
        "new_xgb_hyperparameter_space": NEW_XGB_HYPERPARAMETER_SPACE,
        "baseline_xgb": {
            "train": baseline_train,
            "test": baseline_test,
            "balanced_accuracy_test": balanced_accuracy_score(
                y_test, baseline.predict(X_test)
            ),
        },
        "tuned_xgb": {
            "best_cv_balanced_accuracy": search.best_score_,
            "best_params": search.best_params_,
            "train": tuned_train,
            "test": tuned_test,
            "balanced_accuracy_test": balanced_accuracy_score(
                y_test, search.best_estimator_.predict(X_test)
            ),
        },
    }

    if include_tabpfn:
        tabpfn = _build_tabpfn_pipeline(X_train)
        if tabpfn is None:
            print("\nTabPFN experiment skipped: tabpfn is not installed.")
            results["tabpfn"] = {"status": "not_installed"}
        else:
            try:
                tabpfn_train, tabpfn_test = _fit_and_score(
                    tabpfn, X_train, X_test, y_train, y_test
                )
            except (OSError, RuntimeError) as exc:
                print(f"\nTabPFN experiment failed: {exc}")
                results["tabpfn"] = {
                    "status": "failed",
                    "reason": str(exc),
                }
            else:
                print_metrics("TabPFN train", tabpfn_train)
                print_metrics("TabPFN test", tabpfn_test)
                results["tabpfn"] = {
                    "status": "completed",
                    "train": tabpfn_train,
                    "test": tabpfn_test,
                    "balanced_accuracy_test": balanced_accuracy_score(
                        y_test, tabpfn.predict(X_test)
                    ),
                }

    RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(_metric_for_json(results), f, indent=2)
    print(f"\nSaved tuning results to {RESULTS_OUTPUT}")


if __name__ == "__main__":
    run_tuning()
