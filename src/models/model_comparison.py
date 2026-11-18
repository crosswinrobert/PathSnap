import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5


def get_models():
    """Return the five initial candidate classifiers."""

    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )),
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance"
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(X, y, groups):
    """
    Evaluate classifiers using GroupKFold.

    X: Feature matrix
    y: Target labels
    groups: Trip IDs corresponding to each row
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y and groups must have equal lengths.")

    if len(np.unique(groups)) < N_SPLITS:
        raise ValueError("Not enough unique trips for GroupKFold.")

    classes = np.unique(y)
    cv = GroupKFold(n_splits=N_SPLITS)

    results = []

    for model_name, model in get_models().items():
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            estimator = clone(model)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            start = time.perf_counter()
            estimator.fit(X_train, y_train)
            training_time = time.perf_counter() - start

            start = time.perf_counter()
            predictions = estimator.predict(X_val)
            inference_time = time.perf_counter() - start

            metrics = {
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y_val, predictions),
                "precision_macro": precision_score(
                    y_val, predictions, average="macro",
                    zero_division=0
                ),
                "recall_macro": recall_score(
                    y_val, predictions, average="macro",
                    zero_division=0
                ),
                "f1_macro": f1_score(
                    y_val, predictions, average="macro",
                    zero_division=0
                ),
                "training_time_sec": training_time,
                "inference_time_sec": inference_time,
            }

            if hasattr(estimator, "predict_proba"):
                probabilities = estimator.predict_proba(X_val)
                model_classes = estimator.classes_

                aligned = np.zeros((len(X_val), len(classes)))

                for j, class_label in enumerate(model_classes):
                    class_index = np.where(classes == class_label)[0][0]
                    aligned[:, class_index] = probabilities[:, j]

                metrics["log_loss"] = log_loss(
                    y_val, aligned, labels=classes
                )
            else:
                metrics["log_loss"] = np.nan

            fold_metrics.append(metrics)

        results.extend(fold_metrics)

        print(f"Completed: {model_name}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "experiments/cross_validation_results.csv",
        index=False
    )

    summary = (
        results_df.groupby("model")
        .agg({
            "accuracy": "mean",
            "precision_macro": "mean",
            "recall_macro": "mean",
            "f1_macro": "mean",
            "log_loss": "mean",
            "training_time_sec": "mean",
            "inference_time_sec": "mean",
        })
        .sort_values("f1_macro", ascending=False)
    )

    summary.to_csv("experiments/model_comparison_summary.csv")

    return summary


if __name__ == "__main__":
    print("Available models:")
    for name in get_models():
        print("-", name)

    print("\nModel comparison framework ready.")
    print("Waiting for Member 1's feature dataset.")
