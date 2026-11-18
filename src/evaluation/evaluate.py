import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS
from src.models.model_comparison import (
    get_models,
    sample_complete_observations,
)
from src.evaluation.map_matching_metrics import observation_top1_accuracy


TRAIN_PATH = Path("data/processed/features_train.csv.gz")
TEST_PATH = Path("data/processed/features_student_test.csv.gz")
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
KNN_TRAIN_LIMIT = 20_000
KNN_TEST_LIMIT = 10_000


def prepare_data(df, name):
    keys = ["trip_id", "obs_idx"]

    df = df[df["label_valid"] == 1].copy()

    has_positive = df.groupby(keys)["y"].transform("max") == 1
    df = df[has_positive].copy()

    df["y"] = df["y"].astype(int)

    if df.empty:
        raise ValueError(f"No eligible observations in {name}.")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int8)
    observation_ids = df[keys].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError(f"{name} features contain NaN or infinity.")

    print(
        f"{name}: {len(df):,} candidate rows, "
        f"{df.groupby(keys).ngroups:,} observations, "
        f"{df['trip_id'].nunique()} trips"
    )

    return df, X, y, observation_ids


def evaluate_one(model_name, model, X_train, y_train,
                 X_test, y_test, observation_ids, test_df,
                 evaluation_type):

    print(f"\nTraining {model_name}...")

    estimator = clone(model)

    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    print(f"Evaluating {model_name}...")

    start = time.perf_counter()
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - start

    metrics = {
        "model": model_name,
        "evaluation_type": evaluation_type,
        "candidate_rows": len(y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
        "top1_accuracy": observation_top1_accuracy(
            y_test, probabilities, observation_ids
        ),
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
    }

    predictions_df = pd.DataFrame({
        "model": model_name,
        "trip_id": observation_ids[:, 0],
        "obs_idx": observation_ids[:, 1],
        "y_true": y_test,
        "y_pred": predictions,
        "probability": probabilities,
    })

    print(
        f"{model_name}: "
        f"Top-1={metrics['top1_accuracy']:.5f}, "
        f"Macro F1={metrics['macro_f1']:.5f}, "
        f"Log Loss={metrics['log_loss']:.5f}"
    )

    del estimator

    return metrics, predictions_df


def main():
    print("Loading training and student-test features...")

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    train_df, X_train, y_train, _ = prepare_data(
        train_raw, "Training data"
    )
    test_df, X_test, y_test, test_obs = prepare_data(
        test_raw, "Student-test data"
    )

    results = []
    all_predictions = []

    # Four models evaluated on all eligible training and test rows.
    for model_name, model in get_models().items():
        metrics, predictions = evaluate_one(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            test_obs,
            test_df,
            "full_student_test",
        )

        results.append(metrics)
        all_predictions.append(predictions)

    # KNN retains the existing sampled-observation approach.
    print("\nPreparing sampled KNN data...")

    rng = np.random.default_rng(RANDOM_STATE)

    sampled_train = sample_complete_observations(
        train_df, KNN_TRAIN_LIMIT, rng
    )
    sampled_test = sample_complete_observations(
        test_df, KNN_TEST_LIMIT, rng
    )

    X_knn_train = sampled_train[FEATURE_COLS].to_numpy(
        dtype=np.float32
    )
    y_knn_train = sampled_train["y"].to_numpy(dtype=np.int8)

    X_knn_test = sampled_test[FEATURE_COLS].to_numpy(
        dtype=np.float32
    )
    y_knn_test = sampled_test["y"].to_numpy(dtype=np.int8)

    knn_obs = sampled_test[["trip_id", "obs_idx"]].to_numpy()

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        )),
    ])

    metrics, predictions = evaluate_one(
        "KNN",
        knn,
        X_knn_train,
        y_knn_train,
        X_knn_test,
        y_knn_test,
        knn_obs,
        sampled_test,
        "sampled_student_test",
    )

    results.append(metrics)
    all_predictions.append(predictions)

    # Save results separately from the existing CV experiments.
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "student_test_metrics.csv",
        index=False,
    )

    predictions_df = pd.concat(
        all_predictions, ignore_index=True
    )
    predictions_df.to_csv(
        OUTPUT_DIR / "student_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    print("\nSTUDENT-TEST RESULTS")
    print(
        results_df[
            ["model", "evaluation_type", "top1_accuracy",
             "macro_f1", "log_loss"]
        ].sort_values("macro_f1", ascending=False).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print("- experiments/student_test_metrics.csv")
    print("- experiments/student_test_predictions.csv.gz")


if __name__ == "__main__":
    main()
