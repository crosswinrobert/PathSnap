import time
from pathlib import Path

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

from src.features import FEATURE_COLS
from src.evaluation.map_matching_metrics import observation_top1_accuracy


RANDOM_STATE = 42
N_SPLITS = 5

# KNN is capped because full-dataset nearest-neighbour prediction
# is too expensive for this experiment.
KNN_TRAIN_LIMIT = 20_000
KNN_VAL_LIMIT = 10_000

OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_models():
    """Return the four models evaluated on the full eligible dataset."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def sample_complete_observations(df, max_rows, rng):
    """Sample whole GPS observations, never partial candidate groups."""
    grouped = list(df.groupby(["trip_id", "obs_idx"], sort=False).indices.values())
    rng.shuffle(grouped)

    selected = []
    count = 0

    for indices in grouped:
        if count + len(indices) > max_rows:
            continue
        selected.extend(indices)
        count += len(indices)

    return df.iloc[selected].copy()


def calculate_metrics(
    model_name, fold, y_true, predictions, probabilities,
    observation_ids, training_time, inference_time, evaluation_type
):
    return {
        "model": model_name,
        "fold": fold,
        "evaluation_type": evaluation_type,
        "candidate_rows": len(y_true),
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "observation_top1_accuracy": observation_top1_accuracy(
            y_true, probabilities, observation_ids
        ),
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
    }


def evaluate_models(df):
    """Run five-fold trip-grouped cross-validation."""

    required = FEATURE_COLS + ["trip_id", "obs_idx", "y"]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Keep only valid labels.
    df = df[df["label_valid"] == 1].copy()

    # Remove observations where the correct candidate is absent.
    observation_keys = ["trip_id", "obs_idx"]
    has_positive = df.groupby(observation_keys)["y"].transform("max") == 1
    df = df[has_positive].copy()

    df["y"] = df["y"].astype(int)

    if set(df["y"].unique()) != {0, 1}:
        raise ValueError("Training data must contain both classes.")

    print(f"Eligible candidate rows: {len(df):,}")
    print(f"Eligible observations: {df.groupby(observation_keys).ngroups:,}")
    print(f"Eligible trips: {df['trip_id'].nunique()}")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int8)
    groups = df["trip_id"].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    cv = GroupKFold(n_splits=N_SPLITS)
    results = []
    oof_predictions = []

    models = get_models()
    rng = np.random.default_rng(RANDOM_STATE)

    for model_name, model in models.items():
        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            estimator = clone(model)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_obs = df.iloc[train_idx][observation_keys].to_numpy()
            val_obs = df.iloc[val_idx][observation_keys].to_numpy()

            start = time.perf_counter()
            estimator.fit(X_train, y_train)
            training_time = time.perf_counter() - start

            start = time.perf_counter()
            predictions = estimator.predict(X_val)
            probabilities = estimator.predict_proba(X_val)[:, 1]
            inference_time = time.perf_counter() - start

            metrics = calculate_metrics(
                model_name, fold, y_val, predictions, probabilities,
                val_obs, training_time, inference_time, "full_validation"
            )
            results.append(metrics)

            oof_predictions.append(pd.DataFrame({
                "model": model_name,
                "fold": fold,
                "trip_id": val_obs[:, 0],
                "obs_idx": val_obs[:, 1],
                "y_true": y_val,
                "y_pred": predictions,
                "probability": probabilities,
                "evaluation_type": "full_validation",
            }))

            print(f"{model_name} — fold {fold}/{N_SPLITS} complete")

            del estimator

    # KNN: sample complete observations in each fold.
    print("\nRunning sampled KNN comparison...")

    knn_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        sampled_train = sample_complete_observations(
            train_df, KNN_TRAIN_LIMIT, rng
        )
        sampled_val = sample_complete_observations(
            val_df, KNN_VAL_LIMIT, rng
        )

        X_train = sampled_train[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_train = sampled_train["y"].to_numpy(dtype=np.int8)
        X_val = sampled_val[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_val = sampled_val["y"].to_numpy(dtype=np.int8)
        val_obs = sampled_val[observation_keys].to_numpy()

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            )),
        ])

        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        start = time.perf_counter()
        predictions = estimator.predict(X_val)
        probabilities = estimator.predict_proba(X_val)[:, 1]
        inference_time = time.perf_counter() - start

        metrics = calculate_metrics(
            "KNN", fold, y_val, predictions, probabilities,
            val_obs, training_time, inference_time, "sampled_observations"
        )
        knn_results.append(metrics)

        oof_predictions.append(pd.DataFrame({
            "model": "KNN",
            "fold": fold,
            "trip_id": val_obs[:, 0],
            "obs_idx": val_obs[:, 1],
            "y_true": y_val,
            "y_pred": predictions,
            "probability": probabilities,
            "evaluation_type": "sampled_observations",
        }))

        print(
            f"KNN — fold {fold}/{N_SPLITS} complete "
            f"({len(sampled_train):,} train rows, "
            f"{len(sampled_val):,} validation rows)"
        )

        del estimator

    results.extend(knn_results)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "cross_validation_results.csv", index=False
    )

    predictions_df = pd.concat(oof_predictions, ignore_index=True)
    predictions_df.to_csv(
        OUTPUT_DIR / "oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    print("Saved out-of-fold predictions.")

    summary = (
        results_df.groupby(["model", "evaluation_type"])
        .agg({
            "accuracy": "mean",
            "precision_macro": "mean",
            "recall_macro": "mean",
            "f1_macro": "mean",
            "log_loss": "mean",
            "observation_top1_accuracy": "mean",
            "training_time_sec": "mean",
            "inference_time_sec": "mean",
        })
        .reset_index()
        .sort_values("f1_macro", ascending=False)
    )

    summary.to_csv(
        OUTPUT_DIR / "model_comparison_summary.csv", index=False
    )

    print("\nMODEL COMPARISON SUMMARY")
    print(summary.to_string(index=False))

    return summary
