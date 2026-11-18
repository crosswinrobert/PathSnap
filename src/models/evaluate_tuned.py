import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupKFold

from src.features import FEATURE_COLS
from src.models.model_comparison import calculate_metrics


DATA_PATH = Path("data/processed/features_train.csv.gz")
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42


def main():
    print("Loading full training dataset...")
    df = pd.read_csv(DATA_PATH)

    # Match the filtering used in the original comparison.
    df = df[df["label_valid"] == 1].copy()

    keys = ["trip_id", "obs_idx"]
    has_positive = df.groupby(keys)["y"].transform("max") == 1
    df = df[has_positive].copy()

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int8)
    groups = df["trip_id"].to_numpy()
    observation_ids = df[keys].to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    print(f"Eligible rows: {len(df):,}")
    print(f"Eligible observations: {df.groupby(keys).ngroups:,}")
    print(f"Eligible trips: {df['trip_id'].nunique()}")

    models = {
        "Random Forest Tuned": RandomForestClassifier(
            n_estimators=180,
            max_depth=35,
            max_features=0.5859357806982803,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "HistGradientBoosting Tuned": HistGradientBoostingClassifier(
            l2_regularization=0.011233621690895234,
            learning_rate=0.11931659619655849,
            max_iter=240,
            max_leaf_nodes=30,
            min_samples_leaf=45,
            random_state=RANDOM_STATE,
        ),
    }

    cv = GroupKFold(n_splits=N_SPLITS)
    results = []
    oof_predictions = []

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            val_obs = observation_ids[val_idx]

            start = time.perf_counter()
            model.fit(X_train, y_train)
            training_time = time.perf_counter() - start

            start = time.perf_counter()
            predictions = model.predict(X_val)
            probabilities = model.predict_proba(X_val)[:, 1]
            inference_time = time.perf_counter() - start

            metrics = calculate_metrics(
                model_name,
                fold,
                y_val,
                predictions,
                probabilities,
                val_obs,
                training_time,
                inference_time,
                "full_validation",
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
            }))

            print(
                f"{model_name} — fold {fold}/{N_SPLITS} complete "
                f"(Macro F1: {metrics['f1_macro']:.5f})"
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "tuned_cross_validation_results.csv",
        index=False,
    )

    predictions_df = pd.concat(oof_predictions, ignore_index=True)
    predictions_df.to_csv(
        OUTPUT_DIR / "tuned_oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    # Compare against the original full-validation fold results.
    baseline = pd.read_csv(
        OUTPUT_DIR / "cross_validation_results.csv"
    )
    baseline = baseline[
        baseline["evaluation_type"] == "full_validation"
    ]

    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "log_loss",
        "observation_top1_accuracy",
        "training_time_sec",
        "inference_time_sec",
    ]

    baseline_summary = (
        baseline.groupby("model")[metrics].mean()
    )

    tuned_summary = (
        results_df.groupby("model")[metrics].mean()
    )

    comparison = pd.concat([
        baseline_summary,
        tuned_summary,
    ])

    comparison.to_csv(
        OUTPUT_DIR / "tuned_vs_untuned_comparison.csv"
    )

    print("\nTUNED VS UNTUNED COMPARISON")
    print(comparison.round(5).to_string())

    print("\nSaved:")
    print("- experiments/tuned_cross_validation_results.csv")
    print("- experiments/tuned_oof_predictions.csv.gz")
    print("- experiments/tuned_vs_untuned_comparison.csv")


if __name__ == "__main__":
    main()
