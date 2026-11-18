import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import GroupKFold

from src.features import FEATURE_COLS


RANDOM_STATE = 42
N_SPLITS = 5
MAX_ROWS = 60_000
N_ITER = 6

DATA_PATH = Path("data/processed/features_train.csv.gz")
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_tuning_data():
    print("Loading training features...")
    df = pd.read_csv(DATA_PATH)

    df = df[df["label_valid"] == 1].copy()

    keys = ["trip_id", "obs_idx"]
    valid_obs = df.groupby(keys)["y"].transform("max") == 1
    df = df[valid_obs].copy()

    # Sample complete GPS observations.
    rng = np.random.default_rng(RANDOM_STATE)
    groups = list(df.groupby(keys, sort=False).indices.values())
    rng.shuffle(groups)

    selected = []
    count = 0

    for indices in groups:
        if count + len(indices) > MAX_ROWS:
            continue
        selected.extend(indices)
        count += len(indices)

    sample = df.iloc[selected].copy()

    X = sample[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = sample["y"].to_numpy(dtype=np.int8)
    trip_groups = sample["trip_id"].to_numpy()

    print(f"Tuning rows: {len(sample):,}")
    print(f"Tuning observations: {sample.groupby(keys).ngroups:,}")
    print(f"Tuning trips: {sample['trip_id'].nunique()}")

    return X, y, trip_groups


def main():
    X, y, groups = load_tuning_data()

    cv = GroupKFold(n_splits=N_SPLITS)

    searches = {
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": Integer(100, 200),
                "max_depth": Categorical([None, 15, 25, 35]),
                "min_samples_leaf": Integer(1, 5),
                "max_features": Real(0.5, 1.0),
            },
        ),

        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            {
                "max_iter": Integer(100, 250),
                "learning_rate": Real(
                    0.03, 0.2, prior="log-uniform"
                ),
                "max_leaf_nodes": Integer(15, 63),
                "min_samples_leaf": Integer(15, 60),
                "l2_regularization": Real(
                    1e-4, 10.0, prior="log-uniform"
                ),
            },
        ),
    }

    all_results = []
    best_parameters = {}

    for name, (estimator, search_space) in searches.items():
        print(f"\nStarting Bayesian optimisation: {name}")

        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_space,
            n_iter=N_ITER,
            scoring="f1_macro",
            cv=cv,
            n_jobs=1,
            random_state=RANDOM_STATE,
            refit=False,
            verbose=1,
            return_train_score=False,
        )

        start = time.perf_counter()
        search.fit(X, y, groups=groups)
        elapsed = time.perf_counter() - start

        best_parameters[name] = {
            key: value
            for key, value in search.best_params_.items()
        }

        print(f"\n{name} best Macro F1: {search.best_score_:.5f}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Time: {elapsed:.1f} seconds")

        results = pd.DataFrame(search.cv_results_)
        results["model"] = name
        results["optimisation_time_sec"] = elapsed
        all_results.append(results)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(
        OUTPUT_DIR / "bayesian_optimisation_results.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "best_hyperparameters.json", "w"
    ) as f:
        json.dump(best_parameters, f, indent=4, default=str)

    print("\nBayesian optimisation completed.")
    print("Saved:")
    print("- experiments/bayesian_optimisation_results.csv")
    print("- experiments/best_hyperparameters.json")


if __name__ == "__main__":
    main()
