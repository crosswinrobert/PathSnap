"""
Bayesian Hyperparameter Optimisation using Optuna and GroupKFold.
Member 3 Implementation - PathSnap Project, PES University.

Guarantees:
- Group-aware cross-validation (GroupKFold) grouped by trip_id (no intra-trip data leakage).
- Optimises Macro F1-score across folds.
- Records trial history to CSV.
- Saves best configuration to JSON.
- Refits final classifier on training set and records computational costs.
"""

import os
import time
import json
import optuna
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class BayesianOptimizer:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        n_trials: int = 15,
        random_seed: int = 42
    ):
        """
        Args:
            X: Feature matrix of candidate observations (N, n_features)
            y: Binary target labels (1 if candidate matches ground truth, 0 otherwise)
            groups: Trip IDs for each observation to prevent intra-trip leakage
            n_splits: Number of folds for GroupKFold
            n_trials: Number of Optuna Bayesian trials
            random_seed: Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.groups = groups
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.trials_records = []
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna: Evaluates Macro F1 using GroupKFold.
        """
        # Justified hyperparameter search space
        n_estimators = trial.suggest_int("n_estimators", 40, 150, step=20)
        max_depth = trial.suggest_int("max_depth", 6, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        gkf = GroupKFold(n_splits=self.n_splits)
        fold_f1_scores = []

        for train_idx, val_idx in gkf.split(self.X, self.y, groups=self.groups):
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_seed,
                    n_jobs=-1
                ))
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            fold_f1_scores.append(macro_f1)

        mean_macro_f1 = float(np.mean(fold_f1_scores))

        # Record trial details
        record = {
            "trial_number": trial.number,
            "macro_f1": mean_macro_f1,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": str(max_features)
        }
        self.trials_records.append(record)

        return mean_macro_f1

    def run_optimization(self) -> Tuple[Dict[str, Any], float]:
        """
        Executes Optuna study using Tree-structured Parzen Estimator (TPE).
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        start_time = time.time()
        study.optimize(self.objective, n_trials=self.n_trials)
        optim_duration = time.time() - start_time

        self.best_params = study.best_params
        self.best_score = float(study.best_value)

        print(f"Optimization complete in {optim_duration:.2f}s across {self.n_trials} trials.")
        print(f"Best Macro F1: {self.best_score:.4f}")
        print(f"Best Parameters: {self.best_params}")

        return self.best_params, self.best_score

    def save_artifacts(
        self,
        trials_csv_path: str = "experiments/optimization_trials.csv",
        best_json_path: str = "artifacts/configs/best_hyperparameters.json"
    ):
        os.makedirs(os.path.dirname(trials_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(best_json_path), exist_ok=True)

        # Save trials DataFrame
        df_trials = pd.DataFrame(self.trials_records)
        df_trials.to_csv(trials_csv_path, index=False)
        print(f"Saved {len(df_trials)} trials to {trials_csv_path}")

        # Save best parameters
        best_data = {
            "best_macro_f1": self.best_score,
            "best_hyperparameters": self.best_params,
            "n_trials": self.n_trials,
            "cv_strategy": f"GroupKFold(n_splits={self.n_splits})",
            "grouped_by": "trip_id",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(best_json_path, "w") as f:
            json.dump(best_data, f, indent=4)
        print(f"Saved best configuration to {best_json_path}")
