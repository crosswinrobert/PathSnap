"""
Candidate road-segment classifier wrapper.
Predicts probability P(candidate is true ground-truth match | features).
"""

import joblib
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_candidate_classifier(**hyperparameters) -> Pipeline:
    """
    Builds a standard scikit-learn classifier pipeline for candidate matching.
    """
    default_params = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    }
    default_params.update(hyperparameters)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(**default_params))
    ])
    return model
