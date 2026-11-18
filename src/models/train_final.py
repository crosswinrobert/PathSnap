import os
import joblib
import pandas as pd

from src.features import FEATURE_COLS
from src.models.model_comparison import get_models

DATA_PATH = "data/processed/features_train.csv.gz"
OUTPUT_PATH = "artifacts/models/member2_random_forest.joblib"

def main():
    print("Loading all training features...")
    df = pd.read_csv(DATA_PATH)

    # Keep only valid labelled candidates
    df = df[df["label_valid"] == 1].copy()

    # Exclude observations without a positive candidate
    valid_obs = (
        df.groupby(["trip_id", "obs_idx"])["y"]
        .transform("sum") > 0
    )
    df = df[valid_obs].copy()

    X = df[FEATURE_COLS]
    y = df["y"].astype(int)

    print(f"Training rows: {len(df):,}")
    print(f"Training trips: {df['trip_id'].nunique()}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Classes: {sorted(y.unique())}")

    if set(y.unique()) != {0, 1}:
        raise ValueError("Training data must contain both classes.")

    print("\nTraining final Random Forest...")
    model = get_models()["Random Forest"]
    model.fit(X, y)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, OUTPUT_PATH, compress=3)

    print("\nFinal model saved successfully:")
    print(OUTPUT_PATH)
    print(f"Model type: {type(model).__name__}")
    print(f"Feature count: {len(FEATURE_COLS)}")

if __name__ == "__main__":
    main()
