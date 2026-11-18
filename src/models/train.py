import pandas as pd

from src.models.model_comparison import evaluate_models


def main():
    path = "data/processed/features_train.csv.gz"

    print("Loading training dataset...")
    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} candidate rows")
    print("Starting five-fold GroupKFold model comparison...")

    summary = evaluate_models(df)

    print("\nTraining comparison completed.")
    print("Results saved in:")
    print("- experiments/cross_validation_results.csv")
    print("- experiments/model_comparison_summary.csv")


if __name__ == "__main__":
    main()
