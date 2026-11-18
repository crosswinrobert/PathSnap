# PathSnap - ML Model Experiments

## Experimental Setup

- Training trips: 80
- Training candidate rows: 1,427,306
- Training observations: 184,210
- Cross-validation: GroupKFold
- Number of folds: 5
- Random seed: 42
- Target: Candidate road-segment classification
- Feature source: Member 1's `src/features.py`
- Number of input features: 25 (`FEATURE_COLS`)
- Model evaluation metrics: Candidate-level Accuracy, Macro Precision, Macro Recall, Macro F1, Log Loss, and observation-level Top-1 Accuracy.

The models evaluated were Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, Extra Trees, and HistGradientBoosting.

KNN was evaluated using sampled training and validation data (up to 20,000 training rows and 10,000 validation rows). The other models used the full validation data.

## Initial Model Comparison

The following results were obtained from the initial five-fold cross-validation.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Log Loss | Observation Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.941181 | 0.875856 | 0.854849 | 0.864836 | 0.150256 | 0.796273 |
| KNN | 0.934420 | 0.863166 | 0.835203 | 0.848181 | 0.702118 | 0.758552 |
| Random Forest | 0.948313 | 0.904656 | 0.856398 | 0.878235 | 0.157485 | 0.798725 |
| Extra Trees | 0.948025 | 0.900886 | 0.858698 | 0.878051 | 0.151472 | 0.795768 |
| HistGradientBoosting | 0.946630 | 0.893897 | 0.861271 | 0.876382 | 0.138659 | 0.793203 |

Candidate-level Accuracy measures classification over individual candidate rows. Observation Top-1 measures whether the highest-scoring candidate for an observation is the correct candidate. These metrics measure different things and should not be directly equated.

## Best Initial Model

Random Forest was selected as the final model based on the initial comparison.

It achieved the highest Macro F1 score (0.878235) and the highest observation-level Top-1 score (0.798725) among the evaluated models.

Extra Trees had a very similar Macro F1 score (0.878051), while HistGradientBoosting achieved the lowest Log Loss (0.138659).

Random Forest was selected because it provided the strongest overall result on the primary Macro F1 and observation Top-1 metrics.

## Bayesian Optimisation

Bayesian optimisation was performed on Random Forest and HistGradientBoosting.

Configuration:
- Number of trials per model: 6
- Cross-validation: GroupKFold with 5 folds
- Maximum sampled rows: 60,000
- Optimisation metric: Macro F1
- Random seed: 42

Best sampled configurations:

### Random Forest

| Hyperparameter | Best Value |
|---|---:|
| n_estimators | 180 |
| max_depth | 35 |
| max_features | 0.5859 |
| min_samples_leaf | 3 |

Best sampled Macro F1: **0.880246**

### HistGradientBoosting

| Hyperparameter | Best Value |
|---|---:|
| max_iter | 240 |
| learning_rate | 0.1193 |
| max_leaf_nodes | 30 |
| min_samples_leaf | 45 |
| l2_regularization | 0.01123 |

Best sampled Macro F1: **0.879953**

## Tuned vs Untuned Comparison

The tuned models were compared against their untuned versions using cross-validation.

| Model | Configuration | Macro F1 | Observation Top-1 | Log Loss |
|---|---|---:|---:|---:|
| Random Forest | Untuned | 0.878235 | 0.798725 | 0.157485 |
| Random Forest | Tuned | 0.877582 | 0.799262 | 0.152652 |
| HistGradientBoosting | Untuned | 0.876382 | 0.793203 | 0.138659 |
| HistGradientBoosting | Tuned | 0.876456 | 0.794565 | 0.134839 |

Tuning produced small changes in performance. Random Forest's tuned version had slightly higher observation Top-1 and lower Log Loss, but its Macro F1 was slightly lower than the untuned version. HistGradientBoosting showed small improvements across the reported metrics.

The tuning results were therefore considered alongside the initial model comparison rather than assuming that tuning always improves every metric.

## Student-Test Evaluation

The selected models were also evaluated on the separate student-test dataset.

- Student-test trips: 10
- Candidate rows: 164,369
- Observations: 19,121

| Model | Observation Top-1 | Macro F1 | Log Loss |
|---|---:|---:|---:|
| Logistic Regression | 0.795722 | 0.873357 | 0.129606 |
| KNN | 0.789562 | 0.867933 | 0.496157 |
| Random Forest | 0.814236 | 0.893278 | 0.122510 |
| Extra Trees | 0.804037 | 0.888831 | 0.119860 |
| HistGradientBoosting | 0.809686 | 0.893216 | 0.110093 |

Random Forest achieved the highest observation Top-1 score (0.814236) and a Macro F1 of 0.893278. HistGradientBoosting achieved a very similar Macro F1 (0.893216) and the lowest Log Loss (0.110093).

The student-test results supported retaining Random Forest as the selected classifier.

## Final Model Artifact

The Member 2 model is saved separately at:

`artifacts/models/member2_random_forest.joblib`

It uses the 25 features defined by Member 1 in `FEATURE_COLS`.

**Important:** The currently saved final artifact is an untuned `RandomForestClassifier` trained using the default model configuration. The Bayesian optimisation results are recorded above, but the final artifact was not retrained with the tuned hyperparameters.

The separate `best_classifier.joblib` artifact was not replaced.

## Generated Outputs

- `experiments/cross_validation_results.csv`
- `experiments/bayesian_optimisation_results.csv`
- `experiments/best_hyperparameters.json`
- `experiments/student_test_metrics.csv`
- `experiments/student_test_predictions.csv.gz`
- `experiments/plot_results.py`
- `experiments/plots/`
