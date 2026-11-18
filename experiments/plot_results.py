import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

OUT = "experiments/plots"
os.makedirs(OUT, exist_ok=True)

# Load saved results
cv = pd.read_csv("experiments/cross_validation_results.csv")
test = pd.read_csv("experiments/student_test_metrics.csv")
pred = pd.read_csv("experiments/student_test_predictions.csv.gz")

# 1. Cross-validation Macro F1
plt.figure(figsize=(10, 6))

for model, group in cv.groupby("model"):
    group = group.sort_values("fold")
    plt.plot(
        group["fold"],
        group["f1_macro"],
        marker="o",
        label=model
    )

plt.xlabel("CV Fold")
plt.ylabel("Macro F1")
plt.title("Five-Fold Cross-Validation Macro F1")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/cv_macro_f1.png", dpi=300)
plt.close()

# 2. Student-test Accuracy and Macro F1
test_sorted = test.sort_values("macro_f1", ascending=False)

ax = test_sorted.set_index("model")[
    ["top1_accuracy", "macro_f1"]
].plot(kind="bar", figsize=(10, 6))

ax.set_title("Student-Test Accuracy and Macro F1")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1)
ax.legend(["Top-1 Accuracy", "Macro F1"])
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/student_test_accuracy_f1.png", dpi=300)
plt.close()

# 3. Student-test Log Loss
plt.figure(figsize=(9, 5))
plt.bar(test["model"], test["log_loss"])
plt.title("Student-Test Log Loss")
plt.ylabel("Log Loss (lower is better)")
plt.xlabel("Model")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/student_test_log_loss.png", dpi=300)
plt.close()

# 4. Candidate-level confusion matrices
for model, group in pred.groupby("model"):
    cm = confusion_matrix(
        group["y_true"],
        group["y_pred"],
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not selected", "Selected"]
    )
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{model} - Student-Test Confusion Matrix")
    plt.tight_layout()

    filename = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    plt.savefig(f"{OUT}/confusion_matrix_{filename}.png", dpi=300)
    plt.close()

print("Plots generated successfully:")
for filename in sorted(os.listdir(OUT)):
    print(f"- {OUT}/{filename}")
