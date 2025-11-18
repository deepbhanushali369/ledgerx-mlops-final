import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from loguru import logger

from .load_data_01 import load_data, split_data

MODEL_PATH = Path("models/best_model.pkl")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(cm, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add text labels in each cell
    num_rows, num_cols = cm.shape
    for i in range(num_rows):
        for j in range(num_cols):
            val = cm[i, j]
            plt.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(out_path)
    plt.close()

def main():

    logger.info("📘 Running Model Evaluation")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Best model not found")

    model = joblib.load(MODEL_PATH)
    logger.info(f"📦 Loaded model from {MODEL_PATH}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info(f"📊 Test Metrics: {metrics}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, REPORT_DIR / "confusion_matrix.png")

    plot_roc_curve(y_test, y_prob, REPORT_DIR / "roc_curve.png")

    report_path = REPORT_DIR / "model_evaluation.txt"
    with open(report_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    logger.success(f"📄 Evaluation report saved → {report_path}")
    logger.success("🖼️ confusion_matrix.png & roc_curve.png generated!")

if __name__ == "__main__":
    main()
