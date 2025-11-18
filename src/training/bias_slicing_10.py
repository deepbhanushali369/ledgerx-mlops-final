from pathlib import Path
import joblib
import pandas as pd
from loguru import logger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .load_data_01 import load_data, split_data

# Paths
MODEL_TUNED_PATH = Path("models/best_model_tuned.pkl")
MODEL_BASE_PATH = Path("models/best_model.pkl")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BIAS_DETAIL_PATH = REPORT_DIR / "bias_slicing.txt"
BIAS_SUMMARY_PATH = REPORT_DIR / "bias_summary_report.txt"


def compute_metrics(y_true, y_pred):
    return {
        "support": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def evaluate_slices(y_true, y_pred, group_series, group_name, min_support=20):
    """
    Evaluate metrics per group for a given slicing variable.
    Returns a list of dicts: one row per group.
    """
    logger.info(f"🔍 Evaluating bias slices for '{group_name}'")

    results = []
    df_group = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, group_name: group_series}
    )

    for value in sorted(df_group[group_name].dropna().unique()):
        mask = df_group[group_name] == value
        y_t = df_group.loc[mask, "y_true"]
        y_p = df_group.loc[mask, "y_pred"]

        if len(y_t) < min_support:
            continue  # skip very small groups, too noisy

        metrics = compute_metrics(y_t, y_p)
        row = {"group": group_name, "value": str(value)}
        row.update(metrics)
        results.append(row)

    return results


def main():
    logger.info("📘 Running Bias Slicing & Fairness Analysis")

    # 1) Load data & split
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Align attributes for test set
    df_test_attrs = df.loc[X_test.index].copy()

    # 2) Load best tuned model if present
    if MODEL_TUNED_PATH.exists():
        model_path = MODEL_TUNED_PATH
    elif MODEL_BASE_PATH.exists():
        model_path = MODEL_BASE_PATH
    else:
        raise FileNotFoundError(
            "Neither best_model_tuned.pkl nor best_model.pkl was found in models/"
        )

    model = joblib.load(model_path)
    logger.info(f"📦 Loaded model for bias analysis → {model_path}")

    # 3) Global performance
    y_pred = model.predict(X_test)
    global_metrics = compute_metrics(y_test, y_pred)
    global_f1 = global_metrics["f1"]

    logger.info(f"🌍 Global Test Metrics: {global_metrics}")

    # 4) Slicing variables
    slices_results = []

    # vendor_name
    if "vendor_name" in df_test_attrs.columns:
        slices_results += evaluate_slices(
            y_test, y_pred, df_test_attrs["vendor_name"], "vendor_name", min_support=20
        )

    # currency
    if "currency" in df_test_attrs.columns:
        slices_results += evaluate_slices(
            y_test, y_pred, df_test_attrs["currency"], "currency", min_support=20
        )

    # blur_flag
    if "blur_flag" in df_test_attrs.columns:
        slices_results += evaluate_slices(
            y_test, y_pred, df_test_attrs["blur_flag"], "blur_flag", min_support=10
        )

    # total_amount buckets
    if "total_amount" in df_test_attrs.columns:
        amount_bucket = pd.cut(
            df_test_attrs["total_amount"],
            bins=[-1, 1000, 5000, 20000, float("inf")],
            labels=["0–1k", "1k–5k", "5k–20k", "20k+"],
        )
        slices_results += evaluate_slices(
            y_test, y_pred, amount_bucket, "amount_bucket", min_support=20
        )

    # invoice_age_days buckets
    if "invoice_age_days" in df_test_attrs.columns:
        age_bucket = pd.cut(
            df_test_attrs["invoice_age_days"],
            bins=[-1, 30, 90, 365, float("inf")],
            labels=["0–30d", "30–90d", "90–365d", "365d+"],
        )
        slices_results += evaluate_slices(
            y_test, y_pred, age_bucket, "age_bucket", min_support=20
        )

    # 5) Write detailed bias report
    logger.info(f"📝 Writing detailed bias report → {BIAS_DETAIL_PATH}")
    with open(BIAS_DETAIL_PATH, "w", encoding="utf-8") as f:
        f.write("LedgerX – Bias Slicing Report\n")
        f.write("================================\n\n")
        f.write("Global Test Metrics:\n")
        for k, v in global_metrics.items():
            f.write(f"- {k}: {v}\n")

        f.write("\nPer-slice Metrics:\n")
        for row in slices_results:
            f.write(
                f"\n[{row['group']} = {row['value']}] "
                f"(n={row['support']}): "
                f"acc={row['accuracy']}, "
                f"prec={row['precision']}, "
                f"rec={row['recall']}, "
                f"f1={row['f1']}"
            )

    # 6) Summary report showing top potential bias slices
    logger.info(f"📝 Writing bias summary report → {BIAS_SUMMARY_PATH}")

    # Sort by lowest F1 (worst slices first)
    sorted_slices = sorted(slices_results, key=lambda r: r["f1"])

    with open(BIAS_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("LedgerX – Bias Summary Report\n")
        f.write("================================\n\n")
        f.write("Global Test Metrics:\n")
        for k, v in global_metrics.items():
            f.write(f"- {k}: {v}\n")

        f.write("\nWorst Slices by F1 Score:\n")
        for row in sorted_slices[:10]:
            delta = round(global_f1 - row["f1"], 4)
            f.write(
                f"\n[{row['group']} = {row['value']}] "
                f"(n={row['support']}): f1={row['f1']} "
                f"(Δ vs global = {delta})"
            )

        f.write(
            "\n\nInterpretation: Any slice with large negative Δ indicates "
            "a subgroup where the model performs worse. Investigate data quality, "
            "representation, or consider mitigation such as re-weighting or additional "
            "training samples."
        )

    logger.success("✅ Bias slicing & summary reports generated successfully.")
    logger.success(f"📄 Detailed: {BIAS_DETAIL_PATH}")
    logger.success(f"📄 Summary:  {BIAS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
