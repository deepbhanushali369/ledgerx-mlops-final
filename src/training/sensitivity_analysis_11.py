from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from .load_data_01 import load_data, split_data

# Try importing shap – user must install it via `pip install shap`
import shap

# Paths
MODEL_TUNED_PATH = Path("models/best_model_tuned.pkl")
MODEL_BASE_PATH = Path("models/best_model.pkl")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FI_FIG_PATH = REPORT_DIR / "feature_importance.png"
SHAP_FIG_PATH = REPORT_DIR / "shap_summary.png"
SENSITIVITY_REPORT_PATH = REPORT_DIR / "sensitivity_report.txt"


def get_pipeline():
    """
    Load the tuned best model pipeline if available, otherwise fall back
    to the baseline best model.
    """
    if MODEL_TUNED_PATH.exists():
        model_path = MODEL_TUNED_PATH
    elif MODEL_BASE_PATH.exists():
        model_path = MODEL_BASE_PATH
    else:
        raise FileNotFoundError(
            "Neither best_model_tuned.pkl nor best_model.pkl was found in models/"
        )

    pipeline = joblib.load(model_path)
    logger.info(f"📦 Loaded pipeline for sensitivity analysis → {model_path}")
    return pipeline, model_path


def get_feature_names(preprocessor, X_sample: pd.DataFrame):
    """
    Try to get transformed feature names from the ColumnTransformer.
    Fallback to generic names if unavailable.
    """
    try:
        # sklearn >= 1.0
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except Exception:
        logger.warning("Could not extract feature names from preprocessor. Using generic names.")
        return [f"feat_{i}" for i in range(preprocessor.transform(X_sample).shape[1])]


def plot_feature_importance(model, feature_names, out_path, top_n=20):
    """
    Plot bar chart of top_n feature importances.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_. Skipping feature importance plot.")
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        logger.warning(
            f"Length mismatch in feature names ({len(feature_names)}) and importances ({len(importances)}). "
            "Truncating to smallest length."
        )
        n = min(len(feature_names), len(importances))
        importances = importances[:n]
        feature_names = feature_names[:n]

    # Get top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    logger.success(f"📊 Feature importance plot saved → {out_path}")


def compute_shap_summary(model, preprocessor, X_train: pd.DataFrame, feature_names, out_path, max_samples=500):
    """
    Compute and save a SHAP summary plot for the tree-based model.
    We compute SHAP on the transformed feature space.
    """
    # Subsample for speed
    if len(X_train) > max_samples:
        X_sample = X_train.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_train.copy()

    # Transform features
    X_trans = preprocessor.transform(X_sample)

    logger.info(f"Using {X_trans.shape[0]} samples and {X_trans.shape[1]} transformed features for SHAP.")

    # TreeExplainer is efficient for tree-based models (CatBoost, XGBoost, LGBM, RF)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)

        plt.figure(figsize=(10, 6))
        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1]
        else:
            sv = shap_values

        shap.summary_plot(sv, X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        logger.success(f"📈 SHAP summary plot saved → {out_path}")

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")


def main():
    logger.info("📘 Running Sensitivity Analysis (Feature Importance & SHAP)")

    # 1) Load data and split
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 2) Load pipeline and extract components
    pipeline, model_path = get_pipeline()

    if "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain 'preprocessor' and 'model' steps.")

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # 3) Determine transformed feature names
    feature_names = get_feature_names(preprocessor, X_train)

    # 4) Plot feature importance (if available)
    plot_feature_importance(model, feature_names, FI_FIG_PATH, top_n=20)

    # 5) Compute SHAP summary
    compute_shap_summary(model, preprocessor, X_train, feature_names, SHAP_FIG_PATH)

    # 6) Write sensitivity report
    logger.info(f"📝 Writing sensitivity report → {SENSITIVITY_REPORT_PATH}")
    with open(SENSITIVITY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("LedgerX – Sensitivity Analysis Report\n")
        f.write("=====================================\n\n")
        f.write(f"Model path used: {model_path}\n\n")
        f.write("Includes:\n")
        f.write("- Top feature importance bar chart (feature_importance.png)\n")
        f.write("- Global SHAP summary plot (shap_summary.png)\n\n")
        f.write(
            "Interpretation guidance:\n"
            "- Features with higher importance (or larger absolute SHAP values) have a stronger\n"
            "  influence on the model's predictions.\n"
            "- Use the SHAP summary plot to see which features consistently push predictions\n"
            "  toward failure vs. success, and whether any features behave unexpectedly.\n"
        )

    logger.success("✅ Sensitivity analysis completed successfully.")
    logger.success(f"📄 Text report: {SENSITIVITY_REPORT_PATH}")
    logger.success(f"🖼️ Feature importance: {FI_FIG_PATH}")
    logger.success(f"🖼️ SHAP summary: {SHAP_FIG_PATH}")


if __name__ == "__main__":
    main()
