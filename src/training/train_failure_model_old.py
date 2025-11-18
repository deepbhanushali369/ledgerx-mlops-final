import pandas as pd
from pathlib import Path
from loguru import logger
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# Optional models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# ===============================
# Paths
# ===============================
DATA_FILE = Path("data/processed/fatura_model_ready.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"


# ===============================
# Load Data
# ===============================
def load_data():
    if not DATA_FILE.exists():
        logger.error(f"❌ Missing dataset: {DATA_FILE}")
        raise FileNotFoundError(f"{DATA_FILE} not found")

    df = pd.read_csv(DATA_FILE)
    logger.info(f"📄 Loaded dataset → {len(df)} rows")
    return df


# ===============================
# Train/Test Split
# ===============================
def split_data(df):
    X = df.drop(columns=["failure_label"])
    y = df["failure_label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ===============================
# Preprocessing Pipeline
# ===============================
def build_preprocessor(X):
    categorical = ["vendor_name", "currency"]
    numerical = [
        "invoice_number_length",
        "invoice_age_days",
        "total_amount",
        "ocr_text_length",
        "blur_flag",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numerical),
        ]
    )

    return preprocessor


# ===============================
# Train with MLflow
# ===============================
def train_with_mlflow(model_name, model, preprocessor, X_train, y_train, X_val, y_val, params=None):
    with mlflow.start_run(run_name=model_name):

        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        f1 = f1_score(y_val, y_pred)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, model_name)

        logger.info(f"📊 {model_name} F1 Score: {f1:.4f}")
        return pipeline, f1


# ===============================
# Main Training Script
# ===============================
def main():

    logger.info("🚀 Step 2: Training 5 models (Full Performance) with MLflow tracking")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    logger.info("🔧 Building preprocessing pipeline")
    preprocessor = build_preprocessor(X_train)

    mlflow.set_experiment("ledgerx_failure_model")

    results = {}

    # ===============================
    # 1. Logistic Regression
    # ===============================
    lr_model = LogisticRegression(max_iter=2000)
    lr_pipe, lr_f1 = train_with_mlflow(
        "LogisticRegression", lr_model, preprocessor,
        X_train, y_train, X_val, y_val,
        params={"max_iter": 2000}
    )
    results["LogisticRegression"] = (lr_pipe, lr_f1)

    # ===============================
    # 2. Random Forest
    # ===============================
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        class_weight="balanced"
    )
    rf_pipe, rf_f1 = train_with_mlflow(
        "RandomForest", rf_model, preprocessor,
        X_train, y_train, X_val, y_val,
        params={"n_estimators": 300, "max_depth": 15}
    )
    results["RandomForest"] = (rf_pipe, rf_f1)

    # ===============================
    # 3. XGBoost
    # ===============================
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="gpu_hist" if xgb.__version__ else "hist",
            random_state=42,
            eval_metric="logloss"
        )

        xgb_pipe, xgb_f1 = train_with_mlflow(
            "XGBoost", xgb_model, preprocessor,
            X_train, y_train, X_val, y_val
        )
        results["XGBoost"] = (xgb_pipe, xgb_f1)
    except Exception as e:
        logger.error(f"❌ XGBoost failed: {e}")

    # ===============================
    # 4. LightGBM
    # ===============================
    try:
        lgb_model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            device="gpu" if lgb.__version__ else "cpu"
        )

        lgb_pipe, lgb_f1 = train_with_mlflow(
            "LightGBM", lgb_model, preprocessor,
            X_train, y_train, X_val, y_val
        )
        results["LightGBM"] = (lgb_pipe, lgb_f1)
    except Exception as e:
        logger.error(f"❌ LightGBM failed: {e}")

    # ===============================
    # 5. CatBoost
    # ===============================
    try:
        cat_model = CatBoostClassifier(
            iterations=500,
            depth=10,
            learning_rate=0.08,
            loss_function="Logloss",
            verbose=False,
            task_type="GPU" if cat_model else "CPU"
        )

        cat_pipe, cat_f1 = train_with_mlflow(
            "CatBoost", cat_model, preprocessor,
            X_train, y_train, X_val, y_val
        )
        results["CatBoost"] = (cat_pipe, cat_f1)
    except Exception as e:
        logger.error(f"❌ CatBoost failed: {e}")

    # ===============================
    # Select Best Model
    # ===============================
    best_model_name = max(results, key=lambda m: results[m][1])
    best_model, best_f1 = results[best_model_name]

    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.success(f"🏆 Best model: {best_model_name} (F1={best_f1:.4f}) saved to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()