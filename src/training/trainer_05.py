import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from loguru import logger
import mlflow

from .mlflow_utils_04 import log_params, log_metrics, log_pipeline

# ======================================================
# CLEAN ALL TRAINING WARNINGS
# ======================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="lightgbm")
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="mlflow")

# Disable CatBoost console spam
import os
os.environ["CATBOOST_VERBOSE"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train_model_with_mlflow(
    model_name: str,
    model,
    preprocessor,
    X_train,
    y_train,
    X_val,
    y_val,
    params=None,
):
    """
    Train a model with MLflow logging and return (pipeline, f1).
    All warnings suppressed for clean output.
    """
    with mlflow.start_run(run_name=model_name):

        if params:
            log_params(params)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        logger.info(f"🚂 Training {model_name} ...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        log_metrics({"f1_score": f1})
        log_pipeline(model_name, pipeline)

        logger.info(f"📊 {model_name} – F1 Score: {f1:.4f}")

        return pipeline, f1
