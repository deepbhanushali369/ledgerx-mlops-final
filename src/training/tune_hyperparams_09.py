import os
import optuna
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .load_data_01 import load_data, split_data
from .preprocessing_02 import build_preprocessor

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


MLFLOW_EXPERIMENT = "ledgerx_failure_model_tuning"
MODEL_SAVE_PATH = "models/best_model_tuned.pkl"
REPORT_PATH = "data/reports/tuning_report.txt"


# ---------------------------------------------------------
# Model Evaluation Using F1 Score (your main metric)
# ---------------------------------------------------------
def evaluate(model, X_val, y_val):
    preds = model.predict(X_val)
    return f1_score(y_val, preds)


# ---------------------------------------------------------
# OBJECTIVE: CatBoost
# ---------------------------------------------------------
def objective_catboost(trial, X_train, y_train, X_val, y_val, preprocessor):

    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "Logloss",
        "verbose": False,
        "task_type": "GPU"
    }

    model = CatBoostClassifier(**params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    f1 = evaluate(pipeline, X_val, y_val)
    return f1


# ---------------------------------------------------------
# OBJECTIVE: XGBoost
# ---------------------------------------------------------
def objective_xgb(trial, X_train, y_train, X_val, y_val, preprocessor):

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "eval_metric": "logloss",
        "tree_method": "hist"
    }

    model = XGBClassifier(**params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    f1 = evaluate(pipeline, X_val, y_val)
    return f1


# ---------------------------------------------------------
# OBJECTIVE: LightGBM
# ---------------------------------------------------------
def objective_lgbm(trial, X_train, y_train, X_val, y_val, preprocessor):

    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)
    }

    model = LGBMClassifier(**params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    f1 = evaluate(pipeline, X_val, y_val)
    return f1


# ---------------------------------------------------------
# RUN OPTUNA STUDY FOR ONE MODEL
# ---------------------------------------------------------
def run_study(model_name, objective_fn, X_train, y_train, X_val, y_val, preprocessor):

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"Tuning_{model_name}"):

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective_fn(trial, X_train, y_train, X_val, y_val, preprocessor),
            n_trials=20,
            show_progress_bar=True
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)

    return study.best_value, study.best_params


# ---------------------------------------------------------
# MAIN TUNING PROCESS
# ---------------------------------------------------------
def main():

    print("\n📘 Starting Hyperparameter Tuning...")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor()

    results = {}

    # ---- CatBoost ----
    print("\n🔍 Tuning CatBoost...")
    results["catboost"] = run_study(
        "CatBoost", objective_catboost,
        X_train, y_train, X_val, y_val, preprocessor
    )

    # ---- XGBoost ----
    print("\n🔍 Tuning XGBoost...")
    results["xgboost"] = run_study(
        "XGBoost", objective_xgb,
        X_train, y_train, X_val, y_val, preprocessor
    )

    # ---- LightGBM ----
    print("\n🔍 Tuning LightGBM...")
    results["lightgbm"] = run_study(
        "LightGBM", objective_lgbm,
        X_train, y_train, X_val, y_val, preprocessor
    )

    # -----------------------------------------------------
    # Select best tuned model
    # -----------------------------------------------------
    best_model_name = max(results, key=lambda k: results[k][0])
    best_params = results[best_model_name][1]

    print(f"\n🏆 Best Tuned Model: {best_model_name.upper()}")
    print(best_params)

    # Build final model
    if best_model_name == "catboost":
        model = CatBoostClassifier(**best_params, verbose=False, task_type="GPU")
    elif best_model_name == "xgboost":
        model = XGBClassifier(**best_params)
    else:
        model = LGBMClassifier(**best_params)

    final_preprocessor = build_preprocessor()

    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("model", model)
    ])

    final_pipeline.fit(X_train, y_train)

    # Save final tuned model
    joblib.dump(final_pipeline, MODEL_SAVE_PATH)
    print(f"💾 Saved Tuned Model → {MODEL_SAVE_PATH}")

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("LedgerX – Hyperparameter Tuning Report\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write("Best Parameters:\n")
        for k, v in best_params.items():
            f.write(f"- {k}: {v}\n")

    print(f"📄 Tuning Report Saved → {REPORT_PATH}")
    print("\n🎉 Hyperparameter Tuning Completed!\n")


if __name__ == "__main__":
    main()
