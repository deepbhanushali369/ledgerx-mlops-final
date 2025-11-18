from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


def get_logistic_regression():
    model = LogisticRegression(max_iter=2000)
    params = {"max_iter": 2000}
    return "LogisticRegression", model, params


def get_random_forest():
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        class_weight="balanced",
    )
    params = {"n_estimators": 300, "max_depth": 15, "class_weight": "balanced"}
    return "RandomForest", model, params


def get_xgboost():
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",  # CPU-safe
        random_state=42,
        eval_metric="logloss",
    )
    params = {
        "n_estimators": 300,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    }
    return "XGBoost", model, params


def get_lightgbm():
    model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        device="gpu",
    )
    params = {
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "gpu",
    }
    return "LightGBM", model, params


def get_catboost():
    model = CatBoostClassifier(
        iterations=500,
        depth=10,
        learning_rate=0.08,
        loss_function="Logloss",
        verbose=False,
        task_type="GPU",
    )
    params = {
        "iterations": 500,
        "depth": 10,
        "learning_rate": 0.08,
        "loss_function": "Logloss",
        "task_type": "GPU",
    }
    return "CatBoost", model, params


def get_all_models():
    models = {}

    for getter in [
        get_logistic_regression,
        get_random_forest,
        get_xgboost,
        get_lightgbm,
        get_catboost,
    ]:
        name, model, params = getter()
        models[name] = (model, params)

    return models
