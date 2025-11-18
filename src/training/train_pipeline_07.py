from loguru import logger

from .load_data_01 import load_data, split_data
from .preprocessing_02 import build_preprocessor
from .model_definitions_03 import get_all_models
from .trainer_05 import train_model_with_mlflow
from .model_selector_06 import select_and_save_best_model
from .mlflow_utils_04 import init_experiment


def main():

    logger.info("🚀 Modular Training Pipeline Started (5 Models)")

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor()

    init_experiment("ledgerx_failure_model")

    models = get_all_models()
    results = {}

    for model_name, (model, params) in models.items():
        try:
            pipeline, f1 = train_model_with_mlflow(
                model_name=model_name,
                model=model,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                params=params,
            )
            results[model_name] = (pipeline, f1)

        except Exception as e:
            logger.error(f"❌ {model_name} failed: {e}")

    best_name, best_f1 = select_and_save_best_model(results)

    logger.info(f"🎉 Pipeline Finished — Best Model: {best_name} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
