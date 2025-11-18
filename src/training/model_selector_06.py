from pathlib import Path
import joblib
from loguru import logger


def select_and_save_best_model(results: dict, out_path=Path("models/best_model.pkl")):
    if not results:
        raise ValueError("No models were successfully trained.")

    best_model_name = max(results, key=lambda m: results[m][1])
    best_model, best_f1 = results[best_model_name]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)

    logger.success(
        f"🏆 Best model: {best_model_name} (F1={best_f1:.4f}) saved to {out_path}"
    )

    return best_model_name, best_f1
