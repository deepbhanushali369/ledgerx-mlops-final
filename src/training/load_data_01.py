from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/processed/fatura_model_ready.csv")


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        logger.error(f"❌ Missing dataset: {DATA_FILE}")
        raise FileNotFoundError(f"{DATA_FILE} not found")

    df = pd.read_csv(DATA_FILE)
    logger.info(f"📄 Loaded dataset → {len(df)} rows")
    return df


def split_data(df: pd.DataFrame):
    if "failure_label" not in df.columns:
        raise KeyError("Column 'failure_label' not found in dataset")

    X = df.drop(columns=["failure_label"])
    y = df["failure_label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(
        f"🔀 Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
