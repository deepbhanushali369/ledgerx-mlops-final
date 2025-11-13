from loguru import logger
import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/processed/fatura_ocr.csv")

def validate_schema():
    if not DATA_FILE.exists():
        logger.error(f"❌ File not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    logger.info(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")

    expected_cols = {"file_name", "ocr_text"}
    missing = expected_cols - set(df.columns)

    if missing:
        logger.error(f"❌ Missing expected columns: {missing}")
        raise SystemExit(1)
    else:
        logger.success("✅ Schema validation passed!")

if __name__ == "__main__":
    validate_schema()
