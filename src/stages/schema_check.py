# src/stages/schema_check.py
import pandas as pd
from loguru import logger
from pathlib import Path

INPUT_FILE = Path("data/processed/fatura_ocr.csv")
REPORT_FILE = Path("reports/schema_check.txt")

def check_schema():
    df = pd.read_csv(INPUT_FILE)
    expected_cols = {"file_name", "ocr_text"}
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols

    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        if missing:
            f.write(f"❌ Missing columns: {missing}\n")
            logger.error(f"Missing columns: {missing}")
        else:
            f.write("✅ Schema validated: all expected columns present.\n")
            logger.success("Schema validated successfully.")

if __name__ == "__main__":
    check_schema()
