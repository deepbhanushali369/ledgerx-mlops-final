import pandas as pd
from pathlib import Path

# Absolute paths inside Airflow container
INPUT_FILE = Path("/opt/airflow/data/processed/fatura_ocr.csv")
OUTPUT_FILE = Path("/opt/airflow/reports/bias_check_summary.txt")


def detect_bias():
    # Ensure directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load OCR output
    df = pd.read_csv(INPUT_FILE)

    results = []

    # Example bias checks:
    # 1. Check for empty OCR extractions
    empty_text_count = df["ocr_text"].isna().sum()
    results.append(f"Empty OCR entries: {empty_text_count}")

    # 2. Check average text length (helps detect truncation issues)
    avg_length = df["ocr_text"].fillna("").str.len().mean()
    results.append(f"Average OCR text length: {avg_length:.2f}")

    # 3. Check for repeated values (indicates OCR failures)
    duplicate_count = df["ocr_text"].duplicated().sum()
    results.append(f"Duplicate OCR entries: {duplicate_count}")

    # Save report
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))

    print(f"Bias check complete → {OUTPUT_FILE}")


if __name__ == "__main__":
    detect_bias()
