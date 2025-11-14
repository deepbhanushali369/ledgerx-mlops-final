import pandas as pd
from pathlib import Path

# 👇 Absolute paths inside Airflow container
INPUT_FILE = Path("/opt/airflow/data/processed/fatura_ocr.csv")
OUTPUT_FILE = Path("/opt/airflow/reports/schema_check.txt")

def check_schema():
    # ✔ Ensure report directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ✔ Load the OCR CSV
    df = pd.read_csv(INPUT_FILE)

    # ✔ Define expected schema
    expected_columns = ["file_name", "ocr_text"]

    results = []

    # Check columns exist
    for col in expected_columns:
        if col in df.columns:
            results.append(f"✔ Column present: {col}")
        else:
            results.append(f"❌ MISSING column: {col}")

    # Check number of records
    results.append(f"Total rows: {len(df)}")

    # Save report
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))

    print(f"Schema check complete → {OUTPUT_FILE}")


if __name__ == "__main__":
    check_schema()
