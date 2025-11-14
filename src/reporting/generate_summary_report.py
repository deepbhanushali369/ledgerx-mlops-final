"""
Generate Summary Report for LedgerX FATURA Pipeline
---------------------------------------------------

This script loads the processed OCR CSV and generates
a simple summary report with:

- Total invoices processed
- Missing OCR text count
- Average OCR text length
- Timestamp of report generation

Output is saved to:
    /opt/airflow/reports/summary_report.txt
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

INPUT_FILE = Path("/opt/airflow/data/processed/fatura_ocr.csv")
OUTPUT_DIR = Path("/opt/airflow/reports")
OUTPUT_FILE = OUTPUT_DIR / "summary_report.txt"


def generate_report():
    print(f"📄 Generating summary report...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check input exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

    # Load data
    df = pd.read_csv(INPUT_FILE)

    total_rows = len(df)
    missing_ocr = df['ocr_text'].isna().sum()
    avg_text_length = df['ocr_text'].astype(str).apply(len).mean()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write report
    with open(OUTPUT_FILE, "w") as f:
        f.write("LedgerX FATURA Pipeline Summary Report\n")
        f.write("--------------------------------------\n")
        f.write(f"Generated at: {timestamp}\n\n")
        f.write(f"Total invoices processed: {total_rows}\n")
        f.write(f"Invoices with missing OCR text: {missing_ocr}\n")
        f.write(f"Average OCR text length: {avg_text_length:.2f}\n")

    print(f"✅ Summary report created at: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_report()
