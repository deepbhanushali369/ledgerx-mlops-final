# src/stages/bias_check.py
import pandas as pd
from loguru import logger
from pathlib import Path

INPUT_FILE = Path("data/processed/fatura_ocr.csv")
REPORT_FILE = Path("reports/bias_check_summary.txt")

def detect_bias():
    df = pd.read_csv(INPUT_FILE)
    df["text_length"] = df["ocr_text"].fillna("").str.len()

    # Example slice: short vs long invoices
    short_mean = df[df["text_length"] < 100]["text_length"].mean()
    long_mean = df[df["text_length"] >= 100]["text_length"].mean()

    diff = abs(short_mean - long_mean)
    bias_flag = diff > 50

    REPORT_FILE.parent.mkdir(exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(f"Short text avg: {short_mean:.2f}\n")
        f.write(f"Long text avg: {long_mean:.2f}\n")
        f.write(f"Bias detected: {bias_flag}\n")

    logger.info(f"Bias check complete — difference = {diff:.2f}, bias={bias_flag}")

if __name__ == "__main__":
    detect_bias()
