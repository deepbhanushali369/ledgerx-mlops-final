import pandas as pd
from pathlib import Path
from loguru import logger

# ===============================
# Paths
# ===============================
CLEANED_FILE = Path("data/processed/fatura_cleaned.csv")
OUT_FILE = Path("data/processed/fatura_model_ready.csv")
REPORT_FILE = Path("data/reports/feature_build_report.txt")


# ===============================
# Helper Functions
# ===============================

def derive_invoice_number_length(df):
    return df["invoice_number"].astype(str).str.len()


def derive_invoice_age_days(df):
    # Convert invoice_date to datetime
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    today = pd.Timestamp.today()
    return (today - df["invoice_date"]).dt.days


def compute_failure_label(df):
    """
    failure_label = 1 if ANY:
    - vendor_name == UNKNOWN_VENDOR
    - currency == UNK
    - total_amount <= 0
    - invoice_number missing
    - invoice_date == "2000-01-01" (invalid placeholder used in cleaning)
    """

    conditions = (
        (df["vendor_name"] == "UNKNOWN_VENDOR") |
        (df["currency"] == "UNK") |
        (df["total_amount"] <= 0) |
        (df["invoice_number"].isna()) |
        (df["invoice_date"].astype(str) == "2000-01-01")
    )

    return conditions.astype(int)


# ===============================
# Main Script
# ===============================

def main():

    logger.info("🚀 Step 1 (Structured-Only): Building model-ready dataset")

    # ============================================
    # 1. Load Cleaned Structured Data
    # ============================================
    if not CLEANED_FILE.exists():
        logger.error(f"❌ Cleaned file missing: {CLEANED_FILE}")
        raise FileNotFoundError(f"Missing: {CLEANED_FILE}")

    df = pd.read_csv(CLEANED_FILE)
    logger.info(f"📄 Loaded cleaned file → {len(df)} rows")

    # ============================================
    # 2. Create Derived Features (Structured Only)
    # ============================================
    df["invoice_number_length"] = derive_invoice_number_length(df)
    df["invoice_age_days"] = derive_invoice_age_days(df)

    # Since we do not have OCR:
    df["ocr_text_length"] = 0
    df["blur_flag"] = 0

    # ============================================
    # 3. Compute Failure Label
    # ============================================
    df["failure_label"] = compute_failure_label(df)

    logger.info(
        f"🏷 failure_label distribution → "
        f"{df['failure_label'].value_counts().to_dict()}"
    )

    # ============================================
    # 4. Save Final Model-Ready Dataset
    # ============================================
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    logger.success(f"💾 Saved model-ready CSV → {OUT_FILE}")

    # ============================================
    # 5. Write Summary Report
    # ============================================
    summary_lines = [
        "=== LedgerX Model-Ready Dataset Report (Structured-Only) ===",
        f"Rows: {len(df)}",
        "",
        "--- Failure Label Distribution ---",
        str(df["failure_label"].value_counts().to_dict()),
        "",
        "--- Missing Values Summary ---",
        str(df.isna().sum().to_dict()),
        "",
        "--- Sample Columns ---",
        str(list(df.columns)),
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(summary_lines))

    logger.success(f"📝 Report written → {REPORT_FILE}")
    logger.info("✅ Step 1 Completed Successfully!")


if __name__ == "__main__":
    main()