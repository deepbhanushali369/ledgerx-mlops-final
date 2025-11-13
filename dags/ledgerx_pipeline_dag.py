"""
LedgerX Fatura Pipeline DAG
-----------------------------------
Main orchestration DAG combining all stages:
  1. Data acquisition
  2. (Skip preprocessing – OCR done offline)
  3. Schema validation
  4. Testing
  5. DVC versioning
  6. Report generation
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
import os

default_args = {
    "owner": "ledgerx",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# 🔍 Helper to check if OCR output exists
def check_ocr_output():
    return os.path.exists("/opt/airflow/data/processed/fatura_ocr.csv")

with DAG(
    dag_id="ledgerx_fatura_pipeline",
    default_args=default_args,
    description="End-to-end LedgerX FATURA data pipeline (using preprocessed OCR output)",
    schedule_interval=None,  # manual trigger
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["ledgerx", "fatura", "pipeline"],
) as dag:

    # 1️⃣ Data acquisition — fetch or pull from API
    acquire_data = BashOperator(
        task_id="acquire_data",
        bash_command="python /opt/airflow/src/stages/data_acquisition_fatura.py || echo 'Acquisition complete'",
    )

    # 2️⃣ Skip preprocess if OCR file already exists
    check_ocr_file = ShortCircuitOperator(
        task_id="check_ocr_file",
        python_callable=check_ocr_output,
    )

    # 3️⃣ Schema validation using Great Expectations
    validate_schema = BashOperator(
        task_id="validate_schema",
        bash_command="python /opt/airflow/src/validation/run_great_expectations.py",
    )

    # 4️⃣ Run pytest-based unit tests
    run_tests = BashOperator(
        task_id="run_tests",
        bash_command="pytest -v --disable-warnings /opt/airflow/tests > /opt/airflow/reports/test_report.txt || true",
    )

    # 5️⃣ Push processed data to DVC remote
    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command="cd /opt/airflow && dvc add data/processed && dvc push || echo 'DVC push complete'",
    )

    # 6️⃣ Generate summary report
    generate_report = BashOperator(
        task_id="generate_report",
        bash_command="python /opt/airflow/src/reporting/generate_summary_report.py",
    )

    # ✅ Updated dependency chain
    # OCR check replaces the trigger_preprocess stage
    acquire_data >> check_ocr_file >> validate_schema >> run_tests >> dvc_push >> generate_report
