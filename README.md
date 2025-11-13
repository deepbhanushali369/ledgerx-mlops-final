# LedgerX - AI Powered Invoice Intelligence Platform (Failure-Aware MLOps)

Team: Lochan Enugula, Jash Bhavesh Shah, Samruddhi Bansod, Rutuja Jadhav, Nirali Hirpara, Deep Bhanushali  
Course: IE 7305 - Machine Learning Operations (Prof. Ramin Mohammadi)  
Institution: Northeastern University, Boston  

---

## Overview
LedgerX is an end-to-end AI-driven Accounts Payable automation system that performs OCR-based invoice extraction, schema validation, failure detection, and analytics using a failure-aware MLOps pipeline orchestrated by Apache Airflow and version-controlled with DVC.

Core capabilities:
- Parallel OCR with Tesseract and Pillow
- Schema validation using Great Expectations
- Failure-aware anomaly and drift detection
- Automated data versioning with DVC
- Continuous integration using Airflow DAGs
- Testing and reproducibility through pytest

---

## Objectives
| # | Objective | Description |
|---|------------|-------------|
| 1 | Automated OCR Extraction | High-throughput Tesseract OCR for invoices across vendors |
| 2 | Schema Validation | Detects missing totals, blur, or layout drift |
| 3 | Failure Prediction | Flags invoices likely to fail downstream |
| 4 | GL Mapping and Forecasting | Suggests accounting codes and predicts cash flow |
| 5 | Vendor Intelligence Graph | Detects duplicate or anomalous vendors |
| 6 | Active Learning Loop | Human feedback retrains models automatically |

---

## Architecture
Invoices -> Preprocess_FATURA (OCR) -> Schema Validation  
       |                                  |  
       v                                  v  
   DVC Tracking & Versioning        Failure Predictor  
       |                                  |  
       -> Airflow DAG Orchestration (ledgerx_fatura_pipeline)  
                       |  
                       v  
            Reports / Monitoring / Retraining

Stack:
- Airflow 2.9 (Python 3.12, Dockerized)
- PostgreSQL metadata database
- DVC + Google Drive remote
- Great Expectations for schema validation
- Tesseract OCR, Pillow, and OpenCV
- pytest for testing
- Prometheus and Grafana for monitoring

---

## Repository Structure
ledgerx-mlops-final/
├── dags/                     - Airflow DAG definitions
│   └── ledgerx_pipeline_dag.py
├── src/                      - Core Python modules
│   ├── preprocess_fatura.py  - OCR pipeline
│   └── validation/           - Great Expectations checks
├── data/
│   ├── raw/                  - Source invoices (e.g., FATURA)
│   └── processed/            - OCR outputs (fatura_ocr.csv)
├── tests/                    - pytest unit tests
├── reports/                  - Schema and validation reports
├── Dockerfile                - Airflow custom image
├── docker-compose.yml        - Multi-service setup
├── requirements.txt          - Python dependencies
├── start_ledgerx.sh          - Container startup script
└── README.md

---

## Local Setup and Usage
python -m venv .venv  
.\.venv\Scripts\activate  
pip install -r requirements.txt  
python src/preprocess_fatura.py  

Output: data/processed/fatura_ocr.csv  

---

## Running Airflow with Docker
docker compose up --build -d  
docker ps  
Access Airflow at http://localhost:8081 (admin/admin)  

Trigger DAG: ledgerx_fatura_pipeline  
Tasks: extract_ocr → validate_schema → bias_detection → unit_tests → dvc_push  

Expected Logs:  
File found data/processed/fatura_ocr.csv  
Schema validation passed  
DVC push complete  

---

## DVC Data Versioning
dvc init  
dvc remote add -d gdrive gdrive://<YOUR_DRIVE_ID>  
dvc add data/processed/fatura_ocr.csv  
git add data/processed/fatura_ocr.csv.dvc  
git commit -m "Add processed OCR dataset"  
dvc push  

---

## Testing
pytest -v --disable-warnings  

---

## Troubleshooting
| Problem | Cause | Fix |
|----------|--------|-----|
| File not found | Volume not mounted | Add - ./data:/opt/airflow/data |
| git not found | Missing in image | Add git in Dockerfile apt-get |
| DAG missing | Wrong mount | Mount ./dags:/opt/airflow/dags |
| Permission denied | File access | chmod -R 777 data/ logs/ reports/ |

---

## Metrics and Targets
OCR F1 >= 90%  
GL Mapping >= 88%  
Failure AUC >= 0.85  
Validation Pass >= 92%  
Latency <= 5 s  
Coverage >= 80%  

---

## Roadmap
[x] OCR Pipeline  
[x] Schema Validation  
[x] DVC Integration and Airflow DAG  
[ ] Failure Prediction (LayoutLM/LSTM)  
[ ] Vendor Graph and Forecasting  
[ ] CI/CD (GitHub Actions + MLflow)  

---

## License and Acknowledgments
Open-source educational project for IE 7305 (MLOps) at Northeastern University.  
Datasets: SROIE, CORD, DocVQA, and synthetic FATURA invoices.  
© 2025 LedgerX Team – For academic use only.
