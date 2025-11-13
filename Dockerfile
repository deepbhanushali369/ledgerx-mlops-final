# ---------- Base Image ----------
FROM apache/airflow:2.9.3-python3.12

# ---------- Switch to root for system-level installs ----------
USER root

# Install OS dependencies (Tesseract, Poppler, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        ghostscript && \
    rm -rf /var/lib/apt/lists/*

# ---------- Switch to airflow user for pip installs ----------
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /requirements.txt

# ---------- Back to root for project setup ----------
USER root

RUN mkdir -p /opt/airflow/{dags,src,data,tests,reports}

# Copy project folders
COPY ./dags /opt/airflow/dags/
COPY ./src /opt/airflow/src/
COPY ./tests /opt/airflow/tests/
COPY ./reports /opt/airflow/reports/
COPY ./start_ledgerx.sh /opt/airflow/start_ledgerx.sh
COPY ./drive_creds.json /opt/airflow/drive_creds.json

# Permissions & defaults
RUN chmod +x /opt/airflow/start_ledgerx.sh
ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH="/opt/airflow/src:${PYTHONPATH}"

WORKDIR /opt/airflow

# ---------- Default command ----------
CMD ["bash", "/opt/airflow/start_ledgerx.sh"]
