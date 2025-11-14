#!/bin/bash
set -e

echo "🚀 Booting LedgerX Airflow Environment..."

# -----------------------------
# 1. Database upgrade
# -----------------------------
echo "🔧 Initializing Airflow DB..."
airflow db upgrade

# -----------------------------
# 2. Create Admin user IF NOT EXISTS
# -----------------------------
echo "👤 Ensuring admin user exists..."

airflow users list | grep -w "admin" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --password admin \
        --role Admin \
        --email admin@example.com
    echo "✅ Admin user created"
else
    echo "ℹ️ Admin user already exists"
fi

# -----------------------------
# 3. Configure Git (for DVC)
# -----------------------------
echo "🌐 Configuring DVC + Git..."
git config --global user.name "LedgerX Pipeline"
git config --global user.email "pipeline@ledgerx.local"

# -----------------------------
# 4. Start Airflow services
# -----------------------------
echo "🚀 Starting Airflow Webserver + Scheduler..."

airflow webserver &
airflow scheduler
