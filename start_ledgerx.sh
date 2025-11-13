#!/bin/bash
set -e

echo "🚀 Booting LedgerX Airflow + DVC Environment..."
cd /opt/airflow || exit 1

echo "🔧 Initializing Airflow database..."
airflow db init

# Create admin user if it doesn't exist
if ! airflow users list | grep -q "admin@example.com"; then
  echo "👤 Creating Airflow admin user..."
  airflow users create \
    --username admin \
    --firstname LedgerX \
    --lastname Admin \
    --role Admin \
    --email admin@example.com \
    --password admin
else
  echo "✅ Admin user already exists."
fi

echo "🌐 Configuring DVC..."
if [ ! -d ".dvc" ]; then
  dvc init --no-scm
fi

# Add Google Drive remote (your ID is already set)
dvc remote add -d gdrive gdrive://1nDGBfjE3BSEPobsMewLJi8lGknsKTgfT >/dev/null 2>&1 || true
dvc remote modify gdrive gdrive_user_credentials_file /opt/airflow/drive_creds.json || true

echo "⚙️ Setting Git identity..."
git init || true
git config user.email "admin@example.com"
git config user.name "LedgerX"

echo "✅ Starting Airflow standalone..."
exec airflow standalone

echo '?? Pulling latest data via DVC...'
dvc pull || true
