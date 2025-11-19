import json
import tarfile
import subprocess
from datetime import datetime
from pathlib import Path
from loguru import logger
import joblib
import os

# ============================================================
# CONFIGURATION (Your values)
# ============================================================

PROJECT_ID = "mlops-ledgerx"
REGION = "us-central1"
REPOSITORY_NAME = "ledgerx-models"
PACKAGE_NAME = "ledgerx-failure-model"

# Full absolute path to gcloud.cmd (Windows)
GCLOUD = os.environ.get(
    "GCLOUD_PATH",
    r"C:\Users\Deep\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
)

MODEL_DIR = Path("models")
TUNED_MODEL_PATH = MODEL_DIR / "best_model_tuned.pkl"
BASE_MODEL_PATH = MODEL_DIR / "best_model.pkl"

REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_REPORT = REPORT_DIR / "model_registry_report.txt"

PACKAGE_OUTPUT_DIR = Path("model_packages")
PACKAGE_OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Utility: Get Next Model Version
# ============================================================

def get_next_version():
    """Find existing versions in Artifact Registry and auto-increment."""
    logger.info("🔍 Checking existing model versions from Artifact Registry...")

    cmd = [
        GCLOUD, "artifacts", "versions", "list",
        f"--project={PROJECT_ID}",
        f"--location={REGION}",
        f"--repository={REPOSITORY_NAME}",
        f"--package={PACKAGE_NAME}",
        "--format=json"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        versions = json.loads(result.stdout)
    except subprocess.CalledProcessError:
        logger.warning("⚠️ No existing versions found. Starting from v1.")
        return "v1"

    if not versions:
        return "v1"

    version_ids = [v["name"].split("/")[-1] for v in versions if "name" in v]
    version_nums = [int(v.replace("v", "")) for v in version_ids if v.startswith("v")]

    next_version = f"v{max(version_nums) + 1}"
    logger.info(f"🔢 Next model version → {next_version}")
    return next_version


# ============================================================
# Utility: Create metadata file
# ============================================================

def create_metadata(version):
    """Create metadata JSON for this model version."""
    metadata = {
        "version": version,
        "project_id": PROJECT_ID,
        "repository": REPOSITORY_NAME,
        "package": PACKAGE_NAME,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "trained_model": "best_model_tuned.pkl",
        "notes": "Auto-uploaded from MLOPs training pipeline."
    }

    metadata_path = PACKAGE_OUTPUT_DIR / f"metadata_{version}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"📝 Metadata created → {metadata_path}")
    return metadata_path


# ============================================================
# Step 1: Select & package model
# ============================================================

def package_model(version):
    logger.info("📦 Preparing model package...")

    if TUNED_MODEL_PATH.exists():
        model_path = TUNED_MODEL_PATH
    else:
        model_path = BASE_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError("❌ No model file found to package.")

    metadata_path = create_metadata(version)

    tar_name = PACKAGE_OUTPUT_DIR / f"{PACKAGE_NAME}_{version}.tar.gz"

    with tarfile.open(tar_name, "w:gz") as tar:
        tar.add(model_path, arcname="model.pkl")
        tar.add(metadata_path, arcname="metadata.json")

    logger.success(f"📦 Model package created → {tar_name}")
    return tar_name


# ============================================================
# Step 2: Upload to Google Artifact Registry
# ============================================================

def upload_to_registry(tar_path, version):
    logger.info("🚀 Uploading model to Google Artifact Registry...")

    cmd = [
        GCLOUD, "artifacts", "generic", "upload",
        f"--project={PROJECT_ID}",
        f"--location={REGION}",
        f"--repository={REPOSITORY_NAME}",
        f"--package={PACKAGE_NAME}",
        f"--version={version}",
        f"--source={tar_path}"
    ]

    logger.info(f"Executing: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)

    logger.success(f"🚀 Model version {version} uploaded successfully!")
    return True


# ============================================================
# Step 3: Write registry report
# ============================================================

def write_registry_report(version, tar_path):
    with open(REGISTRY_REPORT, "w", encoding="utf-8") as f:
        f.write("LedgerX – Model Registry Upload Report\n")
        f.write("=======================================\n\n")
        f.write(f"Model Package: {PACKAGE_NAME}\n")
        f.write(f"Version Uploaded: {version}\n")
        f.write(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Tarball Path: {tar_path}\n")
        f.write(f"GCP Project: {PROJECT_ID}\n")
        f.write(f"Repository: {REPOSITORY_NAME}\n")
        f.write(f"Region: {REGION}\n\n")

    logger.success(f"📄 Registry report generated → {REGISTRY_REPORT}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("🚀 Starting Model Registry Upload (Artifact Registry)")

    version = get_next_version()
    tar_path = package_model(version)
    upload_to_registry(tar_path, version)
    write_registry_report(version, tar_path)

    logger.success("🎉 Model Registry Process Completed!")


if __name__ == "__main__":
    main()
