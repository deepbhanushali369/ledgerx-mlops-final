from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from loguru import logger

CATEGORICAL_FEATURES = ["vendor_name", "currency"]
NUMERICAL_FEATURES = [
    "invoice_number_length",
    "invoice_age_days",
    "total_amount",
    "ocr_text_length",
    "blur_flag",
]


def build_preprocessor():
    logger.info("🔧 Building preprocessing transformer")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )
    return preprocessor
