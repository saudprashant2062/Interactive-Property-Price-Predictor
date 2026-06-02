from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_INPUT_FILE = RAW_DATA_DIR / "properties.csv"
DEFAULT_MODEL_FILE = MODEL_DIR / "xgboost_model.joblib"
DEFAULT_SCALER_FILE = MODEL_DIR / "scaler.joblib"
DEFAULT_ENCODER_FILE = MODEL_DIR / "encoder.joblib"

TARGET_COLUMN = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2
