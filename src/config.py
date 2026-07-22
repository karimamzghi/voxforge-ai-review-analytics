from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SENTIMENT_ARTIFACTS_DIR = ARTIFACTS_DIR / "sentiment"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOT_DIR = ARTIFACTS_DIR / "plots"


RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_FILES = [
    RAW_DATA_DIR / "1429_1.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
]

RANDOM_STATE = 42
