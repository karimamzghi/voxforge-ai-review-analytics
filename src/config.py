"""Central project configuration and artifact paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLAB_PROJECT_ROOT = Path(
        "/content/drive/MyDrive/Ironhack-challenges/"
        "voxforge-ai-review-analytics"
    )

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_FILES = [
    RAW_DATA_DIR / "1429_1.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
]
CLEANED_REVIEWS_PATH = PROCESSED_DATA_DIR / "cleaned_reviews.csv"

MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODELS_DIR = MODELS_DIR / "sentiment"

RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
MODEL_TRACKING_PATH = RESULTS_DIR / "model_tracking.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

RANDOM_STATE = 42
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

def create_project_directories() -> None:
    """Create all reusable output directories."""
    for directory in (
        SENTIMENT_MODELS_DIR, RESULTS_DIR, CONFUSION_MATRICES_DIR,
        PLOTS_DIR, PREDICTIONS_DIR, CLASSIFICATION_REPORTS_DIR, REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
