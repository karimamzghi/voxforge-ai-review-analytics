"""Central project configuration and artifact paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SENTIMENT_ARTIFACTS_DIR = MODELS_DIR / "sentiment"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SENTIMENT_PLOT_DIR = PLOTS_DIR / "sentiment"
CLASSIFICATION_REPORTS_DIR = ARTIFACTS_DIR / "classification_reports"
CONFUSION_MATRICES_DIR = ARTIFACTS_DIR / "confusion_matrices"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
MODEL_TRACKING_PATH = ARTIFACTS_DIR / "model_tracking.csv"

RAW_FILES = [
    RAW_DATA_DIR / "1429_1.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
]

CLEANED_REVIEWS_PATH = PROCESSED_DATA_DIR / "cleaned_reviews.csv"
RANDOM_STATE = 42
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"

MODELS_DIR = PROJECT_ROOT / "models" / "sentiment"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

MODEL_PATH = MODELS_DIR / "tfidf_logistic_regression.joblib"
CONFUSION_MATRIX_PATH = (
    CONFUSION_MATRICES_DIR
    / "tfidf_logistic_regression.png"
)
CLASS_DISTRIBUTION_PATH = (
    PLOTS_DIR
    / "sentiment_class_distribution.png"
)
TOP_FEATURES_PATH = (
    PLOTS_DIR
    / "tfidf_logistic_top_features.png"
)
PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "tfidf_logistic_validation.csv"
)
MODEL_TRACKING_PATH = RESULTS_DIR / "model_tracking.csv"

def create_project_directories() -> None:
    for directory in (
        MODELS_DIR,
        CONFUSION_MATRICES_DIR,
        PLOTS_DIR,
        PREDICTIONS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
