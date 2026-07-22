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
PLOT_DIR = ARTIFACTS_DIR / "plots"
SENTIMENT_PLOT_DIR = PLOT_DIR / "sentiment"
SENTIMENT_REPORTS_DIR = REPORTS_DIR / "sentiment"

RAW_FILES = [
    RAW_DATA_DIR / "1429_1.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
    RAW_DATA_DIR / "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv",
]

CLEANED_REVIEWS_PATH = PROCESSED_DATA_DIR / "cleaned_reviews.csv"
RANDOM_STATE = 42
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

TFIDF_LOGISTIC_MODEL_PATH = (
    SENTIMENT_ARTIFACTS_DIR / "tfidf_logistic_regression_pipeline.joblib"
)
TFIDF_LOGISTIC_METRICS_PATH = (
    SENTIMENT_REPORTS_DIR / "tfidf_logistic_regression_metrics.json"
)
TFIDF_LOGISTIC_REPORT_PATH = (
    SENTIMENT_REPORTS_DIR / "tfidf_logistic_regression_classification_report.csv"
)
TFIDF_LOGISTIC_RESULT_PATH = (
    SENTIMENT_REPORTS_DIR / "tfidf_logistic_regression_result.csv"
)
TFIDF_LOGISTIC_TOP_FEATURES_PATH = (
    SENTIMENT_REPORTS_DIR / "tfidf_logistic_regression_top_features.csv"
)
TFIDF_LOGISTIC_CONFUSION_MATRIX_PATH = (
    SENTIMENT_PLOT_DIR / "tfidf_logistic_regression_confusion_matrix.png"
)
TFIDF_LOGISTIC_CONFUSION_MATRIX_CSV_PATH = (
    SENTIMENT_REPORTS_DIR / "tfidf_logistic_regression_confusion_matrix.csv"
)
EXPERIMENT_TRACKING_PATH = REPORTS_DIR / "experiment_tracking.csv"


def create_project_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        SENTIMENT_ARTIFACTS_DIR,
        SENTIMENT_REPORTS_DIR,
        SENTIMENT_PLOT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
