"""Central project configuration and artifact paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLAB_PROJECT_ROOT = Path("/content/voxforge-ai-review-analytics")

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
ENRICHED_REVIEWS_PATH = PROCESSED_DATA_DIR / "reviews_with_sentiment.csv"
CLUSTERED_REVIEWS_PATH = PROCESSED_DATA_DIR / "reviews_with_sentiment_and_topics.csv"

MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODELS_DIR = MODELS_DIR / "sentiment"
CLUSTERING_MODELS_DIR = MODELS_DIR / "clustering"
MODEL_PATH = SENTIMENT_MODELS_DIR / "tfidf_logistic_regression.joblib"

RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
ABLATIONS_DIR = RESULTS_DIR / "ablations"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
MODEL_TRACKING_PATH = RESULTS_DIR / "model_tracking.csv"
MODEL_BENCHMARK_PATH = BENCHMARKS_DIR / "sentiment_model_benchmark.csv"
MODEL_COMPARISON_REPORT_PATH = RESULTS_DIR / "sentiment_model_comparison.md"
MODEL_SELECTION_PATH = RESULTS_DIR / "selected_sentiment_model.json"
INFERENCE_SPEED_PATH = BENCHMARKS_DIR / "inference_speed.csv"

# Backward-compatible notebook constants.
CLASS_DISTRIBUTION_PATH = PLOTS_DIR / "sentiment_class_distribution.png"
TOP_FEATURES_PATH = PLOTS_DIR / "tfidf_top_features.png"
PREDICTIONS_PATH = PREDICTIONS_DIR / "tfidf_logistic_validation.csv"
CONFUSION_MATRIX_PATH = CONFUSION_MATRICES_DIR / "tfidf_logistic_regression.png"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

RANDOM_STATE = 42
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def create_project_directories() -> None:
    """Create directories used by training, evaluation and clustering."""
    directories = (
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        SENTIMENT_MODELS_DIR, CLUSTERING_MODELS_DIR,
        RESULTS_DIR, CONFUSION_MATRICES_DIR, PLOTS_DIR, PREDICTIONS_DIR,
        CLASSIFICATION_REPORTS_DIR, ERROR_ANALYSIS_DIR, BENCHMARKS_DIR,
        ABLATIONS_DIR, CLUSTERING_RESULTS_DIR, REPORTS_DIR,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
