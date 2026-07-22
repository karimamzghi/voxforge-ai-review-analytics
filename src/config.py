"""Central project configuration and artifact paths."""

from pathlib import Path

# =============================================================================
# Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Optional helper for Colab notebooks
COLAB_PROJECT_ROOT = Path("/content/voxforge-ai-review-analytics")

# =============================================================================
# Data
# =============================================================================

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
CLUSTERED_REVIEWS_PATH = (
    PROCESSED_DATA_DIR / "reviews_with_sentiment_and_topics.csv"
)

# =============================================================================
# Models
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"

SENTIMENT_MODELS_DIR = MODELS_DIR / "sentiment"
CLUSTERING_MODELS_DIR = MODELS_DIR / "clustering"

# Classical model
TFIDF_LOGISTIC_MODEL_PATH = (
    SENTIMENT_MODELS_DIR / "tfidf_logistic_regression.joblib"
)

# Transformer models
DISTILBERT_MODEL_DIR = (
    SENTIMENT_MODELS_DIR / "distilbert_sentiment"
)

# Advanced transformer models
DEBERTA_MODEL_DIR = (
    SENTIMENT_MODELS_DIR / "deberta_v3_sentiment"
)

# Clutering
TFIDF_SVD_KMEANS_MODEL_PATH = (
    CLUSTERING_MODELS_DIR / "tfidf_svd_kmeans.joblib"
)

# =============================================================================
# Results
# =============================================================================

RESULTS_DIR = PROJECT_ROOT / "results"

PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
CLASSIFICATION_REPORTS_DIR = RESULTS_DIR / "classification_reports"

ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
ABLATIONS_DIR = RESULTS_DIR / "ablations"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"

# =============================================================================
# TF-IDF outputs
# =============================================================================

TFIDF_LOGISTIC_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "tfidf_logistic_validation.csv"
)

TFIDF_LOGISTIC_CONFUSION_MATRIX_PATH = (
    CONFUSION_MATRICES_DIR / "tfidf_logistic_regression.png"
)

TFIDF_LOGISTIC_CONFUSION_MATRIX_CSV_PATH = (
    CONFUSION_MATRICES_DIR / "tfidf_logistic_regression.csv"
)

TFIDF_LOGISTIC_CLASSIFICATION_REPORT_PATH = (
    CLASSIFICATION_REPORTS_DIR / "tfidf_logistic_regression.csv"
)

# =============================================================================
# DistilBERT outputs
# =============================================================================

DISTILBERT_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "distilbert_validation_predictions.csv"
)

DISTILBERT_CONFUSION_MATRIX_PATH = (
    CONFUSION_MATRICES_DIR / "distilbert_sentiment.png"
)

DISTILBERT_CONFUSION_MATRIX_CSV_PATH = (
    CONFUSION_MATRICES_DIR / "distilbert_sentiment.csv"
)

DISTILBERT_CLASSIFICATION_REPORT_PATH = (
    CLASSIFICATION_REPORTS_DIR / "distilbert_sentiment.csv"
)

DISTILBERT_TRAINING_HISTORY_PATH = (
    PLOTS_DIR / "distilbert_training_history.png"
)

# =============================================================================
# DeBERTa outputs
# =============================================================================

DEBERTA_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "deberta_v3_sentiment_validation.csv"
)

DEBERTA_CONFUSION_MATRIX_PATH = (
    CONFUSION_MATRICES_DIR / "deberta_v3_sentiment.png"
)

DEBERTA_CONFUSION_MATRIX_CSV_PATH = (
    CONFUSION_MATRICES_DIR / "deberta_v3_sentiment.csv"
)

DEBERTA_CLASSIFICATION_REPORT_PATH = (
    CLASSIFICATION_REPORTS_DIR / "deberta_v3_sentiment.csv"
)

DEBERTA_TRAINING_HISTORY_PATH = (
    PLOTS_DIR / "deberta_v3_training_history.png"
)

# =============================================================================
# Project Reports
# =============================================================================

MODEL_TRACKING_PATH = RESULTS_DIR / "model_tracking.csv"

MODEL_BENCHMARK_PATH = (
    BENCHMARKS_DIR / "sentiment_model_benchmark.csv"
)

MODEL_COMPARISON_REPORT_PATH = (
    RESULTS_DIR / "sentiment_model_comparison.md"
)

MODEL_SELECTION_PATH = (
    RESULTS_DIR / "selected_sentiment_model.json"
)

INFERENCE_SPEED_PATH = (
    BENCHMARKS_DIR / "inference_speed.csv"
)

# =============================================================================
# Legacy notebook compatibility
# =============================================================================

CLASS_DISTRIBUTION_PATH = (
    PLOTS_DIR / "sentiment_class_distribution.png"
)

TOP_FEATURES_PATH = (
    PLOTS_DIR / "tfidf_top_features.png"
)

PREDICTIONS_PATH = TFIDF_LOGISTIC_PREDICTIONS_PATH

CONFUSION_MATRIX_PATH = (
    TFIDF_LOGISTIC_CONFUSION_MATRIX_PATH
)

# =============================================================================
# Reports
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# =============================================================================
# General
# =============================================================================

SENTIMENT_LABELS = [
    "negative",
    "neutral",
    "positive",
]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(SENTIMENT_LABELS)
}

ID_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_ID.items()
}

# =============================================================================
# Directory creation
# =============================================================================


def create_project_directories() -> None:
    """Create all project directories."""

    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        SENTIMENT_MODELS_DIR,
        CLUSTERING_MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        PREDICTIONS_DIR,
        CONFUSION_MATRICES_DIR,
        CLASSIFICATION_REPORTS_DIR,
        ERROR_ANALYSIS_DIR,
        BENCHMARKS_DIR,
        ABLATIONS_DIR,
        CLUSTERING_RESULTS_DIR,
        REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


create_project_directories()

# ------------------------------------------------------------------
# Topic clustering configuration
# ------------------------------------------------------------------
RANDOM_STATE = 42

CLUSTER_TEXT_COLUMN = "classical_text"
DEFAULT_CLUSTER_COUNTS = tuple(range(2, 11))
DEFAULT_N_CLUSTERS = 6
DEFAULT_TOP_N_TERMS = 12
DEFAULT_REVIEWS_PER_CLUSTER = 5

TFIDF_MAX_FEATURES = 10_000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.85
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = True

SVD_COMPONENTS = 100
KMEANS_N_INIT = 20
SILHOUETTE_SAMPLE_SIZE = 10_000

DOMAIN_STOP_WORDS = frozenset(
    {
        "star",
        "stars",
        "rating",
        "ratings",
        "rated",
        "five",
        "four",
        "three",
        "two",
        "one",
    }
)
