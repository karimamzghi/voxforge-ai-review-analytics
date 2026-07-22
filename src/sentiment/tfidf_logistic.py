"""TF-IDF + Logistic Regression model construction."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def build_tfidf_logistic_pipeline(
    *,
    max_features: int = 30000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build the classical sentiment baseline as one sklearn Pipeline."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight=class_weight,
                max_iter=max_iter,
                solver="lbfgs",
                random_state=random_state,
            ),
        ),
    ])
