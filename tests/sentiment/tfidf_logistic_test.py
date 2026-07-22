"""Tests for the classical TF-IDF + Logistic Regression baseline."""
from sklearn.pipeline import Pipeline

from src.sentiment.tfidf_logistic import build_tfidf_logistic_pipeline


def test_pipeline_has_tfidf_and_classifier_steps() -> None:
    pipeline = build_tfidf_logistic_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["tfidf", "classifier"]


def test_pipeline_fits_and_predicts() -> None:
    texts = [
        "great product love it", "excellent quality very happy",
        "terrible waste of money", "broke immediately awful",
        "it is fine nothing special", "average okay experience",
    ]
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
    pipeline = build_tfidf_logistic_pipeline(min_df=1)
    pipeline.fit(texts, labels)
    predictions = pipeline.predict(texts)
    assert len(predictions) == len(texts)
    assert set(predictions).issubset({"positive", "negative", "neutral"})
