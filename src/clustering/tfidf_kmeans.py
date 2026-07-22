"""Reusable TF-IDF + LSA + K-Means topic-clustering utilities.

Pipeline
--------
review text
    -> TF-IDF
    -> TruncatedSVD (Latent Semantic Analysis)
    -> L2 normalization
    -> K-Means

The public functions preserve the API used by the clustering notebook:
- evaluate_cluster_counts
- fit_topic_model
- top_terms
- representative_reviews
- add_topic_labels
- topic_sentiment_summary
- create_svd_coordinates
- save_topic_model
- load_topic_model
- predict_topics
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer

from src.config import (
    DEFAULT_CLUSTER_COUNTS,
    DEFAULT_N_CLUSTERS,
    DEFAULT_REVIEWS_PER_CLUSTER,
    DEFAULT_TOP_N_TERMS,
    DOMAIN_STOP_WORDS,
    KMEANS_N_INIT,
    RANDOM_STATE,
    SILHOUETTE_SAMPLE_SIZE,
    SVD_COMPONENTS,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)

from src.config import (
    DEFAULT_CLUSTER_COUNTS,
    DEFAULT_N_CLUSTERS,
    RANDOM_STATE,
    SVD_COMPONENTS,
)

RANDOM_STATE = 42

def _prepare_texts(texts: Sequence[str] | pd.Series) -> pd.Series:
    """Return a clean, index-reset text series suitable for vectorization."""
    prepared = pd.Series(texts).fillna("").astype(str).str.strip().reset_index(drop=True)

    if prepared.empty:
        raise ValueError("No review texts were provided.")

    if not prepared.str.len().gt(0).any():
        raise ValueError("All review texts are empty after cleaning.")

    return prepared


def build_clustering_stop_words(
    extra_stop_words: Iterable[str] | None = None,
) -> list[str]:
    """Combine scikit-learn English stop words with domain-specific terms."""
    words = set(ENGLISH_STOP_WORDS)
    words.update(DOMAIN_STOP_WORDS)

    if extra_stop_words is not None:
        words.update(
            str(word).strip().lower()
            for word in extra_stop_words
            if str(word).strip()
        )

    return sorted(words)


def create_tfidf_vectorizer(
    *,
    max_features: int = TFIDF_MAX_FEATURES,
    min_df: int | float = TFIDF_MIN_DF,
    max_df: int | float = TFIDF_MAX_DF,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    extra_stop_words: Iterable[str] | None = None,
) -> TfidfVectorizer:
    """Create the reusable TF-IDF vectorizer for topic clustering."""
    return TfidfVectorizer(
        stop_words=build_clustering_stop_words(extra_stop_words),
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        dtype=np.float32,
    )


def _validated_svd_components(
    matrix: sparse.spmatrix,
    requested_components: int,
) -> int:
    """Return a valid SVD dimension for the fitted TF-IDF matrix."""
    maximum = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    components = min(requested_components, maximum)

    if components < 2:
        raise ValueError(
            "TruncatedSVD requires at least two valid components. "
            f"Received TF-IDF shape {matrix.shape}."
        )

    return components


def _fit_text_representation(
    texts: Sequence[str] | pd.Series,
    *,
    svd_components: int = SVD_COMPONENTS,
    random_state: int = RANDOM_STATE,
    extra_stop_words: Iterable[str] | None = None,
) -> dict:
    """Fit TF-IDF, TruncatedSVD and L2 normalization once."""
    clean_texts = _prepare_texts(texts)

    vectorizer = create_tfidf_vectorizer(
        extra_stop_words=extra_stop_words,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)

    n_components = _validated_svd_components(
        tfidf_matrix,
        requested_components=svd_components,
    )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
    )
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    normalizer = Normalizer(norm="l2", copy=True)
    normalized_matrix = normalizer.fit_transform(reduced_matrix)

    return {
        "texts": clean_texts,
        "vectorizer": vectorizer,
        "svd": svd,
        "normalizer": normalizer,
        "tfidf_matrix": tfidf_matrix,
        "reduced_matrix": reduced_matrix,
        "normalized_matrix": normalized_matrix,
        "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
    }


def _safe_silhouette_score(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
    sample_size: int = SILHOUETTE_SAMPLE_SIZE,
) -> float:
    """Calculate silhouette safely, sampling large datasets for efficiency."""
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return float("nan")

    effective_sample_size = min(sample_size, len(labels))

    return float(
        silhouette_score(
            matrix,
            labels,
            metric="euclidean",
            sample_size=effective_sample_size,
            random_state=random_state,
        )
    )


def evaluate_cluster_counts(
    texts: Sequence[str] | pd.Series,
    *,
    cluster_counts: Iterable[int] = DEFAULT_CLUSTER_COUNTS,
    svd_components: int = SVD_COMPONENTS,
    random_state: int = RANDOM_STATE,
    n_init: int = KMEANS_N_INIT,
    silhouette_sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    extra_stop_words: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Evaluate candidate K values on one shared normalized LSA representation."""
    representation = _fit_text_representation(
        texts,
        svd_components=svd_components,
        random_state=random_state,
        extra_stop_words=extra_stop_words,
    )
    matrix = representation["normalized_matrix"]

    rows: list[dict] = []

    for n_clusters in cluster_counts:
        n_clusters = int(n_clusters)

        if n_clusters < 2:
            continue

        if n_clusters >= len(matrix):
            continue

        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )
        labels = model.fit_predict(matrix)

        rows.append(
            {
                "n_clusters": n_clusters,
                "inertia": float(model.inertia_),
                "silhouette_score": _safe_silhouette_score(
                    matrix,
                    labels,
                    random_state=random_state,
                    sample_size=silhouette_sample_size,
                ),
                "svd_components": int(representation["svd"].n_components),
                "svd_explained_variance_ratio": representation[
                    "explained_variance_ratio"
                ],
            }
        )

    if not rows:
        raise ValueError("No valid cluster counts were provided.")

    return pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)


def fit_topic_model(
    texts: Sequence[str] | pd.Series,
    *,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    svd_components: int = SVD_COMPONENTS,
    random_state: int = RANDOM_STATE,
    n_init: int = KMEANS_N_INIT,
    silhouette_sample_size: int = SILHOUETTE_SAMPLE_SIZE,
    extra_stop_words: Iterable[str] | None = None,
) -> dict:
    """Fit the final TF-IDF -> SVD -> normalization -> K-Means model."""
    representation = _fit_text_representation(
        texts,
        svd_components=svd_components,
        random_state=random_state,
        extra_stop_words=extra_stop_words,
    )

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if n_clusters >= len(representation["texts"]):
        raise ValueError(
            "n_clusters must be smaller than the number of review texts."
        )

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(representation["normalized_matrix"])

    return {
        **representation,
        "model": model,
        "labels": labels,
        "n_clusters": int(n_clusters),
        "silhouette_score": _safe_silhouette_score(
            representation["normalized_matrix"],
            labels,
            random_state=random_state,
            sample_size=silhouette_sample_size,
        ),
        "random_state": int(random_state),
    }


def top_terms(
    run: Mapping,
    *,
    top_n: int = DEFAULT_TOP_N_TERMS,
) -> pd.DataFrame:
    """Return terms with the highest mean original TF-IDF weight per cluster."""
    feature_names = np.asarray(
        run["vectorizer"].get_feature_names_out()
    )
    labels = np.asarray(run["labels"])
    tfidf_matrix = run["tfidf_matrix"]

    rows: list[dict] = []

    for cluster_id in sorted(np.unique(labels)):
        cluster_mask = labels == cluster_id
        cluster_matrix = tfidf_matrix[cluster_mask]
        mean_scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_indices = mean_scores.argsort()[::-1][:top_n]

        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "rank": int(rank),
                    "term": str(feature_names[feature_index]),
                    "mean_tfidf": float(mean_scores[feature_index]),
                }
            )

    return pd.DataFrame(rows)


def representative_reviews(
    texts: Sequence[str] | pd.Series,
    run: Mapping,
    *,
    reviews_per_cluster: int = DEFAULT_REVIEWS_PER_CLUSTER,
) -> pd.DataFrame:
    """Return the reviews closest to each K-Means centroid."""
    clean_texts = _prepare_texts(texts)
    labels = np.asarray(run["labels"])
    model = run["model"]
    matrix = run["normalized_matrix"]

    if len(clean_texts) != len(labels):
        raise ValueError(
            "The number of texts must match the number of fitted cluster labels."
        )

    distances = model.transform(matrix)
    rows: list[dict] = []

    for cluster_id in sorted(np.unique(labels)):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        cluster_distances = distances[cluster_indices, cluster_id]
        selected = cluster_indices[
            np.argsort(cluster_distances)[:reviews_per_cluster]
        ]

        for rank, review_index in enumerate(selected, start=1):
            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "rank": int(rank),
                    "review_index": int(review_index),
                    "review_text": clean_texts.iloc[review_index],
                    "distance_to_centroid": float(
                        distances[review_index, cluster_id]
                    ),
                }
            )

    return pd.DataFrame(rows)


def cluster_size_summary(run: Mapping) -> pd.DataFrame:
    """Create review counts and percentages for each cluster."""
    labels = pd.Series(run["labels"], name="cluster_id")

    summary = (
        labels.value_counts()
        .sort_index()
        .rename("review_count")
        .reset_index()
    )
    summary["percentage"] = (
        summary["review_count"] / summary["review_count"].sum() * 100
    ).round(2)

    return summary


def add_topic_labels(
    dataframe: pd.DataFrame,
    run: Mapping,
    topic_names: Mapping[int, str],
    *,
    cluster_column: str = "cluster_id",
    topic_column: str = "topic_name",
) -> pd.DataFrame:
    """Add K-Means cluster IDs and human-readable names to a dataframe."""
    output = dataframe.reset_index(drop=True).copy()
    labels = np.asarray(run["labels"])

    if len(output) != len(labels):
        raise ValueError(
            "The dataframe row count must match the number of fitted labels."
        )

    output[cluster_column] = labels.astype(int)
    output[topic_column] = output[cluster_column].map(topic_names)

    missing_ids = sorted(
        output.loc[output[topic_column].isna(), cluster_column].unique()
    )
    if missing_ids:
        raise ValueError(
            "Missing human-readable names for cluster IDs: "
            f"{missing_ids}"
        )

    return output


def topic_sentiment_summary(
    dataframe: pd.DataFrame,
    *,
    topic_column: str = "topic_name",
    sentiment_column: str = "sentiment_label",
) -> pd.DataFrame:
    """Summarize review counts and sentiment percentages by topic."""
    required = {topic_column, sentiment_column}
    missing = required.difference(dataframe.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    counts = (
        dataframe.groupby([topic_column, sentiment_column], dropna=False)
        .size()
        .rename("review_count")
        .reset_index()
    )
    totals = (
        counts.groupby(topic_column)["review_count"]
        .sum()
        .rename("topic_review_count")
        .reset_index()
    )
    summary = counts.merge(totals, on=topic_column, how="left")
    summary["sentiment_percentage"] = (
        summary["review_count"] / summary["topic_review_count"] * 100
    ).round(2)

    return summary.sort_values(
        [topic_column, "review_count"],
        ascending=[True, False],
    ).reset_index(drop=True)


def create_svd_coordinates(
    run: Mapping,
    *,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Project the final normalized clustering representation into two dimensions."""
    projection = TruncatedSVD(
        n_components=2,
        random_state=random_state,
    )
    coordinates = projection.fit_transform(
        run["normalized_matrix"]
    )

    return pd.DataFrame(
        {
            "component_1": coordinates[:, 0],
            "component_2": coordinates[:, 1],
            "cluster_id": np.asarray(run["labels"]).astype(int),
        }
    )


def save_topic_model(
    run: Mapping,
    path: str | Path,
    *,
    topic_names: Mapping[int, str] | None = None,
) -> Path:
    """Save only fitted components needed for future topic prediction."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "vectorizer": run["vectorizer"],
        "svd": run["svd"],
        "normalizer": run["normalizer"],
        "model": run["model"],
        "topic_names": dict(topic_names) if topic_names is not None else None,
        "metadata": {
            "n_clusters": int(run["n_clusters"]),
            "svd_components": int(run["svd"].n_components),
            "svd_explained_variance_ratio": float(
                run["explained_variance_ratio"]
            ),
            "silhouette_score": float(run["silhouette_score"]),
            "random_state": int(run["random_state"]),
        },
    }

    joblib.dump(artifact, output_path)
    return output_path


def load_topic_model(path: str | Path) -> dict:
    """Load a saved topic-model artifact."""
    artifact = joblib.load(Path(path))

    required = {"vectorizer", "svd", "normalizer", "model"}
    missing = required.difference(artifact)

    if missing:
        raise ValueError(
            f"Invalid topic-model artifact; missing keys: {sorted(missing)}"
        )

    return artifact


def predict_topics(
    texts: Sequence[str] | pd.Series,
    artifact: Mapping,
) -> pd.DataFrame:
    """Predict cluster IDs and optional topic names for new review texts."""
    clean_texts = _prepare_texts(texts)

    tfidf_matrix = artifact["vectorizer"].transform(clean_texts)
    reduced_matrix = artifact["svd"].transform(tfidf_matrix)
    normalized_matrix = artifact["normalizer"].transform(reduced_matrix)
    cluster_ids = artifact["model"].predict(normalized_matrix).astype(int)

    output = pd.DataFrame(
        {
            "review_text": clean_texts,
            "cluster_id": cluster_ids,
        }
    )

    topic_names = artifact.get("topic_names")
    if topic_names:
        output["topic_name"] = output["cluster_id"].map(topic_names)

    return output
