"""Reusable TF-IDF + K-Means topic discovery utilities."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def build_topic_model(*, n_clusters: int, max_features: int = 20000, min_df: int = 3, max_df: float = 0.95, random_state: int = 42):
    vectorizer = TfidfVectorizer(
        max_features=max_features, min_df=min_df, max_df=max_df,
        ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
    )
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return vectorizer, model


def fit_topic_model(texts, *, n_clusters: int, random_state: int = 42):
    values = pd.Series(texts, dtype="string").fillna("").astype(str)
    vectorizer, model = build_topic_model(n_clusters=n_clusters, random_state=random_state)
    matrix = vectorizer.fit_transform(values)
    labels = model.fit_predict(matrix)
    return {"vectorizer": vectorizer, "model": model, "matrix": matrix, "labels": labels}


def evaluate_cluster_counts(texts, cluster_counts=range(2, 11), *, sample_size: int = 5000, random_state: int = 42) -> pd.DataFrame:
    values = pd.Series(texts, dtype="string").fillna("").astype(str)
    vectorizer = TfidfVectorizer(max_features=15000, min_df=3, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    matrix = vectorizer.fit_transform(values)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(matrix.shape[0], size=min(sample_size, matrix.shape[0]), replace=False)
    rows = []
    for k in cluster_counts:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
        score = silhouette_score(matrix[sample_idx], labels[sample_idx])
        rows.append({"n_clusters": k, "inertia": model.inertia_, "silhouette_score": score})
    return pd.DataFrame(rows)


def top_terms(run, *, top_n: int = 12) -> pd.DataFrame:
    terms = run["vectorizer"].get_feature_names_out()
    rows = []
    for cluster_id, center in enumerate(run["model"].cluster_centers_):
        for rank, index in enumerate(center.argsort()[-top_n:][::-1], start=1):
            rows.append({"cluster_id": cluster_id, "rank": rank, "term": terms[index], "weight": center[index]})
    return pd.DataFrame(rows)


def representative_reviews(texts, run, *, reviews_per_cluster: int = 5) -> pd.DataFrame:
    values = pd.Series(texts, dtype="string").fillna("").astype(str).reset_index(drop=True)
    matrix, labels, centers = run["matrix"], run["labels"], run["model"].cluster_centers_
    rows = []
    for cluster_id in range(run["model"].n_clusters):
        indices = np.where(labels == cluster_id)[0]
        cluster_matrix = matrix[indices]
        center = centers[cluster_id]
        center_norm = np.linalg.norm(center) or 1.0
        row_norms = np.sqrt(cluster_matrix.multiply(cluster_matrix).sum(axis=1)).A1
        similarities = np.asarray(cluster_matrix @ center).ravel() / (row_norms * center_norm + 1e-12)
        for rank, local_index in enumerate(np.argsort(similarities)[::-1][:reviews_per_cluster], start=1):
            source_index = indices[local_index]
            rows.append({"cluster_id": cluster_id, "rank": rank, "text": values.iloc[source_index], "similarity_to_center": similarities[local_index]})
    return pd.DataFrame(rows)


def add_topic_labels(reviews: pd.DataFrame, run, topic_names: dict[int, str] | None = None) -> pd.DataFrame:
    result = reviews.reset_index(drop=True).copy()
    result["topic_cluster"] = run["labels"]
    if topic_names:
        result["topic_name"] = result["topic_cluster"].map(topic_names).fillna("unlabeled")
    return result


def topic_sentiment_summary(reviews: pd.DataFrame, *, topic_column: str = "topic_name", sentiment_column: str = "predicted_sentiment") -> pd.DataFrame:
    counts = pd.crosstab(reviews[topic_column], reviews[sentiment_column])
    percentages = counts.div(counts.sum(axis=1), axis=0).mul(100).round(2).add_suffix("_pct")
    return counts.join(percentages).reset_index()


def create_svd_coordinates(run, *, random_state: int = 42) -> pd.DataFrame:
    coordinates = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(run["matrix"])
    return pd.DataFrame({"component_1": coordinates[:, 0], "component_2": coordinates[:, 1], "cluster_id": run["labels"]})


def save_topic_model(run, output_path: str | Path) -> Path:
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": run["vectorizer"], "model": run["model"]}, path)
    return path
