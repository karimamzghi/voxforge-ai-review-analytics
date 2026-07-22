"""Reusable classification evaluation utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def print_classification_metrics(metrics: dict[str, float]) -> None:
    print("Classification metrics")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")


def create_classification_report(
    y_true,
    y_pred,
    *,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def create_confusion_matrix_dataframe(
    y_true,
    y_pred,
    *,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in matrix_labels],
        columns=[f"predicted_{label}" for label in matrix_labels],
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    display.plot(values_format="d", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def measure_inference_time(
    model,
    X,
    *,
    number_of_runs: int = 5,
) -> dict[str, float]:
    if number_of_runs < 1:
        raise ValueError("number_of_runs must be at least 1.")
    if len(X) == 0:
        raise ValueError("X must contain at least one example.")

    run_times: list[float] = []
    for _ in range(number_of_runs):
        start = time.perf_counter()
        model.predict(X)
        run_times.append(time.perf_counter() - start)

    average_total = sum(run_times) / number_of_runs
    return {
        "total_inference_seconds": float(average_total),
        "average_inference_ms": float((average_total / len(X)) * 1000),
    }


def get_top_features_per_class(
    pipeline,
    *,
    top_n: int = 20,
    vectorizer_step: str = "tfidf",
    classifier_step: str = "classifier",
) -> dict[str, pd.DataFrame]:
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if vectorizer_step not in pipeline.named_steps:
        raise KeyError(f"Pipeline does not contain '{vectorizer_step}'.")
    if classifier_step not in pipeline.named_steps:
        raise KeyError(f"Pipeline does not contain '{classifier_step}'.")

    vectorizer = pipeline.named_steps[vectorizer_step]
    classifier = pipeline.named_steps[classifier_step]
    if not hasattr(vectorizer, "get_feature_names_out"):
        raise TypeError("Vectorizer must expose get_feature_names_out().")
    if not hasattr(classifier, "coef_") or not hasattr(classifier, "classes_"):
        raise TypeError("Classifier must expose coef_ and classes_ attributes.")

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_
    classes = classifier.classes_

    # Binary classifiers expose one coefficient row; create the opposite row.
    if len(classes) == 2 and coefficients.shape[0] == 1:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])

    result: dict[str, pd.DataFrame] = {}
    for class_index, class_name in enumerate(classes):
        class_coefficients = coefficients[class_index]
        indices = np.argsort(class_coefficients)[-top_n:][::-1]
        result[str(class_name)] = pd.DataFrame({
            "feature": feature_names[indices],
            "coefficient": class_coefficients[indices],
        })
    return result


def combine_top_features(top_features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-class feature tables into one saveable DataFrame."""
    frames = []
    for class_name, frame in top_features.items():
        class_frame = frame.copy()
        class_frame.insert(0, "sentiment_class", class_name)
        class_frame.insert(1, "rank", range(1, len(class_frame) + 1))
        frames.append(class_frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def display_top_features_per_class(top_features: dict[str, pd.DataFrame]) -> None:
    for class_name, frame in top_features.items():
        print(f"\nTop features for: {class_name}")
        print("-" * 40)
        print(frame.to_string(index=False))


def evaluate_model(
    model,
    X,
    y_true,
    *,
    model_name: str = "Model",
    labels: list[str] | None = None,
    confusion_matrix_path: str | Path | None = None,
    show_confusion_matrix: bool = True,
) -> dict[str, Any]:
    predictions = model.predict(X)
    metrics = calculate_classification_metrics(y_true, predictions)
    report = create_classification_report(y_true, predictions, labels=labels)
    matrix_df = create_confusion_matrix_dataframe(y_true, predictions, labels=labels)
    inference = measure_inference_time(model, X)

    print(f"Evaluating: {model_name}")
    print("=" * 50)
    print_classification_metrics(metrics)
    print(f"Average inference per review: {inference['average_inference_ms']:.4f} ms")

    plot_confusion_matrix(
        y_true,
        predictions,
        labels=labels,
        title=f"{model_name} Confusion Matrix",
        save_path=confusion_matrix_path,
        show=show_confusion_matrix,
    )

    return {
        "model_name": model_name,
        "predictions": predictions,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix_df,
        "inference": inference,
    }


def create_model_result_row(
    *,
    model_name: str,
    metrics: dict[str, float],
    training_time_seconds: float | None = None,
    inference_time_ms: float | None = None,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "model_name": model_name,
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "training_time_seconds": training_time_seconds,
        "inference_time_ms": inference_time_ms,
    }])


def compare_model_results(
    results: list[pd.DataFrame],
    *,
    sort_by: str = "macro_f1",
) -> pd.DataFrame:
    if not results:
        raise ValueError("At least one result DataFrame is required.")
    comparison = pd.concat(results, ignore_index=True)
    if sort_by not in comparison.columns:
        raise KeyError(f"Column '{sort_by}' not found in model results.")
    return comparison.sort_values(sort_by, ascending=False).reset_index(drop=True)
