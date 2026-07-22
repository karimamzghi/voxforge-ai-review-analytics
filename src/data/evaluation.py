"""
Evaluation utilities for sentiment classification models.

This module contains reusable methods for evaluating classical machine
learning and transformer-based sentiment models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate the main classification metrics.

    Parameters
    ----------
    y_true:
        Correct sentiment labels.

    y_pred:
        Predicted sentiment labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall and F1 scores.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics


def print_classification_metrics(
    metrics: dict[str, float],
) -> None:
    """
    Print classification metrics in a readable format.
    """

    print("Classification metrics")
    print("-" * 30)

    for metric_name, metric_value in metrics.items():
        readable_name = metric_name.replace("_", " ").title()
        print(f"{readable_name}: {metric_value:.4f}")


def create_classification_report(
    y_true,
    y_pred,
) -> pd.DataFrame:
    """
    Create a classification report as a pandas DataFrame.

    Returns precision, recall, F1-score and support for each class.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    return report_df


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true:
        Correct labels.

    y_pred:
        Predicted labels.

    labels:
        Ordered list of class labels.

    title:
        Plot title.

    save_path:
        Optional path used to save the figure.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot(
        values_format="d",
        cmap="Blues",
    )

    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            bbox_inches="tight",
        )

    plt.show()


def measure_inference_time(
    model,
    X,
    number_of_runs: int = 5,
) -> dict[str, float]:
    """
    Measure average model inference time.

    The prediction is repeated several times to make the result more stable.

    Parameters
    ----------
    model:
        Trained model or sklearn Pipeline.

    X:
        Input examples.

    number_of_runs:
        Number of repeated prediction runs.

    Returns
    -------
    dict
        Total and average inference times.
    """

    if number_of_runs < 1:
        raise ValueError("number_of_runs must be at least 1.")

    run_times = []

    for _ in range(number_of_runs):
        start_time = time.perf_counter()

        model.predict(X)

        end_time = time.perf_counter()
        run_times.append(end_time - start_time)

    average_total_seconds = sum(run_times) / number_of_runs
    number_of_examples = len(X)

    if number_of_examples == 0:
        raise ValueError("X must contain at least one example.")

    average_per_example_seconds = (
        average_total_seconds / number_of_examples
    )

    return {
        "total_inference_seconds": average_total_seconds,
        "average_inference_ms": average_per_example_seconds * 1000,
    }


def get_top_features_per_class(
    pipeline,
    top_n: int = 20,
    vectorizer_step: str = "tfidf",
    classifier_step: str = "classifier",
) -> dict[str, pd.DataFrame]:
    """
    Get the most important TF-IDF features for each Logistic Regression class.

    This method expects a trained sklearn Pipeline containing:

    - a TfidfVectorizer
    - a LogisticRegression classifier

    Parameters
    ----------
    pipeline:
        Trained sklearn Pipeline.

    top_n:
        Number of features to return for each class.

    vectorizer_step:
        Name of the TF-IDF step inside the pipeline.

    classifier_step:
        Name of the classifier step inside the pipeline.

    Returns
    -------
    dict
        One DataFrame per sentiment class.
    """

    if vectorizer_step not in pipeline.named_steps:
        raise KeyError(
            f"Pipeline does not contain a '{vectorizer_step}' step."
        )

    if classifier_step not in pipeline.named_steps:
        raise KeyError(
            f"Pipeline does not contain a '{classifier_step}' step."
        )

    vectorizer = pipeline.named_steps[vectorizer_step]
    classifier = pipeline.named_steps[classifier_step]

    if not hasattr(vectorizer, "get_feature_names_out"):
        raise TypeError(
            "The vectorizer must support get_feature_names_out()."
        )

    if not hasattr(classifier, "coef_"):
        raise TypeError(
            "The classifier must expose coefficients through coef_."
        )

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_
    class_names = classifier.classes_

    top_features = {}

    for class_index, class_name in enumerate(class_names):
        class_coefficients = coefficients[class_index]

        top_indices = np.argsort(class_coefficients)[-top_n:][::-1]

        class_features = pd.DataFrame(
            {
                "feature": feature_names[top_indices],
                "coefficient": class_coefficients[top_indices],
            }
        )

        top_features[str(class_name)] = class_features

    return top_features


def display_top_features_per_class(
    top_features: dict[str, pd.DataFrame],
) -> None:
    """
    Print the top features for each class.
    """

    for class_name, features_df in top_features.items():
        print(f"\nTop features for: {class_name}")
        print("-" * 40)
        print(features_df.to_string(index=False))


def evaluate_model(
    model,
    X,
    y_true,
    model_name: str = "Model",
    labels: list[str] | None = None,
    confusion_matrix_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run the complete evaluation process for a trained model.

    This method:

    1. Generates predictions
    2. Calculates classification metrics
    3. Creates the classification report
    4. Measures inference time
    5. Plots the confusion matrix

    Returns
    -------
    dict
        Predictions, metrics, report and inference information.
    """

    print(f"Evaluating: {model_name}")
    print("=" * 50)

    predictions = model.predict(X)

    metrics = calculate_classification_metrics(
        y_true,
        predictions,
    )

    report_df = create_classification_report(
        y_true,
        predictions,
    )

    inference_metrics = measure_inference_time(
        model,
        X,
    )

    print_classification_metrics(metrics)

    print("\nInference performance")
    print("-" * 30)
    print(
        "Average inference time per review: "
        f"{inference_metrics['average_inference_ms']:.4f} ms"
    )

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=predictions,
        labels=labels,
        title=f"{model_name} Confusion Matrix",
        save_path=confusion_matrix_path,
    )

    return {
        "model_name": model_name,
        "predictions": predictions,
        "metrics": metrics,
        "classification_report": report_df,
        "inference": inference_metrics,
    }


def create_model_result_row(
    model_name: str,
    metrics: dict[str, float],
    training_time_seconds: float | None = None,
    inference_time_ms: float | None = None,
) -> pd.DataFrame:
    """
    Create one model-results row for an experiment comparison table.
    """

    result = {
        "model_name": model_name,
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "training_time_seconds": training_time_seconds,
        "inference_time_ms": inference_time_ms,
    }

    return pd.DataFrame([result])


def compare_model_results(
    results: list[pd.DataFrame],
    sort_by: str = "macro_f1",
) -> pd.DataFrame:
    """
    Combine several model result rows into one comparison table.
    """

    if not results:
        raise ValueError("At least one result DataFrame is required.")

    comparison_df = pd.concat(
        results,
        ignore_index=True,
    )

    if sort_by in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            by=sort_by,
            ascending=False,
        )

    return comparison_df.reset_index(drop=True)