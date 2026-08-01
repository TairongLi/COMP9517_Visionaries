"""Shared metrics and result saving for all project methods."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, y_score, train_s=0.0, test_s=0.0):
    """Compute the required classification and timing metrics."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1 or y_pred.shape != y_true.shape:
        raise ValueError("y_true and y_pred must have shape (N,)")
    if y_score.ndim != 2 or y_score.shape[0] != len(y_true):
        raise ValueError("y_score must have shape (N, C)")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    k = min(5, y_score.shape[1])
    top5 = np.argpartition(y_score, -k, axis=1)[:, -k:]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(y_score.shape[1]),
        average="macro",
        zero_division=0,
    )

    return {
        "top1_accuracy": float(np.mean(y_true == y_pred)),
        "top5_accuracy": float(np.mean(np.any(top5 == y_true[:, None], axis=1))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "train_s": float(train_s),
        "test_s": float(test_s),
        "ms_per_image": float(test_s * 1000.0 / len(y_true)),
        "num_samples": int(len(y_true)),
    }


def save_result(
    results_dir,
    method_name,
    y_true,
    y_pred,
    y_score,
    class_names,
    timing=None,
    filepaths=None,
):
    """Save aligned predictions and return the output path and metrics."""

    timing = timing or {}
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    class_names = np.asarray(class_names, dtype=str)

    if y_score.shape != (len(y_true), len(class_names)):
        raise ValueError(
            f"Expected score shape {(len(y_true), len(class_names))}, "
            f"got {y_score.shape}"
        )

    scores = compute_metrics(
        y_true,
        y_pred,
        y_score,
        train_s=timing.get("train_s", 0.0),
        test_s=timing.get("test_s", 0.0),
    )

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{method_name}.npz"

    values = {
        "method_name": np.asarray(method_name),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "class_names": class_names,
        **{name: np.asarray(value) for name, value in scores.items()},
    }
    if filepaths is not None:
        values["filepaths"] = np.asarray(filepaths, dtype=str)

    np.savez_compressed(path, **values)
    return path, scores

