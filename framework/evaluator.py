"""
Evaluator
==========
Comprehensive evaluation module with metrics computation,
confusion matrix analysis, and structured result output.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics:
        - Precision, Recall, F1-score
        - ROC-AUC
        - Confusion matrix (TP, FP, FN, TN)
        - False positive rate, False negative rate
        - Accuracy
        - Specificity (TNR)

    Parameters
    ----------
    y_true : np.ndarray of int (0 or 1)
    y_scores : np.ndarray of float (predicted probabilities)
    threshold : float

    Returns
    -------
    dict with all metrics
    """
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        accuracy_score,
        confusion_matrix,
    )

    y_pred = (y_scores >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Rates
    total = len(y_true)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auc = 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_samples": total,
        "threshold": threshold,
    }

    return metrics


def evaluate_model(
    model,
    loader,
    device,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """
    Run inference on a DataLoader and compute all metrics.

    Works with both CNN (torch) models.
    """
    import torch
    from torch.cuda.amp import autocast

    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with autocast():
                    logits, _ = model(images)
            else:
                logits, _ = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_scores)

    return compute_all_metrics(y_true, y_scores, threshold=threshold)


def evaluate_classical_model(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a classical ML model."""
    y_scores = model.predict_proba(X)[:, 1]
    return compute_all_metrics(y_true, y_scores, threshold=threshold)


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Save metrics dict as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


def print_metrics(metrics: Dict[str, Any], title: str = "Evaluation Results") -> None:
    """Pretty-print metrics to console."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1 Score      : {metrics['f1']:.4f}")
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
    print(f"  Specificity   : {metrics['specificity']:.4f}")
    print(f"  FP Rate       : {metrics['false_positive_rate']:.4f}")
    print(f"  FN Rate       : {metrics['false_negative_rate']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"  Total samples : {metrics['total_samples']}")
    print(f"  Threshold     : {metrics['threshold']}")
    print(f"{'=' * 60}\n")


def compare_experiments(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """
    Compare metrics across multiple experiments.

    Parameters
    ----------
    results : dict mapping experiment_name -> metrics dict
    output_path : str, optional
        If given, save comparison table to file.

    Returns
    -------
    str : formatted comparison table
    """
    names = list(results.keys())
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "fp", "fn"]
    header = f"{'Experiment':<30}" + "".join(f"{c:<12}" for c in cols)
    lines = [header, "-" * len(header)]

    for name in names:
        m = results[name]
        row = f"{name:<30}"
        for c in cols:
            val = m.get(c, "N/A")
            if isinstance(val, float):
                row += f"{val:<12.4f}"
            else:
                row += f"{str(val):<12}"
        lines.append(row)

    table = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table)

    return table
