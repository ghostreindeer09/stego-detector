"""
Evaluation metrics: Weighted AUC, F1-Score, and confusion matrix utilities.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def weighted_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1,
) -> float:
    """
    Weighted AUC (equivalent to sklearn's roc_auc_score with average='weighted'
    for binary: same as standard AUC).
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError("Install scikit-learn for AUC: pip install scikit-learn")

    if np.unique(y_true).size <= 1:
        return 0.0
    return float(roc_auc_score(y_true, y_score, average="weighted"))


def f1_score_from_cm(tp: int, fp: int, fn: int, tn: int) -> Tuple[float, float, float]:
    """Precision, Recall, F1 (binary)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def confusion_matrix_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Returns (TP, FP, FN, TN)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tp, fp, fn, tn


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Returns dict with weighted_auc, f1, precision, recall, tp, fp, fn, tn."""
    y_pred = (y_score >= threshold).astype(np.int64)
    tp, fp, fn, tn = confusion_matrix_binary(y_true, y_pred)
    precision, recall, f1 = f1_score_from_cm(tp, fp, fn, tn)
    auc = weighted_auc(y_true, y_score)
    return {
        "weighted_auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
