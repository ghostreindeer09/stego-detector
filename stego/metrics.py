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
    
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    fpr_val = fp / max(1, fp + tn)
    fnr_val = fn / max(1, tp + fn)
    pe = (fpr_val + fnr_val) / 2.0
    
    try:
        from sklearn.metrics import roc_curve
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_score)
        fnr_curve = 1 - tpr_curve
        eer_idx = np.nanargmin(np.absolute((fnr_curve - fpr_curve)))
        eer = float(fpr_curve[eer_idx])
        
        idx_1 = np.where(fpr_curve <= 0.01)[0]
        tpr_at_1 = float(tpr_curve[idx_1[-1]]) if len(idx_1) > 0 else 0.0
        
        idx_5 = np.where(fpr_curve <= 0.05)[0]
        tpr_at_5 = float(tpr_curve[idx_5[-1]]) if len(idx_5) > 0 else 0.0
    except ImportError:
        eer, tpr_at_1, tpr_at_5 = 0.0, 0.0, 0.0

    return {
        "weighted_auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "pe": pe,
        "eer": eer,
        "tpr_at_fpr_0.01": tpr_at_1,
        "tpr_at_fpr_0.05": tpr_at_5,
    }
