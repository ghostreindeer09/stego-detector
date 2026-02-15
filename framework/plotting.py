"""
Plotting Module
================
Publication-quality plots for steganalysis research.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_roc_curve(y_true, y_scores, save_path="roc_curve.png", title="ROC Curve", label="Model"):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"{label} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.savefig(save_path); plt.close()
    logger.info("ROC curve saved to %s", save_path)


def plot_precision_recall_curve(y_true, y_scores, save_path="pr_curve.png", title="PR Curve", label="Model"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#4CAF50", lw=2, label=f"{label} (AP={ap:.4f})")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)
    plt.savefig(save_path); plt.close()
    logger.info("PR curve saved to %s", save_path)


def plot_confusion_matrix(tp, fp, fn, tn, save_path="confusion_matrix.png", title="Confusion Matrix"):
    cm_arr = np.array([[tn, fp], [fn, tp]])
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Cover", "Pred: Stego"])
    ax.set_yticklabels(["True: Cover", "True: Stego"])
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            c = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color=c, fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Count", shrink=0.8)
    plt.savefig(save_path); plt.close()
    logger.info("Confusion matrix saved to %s", save_path)


def plot_training_curves(history: Dict[str, List[float]], save_dir: str = "plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if "train_loss" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["train_loss"], "o-", color="#F44336", label="Train Loss", ms=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "loss_curve.png")); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    if "train_auc" in history:
        ax.plot(epochs, history["train_auc"], "o-", color="#2196F3", label="Train AUC", ms=3)
    if "val_auc" in history:
        ax.plot(epochs, history["val_auc"], "s-", color="#4CAF50", label="Val AUC", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC"); ax.set_title("AUC over Epochs")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "auc_curves.png")); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    if "train_f1" in history:
        ax.plot(epochs, history["train_f1"], "o-", color="#FF9800", label="Train F1", ms=3)
    if "val_f1" in history:
        ax.plot(epochs, history["val_f1"], "s-", color="#9C27B0", label="Val F1", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); ax.set_title("F1 over Epochs")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "f1_curves.png")); plt.close()

    if "lr" in history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["lr"], "-", color="#607D8B", lw=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("LR Schedule")
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "lr_schedule.png")); plt.close()
    logger.info("Training curves saved to %s/", save_dir)


def plot_robustness_results(results, save_dir="plots/robustness", metric_key="f1"):
    os.makedirs(save_dir, exist_ok=True)
    colors = {"jpeg_compression": "#2196F3", "gaussian_noise": "#F44336",
              "resize": "#4CAF50", "crop": "#FF9800"}
    for pert_name, pert_results in results.items():
        if not pert_results:
            continue
        params, values = [], []
        for label, metrics in pert_results.items():
            params.append(str(label.split("=")[-1] if "=" in label else label))
            values.append(metrics.get(metric_key, 0.0))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(params)), values, color=colors.get(pert_name, "#607D8B"), alpha=0.8)
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha="right")
        ax.set_ylabel(metric_key.upper())
        ax.set_title(f"Robustness: {pert_name.replace('_', ' ').title()}")
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis="y")
        plt.savefig(os.path.join(save_dir, f"robustness_{pert_name}.png")); plt.close()
    logger.info("Robustness plots saved to %s/", save_dir)


def plot_model_comparison(results, metrics_to_plot=None, save_path="plots/model_comparison.png"):
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    _ensure_dir(save_path)
    names = list(results.keys())
    n = len(names); nm = len(metrics_to_plot)
    x = np.arange(nm); w = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = plt.cm.Set2(np.linspace(0, 1, n))
    for i, name in enumerate(names):
        vals = [results[name].get(m, 0.0) for m in metrics_to_plot]
        ax.bar(x + i * w, vals, w, label=name, color=colors_list[i])
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
    ax.set_ylabel("Score"); ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.savefig(save_path); plt.close()
    logger.info("Model comparison saved to %s", save_path)
