"""
Evaluate SRNet checkpoint and output confusion matrix.
Useful to track False Positives (innocent images flagged as stego) in social-engineering contexts.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from stego.datasets import (
    PairConstraintStegoDataset,
    build_alaska2_pairs,
    get_val_transform,
    pair_constraint_collate,
)
from stego.features import get_device
from stego.metrics import compute_metrics, confusion_matrix_binary
from stego.model import SRNet


def plot_confusion_matrix(tp: int, fp: int, fn: int, tn: int, save_path: str) -> None:
    """Plot and save confusion matrix as text + optional matplotlib figure."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Cover", "Pred: Stego"])
    ax.set_yticklabels(["True: Cover", "True: Stego"])
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")


def main():
    p = argparse.ArgumentParser(description="Evaluate SRNet and print/plot confusion matrix.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint.")
    p.add_argument("--data-root", type=str, default=None, help="ALASKA2 root (val split).")
    p.add_argument("--cover-dir", type=str, default=None)
    p.add_argument("--stego-dir", type=str, default=None)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", type=str, default=None, help="Save confusion matrix plot here.")
    p.add_argument("--no-kv-hpf", action="store_true")
    args = p.parse_args()

    device = get_device()
    model = SRNet(num_classes=1, use_kv_hpf=not args.no_kv_hpf).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if args.data_root:
        cover_dir, stego_dir = build_alaska2_pairs(args.data_root, "val")
    elif args.cover_dir and args.stego_dir:
        cover_dir, stego_dir = args.cover_dir, args.stego_dir
    else:
        raise SystemExit("Provide --data-root or --cover-dir and --stego-dir.")

    val_t = get_val_transform(args.image_size)
    dataset = PairConstraintStegoDataset(cover_dir, stego_dir, image_size=args.image_size, transform=val_t)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pair_constraint_collate,
    )

    all_scores = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits, _ = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    y_pred = (all_scores >= 0.5).astype(np.int64)

    tp, fp, fn, tn = confusion_matrix_binary(all_labels, y_pred)
    metrics = compute_metrics(all_labels, all_scores, threshold=0.5)

    print("Confusion matrix (TP, FP, FN, TN):", (tp, fp, fn, tn))
    print("Weighted AUC:", metrics["weighted_auc"])
    print("F1:", metrics["f1"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("False Positives (innocent flagged as stego):", fp)
    print("False Negatives (stego missed):", fn)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_confusion_matrix(tp, fp, fn, tn, os.path.join(args.output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
