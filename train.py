"""
Robust training pipeline for SRNet-based steganalysis.
- Pair-constraint batches (cover + corresponding stego).
- KV HPF as first non-trainable layer.
- Albumentations: RandomRotate90, Flip, JPEG compression (70–95).
- AdamW 1e-3, OneCycleLR, BCEWithLogitsLoss with label smoothing 0.1.
- AMP, batch size 32 or 64.
- Evaluation: Weighted AUC, F1.
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from stego.datasets import (
    PairConstraintStegoDataset,
    build_alaska2_pairs,
    get_train_transform,
    get_val_transform,
    pair_constraint_collate,
)
from stego.features import get_device
from stego.metrics import compute_metrics
from stego.model import SRNet


def bce_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    BCEWithLogitsLoss with label smoothing to prevent overfitting on subtle noise.
    Smooth labels: 0 -> smoothing/2, 1 -> 1 - smoothing/2 (e.g. 0.05 and 0.95 for 0.1).
    """
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(1), targets_smooth.squeeze(1)
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    scaler: Optional[GradScaler],
    label_smoothing: float = 0.1,
    use_amp: bool = True,
) -> tuple:
    """
    One training epoch with AMP and label-smoothed BCE.
    Returns (mean_loss, all_preds, all_labels) for epoch-level metrics if needed.
    """
    model.train()
    running_loss = 0.0
    all_scores = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                logits, _ = model(images)
                loss = bce_with_label_smoothing(logits, labels, smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(images)
            loss = bce_with_label_smoothing(logits, labels, smoothing=label_smoothing)
            loss.backward()
            optimizer.step()

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        running_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.squeeze(1).cpu().numpy())

    mean_loss = running_loss / max(1, len(loader))
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return mean_loss, all_scores, all_labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Returns metrics dict: weighted_auc, f1, precision, recall, tp, fp, fn, tn."""
    model.eval()
    all_scores = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast():
                logits, _ = model(images)
        else:
            logits, _ = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return compute_metrics(all_labels, all_scores, threshold=0.5)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train SRNet steganalysis with pair-constraint and AMP.")
    p.add_argument("--alaska2-root", type=str, default=None, help="ALASKA2 root (e.g. data/ALASKA2).")
    p.add_argument("--bossbase-cover", type=str, default=None, help="BOSSBase cover directory.")
    p.add_argument("--bossbase-stego", type=str, default=None, help="BOSSBase stego directory.")
    p.add_argument("--data-root", type=str, default=None, help="Single root for ALASKA2 (overrides --alaska2-root).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--jpeg-min", type=int, default=70)
    p.add_argument("--jpeg-max", type=int, default=95)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--save-name", type=str, default="srnet_best.pth")
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    p.add_argument("--no-kv-hpf", action="store_true", help="Disable KV HPF first layer.")
    return p


def main():
    args = build_arg_parser().parse_args()
    device = get_device()
    use_amp = not getattr(args, "no_amp", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # Data roots
    alaska2_root = args.data_root or args.alaska2_root
    if alaska2_root is None and (args.bossbase_cover is None or args.bossbase_stego is None):
        raise SystemExit("Provide --data-root (ALASKA2) or --bossbase-cover and --bossbase-stego.")

    train_transforms = get_train_transform(
        image_size=args.image_size,
        jpeg_quality_min=args.jpeg_min,
        jpeg_quality_max=args.jpeg_max,
    )
    val_transforms = get_val_transform(image_size=args.image_size)

    datasets = []
    if alaska2_root:
        cover_dir, stego_dir = build_alaska2_pairs(alaska2_root, "train")
        if os.path.isdir(cover_dir) and os.path.isdir(stego_dir):
            datasets.append(
                PairConstraintStegoDataset(
                    cover_dir, stego_dir,
                    image_size=args.image_size,
                    transform=train_transforms,
                )
            )
    if args.bossbase_cover and args.bossbase_stego:
        datasets.append(
            PairConstraintStegoDataset(
                args.bossbase_cover,
                args.bossbase_stego,
                image_size=args.image_size,
                transform=train_transforms,
            )
        )

    if not datasets:
        raise SystemExit("No dataset found. Check --data-root or --bossbase-cover/--bossbase-stego.")

    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(datasets)

    val_datasets = []
    if alaska2_root:
        cover_dir_v, stego_dir_v = build_alaska2_pairs(alaska2_root, "val")
        if os.path.isdir(cover_dir_v) and os.path.isdir(stego_dir_v):
            val_datasets.append(
                PairConstraintStegoDataset(
                    cover_dir_v, stego_dir_v,
                    image_size=args.image_size,
                    transform=val_transforms,
                )
            )
    if not val_datasets:
        val_dataset = train_dataset  # fallback
    else:
        from torch.utils.data import ConcatDataset
        val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
    )

    model = SRNet(num_classes=1, use_kv_hpf=not args.no_kv_hpf).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_scores, train_labels = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            label_smoothing=args.label_smoothing,
            use_amp=use_amp,
        )
        train_metrics = compute_metrics(train_labels, train_scores)
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Train AUC: {train_metrics['weighted_auc']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['weighted_auc']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"Val FP: {val_metrics['fp']} FN: {val_metrics['fn']}"
        )

        if val_metrics["weighted_auc"] > best_auc:
            best_auc = val_metrics["weighted_auc"]
            path = os.path.join(args.checkpoint_dir, args.save_name)
            torch.save(model.state_dict(), path)
            print(f"Saved best model to {path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
