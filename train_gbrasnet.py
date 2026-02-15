#!/usr/bin/env python3
"""
train_gbrasnet.py
==================
Train SRNet on the GBRAS-Net dataset (BOSSbase-1.01 / BOWS2).

Dataset layout:
    dataset/GBRASNET/BOSSbase-1.01/cover/           -> 10,000 .pgm
    dataset/GBRASNET/BOSSbase-1.01/stego/WOW/0.4bpp/stego/  -> 10,000 .pgm
    dataset/GBRASNET/BOWS2/cover/                   -> 10,000 .pgm
    dataset/GBRASNET/BOWS2/stego/WOW/0.4bpp/stego/  -> ...

Usage:
    python train_gbrasnet.py --epochs 50
    python train_gbrasnet.py --epochs 50 --stego-method WOW --bpp 0.4bpp --batch-size 16
"""

import argparse
import os
import sys
import time
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stego.datasets import (
    PairConstraintStegoDataset,
    get_train_transform,
    get_val_transform,
    pair_constraint_collate,
)
from stego.metrics import compute_metrics
from stego.model import SRNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_output.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_gbrasnet")


def bce_with_label_smoothing(logits, targets, smoothing=0.1):
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(
        logits.view(-1), targets_smooth.view(-1)
    )


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, label_smoothing=0.1, use_amp=True):
    model.train()
    total_loss = 0.0
    all_scores, all_labels = [], []
    num_batches = 0
    total_batches = len(loader)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                logits, _ = model(images)
                loss = bce_with_label_smoothing(logits, labels, smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(images)
            loss = bce_with_label_smoothing(logits, labels, smoothing=label_smoothing)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.cpu().numpy())

        if num_batches % 100 == 0 or num_batches == total_batches:
            avg = total_loss / num_batches
            logger.info("  Batch %d/%d | Running Loss: %.4f", num_batches, total_batches, avg)

    avg_loss = total_loss / max(num_batches, 1)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    return avg_loss, all_scores, all_labels


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    all_scores, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.cpu().numpy()

        if use_amp:
            with autocast():
                logits, _ = model(images)
        else:
            logits, _ = model(images)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    return compute_metrics(all_labels, all_scores, threshold=0.5)


def main():
    parser = argparse.ArgumentParser(description="Train SRNet on GBRAS-Net dataset (BOSSbase/BOWS2).")
    parser.add_argument("--dataset-root", type=str, default="dataset/GBRASNET",
                        help="Root of the GBRAS-Net dataset.")
    parser.add_argument("--subdataset", type=str, default="BOSSbase-1.01",
                        choices=["BOSSbase-1.01", "BOWS2", "both"],
                        help="Which sub-dataset to use.")
    parser.add_argument("--stego-method", type=str, default="WOW",
                        choices=["WOW", "HILL", "HUGO", "MiPOD", "S-UNIWARD"],
                        help="Steganographic algorithm.")
    parser.add_argument("--bpp", type=str, default="0.4bpp",
                        choices=["0.2bpp", "0.4bpp"],
                        help="Bits-per-pixel payload rate.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (pair-constraint doubles effective batch).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data for validation.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Checkpoint filename (default: auto-generated).")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 = disabled).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    logger.info("=" * 70)
    logger.info("  SRNet Training on GBRAS-Net Dataset")
    logger.info("=" * 70)
    logger.info("  Device:        %s", device)
    logger.info("  AMP:           %s", use_amp)
    logger.info("  Sub-dataset:   %s", args.subdataset)
    logger.info("  Stego method:  %s @ %s", args.stego_method, args.bpp)
    logger.info("  Epochs:        %d", args.epochs)
    logger.info("  Batch size:    %d (effective %d with pair-constraint)", args.batch_size, args.batch_size * 2)
    logger.info("  LR:            %s", args.lr)
    logger.info("  Image size:    %d", args.image_size)
    logger.info("  Patience:      %d", args.patience)
    logger.info("=" * 70)

    # Build dataset paths
    cover_dirs = []
    stego_dirs = []
    subdatasets = ["BOSSbase-1.01", "BOWS2"] if args.subdataset == "both" else [args.subdataset]

    for sub in subdatasets:
        cover_dir = os.path.join(args.dataset_root, sub, "cover")
        stego_dir = os.path.join(args.dataset_root, sub, "stego", args.stego_method, args.bpp, "stego")

        if not os.path.isdir(cover_dir):
            logger.error("Cover dir not found: %s", cover_dir)
            continue
        if not os.path.isdir(stego_dir):
            logger.error("Stego dir not found: %s", stego_dir)
            logger.info("Available stego methods: %s",
                        os.listdir(os.path.join(args.dataset_root, sub, "stego"))
                        if os.path.isdir(os.path.join(args.dataset_root, sub, "stego")) else "N/A")
            continue

        cover_dirs.append(cover_dir)
        stego_dirs.append(stego_dir)
        logger.info("  Found: %s -> %d cover files",
                     sub, len([f for f in os.listdir(cover_dir) if f.endswith('.pgm')]))

    if not cover_dirs:
        raise SystemExit("No valid dataset directories found! Check --dataset-root and --stego-method/--bpp.")

    # Transforms
    train_transforms = get_train_transform(image_size=args.image_size)
    val_transforms = get_val_transform(image_size=args.image_size)

    # Build dataset — add .pgm extension support
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tif", ".tiff")
    from torch.utils.data import ConcatDataset

    full_datasets = []
    for cdir, sdir in zip(cover_dirs, stego_dirs):
        ds = PairConstraintStegoDataset(
            cdir, sdir,
            image_size=args.image_size,
            transform=train_transforms,
            extensions=extensions,
        )
        logger.info("  Loaded %d pairs from %s", len(ds), os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sdir))))))
        full_datasets.append(ds)

    if len(full_datasets) > 1:
        full_dataset = ConcatDataset(full_datasets)
    else:
        full_dataset = full_datasets[0]

    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size

    logger.info("  Total pairs: %d | Train: %d | Val: %d", total_size, train_size, val_size)

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
    )

    logger.info("  Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # Model
    model = SRNet(num_classes=1, use_kv_hpf=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Model params: %s total, %s trainable",
                f"{total_params:,}", f"{trainable_params:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # Checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.save_name is None:
        args.save_name = f"srnet_{args.stego_method}_{args.bpp}_best.pth"
    save_path = os.path.join(args.checkpoint_dir, args.save_name)

    # Training loop
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [], "train_auc": [], "train_f1": [],
        "val_auc": [], "val_f1": [], "val_precision": [], "val_recall": [],
        "lr": [],
    }

    logger.info("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_scores, train_labels = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler,
            label_smoothing=args.label_smoothing, use_amp=use_amp,
        )
        train_metrics = compute_metrics(train_labels, train_scores)

        # Validate
        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_metrics["weighted_auc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_auc"].append(val_metrics["weighted_auc"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["lr"].append(current_lr)

        # Log
        improved = ""
        if val_metrics["weighted_auc"] > best_auc:
            best_auc = val_metrics["weighted_auc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            improved = " ★ BEST"
        else:
            patience_counter += 1

        logger.info(
            "Epoch %3d/%d [%5.1fs] | Loss: %.4f | "
            "Train AUC: %.4f F1: %.4f | "
            "Val AUC: %.4f F1: %.4f P: %.4f R: %.4f | "
            "LR: %.2e%s",
            epoch, args.epochs, epoch_time, train_loss,
            train_metrics["weighted_auc"], train_metrics["f1"],
            val_metrics["weighted_auc"], val_metrics["f1"],
            val_metrics["precision"], val_metrics["recall"],
            current_lr, improved,
        )

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, args.patience)
            break

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  Training Complete!")
    logger.info("  Best Val AUC: %.4f at epoch %d", best_auc, best_epoch)
    logger.info("  Best model saved to: %s", save_path)
    logger.info("=" * 70)

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f"training_history_{args.stego_method}_{args.bpp}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to: %s", history_path)

    # Plot training curves if matplotlib available
    try:
        from framework.plotting import plot_training_curves
        plot_dir = os.path.join(args.checkpoint_dir, "plots")
        plot_training_curves(history, save_dir=plot_dir)
        logger.info("Training plots saved to: %s/", plot_dir)
    except Exception as e:
        logger.warning("Could not generate plots: %s", e)

    return best_auc


if __name__ == "__main__":
    main()
