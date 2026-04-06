#!/usr/bin/env python3
"""
train_openimages.py
====================
End-to-end pipeline:
  1. Download a configurable number of images from Open Images V7 (train split).
  2. Generate stego versions using LSB embedding.
  3. Train SRNet steganalysis model on cover/stego pairs for 20 epochs on GPU.

Usage:
    python train_openimages.py                          # Download 1000 images + train 20 epochs
    python train_openimages.py --num-images 2000        # More images
    python train_openimages.py --skip-download          # Skip download if images already exist
    python train_openimages.py --epochs 30 --batch-size 64
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stego.model import SRNet
from stego.features import get_device
from stego.metrics import compute_metrics
from stego.datasets import (
    PairConstraintStegoDataset,
    get_train_transform,
    get_val_transform,
    pair_constraint_collate,
)
from framework.embedding import LSBEmbedder, generate_random_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Step 1: Download Open Images V7 train images
# ==============================================================================

OPEN_IMAGES_TRAIN_METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/train/"
    "train-images-boxable-with-rotation.csv"
)

# Open Images V7 images are publicly accessible via this URL pattern:
# https://storage.googleapis.com/openimages/fiftyone/<split>/<image_id>.jpg
# But actually hosted on various providers. We'll use the AWS S3 mirror from CVDF.
# Format: https://s3.amazonaws.com/open-images-dataset/train/<image_id>.jpg

OPEN_IMAGES_DOWNLOAD_URL_TEMPLATE = (
    "https://s3.amazonaws.com/open-images-dataset/train/{image_id}.jpg"
)


def download_image_ids_list(metadata_csv_path: str, num_images: int, seed: int = 42) -> list:
    """
    Download the train image metadata CSV and return a list of image IDs.
    If the CSV already exists locally, just read it.
    """
    import urllib.request

    if not os.path.exists(metadata_csv_path):
        logger.info("Downloading Open Images train metadata CSV...")
        logger.info("URL: %s", OPEN_IMAGES_TRAIN_METADATA_URL)
        urllib.request.urlretrieve(OPEN_IMAGES_TRAIN_METADATA_URL, metadata_csv_path)
        logger.info("Metadata CSV saved to %s", metadata_csv_path)
    else:
        logger.info("Using existing metadata CSV: %s", metadata_csv_path)

    # Read image IDs from the CSV
    image_ids = []
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(row["ImageID"])

    logger.info("Total available train image IDs: %d", len(image_ids))

    # Randomly sample the requested number
    rng = random.Random(seed)
    if num_images < len(image_ids):
        image_ids = rng.sample(image_ids, num_images)
    else:
        rng.shuffle(image_ids)

    logger.info("Selected %d image IDs for download", len(image_ids))
    return image_ids


def download_single_image(image_id: str, output_dir: str, min_size: int = 256) -> Optional[str]:
    """Download a single image from Open Images. Returns the saved path or None on failure."""
    import urllib.request
    import urllib.error

    url = OPEN_IMAGES_DOWNLOAD_URL_TEMPLATE.format(image_id=image_id)
    output_path = os.path.join(output_dir, f"{image_id}.jpg")

    if os.path.exists(output_path):
        return output_path

    try:
        urllib.request.urlretrieve(url, output_path)
        # Verify it's a valid image and meets minimum size
        with Image.open(output_path) as img:
            w, h = img.size
            if w < min_size or h < min_size:
                os.remove(output_path)
                return None
        return output_path
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        return None


def download_open_images(
    output_dir: str,
    metadata_csv_path: str,
    num_images: int = 1000,
    num_workers: int = 8,
    seed: int = 42,
) -> int:
    """
    Download images from Open Images V7 train split.
    Returns the number of successfully downloaded images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check how many images already exist
    existing = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
    if len(existing) >= num_images:
        logger.info(
            "Already have %d images in %s (requested %d). Skipping download.",
            len(existing), output_dir, num_images,
        )
        return len(existing)

    # How many more do we need
    needed = num_images - len(existing)
    # Request more IDs than needed to account for failures
    request_count = int(needed * 1.5) + 100

    image_ids = download_image_ids_list(metadata_csv_path, request_count, seed)

    # Filter out IDs we already have
    existing_ids = {os.path.splitext(f)[0] for f in existing}
    image_ids = [iid for iid in image_ids if iid not in existing_ids]

    logger.info("Downloading up to %d images using %d workers...", needed, num_workers)

    downloaded = len(existing)
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for img_id in image_ids:
            if downloaded >= num_images:
                break
            future = executor.submit(download_single_image, img_id, output_dir)
            futures[future] = img_id

        for future in as_completed(futures):
            if downloaded >= num_images:
                # Cancel remaining futures
                break
            result = future.result()
            if result is not None:
                downloaded += 1
                if downloaded % 50 == 0:
                    logger.info("Downloaded %d / %d images...", downloaded, num_images)
            else:
                failed += 1

    logger.info(
        "Download complete: %d images saved, %d failed. Directory: %s",
        downloaded, failed, output_dir,
    )
    return downloaded


# ==============================================================================
# Step 2: Generate stego images from downloaded covers
# ==============================================================================

def generate_stego_from_covers(
    cover_dir: str,
    stego_dir: str,
    payload_length: int = 16,
    seed: int = 42,
    variable_payload: bool = True,
    payload_range: Tuple[int, int] = (8, 64),
) -> int:
    """
    Generate stego images from cover images using LSB embedding.
    Returns the number of stego images generated.
    """
    os.makedirs(stego_dir, exist_ok=True)

    embedder = LSBEmbedder()
    rng = random.Random(seed)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    cover_files = sorted(
        f for f in os.listdir(cover_dir)
        if os.path.splitext(f)[1].lower() in extensions
    )

    # Check how many stego images already exist
    existing_stego = set(os.listdir(stego_dir))
    covers_needing_stego = []
    for f in cover_files:
        name_no_ext = os.path.splitext(f)[0]
        stego_name = f"{name_no_ext}.png"
        if stego_name not in existing_stego:
            covers_needing_stego.append(f)

    if not covers_needing_stego:
        logger.info("All %d stego images already exist. Skipping generation.", len(cover_files))
        return len(cover_files)

    logger.info(
        "Generating stego images for %d / %d covers...",
        len(covers_needing_stego), len(cover_files),
    )

    generated = 0
    for idx, filename in enumerate(covers_needing_stego):
        cover_path = os.path.join(cover_dir, filename)
        name_no_ext = os.path.splitext(filename)[0]

        if variable_payload:
            plen = rng.randint(payload_range[0], payload_range[1])
        else:
            plen = payload_length

        message = generate_random_message(length=plen, charset="alphanumeric", seed=seed + idx)

        try:
            # Crop center 256x256 to ensure payload fills the evaluated patch
            cover_img = Image.open(cover_path).convert("RGB")
            w, h = cover_img.size
            if w > 256 or h > 256:
                # Center crop
                left = (w - 256) // 2
                top = (h - 256) // 2
                cover_img = cover_img.crop((left, top, left + 256, top + 256))
            
            # Save the cropped cover back so it natively matches the stego image!
            cover_png_path = os.path.join(cover_dir, f"{name_no_ext}.png")
            cover_img.save(cover_png_path, format="PNG") # Use PNG to avoid JPEG artifacts ruining cover LSBs
            if cover_path != cover_png_path and os.path.exists(cover_path):
                os.remove(cover_path)

            stego_img = embedder.embed(cover_img, message)

            # Save as PNG to preserve LSB embedding (no lossy compression)
            stego_path = os.path.join(stego_dir, f"{name_no_ext}.png")
            stego_img.save(stego_path, format="PNG")
            generated += 1

            if (generated) % 100 == 0:
                logger.info("Generated %d stego images...", generated)

        except Exception as e:
            logger.warning("Failed to embed in '%s': %s", filename, e)

    total = len(cover_files) - len(covers_needing_stego) + generated
    logger.info("Stego generation complete: %d total stego images.", total)
    return total


# ==============================================================================
# Step 3: Training with SRNet
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target, smoothing=0.0):
        targets_smooth = target * (1.0 - smoothing) + 0.5 * smoothing
        bce = F.binary_cross_entropy_with_logits(
            pred.squeeze(1), targets_smooth.squeeze(1), reduction='none'
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

focal_criterion = FocalLoss(gamma=2, alpha=0.25)


class PairListDataset(Dataset):
    """Pair dataset backed by an explicit (cover_path, stego_path) list."""

    def __init__(self, pairs, transform):
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cover_path, stego_path = self.pairs[idx]
        cover = np.array(Image.open(cover_path).convert("RGB"))
        stego = np.array(Image.open(stego_path).convert("RGB"))

        if self.transform is not None and "stego_image" in getattr(self.transform, "additional_targets", {}):
            out = self.transform(image=cover, stego_image=stego)
            cover_t = out["image"]
            stego_t = out["stego_image"]
        elif self.transform is not None:
            cover_t = self.transform(image=cover)["image"]
            stego_t = self.transform(image=stego)["image"]
        else:
            raise RuntimeError("Transform must be provided.")

        label_cover = torch.tensor(0.0, dtype=torch.float32)
        label_stego = torch.tensor(1.0, dtype=torch.float32)
        return cover_t, stego_t, label_cover, label_stego


def estimate_pair_bpp(cover_path: str, stego_path: str) -> float:
    cover = np.array(Image.open(cover_path).convert("RGB"))
    stego = np.array(Image.open(stego_path).convert("RGB"))
    diff = np.sum(cover != stego)
    payload_bits = diff * 2.0
    return float(payload_bits / max(1, cover.size))


def build_stratified_splits(pairs, val_split: float, test_split: float, seed: int):
    bpps = np.array([estimate_pair_bpp(c, s) for c, s in pairs], dtype=np.float32)
    bins = np.digitize(bpps, np.quantile(bpps, [0.2, 0.4, 0.6, 0.8]), right=True)
    indices = np.arange(len(pairs))

    train_idx, tmp_idx = train_test_split(
        indices,
        test_size=val_split + test_split,
        random_state=seed,
        stratify=bins,
    )
    tmp_bins = bins[tmp_idx]
    relative_val = val_split / (val_split + test_split)
    val_rel_idx, test_rel_idx = train_test_split(
        np.arange(len(tmp_idx)),
        test_size=1.0 - relative_val,
        random_state=seed,
        stratify=tmp_bins,
    )
    val_idx = tmp_idx[val_rel_idx]
    test_idx = tmp_idx[test_rel_idx]
    return train_idx, val_idx, test_idx, bpps


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).float()
        if images.max() > 1.0:
            images = images / 255.0
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        if use_amp:
            with autocast():
                logits, _ = model(images)
        else:
            logits, _ = model(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.squeeze(1).cpu().numpy())
    return np.concatenate(all_scores, axis=0), np.concatenate(all_labels, axis=0)


def mine_hard_negative_pairs(
    model: nn.Module,
    val_pairs,
    val_transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    threshold: float = 0.5,
):
    val_dataset = PairListDataset(val_pairs, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
    )
    scores, labels = predict_scores(model, val_loader, device=device, use_amp=use_amp)
    preds = (scores >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    hard_pairs = []
    for pair_idx in range(len(val_pairs)):
        i_cover = 2 * pair_idx
        i_stego = i_cover + 1
        cover_wrong = preds[i_cover] != labels[i_cover]
        stego_wrong = preds[i_stego] != labels[i_stego]
        if cover_wrong or stego_wrong:
            hard_pairs.append(val_pairs[pair_idx])
    return hard_pairs


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: Optional[GradScaler],
    label_smoothing: float = 0.0,
    use_amp: bool = True,
    mixup_alpha: float = 0.0,
) -> tuple:
    """One training epoch with AMP and label-smoothed BCE."""
    model.train()
    running_loss = 0.0
    all_scores = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True).float()
        if images.max() > 1.0:
            images = images / 255.0
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if mixup_alpha > 0.0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(images.size(0), device=images.device)
            mixed_images = lam * images + (1.0 - lam) * images[perm]
            mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
        else:
            mixed_images = images
            mixed_labels = labels

        if use_amp and scaler is not None:
            with autocast():
                logits, _ = model(mixed_images)
                loss = focal_criterion(logits, mixed_labels, smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(mixed_images)
            loss = focal_criterion(logits, mixed_labels, smoothing=label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
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
    label_smoothing: float = 0.0,
) -> tuple:
    """Returns metric lists/mean_loss."""
    model.eval()
    all_scores = []
    all_labels = []
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True).float()
        if images.max() > 1.0:
            images = images / 255.0
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        if use_amp:
            with autocast():
                logits, _ = model(images)
                loss = focal_criterion(logits, labels, smoothing=label_smoothing)
        else:
            logits, _ = model(images)
            loss = focal_criterion(logits, labels, smoothing=label_smoothing)
        running_loss += loss.item()
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.squeeze(1).cpu().numpy())

    mean_loss = running_loss / max(1, len(loader))
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return mean_loss, compute_metrics(all_labels, all_scores, threshold=0.5)

def print_status_report(epoch, max_epochs, phase, epoch_time, t_loss, v_loss,
                        t_met, v_met, best_ep, best_auc, best_acc, status):
    mm, ss = divmod(int(epoch_time), 60)
    hh, mm = divmod(mm, 60)
    t_str = f"{hh:02d}:{mm:02d}"

    def g(d, k, default=0.0): return d.get(k, default)
    
    t_acc = g(t_met, 'accuracy') * 100
    v_acc = g(v_met, 'accuracy') * 100
    v_auc = g(v_met, 'weighted_auc')
    v_f1 = g(v_met, 'f1')
    v_prec = g(v_met, 'precision')
    v_rec = g(v_met, 'recall')
    v_pe = g(v_met, 'pe', 0.0)
    v_eer = g(v_met, 'eer', 0.0)
    v_tpr_1 = g(v_met, 'tpr_at_fpr_0.01', 0.0)
    v_tpr_5 = g(v_met, 'tpr_at_fpr_0.05', 0.0)
    tp, tn = g(v_met, 'tp', 0), g(v_met, 'tn', 0)
    fp, fn = g(v_met, 'fp', 0), g(v_met, 'fn', 0)
    
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)

    line = "+" + "-" * 62 + "+"
    print(line)
    print(f"| Epoch {epoch:<3}/{max_epochs:<3} | Phase {phase:<5} | Time: {t_str:<14} |")
    print(line)
    print(f"| LOSS      Train: {t_loss:.4f}  |  Val: {v_loss:.4f}{' ':11} |")
    print(f"| ACCURACY  Train: {t_acc:5.1f}%   |  Val: {v_acc:5.1f}%{' ':12} |")
    print(f"| AUC              {v_auc:.4f}  |  F1:  {v_f1:.4f}{' ':11} |")
    print(f"| PRECISION        {v_prec:.4f}  |  RECALL: {v_rec:.4f}{' ':8} |")
    print(line)
    print(f"| STEGANALYSIS METRICS{' ':41} |")
    print(f"| PE: {v_pe:.4f}  |  EER: {v_eer:.4f}{' ':24} |")
    print(f"| TPR@FPR=1%: {v_tpr_1:.4f}  |  TPR@FPR=5%: {v_tpr_5:.4f}{' ':8} |")
    print(line)
    print(f"| CONFUSION MATRIX{' ':45} |")
    print(f"| TP: {tp:<4}  TN: {tn:<4}  FP: {fp:<4}  FN: {fn:<4}{' ':7} |")
    print(f"| FPR: {fpr:.4f}  |  FNR: {fnr:.4f}{' ':23} |")
    print(line)
    print(f"| STATUS: {status:<45} |")
    print(f"| BEST MODEL: Epoch {best_ep:<3} | AUC {best_auc:.4f} | Acc {best_acc:5.1f}%   |")
    print(line)


def ping_epoch_trained(
    checkpoint_dir: str,
    epoch: int,
    max_epochs: int,
    val_auc: float,
    val_acc_pct: float,
    status: str,
) -> None:
    """Notify that an epoch finished: log file, logger line, short Windows beep."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ping_path = os.path.join(checkpoint_dir, "epoch_pings.log")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"PING epoch {epoch}/{max_epochs} done | val_AUC={val_auc:.4f} "
        f"val_Acc={val_acc_pct:.2f}% | {status}"
    )
    logger.info(">>> %s", line)
    try:
        with open(ping_path, "a", encoding="utf-8") as pf:
            pf.write(f"{ts} {line}\n")
    except OSError:
        pass
    try:
        import winsound

        winsound.Beep(880, 160)
    except Exception:
        pass
    sys.stdout.flush()


def run_training(
    cover_dir: str,
    stego_dir: str,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    image_size: int = 256,
    label_smoothing: float = 0.0,
    num_workers: int = 4,
    checkpoint_dir: str = "checkpoints",
    save_name: str = "srnet_openimages_best.pth",
    val_split: float = 0.10,
    test_split: float = 0.10,
    use_amp: bool = True,
    use_kv_hpf: bool = True,
    use_channel_attention: bool = False,
    use_learnable_hpf: bool = False,
    enable_mixup: bool = False,
    mixup_alpha: float = 0.2,
    enable_curriculum: bool = False,
    early_stopping_patience: int = 20,
    enable_hard_negative_mining: bool = False,
    hard_negative_ratio: float = 2.0,
    seed: int = 42,
    split_seed: int = 42,
    stage_name: str = "unnamed_stage",
    epoch_ping: bool = False,
    enable_diagnostic_stops: bool = True,
):
    """Train SRNet on cover/stego pairs from Open Images."""
    set_global_seed(seed)
    device = get_device()
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("Epochs: %d", epochs)
    logger.info("Batch size: %d", batch_size)
    logger.info("Learning rate: %s", lr)
    logger.info("Image size: %d", image_size)
    logger.info("Label smoothing: %s", label_smoothing)
    logger.info("AMP enabled: %s", use_amp and device.type == "cuda")
    logger.info("KV HPF: %s", use_kv_hpf)
    logger.info("Learnable HPF: %s", use_learnable_hpf)
    logger.info("Channel Attention: %s", use_channel_attention)
    logger.info("Mixup: %s", enable_mixup)
    logger.info("Curriculum: %s", enable_curriculum)
    logger.info("Hard negative mining: %s", enable_hard_negative_mining)
    logger.info("Early stopping patience: %s", early_stopping_patience if early_stopping_patience > 0 else "disabled")
    logger.info("Epoch ping: %s", epoch_ping)
    logger.info("Diagnostic early stops (FP/FN symmetry, etc.): %s", enable_diagnostic_stops)
    logger.info("Seed: %d | Split seed: %d", seed, split_seed)
    logger.info("Cover dir: %s", cover_dir)
    logger.info("Stego dir: %s", stego_dir)
    logger.info("=" * 70)

    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    # Build transforms
    train_transforms = get_train_transform(image_size=image_size)
    val_transforms = get_val_transform(image_size=image_size)

    # Build full pair list once, then create split-specific datasets with proper transforms
    pair_source = PairConstraintStegoDataset(
        cover_dir=cover_dir,
        stego_dir=stego_dir,
        image_size=image_size,
        transform=val_transforms,
    )
    logger.info("Total cover/stego pairs: %d", len(pair_source))

    if len(pair_source) == 0:
        raise RuntimeError(
            f"No matching cover/stego pairs found!\n"
            f"  Cover dir: {cover_dir} ({len(os.listdir(cover_dir))} files)\n"
            f"  Stego dir: {stego_dir} ({len(os.listdir(stego_dir))} files)"
        )

    train_idx, val_idx, test_idx, bpps = build_stratified_splits(
        pair_source.pairs,
        val_split=val_split,
        test_split=test_split,
        seed=split_seed,
    )
    train_pairs = [pair_source.pairs[i] for i in train_idx]
    val_pairs = [pair_source.pairs[i] for i in val_idx]
    test_pairs = [pair_source.pairs[i] for i in test_idx]

    train_dataset = PairListDataset(train_pairs, train_transforms)
    val_dataset_raw = PairListDataset(val_pairs, val_transforms)
    test_dataset_raw = PairListDataset(test_pairs, val_transforms)

    train_names = {os.path.basename(p[0]) for p in train_pairs}
    val_names = {os.path.basename(p[0]) for p in val_pairs}
    test_names = {os.path.basename(p[0]) for p in test_pairs}
    leakage = (train_names & val_names) or (train_names & test_names) or (val_names & test_names)
    if leakage:
        raise RuntimeError("Data leakage detected across train/val/test splits.")

    logger.info("Train pairs: %d | Val pairs: %d | Test pairs: %d", len(train_pairs), len(val_pairs), len(test_pairs))
    logger.info("Payload bpp (global): min=%.4f max=%.4f mean=%.4f", float(np.min(bpps)), float(np.max(bpps)), float(np.mean(bpps)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset_raw,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
    )
    test_loader = DataLoader(
        test_dataset_raw,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pair_constraint_collate,
    )

    # Model
    model = SRNet(
        num_classes=1,
        use_kv_hpf=use_kv_hpf,
        use_learnable_hpf=use_learnable_hpf,
        use_channel_attention=use_channel_attention,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: SRNet | Total params: %s | Trainable: %s",
                f"{total_params:,}", f"{trainable_params:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    max_train_pairs = len(train_pairs) + int(max(0.0, hard_negative_ratio) * len(val_pairs))
    steps_per_epoch = max(1, math.ceil(max_train_pairs / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Checkpointing
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_auc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    best_acc_val = 0.0
    epochs_without_improvement = 0
    mined_pairs = []

    logger.info("")
    logger.info("Starting training for %d epochs...", epochs)
    logger.info("-" * 90)

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        current_phase = 1
        current_train_pairs = list(train_pairs)
        if enable_curriculum:
            sorted_idx = np.argsort(-bpps[train_idx])
            if epoch <= 50:
                use_n = max(1, int(len(sorted_idx) * 0.30))
                current_phase = 1
            elif epoch <= 100:
                use_n = max(1, int(len(sorted_idx) * 0.60))
                current_phase = 2
            else:
                use_n = len(sorted_idx)
                current_phase = 3
            selected = sorted_idx[:use_n]
            current_train_pairs = [train_pairs[i] for i in selected]
        if enable_hard_negative_mining and epoch >= 30 and len(mined_pairs) > 0:
            current_train_pairs = current_train_pairs + mined_pairs

        train_loader = DataLoader(
            PairListDataset(current_train_pairs, train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pair_constraint_collate,
            drop_last=True,
        )

        train_loss, train_scores, train_labels = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            label_smoothing=label_smoothing,
            use_amp=use_amp,
            mixup_alpha=mixup_alpha if enable_mixup else 0.0,
        )
        train_metrics = compute_metrics(train_labels, train_scores)
        val_loss, val_metrics = evaluate(model, val_loader, device, use_amp=use_amp, label_smoothing=label_smoothing)

        epoch_time = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]["lr"]

        val_auc = val_metrics.get("weighted_auc", 0)
        val_f1 = val_metrics.get("f1", 0)
        val_acc = val_metrics.get("accuracy", 0) * 100
        
        status = "ON_TRACK"
        if epoch == 30 and (val_acc < 60 or val_auc < 0.62): status = "CHECKPOINT_MISSED (Ep 30)"
        elif epoch == 50 and (val_acc < 70 or val_auc < 0.72): status = "CHECKPOINT_MISSED (Ep 50)"
        elif epoch == 75 and (val_acc < 78 or val_auc < 0.82): status = "CHECKPOINT_MISSED (Ep 75)"
        elif epoch == 100 and (val_acc < 82 or val_auc < 0.87): status = "CHECKPOINT_MISSED (Ep 100)"
        elif epoch == 150 and (val_acc < 85 or val_auc < 0.90): status = "CHECKPOINT_MISSED (Ep 150)"

        if val_auc > best_auc + 1e-3:
            best_auc = val_auc
            best_f1 = val_f1
            best_epoch = epoch
            best_acc_val = val_acc
            epochs_without_improvement = 0
            path = os.path.join(checkpoint_dir, save_name)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "best_f1": best_f1,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            }, path)
        else:
            epochs_without_improvement += 1
        
        print_status_report(epoch, epochs, str(current_phase), epoch_time, train_loss, val_loss,
                            train_metrics, val_metrics, best_epoch, best_auc, best_acc_val, status)

        if epoch_ping:
            ping_epoch_trained(checkpoint_dir, epoch, epochs, val_auc, val_acc, status)

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            logger.error("Loss became NaN/Inf; stopping immediately for diagnosis.")
            break
        if enable_diagnostic_stops:
            if epoch > 15 and val_auc < 0.50:
                logger.error("AUC dropped below 0.50 after epoch 15; stopping for diagnosis.")
                break
            if epoch > 20 and abs(val_metrics.get("fp", 0) - val_metrics.get("fn", 0)) <= 2:
                logger.error("FP/FN remain nearly symmetric after epoch 20; stopping for diagnosis.")
                break
            if epoch == 30 and val_acc <= 55.0:
                logger.error("Accuracy did not exceed 55%% by epoch 30; stopping for diagnosis.")
                break
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.info("Early stopping triggered on weighted_auc patience=%d", early_stopping_patience)
            break

        if enable_hard_negative_mining and epoch >= 30:
            hard_pairs = mine_hard_negative_pairs(
                model=model,
                val_pairs=val_pairs,
                val_transform=val_transforms,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                use_amp=use_amp,
            )
            repeat_count = max(1, int(round(hard_negative_ratio)))
            mined_pairs = hard_pairs * repeat_count
            logger.info(
                "Hard negative mining: %d difficult val pairs, adding %d mined pairs next epoch.",
                len(hard_pairs),
                len(mined_pairs),
            )

        # Save last checkpoint
        last_path = os.path.join(checkpoint_dir, "srnet_openimages_last.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
        }, last_path)

    total_time = time.time() - train_start
    logger.info("-" * 90)
    logger.info("Training complete!")
    logger.info("Total time: %.1f minutes (%.1f hours)", total_time / 60, total_time / 3600)
    logger.info("Best Val AUC: %.4f | Best Val F1: %.4f", best_auc, best_f1)
    logger.info("Best model saved to: %s", os.path.join(checkpoint_dir, save_name))

    # Final test evaluation from best checkpoint
    best_path = os.path.join(checkpoint_dir, save_name)
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    test_scores, test_labels = predict_scores(model, test_loader, device=device, use_amp=use_amp)
    test_metrics = compute_metrics(test_labels, test_scores, threshold=0.5)
    logger.info(
        "Test metrics | Acc: %.4f AUC: %.4f F1: %.4f P: %.4f R: %.4f",
        test_metrics.get("accuracy", 0.0),
        test_metrics.get("weighted_auc", 0.0),
        test_metrics.get("f1", 0.0),
        test_metrics.get("precision", 0.0),
        test_metrics.get("recall", 0.0),
    )
    return {
        "best_auc": best_auc,
        "best_acc": best_acc_val / 100.0,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "best_path": best_path,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "test_metrics": test_metrics,
    }


def update_upgrade_tracking(checkpoint_dir: str, stage_name: str, run_out: dict) -> dict:
    os.makedirs(checkpoint_dir, exist_ok=True)
    track_path = os.path.join(checkpoint_dir, "upgrade_tracking.json")
    if os.path.exists(track_path):
        with open(track_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {"runs": []}

    current = {
        "stage_name": stage_name,
        "best_auc": float(run_out.get("best_auc", 0.0)),
        "best_acc": float(run_out.get("best_acc", 0.0)),
        "best_f1": float(run_out.get("best_f1", 0.0)),
        "test_auc": float(run_out.get("test_metrics", {}).get("weighted_auc", 0.0)),
        "test_acc": float(run_out.get("test_metrics", {}).get("accuracy", 0.0)),
        "test_f1": float(run_out.get("test_metrics", {}).get("f1", 0.0)),
    }
    prev = history["runs"][-1] if history["runs"] else None
    if prev is None:
        current["delta_test_acc"] = 0.0
        current["delta_test_auc"] = 0.0
        current["delta_test_f1"] = 0.0
    else:
        current["delta_test_acc"] = current["test_acc"] - float(prev.get("test_acc", 0.0))
        current["delta_test_auc"] = current["test_auc"] - float(prev.get("test_auc", 0.0))
        current["delta_test_f1"] = current["test_f1"] - float(prev.get("test_f1", 0.0))

    history["runs"].append(current)
    with open(track_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Upgrade delta (%s) | dAcc: %+0.4f dAUC: %+0.4f dF1: %+0.4f",
        stage_name,
        current["delta_test_acc"],
        current["delta_test_auc"],
        current["delta_test_f1"],
    )
    return current


def run_ensemble_training(
    cover_dir: str,
    stego_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    label_smoothing: float,
    num_workers: int,
    checkpoint_dir: str,
    val_split: float,
    test_split: float,
    use_amp: bool,
    use_kv_hpf: bool,
    enable_mixup: bool,
    mixup_alpha: float,
    enable_curriculum: bool,
    early_stopping_patience: int,
    enable_hard_negative_mining: bool,
    hard_negative_ratio: float,
    split_seed: int,
    seeds: list[int],
    stage_name: str,
):
    variants = [
        ("A_channel_attention", True, False),
        ("B_learnable_hpf", False, True),
        ("C_channel_and_hpf", True, True),
    ]
    if len(seeds) < len(variants):
        raise ValueError("Provide at least 3 seeds for ensemble training.")

    results = []
    for i, (name, use_ca, use_lhpf) in enumerate(variants):
        logger.info("=" * 70)
        logger.info("ENSEMBLE VARIANT %s | seed=%d", name, seeds[i])
        logger.info("=" * 70)
        out = run_training(
            cover_dir=cover_dir,
            stego_dir=stego_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            image_size=image_size,
            label_smoothing=label_smoothing,
            num_workers=num_workers,
            checkpoint_dir=checkpoint_dir,
            save_name=f"srnet_openimages_{name}.pth",
            val_split=val_split,
            test_split=test_split,
            use_amp=use_amp,
            use_kv_hpf=use_kv_hpf,
            use_channel_attention=use_ca,
            use_learnable_hpf=use_lhpf,
            enable_mixup=enable_mixup,
            mixup_alpha=mixup_alpha,
            enable_curriculum=enable_curriculum,
            early_stopping_patience=early_stopping_patience,
            enable_hard_negative_mining=enable_hard_negative_mining,
            hard_negative_ratio=hard_negative_ratio,
            seed=seeds[i],
            split_seed=split_seed,
            stage_name=f"{stage_name}_{name}",
        )
        results.append((name, out))

    ensemble_scores = np.mean([r[1]["test_scores"] for r in results], axis=0)
    ensemble_labels = results[0][1]["test_labels"]
    ensemble_metrics = compute_metrics(ensemble_labels, ensemble_scores, threshold=0.5)
    logger.info("=" * 70)
    logger.info(
        "ENSEMBLE TEST | Acc: %.4f AUC: %.4f F1: %.4f P: %.4f R: %.4f",
        ensemble_metrics.get("accuracy", 0.0),
        ensemble_metrics.get("weighted_auc", 0.0),
        ensemble_metrics.get("f1", 0.0),
        ensemble_metrics.get("precision", 0.0),
        ensemble_metrics.get("recall", 0.0),
    )
    logger.info("=" * 70)
    ensemble_out = {"variants": results, "ensemble_metrics": ensemble_metrics}
    update_upgrade_tracking(
        checkpoint_dir=checkpoint_dir,
        stage_name=f"{stage_name}_ensemble",
        run_out={
            "best_auc": max(r[1]["best_auc"] for r in results),
            "best_acc": max(r[1]["best_acc"] for r in results),
            "best_f1": max(r[1]["best_f1"] for r in results),
            "test_metrics": ensemble_metrics,
        },
    )
    return ensemble_out


# ==============================================================================
# Main
# ==============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Train SRNet steganalysis on Open Images V7 dataset (GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_openimages.py                              # Download 1000 images, train 20 epochs
  python train_openimages.py --num-images 5000            # More training data
  python train_openimages.py --skip-download              # Re-use already downloaded images
  python train_openimages.py --epochs 30 --batch-size 64  # Custom training params
        """,
    )

    # Download options
    p.add_argument("--num-images", type=int, default=1000,
                    help="Number of Open Images to download (default: 1000)")
    p.add_argument("--skip-download", action="store_true",
                    help="Skip downloading, use existing images")
    p.add_argument("--skip-stego-gen", action="store_true",
                    help="Skip stego generation, use existing stego images")
    p.add_argument("--download-workers", type=int, default=8,
                    help="Parallel download threads (default: 8)")

    # Data directories
    p.add_argument("--data-dir", type=str, default="data/openimages",
                    help="Base directory for Open Images data (default: data/openimages)")
    p.add_argument("--cover-dir", type=str, default=None,
                    help="Override cover images directory")
    p.add_argument("--stego-dir", type=str, default=None,
                    help="Override stego images directory")

    # Training options
    p.add_argument("--epochs", type=int, default=150, help="Training epochs (default: 150)")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--image-size", type=int, default=256, help="Image size (default: 256)")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing (default: 0.0)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    p.add_argument("--val-split", type=float, default=0.10, help="Validation split ratio (default: 0.10)")
    p.add_argument("--test-split", type=float, default=0.10, help="Test split ratio (default: 0.10)")

    # Stego generation
    p.add_argument("--payload-length", type=int, default=4000, help="Default payload length (default: 4000)")
    p.add_argument("--variable-payload", action="store_true", default=True,
                    help="Vary payload length per image (default: True)")
    p.add_argument("--payload-min", type=int, default=1000, help="Min payload when variable (default: 1000)")
    p.add_argument("--payload-max", type=int, default=8000, help="Max payload when variable (default: 8000)")

    # Model options
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    p.add_argument("--no-kv-hpf", action="store_true", help="Disable KV HPF first layer")
    p.add_argument("--use-channel-attention", action="store_true", help="Enable channel attention block")
    p.add_argument("--use-learnable-hpf", action="store_true", help="Enable learnable HPF initialized from KV")
    p.add_argument("--enable-mixup", action="store_true", help="Enable mixup during training")
    p.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup alpha (default: 0.2)")
    p.add_argument("--enable-curriculum", action="store_true", help="Enable payload-density curriculum")
    p.add_argument("--early-stopping-patience", type=int, default=20, help="AUC patience for early stopping (0 = disabled)")
    p.add_argument("--no-early-stopping", action="store_true", help="Train all epochs; disable early stopping")
    p.add_argument("--epoch-ping", action="store_true", help="After each epoch: log checkpoints/epoch_pings.log + Windows beep")
    p.add_argument(
        "--no-diagnostic-stops",
        action="store_true",
        help="Disable FP/FN symmetry, epoch-30 acc floor, and post-epoch-15 AUC guards (NaN loss still stops)",
    )
    p.add_argument("--enable-hard-negative-mining", action="store_true", help="Mine and oversample hard validation pairs from epoch 30")
    p.add_argument("--hard-negative-ratio", type=float, default=2.0, help="Oversampling ratio for hard negatives")
    p.add_argument("--run-ensemble", action="store_true", help="Train 3 configured variants and report ensemble metrics")
    p.add_argument("--ensemble-seeds", type=str, default="42,52,62", help="Comma-separated seeds for ensemble variants")
    p.add_argument("--split-seed", type=int, default=42, help="Seed controlling train/val/test split")
    p.add_argument("--stage-name", type=str, default="manual_stage", help="Tracking label for this run (e.g., week1_baseline)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                    help="Checkpoint directory (default: checkpoints)")
    p.add_argument("--save-name", type=str, default="srnet_openimages_best.pth",
                    help="Best model filename (default: srnet_openimages_best.pth)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return p


def main():
    args = build_arg_parser().parse_args()
    early_patience = 0 if args.no_early_stopping else args.early_stopping_patience

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve directories
    base_dir = args.data_dir
    cover_dir = args.cover_dir or os.path.join(base_dir, "cover")
    stego_dir = args.stego_dir or os.path.join(base_dir, "stego")
    metadata_csv = os.path.join(base_dir, "train-images-metadata.csv")

    logger.info("=" * 70)
    logger.info("STEGANALYSIS TRAINING ON OPEN IMAGES V7")
    logger.info("=" * 70)

    # --- Step 1: Download ---
    if not args.skip_download:
        logger.info("")
        logger.info("STEP 1: Downloading Open Images V7 train images...")
        logger.info("-" * 70)
        os.makedirs(base_dir, exist_ok=True)
        num_downloaded = download_open_images(
            output_dir=cover_dir,
            metadata_csv_path=metadata_csv,
            num_images=args.num_images,
            num_workers=args.download_workers,
            seed=args.seed,
        )
        logger.info("Step 1 complete: %d cover images available.", num_downloaded)
    else:
        num_existing = len([f for f in os.listdir(cover_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
        logger.info("Skipping download. Using %d existing images in %s", num_existing, cover_dir)

    # --- Step 2: Generate stego ---
    if not args.skip_stego_gen:
        logger.info("")
        logger.info("STEP 2: Generating stego images via LSB embedding...")
        logger.info("-" * 70)
        num_stego = generate_stego_from_covers(
            cover_dir=cover_dir,
            stego_dir=stego_dir,
            payload_length=args.payload_length,
            seed=args.seed,
            variable_payload=args.variable_payload,
            payload_range=(args.payload_min, args.payload_max),
        )
        logger.info("Step 2 complete: %d stego images available.", num_stego)
    else:
        logger.info("Skipping stego generation. Using existing stego images in %s", stego_dir)

    # --- Step 3: Train ---
    logger.info("")
    logger.info("STEP 3: Training SRNet model...")
    logger.info("-" * 70)
    if args.run_ensemble:
        ensemble_seeds = [int(x.strip()) for x in args.ensemble_seeds.split(",") if x.strip()]
        run_ensemble_training(
            cover_dir=cover_dir,
            stego_dir=stego_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            val_split=args.val_split,
            test_split=args.test_split,
            use_amp=not args.no_amp,
            use_kv_hpf=not args.no_kv_hpf,
            enable_mixup=args.enable_mixup,
            mixup_alpha=args.mixup_alpha,
            enable_curriculum=args.enable_curriculum,
            early_stopping_patience=early_patience,
            enable_hard_negative_mining=args.enable_hard_negative_mining,
            hard_negative_ratio=args.hard_negative_ratio,
            split_seed=args.split_seed,
            seeds=ensemble_seeds,
            stage_name=args.stage_name,
        )
    else:
        out = run_training(
            cover_dir=cover_dir,
            stego_dir=stego_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            checkpoint_dir=args.checkpoint_dir,
            save_name=args.save_name,
            val_split=args.val_split,
            test_split=args.test_split,
            use_amp=not args.no_amp,
            use_kv_hpf=not args.no_kv_hpf,
            use_channel_attention=args.use_channel_attention,
            use_learnable_hpf=args.use_learnable_hpf,
            enable_mixup=args.enable_mixup,
            mixup_alpha=args.mixup_alpha,
            enable_curriculum=args.enable_curriculum,
            early_stopping_patience=early_patience,
            enable_hard_negative_mining=args.enable_hard_negative_mining,
            hard_negative_ratio=args.hard_negative_ratio,
            seed=args.seed,
            split_seed=args.split_seed,
            stage_name=args.stage_name,
            epoch_ping=args.epoch_ping,
            enable_diagnostic_stops=not args.no_diagnostic_stops,
        )
        update_upgrade_tracking(
            checkpoint_dir=args.checkpoint_dir,
            stage_name=args.stage_name,
            run_out=out,
        )


if __name__ == "__main__":
    main()
