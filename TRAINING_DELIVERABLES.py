"""
==============================================================================
DELIVERABLE: Dataset class, SRNet architecture, and train_one_epoch
==============================================================================
Robust training pipeline for SRNet-based steganalysis (ALASKA2 + BOSSBase).
- Binary labels: 0 = Cover, 1 = Stego.
- Pair-constraint: cover and corresponding stego in the same batch.
- KV HPF as first non-trainable layer; Albumentations (Rotate90, Flip, JPEG 70-95).
- AdamW 1e-3, OneCycleLR, BCEWithLogitsLoss + label smoothing 0.1, AMP.
- Metrics: Weighted AUC, F1; confusion matrix script for False Positives.
==============================================================================
Copy the sections you need. Dependencies: torch, albumentations, numpy, PIL.
==============================================================================
"""

from __future__ import annotations

import os
from glob import glob
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# 1. DATASET CLASS (Pair-Constraint for ALASKA2 / BOSSBase)
# -----------------------------------------------------------------------------

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False


def get_train_transform(
    image_size: int = 256,
    jpeg_quality_min: int = 70,
    jpeg_quality_max: int = 95,
    additional_targets: Optional[dict] = None,
):
    """
    Albumentations: RandomRotate90, Flip, JPEG compression (70–95).
    additional_targets={"stego_image": "image"} applies the same random
    transform to cover and stego (pair-constraint).
    """
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("pip install albumentations")
    if additional_targets is None:
        additional_targets = {"stego_image": "image"}
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ImageCompression(
                quality_lower=jpeg_quality_min,
                quality_upper=jpeg_quality_max,
                p=0.5,
            ),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


class PairConstraintStegoDataset(Dataset):
    """
    Binary steganalysis dataset with pair-constraint.
    Labels: 0 = Cover, 1 = Stego.
    Each sample is a (cover, stego) pair; the same augmentation is applied
    to both so the model sees matching content and minimizes content interference.
    Directory layout: cover_dir/*.jpg and stego_dir/*.jpg with matching names.
    """

    def __init__(
        self,
        cover_dir: str,
        stego_dir: str,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.image_size = image_size
        self.transform = transform
        self.extensions = extensions
        self.pairs: List[Tuple[str, str]] = []
        self._collect_pairs()

    def _collect_pairs(self) -> None:
        cover_files = []
        for ext in self.extensions:
            cover_files.extend(glob(os.path.join(self.cover_dir, "*" + ext)))
        cover_files = sorted(cover_files)
        for cover_path in cover_files:
            name, _ = os.path.splitext(os.path.basename(cover_path))
            stego_path = None
            for e in self.extensions:
                candidate = os.path.join(self.stego_dir, name + e)
                if os.path.isfile(candidate):
                    stego_path = candidate
                    break
            if stego_path is not None:
                self.pairs.append((cover_path, stego_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cover_path, stego_path = self.pairs[idx]
        cover = _load_image(cover_path)
        stego = _load_image(stego_path)
        if self.transform is not None:
            if "stego_image" in getattr(self.transform, "additional_targets", {}):
                out = self.transform(image=cover, stego_image=stego)
                cover_t, stego_t = out["image"], out["stego_image"]
            else:
                cover_t = self.transform(image=cover)["image"]
                stego_t = self.transform(image=stego)["image"]
        else:
            if _HAS_ALBUMENTATIONS:
                t = A.Compose(
                    [
                        A.Resize(self.image_size, self.image_size),
                        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ToTensorV2(),
                    ]
                )
                cover_t = t(image=cover)["image"]
                stego_t = t(image=stego)["image"]
            else:
                raise RuntimeError("Transform required or install albumentations.")
        label_cover = torch.tensor(0.0, dtype=torch.float32)
        label_stego = torch.tensor(1.0, dtype=torch.float32)
        return cover_t, stego_t, label_cover, label_stego


def pair_constraint_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate so batch has images [2*B, C, H, W] and labels [2*B]
    with (cover_0, stego_0, cover_1, stego_1, ...) and (0, 1, 0, 1, ...).
    """
    covers = torch.stack([b[0] for b in batch], dim=0)
    stegos = torch.stack([b[1] for b in batch], dim=0)
    labels_c = torch.stack([b[2] for b in batch], dim=0)
    labels_s = torch.stack([b[3] for b in batch], dim=0)
    B = covers.size(0)
    images = torch.stack([covers, stegos], dim=1).view(2 * B, *covers.shape[1:])
    labels = torch.stack([labels_c, labels_s], dim=1).view(2 * B)
    return images, labels


# -----------------------------------------------------------------------------
# 2. SRNet ARCHITECTURE (KV HPF + Residual Backbone)
# -----------------------------------------------------------------------------


def get_kv_kernel_5x5() -> np.ndarray:
    """KV-style 5x5 high-pass kernel (fixed, non-trainable)."""
    return np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [-1, 2, 4, 2, -1],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float32,
    )


class KVHighPassFilter(nn.Module):
    """
    Fixed 5x5 KV high-pass filter as the first non-trainable layer.
    Forces the model to look at pixel residuals. Input: [B, 3, H, W]; output: [B, 3, H, W].
    """

    def __init__(self):
        super().__init__()
        kernel = get_kv_kernel_5x5().reshape(1, 1, 5, 5)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weight = nn.Parameter(torch.from_numpy(kernel), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None, padding=2, groups=3)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)


class SRNetBackbone(nn.Module):
    """Residual CNN backbone for steganalysis on noise residual maps."""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.layer4 = BasicBlock(128, 256, stride=2)
        self.layer5 = BasicBlock(256, 256, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        feat_map = x
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, feat_map


class SRNet(nn.Module):
    """
    SRNet: fixed KV HPF (first non-trainable layer) + residual backbone.
    Input: [B, 3, H, W] RGB. Output: logits [B, 1], feat_map for Grad-CAM.
    """

    def __init__(self, num_classes: int = 1, use_kv_hpf: bool = True):
        super().__init__()
        self.kv_hpf = KVHighPassFilter() if use_kv_hpf else nn.Identity()
        self.backbone = SRNetBackbone(in_channels=3, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.kv_hpf(x)
        return self.backbone(x)


# -----------------------------------------------------------------------------
# 3. TRAIN_ONE_EPOCH (AdamW, OneCycleLR, BCE + label smoothing 0.1, AMP)
# -----------------------------------------------------------------------------


def bce_with_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    BCEWithLogitsLoss with label smoothing (0.1) to prevent overfitting
    on subtle noise patterns. Labels 0 -> 0.05, 1 -> 0.95.
    """
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits.squeeze(1), targets_smooth)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    scaler: Optional[GradScaler],
    label_smoothing: float = 0.1,
    use_amp: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    One training epoch with AMP and label-smoothed BCE.
    Returns (mean_loss, all_scores, all_labels) for epoch-level metrics.
    """
    model.train()
    running_loss = 0.0
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
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
    return mean_loss, np.concatenate(all_scores), np.concatenate(all_labels)


# -----------------------------------------------------------------------------
# USAGE (Weighted AUC / F1 and confusion matrix: use stego.metrics and
# evaluate_confusion_matrix.py)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: build dataset and run one batch through SRNet + train_one_epoch
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cover_dir = "data/ALASKA2/train/cover"
    stego_dir = "data/ALASKA2/train/stego"
    if os.path.isdir(cover_dir) and os.path.isdir(stego_dir):
        transform = get_train_transform(256, 70, 95)
        ds = PairConstraintStegoDataset(cover_dir, stego_dir, 256, transform)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, shuffle=True, collate_fn=pair_constraint_collate
        )
        model = SRNet(num_classes=1, use_kv_hpf=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=1, steps_per_epoch=len(loader)
        )
        scaler = GradScaler() if device.type == "cuda" else None
        loss, scores, labels = train_one_epoch(
            model, loader, optimizer, scheduler, device, scaler, 0.1, use_amp=(device.type == "cuda")
        )
        print("Example epoch loss:", loss, "scores shape:", scores.shape, "labels shape:", labels.shape)
    else:
        print("Place ALASKA2 train/cover and train/stego and re-run for a quick test.")
