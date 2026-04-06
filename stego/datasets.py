"""
Pair-constraint steganalysis datasets v3.1.
Binary labels: 0 = Cover, 1 = Stego.
Each batch contains (cover, stego) pairs so the model sees matching content.

AUGMENTATION RULES (LSB-SAFE):
  Phase 1 (Epochs 1-50):   horizontal flip, vertical flip, random crop ONLY
  Phase 2 (Epochs 51-100): horizontal flip, vertical flip ONLY
  Phase 3 (Epochs 101-150): flips + JPEG quality >= 90 ONLY

FORBIDDEN in Phase 1-2: JPEG compression, color jitter, rotation, blur
"""

import os
from glob import glob
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Albumentations optional for training
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def get_phase1_transform(
    image_size: int = 256,
    additional_targets: Optional[dict] = None,
) -> "A.Compose":
    """
    Phase 1 (Epochs 1-50): ONLY flips + random crop.
    NO JPEG, NO color jitter, NO rotation.
    These augmentations are safe for LSB signal preservation.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Install albumentations: pip install albumentations")

    if additional_targets is None:
        additional_targets = {"stego_image": "image"}

    return A.Compose(
        [
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def get_phase2_transform(
    image_size: int = 256,
    additional_targets: Optional[dict] = None,
) -> "A.Compose":
    """
    Phase 2 (Epochs 51-100): ONLY flips (no crop randomness).
    Tightest augmentation to protect weak signal learning.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Install albumentations: pip install albumentations")

    if additional_targets is None:
        additional_targets = {"stego_image": "image"}

    return A.Compose(
        [
            A.CenterCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def get_phase3_transform(
    image_size: int = 256,
    additional_targets: Optional[dict] = None,
) -> "A.Compose":
    """
    Phase 3 (Epochs 101-150): flips + JPEG quality >= 90 ONLY.
    Light JPEG to build robustness after signal is already learned.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Install albumentations: pip install albumentations")

    if additional_targets is None:
        additional_targets = {"stego_image": "image"}

    return A.Compose(
        [
            A.CenterCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ImageCompression(
                quality_range=(90, 100),
                compression_type=A.ImageCompression.ImageCompressionType.JPEG,
                p=0.3,
            ),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def get_train_transform(
    image_size: int = 256,
    phase: int = 1,
    additional_targets: Optional[dict] = None,
    **kwargs,
) -> "A.Compose":
    """
    Get the appropriate training transform for the given phase.
    This is the main entry point — backwards compatible.
    """
    if phase == 1:
        return get_phase1_transform(image_size, additional_targets)
    elif phase == 2:
        return get_phase2_transform(image_size, additional_targets)
    elif phase == 3:
        return get_phase3_transform(image_size, additional_targets)
    else:
        return get_phase1_transform(image_size, additional_targets)


def get_val_transform(image_size: int = 256) -> "A.Compose":
    """Validation/test transform: deterministic center crop only."""
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Install albumentations: pip install albumentations")
    return A.Compose(
        [
            A.CenterCrop(image_size, image_size),
            ToTensorV2(),
        ]
    )


def get_inference_transform(image_size: int = 256) -> "A.Compose":
    """Inference transform: identical to validation — no augmentation."""
    return get_val_transform(image_size)


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


class PairConstraintStegoDataset(Dataset):
    """
    Binary steganalysis dataset with pair-constraint: each sample is a (cover, stego) pair.
    Labels: 0 = Cover, 1 = Stego.
    The same spatial augmentation is applied to both images so the model sees matching content.
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
            basename = os.path.basename(cover_path)
            name, ext = os.path.splitext(basename)
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cover_path, stego_path = self.pairs[idx]
        cover = _load_image(cover_path)
        stego = _load_image(stego_path)

        if self.transform is not None:
            # Pair-constraint: same augmentation for cover and stego via additional_targets
            if "stego_image" in getattr(self.transform, "additional_targets", {}):
                out = self.transform(image=cover, stego_image=stego)
                cover_t = out["image"]
                stego_t = out["stego_image"]
            else:
                t_cover = self.transform(image=cover)
                t_stego = self.transform(image=stego)
                cover_t = t_cover["image"]
                stego_t = t_stego["image"]
        else:
            if ALBUMENTATIONS_AVAILABLE:
                t = get_val_transform(self.image_size)
                cover_t = t(image=cover)["image"]
                stego_t = t(image=stego)["image"]
            else:
                raise RuntimeError("Transform must be provided or install albumentations.")

        # Labels: 0 = cover, 1 = stego
        label_cover = torch.tensor(0.0, dtype=torch.float32)
        label_stego = torch.tensor(1.0, dtype=torch.float32)
        return cover_t, stego_t, label_cover, label_stego


def pair_constraint_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate so that each batch has shape [2*B, C, H, W] and labels [2*B]
    with (cover_0, stego_0, cover_1, stego_1, ...) and (0, 1, 0, 1, ...).
    """
    covers = torch.stack([b[0] for b in batch], dim=0)
    stegos = torch.stack([b[1] for b in batch], dim=0)
    labels_c = torch.stack([b[2] for b in batch], dim=0)
    labels_s = torch.stack([b[3] for b in batch], dim=0)

    # Interleave: [c0, s0, c1, s1, ...]
    B = covers.size(0)
    images = torch.stack([covers, stegos], dim=1).view(2 * B, *covers.shape[1:])
    labels = torch.stack([labels_c, labels_s], dim=1).view(2 * B)
    return images, labels


def build_alaska2_pairs(
    data_root: str,
    split: str = "train",
) -> Tuple[str, str]:
    """Return (cover_dir, stego_dir) for ALASKA2 layout."""
    cover_dir = os.path.join(data_root, split, "cover")
    stego_dir = os.path.join(data_root, split, "stego")
    return cover_dir, stego_dir


def build_bossbase_pairs(
    cover_root: str,
    stego_root: str,
) -> Tuple[str, str]:
    """Return (cover_dir, stego_dir) for BOSSBase."""
    return cover_root, stego_root
