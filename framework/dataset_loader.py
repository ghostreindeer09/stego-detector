"""
Dataset Loader
===============
Unified data loading for the steganalysis framework.

- Reads the CSV mapping produced by dataset_generator.
- Supports ALASKA2-style pair-constraint loading (wraps stego.datasets).
- Supports a flat CSV-based loading mode for generated stego datasets.
- Splits data into train / val / test reproducibly.
"""

import csv
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV-based Dataset (for framework-generated datasets)
# ---------------------------------------------------------------------------

class CSVStegoDataset(Dataset):
    """
    Loads images and labels from a CSV mapping file produced by dataset_generator.

    CSV columns: image_name, label, payload, payload_length, source_path
    """

    def __init__(
        self,
        csv_path: str,
        image_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        include_cover: bool = True,
        include_stego: bool = True,
    ):
        self.csv_path = csv_path
        self.image_root = image_root
        self.transform = transform

        self.samples: List[Dict] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row["label"])
                if (label == 0 and not include_cover) or (label == 1 and not include_stego):
                    continue
                self.samples.append(row)

        logger.info("CSVStegoDataset: loaded %d samples from '%s'", len(self.samples), csv_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.samples[idx]
        label = int(row["label"])

        # Resolve image path
        path = row.get("source_path", "")
        if not os.path.isfile(path):
            # Fallback: try image_root / image_name
            if self.image_root:
                if label == 0:
                    path = os.path.join(self.image_root, row["image_name"])
                else:
                    path = os.path.join(self.image_root, row["image_name"])

        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        if self.transform is not None:
            transformed = self.transform(image=img_np)
            img_tensor = transformed["image"]
        else:
            # Minimal fallback: HWC uint8 -> CHW float32
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return img_tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split a dataset into train, val, test subsets."""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg,
    train_transform=None,
    val_transform=None,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from either:
      1. A CSV mapping file (framework-generated dataset), OR
      2. ALASKA2-style cover/stego directories (existing pipeline).

    Parameters
    ----------
    cfg : ExperimentConfig
    train_transform, val_transform
        Albumentations Compose or similar.

    Returns
    -------
    dict  {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dl_cfg = cfg.dataloader
    ds_cfg = cfg.dataset
    batch_size = dl_cfg.get("batch_size", 32)
    num_workers = dl_cfg.get("num_workers", 4)
    pin_memory = dl_cfg.get("pin_memory", True)
    seed = cfg.seed

    csv_path = ds_cfg.get("mapping_csv", "dataset_mapping.csv")

    if os.path.isfile(csv_path):
        logger.info("Loading CSV-based dataset from '%s'", csv_path)
        full_dataset = CSVStegoDataset(
            csv_path=csv_path,
            transform=train_transform,
        )
    else:
        raise FileNotFoundError(
            f"No mapping CSV found at '{csv_path}'. "
            "Run dataset generation first, or provide --data-root for ALASKA2."
        )

    train_set, val_set, test_set = split_dataset(
        full_dataset,
        train_ratio=dl_cfg.get("train_split", 0.7),
        val_ratio=dl_cfg.get("val_split", 0.15),
        test_ratio=dl_cfg.get("test_split", 0.15),
        seed=seed,
    )

    # Optionally rebind transforms (val/test should use val_transform)
    # Since Subset wraps the parent, we provide two different dataset instances
    val_dataset_full = CSVStegoDataset(csv_path=csv_path, transform=val_transform)
    test_dataset_full = CSVStegoDataset(csv_path=csv_path, transform=val_transform)

    val_indices = val_set.indices
    test_indices = test_set.indices
    val_set = Subset(val_dataset_full, val_indices)
    test_set = Subset(test_dataset_full, test_indices)

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    logger.info(
        "DataLoaders built: train=%d, val=%d, test=%d",
        len(train_set), len(val_set), len(test_set),
    )
    return loaders
