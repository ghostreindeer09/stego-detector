import argparse
import os
from glob import glob
from typing import List, Tuple

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from stego.detector import StegoDetector
from stego.features import get_device


class Alaska2Dataset(Dataset):
    """
    Simple ALASKA2-style dataset wrapper.
    Expects directory structure like:

    data_root/
      train/
        cover/*.jpg
        stego/*.jpg
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        self.root = root
        self.split = split

        cover_dir = os.path.join(root, split, "cover")
        stego_dir = os.path.join(root, split, "stego")

        self.cover_paths = sorted(glob(os.path.join(cover_dir, "*.jpg")))
        self.stego_paths = sorted(glob(os.path.join(stego_dir, "*.jpg")))

        self.samples: List[Tuple[str, int]] = []
        for p in self.cover_paths:
            self.samples.append((p, 0))
        for p in self.stego_paths:
            self.samples.append((p, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, torch.tensor(label, dtype=torch.float32)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Train SRNet-like stego detector on ALASKA2.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to ALASKA2 root directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser


def train(args):
    device = get_device()

    train_dataset = Alaska2Dataset(args.data_root, split="train")
    val_dataset = Alaska2Dataset(args.data_root, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Use StegoDetector to get model and feature extractor, but we will manage the loop
    detector = StegoDetector(image_size=args.image_size, model_weights=None, device=device)
    model = detector.model
    feature_extractor = detector.feature_extractor

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_acc = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            batch_size = imgs.size(0)
            logits_list = []

            for i in range(batch_size):
                img = imgs[i]
                pil_img = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8"))
                img_format = "JPEG"
                features = feature_extractor.extract_features(pil_img, img_format)
                x = detector._prepare_input_tensor(features).to(device)
                logits, _ = model(x)
                logits_list.append(logits)

            logits_batch = torch.vstack(logits_list).squeeze(1)
            labels_batch = labels.to(device)

            loss = criterion(logits_batch, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            preds = (torch.sigmoid(logits_batch) > 0.5).float()
            correct += (preds == labels_batch).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                batch_size = imgs.size(0)
                logits_list = []
                for i in range(batch_size):
                    img = imgs[i]
                    pil_img = Image.fromarray(
                        (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
                    )
                    img_format = "JPEG"
                    features = feature_extractor.extract_features(pil_img, img_format)
                    x = detector._prepare_input_tensor(features).to(device)
                    logits, _ = model(x)
                    logits_list.append(logits)

                logits_batch = torch.vstack(logits_list).squeeze(1)
                labels_batch = labels.to(device)

                preds = (torch.sigmoid(logits_batch) > 0.5).float()
                val_correct += (preds == labels_batch).sum().item()
                val_total += batch_size

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "srnet_alaska2_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path}")


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

