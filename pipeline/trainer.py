"""
pipeline/trainer.py — Training engine v3.1
Handles: mixed precision, gradient monitoring, checkpoint save/verify,
epoch reporting, curriculum learning, early stopping.
"""
import os
import sys
import math
import time
import json
import datetime
from typing import Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from stego.model import SRNet
from stego.metrics import compute_metrics
from stego.datasets import (
    PairConstraintStegoDataset, pair_constraint_collate,
    get_train_transform, get_val_transform,
)
from pipeline.preflight import (
    check_disk_space, monitor_gradients, GPUMonitor,
    create_manifest, verify_manifest, find_optimal_batch_size,
    benchmark_dataloader,
)

import logging
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(
            pred.squeeze(1), target.squeeze(1), reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class PairListDataset(torch.utils.data.Dataset):
    """Dataset backed by explicit (cover, stego) path list."""
    def __init__(self, pairs, transform):
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        cover_path, stego_path = self.pairs[idx]
        cover = np.array(PILImage.open(cover_path).convert('RGB'))
        stego = np.array(PILImage.open(stego_path).convert('RGB'))

        if self.transform is not None and "stego_image" in getattr(
                self.transform, "additional_targets", {}):
            out = self.transform(image=cover, stego_image=stego)
            cover_t, stego_t = out["image"], out["stego_image"]
        elif self.transform is not None:
            cover_t = self.transform(image=cover)["image"]
            stego_t = self.transform(image=stego)["image"]
        else:
            raise RuntimeError("Transform required")

        return (cover_t, stego_t,
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32))


def get_phase(epoch: int) -> int:
    if epoch <= 50:
        return 1
    elif epoch <= 100:
        return 2
    else:
        return 3


def train_one_epoch(model, loader, optimizer, scheduler, device,
                    scaler, amp_dtype, use_amp, criterion, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    all_scores, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True).float()
        if images.max() > 1.0:
            images = images / 255.0
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)

        # Mixup
        if mixup_alpha > 0.0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(images.size(0), device=device)
            images = lam * images + (1.0 - lam) * images[perm]
            labels = lam * labels + (1.0 - lam) * labels[perm]

        if use_amp and scaler is not None:
            with autocast('cuda', dtype=amp_dtype):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        elif use_amp and amp_dtype == torch.bfloat16:
            with autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        running_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.squeeze(1).float().cpu().numpy())

    return (running_loss / max(1, len(loader)),
            np.concatenate(all_scores), np.concatenate(all_labels))


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, use_amp, criterion):
    model.eval()
    running_loss = 0.0
    all_scores, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True).float()
        if images.max() > 1.0:
            images = images / 255.0
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        ctx = autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()
        with ctx:
            logits, _ = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        probs = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
        all_scores.append(probs)
        all_labels.append(labels.squeeze(1).float().cpu().numpy())

    scores = np.concatenate(all_scores)
    labels_arr = np.concatenate(all_labels)
    return running_loss / max(1, len(loader)), compute_metrics(labels_arr, scores)


def print_epoch_report(epoch, max_epochs, phase, epoch_time, t_loss, v_loss,
                       t_met, v_met, best_ep, best_auc, best_acc,
                       grad_mean, grad_max, gpu_report, status, lr, amp_dtype,
                       batch_size, throughput):
    mm, ss = divmod(int(epoch_time), 60)
    hh, mm = divmod(mm, 60)
    g = lambda d, k, default=0.0: d.get(k, default)

    v_auc = g(v_met, 'weighted_auc')
    v_f1 = g(v_met, 'f1')
    v_acc = g(v_met, 'accuracy') * 100
    v_prec = g(v_met, 'precision')
    v_rec = g(v_met, 'recall')
    v_pe = g(v_met, 'pe')
    v_eer = g(v_met, 'eer')
    tp, tn = g(v_met, 'tp', 0), g(v_met, 'tn', 0)
    fp, fn = g(v_met, 'fp', 0), g(v_met, 'fn', 0)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    gpu_util = gpu_report.get('mean_util', 0) if gpu_report else 0

    line = "+" + "=" * 60 + "+"
    print(line)
    print(f"| Epoch {epoch:<3}/{max_epochs:<3} | Phase {phase} | Time: {hh:02d}:{mm:02d}:{ss:02d}"
          f" | LR: {lr:.6f}  |")
    print(f"| Batch: {batch_size} | AMP: {amp_dtype} | Throughput: {throughput:.0f} img/s"
          f"{' ' * max(0, 13 - len(str(int(throughput))))}|")
    print("+" + "-" * 60 + "+")
    print(f"| GRAD    Mean: {grad_mean:.2e}  |  Max: {grad_max:.2e}"
          f"     GPU: {gpu_util:.0f}%  |")
    print(f"| LOSS    Train: {t_loss:.4f}     |  Val: {v_loss:.4f}"
          f"                |")
    print(f"| ACC     Train: {g(t_met,'accuracy')*100:5.1f}%     "
          f"|  Val: {v_acc:5.1f}%               |")
    print(f"| AUC           {v_auc:.4f}     |  F1:  {v_f1:.4f}"
          f"                |")
    print(f"| PREC          {v_prec:.4f}     |  REC: {v_rec:.4f}"
          f"                |")
    print("+" + "-" * 60 + "+")
    print(f"| PE: {v_pe:.4f} | EER: {v_eer:.4f} "
          f"| TPR@1%: {g(v_met,'tpr_at_fpr_0.01'):.4f}"
          f" | TPR@5%: {g(v_met,'tpr_at_fpr_0.05'):.4f} |")
    print(f"| TP:{tp:<5} TN:{tn:<5} FP:{fp:<5} FN:{fn:<5}"
          f" FPR:{fpr:.4f} FNR:{fnr:.4f} |")
    print("+" + "-" * 60 + "+")
    print(f"| STATUS: {status:<50} |")
    print(f"| BEST: Epoch {best_ep:<3} | AUC {best_auc:.4f}"
          f" | Acc {best_acc:5.1f}%               |")
    print(line)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics,
                    manifest, config, path):
    check_disk_space(os.path.dirname(path), required_gb=1.0)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'dataset_manifest': manifest,
        'config': config,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def run_training(
    train_pairs, val_pairs, test_pairs,
    data_dir: str,
    checkpoint_dir: str = "checkpoints",
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    image_size: int = 256,
    num_workers: int = 4,
    use_kv_hpf: bool = True,
    use_channel_attention: bool = False,
    use_learnable_hpf: bool = False,
    enable_mixup: bool = False,
    mixup_alpha: float = 0.2,
    enable_curriculum: bool = True,
    early_stopping_patience: int = 20,
    amp_dtype=torch.bfloat16,
    seed: int = 42,
    stage_name: str = "v3.1",
    bpp_values=None,
):
    """Full training loop with all monitoring and safeguards."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = SRNet(
        num_classes=1,
        use_kv_hpf=use_kv_hpf,
        use_learnable_hpf=use_learnable_hpf,
        use_channel_attention=use_channel_attention,
    ).to(device)

    # torch.compile if available (requires Triton — Linux only)
    import platform
    if hasattr(torch, 'compile') and platform.system() == 'Linux':
        try:
            model = torch.compile(model)
            logger.info("torch.compile: ENABLED")
        except Exception as e:
            logger.info(f"torch.compile skipped: {e}")
    else:
        logger.info("torch.compile: SKIPPED (Windows/no Triton)")

    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_p:,}")

    # AMP setup
    use_amp = amp_dtype in (torch.float16, torch.bfloat16)
    scaler = GradScaler() if amp_dtype == torch.float16 else None

    # Criterion
    criterion = FocalLoss(gamma=2, alpha=0.25)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler (OneCycleLR)
    steps_per_epoch = max(1, math.ceil(len(train_pairs) / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.3, div_factor=25.0, final_div_factor=1000.0,
    )

    # Datasets
    val_transform = get_val_transform(image_size)
    val_dataset = PairListDataset(val_pairs, val_transform)
    test_dataset = PairListDataset(test_pairs, val_transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=pair_constraint_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=pair_constraint_collate)

    # Manifest for integrity
    manifest = create_manifest(data_dir) if os.path.isdir(
        os.path.join(data_dir, 'train', 'cover')) else {}
    config = {
        'stage_name': stage_name, 'epochs': epochs, 'batch_size': batch_size,
        'lr': lr, 'seed': seed, 'amp_dtype': str(amp_dtype),
    }

    # GPU monitor
    gpu_monitor = GPUMonitor(interval=30)
    gpu_monitor.start()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_auc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    best_acc = 0.0
    no_improve = 0
    loss_history = []
    auc_history = []

    # BPP sorting for curriculum
    if bpp_values is not None and enable_curriculum:
        bpp_arr = np.array(bpp_values[:len(train_pairs)])
        sorted_idx = np.argsort(-bpp_arr)
    else:
        sorted_idx = np.arange(len(train_pairs))

    logger.info(f"\nStarting training: {epochs} epochs, batch={batch_size}")
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        phase = get_phase(epoch)

        # Curriculum: select subset of training data
        if enable_curriculum:
            if phase == 1:
                use_n = max(1, int(len(sorted_idx) * 0.30))
            elif phase == 2:
                use_n = max(1, int(len(sorted_idx) * 0.60))
            else:
                use_n = len(sorted_idx)
            selected = sorted_idx[:use_n]
            current_pairs = [train_pairs[i] for i in selected]
        else:
            current_pairs = list(train_pairs)

        # Phase-aware transform
        train_transform = get_train_transform(image_size, phase=phase)
        train_dataset = PairListDataset(current_pairs, train_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            collate_fn=pair_constraint_collate, drop_last=True,
        )

        # Train
        mixup_a = mixup_alpha if enable_mixup and phase >= 2 else 0.0
        t_loss, t_scores, t_labels = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            scaler, amp_dtype, use_amp, criterion, mixup_alpha=mixup_a,
        )
        t_met = compute_metrics(t_labels, t_scores)

        # Gradient health
        grad_mean, grad_max = monitor_gradients(model, epoch)

        # Validate
        v_loss, v_met = evaluate(model, val_loader, device, amp_dtype, use_amp, criterion)

        epoch_time = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]["lr"]

        v_auc = v_met.get("weighted_auc", 0)
        v_f1 = v_met.get("f1", 0)
        v_acc = v_met.get("accuracy", 0) * 100
        n_train_imgs = len(current_pairs) * 2
        throughput = n_train_imgs / max(0.01, epoch_time)

        # Tracking
        loss_history.append(v_loss)
        auc_history.append(v_auc)

        # Status
        status = "ON_TRACK"
        if epoch == 30 and (v_acc < 55 or v_auc < 0.55):
            status = "BEHIND (Ep30)"
        elif epoch == 50 and (v_acc < 65 or v_auc < 0.65):
            status = "BEHIND (Ep50)"
        elif epoch == 75 and (v_acc < 72 or v_auc < 0.75):
            status = "BEHIND (Ep75)"

        # Best model
        if v_auc > best_auc + 1e-4:
            best_auc = v_auc
            best_f1 = v_f1
            best_epoch = epoch
            best_acc = v_acc
            no_improve = 0
            save_path = os.path.join(checkpoint_dir, f"srnet_{stage_name}_best.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            v_met, manifest, config, save_path)
        else:
            no_improve += 1

        # Report
        gpu_report = gpu_monitor.report()
        print_epoch_report(
            epoch, epochs, phase, epoch_time, t_loss, v_loss,
            t_met, v_met, best_epoch, best_auc, best_acc,
            grad_mean, grad_max, gpu_report, status, lr_current,
            amp_dtype, batch_size, throughput,
        )

        # Loss/AUC correlation check
        if len(loss_history) >= 10:
            if loss_history[-1] < loss_history[-10] and auc_history[-1] <= auc_history[-10] + 0.005:
                print("  WARNING: Loss falling but AUC flat — check embedding integrity")

        # Stopping conditions
        if not np.isfinite(t_loss) or not np.isfinite(v_loss):
            logger.error("Loss NaN/Inf — stopping")
            break
        if epoch > 15 and v_auc < 0.50:
            logger.error(f"AUC {v_auc:.4f} < 0.50 after epoch 15 — stopping")
            break
        if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={early_stopping_patience})")
            break

        # Periodic checkpoint
        if epoch in [1, 5, 10, 20, 30, 50, 75, 100, 150]:
            ckpt_path = os.path.join(checkpoint_dir, f"srnet_{stage_name}_ep{epoch}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            v_met, manifest, config, ckpt_path)

    # Stop GPU monitor
    gpu_monitor.stop()
    total_time = time.time() - train_start

    logger.info(f"\nTraining complete: {total_time/60:.1f} min")
    logger.info(f"Best: epoch={best_epoch} AUC={best_auc:.4f} Acc={best_acc:.1f}%")

    # Load best and evaluate on test
    best_path = os.path.join(checkpoint_dir, f"srnet_{stage_name}_best.pth")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])

    _, test_met = evaluate(model, test_loader, device, amp_dtype, use_amp, criterion)
    logger.info(f"TEST: Acc={test_met.get('accuracy',0)*100:.2f}% "
                f"AUC={test_met.get('weighted_auc',0):.4f} "
                f"F1={test_met.get('f1',0):.4f}")

    # GPU report
    final_gpu = gpu_monitor.report()
    if final_gpu:
        logger.info(f"GPU: mean_util={final_gpu.get('mean_util',0):.0f}% "
                     f"peak_vram={final_gpu.get('peak_vram_mb',0):.0f}MB")

    return {
        'best_auc': best_auc, 'best_acc': best_acc / 100.0,
        'best_f1': best_f1, 'best_epoch': best_epoch,
        'test_metrics': test_met, 'gpu_report': final_gpu,
        'total_time_min': total_time / 60,
    }
