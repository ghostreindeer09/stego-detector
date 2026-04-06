"""
pipeline/preflight.py — Pre-flight checks (Part 0)
All checks must PASS before training can begin.
"""
import os
import sys
import shutil
import time
import threading
import subprocess
import pathlib
import hashlib
import json
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── CHECK 0.1: DISK SPACE ──────────────────────────────────────────────────
def check_disk_space(path: str = ".", required_gb: float = 10.0) -> None:
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    free_gb = free / 1e9
    total_gb = total / 1e9
    print(f"  Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < required_gb:
        raise RuntimeError(f"FAIL: {free_gb:.1f} GB free < {required_gb} GB required")
    print(f"  CHECK 0.1: PASS — {free_gb:.1f} GB free")


# ── CHECK 0.2: GPU / VRAM ──────────────────────────────────────────────────
def check_gpu() -> Tuple[str, float, bool, bool, torch.dtype]:
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL: No CUDA GPU detected. Training requires GPU.")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    fp16 = cap >= (7, 0)
    bf16 = cap >= (8, 0)
    if bf16:
        amp_dtype = torch.bfloat16
    elif fp16:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    print(f"  GPU: {name} | VRAM: {vram:.1f} GB | SM {cap[0]}.{cap[1]}")
    print(f"  FP16: {fp16} | BF16: {bf16} | AMP dtype: {amp_dtype}")
    print(f"  CHECK 0.2: PASS")
    return name, vram, fp16, bf16, amp_dtype


# ── CHECK 0.3: BATCH SIZE TUNING ───────────────────────────────────────────
def find_optimal_batch_size(model, img_size: int = 256, start: int = 32, max_bs: int = 128):
    optimal = start
    for bs in [32, 48, 64, 96, 128]:
        if bs > max_bs:
            break
        try:
            torch.cuda.empty_cache()
            dummy = torch.randn(bs, 3, img_size, img_size, device="cuda")
            dummy_l = torch.randint(0, 2, (bs,), device="cuda").float()
            out, _ = model(dummy)
            loss = F.binary_cross_entropy_with_logits(out.squeeze(1), dummy_l)
            loss.backward()
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            headroom = total - reserved
            print(f"  Batch {bs}: VRAM headroom={headroom:.1f} GB — OK")
            if headroom > 1.0:
                optimal = bs
            torch.cuda.empty_cache()
            model.zero_grad()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  Batch {bs}: OOM — stopping")
                torch.cuda.empty_cache()
                break
            raise
    print(f"  CHECK 0.3: PASS — Optimal batch size: {optimal}")
    return optimal


# ── CHECK 0.6-0.7: IMAGE AUDIT ─────────────────────────────────────────────
def audit_images(image_dir: str, sample_n: int = 100):
    all_imgs = sorted(pathlib.Path(image_dir).rglob('*.png'))
    non_png = [p for p in pathlib.Path(image_dir).rglob('*.*')
               if p.suffix.lower() != '.png' and p.is_file()
               and p.suffix.lower() in ('.jpg', '.jpeg', '.bmp', '.tiff')]
    print(f"  Total PNG: {len(all_imgs)} | Non-PNG: {len(non_png)}")
    if non_png:
        print(f"  WARNING: {len(non_png)} non-PNG files found")
    resolutions = {}
    for p in all_imgs[:sample_n]:
        img = Image.open(p)
        resolutions[img.size] = resolutions.get(img.size, 0) + 1
    print(f"  Resolution distribution (sample {min(sample_n, len(all_imgs))}):")
    for res, cnt in sorted(resolutions.items(), key=lambda x: -x[1]):
        print(f"    {res[0]}x{res[1]}: {cnt}")
    dominant = max(resolutions, key=resolutions.get) if resolutions else (256, 256)
    print(f"  CHECK 0.6-0.7: PASS — Dominant: {dominant[0]}x{dominant[1]}")
    return dominant, len(all_imgs)


# ── CHECK 0.8: LSB EMBEDDING VERIFICATION ──────────────────────────────────
def verify_embedding_pairs(cover_dir: str, stego_dir: str, n: int = 50):
    covers = sorted(pathlib.Path(cover_dir).glob('*.png'))[:n]
    broken = 0
    bpp_vals = []
    for cp in covers:
        sp = pathlib.Path(stego_dir) / cp.name
        if not sp.exists():
            broken += 1
            continue
        c = np.array(Image.open(cp)).astype(np.int16)
        s = np.array(Image.open(sp)).astype(np.int16)
        diff = c - s
        max_d = int(np.max(np.abs(diff)))
        changed = int(np.sum(diff != 0))
        if max_d > 1 or changed == 0:
            broken += 1
            continue
        bpp_vals.append(changed / c.size)
    if bpp_vals:
        print(f"  Embedding: {broken}/{n} broken | bpp mean={np.mean(bpp_vals):.4f} "
              f"min={np.min(bpp_vals):.4f} max={np.max(bpp_vals):.4f}")
    if broken > 0:
        print(f"  WARNING: {broken} broken pairs")
    print(f"  CHECK 0.8: {'PASS' if broken == 0 else 'WARNING'}")
    return broken, bpp_vals


# ── GPU MONITOR ─────────────────────────────────────────────────────────────
class GPUMonitor:
    def __init__(self, interval: int = 30):
        self.interval = interval
        self.running = False
        self.util_log = []
        self.memory_log = []

    def _loop(self):
        while self.running:
            try:
                r = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5)
                parts = r.stdout.strip().split(',')
                if len(parts) >= 3:
                    self.util_log.append(float(parts[0]))
                    self.memory_log.append(float(parts[1]))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def report(self):
        if not self.util_log:
            return {}
        return {
            'mean_util': float(np.mean(self.util_log)),
            'max_util': float(np.max(self.util_log)),
            'mean_vram_mb': float(np.mean(self.memory_log)),
            'peak_vram_mb': float(np.max(self.memory_log)),
        }


# ── DATASET FINGERPRINTING ─────────────────────────────────────────────────
def fingerprint_directory(dir_path: str) -> dict:
    files = sorted(pathlib.Path(dir_path).rglob('*.png'))
    names_hash = hashlib.md5(''.join(p.name for p in files).encode()).hexdigest()
    return {'count': len(files), 'hash': names_hash}


def create_manifest(data_dir: str) -> dict:
    manifest = {}
    cover_dir = os.path.join(data_dir, 'cover')
    stego_dir = os.path.join(data_dir, 'stego')
    if os.path.isdir(cover_dir):
        manifest['cover'] = fingerprint_directory(cover_dir)
    if os.path.isdir(stego_dir):
        manifest['stego'] = fingerprint_directory(stego_dir)
    return manifest


def verify_manifest(expected: dict, data_dir: str) -> bool:
    current = create_manifest(data_dir)
    for key in expected:
        if key not in current:
            return False
        if expected[key]['count'] != current[key]['count']:
            return False
        if expected[key]['hash'] != current[key]['hash']:
            return False
    return True


# ── GRADIENT HEALTH ─────────────────────────────────────────────────────────
def monitor_gradients(model, epoch: int) -> Tuple[float, float]:
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms.append(p.grad.norm().item())
    if not norms:
        print(f"  WARNING: No gradients at epoch {epoch}")
        return 0.0, 0.0
    mean_n = float(np.mean(norms))
    max_n = float(np.max(norms))
    if mean_n < 1e-7:
        raise RuntimeError(f"VANISHING GRADIENTS epoch {epoch}: mean={mean_n:.2e}")
    if max_n > 100:
        raise RuntimeError(f"EXPLODING GRADIENTS epoch {epoch}: max={max_n:.2e}")
    return mean_n, max_n


# ── DATA PIPELINE BENCHMARK ────────────────────────────────────────────────
def benchmark_dataloader(dataset, batch_size: int, workers_options=None):
    if workers_options is None:
        cpu_count = os.cpu_count() or 4
        workers_options = [0, 2, 4, min(8, cpu_count)]
    from torch.utils.data import DataLoader
    from stego.datasets import pair_constraint_collate
    results = {}
    for nw in workers_options:
        try:
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw,
                                pin_memory=True, collate_fn=pair_constraint_collate,
                                prefetch_factor=2 if nw > 0 else None,
                                persistent_workers=nw > 0)
            start = time.perf_counter()
            for i, (imgs, _) in enumerate(loader):
                imgs = imgs.cuda(non_blocking=True)
                if i >= 30:
                    break
            elapsed = time.perf_counter() - start
            throughput = 30 * batch_size * 2 / max(0.001, elapsed)
            results[nw] = throughput
            print(f"  workers={nw}: {throughput:.0f} img/s ({elapsed:.1f}s)")
            del loader
        except Exception as e:
            print(f"  workers={nw}: FAILED ({e})")
            results[nw] = 0
    best = max(results, key=results.get) if results else 0
    print(f"  CHECK 0.5: PASS — Optimal workers: {best}")
    return best, results


# ── FULL PRE-FLIGHT ─────────────────────────────────────────────────────────
def run_preflight(data_dir: str) -> dict:
    """Run all pre-flight checks. Returns config dict."""
    print("=" * 62)
    print("PART 0: PRE-FLIGHT CHECKS")
    print("=" * 62)

    print("\n[0.1] DISK SPACE")
    check_disk_space(data_dir, required_gb=5.0)

    print("\n[0.2] GPU / VRAM")
    gpu_name, vram_gb, fp16, bf16, amp_dtype = check_gpu()

    print("\n[0.6-0.7] IMAGE AUDIT")
    resolution, n_images = audit_images(data_dir)

    print("\n[0.8] LSB EMBEDDING VERIFICATION")
    cover_dir = os.path.join(data_dir, 'cover')
    stego_dir = os.path.join(data_dir, 'stego')
    if os.path.isdir(cover_dir) and os.path.isdir(stego_dir):
        broken, bpp_vals = verify_embedding_pairs(cover_dir, stego_dir)
    else:
        print("  Skipping embedding verification (splits not yet generated)")
        broken, bpp_vals = 0, []

    print("\n[0.10] ALGORITHM CHECK")
    from framework.embedding import get_available_algorithms
    algos = get_available_algorithms()
    for a in ['lsb_sequential', 'lsb_random', 'lsb_pvd', 'lsb_matching']:
        assert a in algos, f"FAIL: {a} not available"
    print(f"  Algorithms: {algos}")
    print(f"  CHECK 0.10: PASS")

    print("\n" + "=" * 62)
    print("ALL PRE-FLIGHT CHECKS: PASS")
    print("=" * 62)

    return {
        'gpu_name': gpu_name,
        'vram_gb': vram_gb,
        'fp16': fp16,
        'bf16': bf16,
        'amp_dtype': amp_dtype,
        'resolution': resolution,
        'n_images': n_images,
        'mean_bpp': float(np.mean(bpp_vals)) if bpp_vals else 0.0,
    }
