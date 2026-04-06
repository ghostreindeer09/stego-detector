#!/usr/bin/env python3
"""
run_pipeline_v31.py — Steganalysis Pipeline v3.1
=================================================
Leakage-free, reproducible, high-performance steganalysis.
Target: >85% accuracy, >0.90 AUC.

Usage:
    python run_pipeline_v31.py                          # Full pipeline
    python run_pipeline_v31.py --skip-data-gen          # Skip data generation
    python run_pipeline_v31.py --stage baseline         # Named stage
    python run_pipeline_v31.py --epochs 50 --batch-size 32
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.preflight import run_preflight
from pipeline.data_gen import build_leakage_free_dataset
from pipeline.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="Steganalysis Pipeline v3.1")
    p.add_argument("--data-dir", default="data/openimages",
                   help="Directory with cover/ and stego/ subdirs")
    p.add_argument("--output-dir", default="data/splits_v31",
                   help="Output directory for leakage-free splits")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint dir")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--embed-seed", type=int, default=123)
    p.add_argument("--stage", default="v31_baseline", help="Stage name for tracking")
    p.add_argument("--skip-data-gen", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--use-channel-attention", action="store_true")
    p.add_argument("--use-learnable-hpf", action="store_true")
    p.add_argument("--enable-mixup", action="store_true")
    p.add_argument("--enable-curriculum", action="store_true", default=True)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--payload-min", type=int, default=2000)
    p.add_argument("--payload-max", type=int, default=10000)
    args = p.parse_args()

    if args.no_curriculum:
        args.enable_curriculum = False

    logger.info("=" * 62)
    logger.info("STEGANALYSIS PIPELINE v3.1")
    logger.info("=" * 62)

    # ── PART 0: PRE-FLIGHT ──
    if not args.skip_preflight:
        preflight = run_preflight(args.data_dir)
        amp_dtype = preflight['amp_dtype']
    else:
        amp_dtype = torch.bfloat16
        logger.info("Pre-flight skipped")

    # ── PART 2: DATA GENERATION ──
    if not args.skip_data_gen:
        source_dir = args.data_dir
        data_result = build_leakage_free_dataset(
            source_dir=source_dir,
            output_dir=args.output_dir,
            target_resolution=(args.image_size, args.image_size),
            payload_range=(args.payload_min, args.payload_max),
            split_seed=args.split_seed,
            embed_seed=args.embed_seed,
        )
        train_pairs = data_result['train_pairs']
        val_pairs = data_result['val_pairs']
        test_pairs = data_result['test_pairs']
        bpp_values = data_result['train_bpp']
    else:
        # Load from existing split directories
        from stego.datasets import PairConstraintStegoDataset
        splits_dir = args.output_dir
        datasets = {}
        for split in ['train', 'val', 'test']:
            cd = os.path.join(splits_dir, split, 'cover')
            sd = os.path.join(splits_dir, split, 'stego')
            if os.path.isdir(cd) and os.path.isdir(sd):
                ds = PairConstraintStegoDataset(cd, sd, args.image_size)
                datasets[split] = ds.pairs
                logger.info(f"Loaded {split}: {len(ds.pairs)} pairs")
        train_pairs = datasets.get('train', [])
        val_pairs = datasets.get('val', [])
        test_pairs = datasets.get('test', [])
        bpp_values = None

    if not train_pairs:
        logger.error("No training pairs found!")
        return

    logger.info(f"\nDataset: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    # ── PART 3-5: TRAINING ──
    result = run_training(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        data_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_kv_hpf=True,
        use_channel_attention=args.use_channel_attention,
        use_learnable_hpf=args.use_learnable_hpf,
        enable_mixup=args.enable_mixup,
        enable_curriculum=args.enable_curriculum,
        early_stopping_patience=args.early_stop_patience,
        amp_dtype=amp_dtype,
        seed=args.seed,
        stage_name=args.stage,
        bpp_values=bpp_values,
    )

    # ── RESULTS ──
    test_met = result.get('test_metrics', {})
    logger.info("\n" + "=" * 62)
    logger.info("FINAL RESULTS")
    logger.info("=" * 62)
    logger.info(f"  Accuracy:    {test_met.get('accuracy',0)*100:.2f}%")
    logger.info(f"  AUC:         {test_met.get('weighted_auc',0):.4f}")
    logger.info(f"  F1:          {test_met.get('f1',0):.4f}")
    logger.info(f"  Precision:   {test_met.get('precision',0):.4f}")
    logger.info(f"  Recall:      {test_met.get('recall',0):.4f}")
    logger.info(f"  PE:          {test_met.get('pe',0):.4f}")
    logger.info(f"  EER:         {test_met.get('eer',0):.4f}")
    logger.info(f"  Best epoch:  {result['best_epoch']}")
    logger.info(f"  Time:        {result.get('total_time_min',0):.1f} min")
    logger.info("=" * 62)

    # Save tracking
    track_path = os.path.join(args.checkpoint_dir, "upgrade_tracking_v31.json")
    tracking = {'runs': []}
    if os.path.exists(track_path):
        with open(track_path) as f:
            tracking = json.load(f)
    tracking['runs'].append({
        'stage': args.stage,
        'test_acc': test_met.get('accuracy', 0),
        'test_auc': test_met.get('weighted_auc', 0),
        'test_f1': test_met.get('f1', 0),
        'best_epoch': result['best_epoch'],
        'time_min': result.get('total_time_min', 0),
    })
    with open(track_path, 'w') as f:
        json.dump(tracking, f, indent=2)

    # Target check
    acc = test_met.get('accuracy', 0)
    auc = test_met.get('weighted_auc', 0)
    if acc >= 0.85 and auc >= 0.90:
        logger.info("🎯 TARGET MET: Accuracy > 85% AND AUC > 0.90")
    else:
        logger.info(f"⚠️  TARGET NOT MET: need Acc>85% (got {acc*100:.1f}%) "
                     f"AUC>0.90 (got {auc:.4f})")


if __name__ == "__main__":
    main()
